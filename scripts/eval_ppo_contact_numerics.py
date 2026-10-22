"""Fixed-actor, cfg-only PhysX solver-velocity4/8 contact diagnostic.

Solver counts are requested cfg and loaded USD-authored evidence, NOT native
solver-state readback. Native tensor shape offsets/materials are read once,
before actor loading, without changing or advancing the simulation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

BASE_ADAPTER_SHA256 = "5353f849d7d4f263c735d2cb000c8000403bf81280b51634065c9b721bb3b277"
BASE_ADAPTER_PATH = Path(__file__).with_name("eval_ppo_contact_damping.py")
TENSOR_API_SHA256 = "5dd16f8a37eccc94ac82338d6c1127e785cccf761cae1f6f18ef03d55b0f325f"
KIND = "fixed_ppo_nominal_contact_numerics_diagnostic"


def load_base_adapter(path=BASE_ADAPTER_PATH):
    path = Path(path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != BASE_ADAPTER_SHA256:
        raise ValueError("Frozen provenance adapter SHA mismatch")
    spec = importlib.util.spec_from_file_location("_contact_numerics_provenance", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


base = load_base_adapter()


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--solver-velocity-iterations", type=int, choices=(4, 8), required=True)
    options, inherited = parser.parse_known_args(list(argv))
    for word in inherited:
        base.require(word.split("=", 1)[0] not in ("--leg-kd", "--leg-kp", "--solver-position-iterations", "--contact-offset", "--rest-offset"), "Combined override is blocked")
    plan = base.prepare_plan(["--leg-kd", "10", *inherited]); plan.pop("leg_kd")
    plan["solver_velocity_iterations"] = options.solver_velocity_iterations
    audit = plan["audit"]
    for key in ("deployment_leg_kd", "trained_leg_kd"):
        audit.pop(key)
    audit.update(kind=KIND, unchanged_leg_kd=10., baseline_configured_solver_velocity_iterations=4,
        configured_solver_velocity_iterations=options.solver_velocity_iterations,
        unchanged_configured_solver_position_iterations=8, only_changed_control_field=None,
        only_changed_numerical_field="solver_velocity_iteration_count" if options.solver_velocity_iterations == 8 else None,
        wrapper_sha256=base.sha256(__file__), provenance_adapter_sha256=BASE_ADAPTER_SHA256,
        solver_count_evidence_kind="requested_cfg_and_loaded_USD_authored_not_native",
        native_solver_iteration_readback={"available": False, "position": None, "velocity": None,
            "reason": "Installed tensor ArticulationView exposes no native solver-count getter; no deprecated extension is enabled"},
        added_physics_frames=0, extra_policy_calls=0, contact_offsets_changed=False, rest_offsets_changed=False,
        material_changed_relative_to_nominal=False, solver_position_iterations_changed=False,
        interpretation="One fixed actor, nominal mu=.7; only cfg solver-velocity count changes. Not a retrained policy or stability certification.")
    return plan


def apply_config(cfg, evaluation_profile, plan):
    props = cfg.scene.robot.spawn.articulation_props
    legs = cfg.scene.robot.actuators["legs"]
    base.require((props.solver_position_iteration_count, props.solver_velocity_iteration_count) == (8, 4), "Unexpected original articulation solver cfg")
    base.require((legs.stiffness, legs.damping, cfg.sim.dt, cfg.decimation) == (500., 10., .005, 4), "Original PD or clock changed")
    base.require(evaluation_profile["contact_friction"] == .7, "Require nominal material")
    props.solver_velocity_iteration_count = plan["solver_velocity_iterations"]


def tensor_api_source(getter):
    path = inspect.getsourcefile(getter)
    base.require(path is not None and Path(path).is_file(), "Native tensor getter Python provenance unavailable")
    digest = base.sha256(path)
    base.require(digest == TENSOR_API_SHA256, "Installed native tensor API source differs from audited version")
    return {"path": str(Path(path).resolve()), "sha256": digest}


def loaded_usd_solver(stage, prim_paths):
    from pxr import Usd, UsdPhysics
    rows = []
    for path in prim_paths:
        start = stage.GetPrimAtPath(path)
        base.require(bool(start), "Native articulation prim path absent in loaded stage")
        roots = [prim for prim in Usd.PrimRange(start) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if not roots:
            ancestor = start.GetParent()
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.ArticulationRootAPI):
                    roots = [ancestor]; break
                ancestor = ancestor.GetParent()
        base.require(len(roots) == 1, "Require one actual articulation-root API prim per native articulation")
        prim = roots[0]; row = {"native_prim_path": str(path), "articulation_root_prim_path": str(prim.GetPath())}
        for key, attribute in (("position", "physxArticulation:solverPositionIterationCount"), ("velocity", "physxArticulation:solverVelocityIterationCount")):
            attr = prim.GetAttribute(attribute)
            base.require(bool(attr) and attr.Get() is not None and attr.HasAuthoredValueOpinion(), "Loaded solver count lacks explicit USD-authored value")
            row[key] = int(attr.Get()); row[key+"_has_authored_value"] = True
        value = prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get()
        row["enabled_self_collisions_usd"] = bool(value) if value is not None else None
        rows.append(row)
    return {"source": "loaded USD articulation-root physxArticulation attributes", "native_backend_readback": False, "roots": rows}


def runtime_readback(env, cfg, plan, *, stage=None, source_inspector=tensor_api_source, usd_reader=loaded_usd_solver):
    import numpy as np
    base.require("runtime_readback" not in plan["audit"], "Readback hook must run exactly once")
    def clock():
        return [int(env._sim_step_counter), int(env.sim.current_time_step_index), float(env.sim.current_time)]
    before = clock(); robot = env.scene["robot"]; view = robot.root_physx_view
    values, sources = {}, {}
    for name, method in (("contact_offsets_m", "get_contact_offsets"), ("rest_offsets_m", "get_rest_offsets"),
                         ("materials_static_dynamic_restitution", "get_material_properties")):
        getter = getattr(view, method); sources[method] = source_inspector(getter)
        value = getter()
        if hasattr(value, "detach"):
            value = value.detach().clone().cpu().numpy()
        else:
            value = np.asarray(value).copy()
        base.require(value.dtype == np.float32 and np.isfinite(value).all(), "Native shape data must be finite float32")
        values[name] = value
    base.require(all(source["sha256"] == TENSOR_API_SHA256 for source in sources.values()), "Native getter source hash differs")
    count, maximum = int(view.count), int(view.max_shapes)
    base.require(count == 1 and maximum > 0, "Require the unchanged single-articulation diagnostic")
    base.require(values["contact_offsets_m"].shape == values["rest_offsets_m"].shape == (count, maximum)
        and values["materials_static_dynamic_restitution"].shape == (count, maximum, 3), "Native collision-shape dimensions differ")
    base.require(np.all(values["contact_offsets_m"] >= values["rest_offsets_m"]-1e-7), "Contact offset is below rest offset")
    materials = values["materials_static_dynamic_restitution"]
    base.require(np.allclose(materials[..., :2], .7, rtol=0, atol=1e-6), "Native robot materials no longer nominal mu=.7")
    base.require(np.all(materials[..., 2] >= 0) and np.all(materials[..., 2] <= 1), "Invalid native restitution")
    paths = list(view.prim_paths); links = [list(row) for row in view.link_paths]
    base.require(len(paths) == count and len(links) == count, "Native shape/articulation identity dimensions differ")
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    authored = usd_reader(stage, paths)
    target = plan["solver_velocity_iterations"]
    base.require(authored["native_backend_readback"] is False and len(authored["roots"]) == count
        and all(row["position"] == 8 and row["velocity"] == target and row["position_has_authored_value"] is True
                and row["velocity_has_authored_value"] is True for row in authored["roots"]), "Loaded USD-authored counts differ from requested cfg")
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    kp, kd = array(robot.actuators["legs"].stiffness), array(robot.actuators["legs"].damping)
    base.require(np.isfinite(kp).all() and np.isfinite(kd).all() and np.all(kp == 500.) and np.all(kd == 10.), "Actual leg PD readback differs")
    limits = getattr(cfg.sim, "physx", None)
    limits = {name: getattr(limits, name, None) for name in ("min_position_iteration_count", "max_position_iteration_count", "min_velocity_iteration_count", "max_velocity_iteration_count")}
    after = clock(); base.require(before == after, "Readback advanced physics; no extra frames are permitted")
    result = {"cfg_requested": {"position": 8, "velocity": target,
                  "enabled_self_collisions": cfg.scene.robot.spawn.articulation_props.enabled_self_collisions},
        "usd_authored_solver": authored,
        "native_shape_readback": {**{key: value.tolist() for key, value in values.items()},
            "shape": [count, maximum], "prim_paths": paths, "link_paths": links, "getter_source": sources,
            "shape_columns_are_body_ids": False, "wheel_shape_column_mapping_claimed": False,
            "offset_units": "m", "material_channels": ["static_friction", "dynamic_friction", "restitution"]},
        "scene_physx_iteration_limits_cfg": limits, "scene_limits_are_native_readback": False,
        "native_solver_counts_available": False, "gain_readback": {"kp_range": [float(kp.min()), float(kp.max())], "kd_range": [float(kd.min()), float(kd.max())]},
        "clock_before": before, "clock_after": after, "added_physics_frames": 0,
        "time": "After normal env construction/material audit, before runner/actor load; no additional reset/forward/step"}
    plan["audit"]["runtime_readback"] = result
    return result


def decorate_contract(contract, plan):
    base.require((contract["leg_kp"], contract["leg_kd"]) == (500., 10.), "Unexpected exported controller")
    base.require("runtime_readback" in plan["audit"], "Actual initialized readback is required before export")
    contract["contact_numerics_diagnostic"] = copy.deepcopy(plan["audit"])
    contract["configured_solver_position_iteration_count"] = 8
    contract["configured_solver_velocity_iteration_count"] = plan["solver_velocity_iterations"]
    contract["solver_counts_are_native_readback"] = False


def materialize_source(source, legacy_root):
    generated = base.materialize_source(source, legacy_root)
    replacements = [
        ('            "contact_stability_diagnostic": copy.deepcopy(_contact_plan["audit"]),',
         '            "contact_numerics_diagnostic": copy.deepcopy(_contact_plan["audit"]),'),
        ('            contact_material_audit = audit_isaac_materials(env, evaluation_profile)',
         '            contact_material_audit = audit_isaac_materials(env, evaluation_profile)\n            _contact_readback(env, cfg, _contact_plan)'),
    ]
    for old, new in replacements:
        base.require(generated.count(old) == 1, "Pinned numerical-readback insertion anchor differs")
        generated = generated.replace(old, new, 1)
    compile(generated, "<pinned_contact_numerics_evaluator>", "exec")
    return generated


def run(argv):
    plan = prepare_plan(argv)
    generated = materialize_source(plan["legacy_evaluator"].read_text(), plan["legacy_evaluator"].parent.parent)
    plan["audit"]["generated_evaluator_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    namespace = {"__name__": "_pinned_contact_numerics_evaluator", "__file__": str(Path(__file__).resolve()),
        "_contact_plan": plan, "_contact_apply_config": apply_config, "_contact_decorate_contract": decorate_contract,
        "_contact_readback": runtime_readback}
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<pinned_contact_numerics_evaluator>", "exec"), namespace); namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    run(sys.argv[1:])

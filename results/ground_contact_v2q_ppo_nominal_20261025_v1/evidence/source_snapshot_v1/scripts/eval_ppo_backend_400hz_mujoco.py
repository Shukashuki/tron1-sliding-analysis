"""Completed new-cohort PPO -> native CPU MuJoCo at 400 Hz (not MJWarp).

The actor already includes its normalizer. A native full-horizon rollout uses
50 Hz actions and the unchanged explicit 400 Hz DC-PD implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys

from ppo_backend_400hz_contract import (DEFAULT_PROTOCOL, prepare_inputs, provenance,
    require, sha256, verify_cohort, verify_legacy, verify_recording, write_json)

BASE_SHA256 = "2251d740d7568874a47a232556241cc8d1ee5f5b7be1351d1c657aea8fa85f87"


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--cohort-dir", type=Path, required=True)
    result.add_argument("--cohort-protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--legacy-evaluator", type=Path, required=True)
    result.add_argument("--motion-file", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--evaluation-contact-friction", type=float, required=True)
    result.add_argument("--allow-smoke-policy", action="store_true")
    return result


def prepare_plan(argv):
    args = parser().parse_args(argv)
    evaluator, legacy_root = verify_legacy(args.legacy_evaluator, BASE_SHA256)
    cohort = verify_cohort(args.cohort_dir, args.cohort_protocol, allow_smoke=args.allow_smoke_policy)
    plan = prepare_inputs(cohort, args.motion_file, args.output_dir, args.evaluation_contact_friction,
                          "native CPU MuJoCo", model=args.model)
    plan.update(args=args, evaluator=evaluator, legacy_root=legacy_root, wrapper_sha256=sha256(__file__))
    return plan


def configure_priorities(model):
    """Explicit native mapping: wheel priority1, all other colliders priority0.

    Equal all-geom mu makes the targeted pair value mu under either priority
    selection or equal-priority max. Explicit pairs could override this, so fail.
    """
    import numpy as np
    require(model.npair == 0, "Explicit MuJoCo contact pairs could override the declared geom friction")
    colliding = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    wheels = []
    for index in colliding:
        name = model.body(int(model.geom_bodyid[index])).name
        model.geom_priority[index] = 1 if name in ("wheel_L_Link", "wheel_R_Link") else 0
        if model.geom_priority[index] == 1:
            wheels.append(int(index))
    require(len(wheels) >= 2, "Missing colliding geoms for two wheel bodies")
    return colliding, wheels


def audit_materials(model, base_audit, plan):
    import numpy as np
    colliding, wheels = configure_priorities(model)
    friction = np.asarray(model.geom_friction[colliding], dtype=float)
    require(np.isfinite(friction).all() and np.allclose(friction[:, 0], plan["mu"], rtol=0, atol=1e-12),
            "Native colliding-geom friction differs from fixed pair coefficient")
    require(model.opt.timestep == .0025, "Native model timestep differs")
    result = copy.deepcopy(base_audit)
    result.update(native_mapping_verified=True, compiled_native_timestep_s=float(model.opt.timestep),
        explicit_contact_pair_count=int(model.npair), wheel_geom_ids=wheels,
        colliding_geom_ids=colliding.tolist(),
        colliding_geom_names=[model.geom(int(i)).name for i in colliding],
        geom_friction=friction.tolist(), geom_priority=model.geom_priority[colliding].tolist(),
        geom_condim=model.geom_condim[colliding].tolist(), geom_solref=model.geom_solref[colliding].tolist(),
        geom_solimp=model.geom_solimp[colliding].tolist(),
        mapping="explicit wheel priority1 / other colliders0; all colliding sliding mu equal target; not mu squared",
        pair_coefficient=float(plan["mu"]), contact_law_equivalence_claimed=False)
    result["native_plant_readback"] = plant_readback(model)
    return result


def plant_readback(model):
    """Native compiled model parameters; no assertion of identical USD inertias."""
    import numpy as np
    names = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_armature", "dof_damping",
             "dof_frictionloss", "geom_margin", "geom_gap")
    result = {"source": "compiled native MuJoCo MjModel arrays after declared configuration",
        "body_names": [model.body(i).name for i in range(model.nbody)],
        "joint_names": [model.joint(i).name for i in range(model.njnt)],
        "inertia_convention": "body_inertia principal moments; body_ipos/body_iquat define local inertial frame",
        "mass_inertia_equivalence_claimed": False, "no_physics_or_state_mutation": True, "readbacks": {}}
    for name in names:
        value = np.asarray(getattr(model, name))
        require(value.size > 0 and np.isfinite(value).all(), "Nonfinite/empty native plant readback: " + name)
        result["readbacks"][name] = {"shape": list(value.shape), "values": value.tolist()}
    result["solver_configuration"] = {name: float(getattr(model.opt, name))
        for name in ("timestep", "iterations", "ls_iterations", "tolerance", "ls_tolerance", "solver", "integrator", "cone", "impratio")}
    return result


def finish_report(report, arrays, plan):
    import numpy as np
    t = np.asarray(arrays["time_s"], dtype=float)
    require(report["status"] == "completed" and report["completed_full_reference"] is True
            and report["policy_steps"] == 349 and report["physics_steps"] == 2792
            and report["hidden_resets"] == 0 and report["root_state_writes"] == 1
            and report["start_reference_frame"] == 0, "Native full-horizon rollout incomplete")
    require(t.shape == (2793,) and np.allclose(t, np.arange(2793) * .0025, rtol=0, atol=1e-10),
            "Native 400 Hz clock/state count differs")
    verify_recording(plan["output"] / "rollout.npz", "native CPU MuJoCo")
    for name, path in plan["cohort"]["paths"].items():
        require(sha256(path) == plan["cohort"]["input_sha256"][name], "Completed training input changed during evaluation: " + name)
    audit = provenance(plan["cohort"])
    audit.update(validation_backend="native CPU MuJoCo", wrapper_sha256=plan["wrapper_sha256"],
        legacy_evaluator_sha256=BASE_SHA256, generated_source_sha256=plan["generated_sha256"],
        native_states=2793, policy_decisions=349, duration_s=6.98, status="verified",
        normalization_applications_per_inference=1, extra_normalization=False,
        native_model_priority_mapping="wheel1_other0_explicit", no_visual_mesh=True,
        recording_sha256=sha256(plan["output"] / "rollout.npz"))
    report["backend_400hz_cohort_evaluation"] = audit
    report["contract"]["backend_400hz_cohort_evaluation"] = copy.deepcopy(audit)
    write_json(plan["input_dir"] / "evaluation_source_audit.json", audit)


def materialize_source(source, legacy_root):
    require(hashlib.sha256(source.encode()).hexdigest() == BASE_SHA256, "Original evaluator source changed")
    replacements = [
        ("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path(" + repr(str(Path(legacy_root).resolve())) + ")"),
        ("        contact_material_audit = configure_mujoco_materials(model, contract[\"task_profile\"])",
         "        contact_material_audit = configure_mujoco_materials(model, contract[\"task_profile\"])\n"
         "        contact_material_audit = _new_audit_materials(model, contact_material_audit, _new_plan)"),
        ("    (output / \"report.json\").write_text", "    _new_finish_report(report, arrays, _new_plan)\n"
         "    (output / \"report.json\").write_text"),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, "Absent/ambiguous frozen source anchor: " + old[:90])
        source = source.replace(old, new, 1)
    compile(source, "<new_400hz_ppo_native_cpu_evaluator>", "exec")
    return source


def run(argv):
    plan = prepare_plan(argv)
    generated = materialize_source(plan["evaluator"].read_text(), plan["legacy_root"])
    plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    # Native evaluator imports eval_tracking and the frozen task helpers. Restore
    # path/module additions afterwards so offline tests cannot contaminate peers.
    previous = list(sys.path)
    namespace = {"__name__": "_new_cohort_400hz_native_cpu", "__file__": str(Path(__file__).resolve()),
        "_new_plan": plan, "_new_audit_materials": audit_materials, "_new_finish_report": finish_report}
    try:
        sys.path[:0] = [str(plan["evaluator"].parent), str(plan["legacy_root"])]
        exec(compile(generated, "<new_400hz_ppo_native_cpu_evaluator>", "exec"), namespace)
        policy = namespace["load_torchscript"](plan["cohort"]["actor"])
        return namespace["run_evaluation"](policy, plan["motion"], plan["args"].model, plan["output"],
            policy_path=plan["cohort"]["actor"], no_visual_mesh=True, contract_path=plan["contract_path"])
    finally:
        sys.path[:] = previous


if __name__ == "__main__":
    run(sys.argv[1:])

"""New-cohort PPO -> PhysX, native 400 Hz, single full-horizon episode.

Counted in-memory integration into an immutable evaluator; no SDK/source edits,
no old checkpoint substitution, no SAC runner, no evaluation-only dt refinement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

from ppo_backend_400hz_contract import (DEFAULT_PROTOCOL, StrictPPOExportRunner,
    prepare_inputs, provenance, require, sha256, verify_cohort, verify_legacy, verify_recording, write_json)

BASE_SHA256 = "76d78337d70519693bc7e8d819a6523a800e1e4889de5a0466c2d34356d2ae9d"
RECIPE_SHA256 = "1365f4eecc6129bef4738e05a491763a2884407d8cf00a3a7d0faffbc24fa699"


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--cohort-dir", type=Path, required=True)
    result.add_argument("--cohort-protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--legacy-evaluator", type=Path, required=True)
    result.add_argument("--motion-file", type=Path, required=True)
    result.add_argument("--asset-path", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--evaluation-contact-friction", type=float, required=True)
    result.add_argument("--physical-gpu-index", type=int, required=True, choices=range(8))
    result.add_argument("--allow-smoke-policy", action="store_true")
    result.add_argument("--headless", action="store_true")
    return result


def prepare_plan(argv):
    args = parser().parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Use explicit unmasked physical GPU ordinal")
    require(os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID") == "PCI_BUS_ID", "Use PCI_BUS_ID CUDA order")
    evaluator, legacy_root = verify_legacy(args.legacy_evaluator, BASE_SHA256)
    recipe = Path(__file__).resolve().parent / "physx_mjlab_recipe.py"
    require(sha256(recipe) == RECIPE_SHA256, "Frozen material recipe source changed")
    cohort = verify_cohort(args.cohort_dir, args.cohort_protocol, allow_smoke=args.allow_smoke_policy)
    plan = prepare_inputs(cohort, args.motion_file, args.output_dir, args.evaluation_contact_friction,
                          "PhysX", asset=args.asset_path)
    runner_config = cohort["directory"] / "runner_config.json"
    from train_ppo_backend import ppo_config
    require(json.loads(runner_config.read_text()) == ppo_config(42, json.loads(runner_config.read_text())["save_interval"]),
            "Actual cohort RSL3 PPO configuration differs")
    index = args.physical_gpu_index
    plan.update(args=args, evaluator=evaluator, legacy_root=legacy_root,
        requested_device=f"cuda:{index}", wrapper_sha256=sha256(__file__), recipe_sha256=sha256(recipe),
        runner_config_sha256=sha256(runner_config))
    kit = (f"--/renderer/activeGpu={index} --/renderer/multiGpu/enabled=false "
           f"--/physics/cudaDevice={index} --/physics/sceneMultiGPUMode=0")
    plan["inherited_argv"] = ["--motion-file", str(Path(args.motion_file).resolve()), "--asset-path", str(Path(args.asset_path).resolve()),
        "--checkpoint", str(cohort["checkpoint"]), "--runner-config", str(runner_config),
        "--output-dir", str(plan["output"]), "--task-profile", str(plan["task_profile_path"]),
        "--evaluation-contact-friction", str(plan["mu"]), "--num-envs", "1", "--seed", "42",
        "--record-physics", "--verify-unmasked-device", "--device", plan["requested_device"],
        "--headless", "--kit_args", kit]
    return plan


def configure(cfg, evaluation_profile, plan):
    from physx_mjlab_recipe import install_physx_mjlab_recipe
    require((cfg.sim.dt, cfg.decimation) == (.005, 4), "Legacy configuration clock differs before new-cohort setup")
    legs = cfg.scene.robot.actuators["legs"]
    require((legs.stiffness, legs.damping) == (500., 10.), "Nominal explicit PD differs")
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .0025, 8, 8
    cfg.scene.contact_forces.update_period = .0025
    # Ground-filtered sensors have a physics-rate update period assigned by
    # the frozen task helper; set every configured ContactSensor consistently.
    for name, value in vars(cfg.scene).items():
        if (name in ("wheel_L_ground", "wheel_R_ground") or name.startswith("nonwheel_ground_")) and hasattr(value, "update_period"):
            value.update_period = .0025
    install_physx_mjlab_recipe(cfg, training=False, fixed_mu=plan["mu"])
    evaluation_profile["contact_friction"] = plan["mu"]


def audit_materials(env, plan):
    import torch
    from physx_mjlab_recipe import audit_physx_mjlab_recipe
    result = audit_physx_mjlab_recipe(env, check_usd=True)
    require(result["recipe"]["training"] is False and result["encoder_bias_range_rad"] == [0., 0.],
            "Evaluation material recipe unexpectedly randomized")
    legs = env.scene["robot"].actuators["legs"]
    kp, kd = legs.stiffness, legs.damping
    require(torch.isfinite(kp).all() and torch.isfinite(kd).all() and torch.all(kp == 500.) and torch.all(kd == 10.),
            "Live explicit leg PD differs")
    result["leg_pd_gain_readback"] = {"kp_range": [float(kp.min()), float(kp.max())],
                                      "kd_range": [float(kd.min()), float(kd.max())]}
    result["effective_pair_mapping"] = copy.deepcopy(plan["contract"]["evaluation_contact_friction_override"])
    result["native_plant_readback"] = plant_readback(env)
    return result


def plant_readback(env):
    """Read, never tune, native plant nuisance parameters after startup events."""
    import inspect
    import numpy as np
    robot = env.scene["robot"]
    view = robot.root_physx_view
    result = {"body_names": list(robot.body_names), "joint_names": list(robot.joint_names),
        "source": "live PhysX articulation view getters; not USD-authored inference",
        "inertia_convention": "native get_inertias flat9, reported without cross-engine frame conversion",
        "mass_inertia_equivalence_claimed": False, "no_physics_or_state_mutation": True,
        "readbacks": {}, "unavailable": {}}
    for name in ("get_masses", "get_inertias", "get_coms", "get_dof_armatures", "get_dof_dampings"):
        method = getattr(view, name, None)
        if method is None:
            result["unavailable"][name] = "getter not exposed by installed articulation view"
            continue
        value = method()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        require(value.size > 0 and np.isfinite(value).all(), "Nonfinite/empty native plant readback: " + name)
        try:
            source = inspect.getsourcefile(method)
        except TypeError:
            source = None
        result["readbacks"][name] = {"shape": list(value.shape), "values": value.tolist(),
            "getter_source_file": source, "getter_source_sha256": sha256(source) if source else None}
    require("get_masses" in result["readbacks"] and "get_inertias" in result["readbacks"],
            "Native mass/inertia getters required for backend attribution")
    return result


def finish_report(report, contract, plan, physical_arrays):
    import numpy as np
    require(report["status"] == "evaluated" and report["deterministic_start_frame"] == 0,
            "Full-horizon evaluation did not finish deterministically")
    t = np.asarray(physical_arrays["time_s"], dtype=float)
    require(t.shape == (2793,) and np.allclose(t, np.arange(2793) * .0025, rtol=0, atol=1e-10),
            "Native 400 Hz recording count/time differs")
    valid = np.asarray(physical_arrays["valid_mask"], dtype=bool)
    require(valid.all(), "First-episode recording became invalid")
    require(report["physics_clock_readback"]["status"] == "verified", "Native clock not verified")
    verify_recording(plan["output"] / "physics_trajectory.npz", "PhysX")
    for name, path in plan["cohort"]["paths"].items():
        require(sha256(path) == plan["cohort"]["input_sha256"][name], "Completed training input changed during evaluation: " + name)
    audit = provenance(plan["cohort"])
    audit.update(validation_backend="PhysX", wrapper_sha256=plan["wrapper_sha256"],
        legacy_evaluator_sha256=BASE_SHA256, generated_source_sha256=plan["generated_sha256"],
        evaluation_dependency_sha256=dict(__import__("ppo_backend_400hz_contract").EVALUATION_DEPENDENCIES),
        material_recipe_sha256=plan["recipe_sha256"], runner_config_sha256=plan["runner_config_sha256"],
        physical_gpu_index_requested=plan["args"].physical_gpu_index, native_states=2793,
        policy_decisions=349, duration_s=6.98, status="verified")
    audit["deployment_actor_sha256"] = sha256(plan["output"] / "actor_normalized.pt")
    report["backend_400hz_cohort_evaluation"] = audit
    contract["backend_400hz_cohort_evaluation"] = copy.deepcopy(audit)
    require(contract["physics_dt"] == .0025 and contract["decimation"] == 8
            and contract["trained_task_profile"] == plan["cohort"]["contract"]["trained_task_profile"],
            "Final exported contract lost trained400Hz identity")
    write_json(plan["input_dir"] / "evaluation_source_audit.json", audit)


def materialize_source(source, legacy_root):
    require(hashlib.sha256(source.encode()).hexdigest() == BASE_SHA256, "Original evaluator content differs")
    replacements = [
        ("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path(" + repr(str(Path(legacy_root).resolve())) + ")"),
        ("Diagnostic requires one seed42 200Hz episode, original gains, and physics recording",
         "New-cohort adapter requires one seed42 native400Hz episode, original gains, and physics recording"),
        ("        cfg.scene.terrain.physics_material.static_friction = mu\n"
         "        cfg.scene.terrain.physics_material.dynamic_friction = mu\n"
         "        cfg.sim.physics_material = cfg.scene.terrain.physics_material\n"
         "        cfg.events.physics_material.params[\"static_friction_range\"] = (mu, mu)\n"
         "        cfg.events.physics_material.params[\"dynamic_friction_range\"] = (mu, mu)",
         "        _new_configure(cfg, evaluation_profile, _new_plan)"),
        ("        clock_readback = (physics_clock_readback(env, cfg, refined_contract)\n"
         "                          if args.physics_dt_s is not None else None)",
         "        clock_readback = physics_clock_readback(env, cfg, _new_plan['contract'])"),
        ("            from training.tron1_drift import audit_isaac_materials\n"
         "            contact_material_audit = audit_isaac_materials(env, evaluation_profile)",
         "            contact_material_audit = _new_audit_materials(env, _new_plan)"),
        ("        # OnPolicyRunner mutates nested class_name fields, hence use a copy.\n"
         "        if runner_cfg.get(\"algorithm\") == \"SAC\":\n"
         "            if args.allow_smoke_policy:\n"
         "                runner_cfg[\"allow_smoke_actor\"] = True\n"
         "            from training.tron1_scripted_actor_runner import ScriptedActorRunner\n"
         "            runner = ScriptedActorRunner(wrapped, runner_cfg, log_dir=None, device=args.device)\n"
         "        else:\n"
         "            runner = OnPolicyRunner(wrapped, copy.deepcopy(runner_cfg), log_dir=None, device=args.device)",
         "        runner = _new_runner(_new_plan['cohort'], args.device)"),
        ("        contract = observation_action_contract()\n", "        contract = copy.deepcopy(_new_plan['contract'])\n"),
        ("        training_manifest = args.checkpoint.parent / \"manifest.json\"\n"
         "        if training_manifest.is_file():\n"
         "            metadata = json.loads(training_manifest.read_text())\n"
         "            contract[\"training_domain_randomization\"] = metadata.get(\"recipe\", metadata.get(\"domain_randomization_profile\"))\n"
         "            if metadata.get(\"height_regularization\") is not None:\n"
         "                contract[\"height_regularization\"] = metadata[\"height_regularization\"]\n"
         "                contract[\"historical_pose_predicates_only\"] = True",
         "        contract['training_domain_randomization'] = copy.deepcopy(_new_plan['cohort']['protocol']['training_randomization'])\n"
         "        contract['evaluation_contact_friction_override'] = copy.deepcopy(_new_plan['contract']['evaluation_contact_friction_override'])"),
        ("        (args.output_dir / \"policy_contract.json\").write_text",
         "        report['physics_clock_readback'] = clock_readback\n"
         "        _new_finish_report(report, contract, _new_plan, physical_arrays)\n"
         "        (args.output_dir / \"policy_contract.json\").write_text"),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, "Absent/ambiguous frozen integration anchor: " + old[:90])
        source = source.replace(old, new, 1)
    compile(source, "<new_400hz_ppo_physx_evaluator>", "exec")
    return source


def run(argv):
    plan = prepare_plan(argv)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    generated = materialize_source(plan["evaluator"].read_text(), plan["legacy_root"])
    plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    namespace = {"__name__": "_new_cohort_400hz_physx", "__file__": str(Path(__file__).resolve()),
        "_new_plan": plan, "_new_configure": configure, "_new_audit_materials": audit_materials,
        "_new_runner": StrictPPOExportRunner, "_new_finish_report": finish_report}
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<new_400hz_ppo_physx_evaluator>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    run(sys.argv[1:])

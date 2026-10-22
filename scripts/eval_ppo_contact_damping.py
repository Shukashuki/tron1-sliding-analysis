"""Pinned, nominal-friction zero-shot leg-damping diagnostic for one frozen PPO.

Execute the historical evaluator in memory, never edit it or its training files.
Only the deployment leg derivative gain changes; this is not a retrained policy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


BASE_EVALUATOR_SHA256 = "76d78337d70519693bc7e8d819a6523a800e1e4889de5a0466c2d34356d2ae9d"
TRAINING_DEPENDENCIES = {
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_drift.py": "5eb40c9b3c14f5bd5c59c9861c1414ae8ac1b219ff0dd70591f2c6fe0bb5344c",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
}
EVALUATION_DEPENDENCIES = {**TRAINING_DEPENDENCIES,
    "training/tron1_drift.py": "5fee46238c825a2bcf474e969e93af65834aa1302ce857aa1c553e8a9fbaefaa"}
PINS = {
    "checkpoint": "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a",
    "motion_file": "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254",
    "asset_path": "327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e",
    "task_profile": "31de29a61eec847e0e39783f3a0d9d8f2db551acd679a1b38dfde2b24e8ae7d6",
    "manifest": "066ddaeb5a23ff44dfca79e785520c57d8786cdc3a2d7b30d118e3a955a94d63",
    "ablation_manifest": "4cf22f9c4e6b5f28aa1711f2e093516522046d259e0322b740e7cad5da05905e",
    "runner_config": "67b8f0a98e072027d66a54c58b43595d7e41abc75a9292e30f6a6eeb5fe512ba",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _option(argv, flag, default=None):
    values = []
    for i, word in enumerate(argv):
        if word == flag:
            require(i + 1 < len(argv), "Missing value: " + flag)
            values.append(argv[i + 1])
        elif word.startswith(flag + "="):
            values.append(word.split("=", 1)[1])
    require(len(values) <= 1, "Duplicate option: " + flag)
    return values[0] if values else default


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--legacy-evaluator", type=Path, required=True)
    parser.add_argument("--leg-kd", type=float, choices=(10., 5.), required=True)
    options, inherited = parser.parse_known_args(list(argv))
    evaluator = options.legacy_evaluator.resolve()
    require(evaluator.is_file() and sha256(evaluator) == BASE_EVALUATOR_SHA256,
            "Pinned legacy evaluator SHA mismatch")
    for flag in ("--deployment-leg-kp", "--deployment-leg-kd", "--physics-dt-s",
                 "--domain-parameters", "--terrain-file", "--allow-smoke-policy"):
        require(not any(x == flag or x.startswith(flag + "=") for x in inherited),
                "Blocked historical/combined override: " + flag)
    for flag in ("--record-physics", "--verify-unmasked-device"):
        require(inherited.count(flag) == 1, "Required exact flag: " + flag)
    for flag, expected in (("--num-envs", 1), ("--seed", 42)):
        require(int(_option(inherited, flag, "-1")) == expected, "Required " + flag)
    require(float(_option(inherited, "--evaluation-contact-friction", "nan")) == .7,
            "This diagnostic is restricted to nominal mu=0.7")
    require(float(_option(inherited, "--contact-threshold-n", "5")) == 5.,
            "Original contact measurement threshold must remain 5 N")
    paths = {}
    for name in ("checkpoint", "motion_file", "asset_path", "task_profile", "runner_config", "output_dir"):
        value = _option(inherited, "--" + name.replace("_", "-"))
        require(value is not None, "Required argument: " + name)
        paths[name] = Path(value).resolve()
    require(not paths["output_dir"].exists(), "Preserve previous output: directory must be fresh")
    paths["manifest"] = paths["checkpoint"].parent / "manifest.json"
    paths["ablation_manifest"] = paths["checkpoint"].parent / "ablation_manifest.json"
    paths["run_status"] = paths["checkpoint"].parent / "run_status.json"
    for name, expected in PINS.items():
        require(paths[name].is_file() and sha256(paths[name]) == expected, "Pinned input SHA mismatch: " + name)
    manifest = json.loads(paths["manifest"].read_text())
    ablation = json.loads(paths["ablation_manifest"].read_text())
    status = json.loads(paths["run_status"].read_text())
    profile = json.loads(paths["task_profile"].read_text())
    require(manifest["task_profile"] == profile and manifest["contract"]["leg_kp"] == 500.
            and manifest["contract"]["leg_kd"] == 10., "Unexpected original task/controller")
    require((manifest["contract"]["physics_dt"], manifest["contract"]["decimation"],
             manifest["contract"]["policy_fps"]) == (.005, 4, 50.), "Original clock differs")
    require(ablation.get("status") == "training_completed" and not ablation.get("smoke_only", True)
            and ablation.get("initialization") == "from_scratch_no_checkpoint_resume"
            and ablation.get("reward_override", {}).get("weight") == -.5
            and ablation.get("actual_environment_transitions") == 12288000
            and ablation.get("checkpoint_sha256") == PINS["checkpoint"], "Incomplete or different ablation")
    require(status.get("status") == "training_completed" and not status.get("smoke_only", True)
            and status.get("environment_transitions") == 12288000
            and status.get("policy_steps_completed") == 24000, "Incomplete formal training")
    require(manifest.get("source_training_modules_sha256") == TRAINING_DEPENDENCIES,
            "Unexpected original training dependency manifest")
    for name, digest in EVALUATION_DEPENDENCIES.items():
        dependency = evaluator.parent.parent / name
        require(dependency.is_file() and sha256(dependency) == digest, "Legacy dependency changed: " + name)
    audit = {
        "schema_version": 1, "kind": "fixed_ppo_nominal_contact_damping_diagnostic",
        "diagnostic_only": True, "deployment_leg_kd": options.leg_kd, "trained_leg_kd": 10.,
        "unchanged_leg_kp": 500., "checkpoint_sha256": PINS["checkpoint"],
        "additional_training_updates": 0, "physics_dt_s": .005, "policy_dt_s": .02,
        "evaluation_contact_friction": .7, "trained_action_rate_weight": -.5,
        "training_at_deployment_gain_claimed": False, "reward_changes": False,
        "reset_changes": False, "force_filter_changes": False,
        "only_changed_control_field": "leg_kd" if options.leg_kd != 10. else None,
        "comparison_baseline": "same frozen actor at Kp500/Kd10 and mu0.7",
        "legacy_evaluator_sha256": BASE_EVALUATOR_SHA256,
        "training_dependency_sha256": dict(TRAINING_DEPENDENCIES),
        "evaluation_dependency_sha256": dict(EVALUATION_DEPENDENCIES),
        "training_evaluation_dependencies_identical": False,
        "dependency_cohort_note": "Retain the exact evaluator dependencies used by the previous frozen -0.5 PPO validations; the historical training drift module differs from the evaluation module.",
        "wrapper_sha256": sha256(__file__),
        "input_sha256": {name: sha256(path) for name, path in paths.items() if name != "output_dir"},
        "hardware_ready": False, "statistical_robustness_claimed": False,
    }
    return {"leg_kd": options.leg_kd, "inherited_argv": inherited,
            "legacy_evaluator": evaluator, "paths": paths, "audit": audit}


def apply_config(cfg, evaluation_profile, plan):
    """Change only the explicit leg actuator's damping before environment creation."""
    legs = cfg.scene.robot.actuators["legs"]
    require(legs.stiffness == 500. and legs.damping == 10., "Unexpected configured original PD")
    require(cfg.sim.dt == .005 and cfg.decimation == 4, "Unexpected physical/policy clock")
    require(evaluation_profile["contact_friction"] == .7, "Expected nominal evaluation material")
    legs.damping = plan["leg_kd"]
    if plan["leg_kd"] != 10.:
        evaluation_profile["leg_kd"] = plan["leg_kd"]


def decorate_contract(contract, plan):
    require(contract["leg_kp"] == 500. and contract["leg_kd"] == 10., "Unexpected exported original PD")
    contract["leg_kd"] = plan["leg_kd"]
    contract["contact_stability_diagnostic"] = copy.deepcopy(plan["audit"])


def materialize_source(source, legacy_root):
    """Exact-count substitutions against a SHA-pinned source; return code only."""
    require(hashlib.sha256(source.encode()).hexdigest() == BASE_EVALUATOR_SHA256,
            "Pinned legacy evaluator source changed")
    replacements = [
        ('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(' + repr(str(Path(legacy_root).resolve())) + ')'),
        ('        if args.physics_dt_s is not None:\n            from training.tron1_physics_clock import configure_physics_clock',
         '        _contact_apply_config(cfg, evaluation_profile, _contact_plan)\n        if args.physics_dt_s is not None:\n            from training.tron1_physics_clock import configure_physics_clock'),
        ('            "only_changed_physics_field": "contact_friction",',
         '            "only_changed_material_field": "contact_friction",'),
        ('        if args.physics_dt_s is not None:\n            contract = apply_physics_clock_to_contract(',
         '        _contact_decorate_contract(contract, _contact_plan)\n        if args.physics_dt_s is not None:\n            contract = apply_physics_clock_to_contract('),
        ('            "schema_version": 1, "status": "evaluated" if not active.any() else "evaluation_incomplete",',
         '            "contact_stability_diagnostic": copy.deepcopy(_contact_plan["audit"]),\n            "schema_version": 1, "status": "evaluated" if not active.any() else "evaluation_incomplete",'),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, "Pinned source insertion anchor differs: " + old)
        source = source.replace(old, new, 1)
    compile(source, "<pinned_contact_damping_evaluator>", "exec")
    return source


def run(argv):
    plan = prepare_plan(argv)
    source = materialize_source(plan["legacy_evaluator"].read_text(), plan["legacy_evaluator"].parent.parent)
    plan["audit"]["generated_evaluator_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    namespace = {"__name__": "_pinned_contact_damping_evaluator", "__file__": str(Path(__file__).resolve()),
                 "_contact_plan": plan, "_contact_apply_config": apply_config,
                 "_contact_decorate_contract": decorate_contract}
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(source, "<pinned_contact_damping_evaluator>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    run(sys.argv[1:])

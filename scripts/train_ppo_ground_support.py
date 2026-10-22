"""Prepared substep ground-support reward replacement of the fixed -0.5 PPO recipe.

Import never launches training. This method replaces the existing grounded term,
not its weight or the reference task. A separate actual Isaac smoke and explicit
parent finalization are required; app.close may exit before main returns.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import MethodType

from train_ppo_smoothness import (
    BASELINE_ASSET_SHA256, BASELINE_HEIGHT_SHA256, BASELINE_MOTION_SHA256,
    BASELINE_PROFILE_SHA256, BASELINE_SOURCES, BASELINE_TRAINER_SHA256,
    _option, require, sha256,
)


ROOT = Path(__file__).resolve().parents[1]
KIND = "historical_physx_ppo_substep_ground_support_replacement"
TERM_NAME = "drift_grounded"
RUNTIME_FILE = "ground_support_runtime_audit.json"
CLOSE_MARKER = "_tron1_ground_support_runtime_close_owner"
_MISSING = object()
_COLLECTOR = None
ACTION_WRAPPER_SHA256 = "a71f9c9fe84919a27ceadae499af9dc4b5ab14c18bfc04fcc42fab3829556093"
BASELINE_METADATA = {
    "manifest.json": "066ddaeb5a23ff44dfca79e785520c57d8786cdc3a2d7b30d118e3a955a94d63",
    "ablation_manifest.json": "4cf22f9c4e6b5f28aa1711f2e093516522046d259e0322b740e7cad5da05905e",
    "run_status.json": "fa46d5780c5753d6e01d9b19fccb2904d3d9cb06a17c1a894f7bd625c83ae773",
}
BASELINE_CHECKPOINT_SHA256 = "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a"


def collector():
    global _COLLECTOR
    if _COLLECTOR is None:
        path = ROOT / "scripts/ppo_substep_ground_support.py"
        require(path.is_file(), "Prepared ground-support collector is missing")
        spec = importlib.util.spec_from_file_location("_ppo_training_ground_support", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _COLLECTOR = module
    return _COLLECTOR


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--baseline-training-dir", type=Path, required=True)
    options, inherited = parser.parse_known_args(list(argv))
    for flag in ("--source-root", "--motion-file", "--asset-path", "--task-profile", "--output-dir"):
        require(_option(inherited, flag) is not None, "Required inherited argument: " + flag)
    require(not any(x.split("=", 1)[0] in ("--resume", "--action-rate-weight", "--leg-kp", "--leg-kd",
                    "--leg-torque-rate-weight", "--ground-support-weight", "--scale-fraction") for x in inherited),
            "Ground-support v1 is fixed, from scratch, action-rate -0.5 and PD500/10")
    output = Path(_option(inherited, "--output-dir")).resolve()
    require(not output.exists(), "Preserve existing output; choose a fresh directory")
    require(_option(inherited, "--reference-id", "drift_drl_traj4") == "drift_drl_traj4", "Require primary traj_4")
    require(int(_option(inherited, "--seed", "42")) == 42 and
            float(_option(inherited, "--ground-friction", "1")) == 1., "Preserve historical seed and training ground friction")
    require(int(_option(inherited, "--smoke-steps", "0")) == 0, "Separate smoke only; no setup control steps")
    smoke = "--smoke-only" in inherited
    envs, iterations = int(_option(inherited, "--num-envs", "512")), int(_option(inherited, "--iterations", "1000"))
    require((envs, iterations) == ((2, 1) if smoke else (512, 1000)), "Require exact smoke or formal budget")
    if _option(inherited, "--smoke-steps") is None:
        inherited += ["--smoke-steps", "0"]
    source = Path(_option(inherited, "--source-root")).resolve()
    trainer = ROOT / "scripts/train_beyondmimic_style.py"
    paths = {trainer: BASELINE_TRAINER_SHA256, ROOT / "scripts/height_smoothness.py": BASELINE_HEIGHT_SHA256,
        ROOT / "scripts/train_ppo_smoothness.py": ACTION_WRAPPER_SHA256,
        Path(_option(inherited, "--task-profile")): BASELINE_PROFILE_SHA256,
        Path(_option(inherited, "--motion-file")): BASELINE_MOTION_SHA256,
        Path(_option(inherited, "--asset-path")): BASELINE_ASSET_SHA256,
        **{source / name: digest for name, digest in BASELINE_SOURCES.items()}}
    for path, digest in paths.items():
        require(path.is_file() and sha256(path) == digest, "Historical source/input SHA mismatch: " + str(path))
    baseline = options.baseline_training_dir.resolve()
    for name, digest in BASELINE_METADATA.items():
        require((baseline / name).is_file() and sha256(baseline / name) == digest,
                "Fixed -0.5 baseline metadata SHA mismatch: " + name)
    ablation = json.loads((baseline / "ablation_manifest.json").read_text())
    require(ablation.get("checkpoint_sha256") == BASELINE_CHECKPOINT_SHA256 and
            ablation.get("reward_override", {}).get("weight") == -.5 and
            ablation.get("actual_environment_transitions") == 12288000 and
            ablation.get("initialization") == "from_scratch_no_checkpoint_resume",
            "Wrong baseline recipe or incomplete baseline")
    module = collector()
    return {"output_dir": output, "trainer": trainer, "source_root": source,
        "inherited_argv": inherited, "smoke_only": smoke, "num_envs": envs, "iterations": iterations,
        "expected_policy_steps": 24 * iterations, "expected_environment_transitions": envs * 24 * iterations,
        "baseline_training_dir": baseline, "baseline_metadata_sha256": dict(BASELINE_METADATA),
        "sampler_module_sha256": sha256(module.__file__)}


def make_pre_env_hook(original, plan):
    def install(cfg, *, ground_friction=1.):
        recipe = original(cfg, ground_friction=ground_friction)
        action, grounded = cfg.rewards.action_rate_l2, getattr(cfg.rewards, TERM_NAME)
        require(action.weight == -.1 and action.func.__name__ == "action_rate_l2" and
                action.func.__module__ == "isaaclab.envs.mdp.rewards", "Historical action-rate term changed")
        require(grounded.weight == .4 and grounded.func.__name__ == "grounded_reward" and
                grounded.func.__module__ == "training.tron1_drift", "Historical grounded term changed")
        require(cfg.scene.robot.actuators["legs"].stiffness == 500. and
                cfg.scene.robot.actuators["legs"].damping == 10., "Historical leg PD changed")
        require(cfg.sim.dt == .005 and cfg.decimation == 4, "Historical clocks changed")
        module = collector()
        require(sha256(module.__file__) == plan["sampler_module_sha256"], "Collector changed after preparation")
        module.configure_ground_support_sensors(cfg.scene)
        action.weight = -.5
        setattr(cfg.rewards, TERM_NAME, module.make_reward_term(weight=.4, scale_fraction=.20))
        metadata = {
            "schema_version": 1, "kind": KIND, "status": "configured_before_environment_creation",
            "prepared_method_runtime_verified": False, "trained_policy_claimed": False,
            "initialization": "from_scratch_no_checkpoint_resume", "smoke_only": plan["smoke_only"],
            "baseline_action_rate_weight": -.5, "source_default_action_rate_weight": -.1,
            "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "baseline_metadata_sha256": plan["baseline_metadata_sha256"],
            "reward_replacement": {"term": TERM_NAME, "weight": .4, "scale_fraction": .20,
                "original_function": "training.tron1_drift.grounded_reward", "original_sampling": "last_substep_50Hz",
                "original_formula": "both_wheel_net_force_norms_above_5N_binary",
                "new_function": "ppo_substep_ground_support.make_reward_term(...).GroundSupportReward",
                "raw_reward_function": "ppo_substep_ground_support.substep_ground_support_reward",
                "new_formula": "1-mean_4_substeps(mean_2_wheels(exp(-clamp(ground_Fz_N,min=0)/(.20*mg_N))))",
                "raw_reward_bounds": [0., 1.], "additional_reward_term": False,
                "reward_manager_dt_factor_applied_once": .02,
                "sensor_source": "two_single_wheel_actual_ground_filtered_force_matrix_w",
                "mass_cache": "poststartupDR_live_total_mass_and_gravity"},
            "expected_policy_steps": plan["expected_policy_steps"],
            "expected_environment_transitions": plan["expected_environment_transitions"],
            "num_envs": plan["num_envs"], "iterations": plan["iterations"], "seed": 42,
            "historical_trainer_sha256": BASELINE_TRAINER_SHA256, "height_adapter_sha256": BASELINE_HEIGHT_SHA256,
            "task_profile_sha256": BASELINE_PROFILE_SHA256, "motion_sha256": BASELINE_MOTION_SHA256,
            "asset_sha256": BASELINE_ASSET_SHA256, "baseline_source_sha256": dict(BASELINE_SOURCES),
            "wrapper_sha256": sha256(__file__), "sampler_module_sha256": plan["sampler_module_sha256"],
            "inherited_training_arguments": plan["inherited_argv"],
            "unchanged_relative_to_minus0p5_baseline": ["action-rate -0.5", "historical DR", "height smoothness",
                "nonzero yaw/beta reference", "Kp500/Kd10", "1-rad action scale", "DC motors", "PPO", "observation layout", "formal budget"],
            "base_task_profile_file_modified": False, "new_absolute_height_target": False,
            "new_zero_yaw_or_beta_target": False, "contact_stability_guaranteed": False,
            "actual_isaac_smoke_required_before_formal_adoption": True,
        }
        require(plan["output_dir"].is_dir(), "Historical trainer must create fresh output first")
        with (plan["output_dir"] / "ablation_manifest.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False); stream.write("\n")
        return recipe
    return install


def make_post_env_hook(original, plan):
    def audit_material(env, recipe):
        result = original(env, recipe)
        if hasattr(env, CLOSE_MARKER):
            return result
        module = collector()
        require(sha256(module.__file__) == plan["sampler_module_sha256"], "Collector source changed")
        handler = module.install_ground_support_accumulator(env)
        previous, original_close = vars(env).get("close", _MISSING), env.close
        def close(owner, *args, **kwargs):
            require(owner is env and getattr(env, CLOSE_MARKER, None) is handler, "Close ownership changed")
            try:
                runtime = handler.audit(check_device_health=True)
                handler.close()
                runtime.update(closed=True, wrapper_sha256=sha256(__file__))
                with (plan["output_dir"] / RUNTIME_FILE).open("x") as stream:
                    json.dump(runtime, stream, indent=2, allow_nan=False); stream.write("\n")
                require(runtime["device_health_finite"] is True, "Nonfinite physical ground force sampled")
            finally:
                if previous is _MISSING:
                    delattr(env, "close")
                else:
                    env.close = previous
                delattr(env, CLOSE_MARKER)
                original_close(*args, **kwargs)
        env.close = MethodType(close, env)
        setattr(env, CLOSE_MARKER, handler)
        return result
    return audit_material


def finalize_manifest(output_dir, *, child_returncode):
    """Parent must call after its exact owned child exits; no zero-exit shortcut."""
    output = Path(output_dir)
    path = output / "ablation_manifest.json"
    metadata = json.loads(path.read_text())
    module = collector()
    require(metadata["kind"] == KIND and metadata["wrapper_sha256"] == sha256(__file__) and
            metadata["sampler_module_sha256"] == sha256(module.__file__), "Finalizer source differs from prepared method")
    smoke = metadata["smoke_only"]
    require(type(smoke) is bool, "Smoke scope must be explicit boolean")
    expected_envs, expected_iterations = (2, 1) if smoke else (512, 1000)
    require(metadata["num_envs"] == expected_envs and metadata["iterations"] == expected_iterations and
            metadata["expected_policy_steps"] == 24 * expected_iterations and
            metadata["expected_environment_transitions"] == expected_envs * 24 * expected_iterations and
            metadata["seed"] == 42 and metadata["initialization"] == "from_scratch_no_checkpoint_resume",
            "Prepared formal/smoke training scope changed")
    require(metadata["baseline_action_rate_weight"] == -.5 and
            metadata["baseline_checkpoint_sha256"] == BASELINE_CHECKPOINT_SHA256 and
            metadata["baseline_metadata_sha256"] == BASELINE_METADATA and
            metadata["reward_replacement"]["term"] == TERM_NAME and
            metadata["reward_replacement"]["weight"] == .4 and
            metadata["reward_replacement"]["scale_fraction"] == .20 and
            metadata["reward_replacement"]["additional_reward_term"] is False,
            "Prepared baseline or same-name contact replacement changed")
    if child_returncode != 0:
        metadata.update(status="failed", child_returncode=child_returncode, trained_policy_claimed=False)
        path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        return metadata
    status_path, runtime_path, manifest_path = output / "run_status.json", output / RUNTIME_FILE, output / "manifest.json"
    status, runtime, manifest = [json.loads(p.read_text()) for p in (status_path, runtime_path, manifest_path)]
    smoke, expected = metadata["smoke_only"], metadata["expected_policy_steps"]
    require(status["status"] == ("smoke_passed" if smoke else "training_completed") and
            status["environment_transitions"] == metadata["expected_environment_transitions"] and
            status["policy_steps_completed"] == expected and
            status.get("trained_policy_claimed") is (not smoke), "Historical child did not complete exact scope")
    require(runtime["module_sha256"] == metadata["sampler_module_sha256"] and
            runtime["wrapper_sha256"] == metadata["wrapper_sha256"], "Runtime source differs")
    require(runtime["steps_started"] == runtime["steps_completed"] == runtime["unique_cost_windows"] == expected and
            runtime["physics_samples_total"] == 4 * expected and runtime["steps_failed"] == 0 and
            runtime["samples_last_window"] == 4 and runtime["device_health_finite"] is True and
            runtime["closed"] is True and runtime["num_envs"] == metadata["num_envs"], "Incomplete physical sampling audit")
    require(runtime["scale_fraction"] == .20, "Changed physical support reward scale")
    require(runtime["physics_dt_s"] == .005 and runtime["policy_dt_s"] == .02 and runtime["decimation"] == 4,
            "Actual ground-sampling clock changed")
    require(runtime["sensor_names"] == ["wheel_L_ground", "wheel_R_ground"] and
            runtime["wheel_body_names"] == ["wheel_L_Link", "wheel_R_Link"] and
            runtime["force_matrix_shape_each_sensor"] == [metadata["num_envs"], 1, 1, 3] and
            runtime["ground_filter"]["path"] == "/World/ground/terrain/GroundPlane/CollisionPlane" and
            runtime["ground_filter"]["collision_api_verified"] is True and
            runtime["cpu_fixture_only"] is False and runtime["device_health_checked"] is True and
            runtime["external_reward_weight"] == .4 and runtime["mass_gravity_reads"] == 1,
            "Physical sensor/filter/device/normalization contract differs")
    require(runtime["clock_helper_sha256"] == module.CLOCK_HELPER_SHA256 ==
            sha256(ROOT / "scripts/ppo_leg_torque_smoothness.py"), "Physical lifecycle dependency changed")
    require(runtime["raw_reward_bounds"] == [0., 1.] and
            runtime["gain_readback"] == {"kp_range": [500., 500.], "kd_range": [10., 10.]},
            "Changed grounded reward range or actual leg gains")
    for key in ("mass_kg_range", "mg_N_range"):
        values = runtime[key]
        require(len(values) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x) and x > 0 for x in values) and values[0] <= values[1],
                "Invalid live mass/gravity cache: " + key)
    gravity = runtime["gravity_m_s2"]
    require(len(gravity) == 3 and gravity[0] == gravity[1] == 0. and
            math.isfinite(gravity[2]) and gravity[2] < 0., "Invalid live gravity readback")
    require(runtime["negative_normal_force_samples"] == 0 and
            len(runtime["positive_normal_force_samples_left_right"]) == 2 and
            all(x > 0 for x in runtime["positive_normal_force_samples_left_right"]),
            "Actual smoke/training did not verify both upward ground-force channels")
    require(manifest.get("task_profile_sha256") == BASELINE_PROFILE_SHA256 and
            manifest.get("source_training_modules_sha256") == BASELINE_SOURCES and
            manifest.get("motion_sha256") == BASELINE_MOTION_SHA256 and
            manifest.get("asset_sha256") == BASELINE_ASSET_SHA256 and
            manifest.get("trainer_sha256") == BASELINE_TRAINER_SHA256,
            "Original profile/source manifest changed")
    checkpoint = output / ("model_smoke.pt" if smoke else "model_final.pt")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Missing nonempty scoped checkpoint")
    manifest["training_reward_overrides"] = {"action_rate_l2": -.5, TERM_NAME: metadata["reward_replacement"]}
    manifest["contact_reward_replacement"] = metadata["reward_replacement"]
    manifest["comparison_baseline_action_rate_weight"] = -.5
    manifest["reward_override_provenance_file"] = "ablation_manifest.json"
    manifest["historical_profile_hash_does_not_include_explicit_reward_override"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    metadata.update(status="smoke_passed" if smoke else "training_completed", child_returncode=0,
        trained_policy_claimed=not smoke, prepared_method_runtime_verified=True,
        actual_environment_transitions=status["environment_transitions"], checkpoint_sha256=sha256(checkpoint),
        final_training_manifest_sha256=sha256(manifest_path), runtime_audit_sha256=sha256(runtime_path),
        run_status_sha256=sha256(status_path),
        finalization_scope="Exact parent-observed child completion plus native four-substep ground-support audit")
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        print(__doc__); print("Required extra: --baseline-training-dir <completed fixed -0.5 metadata directory>")
        return
    plan = prepare_plan(arguments)
    spec = importlib.util.spec_from_file_location("_historical_ground_support_delegate", plan["trainer"])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    before_pre, before_post = module.install_beyondmimic_events, module.audit_material_draw
    module.install_beyondmimic_events = make_pre_env_hook(before_pre, plan)
    module.audit_material_draw = make_post_env_hook(before_post, plan)
    previous_argv = sys.argv
    try:
        sys.argv = [str(plan["trainer"]), *plan["inherited_argv"]]
        module.main()
    finally:
        module.install_beyondmimic_events, module.audit_material_draw = before_pre, before_post
        sys.argv = previous_argv


if __name__ == "__main__":
    main()

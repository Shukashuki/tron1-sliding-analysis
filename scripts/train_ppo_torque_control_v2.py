"""Prepared torque-rate addition to the original-grounded, action-rate -0.5 PPO.

From scratch, never warm-started. This is NOT the older -0.1 torque protocol or
the failed grounded-replacement experiment. Import launches no job; the owned
parent must finalize after Isaac app.close exits the exact training child.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType

import ppo_leg_torque_smoothness as torque
from train_ppo_smoothness import (
    BASELINE_ASSET_SHA256, BASELINE_HEIGHT_SHA256, BASELINE_MOTION_SHA256,
    BASELINE_PROFILE_SHA256, BASELINE_SOURCES, BASELINE_TRAINER_SHA256, _option, require, sha256,
)


ROOT = Path(__file__).resolve().parents[1]
KIND = "historical_physx_ppo_minus0p5_original_grounded_add_leg_torque_rate_v2"
TERM_NAME = "leg_applied_torque_rate"
WEIGHT = -.05
COLLECTOR_SHA256 = "2649dc39f08f8b540430617f7a2c8182527d895b62d5c515672e66c7c3e03aa7"
BASELINE_CHECKPOINT_SHA256 = "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a"
BASELINE_METADATA = {
    "manifest.json": "066ddaeb5a23ff44dfca79e785520c57d8786cdc3a2d7b30d118e3a955a94d63",
    "ablation_manifest.json": "4cf22f9c4e6b5f28aa1711f2e093516522046d259e0322b740e7cad5da05905e",
    "run_status.json": "fa46d5780c5753d6e01d9b19fccb2904d3d9cb06a17c1a894f7bd625c83ae773",
}
RUNTIME_FILE = "torque_rate_runtime_audit.json"
CLOSE_MARKER = "_tron1_torque_control_v2_close_owner"
GROUND_CONFLICTS = ("_tron1_substep_ground_support_accumulator", "_tron1_substep_ground_support_owner")
_MISSING = object()


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--baseline-training-dir", type=Path, required=True)
    parser.add_argument("--leg-torque-rate-weight", type=float, default=WEIGHT)
    options, inherited = parser.parse_known_args(list(argv))
    require(options.leg_torque_rate_weight == WEIGHT, "Prepared v2 fixes the one additional torque-rate weight at -0.05")
    require(not any(x.split("=", 1)[0] in ("--resume", "--action-rate-weight", "--checkpoint") for x in inherited),
            "Recipe-only baseline: v2 always starts from scratch, no resume or action-rate override")
    for flag in ("--source-root", "--motion-file", "--asset-path", "--task-profile", "--output-dir"):
        require(_option(inherited, flag) is not None, "Required historical trainer argument: "+flag)
    output = Path(_option(inherited, "--output-dir")).resolve()
    require(not output.exists(), "Preserve existing output: choose a fresh v2 directory")
    require(_option(inherited, "--reference-id", "drift_drl_traj4") == "drift_drl_traj4" and
            int(_option(inherited, "--seed", "42")) == 42 and
            _option(inherited, "--device", "cuda:1") == "cuda:1" and
            float(_option(inherited, "--ground-friction", "1")) == 1. and
            int(_option(inherited, "--smoke-steps", "0")) == 0, "Historical task/seed/ground/setup scope changed")
    smoke = "--smoke-only" in inherited
    envs, iterations = int(_option(inherited, "--num-envs", "512")), int(_option(inherited, "--iterations", "1000"))
    require((envs, iterations) == ((2, 1) if smoke else (512, 1000)), "Use exact 2x1 smoke or512x1000 formal scope")
    if _option(inherited, "--smoke-steps") is None:
        inherited += ["--smoke-steps", "0"]
    baseline = options.baseline_training_dir.resolve()
    for name, expected in BASELINE_METADATA.items():
        require((baseline/name).is_file() and sha256(baseline/name) == expected, "Fixed -0.5 baseline metadata differs: "+name)
    baseline_ablation = json.loads((baseline/"ablation_manifest.json").read_text())
    require(baseline_ablation.get("checkpoint_sha256") == BASELINE_CHECKPOINT_SHA256 and
            baseline_ablation.get("status") == "training_completed" and baseline_ablation.get("smoke_only") is False,
            "Baseline is a recipe reference only and must be the completed -0.5 cohort")
    source = Path(_option(inherited, "--source-root")).resolve()
    trainer = ROOT/"scripts/train_beyondmimic_style.py"
    paths = {trainer: BASELINE_TRAINER_SHA256, ROOT/"scripts/height_smoothness.py": BASELINE_HEIGHT_SHA256,
        Path(torque.__file__): COLLECTOR_SHA256, Path(_option(inherited, "--motion-file")): BASELINE_MOTION_SHA256,
        Path(_option(inherited, "--asset-path")): BASELINE_ASSET_SHA256,
        Path(_option(inherited, "--task-profile")): BASELINE_PROFILE_SHA256,
        **{source/name: expected for name, expected in BASELINE_SOURCES.items()}}
    for path, expected in paths.items():
        require(path.is_file() and sha256(path) == expected, "Historical/pinned source or input differs: "+str(path))
    return {"output_dir": output, "inherited_argv": inherited, "trainer": trainer, "source_root": source,
        "baseline_training_dir": baseline, "smoke_only": smoke, "num_envs": envs, "iterations": iterations,
        "expected_policy_steps": 24*iterations, "expected_environment_transitions": envs*24*iterations}


def _reward_names(rewards, term_type):
    return sorted(name for name in dir(rewards) if not name.startswith("_") and
        isinstance(getattr(rewards, name), term_type) and getattr(rewards, name).weight != 0.)


def _original_grounded(term):
    return term.weight == .4 and getattr(term.func, "__name__", None) == "grounded_reward" and \
        getattr(term.func, "__module__", None) == "training.tron1_drift"


def make_pre_env_hook(original, plan):
    def install(cfg, *, ground_friction=1.):
        recipe = original(cfg, ground_friction=ground_friction)
        from isaaclab.managers import RewardTermCfg
        action = cfg.rewards.action_rate_l2
        require(action.weight == -.1 and action.func.__name__ == "action_rate_l2" and
            action.func.__module__ == "isaaclab.envs.mdp.rewards", "Original source action-rate term differs")
        require(_original_grounded(cfg.rewards.drift_grounded), "Must restore original binary grounded+0.4, not the replacement reward")
        require(cfg.scene.robot.actuators["legs"].stiffness == 500. and cfg.scene.robot.actuators["legs"].damping == 10. and
            cfg.sim.dt == .005 and cfg.decimation == 4, "Historical PD/physics clock differs")
        require(all(getattr(cfg.scene, name, None) is None for name in ("wheel_L_ground", "wheel_R_ground")),
            "V2 adds no ground-support sensors or sensor observations")
        before = _reward_names(cfg.rewards, RewardTermCfg)
        require(len(before) == 14 and TERM_NAME not in before and getattr(cfg.rewards, TERM_NAME, None) is None and
            getattr(cfg.rewards, "height_vertical_acceleration", None) is None,
            "Event hook precedes pinned height adapter: require original14 pre-height rewards")
        require(sha256(torque.__file__) == COLLECTOR_SHA256, "Pinned all-substep torque collector changed")
        action.weight = -.5
        setattr(cfg.rewards, TERM_NAME, RewardTermCfg(func=torque.make_reward_term(), weight=WEIGHT))
        after = _reward_names(cfg.rewards, RewardTermCfg)
        require(len(after) == 15 and set(after)-set(before) == {TERM_NAME}, "V2 adds only one pre-height reward; pinned height adapter then makes16")
        expected_before = sorted([*before, "height_vertical_acceleration"])
        expected_after = sorted([*after, "height_vertical_acceleration"])
        metadata = {"schema_version": 1, "kind": KIND, "status": "configured_before_environment_creation",
            "initialization": "from_scratch_no_checkpoint_resume", "smoke_only": plan["smoke_only"],
            "trained_policy_claimed": False, "prepared_method_runtime_verified": False,
            "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256, "baseline_metadata_sha256": dict(BASELINE_METADATA),
            "baseline_checkpoint_used_for_initialization": False, "baseline_action_rate_weight": -.5,
            "source_default_action_rate_weight": -.1, "original_grounded_reward_preserved": True,
            "original_grounded_weight": .4, "original_grounded_function": "training.tron1_drift.grounded_reward",
            "pre_height_active_reward_terms_before": before, "pre_height_active_reward_terms_after": after,
            "expected_final_active_reward_terms_before": expected_before,
            "expected_final_active_reward_terms_after": expected_after,
            "reward_count_lifecycle": "event hook14->15; pinned height adapter adds vertical acceleration; actual RewardManager16 must be audited after environment creation",
            "additional_reward": {"term": TERM_NAME, "weight": WEIGHT, "additional_reward_term": True,
                "formula": "mean_4_substeps(mean_6_legs((delta_applied_tau_Nm/20_Nm)^2))",
                "delta_torque_scale_nm": 20., "physics_dt_s": .005, "policy_dt_s": .02,
                "rated_torque_nm": 80., "torque_rate_scale_nm_s": 4000., "first_actual_substep_after_reset_cost": 0.,
                "denominator_after_reset": 4, "wheel_cost_included": False, "reward_manager_dt_factor_applied_once": .02},
            "num_envs": plan["num_envs"], "iterations": plan["iterations"], "seed": 42,
            "expected_policy_steps": plan["expected_policy_steps"], "expected_environment_transitions": plan["expected_environment_transitions"],
            "historical_trainer_sha256": BASELINE_TRAINER_SHA256, "height_adapter_sha256": BASELINE_HEIGHT_SHA256,
            "task_profile_sha256": BASELINE_PROFILE_SHA256, "motion_sha256": BASELINE_MOTION_SHA256,
            "asset_sha256": BASELINE_ASSET_SHA256, "baseline_source_sha256": dict(BASELINE_SOURCES),
            "wrapper_sha256": sha256(__file__), "sampler_module_sha256": COLLECTOR_SHA256,
            "inherited_training_arguments": plan["inherited_argv"], "base_task_profile_file_modified": False,
            "new_observation_terms": False, "new_absolute_height_target": False, "new_zero_yaw_or_beta_target": False,
            "actual_isaac_smoke_required_before_formal_adoption": True}
        require(plan["output_dir"].is_dir(), "Historical trainer must first create fresh output")
        with (plan["output_dir"]/"ablation_manifest.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False); stream.write("\n")
        return recipe
    return install


def _runtime_recipe(env, handler):
    """Startup-only explicit-actuator PD audit; not native PhysX drive gains."""
    import torch
    active = sorted(env.reward_manager.active_terms)
    require(len(active) == 16 and TERM_NAME in active, "Actual RewardManager must contain exactly16 active terms")
    require(_original_grounded(env.cfg.rewards.drift_grounded) and env.cfg.rewards.action_rate_l2.weight == -.5,
        "Actual original-grounded/action-rate baseline changed")
    require(not any(hasattr(owner, name) for owner in (env, env.scene) for name in GROUND_CONFLICTS),
        "Grounded-replacement collector must not coexist with torque-control v2")
    gain = {}
    legs = env.scene["robot"].actuators["legs"]
    for name, expected, label in (("stiffness", 500., "kp_range"), ("damping", 10., "kd_range")):
        values = getattr(legs, name)
        require(torch.isfinite(values).all().item() and torch.allclose(values, torch.full_like(values, expected)),
            "Actual explicit DC motor leg PD differs: "+name)
        gain[label] = [float(values.min().item()), float(values.max().item())]
    return {"gain_readback": gain, "gain_readback_source": "robot.actuators['legs'].stiffness/damping; explicit DC motor software PD, not native PhysX drive gains",
        "active_reward_terms": active, "active_reward_count": 16,
        "original_grounded_reward_preserved": True, "original_grounded_weight": .4,
        "original_grounded_function": "training.tron1_drift.grounded_reward", "action_rate_weight": -.5,
        "additional_reward_weight": WEIGHT, "new_contact_sensors_added": False, "new_observation_terms": False}


def make_post_env_hook(original, plan):
    def audit_material(env, recipe):
        result = original(env, recipe)
        require(not any(hasattr(owner, name) for owner in (env, env.scene) for name in GROUND_CONFLICTS),
            "Ground-support replacement runtime owner exists")
        handler = torque.install_accumulator(env)
        if hasattr(env, CLOSE_MARKER):
            require(getattr(env, CLOSE_MARKER) is handler, "Conflicting v2 close owner")
            return result
        runtime_recipe = _runtime_recipe(env, handler)
        previous, original_close = vars(env).get("close", _MISSING), env.close
        def close(owner, *args, **kwargs):
            require(owner is env and getattr(env, CLOSE_MARKER, None) is handler, "V2 close ownership changed")
            try:
                audit = handler.audit(check_device_health=True)
                audit.update(runtime_recipe, wrapper_sha256=sha256(__file__),
                    last_native_leg_applied_torque_abs_peak_nm=float(handler.previous.abs().max().item()),
                    last_window_raw_cost_range=[float(handler.cached_cost.min().item()), float(handler.cached_cost.max().item())],
                    torque_telemetry_scope="Last actual native sample and last control window only; not all-substep extrema")
                handler.close(); audit["closed"] = True
                with (plan["output_dir"]/RUNTIME_FILE).open("x") as stream:
                    json.dump(audit, stream, indent=2, allow_nan=False); stream.write("\n")
                require(audit["device_health_finite"] is True, "Nonfinite native applied torque sampled")
            finally:
                if previous is _MISSING: delattr(env, "close")
                else: env.close = previous
                delattr(env, CLOSE_MARKER)
                original_close(*args, **kwargs)
        env.close = MethodType(close, env); setattr(env, CLOSE_MARKER, handler)
        return result
    return audit_material


def verify_runtime_audit(runtime, *, smoke, wrapper_sha256=None):
    import math
    steps, envs = (24, 2) if smoke else (24000, 512)
    require(runtime.get("module_sha256") == COLLECTOR_SHA256 and
        (wrapper_sha256 is None or runtime.get("wrapper_sha256") == wrapper_sha256), "Torque runtime source differs")
    require(runtime.get("steps_started") == runtime.get("steps_completed") == runtime.get("unique_cost_windows") == steps and
        runtime.get("physics_samples_total") == 4*steps and runtime.get("steps_failed") == 0 and
        runtime.get("samples_last_window") == 4 and runtime.get("num_envs") == envs and runtime.get("closed") is True and
        runtime.get("device_health_checked") is True and runtime.get("device_health_finite") is True and runtime.get("device") == "cuda:1",
        "Incomplete/unhealthy actual GPU torque sampling audit")
    require(runtime.get("physics_dt_s") == .005 and runtime.get("policy_dt_s") == .02 and runtime.get("decimation") == 4 and
        runtime.get("delta_torque_scale_nm") == 20. and runtime.get("torque_rate_scale_nm_s") == 4000. and
        runtime.get("wheel_cost_included") is False and runtime.get("first_actual_substep_after_reset_cost") == 0. and
        runtime.get("denominator_after_reset") == 4 and runtime.get("leg_joint_names") == list(torque.LEG_JOINT_NAMES),
        "Changed torque units/six-leg/reset/clock contract")
    require(runtime.get("gain_readback") == {"kp_range": [500., 500.], "kd_range": [10., 10.]} and
        runtime.get("active_reward_count") == len(runtime.get("active_reward_terms", [])) == 16 and
        TERM_NAME in runtime["active_reward_terms"] and runtime.get("original_grounded_reward_preserved") is True and
        len(set(runtime["active_reward_terms"])) == 16 and
        runtime.get("original_grounded_function") == "training.tron1_drift.grounded_reward" and
        runtime.get("original_grounded_weight") == .4 and runtime.get("action_rate_weight") == -.5 and
        runtime.get("additional_reward_weight") == WEIGHT and runtime.get("new_contact_sensors_added") is False and
        runtime.get("new_observation_terms") is False, "Runtime changed the baseline beyond the16th torque term")
    peak, costs = runtime.get("last_native_leg_applied_torque_abs_peak_nm"), runtime.get("last_window_raw_cost_range", [])
    require(isinstance(peak, (float, int)) and math.isfinite(peak) and 0 < peak <= 80.0001 and
        len(costs) == 2 and all(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in costs) and costs[0] <= costs[1],
        "Actual torque smoke lacks finite nonzero bounded Nm telemetry/normalized cost")


def finalize_manifest(output_dir, *, child_returncode):
    output = Path(output_dir); path = output/"ablation_manifest.json"
    metadata = json.loads(path.read_text())
    require(metadata.get("kind") == KIND and metadata.get("wrapper_sha256") == sha256(__file__) and
        metadata.get("sampler_module_sha256") == sha256(torque.__file__) == COLLECTOR_SHA256, "V2 finalizer source differs")
    smoke = metadata.get("smoke_only")
    require(type(smoke) is bool, "Smoke scope must be explicit")
    steps, envs, iterations = (24, 2, 1) if smoke else (24000, 512, 1000)
    require(metadata.get("initialization") == "from_scratch_no_checkpoint_resume" and
        metadata.get("baseline_checkpoint_used_for_initialization") is False and
        metadata.get("num_envs") == envs and metadata.get("iterations") == iterations and metadata.get("seed") == 42 and
        metadata.get("expected_policy_steps") == steps and metadata.get("expected_environment_transitions") == steps*envs and
        metadata.get("baseline_checkpoint_sha256") == BASELINE_CHECKPOINT_SHA256 and metadata.get("baseline_metadata_sha256") == BASELINE_METADATA and
        metadata.get("baseline_action_rate_weight") == -.5 and metadata.get("original_grounded_reward_preserved") is True and
        metadata.get("additional_reward", {}).get("weight") == WEIGHT, "V2 configured comparison scope changed")
    if child_returncode != 0:
        metadata.update(status="failed", child_returncode=child_returncode, trained_policy_claimed=False)
        path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n"); return metadata
    status, runtime, manifest = [json.loads((output/name).read_text()) for name in ("run_status.json", RUNTIME_FILE, "manifest.json")]
    require(status.get("status") == ("smoke_passed" if smoke else "training_completed") and
        status.get("trained_policy_claimed") is (not smoke) and status.get("policy_steps_completed") == steps and
        status.get("environment_transitions") == steps*envs, "Owned child did not complete exact collection budget")
    verify_runtime_audit(runtime, smoke=smoke, wrapper_sha256=metadata["wrapper_sha256"])
    require(runtime["active_reward_terms"] == metadata.get("expected_final_active_reward_terms_after"), "Actual post-height reward names differ from pre-environment declaration")
    for key, value in (("trainer_sha256", BASELINE_TRAINER_SHA256), ("task_profile_sha256", BASELINE_PROFILE_SHA256),
        ("source_training_modules_sha256", BASELINE_SOURCES), ("motion_sha256", BASELINE_MOTION_SHA256), ("asset_sha256", BASELINE_ASSET_SHA256)):
        require(manifest.get(key) == value, "Historical training manifest changed: "+key)
    checkpoint = output/("model_smoke.pt" if smoke else "model_final.pt")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Missing nonempty scoped checkpoint")
    manifest.update(training_reward_overrides={"action_rate_l2": -.5}, additional_training_reward_terms={TERM_NAME: metadata["additional_reward"]},
        original_grounded_reward_preserved=True, baseline_action_rate_weight=-.5, reward_override_provenance_file="ablation_manifest.json",
        historical_profile_hash_does_not_include_additional_reward=True)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    metadata.update(status="smoke_passed" if smoke else "training_completed", child_returncode=0, trained_policy_claimed=not smoke,
        prepared_method_runtime_verified=True, actual_environment_transitions=steps*envs, checkpoint_sha256=sha256(checkpoint),
        final_training_manifest_sha256=sha256(output/"manifest.json"), runtime_audit_sha256=sha256(output/RUNTIME_FILE),
        run_status_sha256=sha256(output/"run_status.json"), finalization_scope="Exact owned child exit plus native4-substep torque and original-grounded baseline audits")
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n"); return metadata


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        print(__doc__); print("Required --baseline-training-dir; fixed --leg-torque-rate-weight -0.05. Historical trainer flags pass through."); return
    plan = prepare_plan(arguments)
    spec = importlib.util.spec_from_file_location("historical_torque_control_v2_delegate", plan["trainer"])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    before_pre, before_post = module.install_beyondmimic_events, module.audit_material_draw
    module.install_beyondmimic_events = make_pre_env_hook(before_pre, plan)
    module.audit_material_draw = make_post_env_hook(before_post, plan)
    previous = sys.argv; sys.argv = [str(plan["trainer"]), *plan["inherited_argv"]]
    try: module.main()
    finally:
        module.install_beyondmimic_events, module.audit_material_draw = before_pre, before_post
        sys.argv = previous


if __name__ == "__main__":
    main()

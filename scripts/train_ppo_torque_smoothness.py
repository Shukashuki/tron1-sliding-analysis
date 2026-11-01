"""Prepared isolated leg applied-torque-rate ablation of historical PhysX PPO.

No job is launched by import. Actual adoption requires a separately authorized
Isaac smoke. Parent orchestration MUST call finalize_manifest after its exact
child exits: Isaac app.close may terminate before Python returns from main.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import MethodType

import ppo_leg_torque_smoothness as torque
from train_ppo_smoothness import (
    BASELINE_ASSET_SHA256, BASELINE_HEIGHT_SHA256, BASELINE_MOTION_SHA256,
    BASELINE_PROFILE_SHA256, BASELINE_SOURCES, BASELINE_TRAINER_SHA256,
    _option, require, sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TERM_NAME = "leg_applied_torque_rate"
_MISSING = object()
CLOSE_MARKER = "_tron1_leg_torque_runtime_close_owner"


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--leg-torque-rate-weight", type=float, default=-.01)
    options, inherited = parser.parse_known_args(list(argv))
    require(math.isfinite(options.leg_torque_rate_weight) and -.5 <= options.leg_torque_rate_weight < 0,
            "Leg torque-rate weight must be finite in [-0.5,0)")
    for flag in ("--source-root", "--motion-file", "--asset-path", "--task-profile", "--output-dir"):
        require(_option(inherited, flag) is not None, "Required inherited argument: "+flag)
    require(not any(value.split("=", 1)[0] in ("--resume", "--action-rate-weight") for value in inherited),
            "This new ablation starts from scratch with baseline action-rate weight -0.1")
    output = Path(_option(inherited, "--output-dir")).resolve()
    require(not output.exists(), "Preserve existing output; choose a fresh directory")
    require(_option(inherited, "--reference-id", "drift_drl_traj4") == "drift_drl_traj4", "Require primary traj_4")
    require(int(_option(inherited, "--seed", "42")) == 42 and
            float(_option(inherited, "--ground-friction", "1")) == 1., "Preserve historical seed and ground friction")
    require(int(_option(inherited, "--smoke-steps", "0")) == 0, "Use separately labelled smoke, no setup control steps")
    smoke = "--smoke-only" in inherited
    envs = int(_option(inherited, "--num-envs", "512")); iterations = int(_option(inherited, "--iterations", "1000"))
    require((envs, iterations) == ((2, 1) if smoke else (512, 1000)), "Preserve formal budget or exact 2-env/1-iteration smoke")
    if _option(inherited, "--smoke-steps") is None:
        inherited += ["--smoke-steps", "0"]
    source = Path(_option(inherited, "--source-root")).resolve()
    trainer = ROOT/"scripts/train_beyondmimic_style.py"
    paths = {trainer: BASELINE_TRAINER_SHA256, ROOT/"scripts/height_smoothness.py": BASELINE_HEIGHT_SHA256,
        Path(_option(inherited, "--task-profile")): BASELINE_PROFILE_SHA256,
        Path(_option(inherited, "--motion-file")): BASELINE_MOTION_SHA256,
        Path(_option(inherited, "--asset-path")): BASELINE_ASSET_SHA256,
        **{source/name: digest for name, digest in BASELINE_SOURCES.items()}}
    for path, digest in paths.items():
        require(path.is_file() and sha256(path) == digest, "Historical source/input SHA mismatch: "+str(path))
    return {"weight": options.leg_torque_rate_weight, "output_dir": output, "inherited_argv": inherited,
            "trainer": trainer, "source_root": source, "smoke_only": smoke,
            "num_envs": envs, "iterations": iterations, "expected_policy_steps": 24*iterations,
            "expected_environment_transitions": envs*24*iterations}


def make_pre_env_hook(original, plan):
    def install(cfg, *, ground_friction=1.):
        recipe = original(cfg, ground_friction=ground_friction)
        action = cfg.rewards.action_rate_l2
        require(action.weight == -.1 and action.func.__name__ == "action_rate_l2" and
                action.func.__module__ == "isaaclab.envs.mdp.rewards", "Historical action-rate term changed")
        require(getattr(cfg.rewards, TERM_NAME, None) is None, "Torque-rate reward already exists")
        require(cfg.scene.robot.actuators["legs"].stiffness == 500. and
                cfg.scene.robot.actuators["legs"].damping == 10., "Historical leg PD changed")
        from isaaclab.managers import RewardTermCfg
        setattr(cfg.rewards, TERM_NAME, RewardTermCfg(func=torque.make_reward_term(), weight=plan["weight"]))
        metadata = {"schema_version": 1, "kind": "historical_physx_ppo_leg_applied_torque_rate_ablation",
            "status": "configured_before_environment_creation", "prepared_method_runtime_verified": False,
            "initialization": "from_scratch_no_checkpoint_resume", "smoke_only": plan["smoke_only"],
            "trained_policy_claimed": False, "baseline_action_rate_weight": -.1,
            "additional_reward": {"term": TERM_NAME, "weight": plan["weight"],
                "formula": "mean_4_substeps(mean_6_legs((delta_applied_tau_Nm/20_Nm)^2))",
                "rated_torque_nm": 80., "reference_time_scale_s": .02, "physics_dt_s": .005,
                "torque_rate_scale_nm_s": 4000., "delta_torque_scale_nm": 20.,
                "first_actual_substep_after_reset_cost": 0., "denominator_after_reset": 4,
                "wheel_cost_included": False, "reward_manager_dt_factor_applied_once": .02},
            "expected_policy_steps": plan["expected_policy_steps"],
            "expected_environment_transitions": plan["expected_environment_transitions"],
            "num_envs": plan["num_envs"], "iterations": plan["iterations"], "seed": 42,
            "historical_trainer_sha256": BASELINE_TRAINER_SHA256,
            "height_adapter_sha256": BASELINE_HEIGHT_SHA256, "task_profile_sha256": BASELINE_PROFILE_SHA256,
            "motion_sha256": BASELINE_MOTION_SHA256, "asset_sha256": BASELINE_ASSET_SHA256,
            "baseline_source_sha256": dict(BASELINE_SOURCES), "wrapper_sha256": sha256(__file__),
            "sampler_module_sha256": sha256(torque.__file__),
            "inherited_training_arguments": plan["inherited_argv"],
            "unchanged": ["baseline action-rate -0.1", "old DR", "height smoothness", "nonzero yaw/beta reference",
                          "Kp=500/Kd=10", "1-rad action scale", "DC motors", "PPO config", "reference", "formal budget"],
            "new_absolute_height_target": False, "new_zero_yaw_or_beta_target": False}
        require(plan["output_dir"].is_dir(), "Historical trainer must create its fresh output first")
        with (plan["output_dir"]/"ablation_manifest.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False); stream.write("\n")
        return recipe
    return install


def make_post_env_hook(original, plan):
    """Install after material audit, before RSL reset/first environment step."""
    def audit_material(env, recipe):
        result = original(env, recipe)
        handler = torque.install_accumulator(env)
        owner = getattr(env, CLOSE_MARKER, None)
        if owner is not None:
            require(owner is handler, "Conflicting environment-close owner")
            return result
        original_close = env.close
        previous_close = vars(env).get("close", _MISSING)
        def close(owner, *args, **kwargs):
            require(owner is env and getattr(env, CLOSE_MARKER, None) is handler,
                    "Environment-close ownership changed")
            runtime_audit = handler.audit(check_device_health=True)
            try:
                handler.close()
                runtime_audit["closed"] = True
                runtime_audit["wrapper_sha256"] = sha256(__file__)
                path = plan["output_dir"]/"torque_rate_runtime_audit.json"
                with path.open("x") as stream:
                    json.dump(runtime_audit, stream, indent=2, allow_nan=False); stream.write("\n")
            finally:
                if previous_close is _MISSING:
                    delattr(env, "close")
                else:
                    env.close = previous_close
                delattr(env, CLOSE_MARKER)
                original_close(*args, **kwargs)
            require(runtime_audit["device_health_finite"], "Nonfinite physical applied torque was sampled")
        env.close = MethodType(close, env); setattr(env, CLOSE_MARKER, handler)
        return result
    return audit_material


def finalize_manifest(output_dir, *, child_returncode):
    """Parent-only completion gate; call after the exact owned child exits.

    A zero exit alone is insufficient: require old trainer status/budget, the
    full physical sampling audit, and the actual saved checkpoint bytes.
    """
    output = Path(output_dir)
    path = output/"ablation_manifest.json"
    metadata = json.loads(path.read_text())
    require(metadata["kind"] == "historical_physx_ppo_leg_applied_torque_rate_ablation", "Wrong ablation kind")
    require(metadata["wrapper_sha256"] == sha256(__file__) and
            metadata["sampler_module_sha256"] == sha256(torque.__file__), "Finalizer source differs from configured method")
    if child_returncode != 0:
        metadata.update(status="failed", trained_policy_claimed=False, child_returncode=child_returncode)
        path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
        return metadata
    status_path, runtime_path = output/"run_status.json", output/"torque_rate_runtime_audit.json"
    status, runtime = json.loads(status_path.read_text()), json.loads(runtime_path.read_text())
    smoke = metadata["smoke_only"]
    require(status["status"] == ("smoke_passed" if smoke else "training_completed"), "Historical child did not complete scope")
    require(status["environment_transitions"] == metadata["expected_environment_transitions"], "Completed budget differs")
    require(status.get("trained_policy_claimed") is (not smoke), "Historical trained-policy claim differs from scope")
    require(runtime["module_sha256"] == metadata["sampler_module_sha256"] and
            runtime["wrapper_sha256"] == metadata["wrapper_sha256"], "Runtime sampler/wrapper source differs")
    expected = metadata["expected_policy_steps"]
    require(runtime["steps_started"] == runtime["steps_completed"] == runtime["unique_cost_windows"] == expected and
            runtime["physics_samples_total"] == 4*expected and runtime["steps_failed"] == 0 and
            runtime["samples_last_window"] == 4 and runtime["device_health_finite"] is True and runtime["closed"] is True,
            "Runtime did not sample exactly every physical substep and reward window")
    require(runtime["num_envs"] == metadata["num_envs"] and runtime["delta_torque_scale_nm"] == 20. and
            runtime["wheel_cost_included"] is False, "Runtime reward configuration differs")
    checkpoint = output/("model_smoke.pt" if smoke else "model_final.pt")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Completed child did not save its nonempty scoped checkpoint")
    manifest_path = output/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["additional_training_reward_terms"] = {TERM_NAME: metadata["additional_reward"]}
    manifest["baseline_action_rate_weight"] = -.1
    manifest["reward_override_provenance_file"] = "ablation_manifest.json"
    manifest["historical_profile_hash_does_not_include_additional_reward"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    metadata.update(status="smoke_passed" if smoke else "training_completed", child_returncode=0,
        trained_policy_claimed=not smoke, actual_environment_transitions=status["environment_transitions"],
        checkpoint_sha256=sha256(checkpoint), final_training_manifest_sha256=sha256(manifest_path),
        runtime_audit_sha256=sha256(runtime_path), run_status_sha256=sha256(status_path),
        prepared_method_runtime_verified=True, finalization_scope="Exact parent-observed child completion plus native counter audit")
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return metadata


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        print(__doc__); print("Additional option: --leg-torque-rate-weight -0.01; all historical trainer arguments pass through.")
        return
    plan = prepare_plan(arguments)
    spec = importlib.util.spec_from_file_location("historical_leg_torque_delegate", plan["trainer"])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    before_pre, before_post = module.install_beyondmimic_events, module.audit_material_draw
    module.install_beyondmimic_events = make_pre_env_hook(before_pre, plan)
    module.audit_material_draw = make_post_env_hook(before_post, plan)
    previous_argv = sys.argv
    sys.argv = [str(plan["trainer"]), *plan["inherited_argv"]]
    try:
        module.main()
    finally:
        module.install_beyondmimic_events, module.audit_material_draw = before_pre, before_post
        sys.argv = previous_argv
    # Deliberately do not finalize here: real Isaac app.close can exit before
    # reaching this line. The parent must inspect its own child's exact exit.


if __name__ == "__main__":
    main()

"""Isolated action-rate ablation of the SHA-locked historical PhysX PPO trainer.

All arguments except --action-rate-weight pass to train_beyondmimic_style.py.
No upstream file is modified. A pre-environment hook changes one reward weight;
the explicit ablation manifest distinguishes this run from its frozen baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BASELINE_TRAINER_SHA256 = "880e84b6394ab99e51aa610ee9db778e113b173012a13db6e76284167983ca5d"
BASELINE_HEIGHT_SHA256 = "e133aca58f671fc22ae1dfc97a03461e7724e4072bf8b74f58a2c14281d198b8"
BASELINE_PROFILE_SHA256 = "31de29a61eec847e0e39783f3a0d9d8f2db551acd679a1b38dfde2b24e8ae7d6"
BASELINE_MOTION_SHA256 = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
BASELINE_ASSET_SHA256 = "327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e"
BASELINE_SOURCES = {
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_drift.py": "5eb40c9b3c14f5bd5c59c9861c1414ae8ac1b219ff0dd70591f2c6fe0bb5344c",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _option(arguments, flag, default=None):
    found = []
    for index, value in enumerate(arguments):
        if value == flag:
            require(index+1 < len(arguments), "Missing value for " + flag)
            found.append(arguments[index+1])
        elif value.startswith(flag+"="):
            found.append(value.split("=", 1)[1])
    require(len(found) <= 1, "Duplicate option: " + flag)
    return found[0] if found else default


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--action-rate-weight", type=float, default=-.3)
    options, inherited = parser.parse_known_args(list(argv))
    weight = options.action_rate_weight
    require(math.isfinite(weight) and -10. <= weight < -.1,
            "Ablation requires a finite stronger negative action-rate weight in [-10,-0.1)")
    for flag in ("--source-root", "--motion-file", "--asset-path", "--task-profile", "--output-dir"):
        require(_option(inherited, flag) is not None, "Required inherited argument: " + flag)
    require(not any(value == "--resume" or value.startswith("--resume=") for value in inherited),
            "This ablation starts from scratch; resume is not supported")
    output = Path(_option(inherited, "--output-dir")).resolve()
    require(not output.exists(), "Preserve existing runs: output directory must not exist")
    source = Path(_option(inherited, "--source-root")).resolve()
    require(_option(inherited, "--reference-id", "drift_drl_traj4") == "drift_drl_traj4",
            "This isolated ablation requires the historical primary traj_4 reference")
    smoke = "--smoke-only" in inherited
    envs = int(_option(inherited, "--num-envs", "512"))
    iterations = int(_option(inherited, "--iterations", "1000"))
    seed = int(_option(inherited, "--seed", "42"))
    require(seed == 42 and float(_option(inherited, "--ground-friction", "1.0")) == 1.,
            "Preserve historical seed=42 and ground friction=1.0")
    require(int(_option(inherited, "--smoke-steps", "0")) == 0,
            "Keep setup smoke steps=0; use the separately labelled --smoke-only run")
    if smoke:
        require(envs == 2 and iterations == 1, "Smoke must use exactly 2 environments and 1 PPO iteration")
    else:
        require(envs == 512 and iterations == 1000, "Formal comparison preserves 512 environments x 1000 iterations")
    if _option(inherited, "--smoke-steps") is None:
        inherited.extend(["--smoke-steps", "0"])
    trainer = ROOT / "scripts/train_beyondmimic_style.py"
    height = ROOT / "scripts/height_smoothness.py"
    profile = Path(_option(inherited, "--task-profile")).resolve()
    expected = {trainer: BASELINE_TRAINER_SHA256, height: BASELINE_HEIGHT_SHA256,
                profile: BASELINE_PROFILE_SHA256,
                Path(_option(inherited, "--motion-file")).resolve(): BASELINE_MOTION_SHA256,
                Path(_option(inherited, "--asset-path")).resolve(): BASELINE_ASSET_SHA256,
                **{source/name: digest for name, digest in BASELINE_SOURCES.items()}}
    for path, digest in expected.items():
        require(path.is_file() and sha256(path) == digest, "Historical source/profile SHA mismatch: " + str(path))
    return {"action_rate_weight": weight, "inherited_argv": inherited,
            "output_dir": output, "source_root": source, "trainer": trainer,
            "smoke_only": smoke, "num_envs": envs, "iterations": iterations, "seed": seed,
            "baseline_source_sha256": dict(BASELINE_SOURCES)}


def make_override_hook(original, plan):
    """Install exactly one reward override after unchanged historical DR setup."""
    def install(cfg, *, ground_friction=1.):
        recipe = original(cfg, ground_friction=ground_friction)
        term = cfg.rewards.action_rate_l2
        require(term.weight == -.1, "Unexpected baseline action-rate weight")
        require(getattr(term.func, "__name__", None) == "action_rate_l2" and
                getattr(term.func, "__module__", None) == "isaaclab.envs.mdp.rewards",
                "Unexpected baseline action-rate reward function")
        require(plan["output_dir"].is_dir(), "Historical trainer must create its fresh output first")
        term.weight = plan["action_rate_weight"]
        metadata = {
            "schema_version": 1, "kind": "historical_physx_ppo_action_rate_ablation",
            "status": "configured_before_environment_creation", "algorithm": "PPO",
            "training_engine": "PhysX", "initialization": "from_scratch_no_checkpoint_resume",
            "smoke_only": plan["smoke_only"], "trained_policy_claimed": False,
            "seed": plan["seed"], "num_envs": plan["num_envs"], "iterations": plan["iterations"],
            "expected_environment_transitions": plan["num_envs"]*plan["iterations"]*24,
            "reward_override": {"term": "action_rate_l2", "baseline_weight": -.1,
                "weight": term.weight, "function": term.func.__module__+"."+term.func.__name__,
                "raw_term": "sum((clipped_normalized_action - previous_clipped_normalized_action)^2)",
                "divided_by_control_dt": False, "applied_before_environment_creation": True},
            "only_intended_experimental_change": "action_rate_l2 weight",
            "unchanged": ["historical DR", "height regularization", "PPO config", "Kp/Kd", "action scale",
                          "actuator model", "reference", "seed", "formal training budget"],
            "historical_trainer_sha256": sha256(plan["trainer"]),
            "height_adapter_sha256": BASELINE_HEIGHT_SHA256,
            "task_profile_sha256": BASELINE_PROFILE_SHA256,
            "motion_sha256": BASELINE_MOTION_SHA256,
            "asset_sha256": BASELINE_ASSET_SHA256,
            "baseline_source_sha256": plan["baseline_source_sha256"],
            "wrapper_sha256": sha256(__file__),
            "inherited_training_arguments": plan["inherited_argv"],
            "DR_recipe_modified": False, "base_task_profile_file_modified": False,
            "baseline_checkpoint_modified": False}
        with (plan["output_dir"]/"ablation_manifest.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False); stream.write("\n")
        return recipe
    return install


def finalize_manifest(plan, *, failed=False):
    path = plan["output_dir"]/"ablation_manifest.json"
    if not path.exists():
        return
    metadata = json.loads(path.read_text())
    status_path = plan["output_dir"]/"run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    metadata["status"] = "failed" if failed else status.get("status", "delegate_returned_without_status")
    metadata["trained_policy_claimed"] = not failed and status.get("trained_policy_claimed") is True
    checkpoint = plan["output_dir"]/("model_smoke.pt" if plan["smoke_only"] else "model_final.pt")
    if not failed:
        require(status.get("status") == ("smoke_passed" if plan["smoke_only"] else "training_completed"),
                "Historical trainer did not complete the requested scope")
        require(status.get("environment_transitions") == metadata["expected_environment_transitions"],
                "Actual transition count differs from the frozen budget")
        require(checkpoint.is_file(), "Completed run has no final checkpoint")
        metadata["checkpoint_sha256"] = sha256(checkpoint)
    metadata["actual_environment_transitions"] = status.get("environment_transitions")
    manifest_path = plan["output_dir"]/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["training_reward_overrides"] = {"action_rate_l2": plan["action_rate_weight"]}
        manifest["reward_override_provenance_file"] = "ablation_manifest.json"
        manifest["historical_profile_hash_does_not_include_explicit_reward_override"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        metadata["final_training_manifest_sha256"] = sha256(manifest_path)
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        print(__doc__)
        print("Additional option: --action-rate-weight FLOAT (default -0.3). All historical trainer arguments are inherited.")
        return
    plan = prepare_plan(arguments)
    spec = importlib.util.spec_from_file_location("tron1_historical_smoothness_delegate", plan["trainer"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.install_beyondmimic_events
    module.install_beyondmimic_events = make_override_hook(original, plan)
    previous_argv = sys.argv
    sys.argv = [str(plan["trainer"]), *plan["inherited_argv"]]
    try:
        module.main()
        finalize_manifest(plan)
    except BaseException:
        finalize_manifest(plan, failed=True)
        raise
    finally:
        module.install_beyondmimic_events = original
        sys.argv = previous_argv


if __name__ == "__main__":
    main()

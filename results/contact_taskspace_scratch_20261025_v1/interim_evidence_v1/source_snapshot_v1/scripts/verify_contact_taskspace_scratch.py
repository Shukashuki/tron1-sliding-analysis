"""Independent admission of fresh K/M task-space training and normalized export."""
from __future__ import annotations

import argparse
from pathlib import Path

import train_contact_taskspace_scratch as training


def verify_training(directory, protocol_path, *, allow_smoke=False, cpu_inference=False):
    root, protocol_path = Path(directory).resolve(), Path(protocol_path).resolve()
    names = ["manifest.json", "cohort_completion_audit.json", "cohort_startup_audit.json",
        "policy_contract.json", "final_policy_contract.json", "runner_config.json", "run_status.json",
        "policy.pt", "model_final.pt", "pre_export_runtime_audit.json", "export_precision_audit.json", "environment_final.json"]
    paths = {n: root/n for n in names}
    assert all(p.is_file() and not p.is_symlink() for p in paths.values())
    hashes = {n: training.sha(p) for n, p in paths.items()}
    read = training.read; m = read(paths["manifest.json"]); a = read(paths["cohort_completion_audit.json"])
    c = read(paths["policy_contract.json"]); report = read(paths["run_status.json"]); p = read(protocol_path)
    assert p["cohort_id"] == a["cohort_id"] == c["cohort_id"] == training.COHORT_ID
    assert p["from_scratch"] is True and not p["historical_checkpoint_used"]
    v = a["variant"]; assert v in ["K", "M"] and v == c["cohort_variant"] == m["reward_variant"]
    smoke = a["smoke_only"]; assert type(smoke) is bool and (allow_smoke or not smoke)
    mode = "smoke" if smoke else "formal"; d = p["smoke"] if smoke else p["algorithm"]
    assert a["run_kind"] == c["run_kind"] == mode and a["status"] == "completed_verified"
    assert report["status"] == "training_completed"
    if "returncode" in report:
        assert report["returncode"] == 0
    assert a["from_scratch"] is True and c["from_scratch"] is True and a["initial_checkpoint"] is None
    assert not a["historical_checkpoint_used"] and not c["historical_checkpoint_used"]
    assert a["trained_policy_claimed"] == c["trained_policy_claimed"] == (not smoke)
    assert a["initial_policy_state_sha256"] == report["initial_policy_state_sha256"] == m["initial_policy_state_sha256"] == training.RANDOM_STATE_SHA
    assert m["seed"] == 42 and m["iterations"] == d["iterations"] and m["num_envs"] == d["num_envs"]
    assert a["training_environment_steps"] == report["training_environment_steps"] == c["training_environment_transitions"] == d["transitions"]
    assert a["protocol_sha256"] == c["training_cohort_protocol_sha256"] == training.sha(protocol_path)
    assert c["checkpoint_sha256"] == hashes["model_final.pt"] and c["exported_policy_sha256"] == hashes["policy.pt"]
    for key, name in [("manifest", "manifest.json"), ("policy_contract", "policy_contract.json"),
                     ("final_policy_contract", "final_policy_contract.json"), ("checkpoint", "model_final.pt"),
                     ("exported_policy", "policy.pt"), ("startup_audit", "cohort_startup_audit.json"),
                     ("pre_export_runtime_audit", "pre_export_runtime_audit.json"), ("export_precision_audit", "export_precision_audit.json")]:
        assert a[key+"_sha256"] == hashes[name]
    assert paths["policy_contract.json"].read_bytes() == paths["final_policy_contract.json"].read_bytes()
    assert m["contract"] == c and a["cohort"] == m["cohort"]
    cohort = m["cohort"]
    assert cohort["from_scratch"] and not cohort["historical_checkpoint_used"] and cohort["initial_checkpoint"] is None
    assert cohort["normalizer_initial_count"] == 0 and cohort["optimizer_initial_state_empty"]
    assert cohort["normalizer_final_count"] == d["transitions"]
    assert cohort["optimizer"] == training.OPTIMIZER
    assert m["algorithm_source_sha256"] == dict(runner=training.RUNNER_SHA, ppo=training.PPO_SHA)
    cfg = read(paths["runner_config.json"])
    for key, value in training.OPTIMIZER.items(): assert cfg["algorithm"][key] == value
    for key, value in dict(num_learning_epochs=5, num_mini_batches=4, entropy_coef=.005, gamma=.99, lam=.95).items():
        assert cfg["algorithm"][key] == value
    assert cfg["num_steps_per_env"] == 24
    assert cfg["policy"]["actor_hidden_dims"] == cfg["policy"]["critic_hidden_dims"] == [256, 128, 128]
    for key, value in dict(physics_dt=.0025, decimation=8, policy_fps=50., actor_dim=51, critic_dim=120,
        action_dim=8, leg_kp=500., leg_kd=10., leg_position_scale=1., wheel_torque_scale_nm=12.,
        actuator_model="dc_motor", raw_action_clip=[-1., 1.], leg_torque_limit_nm=80., wheel_torque_limit_nm=12.,
        leg_motor_velocity_limit_rad_s=15., wheel_motor_velocity_limit_rad_s=100.,
        leg_command_execution="original_position_target_hold").items(): assert c[key] == value, key
    task = c["task_profile"]
    assert task == c["trained_task_profile"] == m["task_profile"]
    assert c["training_domain_randomization"] == p["training_randomization"]
    assert task["reward_weights"]["sideslip"] == 2. and task["action_rate_weight"] == -.5
    assert task["height_regularization"]["mode"] == "vertical_acceleration_only"
    assert c["ground_contact_reward_v2"] == m["contact_settings"] == task["ground_contact_reward_v2"] == p["variant_contact_settings"][v]
    assert c["reward_definition_version"] == "ground_contact_v2"+v.lower()+"_any_wheel"
    assert task["task_space_tracking"] == training.targets.settings(p["task_space_tracking"])
    for name, count in [("live_startup", 8), ("live_final", 8+d["iterations"]*24)]:
        live = cohort[name]; contact = live["contact"]; runtime = live["runtime"]
        assert contact["steps_started"] == contact["steps_completed"] == contact["unique_reward_windows"] == count
        assert contact["physics_samples"] == 8*count and contact["sampled_environment_substeps"] == 8*count*d["num_envs"]
        assert contact["settings"] == c["ground_contact_reward_v2"]
        assert contact["device_finite_normal_force_health"] and contact["device_finite_reward_health"] and not contact["cpu_fixture_only"]
        assert runtime["status"] == "verified" and runtime["fields"]["reward_term_count"] == 15
        assert not runtime["fields"]["absolute_height_termination"] and not runtime["fields"]["absolute_xy_termination"]
        targets = live["task_space_tracking"]
        assert targets["status"] == "live_target_selection_verified"
        assert targets["configured"]["settings"] == p["task_space_tracking"]
        assert targets["active_term_weights"]["drift_grounded"] == .4
    assert read(paths["pre_export_runtime_audit.json"]) == cohort["live_final"]
    precision = read(paths["export_precision_audit.json"]); assert precision == cohort["export_precision"]
    assert precision["CPU_source_module_export_bit_identical"] and precision["CPU_max_abs_error"] == 0.
    assert precision["exported_weights_and_normalizer_bit_identical"] and precision["original_precision_restored"]
    assert precision["GPU_FP32_max_abs_error"] <= precision["GPU_FP32_absolute_tolerance"] == 1e-4
    assert precision["GPU_reference_TF32"] is False
    assert all(training.sha(path) == digest for path, digest in a["source_sha256"].items())
    result = dict(directory=root, protocol_path=protocol_path, protocol=p, manifest=m, contract=c,
        completion=a, training_backend="physx", smoke_only=smoke, variant=v, checkpoint=paths["model_final.pt"],
        actor=paths["policy.pt"], paths=paths, input_sha256=hashes)
    if cpu_inference:
        import torch
        state = torch.load(paths["model_final.pt"], map_location="cpu", weights_only=True)
        assert state["iter"] == d["iterations"]-1
        state = state["model_state_dict"]
        assert all(bool(torch.isfinite(t).all()) for t in state.values())
        assert int(state["actor_obs_normalizer.count"]) == int(state["critic_obs_normalizer.count"]) == d["transitions"]
        actor = torch.jit.load(str(paths["policy.pt"]), map_location="cpu")
        exported = actor.state_dict()
        expected = {**{"0."+k.removeprefix("actor_obs_normalizer."): t for k, t in state.items() if k.startswith("actor_obs_normalizer.")},
                    **{"1."+k.removeprefix("actor."): t for k, t in state.items() if k.startswith("actor.")}}
        assert exported.keys() == expected.keys() and all(torch.equal(t, expected[k]) for k, t in exported.items())
        with torch.inference_mode(): output = actor(torch.zeros(1, 51))
        assert tuple(output.shape) == (1, 8) and torch.isfinite(output).all()
        result["CPU_actor_output_shape"] = [1, 8]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--training-dir", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--allow-smoke", action="store_true"); p.add_argument("--cpu-inference", action="store_true")
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    c = verify_training(args.training_dir, args.protocol, allow_smoke=args.allow_smoke, cpu_inference=args.cpu_inference)
    a = c["completion"]
    result = dict(kind="strict_scratch_taskspace_training_verification", status="completed_verified",
        cohort_id=training.COHORT_ID, variant=c["variant"], run_kind=a["run_kind"], transitions=a["training_environment_steps"],
        initial_policy_state_sha256=training.RANDOM_STATE_SHA, from_scratch=True, historical_checkpoint_used=False,
        actor_sha256=c["input_sha256"]["policy.pt"], checkpoint_sha256=c["input_sha256"]["model_final.pt"],
        completion_audit_sha256=c["input_sha256"]["cohort_completion_audit.json"],
        CPU_actor_output_shape=c.get("CPU_actor_output_shape"), task_space_tracking_verified=True,
        verifier_sha256=training.sha(__file__), task_success_claimed=False)
    with args.output.open("x") as f:
        import json
        json.dump(result, f, indent=2, allow_nan=False)
    print(result)


if __name__ == "__main__":
    main()

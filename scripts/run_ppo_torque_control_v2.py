"""Prepared, guarded smoke-first torque-control v2 dispatcher; no SSH/install.

The default is smoke ONLY. Formal training requires explicit adoption, always
after a fresh successful actual smoke within the same fresh run tree. No old
torque/contact protocol, low-mu test, retry, resume or foreign-process action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import run_ppo_smoothness_sequence as common
import train_ppo_torque_control_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
DEVICE, GPU_INDEX, KIT_ARGS = common.DEVICE, common.GPU_INDEX, common.KIT_ARGS
REFERENCE_SHA256 = "afc22f19b813cf10b3e3a90affb0f26f5a7743074f98b262394c69de786e9f6e"
MODEL_SHA256 = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
TRAINER_SHA256 = "2e82aba1058af213ebbbc8639c6d967db18992cb5c545a6b823438355e7ba4b0"
LEGACY_EVALUATOR_SHA256 = "76d78337d70519693bc7e8d819a6523a800e1e4889de5a0466c2d34356d2ae9d"


def require(condition, message):
    if not condition: raise ValueError(message)


def resolve_args(args):
    names = ("snapshot", "source_root", "validation_source_root", "beyondmimic_source", "motion_file", "reference_file",
        "asset_path", "mujoco_model", "task_profile", "protocol", "baseline_root", "baseline_training_dir", "output_dir")
    for name in names: setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    require(args.snapshot == ROOT.resolve(), "Execute from declared frozen torque-control v2 snapshot")
    require(not args.output_dir.exists(), "Preserve existing results: output must be fresh")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "GPU1 must remain physically unmasked")
    for directory in (args.snapshot, args.source_root, args.validation_source_root, args.beyondmimic_source, args.baseline_root, args.baseline_training_dir):
        require(directory.is_dir() and not args.output_dir.is_relative_to(directory), "Missing source or output overlaps frozen tree: "+str(directory))
    require(args.python.is_file() and os.access(args.python, os.X_OK), "Explicit Python executable unavailable")
    require((args.beyondmimic_source/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py").is_file(),
        "Explicit official BeyondMimic source missing")
    args.protocol_data = json.loads(args.protocol.read_text())
    expected = {"method": "add_leg_applied_torque_rate_to_minus0p5_original_grounded_recipe", "backend": "physx",
        "reference_id": "drift_drl_traj4", "seed": 42, "num_envs": 512, "iterations": 1000,
        "steps_per_environment_per_iteration": 24, "environment_transitions": 12288000,
        "smoke_num_envs": 2, "smoke_iterations": 1, "smoke_environment_transitions": 48, "setup_smoke_control_steps": 0,
        "from_scratch": True, "baseline_checkpoint_used_for_initialization": False, "baseline_action_rate_weight": -.5,
        "original_grounded_reward_preserved": True, "grounded_reward_weight": .4, "leg_applied_torque_rate_weight": -.05,
        "active_reward_terms_before": 15, "active_reward_terms_after": 16, "delta_torque_scale_nm": 20.,
        "first_actual_substep_after_reset_cost": 0., "denominator_after_reset": 4, "wheel_cost_included": False,
        "new_contact_sensors_added": False, "new_observation_terms": False, "leg_kp": 500., "leg_kd": 10.,
        "physics_dt_s": .005, "policy_dt_s": .02, "validation_mu_in_order": [.7],
        "baseline_checkpoint_sha256": trainer.BASELINE_CHECKPOINT_SHA256,
        "baseline_task_profile_sha256": trainer.BASELINE_PROFILE_SHA256, "trainer_wrapper_sha256": TRAINER_SHA256,
        "collector_sha256": trainer.COLLECTOR_SHA256, "legacy_physx_evaluator_sha256": LEGACY_EVALUATOR_SHA256,
        "default_smoke_only": True, "formal_requires_explicit_flag_and_verified_actual_smoke": True, "automatic_retry": False,
        "height_or_flight_failure_gate": False, "sideslip_zero_target": False,
        "final_status": "requires_visual_and_contact_review", "statistical_or_hardware_qualification_claimed": False}
    require(all(args.protocol_data.get(key) == value for key, value in expected.items()), "Unsupported v2 protocol/weight/scope")
    pins = {"scripts/train_ppo_torque_control_v2.py": TRAINER_SHA256,
        "scripts/ppo_leg_torque_smoothness.py": trainer.COLLECTOR_SHA256,
        "scripts/train_beyondmimic_style.py": trainer.BASELINE_TRAINER_SHA256,
        "scripts/height_smoothness.py": trainer.BASELINE_HEIGHT_SHA256,
        "scripts/train_ppo_smoothness.py": "a71f9c9fe84919a27ceadae499af9dc4b5ab14c18bfc04fcc42fab3829556093"}
    for name, expected_sha in pins.items():
        require((args.snapshot/name).is_file() and common.sha256(args.snapshot/name) == expected_sha, "Frozen implementation SHA differs: "+name)
    for name, expected_sha in trainer.BASELINE_SOURCES.items():
        require((args.source_root/name).is_file() and common.sha256(args.source_root/name) == expected_sha, "Historical training source differs: "+name)
    for name, expected_sha in trainer.BASELINE_METADATA.items():
        require((args.baseline_training_dir/name).is_file() and common.sha256(args.baseline_training_dir/name) == expected_sha, "Fixed -0.5 metadata differs: "+name)
    for path, expected_sha in ((args.motion_file, trainer.BASELINE_MOTION_SHA256), (args.asset_path, trainer.BASELINE_ASSET_SHA256),
        (args.task_profile, trainer.BASELINE_PROFILE_SHA256), (args.reference_file, REFERENCE_SHA256), (args.mujoco_model, MODEL_SHA256),
        (args.validation_source_root/"scripts/eval_tracking_friction_isaac.py", LEGACY_EVALUATOR_SHA256)):
        require(path.is_file() and common.sha256(path) == expected_sha, "Locked task/validation input differs: "+str(path))
    for name in ("run_car_validation.py", "run_ppo_smoothness_sequence.py", "analyze_ppo_smoothness.py", "analyze_ppo_contact_stability.py"):
        require((args.snapshot/"scripts"/name).is_file(), "Missing necessary helper: "+name)


def freeze_inputs(args):
    payload = {"source_snapshot": common.snapshot_hashes(args), "baseline_metadata": {
        str(args.baseline_training_dir/name): common.sha256(args.baseline_training_dir/name) for name in trainer.BASELINE_METADATA}}
    payload["combined_tree_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def verify_frozen(args, frozen):
    common.verify_snapshot(args, frozen["source_snapshot"])
    if any(common.sha256(name) != expected for name, expected in frozen["baseline_metadata"].items()):
        raise common.SourceChangedError("Frozen baseline metadata changed")


def training_command(args, output, *, smoke):
    command = [str(args.python), "-u", str(args.snapshot/"scripts/train_ppo_torque_control_v2.py"),
        "--baseline-training-dir", str(args.baseline_training_dir), "--leg-torque-rate-weight", "-0.05",
        "--source-root", str(args.source_root), "--motion-file", str(args.motion_file), "--asset-path", str(args.asset_path),
        "--task-profile", str(args.task_profile), "--output-dir", str(output), "--num-envs", "2" if smoke else "512",
        "--iterations", "1" if smoke else "1000", "--seed", "42", "--smoke-steps", "0", "--ground-friction", "1.0",
        "--save-interval", "100", "--device", DEVICE, "--headless", "--kit_args", KIT_ARGS]
    if smoke: command.append("--smoke-only")
    return command


def finalize_owned_training(args, output, child, *, smoke, finalizer):
    output = Path(output)
    require(child.get("phase") == ("smoke" if smoke else "training") and "returncode" in child and
        output.is_relative_to(args.output_dir) and str(output) in child["command"], "Only exact owned exited child can be finalized")
    finalized = finalizer(output, child_returncode=child["returncode"])
    if child["returncode"] != 0: raise common.ChildFailedError("Owned v2 training failed; no retry")
    manifest, status, runner, runtime = [json.loads((output/name).read_text()) for name in
        ("manifest.json", "run_status.json", "runner_config.json", trainer.RUNTIME_FILE)]
    steps, envs, iterations = (24, 2, 1) if smoke else (24000, 512, 1000)
    require(finalized.get("kind") == trainer.KIND and finalized.get("status") == status.get("status") == ("smoke_passed" if smoke else "training_completed") and
        finalized.get("smoke_only") is smoke and finalized.get("trained_policy_claimed") is (not smoke) and
        finalized.get("prepared_method_runtime_verified") is True and finalized.get("baseline_checkpoint_used_for_initialization") is False,
        "Parent finalizer did not verify exact fresh v2 scope")
    require(status.get("environment_transitions") == finalized.get("actual_environment_transitions") == manifest.get("environment_transitions_requested") == steps*envs and
        status.get("policy_steps_completed") == steps and manifest.get("num_envs") == envs and manifest.get("iterations") == iterations and
        manifest.get("seed") == 42 and manifest.get("setup_smoke_control_steps") == 0 and runner.get("num_steps_per_env") == 24,
        "V2 exact collection budget/seed/setup changed")
    require(manifest.get("training_reward_overrides") == {"action_rate_l2": -.5} and
        manifest.get("additional_training_reward_terms") == {trainer.TERM_NAME: finalized.get("additional_reward")} and
        manifest.get("original_grounded_reward_preserved") is True and "contact_reward_replacement" not in manifest,
        "V2 did not preserve original grounded and add only torque penalty")
    trainer.verify_runtime_audit(runtime, smoke=smoke, wrapper_sha256=TRAINER_SHA256)
    checkpoint = output/("model_smoke.pt" if smoke else "model_final.pt")
    for key, path in (("checkpoint_sha256", checkpoint), ("final_training_manifest_sha256", output/"manifest.json"),
        ("runtime_audit_sha256", output/trainer.RUNTIME_FILE), ("run_status_sha256", output/"run_status.json")):
        require(path.is_file() and path.stat().st_size > 0 and common.sha256(path) == finalized.get(key), "Finalized v2 artifact differs: "+key)
    return {"status": finalized["status"], "environment_transitions": steps*envs, "policy_steps": steps,
        "native_torque_sampling_verified": True, "original_grounded_preserved": True, "checkpoint_sha256": finalized["checkpoint_sha256"],
        "ablation_manifest_sha256": common.sha256(output/"ablation_manifest.json"), "physical_sampling_audit_sha256": finalized["runtime_audit_sha256"]}


def contact_analysis(folder):
    import numpy as np
    from analyze_ppo_contact_stability import analyze_signals, _field
    with np.load(Path(folder)/"physics_trajectory.npz", allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    require(raw["time_s"].shape == (1397,) and np.array_equal(raw["time_s"], np.arange(1397)*.005) and
        raw["valid_mask"].shape == (1397, 1) and raw["valid_mask"].all(), "Incomplete original native200Hz recording")
    return analyze_signals(raw["time_s"], _field(raw, "wheel_ground_contact_count"), _field(raw, "wheel_ground_contact_force_w_n"),
        _field(raw, "root_linear_velocity_w_m_s"), _field(raw, "root_angular_velocity_w_rad_s"),
        nonwheel_counts=_field(raw, "nonwheel_ground_contact_count") if "nonwheel_ground_contact_count" in raw else None)


def run_trial(args, *, gpu_probe=common.query_gpu_one, child_runner=common.run_child, validation_runner=common.run_validation,
              training_finalizer=trainer.finalize_manifest, trajectory_analyzer=None, contact_analyzer=contact_analysis, quiesce=time.sleep):
    resolve_args(args)
    if trajectory_analyzer is None:
        from analyze_ppo_smoothness import analyze_trajectory
        trajectory_analyzer = analyze_trajectory
    frozen = freeze_inputs(args); args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir/"torque_control_v2_status.json"
    status = {"schema_version": 1, "kind": trainer.KIND, "status": "initializing", "protocol": args.protocol_data,
        "protocol_sha256": common.sha256(args.protocol), "source_tree_sha256": frozen["combined_tree_sha256"],
        "pid": os.getpid(), "host": platform.node(), "started_unix_time": time.time(), "device": DEVICE, "physical_gpu_index": GPU_INDEX,
        "cuda_visible_devices": None, "formal_after_verified_smoke_requested": args.run_formal_after_verified_smoke,
        "automatic_retry": False, "exclusive_gpu_reservation_claimed": False, "new_policy_quality_claimed": False,
        "continuous_contact_certified": False, "requires_visual_contact_review": True, "children": []}
    common.write_json(args.output_dir/"frozen_source_and_inputs.json", frozen); common.write_json(status_path, status)
    environment = os.environ.copy(); environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1", TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    def launch_checked(command, cwd, phase):
        verify_frozen(args, frozen); evidence = gpu_probe()
        child = {"phase": phase, "command": list(command), "cwd": str(cwd), "gpu_preflight": evidence, "status": "pending"}
        status["children"].append(child)
        if not evidence.get("idle") or evidence.get("physical_index") != GPU_INDEX:
            child["status"] = "not_started_gpu_busy"; common.write_json(status_path, status)
            raise common.GpuBusyError("GPU1 not freshly fully idle; no child started")
        child.update(status="running", started_unix_time=time.time()); status["status"] = "running_"+phase
        log = args.output_dir/f"child_{len(status['children']):03d}_{phase}.log"; child["log_path"] = str(log)
        common.write_json(status_path, status)
        def spawned(pid): child["pid"] = int(pid); common.write_json(status_path, status)
        code = child_runner(command, log, cwd, environment, spawned)
        child.update(returncode=code, status="completed" if code == 0 else "failed", finished_unix_time=time.time())
        common.write_json(status_path, status); quiesce(2.); verify_frozen(args, frozen)
        if code != 0 and phase not in ("smoke", "training"): raise common.ChildFailedError("Owned validation failed; no retry")
        return child
    try:
        baseline = args.baseline_root/"mu0p700/isaac"; motion_sha = common.sha256(args.motion_file)
        baseline_report = common.verify_evaluation(baseline, .7, trainer.BASELINE_CHECKPOINT_SHA256, motion_sha)
        require(baseline_report.get("evaluator_source_sha256") == LEGACY_EVALUATOR_SHA256, "Baseline evaluator differs")
        common.write_json(args.output_dir/"baseline_nominal.json", {"trajectory": trajectory_analyzer(baseline/"physics_trajectory.npz",
            reference=args.motion_file, report=baseline/"report.json", high_frequency_hz=10.), "contact": contact_analyzer(baseline)})
        for smoke in (True, False):
            output = args.output_dir/("smoke" if smoke else "training")
            child = launch_checked(training_command(args, output, smoke=smoke), args.snapshot, "smoke" if smoke else "training")
            proof = finalize_owned_training(args, output, child, smoke=smoke, finalizer=training_finalizer)
            status["smoke" if smoke else "training"] = proof; child["parent_finalization"] = proof
            common.write_json(status_path, status)
            if smoke and not args.run_formal_after_verified_smoke:
                status.update(status="smoke_passed_requires_explicit_formal_adoption", finished_unix_time=time.time())
                common.write_json(status_path, status); return 0
        validation = args.output_dir/"validation"
        validation_runner(args, args.output_dir/"training", validation, launch_checked)
        isaac = validation/"mu0p700/isaac"
        report = common.verify_evaluation(isaac, .7, status["training"]["checkpoint_sha256"], motion_sha, require_gpu_one=True)
        require(report.get("evaluator_source_sha256") == LEGACY_EVALUATOR_SHA256 and
            report["contact_material_audit"]["leg_pd_gain_readback"] == {"kp_range": [500., 500.], "kd_range": [10., 10.]},
            "Nominal evaluation source/PD differs")
        measured = trajectory_analyzer(isaac/"physics_trajectory.npz", reference=args.motion_file, report=isaac/"report.json", high_frequency_hz=10.)
        common.write_json(args.output_dir/"candidate_nominal.json", {"trajectory": measured, "contact": contact_analyzer(isaac)})
        mujoco = validation/"mu0p700/mujoco"; mj = json.loads((mujoco/"report.json").read_text())
        require(mj.get("status") == "completed" and mj.get("completed_full_reference") is True and mj.get("hidden_resets") == 0 and
            mj.get("root_state_writes") == 1 and mj.get("motion_sha256") == motion_sha and mj.get("model_sha256") == common.sha256(args.mujoco_model) and
            mj.get("policy_sha256") == common.sha256(isaac/"actor_normalized.pt") and
            (mujoco/"rollout.npz").is_file() and (mujoco/"rollout.npz").stat().st_size > 0, "Nominal MuJoCo/actor/model evidence differs")
        status["validation"] = {"mu": .7, "isaac_report_sha256": common.sha256(isaac/"report.json"),
            "isaac_raw_sha256": report["physics_trajectory_sha256"], "mujoco_report_sha256": common.sha256(mujoco/"report.json"),
            "mujoco_rollout_sha256": common.sha256(mujoco/"rollout.npz")}
        status.update(status="requires_visual_and_contact_review", finished_unix_time=time.time())
        common.write_json(status_path, status); return 0
    except BaseException as error:
        status.update(status="stopped_gpu_busy" if isinstance(error, common.GpuBusyError) else "failed",
            error=f"{type(error).__name__}: {error}", finished_unix_time=time.time())
        if status["children"] and status["children"][-1]["status"] == "running": status["children"][-1]["status"] = "failed_or_interrupted"
        common.write_json(status_path, status); return 3 if isinstance(error, common.GpuBusyError) else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "source-root", "validation-source-root", "beyondmimic-source", "python", "motion-file", "reference-file",
        "asset-path", "mujoco-model", "task-profile", "baseline-root", "baseline-training-dir", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"config/ppo_torque_control_v2.json")
    parser.add_argument("--run-formal-after-verified-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    try: return run_trial(parse_args(argv))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Torque-control v2 not started: {type(error).__name__}: {error}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())

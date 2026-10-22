"""Guarded smoke-first dispatcher for the new substep ground-support PPO.

Default invocation runs ONLY the real Isaac smoke. Formal adoption requires an
explicit flag and another fresh run tree; every owned child is preceded by a
fresh fully-idle unmasked physical GPU1 check. No SSH, retry, resume or foreign
process control is performed. Final evaluation requires visual/contact review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import run_ppo_smoothness_sequence as common
from train_ppo_ground_support import BASELINE_METADATA, BASELINE_CHECKPOINT_SHA256, KIND, RUNTIME_FILE
from train_ppo_smoothness import BASELINE_SOURCES, BASELINE_TRAINER_SHA256, BASELINE_HEIGHT_SHA256, BASELINE_MOTION_SHA256, BASELINE_ASSET_SHA256, BASELINE_PROFILE_SHA256

ROOT = Path(__file__).resolve().parents[1]
DEVICE, GPU_INDEX, KIT_ARGS = common.DEVICE, common.GPU_INDEX, common.KIT_ARGS
TERM = "drift_grounded"
REFERENCE_SHA256 = "afc22f19b813cf10b3e3a90affb0f26f5a7743074f98b262394c69de786e9f6e"
MODEL_SHA256 = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
PINS = {
    "scripts/train_ppo_ground_support.py": "a834e864689c77d622808f2a2190879af3fb757016645b219c48a316fb4c32cb",
    "scripts/ppo_substep_ground_support.py": "4b8c7e1e34487cee8576e660eccb20dafd2cf599aecb7c64a59707c0a789e62f",
    "scripts/ppo_leg_torque_smoothness.py": "2649dc39f08f8b540430617f7a2c8182527d895b62d5c515672e66c7c3e03aa7",
    "scripts/train_ppo_smoothness.py": "a71f9c9fe84919a27ceadae499af9dc4b5ab14c18bfc04fcc42fab3829556093",
    "scripts/train_beyondmimic_style.py": BASELINE_TRAINER_SHA256,
    "scripts/height_smoothness.py": BASELINE_HEIGHT_SHA256,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_args(args):
    for name in ("snapshot", "source_root", "validation_source_root", "beyondmimic_source", "motion_file", "reference_file",
                 "asset_path", "mujoco_model", "task_profile", "protocol", "baseline_root", "baseline_training_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    require(args.snapshot == ROOT.resolve(), "Run from the declared frozen ground-support snapshot")
    require(not args.output_dir.exists(), "Preserve existing runs: choose a fresh output directory")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Unmasked GPU1 requires CUDA_VISIBLE_DEVICES unset")
    for directory in (args.snapshot, args.source_root, args.validation_source_root, args.beyondmimic_source, args.baseline_training_dir, args.baseline_root):
        require(directory.is_dir(), "Missing explicit source/baseline directory: "+str(directory))
        require(not args.output_dir.is_relative_to(directory), "Outputs must be outside frozen source/baseline trees")
    require(args.python.is_file() and os.access(args.python, os.X_OK), "Explicit Python executable unavailable")
    bm_config = args.beyondmimic_source/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    require(bm_config.is_file(), "Explicit official BeyondMimic source/config is missing")
    args.protocol_data = json.loads(args.protocol.read_text())
    expected = {"method": "replace_existing_grounded_reward_with_all_substep_ground_support", "backend": "physx",
        "reference_id": "drift_drl_traj4", "seed": 42, "num_envs": 512, "iterations": 1000,
        "steps_per_environment_per_iteration": 24, "environment_transitions": 12288000,
        "smoke_num_envs": 2, "smoke_iterations": 1, "smoke_environment_transitions": 48,
        "setup_smoke_control_steps": 0, "from_scratch": True, "baseline_action_rate_weight": -.5,
        "grounded_reward_weight": .4, "ground_support_scale_fraction": .2, "leg_kp": 500., "leg_kd": 10.,
        "physics_dt_s": .005, "policy_dt_s": .02, "validation_mu_in_order": [.7],
        "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
        "baseline_task_profile_sha256": BASELINE_PROFILE_SHA256,
        "trainer_wrapper_sha256": PINS["scripts/train_ppo_ground_support.py"],
        "collector_sha256": PINS["scripts/ppo_substep_ground_support.py"],
        "clock_helper_sha256": PINS["scripts/ppo_leg_torque_smoothness.py"],
        "legacy_physx_evaluator_sha256": "76d78337d70519693bc7e8d819a6523a800e1e4889de5a0466c2d34356d2ae9d",
        "default_smoke_only": True, "formal_requires_explicit_flag_and_verified_actual_smoke": True,
        "automatic_retry": False, "height_or_flight_failure_gate": False, "sideslip_zero_target": False,
        "final_status": "requires_visual_and_contact_review", "statistical_or_hardware_qualification_claimed": False}
    for key, value in expected.items():
        require(args.protocol_data.get(key) == value, "Unsupported ground-support protocol: "+key)
    for name, digest in PINS.items():
        require((args.snapshot/name).is_file() and common.sha256(args.snapshot/name) == digest, "Frozen implementation SHA mismatch: "+name)
    for name, digest in BASELINE_SOURCES.items():
        require((args.source_root/name).is_file() and common.sha256(args.source_root/name) == digest, "Historical training dependency SHA mismatch: "+name)
    for name, digest in BASELINE_METADATA.items():
        require((args.baseline_training_dir/name).is_file() and common.sha256(args.baseline_training_dir/name) == digest, "Fixed -0.5 baseline metadata SHA mismatch: "+name)
    for name in ("run_car_validation.py", "run_ppo_smoothness_sequence.py", "analyze_ppo_smoothness.py", "analyze_ppo_contact_stability.py", "summarize_ppo_smoothness.py"):
        require((args.snapshot/"scripts"/name).is_file(), "Missing frozen helper: "+name)
    require(common.sha256(args.task_profile) == args.protocol_data["baseline_task_profile_sha256"], "Historical task-profile SHA differs")
    require(common.sha256(args.motion_file) == BASELINE_MOTION_SHA256 and common.sha256(args.asset_path) == BASELINE_ASSET_SHA256, "Historical motion/asset differs")
    require(common.sha256(args.reference_file) == REFERENCE_SHA256, "Locked robot-reference SHA differs")
    require(common.sha256(args.mujoco_model) == MODEL_SHA256, "Locked MuJoCo model SHA differs")
    require(common.sha256(args.validation_source_root/"scripts/eval_tracking_friction_isaac.py") == args.protocol_data["legacy_physx_evaluator_sha256"], "Legacy PhysX evaluator SHA differs")
    for name in ("physics_trajectory.npz", "report.json"):
        require((args.baseline_root/"mu0p700/isaac"/name).is_file(), "Missing nominal -0.5 baseline recording: "+name)


def freeze_inputs(args):
    baseline = {str(args.baseline_training_dir/name): common.sha256(args.baseline_training_dir/name) for name in BASELINE_METADATA}
    payload = {"source_snapshot": common.snapshot_hashes(args), "baseline_training_metadata": baseline}
    payload["combined_tree_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def verify_frozen(args, frozen):
    common.verify_snapshot(args, frozen["source_snapshot"])
    for name, digest in frozen["baseline_training_metadata"].items():
        if common.sha256(name) != digest:
            raise common.SourceChangedError("Frozen baseline metadata changed: "+name)


def training_command(args, output, *, smoke):
    command = [str(args.python), "-u", str(args.snapshot/"scripts/train_ppo_ground_support.py"),
        "--baseline-training-dir", str(args.baseline_training_dir), "--source-root", str(args.source_root),
        "--motion-file", str(args.motion_file), "--asset-path", str(args.asset_path), "--task-profile", str(args.task_profile),
        "--output-dir", str(output), "--num-envs", "2" if smoke else "512", "--iterations", "1" if smoke else "1000",
        "--seed", "42", "--smoke-steps", "0", "--ground-friction", "1.0", "--save-interval", "100",
        "--device", DEVICE, "--headless", "--kit_args", KIT_ARGS]
    if smoke:
        command.append("--smoke-only")
    return command


def verify_physical_audit(runtime, *, smoke):
    steps, envs = (24, 2) if smoke else (24000, 512)
    require(runtime.get("steps_started") == runtime.get("steps_completed") == runtime.get("unique_cost_windows") == steps
        and runtime.get("physics_samples_total") == 4*steps and runtime.get("physics_environment_substeps") == 4*steps*envs
        and runtime.get("steps_failed") == 0 and runtime.get("samples_last_window") == 4 and runtime.get("num_envs") == envs,
        "Incomplete native four-substep support audit")
    require(runtime.get("closed") is True and runtime.get("device_health_checked") is True and runtime.get("device_health_finite") is True
        and runtime.get("cpu_fixture_only") is False and runtime.get("device") == DEVICE, "Actual GPU physical audit is missing or unhealthy")
    require(runtime.get("module_sha256") == PINS["scripts/ppo_substep_ground_support.py"]
        and runtime.get("wrapper_sha256") == PINS["scripts/train_ppo_ground_support.py"]
        and runtime.get("clock_helper_sha256") == PINS["scripts/ppo_leg_torque_smoothness.py"], "Runtime source hashes differ")
    require((runtime.get("physics_dt_s"), runtime.get("policy_dt_s"), runtime.get("decimation")) == (.005, .02, 4)
        and runtime.get("scale_fraction") == .2 and runtime.get("raw_reward_bounds") == [0., 1.]
        and runtime.get("external_reward_weight") == .4, "Physical sampling/reward normalization differs")
    require(runtime.get("sensor_names") == ["wheel_L_ground", "wheel_R_ground"] and runtime.get("wheel_body_names") == ["wheel_L_Link", "wheel_R_Link"]
        and runtime.get("force_matrix_shape_each_sensor") == [envs, 1, 1, 3]
        and runtime.get("ground_filter", {}).get("path") == "/World/ground/terrain/GroundPlane/CollisionPlane"
        and runtime.get("ground_filter", {}).get("collision_api_verified") is True, "Ground-filtered physical sensor contract differs")
    require(runtime.get("gain_readback") == {"kp_range": [500., 500.], "kd_range": [10., 10.]} and runtime.get("mass_gravity_reads") == 1,
        "Actual gain or poststartup live-mass audit differs")
    for key in ("mass_kg_range", "mg_N_range"):
        value = runtime.get(key, [])
        require(len(value) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in value)
            and value[0] <= value[1], "Invalid live mass/mg cache: "+key)
    gravity = runtime.get("gravity_m_s2", [])
    require(len(gravity) == 3 and gravity[0] == gravity[1] == 0 and math.isfinite(gravity[2]) and gravity[2] < 0, "Invalid live gravity")
    require(all(math.isclose(mg, mass*abs(gravity[2]), rel_tol=1e-5) for mass, mg in zip(runtime["mass_kg_range"], runtime["mg_N_range"])), "Live mg is inconsistent with mass/gravity")
    positive = runtime.get("positive_normal_force_samples_left_right", [])
    require(len(positive) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and 0 < x <= 4*steps*envs for x in positive)
        and runtime.get("negative_normal_force_samples") == 0, "Both real upward wheel-force channels must be measured")
    for key in ("observed_ground_fn_z_min_N", "observed_ground_fn_z_max_N"):
        value = runtime.get(key, [])
        require(len(value) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in value), "Missing finite measured forces: "+key)
    require(min(runtime["observed_ground_fn_z_min_N"]) >= -1e-4 and min(runtime["observed_ground_fn_z_max_N"]) > 0,
        "Measured force direction is not verified upward on both wheels")
    require(isinstance(runtime.get("observed_normal_force_xy_abs_peak_N"), (int, float)) and math.isfinite(runtime["observed_normal_force_xy_abs_peak_N"]), "Missing finite force-vector audit")


def finalize_owned_training(args, output, child, *, smoke, finalizer):
    output = Path(output)
    require(child.get("phase") == ("smoke" if smoke else "training") and "returncode" in child
        and output.is_relative_to(args.output_dir) and str(output) in child["command"], "Only the exact exited owned training child may be finalized")
    finalized = finalizer(output, child_returncode=child["returncode"])
    if child["returncode"] != 0:
        raise common.ChildFailedError("Owned training child failed; no retry or formal adoption")
    status, manifest, runtime, runner = [json.loads((output/name).read_text()) for name in ("run_status.json", "manifest.json", RUNTIME_FILE, "runner_config.json")]
    steps, envs, iterations = (24, 2, 1) if smoke else (24000, 512, 1000)
    scope_status = "smoke_passed" if smoke else "training_completed"
    require(finalized.get("kind") == KIND and finalized.get("status") == status.get("status") == scope_status
        and finalized.get("trained_policy_claimed") is (not smoke) and finalized.get("smoke_only") is smoke
        and finalized.get("prepared_method_runtime_verified") is True
        and finalized.get("initialization") == "from_scratch_no_checkpoint_resume", "Finalizer did not verify exact smoke/formal scope")
    require(status.get("environment_transitions") == finalized.get("actual_environment_transitions")
        == finalized.get("expected_environment_transitions") == manifest.get("environment_transitions_requested") == steps*envs
        and status.get("policy_steps_completed") == finalized.get("expected_policy_steps") == steps
        and manifest.get("num_envs") == finalized.get("num_envs") == envs
        and manifest.get("iterations") == finalized.get("iterations") == iterations
        and manifest.get("seed") == finalized.get("seed") == 42 and manifest.get("setup_smoke_control_steps") == 0
        and runner.get("num_steps_per_env") == 24, "Incomplete formal/smoke budget, seed or setup")
    reward = finalized.get("reward_replacement", {})
    require(finalized.get("baseline_action_rate_weight") == -.5 and reward.get("term") == TERM
        and reward.get("weight") == .4 and reward.get("scale_fraction") == .2 and reward.get("additional_reward_term") is False
        and manifest.get("training_reward_overrides") == {"action_rate_l2": -.5, TERM: reward}
        and manifest.get("contact_reward_replacement") == reward, "Ground-support replacement/action-rate recipe differs")
    verify_physical_audit(runtime, smoke=smoke)
    checkpoint = output/("model_smoke.pt" if smoke else "model_final.pt")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0 and finalized.get("checkpoint_sha256") == common.sha256(checkpoint)
        and finalized.get("final_training_manifest_sha256") == common.sha256(output/"manifest.json")
        and finalized.get("runtime_audit_sha256") == common.sha256(output/RUNTIME_FILE)
        and finalized.get("run_status_sha256") == common.sha256(output/"run_status.json"), "Finalized artifacts are not byte-bound")
    return {"status": scope_status, "environment_transitions": steps*envs, "policy_steps": steps,
        "physical_sampling_verified": True, "actual_both_upward_force_channels_verified": True,
        "checkpoint_sha256": finalized["checkpoint_sha256"], "ablation_manifest_sha256": common.sha256(output/"ablation_manifest.json"),
        "physical_sampling_audit_sha256": finalized["runtime_audit_sha256"]}


def contact_analysis(folder):
    import numpy as np
    from analyze_ppo_contact_stability import analyze_signals, _field
    from summarize_ppo_smoothness import posture_diagnostics
    with np.load(Path(folder)/"physics_trajectory.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    require(raw["time_s"].shape == (1397,) and np.array_equal(raw["time_s"], np.arange(1397)*.005)
        and raw["valid_mask"].shape == (1397, 1) and raw["valid_mask"].all(), "Incomplete native contact recording")
    return {"windows": analyze_signals(raw["time_s"], _field(raw, "wheel_ground_contact_count"), _field(raw, "wheel_ground_contact_force_w_n"),
        _field(raw, "root_linear_velocity_w_m_s"), _field(raw, "root_angular_velocity_w_rad_s"),
        nonwheel_counts=_field(raw, "nonwheel_ground_contact_count") if "nonwheel_ground_contact_count" in raw else None), "posture": posture_diagnostics(raw)}


def run_trial(args, *, gpu_probe=common.query_gpu_one, child_runner=common.run_child,
              validation_runner=common.run_validation, trajectory_analyzer=None, contact_analyzer=None,
              training_finalizer=None, quiesce=time.sleep):
    resolve_args(args)
    if trajectory_analyzer is None:
        from analyze_ppo_smoothness import analyze_trajectory
        trajectory_analyzer = analyze_trajectory
    contact_analyzer = contact_analyzer or contact_analysis
    if training_finalizer is None:
        from train_ppo_ground_support import finalize_manifest
        training_finalizer = finalize_manifest
    frozen = freeze_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir/"ground_support_status.json"
    status = {"schema_version": 1, "status": "initializing", "protocol": args.protocol_data,
        "protocol_sha256": common.sha256(args.protocol), "source_tree_sha256": frozen["combined_tree_sha256"],
        "pid": os.getpid(), "host": platform.node(), "started_unix_time": time.time(),
        "device": DEVICE, "physical_gpu_index": GPU_INDEX, "cuda_visible_devices": None,
        "beyondmimic_source": str(args.beyondmimic_source), "formal_after_verified_smoke_requested": args.run_formal_after_verified_smoke,
        "automatic_retry": False, "exclusive_gpu_reservation_claimed": False,
        "new_policy_quality_claimed": False, "continuous_contact_certified": False, "requires_visual_contact_review": True,
        "children": []}
    common.write_json(args.output_dir/"frozen_source_and_inputs.json", frozen); common.write_json(status_path, status)
    environment = os.environ.copy(); environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1", TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))

    def launch_checked(command, cwd, phase):
        verify_frozen(args, frozen)
        evidence = gpu_probe()
        child = {"phase": phase, "command": list(command), "cwd": str(cwd), "gpu_preflight": evidence,
            "status": "pending", "source_tree_sha256": frozen["combined_tree_sha256"]}
        status["children"].append(child)
        if not evidence.get("idle") or evidence.get("physical_index") != GPU_INDEX:
            child["status"] = "not_started_gpu_busy"; common.write_json(status_path, status)
            raise common.GpuBusyError("GPU1 not freshly fully idle; no child started")
        child.update(status="running", started_unix_time=time.time())
        status["status"] = "running_"+phase
        log = args.output_dir/f"child_{len(status['children']):03d}_{phase}.log"
        child["log_path"] = str(log); common.write_json(status_path, status)
        def spawned(pid):
            child["pid"] = int(pid); common.write_json(status_path, status)
        code = child_runner(command, log, cwd, environment, spawned)
        child.update(returncode=code, finished_unix_time=time.time(), status="completed" if code == 0 else "failed")
        common.write_json(status_path, status); quiesce(2.)
        verify_frozen(args, frozen)
        if code != 0 and phase not in ("smoke", "training"):
            raise common.ChildFailedError("Validation child failed; no retry")
        return child

    try:
        baseline_folder = args.baseline_root/"mu0p700/isaac"
        motion_hash = common.sha256(args.motion_file)
        baseline_report = common.verify_evaluation(baseline_folder, .7, BASELINE_CHECKPOINT_SHA256, motion_hash)
        require(baseline_report.get("evaluator_source_sha256") == args.protocol_data["legacy_physx_evaluator_sha256"], "Baseline evaluator cohort differs")
        baseline = trajectory_analyzer(baseline_folder/"physics_trajectory.npz", reference=args.motion_file, report=baseline_folder/"report.json", high_frequency_hz=10.)
        common.write_json(args.output_dir/"baseline_nominal.json", {"trajectory": baseline, "contact": contact_analyzer(baseline_folder)})
        for smoke in (True, False):
            folder = args.output_dir/("smoke" if smoke else "training")
            child = launch_checked(training_command(args, folder, smoke=smoke), args.snapshot, "smoke" if smoke else "training")
            proof = finalize_owned_training(args, folder, child, smoke=smoke, finalizer=training_finalizer)
            status["smoke" if smoke else "training"] = proof; child["parent_finalization"] = proof
            common.write_json(status_path, status)
            if smoke and not args.run_formal_after_verified_smoke:
                status.update(status="smoke_passed_requires_explicit_formal_adoption", finished_unix_time=time.time())
                common.write_json(status_path, status); return 0
        validation = args.output_dir/"validation"
        validation_runner(args, args.output_dir/"training", validation, launch_checked)
        isaac = validation/"mu0p700/isaac"
        report = common.verify_evaluation(isaac, .7, status["training"]["checkpoint_sha256"], motion_hash, require_gpu_one=True)
        require(report.get("evaluator_source_sha256") == args.protocol_data["legacy_physx_evaluator_sha256"]
            and report["contact_material_audit"]["leg_pd_gain_readback"] == {"kp_range": [500., 500.], "kd_range": [10., 10.]}, "Validation evaluator/gain cohort differs")
        measured = trajectory_analyzer(isaac/"physics_trajectory.npz", reference=args.motion_file, report=isaac/"report.json", high_frequency_hz=10.)
        require(measured.get("checkpoint_sha256") == status["training"]["checkpoint_sha256"] and measured.get("reference_sha256") == motion_hash
            and measured.get("evaluation_mu") == .7, "Measured nominal candidate identity differs")
        common.write_json(args.output_dir/"candidate_nominal.json", {"trajectory": measured, "contact": contact_analyzer(isaac)})
        mujoco = validation/"mu0p700/mujoco"; mj_report = json.loads((mujoco/"report.json").read_text())
        require(mj_report.get("status") == "completed" and mj_report.get("completed_full_reference") is True
            and mj_report.get("motion_sha256") == motion_hash and mj_report.get("model_sha256") == common.sha256(args.mujoco_model)
            and mj_report.get("policy_sha256") == common.sha256(isaac/"actor_normalized.pt")
            and (mujoco/"rollout.npz").is_file() and (mujoco/"rollout.npz").stat().st_size > 0, "MuJoCo completed rollout/actor/model evidence differs")
        status["validation"] = {"mu": .7, "isaac_report_sha256": common.sha256(isaac/"report.json"),
            "isaac_raw_sha256": report["physics_trajectory_sha256"], "mujoco_report_sha256": common.sha256(mujoco/"report.json"),
            "mujoco_rollout_sha256": common.sha256(mujoco/"rollout.npz")}
        status.update(status="requires_visual_and_contact_review", finished_unix_time=time.time())
        common.write_json(status_path, status); return 0
    except BaseException as error:
        status.update(status="stopped_gpu_busy" if isinstance(error, common.GpuBusyError) else "failed",
            error=f"{type(error).__name__}: {error}", finished_unix_time=time.time())
        if status["children"] and status["children"][-1]["status"] == "running":
            status["children"][-1]["status"] = "failed_or_interrupted"
        common.write_json(status_path, status)
        return 3 if isinstance(error, common.GpuBusyError) else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "source-root", "validation-source-root", "beyondmimic-source", "python", "motion-file", "reference-file",
                 "asset-path", "mujoco-model", "task-profile", "baseline-root", "baseline-training-dir", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"config/ppo_ground_support_v1.json")
    parser.add_argument("--run-formal-after-verified-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return run_trial(parse_args(argv))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Ground-support job not started: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

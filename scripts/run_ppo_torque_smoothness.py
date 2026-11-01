"""Prepared single torque-rate PPO trial, gated on the completed first method.

No work starts on import. Invocation requires a terminal ``needs_next_method``
action-rate sequence, fresh fully-idle physical GPU 1 before every child, and a
new output tree. No retry, GPU substitution, installation or SSH is performed.
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


ROOT = Path(__file__).resolve().parents[1]
DEVICE, GPU_INDEX, KIT_ARGS = common.DEVICE, common.GPU_INDEX, common.KIT_ARGS
TERM = "leg_applied_torque_rate"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_previous_sequence(path):
    previous = json.loads(Path(path).read_text())
    require(previous.get("status") == "needs_next_method",
            "Previous method must be terminal needs_next_method; running or screen-passed sequences do not authorize this trial")
    protocol = previous.get("protocol", {})
    require(protocol.get("method") == "action_rate_l2_weight_only"
            and protocol.get("candidate_action_rate_weights_in_order") == [-.3, -.5, -1.],
            "Previous sequence is not the declared completed action-rate method")
    candidates = previous.get("candidates", [])
    require([item.get("action_rate_weight") for item in candidates] == [-.3, -.5, -1.],
            "Previous method does not contain all three declared trials")
    for candidate in candidates:
        comparisons = candidate.get("comparisons", {})
        require(candidate.get("status") == "failed_engineering_screen"
                and set(comparisons) == {"0.7", "0.15"}
                and all(isinstance(value.get("passes"), bool) for value in comparisons.values())
                and not all(value["passes"] for value in comparisons.values()),
                "Previous action-rate trial is incomplete or did not fail its declared screen")
    require(previous.get("finished_unix_time") is not None
            and all(child.get("status") == "completed" for child in previous.get("children", [])),
            "Previous method contains unfinished/failed children")
    return previous


def resolve_args(args):
    for name in ("snapshot", "source_root", "validation_source_root", "beyondmimic_source",
                 "motion_file", "reference_file", "asset_path", "mujoco_model", "task_profile",
                 "protocol", "baseline_root", "output_dir", "previous_sequence_status"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    require(args.snapshot == ROOT.resolve(), "Execute from the declared new frozen torque snapshot")
    require(not args.output_dir.exists(), "Preserve existing runs: choose a fresh torque output directory")
    verify_previous_sequence(args.previous_sequence_status)
    require(not args.output_dir.is_relative_to(args.previous_sequence_status.parent),
            "New torque outputs must not be placed inside the previous sequence tree")
    for base in (args.snapshot, args.source_root, args.validation_source_root, args.beyondmimic_source):
        require(base.is_dir(), "Source directory is missing: "+str(base))
        require(not args.output_dir.is_relative_to(base), "Outputs must be outside every frozen source tree")
    args.protocol_data = json.loads(args.protocol.read_text())
    expected = {"backend": "physx", "method": "additional_leg_applied_torque_rate_only",
        "reference_id": "drift_drl_traj4", "seed": 42, "num_envs": 512, "iterations": 1000,
        "steps_per_environment_per_iteration": 24, "environment_transitions_per_candidate": 12288000,
        "setup_smoke_control_steps": 0, "from_scratch_per_candidate": True,
        "baseline_action_rate_weight": -.1, "leg_torque_rate_weight": -.01,
        "validation_mu_in_order": [.7, .15], "physics_dt_s": .005, "policy_dt_s": .02,
        "core_interval_s": [1.44, 4.], "spectrum_cutoff_hz": 10.}
    for name, value in expected.items():
        require(args.protocol_data.get(name) == value, "Unsupported torque protocol field: "+name)
    screen = args.protocol_data["screening"]
    expected_screen = {
        "required_reduction_of_high_frequency_base_world_xy_angular_velocity_rms": .5,
        "required_reduction_of_leg_action_step_rms": .5, "maximum_core_xy_rmse_m": .15,
        "maximum_core_xy_rmse_increase_over_matched_mu_baseline_m": .03,
        "minimum_reference_speed_retention": .9, "maximum_beta_rmse_increase_over_matched_mu_baseline_deg": 5.,
        "height_or_airborne_failure_gate": False, "sideslip_zero_target": False,
        "translation_and_root_height_hf_diagnostics_required": True}
    for name, value in expected_screen.items():
        require(screen.get(name) == value, "Predeclared screen changed: "+name)
    require(common.sha256(args.task_profile) == args.protocol_data["baseline_task_profile_sha256"],
            "Frozen historical task profile SHA differs")
    for name in ("train_ppo_torque_smoothness.py", "ppo_leg_torque_smoothness.py", "train_beyondmimic_style.py",
                 "height_smoothness.py", "run_car_validation.py", "analyze_ppo_smoothness.py"):
        require((args.snapshot/"scripts"/name).is_file(), "Missing frozen source: "+name)
    require(args.python.is_file() and os.access(args.python, os.X_OK), "Explicit Python executable is unavailable")
    for mu in args.protocol_data["validation_mu_in_order"]:
        for name in ("physics_trajectory.npz", "report.json"):
            require((args.baseline_root/common.case_name(mu)/"isaac"/name).is_file(), "Missing baseline recording: "+name)


def freeze_inputs(args):
    snapshot = common.snapshot_hashes(args)
    previous = {"path": str(args.previous_sequence_status), "sha256": common.sha256(args.previous_sequence_status)}
    payload = {"source_snapshot": snapshot, "previous_sequence_status": previous}
    payload["combined_tree_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def verify_frozen(args, frozen):
    common.verify_snapshot(args, frozen["source_snapshot"])
    if common.sha256(args.previous_sequence_status) != frozen["previous_sequence_status"]["sha256"]:
        raise common.SourceChangedError("Previous completed sequence status changed after freezing")
    verify_previous_sequence(args.previous_sequence_status)


def training_command(args, output, *, smoke):
    command = [str(args.python), "-u", str(args.snapshot/"scripts/train_ppo_torque_smoothness.py"),
        "--source-root", str(args.source_root), "--motion-file", str(args.motion_file),
        "--asset-path", str(args.asset_path), "--task-profile", str(args.task_profile),
        "--leg-torque-rate-weight", str(args.protocol_data["leg_torque_rate_weight"]), "--output-dir", str(output),
        "--num-envs", "2" if smoke else "512", "--iterations", "1" if smoke else "1000",
        "--seed", "42", "--smoke-steps", "0", "--ground-friction", "1.0", "--save-interval", "100",
        "--device", DEVICE, "--headless", "--kit_args", KIT_ARGS]
    if smoke:
        command.append("--smoke-only")
    return command


def finalize_owned_training(args, output, child, *, smoke, finalizer):
    output = Path(output)
    require(child.get("phase") == ("smoke" if smoke else "training") and "returncode" in child
            and output.is_relative_to(args.output_dir) and str(output) in child["command"],
            "Only the exact exited owned training child may be finalized")
    # Failure finalization is also attempted: it never promotes a failed child.
    finalized = finalizer(output, child_returncode=child["returncode"])
    if child["returncode"] != 0:
        raise common.ChildFailedError("Owned torque training child exited nonzero; no retry")
    status = json.loads((output/"run_status.json").read_text())
    manifest = json.loads((output/"manifest.json").read_text())
    runtime = json.loads((output/"torque_rate_runtime_audit.json").read_text())
    expected_steps, envs, iterations = (24, 2, 1) if smoke else (24000, 512, 1000)
    expected_status = "smoke_passed" if smoke else "training_completed"
    transitions = expected_steps*envs
    require(finalized.get("kind") == "historical_physx_ppo_leg_applied_torque_rate_ablation"
            and finalized.get("status") == status.get("status") == expected_status
            and finalized.get("trained_policy_claimed") is (not smoke)
            and finalized.get("smoke_only") is smoke and finalized.get("prepared_method_runtime_verified") is True
            and finalized.get("initialization") == "from_scratch_no_checkpoint_resume",
            "Torque child/finalizer did not complete its declared scope")
    require(status.get("environment_transitions") == finalized.get("expected_environment_transitions")
            == finalized.get("actual_environment_transitions") == manifest.get("environment_transitions_requested") == transitions
            and finalized.get("expected_policy_steps") == expected_steps
            and finalized.get("num_envs") == manifest.get("num_envs") == envs
            and finalized.get("iterations") == manifest.get("iterations") == iterations
            and finalized.get("seed") == manifest.get("seed") == 42
            and manifest.get("setup_smoke_control_steps") == 0,
            "Torque training budget or seed differs")
    reward = finalized["additional_reward"]
    require(finalized.get("baseline_action_rate_weight") == manifest.get("baseline_action_rate_weight") == -.1
            and reward.get("term") == TERM and reward.get("weight") == -.01
            and reward.get("delta_torque_scale_nm") == 20. and reward.get("wheel_cost_included") is False
            and manifest.get("additional_training_reward_terms") == {TERM: reward},
            "New torque reward or baseline action-rate configuration differs")
    require(runtime.get("steps_started") == runtime.get("steps_completed") == runtime.get("unique_cost_windows") == expected_steps
            and runtime.get("physics_samples_total") == 4*expected_steps and runtime.get("steps_failed") == 0
            and runtime.get("samples_last_window") == 4 and runtime.get("device_health_finite") is True
            and runtime.get("closed") is True, "Missing exact 200-Hz physical sampling audit")
    checkpoint = output/("model_smoke.pt" if smoke else "model_final.pt")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0
            and finalized.get("checkpoint_sha256") == common.sha256(checkpoint)
            and finalized.get("final_training_manifest_sha256") == common.sha256(output/"manifest.json")
            and finalized.get("runtime_audit_sha256") == common.sha256(output/"torque_rate_runtime_audit.json")
            and finalized.get("run_status_sha256") == common.sha256(output/"run_status.json"),
            "Finalized torque artifacts are not hash-bound")
    return {"status": expected_status, "environment_transitions": transitions,
        "expected_policy_steps": expected_steps, "leg_torque_rate_weight": -.01, "baseline_action_rate_weight": -.1,
        "checkpoint_sha256": finalized["checkpoint_sha256"],
        "ablation_manifest_sha256": common.sha256(output/"ablation_manifest.json"),
        "physical_sampling_audit_sha256": finalized["runtime_audit_sha256"],
        "physical_sampling_verified": True}


def run_trial(args, *, gpu_probe=common.query_gpu_one, child_runner=common.run_child,
              validation_runner=common.run_validation, trajectory_analyzer=None,
              candidate_comparator=None, training_finalizer=None, quiesce=time.sleep):
    resolve_args(args)
    if trajectory_analyzer is None or candidate_comparator is None:
        from analyze_ppo_smoothness import analyze_trajectory, compare_candidate
        trajectory_analyzer = trajectory_analyzer or analyze_trajectory
        candidate_comparator = candidate_comparator or compare_candidate
    if training_finalizer is None:
        from train_ppo_torque_smoothness import finalize_manifest
        training_finalizer = finalize_manifest
    frozen = freeze_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir/"torque_trial_status.json"
    status = {"status": "initializing", "method": args.protocol_data["method"], "protocol": args.protocol_data,
        "protocol_sha256": common.sha256(args.protocol), "source_tree_sha256": frozen["combined_tree_sha256"],
        "previous_sequence_status_sha256": frozen["previous_sequence_status"]["sha256"],
        "pid": os.getpid(), "host": platform.node(), "started_unix_time": time.time(),
        "physical_gpu_index": GPU_INDEX, "device": DEVICE, "cuda_visible_devices": None,
        "automatic_retries": False, "exclusive_gpu_reservation_claimed": False,
        "children": [], "comparisons": {}, "jitter_resolved_claimed": False,
        "whole_body_jitter_reduction_claimed": False, "requires_fullbody_visual_review": True,
        "requires_translation_and_root_height_hf_diagnostics": True,
        "safety_or_statistical_qualification_claimed": False}
    common.write_json(args.output_dir/"frozen_source_and_inputs.json", frozen)
    common.write_json(status_path, status)
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
            raise common.GpuBusyError("Physical GPU 1 not freshly fully idle; no child launched")
        child.update(status="running", started_unix_time=time.time())
        status["status"] = "running_"+phase
        log_path = args.output_dir/f"child_{len(status['children']):03d}_{phase}.log"
        child["log_path"] = str(log_path); common.write_json(status_path, status)
        def spawned(pid):
            child["pid"] = int(pid); common.write_json(status_path, status)
        code = child_runner(command, log_path, cwd, environment, spawned)
        child.update(returncode=code, finished_unix_time=time.time(), status="completed" if code == 0 else "failed")
        common.write_json(status_path, status)
        child["post_owned_child_quiescence_s"] = 2.; quiesce(2.)
        verify_frozen(args, frozen)
        if code != 0 and phase not in ("smoke", "training"):
            raise common.ChildFailedError("Validation child failed; no retry")
        return child

    try:
        baselines = {}; motion_hash = common.sha256(args.motion_file)
        for mu in args.protocol_data["validation_mu_in_order"]:
            folder = args.baseline_root/common.case_name(mu)/"isaac"
            common.verify_evaluation(folder, mu, args.protocol_data["baseline_checkpoint_sha256"], motion_hash)
            measured = trajectory_analyzer(folder/"physics_trajectory.npz", reference=args.motion_file,
                report=folder/"report.json", high_frequency_hz=10.)
            require(measured.get("evaluation_mu") == mu and measured.get("reference_sha256") == motion_hash
                    and measured.get("checkpoint_sha256") == args.protocol_data["baseline_checkpoint_sha256"],
                    "Baseline analysis identity differs")
            baselines[mu] = measured
        common.write_json(args.output_dir/"baseline_smoothness.json", {str(mu): value for mu, value in baselines.items()})
        for smoke in (True, False):
            folder = args.output_dir/("smoke" if smoke else "training")
            child = launch_checked(training_command(args, folder, smoke=smoke), args.snapshot, "smoke" if smoke else "training")
            proof = finalize_owned_training(args, folder, child, smoke=smoke, finalizer=training_finalizer)
            status["smoke" if smoke else "training"] = proof
            child["parent_finalization"] = proof; common.write_json(status_path, status)
        validation = args.output_dir/"validation"
        validation_runner(args, args.output_dir/"training", validation, launch_checked)
        for mu in args.protocol_data["validation_mu_in_order"]:
            folder = validation/common.case_name(mu)/"isaac"
            checkpoint = status["training"]["checkpoint_sha256"]
            common.verify_evaluation(folder, mu, checkpoint, motion_hash, require_gpu_one=True)
            measured = trajectory_analyzer(folder/"physics_trajectory.npz", reference=args.motion_file,
                report=folder/"report.json", high_frequency_hz=10.)
            require(measured.get("evaluation_mu") == mu and measured.get("reference_sha256") == motion_hash
                    and measured.get("checkpoint_sha256") == checkpoint, "Candidate analysis identity differs")
            comparison = candidate_comparator(measured, baselines[mu], reduction_fraction=.5,
                xy_max_rmse_m=.15, xy_extra_rmse_m=.03, speed_retention_min=.9, beta_extra_rmse_deg=5.)
            status["comparisons"][str(mu)] = comparison
            common.write_json(args.output_dir/(common.case_name(mu)+"_smoothness.json"),
                              {"candidate": measured, "baseline": baselines[mu], "comparison": comparison})
        passed = len(status["comparisons"]) == 2 and all(value.get("passes") is True for value in status["comparisons"].values())
        status.update(status="screen_passed_requires_fullbody_and_translation_review" if passed else "needs_next_method",
                      finished_unix_time=time.time())
        common.write_json(status_path, status)
        return 0 if passed else 4
    except BaseException as error:
        status.update(status="stopped_gpu_busy" if isinstance(error, common.GpuBusyError) else "failed",
            error=f"{type(error).__name__}: {error}", finished_unix_time=time.time())
        if status["children"] and status["children"][-1]["status"] == "running":
            status["children"][-1]["status"] = "failed_or_interrupted"
        common.write_json(status_path, status)
        return 3 if isinstance(error, common.GpuBusyError) else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "source-root", "validation-source-root", "beyondmimic-source", "python",
                 "motion-file", "reference-file", "asset-path", "mujoco-model", "task-profile",
                 "baseline-root", "output-dir", "previous-sequence-status"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"config/ppo_leg_torque_smoothness_v1.json")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return run_trial(parse_args(argv))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Torque trial not started: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

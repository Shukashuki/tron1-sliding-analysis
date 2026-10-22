"""Fresh-idle selected-GPU episodes:5ms identity, then2.5ms physics/PD updates.

No training, LPF, gain changes, retries, GPU fallback or foreign-process actions.
The policy remains50Hz. Root must separately authorize/run this prepared driver.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import run_ppo_contact_damping as base
import run_ppo_leg_target_filter as gpu
import eval_ppo_contact_timestep as evaluator


GPU_HELPER_SHA256 = "cbc7bdbd9fd75a6ce98c04e0e0e9dc253956be3fa93acf5e246b73e5e11dd071"
BASELINE_RAW_SHA256 = "54e01f57b01da11c5b0736c572f8b2ebc81ebe2cac8f06c9b3b5be29b1637780"


def command(args, dt, output):
    index = args.physical_gpu_index
    result = base.command(args, 10., output, gpu.evaluator.kit_args_for(index))
    result[2] = str(Path(__file__).with_name("eval_ppo_contact_timestep.py"))
    start = result.index("--leg-kd"); result[start:start+2] = ["--physics-dt-s", str(dt)]
    result[result.index("--device")+1] = f"cuda:{index}"
    result += ["--physical-gpu-index", str(index)]
    return result


def validate_output(folder, physics_dt_s, physical_gpu_index=1):
    import numpy as np
    from analyze_ppo_smoothness import deduplicate_actions
    dt = physics_dt_s
    folder = Path(folder); report = json.loads((folder/"report.json").read_text())
    audit, contract = report["contact_timestep_diagnostic"], report["policy_export"]
    base.require(dt in evaluator.DECIMATION and physical_gpu_index in evaluator.AUTHORIZED_GPU_INDICES, "Undeclared clock/device")
    decimation, device = evaluator.DECIMATION[dt], f"cuda:{physical_gpu_index}"
    count = 349*decimation+1
    base.require(report["status"] == "evaluated" and report["simulator"] == "IsaacLab/PhysX" and
        report["checkpoint_sha256"] == base.CHECKPOINT_SHA and report["reference_frames"] == 349 and
        report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False and
        report["full_horizon_diagnostic"] is True and report["episodes_per_environment"] == 1 and
        report["deterministic_start_frame"] == 0 and report["reference_state_initialization_during_evaluation"] is False and
        report["terminal_states_captured_before_auto_reset"] is True and
        report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .7,
        "Completed deterministic nominal frozen-actor first episode differs")
    base.require(audit["kind"] == evaluator.KIND and audit["physics_dt_s"] == dt and audit["evaluation_decimation"] == decimation and
        audit["physical_gpu_index_requested"] == physical_gpu_index and audit["requested_device"] == device and
        audit["unchanged_leg_kp"] == 500. and audit["unchanged_leg_kd"] == 10. and audit["additional_training_updates"] == 0 and
        audit["leg_target_filter_installed"] is False and audit["additional_torque_reward"] is False and
        contract["contact_timestep_diagnostic"] == audit, "Clock-only/explicit-PD-rate intervention is not isolated")
    base.require(audit["wrapper_sha256"] == base.sha256(evaluator.__file__) == report["evaluator_source_sha256"] and
        audit["physics_clock_helper_sha256"] == evaluator.CLOCK_HELPER_SHA256 and
        audit["provenance_adapter_sha256"] == evaluator.BASE_ADAPTER_SHA256 and
        all(audit["input_sha256"].get(key) == value for key, value in evaluator.base.PINS.items()) and
        audit["evaluation_dependency_sha256"] == evaluator.base.EVALUATION_DEPENDENCIES, "Source/input provenance differs")
    runtime = report["runtime_device_audit"]
    base.require(runtime["status"] == "verified" and runtime["requested_device"] == device and
        runtime["environment_device"] == runtime["physics_sim_view_frontend_device"] == runtime["robot_native_transform_tensor_device"] == device and
        runtime["cuda_visible_devices"] is None and runtime["native_and_tensor_device_consistency_verified"] is True and
        runtime["native_physics_backend_device_ordinal"] == runtime["torch_current_device"] == runtime["physics_cudaDevice_setting"] == physical_gpu_index,
        "Native/unmasked selected-GPU consistency is not verified")
    clock = report["physics_clock_readback"]
    base.require(clock["status"] == "verified" and clock["decimation"] == decimation and
        clock["simulator_physics_dt_s"] == clock["environment_physics_dt_s"] == clock["net_contact_sensor_update_period_s"] == dt and
        clock["policy_dt_s"] == .02 and report["physics_dt_s"] == contract["physics_dt"] == dt and
        contract["decimation"] == decimation and contract["policy_fps"] == 50. and contract["actor_dim"] == 51 and
        contract["critic_dim"] == 120 and contract["action_dim"] == 8,
        "Actual simulator/net-sensor/actor clocks differ from declared timestep")
    refined = contract["physics_clock_refinement"]
    base.require(refined == report["physics_clock_refinement"] and refined["trained_physics_dt_s"] == .005 and
        refined["trained_decimation"] == 4 and refined["actual_physics_dt_s"] == dt and refined["actual_decimation"] == decimation and
        refined["policy_dt_s"] == .02 and refined["policy_fps"] == 50., "Original trained clock evidence differs")
    trained, actual = contract["trained_task_profile"], contract["task_profile"]
    changed = dict(actual); changed["contact_friction"] = trained["contact_friction"]
    bridge = audit["final_export_profile_bridge"]
    base.require(changed == trained and trained["contact_friction"] == 1. and actual["contact_friction"] == .7 and
        bridge["trained_task_profile_canonical_sha256"] == evaluator._canonical_sha(trained) and
        bridge["evaluation_task_profile_canonical_sha256"] == evaluator._canonical_sha(actual) and
        bridge["trained_profile_modified"] is False and bridge["runtime_material_modified_by_bridge"] is False,
        "Export bridge or truthful trained/evaluated profile evidence differs")
    materials = report["contact_material_audit"]
    base.require(materials["leg_pd_gain_readback"] == {"kp_range": [500., 500.], "kd_range": [10., 10.]} and
        np.allclose(materials["robot_runtime_friction_range"], [.7, .7], rtol=0, atol=1e-6) and
        all(np.isclose(row["static_friction"], .7, rtol=0, atol=1e-6) and np.isclose(row["dynamic_friction"], .7, rtol=0, atol=1e-6)
            for row in materials["ground_colliders"]) and len(materials["ground_colliders"]) > 0,
        "Runtime material or explicit DC-motor PD differs")
    path = folder/"physics_trajectory.npz"; digest = base.sha256(path)
    base.require(digest == report["physics_trajectory_sha256"], "Physical recording/report SHA differs")
    with np.load(path, allow_pickle=False) as archive: raw = {key: archive[key] for key in archive.files}
    base.require(raw["time_s"].shape == (count,) and np.array_equal(raw["time_s"], np.arange(count)*dt) and
        raw["valid_mask"].shape == (count, 1) and raw["valid_mask"].dtype.kind == "b" and raw["valid_mask"].all() and
        float(raw["physics_dt_s"]) == dt and float(raw["policy_dt_s"]) == .02 and
        np.isclose(raw["time_s"][-1], 6.98, rtol=0, atol=1e-9), "Require full native6.98s original episode, no resets/resampling")
    for name, width in (("root_position_m", 3), ("root_quaternion_wxyz", 4), ("root_linear_velocity_w_m_s", 3),
        ("root_angular_velocity_w_rad_s", 3), ("joint_position_rad", 8), ("joint_velocity_rad_s", 8), ("applied_joint_torque_nm", 8)):
        base.require(raw[name].shape == (count, 1, width) and np.isfinite(raw[name]).all(), "Invalid native physical state: "+name)
    decision = deduplicate_actions(raw["time_s"], raw["actions_clipped"][:, 0], raw["action_was_applied"][:, 0], .02,
        frames=raw["reference_frame_for_action"][:, 0], counters=raw["simulation_step_counter"])
    base.require(decision["decisions"] == 349 and decision["decimation"] == decimation and decision["held_physical_records"] == 349*decimation,
        "Require actual349 original50Hz decisions and every native physical step")
    if dt == .005: base.require(digest == BASELINE_RAW_SHA256, "5ms identity must reproduce original54e01 baseline before2.5ms")
    return {"status": "verified", "physics_dt_s": dt, "decimation": decimation, "native_states": count,
        "policy_decisions": 349, "duration_s": 6.98, "report_sha256": base.sha256(folder/"report.json"), "recording_sha256": digest,
        "physical_gpu_index_requested": physical_gpu_index, "native_clock_and_device_verified": True}


def parser():
    result = base.parser()
    result.add_argument("--physical-gpu-index", type=int, choices=evaluator.AUTHORIZED_GPU_INDICES, required=True)
    return result


def run(args, *, guard=None, gpu_probe=None, quiesce=time.sleep):
    for name in ("guard_source", "legacy_evaluator", "motion_file", "asset_path", "checkpoint", "runner_config", "task_profile", "output_dir", "beyondmimic_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    base.require(args.physical_gpu_index in evaluator.AUTHORIZED_GPU_INDICES and os.environ.get("CUDA_VISIBLE_DEVICES") is None,
        "Explicit authorized unmasked physical GPU required")
    base.require(not args.output_dir.exists(), "Preserve existing results: choose a fresh output directory")
    base.require(base.sha256(gpu.__file__) == GPU_HELPER_SHA256 and base.sha256(args.checkpoint) == base.CHECKPOINT_SHA and
        base.sha256(args.legacy_evaluator) == base.BASE_EVAL_SHA, "Frozen guard, checkpoint or evaluator differs")
    official = args.beyondmimic_source/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    base.require(official.is_file(), "Explicit official BeyondMimic source missing")
    for source in (Path(__file__).resolve().parents[1], args.legacy_evaluator.parent.parent, args.beyondmimic_source, args.checkpoint.parent):
        base.require(not args.output_dir.is_relative_to(source), "Output overlaps frozen inputs/source")
    guard = guard or base.load_guard(args.guard_source); frozen = base.freeze(args)
    args.output_dir.mkdir(parents=True, exist_ok=False); status_path = args.output_dir/"contact_timestep_status.json"
    status = {"kind": evaluator.KIND, "status": "initializing", "pid": os.getpid(), "dispatcher_argv": list(sys.argv),
        "started_unix_time": time.time(), "physics_dt_s_in_order": [.005, .0025], "policy_fps": 50.,
        "physical_gpu_index": args.physical_gpu_index, "additional_training_updates": 0, "automatic_retry": False,
        "automatic_gpu_fallback": False, "exclusive_gpu_reservation_claimed": False, "contact_success_claimed": False, "children": []}
    guard.write_json(args.output_dir/"frozen_source_and_inputs.json", frozen); guard.write_json(status_path, status)
    environment = os.environ.copy(); environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1", TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    try:
        selected_identity = None
        for dt in (.005, .0025):
            base.require(base.freeze(args) == frozen, "Frozen source/input changed")
            evidence = (gpu_probe or gpu.query_gpu)(args.physical_gpu_index)
            base.require(evidence["idle"] is True and evidence["physical_index"] == args.physical_gpu_index and
                evidence["pci_bus_id_cuda_ordinal_verified"] is True, "Selected GPU not freshly fully idle/PCI-matched; no fallback")
            identity = (evidence["uuid"], evidence["pci_bus_id"])
            if selected_identity is None: selected_identity = identity
            else: base.require(identity == selected_identity, "Selected GPU identity changed")
            label = "dt0p005" if dt == .005 else "dt0p0025"; folder = args.output_dir/label
            child = {"physics_dt_s": dt, "gpu_preflight": evidence, "command": command(args, dt, folder), "status": "running", "started_unix_time": time.time()}
            status["children"].append(child); status["status"] = "running_"+label; guard.write_json(status_path, status)
            def spawned(pid): child["pid"] = int(pid); guard.write_json(status_path, status)
            code = guard.run_child(child["command"], args.output_dir/(label+".log"), Path(__file__).resolve().parents[1], environment, spawned)
            child.update(returncode=code, finished_unix_time=time.time()); guard.write_json(status_path, status)
            base.require(code == 0, "Owned evaluator failed; no retry/next timestep")
            child.update(verification=validate_output(folder, dt, args.physical_gpu_index), status="completed_verified")
            guard.write_json(status_path, status); quiesce(2.); base.require(base.freeze(args) == frozen, "Frozen source/input changed after child")
        status["status"] = "episodes_complete_requires_contact_and_tracking_review"
    except BaseException as error:
        status.update(status="stopped_without_retry", error=f"{type(error).__name__}: {error}")
        if status["children"] and status["children"][-1]["status"] == "running": status["children"][-1]["status"] = "failed_or_unverified"
        raise
    finally:
        status["finished_unix_time"] = time.time(); guard.write_json(status_path, status)
    return status


if __name__ == "__main__": print(json.dumps(run(parser().parse_args())["status"]))

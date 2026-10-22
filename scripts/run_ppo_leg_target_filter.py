"""Sequential alpha1 identity then alpha.5 leg-target diagnostic, fixed actor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import run_ppo_contact_damping as base
import eval_ppo_leg_target_filter as evaluator

BASELINE_RAW_SHA256 = "54e01f57b01da11c5b0736c572f8b2ebc81ebe2cac8f06c9b3b5be29b1637780"


def _pci_key(value):
    domain, bus, slot = value.lower().split(":")
    device, function = slot.split(".")
    return tuple(int(part, 16) for part in (domain, bus, device, function))


def parse_gpu_snapshot(xml_text, physical_gpu_index):
    """All NVML compute+graphics processes; fail closed on ordinal ambiguity."""
    base.require(physical_gpu_index in evaluator.AUTHORIZED_GPU_INDICES, "Unauthorized physical GPU index")
    devices = ET.fromstring(xml_text).findall("gpu")
    base.require(len(devices) > physical_gpu_index, "Selected physical GPU unavailable")
    selected = devices[physical_gpu_index]
    identities = [(item.findtext("uuid"), item.findtext("pci/pci_bus_id")) for item in devices]
    base.require(all(uuid and pci for uuid, pci in identities) and len({uuid for uuid, _ in identities}) == len(devices)
        and len({_pci_key(pci) for _, pci in identities}) == len(devices), "GPU UUID/PCI identities unavailable or ambiguous")
    uuid, pci = identities[physical_gpu_index]
    ordered = sorted(devices, key=lambda item: _pci_key(item.findtext("pci/pci_bus_id")))
    base.require(ordered[physical_gpu_index].findtext("uuid") == uuid, "Physical GPU index is not matching unmasked PCI_BUS_ID CUDA ordinal")
    try:
        memory = int(selected.findtext("fb_memory_usage/used").split()[0])
        utilization = int(selected.findtext("utilization/gpu_util").split()[0])
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("GPU memory/utilization unavailable; fail closed") from error
    base.require(memory >= 0 and 0 <= utilization <= 100, "Invalid GPU memory/utilization")
    process_node = selected.find("processes")
    base.require(process_node is not None and (process_node.text or "").strip().upper() not in ("N/A", "NOT SUPPORTED"), "Complete GPU process enumeration unavailable")
    processes = [{child.tag: child.text for child in row} for row in process_node.findall("process_info")]
    return {"physical_index": physical_gpu_index, "uuid": uuid, "pci_bus_id": pci, "memory_used_mib": memory,
        "utilization_percent": utilization, "all_reported_compute_and_graphics_processes": processes,
        "idle": not processes and memory < 64 and utilization == 0, "observed_unix_time": time.time(),
        "reservation_claimed": False, "pci_bus_id_cuda_ordinal_verified": True,
        "source": "read-only nvidia-smi -q -x (NVML-derived full process snapshot)"}


def query_gpu(physical_gpu_index):
    result = subprocess.run(["nvidia-smi", "-q", "-x"], text=True, capture_output=True, timeout=20, check=True)
    return parse_gpu_snapshot(result.stdout, physical_gpu_index)


def command(args, alpha, output, kit_args):
    index = getattr(args, "physical_gpu_index", 1)
    result = base.command(args, 10., output, evaluator.kit_args_for(index)); result[2] = str(Path(__file__).with_name("eval_ppo_leg_target_filter.py"))
    start = result.index("--leg-kd"); result[start:start+2] = ["--alpha", str(alpha)]
    result[result.index("--device")+1] = f"cuda:{index}"
    result.extend(["--physical-gpu-index", str(index)])
    return result


def validate_output(folder, alpha, physical_gpu_index=1):
    import numpy as np
    report = json.loads((folder/"report.json").read_text()); audit = report["leg_target_filter_diagnostic"]
    requested_device = f"cuda:{physical_gpu_index}"
    base.require(physical_gpu_index in evaluator.AUTHORIZED_GPU_INDICES
        and audit["physical_gpu_index_requested"] == physical_gpu_index and audit["requested_device"] == requested_device,
        "Diagnostic selected GPU differs from requested device")
    device = report["runtime_device_audit"]
    base.require(device["status"] == "verified" and device["requested_device"] == requested_device
        and device["environment_device"] == device["physics_sim_view_frontend_device"] == device["robot_native_transform_tensor_device"] == requested_device
        and device["cuda_visible_devices"] is None and device["native_and_tensor_device_consistency_verified"] is True
        and device["native_physics_backend_device_ordinal"] == device["torch_current_device"] == device["physics_cudaDevice_setting"] == physical_gpu_index,
        "Native/tensor/renderer physical GPU audit is inconsistent or masked")
    base.require(report["status"] == "evaluated" and report["simulator"] == "IsaacLab/PhysX"
        and report["checkpoint_sha256"] == base.CHECKPOINT_SHA and report["physics_dt_s"] == .005
        and report["reference_frames"] == 349 and report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .7,
        "Completed nominal frozen-actor identity differs")
    base.require(audit["kind"] == evaluator.KIND and audit["alpha"] == alpha and audit["unchanged_leg_kp"] == 500.
        and audit["unchanged_leg_kd"] == 10. and audit["configured_solver_position_iterations"] == 8
        and audit["configured_solver_velocity_iterations"] == 4 and audit["additional_training_updates"] == 0
        and audit["diagnostic_only"] is True and audit["processed_leg_position_targets_only"] is True
        and audit["only_changed_control_field"] == ("six_leg_processed_position_target_lowpass" if alpha < 1 else None)
        and audit["actor_observation_layout_modified"] is False and audit["requested_action_history_modified"] is False
        and audit["wheel_requests_modified"] is False, "Target-filter intervention is not isolated")
    base.require(audit["wrapper_sha256"] == base.sha256(evaluator.__file__) and report["evaluator_source_sha256"] == audit["wrapper_sha256"]
        and audit["filter_module_sha256"] == evaluator.FILTER_MODULE_SHA256
        and audit["provenance_adapter_sha256"] == evaluator.BASE_ADAPTER_SHA256
        and audit["evaluation_dependency_sha256"] == evaluator.base.EVALUATION_DEPENDENCIES
        and all(audit["input_sha256"][key] == value for key, value in evaluator.base.PINS.items()), "Frozen filter source/input provenance differs")
    base.require(report["policy_export"]["leg_target_filter_diagnostic"] == audit
        and report["policy_export"]["actor_export_includes_leg_target_filter"] is False
        and report["contact_material_audit"]["leg_pd_gain_readback"] == {"kp_range": [500., 500.], "kd_range": [10., 10.]}, "Final contract/gain evidence differs")
    evidence = audit["final_runtime_evidence"]
    base.require(evidence["finalized_after_rollout"] is True and evidence["verification_status"] == "verified"
        and evidence["control_steps"] == 349 and evidence["physics_apply_calls"] == 1396,
        "Final runtime evidence—not installation-time counters—is required")
    for name, key in ((evaluator.RUNTIME_FILE, "runtime_audit_sha256"), (evaluator.CONTROL_FILE, "control_npz_sha256"), (evaluator.PHYSICS_FILE, "physics_npz_sha256")):
        base.require(base.sha256(folder/name) == evidence[key], "Separate filter evidence hash differs: "+name)
    final = json.loads((folder/evaluator.RUNTIME_FILE).read_text())
    base.require(final["kind"] == evaluator.KIND and final["alpha"] == alpha and final["finalized_after_rollout"] is True
        and final["verification_status"] == "verified"
        and final["control_npz_sha256"] == evidence["control_npz_sha256"] and final["physics_npz_sha256"] == evidence["physics_npz_sha256"], "Final runtime/telemetry binding differs")
    with np.load(folder/evaluator.CONTROL_FILE, allow_pickle=False) as archive: control = {key: archive[key] for key in archive.files}
    with np.load(folder/evaluator.PHYSICS_FILE, allow_pickle=False) as archive: physics = {key: archive[key] for key in archive.files}
    path = folder/"physics_trajectory.npz"; digest = base.sha256(path)
    base.require(digest == report["physics_trajectory_sha256"], "Original physical trajectory/report SHA differs")
    with np.load(path, allow_pickle=False) as archive: raw = {key: archive[key] for key in archive.files}
    base.require(np.array_equal(raw["time_s"], np.arange(1397)*.005) and raw["valid_mask"].shape == (1397, 1)
        and raw["valid_mask"].dtype.kind == "b" and raw["valid_mask"].all() and int(raw["action_was_applied"].sum()) == 1396,
        "Require unchanged complete native first-episode physical recording")
    base.require(np.array_equal(control["requested_actions_clipped"], raw["actions_clipped"][1::4, 0]), "Original recorder must contain requested—not filtered—actions")
    names = tuple(raw["joint_names"].tolist()); legs = report["policy_export"]["leg_joint_names"]
    initial = raw["joint_position_rad"][0, 0, [names.index(name) for name in legs]]
    proof = evaluator.verify_telemetry(control, physics, final["filter_handler_audit"], alpha,
        initial_joint_position=initial, requested_device=requested_device)
    observations = raw["action_observation"][1::4, 0]
    previous = np.concatenate((np.zeros((1, 8), dtype=control["requested_actions_clipped"].dtype), control["requested_actions_clipped"][:-1]), axis=0)
    base.require(observations.shape == (349, 51) and np.array_equal(observations[:, -8:], previous), "Actor previous-action observations were not original requested clipped actions")
    if alpha == 1.:
        base.require(digest == BASELINE_RAW_SHA256, "Alpha1 same-hook identity must reproduce original54e01 baseline before alpha.5")
    return {"status": "verified", "alpha": alpha, "report_sha256": base.sha256(folder/"report.json"), "recording_sha256": digest,
        "runtime_audit_sha256": evidence["runtime_audit_sha256"], "control_npz_sha256": evidence["control_npz_sha256"],
        "physics_npz_sha256": evidence["physics_npz_sha256"], "telemetry_verification": proof}


def run(args, *, guard=None, quiesce=time.sleep, gpu_probe=None):
    for name in ("guard_source", "legacy_evaluator", "motion_file", "asset_path", "checkpoint", "runner_config", "task_profile", "output_dir", "beyondmimic_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    args.physical_gpu_index = getattr(args, "physical_gpu_index", 1)
    base.require(args.physical_gpu_index in evaluator.AUTHORIZED_GPU_INDICES, "Unauthorized physical GPU index")
    base.require(not args.output_dir.exists(), "Preserve existing outputs: choose a fresh directory")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Require an unmasked selected physical GPU")
    base.require(base.sha256(args.checkpoint) == base.CHECKPOINT_SHA and base.sha256(args.legacy_evaluator) == base.BASE_EVAL_SHA, "Frozen actor/base evaluator SHA differs")
    base.require(base.sha256(Path(evaluator.__file__).with_name("ppo_leg_target_filter.py")) == evaluator.FILTER_MODULE_SHA256, "Frozen filter source differs")
    official = args.beyondmimic_source/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    base.require(official.is_file(), "Explicit official BeyondMimic source/config is missing")
    for source in (Path(__file__).resolve().parents[1], args.legacy_evaluator.parent.parent, args.beyondmimic_source, args.checkpoint.parent):
        base.require(not args.output_dir.is_relative_to(source), "Output must remain outside frozen source/input trees")
    guard = guard or base.load_guard(args.guard_source); frozen = base.freeze(args)
    args.output_dir.mkdir(parents=True, exist_ok=False); status_path = args.output_dir/"leg_target_filter_status.json"
    status = {"kind": "single_factor_six_leg_processed_target_lowpass_diagnostic", "status": "initializing", "pid": os.getpid(),
        "dispatcher_argv": list(sys.argv), "started_unix_time": time.time(), "alpha_in_order": [1., .5], "mu": .7,
        "leg_kp": 500., "leg_kd": 10., "configured_solver_position_iterations": 8, "configured_solver_velocity_iterations": 4,
        "physical_gpu_index": args.physical_gpu_index, "requested_device": f"cuda:{args.physical_gpu_index}",
        "automatic_gpu_fallback": False, "additional_training_updates": 0, "actor_observation_layout_modified": False,
        "requested_action_history_modified": False, "automatic_retry": False, "exclusive_gpu_reservation_claimed": False,
        "contact_success_claimed": False, "children": []}
    guard.write_json(args.output_dir/"frozen_source_and_inputs.json", frozen); guard.write_json(status_path, status)
    environment = os.environ.copy(); environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1", TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    try:
        selected_identity = None
        for alpha in (1., .5):
            base.require(base.freeze(args) == frozen, "Frozen source/input changed")
            gpu = (gpu_probe or query_gpu)(args.physical_gpu_index); status["last_gpu_preflight"] = gpu; guard.write_json(status_path, status)
            base.require(gpu["idle"] is True and gpu["physical_index"] == args.physical_gpu_index,
                "Selected GPU is not freshly fully idle; no child started and no fallback")
            identity = (gpu["uuid"], gpu["pci_bus_id"])
            base.require(gpu["pci_bus_id_cuda_ordinal_verified"] is True, "Unmasked physical/CUDA ordinal not verified")
            if selected_identity is None:
                selected_identity = identity; status["selected_gpu_identity"] = {"uuid": identity[0], "pci_bus_id": identity[1]}
            else:
                base.require(identity == selected_identity, "Selected GPU UUID/PCI identity changed between episodes")
            label = "alpha1p0" if alpha == 1 else "alpha0p5"; folder = args.output_dir/label
            child = {"alpha": alpha, "gpu_preflight": gpu, "command": command(args, alpha, folder, guard.KIT_ARGS),
                "status": "running", "started_unix_time": time.time()}
            status["children"].append(child); status["status"] = "running_"+label; guard.write_json(status_path, status)
            def spawned(pid):
                child["pid"] = int(pid); guard.write_json(status_path, status)
            code = guard.run_child(child["command"], args.output_dir/(label+".log"), Path(__file__).resolve().parents[1], environment, spawned)
            child.update(returncode=code, finished_unix_time=time.time()); guard.write_json(status_path, status)
            base.require(code == 0, "Owned evaluator failed; no retry or next alpha")
            child.update(verification=validate_output(folder, alpha, args.physical_gpu_index), status="completed_verified"); guard.write_json(status_path, status)
            quiesce(2.); base.require(base.freeze(args) == frozen, "Frozen source/input changed after evaluation")
        status["status"] = "episodes_complete_requires_contact_and_tracking_review"
    except BaseException as error:
        status.update(status="stopped_without_retry", error=f"{type(error).__name__}: {error}")
        if status["children"] and status["children"][-1]["status"] == "running": status["children"][-1]["status"] = "failed_or_unverified"
        raise
    finally:
        status["finished_unix_time"] = time.time(); guard.write_json(status_path, status)
    return status


def parser():
    result = base.parser()
    result.description = __doc__
    result.add_argument("--physical-gpu-index", type=int, choices=evaluator.AUTHORIZED_GPU_INDICES, default=1,
        help="Select one unmasked physical GPU for both episodes; each requires a fresh complete idle check, with no fallback")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())["status"]))

"""Sequential, freshly-idle-GPU 400 Hz PPO backend training cohort.

Runs on the authorized host: no SSH, installation, retries, GPU fallback or
foreign-process control. Completion means trained artifacts are verified;
cross-backend/friction validation remains pending, not a success claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GUARD_SHA256 = "d07e7b04af9554744ac70fa7dc149760171e55978da2791d9721ce4916f472e0"
GPU_PROBE_SHA256 = "cbc7bdbd9fd75a6ce98c04e0e0e9dc253956be3fa93acf5e246b73e5e11dd071"
AUTHORIZED_GPU_INDICES = (3, 4, 5)
STAGES = (("parity", None, None), ("physx_smoke", "physx", "smoke"),
          ("mjlab_smoke", "mjlab", "smoke"), ("physx_formal", "physx", "formal"),
          ("mjlab_formal", "mjlab", "formal"))
SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".cpp", ".c", ".h"}
IGNORED = {".git", "__pycache__", ".pytest_cache"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text())
    require(isinstance(result, dict), "Expected JSON object: " + str(path))
    return result


def _load(path, name, expected=None):
    path = Path(path)
    if expected is not None:
        require(sha256(path) == expected, "Pinned helper source differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_snapshot(args):
    """Freeze source and asset bytes, excluding generated outputs/caches."""
    paths = {args.protocol, args.contract, args.task_profile, args.model, args.motion,
             args.asset_path, args.guard_source, args.gpu_probe_source,
             args.isaac_device_audit_source}
    snapshot_directories = ("scripts", "config", "native")
    if (args.snapshot / "upstream").is_dir():
        snapshot_directories += ("upstream",)
    for root, names in ((args.snapshot, snapshot_directories),
                        (args.tron1_source, ("scripts", "training")),
                        (args.beyondmimic_source, ("source",)),
                        (args.mjlab_source, ("src",))):
        for name in names:
            directory = root / name
            require(directory.is_dir(), "Frozen source directory unavailable: " + str(directory))
            for path in directory.rglob("*"):
                if IGNORED.intersection(path.relative_to(directory).parts) or path.suffix not in SOURCE_SUFFIXES:
                    continue
                require(not path.is_symlink(), "Refuse symlinked source: " + str(path))
                if path.is_file():
                    paths.add(path)
    paths.update((args.parity_contract, args.parity_task_profile))
    files = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(paths)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"tree_sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def verify_snapshot(args, frozen):
    require(source_snapshot(args) == frozen, "Frozen source or input bytes changed; no retry or further child")


def training_command(args, backend, mode, folder):
    protocol = args.protocol_data
    settings = protocol["smoke"] if mode == "smoke" else protocol["algorithm"]
    return [str(args.python), "-u", str(args.snapshot / "scripts/train_ppo_backend_400hz.py"),
        "--cohort-protocol", str(args.protocol), "--cohort-run-kind", mode,
        "--declared-input-dir", str(folder.with_name(folder.name + "_declared_inputs")),
        "--isaac-device-audit-source", str(args.isaac_device_audit_source),
        "--backend", backend, "--mjlab-task-implementation", "gpu",
        "--model", str(args.model), "--motion", str(args.motion),
        "--contract", str(args.contract), "--task-profile", str(args.task_profile),
        "--asset-path", str(args.asset_path), "--tron1-source", str(args.tron1_source),
        "--beyondmimic-source", str(args.beyondmimic_source), "--mjlab-source", str(args.mjlab_source),
        "--output-dir", str(folder), "--num-envs", str(settings["num_envs"]),
        "--iterations", str(settings["iterations"]), "--seed", "42", "--smoke-steps", "8",
        "--save-interval", "100", "--device", f"cuda:{args.physical_gpu_index}"]


def parity_command(args, folder):
    # Existing audited benchmark also measures one tiny timing window. It is
    # not a PPO throughput benchmark and does not change the parity gate.
    return [str(args.python), "-u", str(args.snapshot / "scripts/benchmark_mjlab_gpu_task.py"),
        "--model", str(args.model), "--motion", str(args.motion),
        "--contract", str(args.parity_contract), "--task-profile", str(args.parity_task_profile),
        "--mjlab-source", str(args.mjlab_source), "--output-dir", str(folder),
        "--device", f"cuda:{args.physical_gpu_index}", "--seed", "42", "--mu", "0.7",
        "--parity-num-envs", "4", "--parity-steps", "8", "--num-envs", "4",
        "--steps", "1", "--warmup-steps", "0", "--repeats", "1", "--performance-dr-mode", "none"]


def validate_parity(folder, args):
    import numpy as np
    report = read_json(folder / "report.json")
    require(report["status"] == "passed_parity_and_measured_environment_timing"
            and report["parity"]["passed"] is True and report["parity_fixed_mu"] == .7
            and report["parity"]["complete_requested_trace"] is True
            and report["parity"]["uninterrupted_no_auto_resets"] is True
            and report["device"] == f"cuda:{args.physical_gpu_index}"
            and report["training_speedup_claimed"] is False,
            "Actual GPU parity did not pass on the selected device")
    for name in ("host", "gpu"):
        metadata = report[name + "_metadata"]
        require(metadata["physics_dt_s"] == .0025 and metadata["decimation"] == 8
                and metadata["policy_dt_s"] == .02 and metadata["num_envs"] == 4
                and metadata["native_backend"]["device"] == f"cuda:{args.physical_gpu_index}"
                and metadata["native_backend"]["physics_dt_s"] == .0025
                and metadata["native_backend"]["decimation"] == 8
                and metadata["dr"]["mode"] == "none",
                "Parity did not use the same declared 400 Hz GPU physics")
    for key, path in (("model", args.model), ("motion", args.motion),
                      ("contract", args.parity_contract), ("task_profile", args.parity_task_profile)):
        require(report["input_sha256"][key] == sha256(path), "Parity input hash differs: " + key)
    with np.load(folder / "parity_arrays.npz", allow_pickle=False) as archive:
        require(archive["shared_actions"].shape == (8, 4, 8), "Parity action trace must be eight steps/four worlds")
        for name in ("host", "gpu"):
            require(archive[name + "/qpos"].shape[:2] == (8, 4), "Incomplete parity physics trace")
    return {"status": "verified", "report_sha256": sha256(folder / "report.json"),
            "parity_arrays_sha256": sha256(folder / "parity_arrays.npz"),
            "physical_gpu_index": args.physical_gpu_index, "physics_dt_s": .0025,
            "decimation": 8, "policy_dt_s": .02, "steps": 8, "num_envs": 4,
            "ppo_training_performance_claimed": False}


def validate_training(folder, backend, mode, args):
    wrapper = args.wrapper
    report = read_json(folder / "run_status.json")
    completion = read_json(folder / "cohort_completion_audit.json")
    manifest = read_json(folder / "manifest.json")
    contract = read_json(folder / "policy_contract.json")
    startup = read_json(folder / "cohort_startup_audit.json")
    protocol = args.protocol_data
    declaration = protocol["smoke"] if mode == "smoke" else protocol["algorithm"]
    transitions = declaration["transitions"]
    smoke = mode == "smoke"
    require(report["status"] == "training_completed" and report["backend"] == backend
            and report["training_environment_steps"] == transitions,
            "Owned training child did not complete the exact declared budget")
    require(completion["status"] == "completed_verified" and completion["backend"] == backend
            and completion["run_kind"] == mode and completion["cohort_id"] == protocol["cohort_id"]
            and completion["training_environment_steps"] == transitions
            and completion["protocol_sha256"] == sha256(args.protocol)
            and completion["source_files_unchanged"] is True
            and completion["historical_checkpoint_used"] is False
            and completion["additional_reward_terms"] is False,
            "Final cohort completion proof is missing or inconsistent")
    for metadata in (report, completion, manifest, contract, startup, manifest["cohort"]):
        require(metadata["smoke_only"] is smoke and metadata["trained_policy_claimed"] is not smoke,
                "Smoke pipeline artifact must not be presented as a formal trained policy")
    for name, expected in (("physics_dt_s", .0025), ("decimation", 8), ("policy_dt_s", .02),
                            ("action_rate_weight", -.5), ("grounded_reward_weight", .4)):
        require(completion[name] == expected, "Final cohort field differs: " + name)
    for filename, key in (("model_final.pt", "checkpoint_sha256"), ("policy.pt", "exported_policy_sha256")):
        path = folder / filename
        require(path.is_file() and path.stat().st_size > 0
                and sha256(path) == report[key] == completion[key] == contract[key],
                "Actual checkpoint/export bytes differ: " + filename)
    for filename, key in (("manifest.json", "manifest_sha256"), ("policy_contract.json", "policy_contract_sha256"),
                          ("final_policy_contract.json", "final_policy_contract_sha256"),
                          ("cohort_startup_audit.json", "startup_audit_sha256")):
        require(sha256(folder / filename) == completion[key], "Completed metadata hash differs: " + filename)
    require((folder / "policy_contract.json").read_bytes() == (folder / "final_policy_contract.json").read_bytes(),
            "Canonical final contract and compatibility alias differ")
    cohort = manifest["cohort"]
    require(manifest["backend"] == backend and manifest["num_envs"] == declaration["num_envs"]
            and manifest["iterations"] == declaration["iterations"]
            and cohort["run_kind"] == mode and cohort["id"] == protocol["cohort_id"]
            and cohort["protocol_sha256"] == sha256(args.protocol)
            and cohort["wrapper_sha256"] == sha256(args.snapshot / "scripts/train_ppo_backend_400hz.py")
            and cohort["from_scratch"] is True and cohort["historical_checkpoint_used"] is False
            and cohort["network_seed_after_environment_setup_and_smoke"] is True
            and cohort["normalized_actor_export_verified"] is True,
            "Training cohort/startup provenance differs")
    require(startup["runtime"] == cohort["runtime"] and startup["initial_policy_state_sha256"] ==
            cohort["initial_policy_state_sha256"] == manifest["initial_policy_state_sha256"] ==
            completion["initial_policy_state_sha256"] == report["initial_policy_state_sha256"],
            "Startup/final network initialization or runtime evidence differs")
    for name, digest in cohort["source_files_sha256"].items():
        require(sha256(args.snapshot / name) == digest, "Cohort source changed: " + name)
    require(cohort["derived_input_sha256"] == args.derived_input_sha256,
            "Training inputs differ from the actual parity inputs")
    declared = folder.with_name(folder.name + "_declared_inputs")
    require(sha256(declared / "contract.json") == args.derived_input_sha256["contract"]
            and sha256(declared / "task_profile.json") == args.derived_input_sha256["task_profile"],
            "Training declared input files changed or differ across stages")
    expected_status = "smoke_completed_normalized_actor_not_formal_policy" if smoke else "trained_completed_normalized_actor"
    require(contract["contract_status"] == expected_status and manifest["contract"] == contract
            and contract["policy_training_backend"] == backend and contract["policy_algorithm"] == "PPO"
            and contract["training_environment_transitions"] == transitions
            and contract["physics_dt"] == .0025 and contract["decimation"] == 8
            and contract["task_profile"] == contract["trained_task_profile"] == manifest["task_profile"]
            and contract["task_profile"]["action_rate_weight"] == -.5
            and contract["actor_includes_observation_normalizer"] is True
            and contract["runtime_clock_and_interface_verified"] is True,
            "Final exported actor contract differs from the trained cohort")
    runtime = cohort["runtime"]
    require(runtime["status"] == "verified" and runtime["no_extra_environment_steps"] is True
            and runtime["shared_DR_recipe"] == protocol["shared_dr_recipe"], "Runtime recipe was not verified")
    wrapper.validate_runtime_fields(runtime["fields"])
    device = runtime["device"]
    require(device["requested_device"] == f"cuda:{args.physical_gpu_index}"
            and device["torch_current_device"] == args.physical_gpu_index
            and device["cuda_visible_devices"] is None, "Training used a masked/different CUDA device")
    if backend == "physx":
        require(device["status"] == "verified" and device["native_and_tensor_device_consistency_verified"] is True
                and device["environment_device"] == device["physics_sim_view_frontend_device"] ==
                device["robot_native_transform_tensor_device"] == f"cuda:{args.physical_gpu_index}"
                and device["native_physics_backend_device_ordinal"] == device["physics_cudaDevice_setting"] == args.physical_gpu_index,
                "PhysX native and tensor device ordinals differ")
    else:
        native = device["native_backend"]
        require(device["task_arrays_device"] == native["device"] == f"cuda:{args.physical_gpu_index}"
                and native["physics_dt_s"] == .0025 and native["decimation"] == 8
                and native["host_task_adapter"] is False, "MJWarp GPU task/physics differ")
        counters = read_json(folder / "environment_final.json")["physical_counters"]
        actual_transitions = transitions + 8 * declaration["num_envs"]
        require(counters["environment_transitions"] == actual_transitions
                and counters["physical_environment_substeps"] == actual_transitions * 8,
                "GPU actual substep counters differ from training plus declared setup")
    return {"status": "verified", "backend": backend, "run_kind": mode,
            "training_environment_steps": transitions, "checkpoint_sha256": report["checkpoint_sha256"],
            "exported_policy_sha256": report["exported_policy_sha256"],
            "initial_policy_state_sha256": manifest["initial_policy_state_sha256"],
            "cohort_completion_audit_sha256": sha256(folder / "cohort_completion_audit.json"),
            "manifest_sha256": completion["manifest_sha256"],
            "validation_matrix_status": "pending_not_evaluated"}


def _resolve(args):
    for name in ("snapshot", "protocol", "contract", "task_profile", "model", "motion", "asset_path",
                 "tron1_source", "beyondmimic_source", "mjlab_source", "guard_source", "gpu_probe_source",
                 "isaac_device_audit_source", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()  # Preserve the existing venv symlink.
    require(args.snapshot == ROOT.resolve(), "Execute dispatcher from its declared frozen snapshot")
    args.stage = getattr(args, "stage", "preflight")
    require(args.stage in ("preflight", "formal"), "Unknown dispatcher stage")
    if getattr(args, "preflight_dir", None) is not None:
        args.preflight_dir = args.preflight_dir.expanduser().resolve()
    require((args.stage == "formal") == (getattr(args, "preflight_dir", None) is not None),
            "Formal launch requires explicit completed --preflight-dir; preflight does not accept one")
    require(args.physical_gpu_index in AUTHORIZED_GPU_INDICES, "Only explicitly approved GPU3/4/5 are allowed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Require unmasked physical GPU ordinals")
    require(os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID") == "PCI_BUS_ID", "Require PCI_BUS_ID CUDA order")
    require(not args.output_dir.exists(), "Preserve previous outputs: choose a fresh directory")
    for source in (args.snapshot, args.tron1_source, args.beyondmimic_source, args.mjlab_source):
        require(source.is_dir() and not args.output_dir.is_relative_to(source), "Output must be outside source trees")


def validate_preflight(folder, args, frozen):
    """Require actual unchanged parity+two-smoke records before formal work."""
    status = read_json(folder / "backend_400hz_pair_status.json")
    expected = [stage[0] for stage in STAGES[:3]]
    require(status["status"] == "preflight_completed_requires_formal_authorization"
            and status["physical_gpu_index"] == args.physical_gpu_index
            and status["stages_in_order"] == expected
            and [child["stage"] for child in status["children"]] == expected
            and all(child["status"] == "completed_verified" and child["returncode"] == 0
                    for child in status["children"]), "Preflight is incomplete, failed, or belongs to a different GPU")
    previous = read_json(folder / "frozen_source_and_inputs.json")
    previous_inputs = {str(folder / "parity_declared_inputs" / name) for name in ("contract.json", "task_profile.json")}
    current_inputs = {str(args.parity_contract), str(args.parity_task_profile)}
    require({key: value for key, value in previous["files"].items() if key not in previous_inputs} ==
            {key: value for key, value in frozen["files"].items() if key not in current_inputs},
            "Formal sources/input bytes differ from the verified preflight")
    for name, record in previous["files"].items():
        require(sha256(name) == record["sha256"], "Verified preflight source/input changed: " + name)
    parity = validate_parity(folder / "parity", args)
    require(parity == status["children"][0]["verification"], "Verified parity evidence changed after preflight")
    network = None
    for index, backend in enumerate(("physx", "mjlab"), start=1):
        proof = validate_training(folder / (backend + "_smoke"), backend, "smoke", args)
        require(proof == status["children"][index]["verification"], "Verified smoke evidence changed after preflight")
        if network is None:
            network = proof["initial_policy_state_sha256"]
        require(proof["initial_policy_state_sha256"] == network, "Preflight network initializations differ")
    require(network == status["initial_policy_state_sha256"], "Preflight summary network hash differs")
    return {"status": "verified", "status_sha256": sha256(folder / "backend_400hz_pair_status.json"),
            "initial_policy_state_sha256": network, "selected_gpu_identity": status["selected_gpu_identity"],
            "parity": parity}


def run(args, *, guard=None, gpu_probe=None, wrapper=None, quiesce=time.sleep, child_runner=None):
    _resolve(args)
    guard = guard or _load(args.guard_source, "_backend400_guard", GUARD_SHA256)
    if gpu_probe is None:
        gpu_probe = _load(args.gpu_probe_source, "_backend400_gpu_probe", GPU_PROBE_SHA256).query_gpu
    wrapper = wrapper or _load(args.snapshot / "scripts/train_ppo_backend_400hz.py", "_backend400_trainer")
    args.wrapper = wrapper
    args.protocol_data = wrapper.validate_protocol(read_json(args.protocol))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir / "backend_400hz_pair_status.json"
    stages = STAGES[:3] if args.stage == "preflight" else STAGES[3:]
    status = {"kind": "fresh_matched_400hz_ppo_backend_training_pair", "status": "initializing", "stage": args.stage,
        "pid": os.getpid(), "dispatcher_argv": list(sys.argv), "started_unix_time": time.time(),
        "physical_gpu_index": args.physical_gpu_index, "requested_device": f"cuda:{args.physical_gpu_index}",
        "automatic_retry": False, "automatic_gpu_fallback": False, "exclusive_gpu_reservation_claimed": False,
        "stages_in_order": [stage[0] for stage in stages], "children": [],
        "validation_matrix_status": "pending_not_evaluated", "hardware_ready": False}
    guard.write_json(status_path, status)
    try:
        derived = wrapper.prepare_declared_inputs(args.protocol, args.output_dir / "parity_declared_inputs",
                                                  nominal_path=args.contract, profile_path=args.task_profile)
        args.parity_contract, args.parity_task_profile = derived["contract"], derived["task_profile"]
        args.derived_input_sha256 = derived["derived_input_sha256"]
        status["parity_declared_inputs"] = {"directory": str(args.parity_contract.parent),
                                           "sha256": args.derived_input_sha256}
        frozen = source_snapshot(args)
        guard.write_json(args.output_dir / "frozen_source_and_inputs.json", frozen)
        environment = os.environ.copy()
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
                           TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
        identity, initial_network = None, None
        if args.stage == "formal":
            proof = validate_preflight(args.preflight_dir, args, frozen)
            status["preflight_verification"] = proof
            initial_network = proof["initial_policy_state_sha256"]
            identity = (proof["selected_gpu_identity"]["uuid"], proof["selected_gpu_identity"]["pci_bus_id"])
            status["selected_gpu_identity"] = proof["selected_gpu_identity"]
        for label, backend, mode in stages:
            verify_snapshot(args, frozen)
            gpu = gpu_probe(args.physical_gpu_index)
            status["last_gpu_preflight"] = gpu
            guard.write_json(status_path, status)
            require(gpu["idle"] is True and gpu["physical_index"] == args.physical_gpu_index
                    and gpu["pci_bus_id_cuda_ordinal_verified"] is True
                    and not gpu["all_reported_compute_and_graphics_processes"],
                    "Selected GPU is not freshly fully idle; no child/fallback/retry")
            current_identity = (gpu["uuid"], gpu["pci_bus_id"])
            if identity is None:
                identity = current_identity
                status["selected_gpu_identity"] = {"uuid": identity[0], "pci_bus_id": identity[1]}
            require(identity == current_identity, "GPU UUID/PCI identity changed between owned children")
            folder = args.output_dir / label
            command = parity_command(args, folder) if backend is None else training_command(args, backend, mode, folder)
            child = {"stage": label, "backend": backend, "run_kind": mode, "command": command,
                     "gpu_preflight": gpu, "status": "running", "started_unix_time": time.time()}
            status["children"].append(child)
            status["status"] = "running_" + label
            guard.write_json(status_path, status)

            def spawned(pid):
                child["pid"] = int(pid)
                guard.write_json(status_path, status)

            code = (child_runner or guard.run_child)(command, args.output_dir / (label + ".log"),
                                                     args.snapshot, environment, spawned)
            child.update(returncode=code, finished_unix_time=time.time())
            guard.write_json(status_path, status)
            require(code == 0, "Owned child failed; no automatic retry or next stage")
            verify_snapshot(args, frozen)
            proof = validate_parity(folder, args) if backend is None else validate_training(folder, backend, mode, args)
            if backend is not None:
                if initial_network is None:
                    initial_network = proof["initial_policy_state_sha256"]
                require(proof["initial_policy_state_sha256"] == initial_network,
                        "Initial policy network differs across training arms/modes")
            child.update(status="completed_verified", verification=proof)
            guard.write_json(status_path, status)
            quiesce(2.)  # Only our returned child; allow stale one-second NVML averages to settle.
        status["status"] = ("preflight_completed_requires_formal_authorization" if args.stage == "preflight"
                            else "training_pair_completed_validation_pending")
        status["initial_policy_state_sha256"] = initial_network
    except BaseException as error:
        status.update(status="stopped_without_retry", error=f"{type(error).__name__}: {error}")
        if status["children"] and status["children"][-1]["status"] == "running":
            status["children"][-1]["status"] = "failed_or_unverified"
        raise
    finally:
        status["finished_unix_time"] = time.time()
        guard.write_json(status_path, status)
    return status


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--snapshot", type=Path, default=ROOT)
    result.add_argument("--protocol", type=Path, default=ROOT / "config/ppo_backend_400hz_comparison_v1.json")
    result.add_argument("--contract", type=Path, default=ROOT / "config/nominal_actuator_contract.json")
    result.add_argument("--task-profile", type=Path, default=ROOT / "config/car_drift_height_smooth_task.json")
    for name in ("model", "motion", "asset-path", "tron1-source", "beyondmimic-source", "mjlab-source",
                 "isaac-device-audit-source", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--guard-source", type=Path, default=ROOT / "scripts/run_ppo_smoothness_sequence.py")
    result.add_argument("--gpu-probe-source", type=Path, default=ROOT / "scripts/run_ppo_leg_target_filter.py")
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument("--physical-gpu-index", type=int, choices=AUTHORIZED_GPU_INDICES, default=3)
    result.add_argument("--stage", choices=("preflight", "formal"), default="preflight",
                        help="Stop for actual smoke review by default; formal needs an explicit separate invocation")
    result.add_argument("--preflight-dir", type=Path,
                        help="Required for formal: completed, unchanged parity and both smoke records")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())["status"]))

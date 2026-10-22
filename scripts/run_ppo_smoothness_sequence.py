"""Run the declared fresh PPO smoothing trials sequentially on physical GPU 1.

This persistent driver runs ON the authorized GPU host. It performs no SSH,
installation, automatic retry, GPU substitution or foreign-process control.
Every training/validation child requires a fresh fully-idle NVML-derived GPU
snapshot. That observation is not an exclusive reservation. Completed failures
are preserved; the first candidate passing both engineering screens stops for
visual review, not a safety or statistical qualification.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:1"
GPU_INDEX = 1
KIT_ARGS = ("--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false "
            "--/physics/cudaDevice=1 --/physics/sceneMultiGPUMode=0")
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}


class GpuBusyError(RuntimeError):
    pass


class SourceChangedError(RuntimeError):
    pass


class ChildFailedError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomic updates confined to this fresh run's generated status/results."""
    path = Path(path)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def case_name(mu):
    return f"mu{float(mu):.3f}".replace(".", "p")


def candidate_name(weight):
    return "action_rate_"+str(abs(float(weight))).replace(".", "p")


def _pci_key(value):
    domain, bus, slot = value.lower().split(":")
    device, function = slot.split(".")
    return tuple(int(part, 16) for part in (domain, bus, device, function))


def parse_gpu_one_snapshot(xml_text):
    """Parse all compute AND graphics processes, not only CUDA applications."""
    devices = ET.fromstring(xml_text).findall("gpu")
    if len(devices) <= GPU_INDEX:
        raise RuntimeError("Physical NVML GPU index 1 unavailable")
    selected = devices[GPU_INDEX]
    uuid = selected.findtext("uuid")
    pci = selected.findtext("pci/pci_bus_id")
    if not uuid or not pci:
        raise RuntimeError("Missing GPU UUID/PCI identity; fail closed")
    ordered = sorted(devices, key=lambda device: _pci_key(device.findtext("pci/pci_bus_id")))
    if ordered[GPU_INDEX].findtext("uuid") != uuid:
        raise RuntimeError("Physical GPU 1 is not CUDA ordinal 1 with PCI_BUS_ID ordering")
    try:
        memory = int(selected.findtext("fb_memory_usage/used").split()[0])
        utilization = int(selected.findtext("utilization/gpu_util").split()[0])
    except (TypeError, AttributeError, ValueError) as error:
        raise RuntimeError("Unavailable GPU memory/utilization; fail closed") from error
    if memory < 0 or not 0 <= utilization <= 100:
        raise RuntimeError("Invalid GPU memory/utilization")
    process_node = selected.find("processes")
    if process_node is None or (process_node.text or "").strip().upper() in ("N/A", "NOT SUPPORTED"):
        raise RuntimeError("Unavailable GPU process enumeration; fail closed")
    processes = [{child.tag: child.text for child in process}
                 for process in process_node.findall("process_info")]
    return {"physical_index": GPU_INDEX, "uuid": uuid, "pci_bus_id": pci,
            "memory_used_mib": memory, "utilization_percent": utilization,
            "all_reported_compute_and_graphics_processes": processes,
            "idle": not processes and memory < 64 and utilization == 0,
            "observed_unix_time": time.time(), "reservation_claimed": False,
            "source": "read-only nvidia-smi -q -x (NVML-derived snapshot)"}


def query_gpu_one():
    completed = subprocess.run(["nvidia-smi", "-q", "-x"], text=True,
                               capture_output=True, timeout=20, check=True)
    return parse_gpu_one_snapshot(completed.stdout)


def _source_files(args):
    files = set()
    for base in (args.snapshot, args.source_root, args.validation_source_root):
        for name in ("scripts", "training", "config"):
            directory = base/name
            if directory.is_dir():
                for path in directory.rglob("*"):
                    if (IGNORED_PARTS.intersection(path.relative_to(directory).parts)
                            or path.suffix not in (".py", ".json", ".yaml", ".yml")):
                        continue
                    if path.is_symlink():
                        raise ValueError("Source snapshot contains symlink: "+str(path))
                    if path.is_file():
                        files.add(path)
        for name in ("wide_friction_source.json", "controller_evaluation_source.json"):
            if (base/name).is_file():
                files.add(base/name)
    for path in args.beyondmimic_source.rglob("*.py"):
        if not IGNORED_PARTS.intersection(path.relative_to(args.beyondmimic_source).parts):
            files.add(path)
    files.update((args.protocol, args.task_profile, args.motion_file, args.reference_file,
                  args.asset_path, args.mujoco_model))
    for mu in args.protocol_data["validation_mu_in_order"]:
        files.update((args.baseline_root/case_name(mu)/"isaac"/"physics_trajectory.npz",
                      args.baseline_root/case_name(mu)/"isaac"/"report.json"))
    return sorted(files)


def snapshot_hashes(args):
    files = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in _source_files(args)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"tree_sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def verify_snapshot(args, frozen):
    current = snapshot_hashes(args)
    if current != frozen:
        before, after = frozen["files"], current["files"]
        changed = [name for name in sorted(set(before)|set(after)) if before.get(name) != after.get(name)]
        raise SourceChangedError("Frozen source/input files changed: "+", ".join(changed[:12]))


def run_child(command, log_path, cwd, environment, on_spawn):
    with Path(log_path).open("xb") as stream:
        child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, shell=False)
        try:
            on_spawn(child.pid)
            return child.wait()
        except BaseException:
            # Only the exact child created here. Never discover/stop GPU users.
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            raise


def training_command(args, weight, output, *, smoke):
    command = [str(args.python), "-u", str(args.snapshot/"scripts/train_ppo_smoothness.py"),
               "--source-root", str(args.source_root), "--motion-file", str(args.motion_file),
               "--asset-path", str(args.asset_path), "--task-profile", str(args.task_profile),
               "--action-rate-weight", str(weight), "--output-dir", str(output),
               "--num-envs", "2" if smoke else str(args.protocol_data["num_envs"]),
               "--iterations", "1" if smoke else str(args.protocol_data["iterations"]),
               "--seed", str(args.protocol_data["seed"]), "--smoke-steps", "0",
               "--ground-friction", "1.0", "--save-interval", "100",
               "--device", DEVICE, "--headless", "--kit_args", KIT_ARGS]
    if smoke:
        command.append("--smoke-only")
    return command


@contextmanager
def _temporary_environment(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def run_validation(args, training_output, validation_output, launch_checked):
    """Reuse the original driver while guarding EVERY actual spawned child."""
    import run_car_validation
    namespace = argparse.Namespace(
        source_root=args.validation_source_root, motion_file=args.motion_file,
        reference_file=args.reference_file, reference_id=args.protocol_data["reference_id"],
        asset_path=args.asset_path, mujoco_model=args.mujoco_model,
        checkpoint=training_output/"model_final.pt", runner_config=training_output/"runner_config.json",
        task_profile=args.task_profile, output_dir=validation_output,
        mu=args.protocol_data["validation_mu_in_order"], device=DEVICE,
        full_horizon_diagnostic=True)
    original = run_car_validation.launch

    def launch(command, cwd):
        command = list(command)
        command[0] = str(args.python)
        if any(str(part).endswith("eval_tracking_friction_isaac.py") for part in command):
            command.extend(["--kit_args", KIT_ARGS])
        launch_checked(command, Path(cwd), "validation")

    run_car_validation.launch = launch
    try:
        with _temporary_environment("TRON1_BEYONDMIMIC_ROOT", str(args.beyondmimic_source)):
            return run_car_validation.run(namespace)
    finally:
        run_car_validation.launch = original


def _resolve(args):
    for name in ("snapshot", "source_root", "validation_source_root", "beyondmimic_source",
                 "motion_file", "reference_file", "asset_path", "mujoco_model", "task_profile",
                 "protocol", "baseline_root", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()  # Preserve a venv symlink.
    if args.snapshot != ROOT.resolve():
        raise ValueError("Execute this driver from the declared frozen --snapshot tree")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory: "+str(args.output_dir))
    for base in (args.snapshot, args.source_root, args.validation_source_root, args.beyondmimic_source):
        if not base.is_dir():
            raise FileNotFoundError(base)
        if args.output_dir.is_relative_to(base):
            raise ValueError("Outputs must be outside all frozen source trees")
    args.protocol_data = json.loads(args.protocol.read_text())
    protocol = args.protocol_data
    required = {"backend": "physx", "reference_id": "drift_drl_traj4", "seed": 42,
                "num_envs": 512, "iterations": 1000,
                "steps_per_environment_per_iteration": 24,
                "environment_transitions_per_candidate": 12288000,
                "setup_smoke_control_steps": 0, "from_scratch_per_candidate": True,
                "candidate_action_rate_weights_in_order": [-.3, -.5, -1.],
                "validation_mu_in_order": [.7, .15]}
    for name, value in required.items():
        if protocol.get(name) != value:
            raise ValueError("Unsupported declared protocol field: "+name)
    screen = protocol["screening"]
    if (screen["required_reduction_of_high_frequency_base_world_xy_angular_velocity_rms"]
            != screen["required_reduction_of_leg_action_step_rms"]):
        raise ValueError("The comparison API requires matching declared HF/action reduction thresholds")
    if sha256(args.task_profile) != protocol["baseline_task_profile_sha256"]:
        raise ValueError("Baseline task profile differs from the frozen protocol")
    for relative in ("scripts/train_ppo_smoothness.py", "scripts/train_beyondmimic_style.py",
                     "scripts/run_car_validation.py", "scripts/analyze_ppo_smoothness.py"):
        if not (args.snapshot/relative).is_file():
            raise FileNotFoundError(args.snapshot/relative)
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise ValueError("Explicit Python executable unavailable")
    for mu in protocol["validation_mu_in_order"]:
        for filename in ("physics_trajectory.npz", "report.json"):
            path = args.baseline_root/case_name(mu)/"isaac"/filename
            if not path.is_file():
                raise FileNotFoundError(path)


def verify_training(output, weight, protocol, *, smoke):
    """Completed status, exact collection budget, weight and artifact identity."""
    output = Path(output)
    try:
        status = json.loads((output/"run_status.json").read_text())
        manifest = json.loads((output/"manifest.json").read_text())
        ablation = json.loads((output/"ablation_manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ChildFailedError("Missing readable training/ablation completion evidence") from error
    expected_status = "smoke_passed" if smoke else "training_completed"
    num_envs, iterations = (2, 1) if smoke else (protocol["num_envs"], protocol["iterations"])
    transitions = num_envs*24*iterations
    if (status.get("status") != expected_status
            or status.get("environment_transitions") != transitions
            or manifest.get("num_envs") != num_envs or manifest.get("iterations") != iterations
            or manifest.get("environment_transitions_requested") != transitions
            or manifest.get("setup_smoke_control_steps") != 0 or manifest.get("seed") != 42
            or manifest.get("smoke_only") is not smoke):
        raise ChildFailedError("Training status or exact declared collection budget differs")
    actual_weight = ablation.get("reward_override", {}).get("weight")
    if actual_weight != weight or ("action_rate_weight" in ablation and ablation["action_rate_weight"] != weight):
        raise ChildFailedError("Completed ablation weight differs from dispatched weight")
    override = ablation.get("reward_override", {})
    if (ablation.get("status") != expected_status
            or ablation.get("initialization") != "from_scratch_no_checkpoint_resume"
            or ablation.get("smoke_only") is not smoke
            or ablation.get("trained_policy_claimed") is not (not smoke)
            or ablation.get("expected_environment_transitions") != transitions
            or ablation.get("actual_environment_transitions") != transitions
            or override.get("term") != "action_rate_l2" or override.get("baseline_weight") != -.1
            or override.get("applied_before_environment_creation") is not True
            or manifest.get("training_reward_overrides") != {"action_rate_l2": weight}
            or ablation.get("final_training_manifest_sha256") != sha256(output/"manifest.json")):
        raise ChildFailedError("Ablation lacks completed from-scratch exact-weight/budget/source evidence")
    if manifest.get("task_profile_sha256") != protocol["baseline_task_profile_sha256"]:
        raise ChildFailedError("Locked task profile bytes differ from declared baseline")
    checkpoint = output/("model_smoke.pt" if smoke else "model_final.pt")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ChildFailedError("Completed checkpoint missing or empty")
    checkpoint_hash = sha256(checkpoint)
    if ablation.get("checkpoint_sha256") != checkpoint_hash:
        raise ChildFailedError("Completed checkpoint hash differs from ablation manifest")
    if not (output/"runner_config.json").is_file():
        raise ChildFailedError("Completed PPO runner configuration missing")
    return {"status": expected_status, "environment_transitions": transitions,
            "action_rate_weight": weight, "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": sha256(output/"manifest.json"),
            "run_status_sha256": sha256(output/"run_status.json"),
            "ablation_manifest_sha256": sha256(output/"ablation_manifest.json")}


def finalize_owned_training(args, output, weight, *, smoke, child_record):
    """Finalize only a successful owned child whose Isaac shutdown skipped return.

    ``app.close(skip_cleanup=True)`` can exit the child before its outer Python
    wrapper returns. The completed trainer status/checkpoint remain mandatory;
    this only performs the wrapper's metadata finalization in the parent. No
    failed, partial or old unrelated run is repaired or promoted to complete.
    """
    output = Path(output)
    if (child_record.get("returncode") != 0 or child_record.get("phase") not in ("smoke", "training")
            or not output.is_relative_to(args.output_dir)
            or str(output) not in child_record.get("command", [])):
        raise ChildFailedError("Parent finalization requires this successful owned training child")
    path = output/"ablation_manifest.json"
    ablation = json.loads(path.read_text())
    if ablation.get("status") != "configured_before_environment_creation":
        return {"performed": False, "reason": "Child already finalized or completion verification will fail"}
    status = json.loads((output/"run_status.json").read_text())
    manifest = json.loads((output/"manifest.json").read_text())
    num_envs, iterations = (2, 1) if smoke else (args.protocol_data["num_envs"], args.protocol_data["iterations"])
    transitions = num_envs*24*iterations
    expected_status = "smoke_passed" if smoke else "training_completed"
    if (status.get("status") != expected_status
            or status.get("environment_transitions") != transitions
            or status.get("trained_policy_claimed") is not (not smoke)
            or manifest.get("num_envs") != num_envs or manifest.get("iterations") != iterations
            or manifest.get("environment_transitions_requested") != transitions
            or manifest.get("setup_smoke_control_steps") != 0
            or manifest.get("smoke_only") is not smoke or manifest.get("seed") != 42):
        raise ChildFailedError("Parent metadata finalization refuses incomplete or mismatched training")
    override = ablation.get("reward_override", {})
    required = {"kind": "historical_physx_ppo_action_rate_ablation", "algorithm": "PPO",
                "training_engine": "PhysX", "initialization": "from_scratch_no_checkpoint_resume",
                "smoke_only": smoke, "trained_policy_claimed": False, "seed": 42,
                "num_envs": num_envs, "iterations": iterations,
                "expected_environment_transitions": transitions,
                "historical_trainer_sha256": sha256(args.snapshot/"scripts/train_beyondmimic_style.py"),
                "height_adapter_sha256": sha256(args.snapshot/"scripts/height_smoothness.py"),
                "task_profile_sha256": sha256(args.task_profile),
                "motion_sha256": sha256(args.motion_file), "asset_sha256": sha256(args.asset_path),
                "wrapper_sha256": sha256(args.snapshot/"scripts/train_ppo_smoothness.py"),
                "DR_recipe_modified": False, "base_task_profile_file_modified": False,
                "baseline_checkpoint_modified": False}
    if any(ablation.get(name) != value for name, value in required.items()):
        raise ChildFailedError("Pre-environment ablation declaration differs from the owned frozen experiment")
    if (override.get("term") != "action_rate_l2" or override.get("baseline_weight") != -.1
            or override.get("weight") != weight
            or override.get("function") != "isaaclab.envs.mdp.rewards.action_rate_l2"
            or override.get("divided_by_control_dt") is not False
            or override.get("applied_before_environment_creation") is not True):
        raise ChildFailedError("Pre-environment action-rate weight/function declaration differs")
    source_names = ("scripts/train_tracking.py", "training/tron1_tracking.py", "training/tron1_drift.py",
                    "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py")
    expected_sources = {name: sha256(args.source_root/name) for name in source_names}
    if (ablation.get("baseline_source_sha256") != expected_sources
            or manifest.get("source_training_modules_sha256") != expected_sources):
        raise ChildFailedError("Pre-environment ablation/trainer source declarations do not match frozen sources")
    checkpoint = output/("model_smoke.pt" if smoke else "model_final.pt")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ChildFailedError("Parent metadata finalization requires a nonempty completed checkpoint")
    before = {"ablation": sha256(path), "manifest": sha256(output/"manifest.json")}
    from train_ppo_smoothness import finalize_manifest
    finalize_manifest({"output_dir": output, "smoke_only": smoke, "action_rate_weight": weight})
    return {"performed": True, "metadata_finalize_location": "parent_after_isaac_exit",
            "reason": "Successful owned Isaac child exited during app.close(skip_cleanup=True), before outer wrapper return; completed trainer status/checkpoint independently verified",
            "before_sha256": before,
            "after_sha256": {"ablation": sha256(path), "manifest": sha256(output/"manifest.json")},
            "run_status_sha256": sha256(output/"run_status.json")}


def verify_evaluation(directory, mu, checkpoint_hash, motion_hash, *, require_gpu_one=False):
    """Bind measured Isaac bytes to the matching frozen policy and fixed-mu run."""
    directory = Path(directory)
    report = json.loads((directory/"report.json").read_text())
    episodes = report.get("summary", {}).get("episodes", [])
    if (report.get("checkpoint_sha256") != checkpoint_hash
            or report.get("motion_file_sha256") != motion_hash
            or report.get("physics_trajectory_sha256") != sha256(directory/"physics_trajectory.npz")
            or report.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") != mu
            or report.get("domain_randomization_enabled") is not False
            or report.get("full_horizon_diagnostic") is not True
            or len(episodes) != 1 or episodes[0].get("completed_full_reference") is not True
            or episodes[0].get("start_reference_frame") != 0):
        raise ChildFailedError("Isaac evaluation lacks matching checkpoint/motion/mu/full-reference evidence")
    if require_gpu_one:
        audit = report.get("runtime_device_audit", {})
        if (audit.get("status") != "verified" or audit.get("requested_device") != DEVICE
                or audit.get("cuda_visible_devices") is not None
                or audit.get("native_and_tensor_device_consistency_verified") is not True):
            raise ChildFailedError("New Isaac evaluation lacks verified unmasked GPU 1 device consistency")
    return report


def run_sequence(args, *, gpu_probe=query_gpu_one, child_runner=run_child,
                 validation_runner=run_validation, trajectory_analyzer=None,
                 candidate_comparator=None, quiesce=time.sleep):
    _resolve(args)
    if trajectory_analyzer is None or candidate_comparator is None:
        from analyze_ppo_smoothness import analyze_trajectory, compare_candidate
        trajectory_analyzer = trajectory_analyzer or analyze_trajectory
        candidate_comparator = candidate_comparator or compare_candidate
    frozen = snapshot_hashes(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir/"sequence_status.json"
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
                       TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    status = {"status": "initializing", "pid": os.getpid(), "host": platform.node(),
              "started_unix_time": time.time(), "protocol": args.protocol_data,
              "protocol_sha256": sha256(args.protocol), "source_tree_sha256": frozen["tree_sha256"],
              "physical_gpu_index": GPU_INDEX, "device": DEVICE,
              "cuda_visible_devices": None, "cuda_device_order": "PCI_BUS_ID",
              "exclusive_gpu_reservation_claimed": False, "automatic_retries": False,
              "children": [], "candidates": [], "jitter_resolved_claimed": False,
              "safety_or_statistical_qualification_claimed": False}
    write_json(args.output_dir/"frozen_source_and_inputs.json", frozen)
    write_json(status_path, status)

    def launch_checked(command, cwd, phase):
        verify_snapshot(args, frozen)
        evidence = gpu_probe()
        child = {"phase": phase, "command": list(command), "cwd": str(cwd),
                 "gpu_preflight": evidence, "status": "pending",
                 "source_tree_sha256": frozen["tree_sha256"]}
        status["children"].append(child)
        if not evidence.get("idle") or evidence.get("physical_index") != GPU_INDEX:
            child["status"] = "not_started_gpu_busy"
            write_json(status_path, status)
            raise GpuBusyError("Physical GPU 1 not freshly fully idle; no new child started")
        child.update(status="running", started_unix_time=time.time())
        status["status"] = "running_"+phase
        write_json(status_path, status)
        log_path = args.output_dir/f"child_{len(status['children']):03d}_{phase}.log"
        child["log_path"] = str(log_path)

        def on_spawn(pid):
            child["pid"] = int(pid)
            write_json(status_path, status)

        returncode = child_runner(command, log_path, cwd, environment, on_spawn)
        child.update(returncode=returncode, finished_unix_time=time.time(),
                     status="completed" if returncode == 0 else "failed")
        write_json(status_path, status)
        # A fixed grace period after OUR child exits permits its utilization
        # counter tail to settle. This is not polling/retry of a busy foreign job.
        child["post_owned_child_quiescence_s"] = 2.
        quiesce(2.)
        verify_snapshot(args, frozen)
        if returncode != 0:
            raise ChildFailedError(f"{phase} child exited {returncode}; no retry")

    try:
        baselines = {}
        motion_hash = sha256(args.motion_file)
        for mu in args.protocol_data["validation_mu_in_order"]:
            directory = args.baseline_root/case_name(mu)/"isaac"
            verify_evaluation(directory, mu, args.protocol_data["baseline_checkpoint_sha256"], motion_hash)
            baselines[mu] = trajectory_analyzer(
                directory/"physics_trajectory.npz", reference=args.motion_file,
                report=directory/"report.json",
                high_frequency_hz=args.protocol_data["spectrum_cutoff_hz"])
            if (baselines[mu].get("evaluation_mu") != mu
                    or baselines[mu].get("checkpoint_sha256") != args.protocol_data["baseline_checkpoint_sha256"]
                    or baselines[mu].get("reference_sha256") != motion_hash):
                raise ChildFailedError("Baseline analysis identity differs from the frozen matched-mu protocol")
        write_json(args.output_dir/"baseline_smoothness.json", {str(mu): value for mu, value in baselines.items()})
        for weight in args.protocol_data["candidate_action_rate_weights_in_order"]:
            candidate_dir = args.output_dir/candidate_name(weight)
            candidate_dir.mkdir(exist_ok=False)
            candidate = {"action_rate_weight": weight, "status": "initializing", "comparisons": {}}
            status["candidates"].append(candidate)
            smoke_output = candidate_dir/"smoke"
            launch_checked(training_command(args, weight, smoke_output, smoke=True), args.snapshot, "smoke")
            status["children"][-1]["parent_metadata_finalization"] = finalize_owned_training(
                args, smoke_output, weight, smoke=True, child_record=status["children"][-1])
            write_json(status_path, status)
            candidate["smoke"] = verify_training(smoke_output, weight, args.protocol_data, smoke=True)
            output = candidate_dir/"training"
            launch_checked(training_command(args, weight, output, smoke=False), args.snapshot, "training")
            status["children"][-1]["parent_metadata_finalization"] = finalize_owned_training(
                args, output, weight, smoke=False, child_record=status["children"][-1])
            write_json(status_path, status)
            candidate["training"] = verify_training(output, weight, args.protocol_data, smoke=False)
            candidate["status"] = "validating"
            write_json(status_path, status)
            validation = candidate_dir/"validation"
            validation_runner(args, output, validation, launch_checked)
            for mu in args.protocol_data["validation_mu_in_order"]:
                directory = validation/case_name(mu)/"isaac"
                verify_evaluation(directory, mu, candidate["training"]["checkpoint_sha256"],
                                  motion_hash, require_gpu_one=True)
                measured = trajectory_analyzer(
                    directory/"physics_trajectory.npz", reference=args.motion_file,
                    report=directory/"report.json",
                    high_frequency_hz=args.protocol_data["spectrum_cutoff_hz"])
                if (measured.get("evaluation_mu") != mu
                        or measured.get("checkpoint_sha256") != candidate["training"]["checkpoint_sha256"]
                        or measured.get("reference_sha256") != motion_hash):
                    raise ChildFailedError("Candidate analysis identity differs from completed training/matched-mu protocol")
                screen = args.protocol_data["screening"]
                comparison = candidate_comparator(
                    measured, baselines[mu],
                    reduction_fraction=screen["required_reduction_of_high_frequency_base_world_xy_angular_velocity_rms"],
                    xy_max_rmse_m=screen["maximum_core_xy_rmse_m"],
                    xy_extra_rmse_m=screen["maximum_core_xy_rmse_increase_over_matched_mu_baseline_m"],
                    speed_retention_min=screen["minimum_reference_speed_retention"],
                    beta_extra_rmse_deg=screen["maximum_beta_rmse_increase_over_matched_mu_baseline_deg"])
                candidate["comparisons"][str(mu)] = comparison
                write_json(candidate_dir/(case_name(mu)+"_smoothness.json"),
                           {"candidate": measured, "baseline": baselines[mu], "comparison": comparison})
            passed = all(value.get("passes") is True for value in candidate["comparisons"].values())
            candidate["status"] = "passed_engineering_screen" if passed else "failed_engineering_screen"
            write_json(status_path, status)
            if passed:
                status.update(status="screen_passed_requires_visual_review", selected_action_rate_weight=weight,
                              finished_unix_time=time.time())
                write_json(status_path, status)
                return 0
        status.update(status="needs_next_method", finished_unix_time=time.time(),
                      explanation="All declared weights tested; no claim that jitter has been resolved")
        write_json(status_path, status)
        return 4
    except BaseException as error:
        status.update(status="stopped_gpu_busy" if isinstance(error, GpuBusyError) else "failed",
                      error=f"{type(error).__name__}: {error}", finished_unix_time=time.time())
        if status["children"] and status["children"][-1]["status"] == "running":
            status["children"][-1]["status"] = "failed_or_interrupted"
        write_json(status_path, status)
        return 3 if isinstance(error, GpuBusyError) else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "source-root", "validation-source-root", "beyondmimic-source",
                 "python", "motion-file", "reference-file", "asset-path", "mujoco-model",
                 "task-profile", "baseline-root", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"config/ppo_smoothness_sequence_v1.json")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return run_sequence(parse_args(argv))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Sequence not started: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

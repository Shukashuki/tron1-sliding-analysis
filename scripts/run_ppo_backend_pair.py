"""Run fresh mjlab and PhysX PPO training sequentially on physical GPU 0.

This dispatcher runs on the authorized GPU host: no SSH, uploads, package
installation, or other users' process manipulation.  A busy GPU stops the
dispatch; there is deliberately no wait/retry/tuning loop.  An idle readback is
a preflight observation, not an exclusive GPU reservation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


MJLAB_PIN = "4360096b49c1bcc478668c11ce7d458db747e49d"
BACKENDS = ("mjlab", "physx")
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Atomic updates only to this new dispatcher's own generated results."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def snapshot_hashes(snapshot):
    snapshot = Path(snapshot)
    files = {}
    for path in sorted(snapshot.rglob("*")):
        relative = path.relative_to(snapshot)
        if IGNORED_PARTS.intersection(relative.parts) or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Source snapshot must contain regular files, not symlinks: {relative}")
        if path.is_file():
            files[relative.as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    if "scripts/train_ppo_backend.py" not in files:
        raise ValueError("Snapshot lacks scripts/train_ppo_backend.py")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"tree_sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def verify_snapshot(snapshot, frozen):
    current = snapshot_hashes(snapshot)
    if current != frozen:
        before, after = frozen["files"], current["files"]
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        raise SourceChangedError("Frozen source snapshot changed: " + ", ".join(changed[:20]))


def _query_csv(fields, kind):
    completed = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=20, check=True)
    rows = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        if not row or not any(part.strip() for part in row):
            continue
        if len(row) != len(fields):
            raise RuntimeError(f"Unexpected nvidia-smi {kind} readback: {row!r}")
        rows.append(dict(zip(fields, (part.strip() for part in row))))
    return rows


def query_gpu_zero():
    devices = _query_csv(("index", "uuid", "pci.bus_id", "memory.used", "utilization.gpu"), "gpu")
    gpu = next((device for device in devices if device["index"] == "0"), None)
    if gpu is None:
        raise RuntimeError("Physical GPU index 0 is unavailable")
    # Unmasked CUDA uses PCI ordering, rather than an inherited ordinal mask.
    # Fail closed if that ordering would not select the requested NVML GPU 0.
    first_pci = min(devices, key=lambda device: device["pci.bus_id"].lower())
    if first_pci["uuid"] != gpu["uuid"]:
        raise RuntimeError("Physical GPU 0 is not CUDA ordinal 0 under PCI_BUS_ID ordering")
    processes = _query_csv(("gpu_uuid", "pid", "process_name", "used_memory"), "compute-apps")
    active = [process for process in processes if process["gpu_uuid"] == gpu["uuid"]]
    try:
        memory = int(gpu["memory.used"])
        utilization = int(gpu["utilization.gpu"])
    except ValueError as error:
        raise RuntimeError("GPU utilization/memory readback is unavailable; refusing dispatch") from error
    if memory < 0 or not 0 <= utilization <= 100:
        raise RuntimeError("Invalid GPU utilization/memory readback")
    evidence = {"physical_index": 0, "uuid": gpu["uuid"], "pci_bus_id": gpu["pci.bus_id"],
                "memory_used_mib": memory, "utilization_percent": utilization,
                "compute_processes": active, "observed_unix_time": time.time(),
                "reservation_claimed": False}
    evidence["idle"] = not active and memory < 96 and utilization <= 1
    return evidence


def child_command(args, backend, output):
    return [str(args.python), "-u", str(args.snapshot / "scripts/train_ppo_backend.py"),
            "--backend", backend, "--model", str(args.model), "--motion", str(args.motion),
            "--contract", str(args.snapshot / "config/nominal_actuator_contract.json"),
            "--task-profile", str(args.snapshot / "config/car_drift_height_smooth_task.json"),
            "--mjlab-source", str(args.mjlab_source), "--tron1-source", str(args.tron1_source),
            "--beyondmimic-source", str(args.beyondmimic_source), "--asset-path", str(args.asset_path),
            "--output-dir", str(output), "--num-envs", str(args.num_envs),
            "--iterations", str(args.iterations), "--save-interval", str(args.save_interval),
            "--smoke-steps", str(args.smoke_steps), "--seed", str(args.seed), "--device", "cuda:0"]


def run_child(command, log_path, cwd, environment):
    with Path(log_path).open("xb") as stream:
        child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, shell=False)
        try:
            return child.wait()
        except BaseException:
            # Only this exact owned child, never a GPU-discovered process.
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            raise


def validate_child(output, backend, transitions, num_envs, iterations):
    output = Path(output)
    try:
        status = json.loads((output / "run_status.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ChildFailedError(f"{backend} lacks readable completion status/manifest") from error
    if status.get("status") != "training_completed":
        raise ChildFailedError(f"{backend} did not report training_completed: {status.get('status')!r}")
    count = status.get("training_environment_steps")
    if type(count) is not int or count != transitions:
        raise ChildFailedError(f"{backend} transition count {count!r} != {transitions}")
    for key, expected in (("backend", backend), ("num_envs", num_envs),
                          ("iterations", iterations), ("transitions_requested", transitions)):
        if manifest.get(key) != expected:
            raise ChildFailedError(f"{backend} manifest {key} differs from the dispatched experiment")
    initial = manifest.get("initial_policy_state_sha256", "")
    if not isinstance(initial, str) or len(initial) != 64 or any(char not in "0123456789abcdef" for char in initial):
        raise ChildFailedError(f"{backend} lacks an initial policy-state SHA256")
    for filename, key in (("model_final.pt", "checkpoint_sha256"), ("policy.pt", "exported_policy_sha256")):
        artifact = output / filename
        if not artifact.is_file() or status.get(key) != sha256(artifact):
            raise ChildFailedError(f"{backend} missing or mismatched completed artifact: {filename}")
    return {"backend": backend, "training_environment_steps": count,
            "initial_policy_state_sha256": initial,
            "manifest_sha256": sha256(output / "manifest.json"),
            "run_status_sha256": sha256(output / "run_status.json"),
            "checkpoint_sha256": status["checkpoint_sha256"],
            "exported_policy_sha256": status["exported_policy_sha256"]}


def _resolve_and_validate(args):
    for name in ("snapshot", "workspace", "mjlab_source", "model", "motion",
                 "asset_path", "tron1_source", "beyondmimic_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # A venv's bin/python may symlink to the base interpreter. Resolving that
    # symlink would bypass pyvenv.cfg and silently select a different runtime.
    args.python = args.python.expanduser().absolute()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       args.workspace / f"ppo_backend_pair_seed{args.seed}_v1")
    if not args.output_dir.is_relative_to(args.workspace):
        raise ValueError("Pair output must be inside the explicit workspace")
    if args.output_dir.is_relative_to(args.snapshot):
        raise ValueError("Pair output must not be inside the frozen source snapshot")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh pair output directory: {args.output_dir}")
    for name in ("snapshot", "mjlab_source", "tron1_source", "beyondmimic_source"):
        if not getattr(args, name).is_dir():
            raise FileNotFoundError(f"Missing {name} directory: {getattr(args, name)}")
    for name in ("python", "model", "motion", "asset_path"):
        if not getattr(args, name).is_file():
            raise FileNotFoundError(f"Missing {name} file: {getattr(args, name)}")
    if not os.access(args.python, os.X_OK):
        raise ValueError("The selected Python executable is not executable")
    for relative in ("scripts/train_ppo_backend.py", "config/nominal_actuator_contract.json",
                     "config/car_drift_height_smooth_task.json"):
        if not (args.snapshot / relative).is_file():
            raise FileNotFoundError(f"Missing snapshot input: {relative}")


def run_pair(args, *, gpu_probe=query_gpu_zero, child_runner=run_child):
    _resolve_and_validate(args)
    frozen = snapshot_hashes(args.snapshot)
    commit = subprocess.check_output(
        ["git", "-C", str(args.mjlab_source), "rev-parse", "HEAD"], text=True).strip()
    if commit != MJLAB_PIN:
        raise ValueError(f"mjlab checkout {commit} != required {MJLAB_PIN}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    requested = args.num_envs * 24 * args.iterations
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["PYTHONUNBUFFERED"] = "1"
    report = {"status": "initializing", "host": platform.node(), "pid": os.getpid(),
              "source_snapshot": str(args.snapshot), "source_tree_sha256": frozen["tree_sha256"],
              "mjlab_commit": commit, "physical_gpu_index": 0,
              "cuda_visible_devices": None, "cuda_device_order": "PCI_BUS_ID",
              "backend_order": list(BACKENDS), "transitions_per_backend": requested,
              "num_envs": args.num_envs, "iterations": args.iterations, "seed": args.seed,
              "automatic_retries": False, "parameter_tuning": False,
              "exclusive_gpu_reservation_claimed": False, "children": []}
    write_json(args.output_dir / "source_snapshot_hashes.json", frozen)
    write_json(args.output_dir / "pair_status.json", report)
    try:
        for backend in BACKENDS:
            verify_snapshot(args.snapshot, frozen)
            evidence = gpu_probe()
            child_record = {"backend": backend, "gpu_preflight": evidence,
                            "output_dir": str(args.output_dir / backend), "status": "pending"}
            report["children"].append(child_record)
            if not evidence.get("idle"):
                child_record["status"] = "not_started_gpu_busy"
                raise GpuBusyError(f"GPU 0 is not fully idle before {backend}; no new child was started")
            command = child_command(args, backend, args.output_dir / backend)
            child_record.update(status="running", command=command, started_unix_time=time.time())
            report["status"] = "running_" + backend
            write_json(args.output_dir / "pair_status.json", report)
            returncode = child_runner(command, args.output_dir / f"{backend}.stdout.log",
                                      args.snapshot, environment)
            child_record.update(returncode=returncode, finished_unix_time=time.time())
            verify_snapshot(args.snapshot, frozen)
            if returncode != 0:
                raise ChildFailedError(f"{backend} child exited {returncode}; no retry or next controller")
            child_record.update(validate_child(args.output_dir / backend, backend, requested,
                                               args.num_envs, args.iterations))
            child_record["status"] = "training_completed_verified"
            write_json(args.output_dir / "pair_status.json", report)
        initial = [child["initial_policy_state_sha256"] for child in report["children"]]
        if len(set(initial)) != 1:
            raise ChildFailedError("Completed backends have different initial policy states; not a matched initialization")
        report.update(status="training_completed", matched_initialization_verified=True,
                      initial_policy_state_sha256=initial[0], finished_unix_time=time.time(),
                      evaluation_required="No controller-quality claim until separate matched-engine/friction validation")
        return 0
    except BaseException as error:
        report.update(status="stopped_gpu_busy" if isinstance(error, GpuBusyError) else "failed",
                      error=f"{type(error).__name__}: {error}", finished_unix_time=time.time())
        if report["children"] and report["children"][-1]["status"] == "running":
            report["children"][-1]["status"] = "failed_completion_verification"
        return 3 if isinstance(error, GpuBusyError) else 1
    finally:
        write_json(args.output_dir / "pair_status.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "workspace", "python", "mjlab-source", "model", "motion",
                 "asset-path", "tron1-source", "beyondmimic-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--smoke-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.smoke_steps < 0:
        parser.error("Positive environment/iteration counts and nonnegative smoke steps are required")
    return args


def main(argv=None):
    try:
        return run_pair(parse_args(argv))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Dispatcher not started: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

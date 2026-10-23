"""Wait for the exact authorized formal PPO, then two nominal evaluations.

No training, retries, process signals, GPU reservations or friction sweep. A
busy GPU is observed, never displaced. Existing evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


DEPENDENCIES = {
    "eval_ppo_ground_contact_v2d.py": "1a4968d4840ffbc672bac5ad630786a6141d4d494f0af2d5ad783adac33abd81",
    "ground_contact_reward_v2.py": "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5",
    "ground_contact_reward_v2b.py": "359997a5e4e1734f14ffd8c64a61bbaf0f8fb1d0dd6b274001a77689bb62ba6c",
    "ground_contact_reward_v2c.py": "758eba1c7234cd43b211262541fce0cab6ad91be8ae0f10a44714d037322ebfb",
    "ground_contact_reward_v2d.py": "1f6245ca3ca08483704d4f42a044ba38b8d09f01d81dc6e71bbf423d206898a7",
    "ppo_backend_400hz_contract.py": "b57b7444698e06f4d9faed6f246bb8a301b505f6c785bdbc2caafdb0b2b6f56e",
    "eval_ppo_backend_400hz_mujoco.py": "9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4",
    "eval_ppo_backend_400hz_isaac.py": "e670d6c88bd85eb8aa36f402301e7f0c9ae5f7d3b97336a78d1a66750a642f99",
    "train_ppo_backend.py": "65367115de9b5608dc2ef57cab701bea31c152c6af576bdb43a5cef025cc6878",
    "physx_mjlab_recipe.py": "1365f4eecc6129bef4738e05a491763a2884407d8cf00a3a7d0faffbc24fa699",
}
GPU_ORDER = (3, 5, 4)
PROTOCOL_SHA256 = "e89a5a3b2336da37097cf7dad163d560b05f38ae15ad6578742b9d567940804a"
COHORT_ID = "tron1_ground_contact_v2d_any_wheel_400hz_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def option(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "Missing/ambiguous launch argument: " + name)
    return argv[argv.index(name) + 1]


def process_snapshot(pid):
    """Read only procfs; never signal or control the original training job."""
    proc = Path("/proc") / str(pid)
    try:
        owner = proc.stat().st_uid
        stat = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        command = [word.decode() for word in (proc / "cmdline").read_bytes().split(b"\0") if word]
    except FileNotFoundError:
        return None
    return {"pid": pid, "uid": owner, "state": stat[0], "start_ticks": int(stat[19]), "argv": command}


def pci_key(value):
    domain, bus, slot = value.lower().split(":")
    device, function = slot.split(".")
    return tuple(int(part, 16) for part in (domain, bus, device, function))


def parse_gpu_snapshot(xml):
    devices = ET.fromstring(xml).findall("gpu")
    identities = [(gpu.findtext("uuid"), gpu.findtext("pci/pci_bus_id")) for gpu in devices]
    require(len(devices) > max(GPU_ORDER) and all(uuid and pci for uuid, pci in identities)
            and len({uuid for uuid, _ in identities}) == len(devices)
            and len({pci_key(pci) for _, pci in identities}) == len(devices), "Unavailable/ambiguous full GPU identities")
    ordered = sorted(range(len(devices)), key=lambda i: pci_key(identities[i][1]))
    result = []
    for index in GPU_ORDER:
        gpu = devices[index]
        require(ordered[index] == index, "Unmasked PCI_BUS_ID CUDA ordinal differs from selected physical GPU")
        node = gpu.find("processes")
        require(node is not None and (node.text or "").strip().upper() not in ("N/A", "NOT SUPPORTED"),
                "Full compute+graphics process enumeration is unavailable")
        processes = [{child.tag: child.text for child in row} for row in node.findall("process_info")]
        try:
            memory = int(gpu.findtext("fb_memory_usage/used").split()[0])
            utilization = int(gpu.findtext("utilization/gpu_util").split()[0])
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("GPU memory/utilization unavailable") from error
        require(memory >= 0 and 0 <= utilization <= 100, "Invalid GPU telemetry")
        result.append({"physical_index": index, "uuid": identities[index][0], "pci_bus_id": identities[index][1],
            "memory_used_mib": memory, "utilization_percent": utilization,
            "all_reported_compute_and_graphics_processes": processes,
            "idle": not processes and memory < 64 and utilization == 0,
            "pci_bus_id_cuda_ordinal_verified": True, "observed_unix_time": time.time(),
            "reservation_claimed": False, "source": "nvidia-smi -q -x full NVML-derived snapshot"})
    return result


def query_gpus():
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True, timeout=20)
    return parse_gpu_snapshot(result.stdout)


def load_evaluator(dependencies):
    for name, digest in DEPENDENCIES.items():
        require(sha256(dependencies / "scripts" / name) == digest, "Pinned evaluation dependency differs: " + name)
    directory = dependencies / "scripts"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("_contact_v2_continuation_evaluator", directory / "eval_ppo_ground_contact_v2d.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_snapshot(args):
    paths = {Path(__file__).resolve(), args.launch_record, args.protocol, args.legacy_native, args.legacy_physx,
             args.motion, args.model, args.asset}
    paths.update(args.evaluation_dependencies / "scripts" / name for name in DEPENDENCIES)
    for root in {args.legacy_native.parent.parent, args.legacy_physx.parent.parent, args.beyondmimic_source}:
        paths.update(path for path in root.rglob("*.py") if "__pycache__" not in path.parts and path.is_file())
    require(all(not path.is_symlink() for path in paths), "Indirect source/input is not allowed")
    return {str(path): sha256(path) for path in sorted(paths)}


def child_command(args, backend, output, index=None):
    command = [str(args.python), "-u", str(args.evaluation_dependencies / "scripts/eval_ppo_ground_contact_v2d.py"),
        "--training-dir", str(args.training_dir), "--protocol", str(args.protocol),
        "--legacy-evaluator", str(args.legacy_native if backend == "mujoco" else args.legacy_physx),
        "--motion-file", str(args.motion), "--output-dir", str(output), "--mu", "0.7", "--backend", backend]
    command += ["--model", str(args.model)] if backend == "mujoco" else ["--asset-path", str(args.asset), "--physical-gpu-index", str(index)]
    return command


def run_child(command, log, cwd, environment, on_spawn, *, deadline=None):
    with Path(log).open("xb") as stream:
        child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, shell=False)
        on_spawn(child.pid)
        # This is our evaluation child only. Never send signals, including on
        # interruption; its PID/log remain available for explicit user review.
        while child.poll() is None:
            require(deadline is None or time.monotonic() < deadline,
                    "Owned evaluation exceeded deadline; its process was preserved, not killed")
            time.sleep(1.)
        return child.returncode


def run(args, *, process_probe=process_snapshot, gpu_probe=query_gpus, evaluator=None,
        child_runner=run_child, sleep=time.sleep, monotonic=time.monotonic):
    for name in ("workspace", "evaluation_dependencies", "protocol", "legacy_native", "legacy_physx", "motion",
                 "model", "asset", "beyondmimic_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    require(0 < args.poll_seconds <= 30 and 0 < args.deadline_seconds <= 7200, "Polling/deadline exceeds declared bounds")
    args.launch_record = (getattr(args, "launch_record", None) or args.workspace / "physx_formal_v1_launch.json").resolve()
    launch = read_json(args.launch_record)
    require(type(args.expected_training_pid) is int and args.expected_training_pid > 0
            and type(launch.get("pid")) is int
            and launch["pid"] == args.expected_training_pid
            and launch["run_kind"] == "formal_from_scratch_12288000_transitions", "Not the exact authorized owned formal launch")
    require(sha256(args.protocol) == PROTOCOL_SHA256
            and read_json(args.protocol).get("cohort_id") == COHORT_ID,
            "Pinned V2d protocol/cohort differs")
    argv = launch["argv"]
    require(isinstance(argv, list) and all(isinstance(word, str) for word in argv)
            and len(argv) > 1 and Path(argv[0]).absolute() == args.python, "Formal launch command differs")
    script_index = 2 if argv[1] == "-u" else 1
    require(len(argv) > script_index
            and Path(argv[script_index]).name == "train_ppo_ground_contact_v2d.py"
            and Path(argv[script_index]).parent.name == "scripts", "Formal launch trainer differs")
    args.training_dir = Path(option(argv, "--output-dir")).resolve()
    require(args.training_dir == args.workspace / "physx_formal_v1"
            and Path(option(argv, "--cohort-protocol")).resolve() == args.protocol
            and option(argv, "--cohort-run-kind") == "formal", "Launch training output/protocol/mode differs")
    for flag, path in (("--motion", args.motion), ("--model", args.model), ("--asset-path", args.asset),
                       ("--beyondmimic-source", args.beyondmimic_source)):
        require(Path(option(argv, flag)).resolve() == path, "Requested input differs from owned training launch: " + flag)
    state_dir = args.workspace / "nominal_validation_continuation_v1"
    outputs = {"mujoco": args.workspace / "mujoco_nominal_eval_v1", "physx": args.workspace / "physx_nominal_eval_v1"}
    require(not state_dir.exists() and all(not path.exists() and not path.with_name(path.name + "_inputs").exists()
            for path in outputs.values()), "Preserve existing continuation/evaluation evidence; no retry/overwrite")
    evaluator = evaluator or load_evaluator(args.evaluation_dependencies)
    frozen = source_snapshot(args)
    state_dir.mkdir(parents=True, exist_ok=False)
    status_path = state_dir / "continuation_status.json"
    status = {"kind": "owned_ground_contact_v2d_formal_nominal_validation_continuation", "pid": os.getpid(),
        "status": "initializing", "started_unix_time": time.time(), "training_pid": launch["pid"],
        "training_launch_record_sha256": sha256(args.launch_record), "automatic_retry": False,
        "process_signals_sent": False, "future_gpu_reservation_claimed": False, "mu": .7, "children": []}
    write_json(state_dir / "frozen_source_and_inputs.json", frozen)
    write_json(status_path, status)
    deadline, pinned_process = monotonic() + args.deadline_seconds, None

    def check():
        require(source_snapshot(args) == frozen, "Frozen evaluation source/protocol/input changed")
        require(monotonic() < deadline, "Bounded continuation deadline exceeded; no job was killed")

    try:
        while True:
            check()
            snapshot = process_probe(launch["pid"])
            if snapshot is not None:
                require(snapshot["uid"] == os.getuid(), "Training PID belongs to a different owner")
                if pinned_process is not None:
                    require(snapshot["start_ticks"] == pinned_process["start_ticks"], "Training PID was reused; no validation launched")
            if snapshot is None or snapshot["state"] == "Z":
                status["training_terminal_observation"] = {"proc_present": snapshot is not None,
                    "zombie": snapshot is not None, "exit_code_known": False, "observed_alive_before": pinned_process is not None}
                break
            require(snapshot["uid"] == os.getuid() and snapshot["argv"] == argv, "Owned training process UID/command identity changed")
            if pinned_process is None:
                pinned_process = snapshot
                status["training_process_identity"] = snapshot
            require(snapshot["start_ticks"] == pinned_process["start_ticks"], "Training PID was reused; no validation launched")
            status.update(status="waiting_for_owned_training", no_gpu_allocation_while_waiting=True)
            current_status = args.training_dir / "run_status.json"
            if current_status.exists():
                require(read_json(current_status).get("status") != "failed", "Owned formal training reported failure")
            write_json(status_path, status)
            sleep(args.poll_seconds)
        check()
        cohort = evaluator.verify_training(args.training_dir, args.protocol, allow_smoke=False)
        require(cohort["smoke_only"] is False, "Formal training artifact required")
        status["training_verification"] = {"status": "verified", "checkpoint_sha256": cohort["input_sha256"]["model_final.pt"],
            "exported_policy_sha256": cohort["input_sha256"]["policy.pt"],
            "training_audit_sha256": cohort["input_sha256"]["ground_contact_v2d_training_audit.json"]}
        write_json(status_path, status)
        for backend in ("mujoco", "physx"):
            check()
            index = None
            if backend == "physx":
                while True:
                    check()
                    snapshots = gpu_probe()
                    by_index = {item["physical_index"]: item for item in snapshots}
                    require(set(by_index) == set(GPU_ORDER), "Missing authorized full GPU snapshots")
                    selected = next((by_index[i] for i in GPU_ORDER if by_index[i]["idle"] is True
                        and by_index[i]["pci_bus_id_cuda_ordinal_verified"] is True
                        and not by_index[i]["all_reported_compute_and_graphics_processes"]), None)
                    status.update(status="waiting_for_fully_idle_gpu", last_gpu_preflight=snapshots)
                    write_json(status_path, status)
                    if selected is not None:
                        index = selected["physical_index"]
                        break
                    sleep(args.poll_seconds)
            environment = os.environ.copy()
            environment.update(PYTHONUNBUFFERED="1", TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
            if backend == "mujoco":
                environment.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="disable")
            else:
                environment.pop("CUDA_VISIBLE_DEVICES", None)
                environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID")
            command = child_command(args, backend, outputs[backend], index)
            child = {"backend": backend, "command": command, "status": "running", "started_unix_time": time.time(),
                "physical_gpu_index": index, "mu": .7}
            status["children"].append(child)
            status["status"] = "running_" + backend
            write_json(status_path, status)

            def spawned(pid):
                child["pid"] = pid
                write_json(status_path, status)

            child_args = (command, state_dir / (backend + ".log"), args.evaluation_dependencies, environment, spawned)
            code = (child_runner(*child_args, deadline=deadline) if child_runner is run_child else child_runner(*child_args))
            child.update(returncode=code, finished_unix_time=time.time())
            write_json(status_path, status)
            require(code == 0, "Owned nominal evaluation failed; no retry/next case")
            check()
            proof = evaluator.validate_output(outputs[backend], args.training_dir, args.protocol, backend=backend, allow_smoke=False)
            require(proof["checkpoint_sha256"] == status["training_verification"]["checkpoint_sha256"], "Nominal evaluations used a different actor")
            require(not (outputs[backend] / "remote_verification.json").exists(), "Existing evaluation proof must not be overwritten")
            write_json(outputs[backend] / "remote_verification.json", proof)
            child.update(status="completed_verified", verification=proof)
            write_json(status_path, status)
            sleep(2.)
        status["status"] = "nominal_dual_engine_validation_completed_requires_task_review"
        write_json(state_dir / "final_audit.json", status)
    except BaseException as error:
        status.update(status="stopped_without_retry", error=f"{type(error).__name__}: {error}")
        if status["children"] and status["children"][-1]["status"] == "running":
            status["children"][-1]["status"] = "failed_or_unverified_process_not_signalled"
        raise
    finally:
        status["finished_unix_time"] = time.time()
        write_json(status_path, status)
    return status


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("workspace", "evaluation-dependencies", "protocol", "legacy-native", "legacy-physx", "motion", "model", "asset", "beyondmimic-source", "python"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--launch-record", type=Path)
    def positive_pid(value):
        number = int(value)
        if number <= 0:
            raise argparse.ArgumentTypeError("Training PID must be a positive integer")
        return number
    result.add_argument("--expected-training-pid", type=positive_pid, required=True,
                        help="Exact already-launched owned V2d formal PID; never starts training")
    result.add_argument("--poll-seconds", type=float, default=30.)
    result.add_argument("--deadline-seconds", type=float, default=7200.)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())["status"]))

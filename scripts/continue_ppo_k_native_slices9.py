"""Wait for one exact model-specific PPO job, verify it, then two nominal arms.

One single-environment GPU child at a time, on a freshly fully idle GPU in
order3/5/4. No process signals, training retries, friction sweep or tuning.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def run(args):
    root = args.workspace.resolve()
    assert 1 <= args.poll_seconds <= 30 and 1 <= args.deadline_seconds <= 7200
    # Keep the lock open for this process's lifetime. It never reserves a GPU.
    with (root/"nominal_continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = root/"nominal_v1"
        directory.mkdir(exist_ok=False)
        status_path = directory/"status.json"
        status = dict(kind="native_slices9_training_then_two_nominal_validation_arms",
            status="verifying_owned_training_identity", cases=[], task_success_claimed=False,
            physical_GPU_order=[3, 5, 4], one_owned_GPU_job_at_a_time=True)
        write(status_path, status)
        try:
            helper = args.idle_audit_source.resolve()
            assert sha(helper) == "6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"
            spec = importlib.util.spec_from_file_location("_readonly_full_GPU_audit", helper)
            audit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(audit)
            launch = read(root/"formal_v1_launch.json")
            assert launch["run_kind"] == "model_specific_formal_from_scratch_12288000_transitions"
            assert launch["pid"] == args.expected_training_pid
            argv = launch["argv"]
            option = lambda name: argv[argv.index(name)+1]
            assert Path(argv[2]).resolve() == args.training_snapshot.resolve()/"scripts/train_ppo_k_native_slices9.py"
            assert option("--cohort-run-kind") == "formal" and option("--num-envs") == "512" and option("--iterations") == "1000"
            assert option("--seed") == "42" and option("--backend") == "physx"
            assert Path(option("--output-dir")).resolve() == root/"formal_v1"
            assert Path(option("--k-snapshot")).resolve() == args.k_snapshot.resolve()
            protocol = Path(option("--cohort-protocol")).resolve()
            assert read(protocol)["cohort_id"] == "tron1_ppo_k_native_slices9_400hz_v1"
            snapshot_manifest = read(root/"source_snapshot_v1.manifest.json")
            assert snapshot_manifest["archive_sha256"] == "ed2d69c2213582ae50193d2d3bc7fb3349f149bc84ad152d45fc21d32a2acd3e"
            assert sha(root/snapshot_manifest["archive"]) == snapshot_manifest["archive_sha256"]
            assert all(sha(args.training_snapshot/item["path"]) == item["sha256"] for item in snapshot_manifest["files"])
            source = {str(p.resolve()): sha(p) for p in (Path(__file__), helper, args.evaluation_snapshot/"scripts/eval_ppo_k_native_slices9.py")}
            source.update({str((args.training_snapshot/item["path"]).resolve()):item["sha256"] for item in snapshot_manifest["files"]})
            status.update(status="waiting_for_exact_formal_training", training_pid=launch["pid"], source_sha256=source,
                training_launch_sha256=sha(root/"formal_v1_launch.json"))
            write(status_path, status)
            deadline = time.monotonic()+args.deadline_seconds

            def bound():
                assert time.monotonic() < deadline, "Declared monitoring deadline reached; owned processes preserved"
                assert all(sha(p) == digest for p, digest in source.items()), "Source changed during continuation"

            def wait_child(child):
                while child.poll() is None:
                    bound()
                    time.sleep(args.poll_seconds)
                assert child.returncode == 0, ("Owned child failed; no retry", child.pid, child.returncode)

            while True:
                bound()
                process = audit.process_snapshot(launch["pid"])
                if process is None or process["state"] == "Z":
                    break
                assert all(process[key] == launch["process"][key] for key in ("pid", "uid", "start_ticks", "argv")), "Original training PID identity changed"
                time.sleep(args.poll_seconds)
            environment = os.environ.copy()
            environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1",
                TRON1_BEYONDMIMIC_ROOT=option("--beyondmimic-source"))
            environment["PYTHONPATH"] = str(args.training_snapshot/"scripts")+os.pathsep+environment.get("PYTHONPATH", "")
            environment["CUDA_VISIBLE_DEVICES"] = ""
            verify = [str(args.python), "-B", str(args.training_snapshot/"scripts/verify_ppo_k_native_slices9.py"),
                "--training-dir", str(root/"formal_v1"), "--protocol", str(protocol), "--k-snapshot", str(args.k_snapshot),
                "--cpu-inference", "--output", str(directory/"formal_training_strict_verification.json")]
            status["status"] = "strict_CPU_training_and_export_verification"
            write(status_path, status)
            with (directory/"training_verification.log").open("xb") as log:
                child = subprocess.Popen(verify, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                wait_child(child)
            proof = read(directory/"formal_training_strict_verification.json")
            assert proof["formal_or_smoke"] == "formal" and proof["transitions"] == 12288000 and proof["CPU_actor_output_shape"] == [1, 8]
            status["formal_training_verification"] = proof
            for geometry in ("slices9", "authored"):
                status["status"] = "waiting_for_fully_idle_GPU_for_"+geometry
                write(status_path, status)
                while True:
                    bound()
                    gpus = audit.query_gpus()
                    selected = next((gpu for gpu in gpus if gpu["idle"]), None)
                    if selected is not None:
                        break
                    time.sleep(args.poll_seconds)
                folder = directory/geometry
                command = [str(args.python), "-B", "-u", str(args.evaluation_snapshot/"scripts/eval_ppo_k_native_slices9.py"),
                    "--k-snapshot", str(args.k_snapshot), "--runtime-source", str(args.runtime_source),
                    "--authored-native-report", option("--authored-native-report"), "--geometry", geometry,
                    "--training-dir", str(root/"formal_v1"), "--protocol", str(protocol), "--backend", "physx",
                    "--legacy-evaluator", str(args.legacy_evaluator), "--motion-file", option("--motion"),
                    "--asset-path", option("--asset-path"), "--output-dir", str(folder), "--mu", ".7",
                    "--physical-gpu-index", str(selected["physical_index"])]
                environment.pop("CUDA_VISIBLE_DEVICES", None)
                with (directory/(geometry+".log")).open("xb") as log:
                    child = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    case = dict(id=geometry, pid=child.pid, argv=command, gpu_audit=gpus,
                        selected_gpu=selected, status="running", launched_unix_time=time.time())
                    status["cases"].append(case)
                    status["status"] = "running_"+geometry
                    write(status_path, status)
                    wait_child(child)
                case["returncode"] = child.returncode
                environment["CUDA_VISIBLE_DEVICES"] = ""
                with (directory/(geometry+"_verification.log")).open("xb") as log:
                    child = subprocess.Popen([*command, "--validate-only"], cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    wait_child(child)
                metadata = read(folder/"model_evaluation_status.json")
                assert metadata["normalized_actor_sha256"] == proof["actor_sha256"]
                case.update(status="completed_verified", metadata_sha256=sha(folder/"model_evaluation_status.json"),
                    physics_trajectory_sha256=sha(folder/"physics_trajectory.npz"),
                    tire_trajectory_sha256=sha(folder/"tire_trajectory.npz"), verification_log_sha256=sha(directory/(geometry+"_verification.log")))
                write(status_path, status)
            status.update(status="both_nominal_arms_completed_require_analysis", finished_unix_time=time.time())
            write(status_path, status)
        except BaseException as error:
            status.update(status="stopped_with_error_no_retry_or_process_signal", error=repr(error))
            write(status_path, status)
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("workspace", "training-snapshot", "evaluation-snapshot", "k-snapshot", "runtime-source", "idle-audit-source", "legacy-evaluator", "python"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--expected-training-pid", type=int, required=True)
    p.add_argument("--poll-seconds", type=float, default=15.)
    p.add_argument("--deadline-seconds", type=float, default=7200.)
    run(p.parse_args())

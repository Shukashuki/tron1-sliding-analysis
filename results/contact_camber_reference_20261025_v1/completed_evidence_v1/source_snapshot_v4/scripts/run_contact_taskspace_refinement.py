"""One native smoke, strict CPU proof, then one K task-space warm refinement."""
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


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def run(a):
    root, snapshot = a.workspace.resolve(), a.snapshot.resolve()
    record = root/"driver_status.json"
    assert not record.exists()
    state = dict(kind="K_taskspace_warm_refinement", status="preflight", children=[], proofs={}, task_success_claimed=False)
    write(record, state)
    with (root/"driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert sha(a.idle_audit_source) == "6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"
            spec = importlib.util.spec_from_file_location("_taskspace_full_idle_gpu_audit", a.idle_audit_source)
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            for pid in [807979, 807984, 808394, 808429, 808861, 808863, 811973, 811975,
                        816742, 815143, 817125, 817135, 818768, 818835, 819555, 820713]:
                process = helper.process_snapshot(pid)
                assert process is None or process["state"] == "Z", ("Previous owned job still active", process)
            manifest = json.loads((root/(snapshot.name+".manifest.json")).read_text())
            ledger = {str(snapshot/r["path"]): r["sha256"] for r in manifest["files"]}

            def bound():
                assert all(sha(p) == h for p, h in ledger.items()), "Source changed during task-space experiment"

            template = json.loads((snapshot/"config/contact_taskspace_refinement_launch_template.json").read_text())["argv"]
            environment = os.environ.copy()
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1")

            def child(argv, label, env, extra=None):
                with (root/(label+".log")).open("xb") as log:
                    p = subprocess.Popen(argv, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
                    row = dict(label=label, pid=p.pid, argv=argv, started_unix_time=time.time(),
                               status="running", process_at_launch=helper.process_snapshot(p.pid), **(extra or {}))
                    state["children"].append(row)
                    state["status"] = "running_"+label
                    write(record, state)
                    while p.poll() is None:
                        time.sleep(2.)
                row["returncode"] = p.returncode
                assert p.returncode == 0, ("Owned job failed; no retry or signal", label, p.returncode)
                row["status"] = "terminal_require_artifact_verification"
                write(record, state)
                return row

            phases = [("smoke", 32, 2)]
            if not a.smoke_only:
                phases.append(("formal", 512, 200))
            for mode, n, iterations in phases:
                bound()
                label = mode+"_K"
                state["status"] = "waiting_for_fully_idle_GPU_"+label
                write(record, state)
                while True:
                    gpus = helper.query_gpus()
                    selected = next((g for g in gpus if g["idle"]), None)
                    if selected:
                        break
                    time.sleep(15.)
                bound()
                argv = template[:]
                argv[0] = str(a.python)
                argv[3] = str(snapshot/"scripts/train_contact_taskspace_refinement.py")
                changes = {"--cohort-protocol": str(snapshot/"config/contact_taskspace_refinement_400hz_v1.json"),
                           "--reward-variant": "K", "--cohort-run-kind": mode, "--num-envs": str(n),
                           "--iterations": str(iterations), "--output-dir": str(root/label),
                           "--declared-input-dir": str(root/(label+"_declared_inputs")),
                           "--device": "cuda:"+str(selected["physical_index"])}
                for flag, value in changes.items():
                    assert argv.count(flag) == 1
                    argv[argv.index(flag)+1] = value
                assert not (root/label).exists()
                job = child(argv, label, environment, dict(selected_gpu=selected, gpu_audit=gpus))
                verify = [str(a.python), "-B", str(snapshot/"scripts/verify_contact_taskspace_refinement.py"),
                          "--training-dir", str(root/label), "--protocol", changes["--cohort-protocol"],
                          "--cpu-inference", "--output", str(root/(label+"_verification.json"))]
                if mode == "smoke":
                    verify.append("--allow-smoke")
                cpu = environment.copy()
                cpu.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                check = child(verify, label+"_CPU_verification", cpu)
                proof = json.loads((root/(label+"_verification.json")).read_text())
                assert proof["status"] == "completed_verified" and proof["task_space_tracking_verified"]
                assert proof["run_kind"] == mode and proof["transitions"] == n*24*iterations
                assert proof["CPU_actor_output_shape"] == [1, 8]
                assert not proof["from_scratch"] and proof["historical_checkpoint_used"]
                state["proofs"][label] = proof
                job["status"] = check["status"] = "completed_strictly_verified"
                write(record, state)
            state["status"] = "smoke_completed_verified" if a.smoke_only else "formal_completed_verified_nominal_pending"
            bound()
            write(record, state)
        except BaseException as error:
            state.update(status="failed_no_retry_or_process_signal", error=repr(error))
            write(record, state)
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("workspace", "snapshot", "idle-audit-source", "python"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    run(p.parse_args())

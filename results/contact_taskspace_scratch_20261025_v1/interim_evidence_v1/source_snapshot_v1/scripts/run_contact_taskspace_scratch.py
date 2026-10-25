"""Serial fresh K/M: both native smokes/CPU proofs, then equal formal budgets.

Use a fresh full compute/graphics NVML audit before every native launch. Never
signal other processes, mask CUDA devices, reserve a GPU or retry a failure.
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

IDLE_SHA = "6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temp, path)


def run(args):
    root, snapshot = args.workspace.resolve(), args.snapshot.resolve()
    record = root/"driver_status.json"
    assert not record.exists()
    state = dict(kind="matched_fresh_taskspace_K_M_PPO", status="preflight", children=[], proofs={}, task_success_claimed=False)
    write(record, state)
    with (root/"driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert sha(args.idle_audit_source) == IDLE_SHA
            spec = importlib.util.spec_from_file_location("_fresh_PPO_idle_audit", args.idle_audit_source)
            helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
            for pid in [915796, 915800, 916198, 916209, 919653, 919716, 919719, 920326]:
                process = helper.process_snapshot(pid)
                assert process is None or process["state"] == "Z", ("Previous owned R process active", process)
            manifest = json.loads((root/(snapshot.name+".manifest.json")).read_text())
            ledger = {str(snapshot/f["path"]): f["sha256"] for f in manifest["files"]}

            def bound():
                assert all(sha(path) == digest for path, digest in ledger.items()), "Source changed"

            environment = os.environ.copy(); environment.pop("CUDA_VISIBLE_DEVICES", None)
            environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1")
            template = json.loads((snapshot/"config/contact_taskspace_scratch_launch_template.json").read_text())["argv"]

            def child(argv, label, env, extra=None):
                with (root/(label+".log")).open("xb") as log:
                    p = subprocess.Popen(argv, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
                    row = dict(label=label, pid=p.pid, argv=argv, started_unix_time=time.time(),
                        status="running", process_at_launch=helper.process_snapshot(p.pid), **(extra or {}))
                    state["children"].append(row); state["status"] = "running_"+label; write(record, state)
                    while p.poll() is None:
                        time.sleep(2.)
                row["returncode"] = p.returncode
                assert p.returncode == 0, ("Owned job failed; preserve evidence, no retry", label, p.returncode)
                row["status"] = "terminal_require_artifact_verification"; write(record, state)
                return row

            phases = [("smoke", 32, 2)]
            if not args.smoke_only:
                phases.append(("formal", 512, 1000))
            for mode, count, iterations in phases:
                for variant in ["K", "M"]:
                    label = mode+"_"+variant
                    bound(); state["status"] = "waiting_fully_idle_GPU_"+label; write(record, state)
                    while True:
                        gpus = helper.query_gpus(); selected = next((g for g in gpus if g["idle"]), None)
                        if selected: break
                        time.sleep(15.)
                    bound(); argv = [v.replace("__SNAPSHOT__", str(snapshot)).replace("__WORKSPACE__", str(root)) for v in template]
                    argv[0] = str(args.python)
                    changes = {"--reward-variant": variant, "--cohort-run-kind": mode, "--num-envs": str(count),
                        "--iterations": str(iterations), "--output-dir": str(root/label),
                        "--declared-input-dir": str(root/(label+"_declared_inputs")),
                        "--device": "cuda:"+str(selected["physical_index"])}
                    for flag, value in changes.items():
                        assert argv.count(flag) == 1; argv[argv.index(flag)+1] = value
                    assert "--initial-checkpoint" not in argv and not (root/label).exists()
                    job = child(argv, label, environment, dict(selected_gpu=selected, gpu_audit=gpus))
                    verify = [str(args.python), "-B", str(snapshot/"scripts/verify_contact_taskspace_scratch.py"),
                        "--training-dir", str(root/label), "--protocol", str(snapshot/"config/contact_taskspace_scratch_400hz_v1.json"),
                        "--cpu-inference", "--output", str(root/(label+"_verification.json"))]
                    if mode == "smoke": verify.append("--allow-smoke")
                    cpu = environment.copy(); cpu.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    check = child(verify, label+"_CPU_verification", cpu)
                    proof = json.loads((root/(label+"_verification.json")).read_text())
                    assert proof["status"] == "completed_verified" and proof["task_space_tracking_verified"]
                    assert proof["from_scratch"] and not proof["historical_checkpoint_used"]
                    assert proof["transitions"] == count*24*iterations and proof["CPU_actor_output_shape"] == [1, 8]
                    state["proofs"][label] = proof; job["status"] = check["status"] = "completed_strictly_verified"; write(record, state)
            state["status"] = "both_smokes_verified" if args.smoke_only else "both_formal_verified_nominal_validation_pending"
            bound(); write(record, state)
        except BaseException as error:
            state.update(status="failed_no_retry_or_process_signal", error=repr(error)); write(record, state)
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("workspace", "snapshot", "idle-audit-source", "python"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    run(p.parse_args())

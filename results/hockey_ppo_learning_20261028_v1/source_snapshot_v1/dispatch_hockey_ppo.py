"""Run bounded hockey PPO arms serially, with a fresh full-idle RTX check."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "prior-launch", "audit-helper", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Fresh output and unmasked devices required")
    if sha(a.audit_helper) != "6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866":
        raise ValueError("GPU audit helper changed")
    spec = importlib.util.spec_from_file_location("gpu_audit", a.audit_helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.GPU_ORDER = (3, 5, 4, 0, 1, 2, 6, 7)
    manifest = json.loads((a.bundle/"manifest.json").read_text())
    for row in manifest["files"]:
        path = (a.bundle/row["path"]).resolve()
        if not path.is_relative_to(a.bundle.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("Snapshot differs")
    cfg = json.loads((a.bundle/"protocol.json").read_text())
    if cfg["formal"] != {"num_envs": 512, "iterations": 1000} or cfg["smoke"] != {"num_envs": 32, "iterations": 2}:
        raise ValueError("Unreviewed training budget")
    old = json.loads(a.prior_launch.read_text())["argv"]
    get = lambda name: old[old.index("--"+name)+1]
    a.output.mkdir()
    records = []
    stages = [(v, "smoke") for v in ("tracking", "support")]
    stages += [(v, m) for v in ("tracking", "support") for m in ("formal", "eval")]
    for variant, mode in stages:
        idle = next((g for g in helper.query_gpus() if g["idle"]), None)
        if idle is None:
            raise RuntimeError("No fully idle RTX; remaining stages not started")
        name = variant+"_"+mode
        argv = [old[0], "-B", "-u", str(a.bundle/"train_hockey_ppo.py"),
            "--protocol", str(a.bundle/"protocol.json"), "--motion", str(a.bundle/"motion.npz"),
            "--mode", mode, "--variant", variant, "--device", "cuda:"+str(idle["physical_index"]),
            "--output-dir", str(a.output/name)]
        for key in ("asset", "tracking-root", "beyondmimic-root"):
            argv += ["--"+key, get(key)]
        if mode == "eval":
            argv += ["--actor", str(a.output/(variant+"_formal")/"actor_normalized.pt")]
        with (a.output/(name+".log")).open("xb") as log:
            child = subprocess.Popen(argv, cwd=a.output, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"), shell=False)
        row = dict(id=name, pid=child.pid, identity=helper.process_snapshot(child.pid), argv=argv,
                   gpu_before_launch=idle, status="running")
        records.append(row)
        status = a.output/"status.json"
        status.write_text(json.dumps(records, indent=2)+"\n")
        code = child.wait(); row["returncode"] = code
        row["status"] = "failed" if code else "completed"
        status.write_text(json.dumps(records, indent=2)+"\n")
        if code:
            raise RuntimeError(name+" failed; no automatic retries or subsequent jobs")
        report = json.loads((a.output/name/"report.json").read_text())
        if mode != "eval" and report["transitions"] != cfg[mode]["num_envs"]*24*cfg[mode]["iterations"]:
            raise ValueError("Unexpected transition count")
        if mode == "formal" and variant == "support":
            baseline = json.loads((a.output/"tracking_formal/report.json").read_text())
            if report["initial_policy_state_sha256"] != baseline["initial_policy_state_sha256"]:
                raise ValueError("Arms did not have matched initial model states")
        row["report_sha256"] = sha(a.output/name/"report.json")
        status.write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()

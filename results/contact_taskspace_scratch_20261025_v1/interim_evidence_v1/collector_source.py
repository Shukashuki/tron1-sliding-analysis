"""Capture source, completed smokes and live formal startup, excluding weights.

Interim evidence is not a training-completion or task-success claim. Read the
exact live procfs identities; never signal, restart or otherwise control jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import time


def collect(args):
    root = args.workspace.resolve()
    helper_path = Path("/data/zengxiong/workspaces/tron1-ground-contact-v2o-20261025/continuation_snapshot_v1/scripts/continue_ground_contact_v2o_validation.py")
    sha = lambda data: hashlib.sha256(data).hexdigest()
    assert sha(helper_path.read_bytes()) == "6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"
    spec = importlib.util.spec_from_file_location("_interim_fresh_PPO_processes", helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    launch = json.loads((root/"driver_v1_launch.json").read_text())
    assert launch["pid"] == args.driver_pid and launch["start_ticks"] == args.driver_start_ticks
    driver = helper.process_snapshot(args.driver_pid)
    assert driver and driver["start_ticks"] == args.driver_start_ticks and driver["state"] != "Z"
    assert driver["argv"] == launch["argv"]
    data = {}

    def add(name):
        p = root/name
        assert p.is_file() and not p.is_symlink() and p.suffix in (".py", ".json", ".log")
        b = p.read_bytes()
        if p.suffix == ".json": json.loads(b)
        data[name] = b

    for name in ("driver_v1_launch.json", "driver_status.json", "source_snapshot_v1.manifest.json",
                 "smoke_K_verification.json", "smoke_M_verification.json"):
        add(name)
    state = json.loads(data["driver_status.json"])
    assert state["status"] == "running_formal_K"
    processes = {str(args.driver_pid): driver}
    for child in state["children"]:
        process = helper.process_snapshot(child["pid"])
        processes[str(child["pid"])] = process
        if child["label"] == "formal_K":
            assert process and process["state"] != "Z"
            assert process["start_ticks"] == child["process_at_launch"]["start_ticks"] and process["argv"] == child["argv"]
        else:
            assert process is None or process["state"] == "Z"
            assert child["status"] == "completed_strictly_verified" and child["returncode"] == 0
    for v in ("K", "M"):
        proof = json.loads(data["smoke_"+v+"_verification.json"])
        assert proof["status"] == "completed_verified" and proof["from_scratch"] and proof["transitions"] == 1536
        for name in ("manifest.json", "cohort_startup_audit.json", "cohort_completion_audit.json", "policy_contract.json",
                     "final_policy_contract.json", "run_status.json", "pre_export_runtime_audit.json", "export_precision_audit.json",
                     "runner_config.json", "environment_final.json"):
            add("smoke_"+v+"/"+name)
        add("smoke_"+v+"_CPU_verification.log")
    source = json.loads(data["source_snapshot_v1.manifest.json"])
    assert source["archive_sha256"] == launch["source_archive_sha256"] == "92886904ed08f1372032ba82256d7c944fe8de813e22964da57ab43059904438"
    for row in source["files"]:
        name = "source_snapshot_v1/"+row["path"]; add(name)
        assert sha(data[name]) == row["sha256"] and len(data[name]) == row["bytes"]
    for name in ("cohort_startup_audit.json", "runner_config.json"):
        add("formal_K/"+name)
    data["formal_K_log_tail_at_collection.log"] = (root/"formal_K.log").read_bytes()[-16000:]
    data["collector_source.py"] = Path(__file__).read_bytes()
    data["process_observation.json"] = (json.dumps(dict(observed_unix_time=time.time(),
        exact_driver_identity_live=True, exact_formal_K_identity_live=True, processes=processes,
        formal_training_completed=False, task_success_claimed=False), indent=2)+"\n").encode()
    target = root/"interim_evidence_v1.tar.gz"; manifest_path = root/"interim_evidence_v1.manifest.json"
    assert not target.exists() and not manifest_path.exists()
    with tarfile.open(target, "x:gz") as archive:
        for name, b in sorted(data.items()):
            info = tarfile.TarInfo(name); info.size = len(b); info.mode = 0o644
            archive.addfile(info, io.BytesIO(b))
    manifest = dict(kind="fresh_taskspace_K_M_smoke_and_live_formal_K_interim", formal_training_completed=False,
        task_success_claimed=False, archive_sha256=sha(target.read_bytes()), archive_bytes=target.stat().st_size,
        models_weights_credentials_native_binaries_included=False,
        files=[dict(path=name, sha256=sha(b), bytes=len(b)) for name, b in sorted(data.items())])
    with manifest_path.open("x") as f: json.dump(manifest, f, indent=2, allow_nan=False)
    print(json.dumps({k: manifest[k] for k in ("kind", "archive_sha256", "archive_bytes", "formal_training_completed")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--workspace", type=Path, required=True); p.add_argument("--driver-pid", type=int, required=True)
    p.add_argument("--driver-start-ticks", type=int, required=True)
    collect(p.parse_args())

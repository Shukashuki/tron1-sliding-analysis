"""Collect an allowlisted completed comparison; never package neural weights.

Run on the existing remote workspace after the sequential verifier completes.
Only code, JSON audits, strict verifier logs and the four native trajectory
NPZ files are included. A live/incomplete comparison is rejected before writes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def collect(workspace, output_prefix="completed_evidence_v1"):
    root = Path(workspace).resolve()
    assert output_prefix.replace("_", "").isalnum() and not output_prefix.startswith(".")
    status = read(root/"nominal_v1/status.json")
    assert status["status"] == "both_nominal_arms_completed_require_analysis", "Formal comparison is still incomplete"
    proof = read(root/"nominal_v1/formal_training_strict_verification.json")
    assert proof["formal_or_smoke"] == "formal" and proof["transitions"] == 12288000
    assert proof["CPU_actor_output_shape"] == [1, 8] and status["formal_training_verification"] == proof
    children = {child["id"]: child for child in status["cases"]}
    assert set(children) == {"authored", "slices9"} and len(status["cases"]) == 2
    files = ["source_snapshot_v1.manifest.json", "evaluation_snapshot_v1.manifest.json",
        "smoke_v1_strict_verification.json", "formal_v1_launch.json", "nominal_continuation_launch.json",
        "nominal_v1/status.json", "nominal_v1/formal_training_strict_verification.json", "nominal_v1/training_verification.log"]
    formal = ("cohort_completion_audit", "cohort_startup_audit", "environment_final", "final_policy_contract",
        "ground_contact_v2k_startup_audit", "ground_contact_v2k_training_audit", "manifest", "policy_contract",
        "run_status", "runner_config", "slices9_model_startup_audit", "slices9_model_training_audit")
    files += ["formal_v1/"+name+".json" for name in formal]
    for snapshot in ("source_snapshot_v1", "evaluation_snapshot_v1"):
        manifest = read(root/(snapshot+".manifest.json"))
        for item in manifest["files"]:
            relative = Path(item["path"])
            assert not relative.is_absolute() and ".." not in relative.parts and relative.suffix in (".py", ".json")
            assert sha((root/snapshot/relative).read_bytes()) == item["sha256"]
            files.append(snapshot+"/"+str(relative))
    assert sha((root/"formal_v1/slices9_model_training_audit.json").read_bytes()) == proof["training_audit_sha256"]
    for geometry, child in children.items():
        assert child["status"] == "completed_verified" and child["returncode"] == 0
        folder = root/"nominal_v1"/geometry
        assert sha((folder/"model_evaluation_status.json").read_bytes()) == child["metadata_sha256"]
        metadata = read(folder/"model_evaluation_status.json")
        assert metadata["normalized_actor_sha256"] == proof["actor_sha256"] and metadata["validation_geometry"] == geometry
        for name in ("physics_trajectory", "tire_trajectory"):
            assert sha((folder/(name+".npz")).read_bytes()) == child[name+"_sha256"] == metadata[name+"_sha256"]
        log = "nominal_v1/"+geometry+"_verification.log"
        assert sha((root/log).read_bytes()) == child["verification_log_sha256"]
        files.append(log)
        files += ["nominal_v1/"+geometry+"/"+name for name in (
            "report.json", "policy_contract.json", "model_evaluation_status.json", "physics_trajectory.npz", "tire_trajectory.npz")]
    assert len(files) == len(set(files))
    payloads = {}
    for name in files:
        path = root/name
        assert path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
        assert path.suffix in (".py", ".json", ".npz", ".log") and path.stat().st_size <= 128*1024**2
        payloads[name] = path.read_bytes()
    assert sum(map(len, payloads.values())) <= 256*1024**2
    assert payloads["nominal_v1/status.json"] == (root/"nominal_v1/status.json").read_bytes()
    archive = root/(output_prefix+".tar.gz")
    manifest_path = root/(output_prefix+".manifest.json")
    assert not archive.exists() and not manifest_path.exists(), "Preserve previous collection"
    with archive.open("xb") as file:
        with tarfile.open(fileobj=file, mode="w:gz") as tar:
            for name, data in payloads.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
    manifest = dict(kind="completed_native_slices9_formal_PhysX_model_comparison_evidence",
        archive=archive.name, archive_sha256=sha(archive.read_bytes()), archive_size_bytes=archive.stat().st_size,
        file_count=len(files), files=[dict(path=name, size_bytes=len(data), sha256=sha(data)) for name, data in payloads.items()],
        includes_model_assets_neural_weights_or_credentials=False, task_success_claimed=False,
        collector_sha256=sha(Path(__file__).read_bytes()))
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(json.dumps({key:value for key,value in manifest.items() if key != "files"}))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output-prefix", default="completed_evidence_v1")
    args = p.parse_args()
    collect(args.workspace, args.output_prefix)

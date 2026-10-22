"""Verify and bundle four completed nominal episodes; never copy policy weights.

Run on the authorized experiment host, using its immutable validator and local
trained artifacts for read-only hash checks. The new archive contains only an
explicit JSON/NPZ allowlist. It neither changes jobs nor performs inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from zoneinfo import ZoneInfo


def digest(data):
    return hashlib.sha256(data).hexdigest()


def add_file(payload, name, path):
    target, path = Path(name), Path(path)
    if (name in payload or target.is_absolute() or ".." in target.parts
            or target.suffix not in (".json", ".npz") or path.is_symlink() or not path.is_file()):
        raise ValueError("Require a unique, regular, explicitly allowed JSON/NPZ artifact")
    payload[name] = path.read_bytes()


def write_archive(output, payload, manifest):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Never overwrite an existing retrieval archive")
    with tarfile.open(output, "x:gz") as archive:
        encoded = (json.dumps(manifest, indent=2, allow_nan=False)+"\n").encode()
        for name, data in (*payload.items(), ("manifest.json", encoded)):
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            archive.addfile(item, io.BytesIO(data))
    return {"archive": str(output), "sha256": digest(output.read_bytes()),
        "bytes": output.stat().st_size, "members": len(payload)+1, "nominal_cases": len(manifest["cases"])}


def run(args):
    base, output = args.workspace.resolve(), args.output.resolve()
    if output.parent != base or output.exists():
        raise ValueError("Use a fresh retrieval archive directly inside the explicit workspace")
    snapshot, matrix = base/"validation_source_v1", base/"matrix_seed42_gpu4_v1"
    sys.path.insert(0, str(snapshot/"scripts"))
    import ppo_backend_400hz_contract as shared
    protocol = snapshot/"config/ppo_backend_400hz_comparison_v1.json"
    state_path = matrix/"backend_400hz_matrix_status.json"
    state = json.loads(state_path.read_text())
    nominal = [c for c in state["cases"] if c["phase"] == "formal" and c["mu"] == .7]
    expected = {(a, b) for a in ("physx", "mjlab") for b in ("physx", "mujoco")}
    if len(nominal) != 4 or {(c["training_backend"], c["engine_tag"]) for c in nominal} != expected:
        raise ValueError("Require the four genuine completed nominal matrix cells")
    payload, training = {}, {}
    add_file(payload, "matrix_status_snapshot.json", state_path)
    add_file(payload, "config/ppo_backend_400hz_comparison_v1.json", protocol)
    for origin in ("physx", "mjlab"):
        folder = base/"pair_seed42_gpu4_v1"/(origin+"_formal")
        shared.verify_cohort(folder, protocol)
        training[origin] = {}
        for key, filename in (("manifest", "manifest.json"), ("startup_audit", "cohort_startup_audit.json"),
                ("completion_audit", "cohort_completion_audit.json"),
                ("final_policy_contract", "final_policy_contract.json"), ("run_status", "run_status.json")):
            relative = "training/"+origin+"/"+filename
            add_file(payload, relative, folder/filename)
            training[origin][key] = relative
    cases = []
    for case in nominal:
        if case["status"] != "completed_verified":
            raise ValueError("Nominal episode did not complete verification")
        folder, origin, engine = Path(case["directory"]), case["training_backend"], case["engine_tag"]
        proof = shared.validate_output(folder, case["cohort_dir"], protocol,
            validation_backend=case["validation_backend"])
        if proof != case["verification"]:
            raise ValueError("Completed original parent proof changed")
        directory = "cases/"+origin+"/"+engine
        raw = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
        for filename in (raw, "report.json"):
            add_file(payload, directory+"/"+filename, folder/filename)
        if (folder/"policy_contract.json").is_file():
            add_file(payload, directory+"/policy_contract.json", folder/"policy_contract.json")
        cases.append({"case_id": origin+"_to_"+engine, "training_backend": origin, "engine_tag": engine,
            "validation_backend": case["validation_backend"], "mu": .7, "directory": directory,
            "remote_directory": str(folder), "raw_file": raw, "report_file": "report.json",
            "recording_sha256": proof["recording_sha256"], "report_sha256": proof["report_sha256"],
            "checkpoint_sha256": case["checkpoint_sha256"], "actor_sha256": case["actor_sha256"], "verification": proof})
    now = datetime.now(timezone.utc)
    manifest = {"schema_version": 1, "kind": "verified_nominal_400hz_backend_partial_matrix_replay_inputs",
        "cohort_id": "traj4_backend_400hz_actionrate0p5_seed42_v1", "nominal_mu": .7,
        "nominal_only": True, "full_matrix_complete": False, "observed_utc": now.isoformat(),
        "observed_new_york": now.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "matrix_status": {"status": state["status"], "counts": state["counts"]},
        "reference_sha256": shared.INPUT_PINS["motion"], "model_sha256": shared.INPUT_PINS["model"],
        "cases": cases, "training_metadata": training,
        "files": [{"path": name, "size": len(data), "sha256": digest(data)} for name, data in sorted(payload.items())],
        "no_weights_or_credentials_in_bundle": True, "new_policy_inference": False,
        "new_physics_simulation": False, "remote_completed_model_hashes_verified": True}
    return write_archive(output, payload, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))

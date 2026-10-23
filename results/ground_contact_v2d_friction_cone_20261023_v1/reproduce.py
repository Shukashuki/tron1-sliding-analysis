"""Replay recorded friction diagnostics without assets, weights, or simulation.

This checks bundled bytes and recomputes all four scientific rows and every
derived compact signal. It does not re-run the policy or authenticate omitted
original rollout bytes. Source modules are loaded from memory, never extracted.
Run: python results/ground_contact_v2d_friction_cone_20261023_v1/reproduce.py
Requires NumPy; makes no filesystem writes.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import types

sys.dont_write_bytecode = True
import numpy as np

BASE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((BASE / "manifest.json").read_text())
    names = set()
    for row in manifest["files"]:
        name = PurePosixPath(row["path"])
        require(not name.is_absolute() and ".." not in name.parts and str(name) not in names,
                "Unsafe or duplicate bundle member")
        names.add(str(name))
        path = BASE / str(name)
        require(path.is_file() and not path.is_symlink(), "Missing direct bundle file")
        data = path.read_bytes()
        require(len(data) == row["size_bytes"] and sha256(data) == row["sha256"], "Bundle bytes differ: " + str(name))
    source_manifest = json.loads((BASE / "source_snapshot.manifest.json").read_text())
    archive = (BASE / "source_snapshot.tar.gz").read_bytes()
    require(sha256(archive) == source_manifest["archive_sha256"], "Frozen source archive changed")
    expected = {row["path"]: row for row in source_manifest["files"]}
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as stream:
        members = stream.getmembers()
        require(len(members) == len(expected) and {member.name for member in members} == set(expected),
                "Unexpected frozen source members")
        for member in members:
            require(member.isfile() and member.size == expected[member.name]["size_bytes"], "Invalid source member")
            data = stream.extractfile(member).read()
            require(sha256(data) == expected[member.name]["sha256"], "Frozen source bytes differ")
            sources[member.name] = data
    for name in ("friction_cone_diagnostics", "analyze_archived_wheel_friction"):
        path = "scripts/" + name + ".py"
        module = types.ModuleType(name)
        module.__file__ = str(BASE / "in_memory_source" / path)
        sys.modules[name] = module
        exec(compile(sources[path], module.__file__, "exec"), module.__dict__)
    summary = json.loads((BASE / "summary.json").read_text())
    require(summary["data_scope"] == "wheel_resultant_lower_bound"
            and summary["actual_per_contact_engine_boundary_computed"] is False
            and summary["reward_or_acceptance_changed"] is False, "Diagnostic scope differs")
    require(summary["analyzer_sha256"] == sha256(sources["scripts/analyze_archived_wheel_friction.py"])
            and summary["shared_diagnostic_sha256"] == sha256(sources["scripts/friction_cone_diagnostics.py"]),
            "Scientific summary/source binding differs")
    with np.load(BASE / "friction_signals.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    rebuilt = sys.modules["analyze_archived_wheel_friction"].recompute_compact(arrays, summary["provenance"])
    require(rebuilt["rows"] == summary["rows"], "Scientific metric rows differ")
    require(rebuilt["thresholds"] == summary["exploratory_thresholds_NOT_acceptance"], "Diagnostic thresholds differ")
    print(json.dumps({"status": "verified_recorded_arithmetic_only", "scientific_rows": len(rebuilt["rows"]),
        "native_samples": len(arrays["time_s"]), "all_compact_derived_arrays_recomputed": True,
        "actual_per_contact_engine_boundary_computed": False,
        "new_physics_inference_training_or_filesystem_writes": False}, indent=2))


if __name__ == "__main__":
    main()

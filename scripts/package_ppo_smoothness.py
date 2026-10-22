"""Create an immutable, pure-code/config snapshot for sequential jitter tests.

Assets, reference motion, checkpoints and frozen upstream source are supplied
separately from existing authorized paths on the experiment host.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/train_beyondmimic_style.py", "scripts/train_ppo_smoothness.py",
    "scripts/height_smoothness.py", "scripts/controller_references.py",
    "scripts/run_ppo_smoothness_sequence.py", "scripts/analyze_ppo_smoothness.py",
    "scripts/run_car_validation.py", "scripts/analyze_friction_grid.py",
    "scripts/analyze_saved_xy.py", "scripts/package_ppo_smoothness.py",
    "config/ppo_smoothness_sequence_v1.json", "config/car_drift_height_smooth_task.json",
    "config/controller_references.json", "config/car_reference_lock.json",
    "config/evaluation_friction.json",
    "tests/test_train_ppo_smoothness.py", "tests/test_analyze_ppo_smoothness.py",
    "tests/test_run_ppo_smoothness_sequence.py",
    "tests/test_package_ppo_smoothness.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(output, root=ROOT):
    output, root = Path(output), Path(root)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Preserving existing source snapshot")
    entries = []
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular allowlisted source file: {name}")
        entries.append({"path": name, "size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"schema_version": 1, "kind": "ppo_smoothness_pure_source_snapshot",
                "contents": "pure Python source, tests and JSON configuration only",
                "excludes": ["robot assets", "checkpoints", "motion data", "credentials", "binaries"],
                "files": entries, "total_files": len(entries),
                "total_source_bytes": sum(entry["size"] for entry in entries)}
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for entry in entries:
            archive.add(root / entry["path"], arcname=entry["path"], recursive=False)
        header = tarfile.TarInfo("snapshot_manifest.json")
        header.size = len(rendered)
        archive.addfile(header, io.BytesIO(rendered))
    with manifest_path.open("xb") as stream:
        stream.write(rendered)
    return {"archive": str(output), "manifest": str(manifest_path),
            "archive_sha256": sha256(output), "bytes": output.stat().st_size,
            "source_files": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))

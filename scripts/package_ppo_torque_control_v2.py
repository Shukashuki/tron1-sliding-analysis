"""Create a fresh source-only snapshot for prepared torque-control v2.

This packager launches nothing and imports no training/simulation modules.
Assets, motion data and weights remain at separately authorized existing paths.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/train_ppo_torque_control_v2.py", "scripts/run_ppo_torque_control_v2.py",
    "scripts/package_ppo_torque_control_v2.py", "scripts/ppo_leg_torque_smoothness.py",
    "scripts/train_ppo_smoothness.py", "scripts/train_beyondmimic_style.py",
    "scripts/height_smoothness.py", "scripts/run_ppo_smoothness_sequence.py",
    "scripts/run_car_validation.py", "scripts/analyze_ppo_smoothness.py",
    "scripts/analyze_ppo_contact_stability.py", "scripts/summarize_ppo_smoothness.py",
    "scripts/analyze_friction_grid.py", "scripts/analyze_saved_xy.py",
    "scripts/controller_references.py", "config/ppo_torque_control_v2.json",
    "config/car_drift_height_smooth_task.json", "config/controller_references.json",
    "config/car_reference_lock.json", "config/evaluation_friction.json",
    "tests/test_train_ppo_torque_control_v2.py", "tests/test_run_ppo_torque_control_v2.py",
    "tests/test_package_ppo_torque_control_v2.py", "tests/test_ppo_leg_torque_smoothness.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(output, root=ROOT):
    output, root = Path(output), Path(root)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or output.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("Preserving existing source snapshot or manifest")
    if len(FILES) != len(set(FILES)):
        raise ValueError("Duplicate source allowlist entry")
    entries, contents = [], {}
    for name in FILES:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix not in (".py", ".json"):
            raise ValueError("Only relative Python/JSON source allowlist paths are permitted")
        path = root / name
        if path.is_symlink() or not path.is_file() or any((root / parent).is_symlink() for parent in relative.parents):
            raise ValueError("Missing regular allowlisted source file: " + name)
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
            if relative.suffix == ".json": json.loads(text)
            else: ast.parse(text, filename=name)
        except (UnicodeError, SyntaxError, ValueError) as error:
            raise ValueError("Allowlisted file must contain Python/JSON source, not binary data: " + name) from error
        contents[name] = data
        entries.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"schema_version": 1, "kind": "ppo_torque_control_v2_pure_source_snapshot",
        "contents": "pure Python source, tests and JSON configuration only", "packaging_launches_jobs": False,
        "excludes": ["robot assets", "checkpoints", "motion data", "credentials", "binaries",
                     "ground-support experiment", "old torque-control protocol"],
        "files": entries, "total_files": len(entries), "total_source_bytes": sum(x["size"] for x in entries)}
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for entry in entries:
            header = tarfile.TarInfo(entry["path"])
            header.size, header.mode = entry["size"], 0o644
            archive.addfile(header, io.BytesIO(contents[entry["path"]]))
        header = tarfile.TarInfo("snapshot_manifest.json")
        header.size, header.mode = len(rendered), 0o644
        archive.addfile(header, io.BytesIO(rendered))
    with manifest_path.open("xb") as stream: stream.write(rendered)
    return {"archive": str(output), "manifest": str(manifest_path), "archive_sha256": sha256(output),
            "bytes": output.stat().st_size, "source_files": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))

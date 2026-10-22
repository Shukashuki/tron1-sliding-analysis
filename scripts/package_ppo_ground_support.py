"""Package the prepared ground-support trial as source/configuration only.

No upload, training, policy inference or asset discovery is performed. Existing
snapshots are preserved; only the fixed allowlist can enter this archive.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import package_ppo_smoothness as base


ROOT = Path(__file__).resolve().parents[1]
FILES = (*base.FILES,
    "scripts/ppo_leg_torque_smoothness.py",
    "scripts/ppo_substep_ground_support.py",
    "scripts/train_ppo_ground_support.py",
    "scripts/run_ppo_ground_support.py",
    "scripts/analyze_ppo_contact_stability.py",
    "scripts/summarize_ppo_smoothness.py",
    "scripts/package_ppo_ground_support.py",
    "config/ppo_ground_support_v1.json",
    "tests/test_ppo_leg_torque_smoothness.py",
    "tests/test_ppo_substep_ground_support.py",
    "tests/test_train_ppo_ground_support.py",
    "tests/test_run_ppo_ground_support.py",
    "tests/test_analyze_ppo_contact_stability.py",
    "tests/test_summarize_ppo_smoothness.py",
    "tests/test_package_ppo_ground_support.py",
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
            if relative.suffix == ".json":
                json.loads(text)
            else:
                ast.parse(text, filename=name)
        except (UnicodeError, SyntaxError, ValueError) as error:
            raise ValueError("Allowlisted file must contain Python/JSON source, not binary data: " + name) from error
        contents[name] = data
        entries.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema_version": 1, "kind": "ppo_ground_support_pure_source_snapshot",
        "contents": "pure Python source, tests and JSON configuration only",
        "packaging_launches_jobs": False,
        "excludes": ["robot assets", "checkpoints", "motion data", "credentials", "binaries"],
        "files": entries, "total_files": len(entries),
        "total_source_bytes": sum(entry["size"] for entry in entries),
    }
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Archive the same captured bytes that were hashed, not a second read of a
    # potentially changing workspace file. Never extract or discover assets.
    with tarfile.open(output, "x:gz") as archive:
        for entry in entries:
            header = tarfile.TarInfo(entry["path"])
            header.size, header.mode = entry["size"], 0o644
            archive.addfile(header, io.BytesIO(contents[entry["path"]]))
        header = tarfile.TarInfo("snapshot_manifest.json")
        header.size, header.mode = len(rendered), 0o644
        archive.addfile(header, io.BytesIO(rendered))
    with manifest_path.open("xb") as stream:
        stream.write(rendered)
    return {"archive": str(output), "manifest": str(manifest_path), "archive_sha256": sha256(output),
            "bytes": output.stat().st_size, "source_files": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(package(parser.parse_args().output), indent=2))

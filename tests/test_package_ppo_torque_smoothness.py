"""Source-only isolated torque snapshot packaging tests; no Isaac import."""
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import package_ppo_smoothness as base
import package_ppo_torque_smoothness as torque


def test_exact_allowlist_and_preserve_shared_packager(tmp_path):
    original = base.FILES
    archive = tmp_path / "pure_source.tar.gz"
    metadata = torque.package(archive)
    assert base.FILES is original
    assert metadata["source_files"] == len(torque.FILES) == len(set(torque.FILES))
    with tarfile.open(archive, "r:gz") as stream:
        assert set(stream.getnames()) == set(torque.FILES) | {"snapshot_manifest.json"}
        manifest = json.load(stream.extractfile("snapshot_manifest.json"))
        assert manifest["total_files"] == len(torque.FILES)
        assert all(Path(item["path"]).suffix in (".py", ".json") for item in manifest["files"])
        assert not any("outputs/" in item["path"] or "results/" in item["path"] for item in manifest["files"])
    with pytest.raises(FileExistsError):
        torque.package(archive)
    assert base.FILES is original


def test_restore_shared_allowlist_after_error(tmp_path):
    original = base.FILES
    with pytest.raises(ValueError, match="Missing regular allowlisted"):
        torque.package(tmp_path / "missing.tar.gz", root=tmp_path)
    assert base.FILES is original

import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "package_ppo_smoothness", Path(__file__).resolve().parents[1] / "scripts/package_ppo_smoothness.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_root(tmp_path):
    root = tmp_path / "source"
    for name in MODULE.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# pure source\n")
    (root / "private_checkpoint.pt").write_bytes(b"excluded")
    return root


def test_allowlist_and_no_overwrite(tmp_path):
    root = fixture_root(tmp_path)
    output = tmp_path / "snapshot.tar.gz"
    result = MODULE.package(output, root=root)
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert set(names) == set(MODULE.FILES) | {"snapshot_manifest.json"}
        manifest = json.load(archive.extractfile("snapshot_manifest.json"))
    assert result["source_files"] == len(MODULE.FILES)
    assert manifest["total_files"] == len(MODULE.FILES)
    assert result["archive_sha256"] == MODULE.sha256(output)
    for entry in manifest["files"]:
        assert entry["sha256"] == MODULE.sha256(root / entry["path"])
    with pytest.raises(FileExistsError):
        MODULE.package(output, root=root)


def test_reject_missing_or_symlink_source(tmp_path):
    root = fixture_root(tmp_path)
    path = root / MODULE.FILES[0]
    path.unlink()
    with pytest.raises(ValueError, match="regular allowlisted"):
        MODULE.package(tmp_path / "missing.tar.gz", root=root)
    path.symlink_to(root / MODULE.FILES[1])
    with pytest.raises(ValueError, match="regular allowlisted"):
        MODULE.package(tmp_path / "symlink.tar.gz", root=root)


def test_existing_manifest_is_preserved(tmp_path):
    root = fixture_root(tmp_path)
    output = tmp_path / "snapshot.tar.gz"
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text("existing")
    with pytest.raises(FileExistsError):
        MODULE.package(output, root=root)
    assert manifest_path.read_text() == "existing"
    assert not output.exists()

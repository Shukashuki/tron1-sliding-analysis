"""Source-only packaging tests; no Isaac, GPU, network or training execution."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import package_ppo_smoothness as base
import package_ppo_ground_support as p


def source_fixture(tmp_path):
    root = tmp_path / "source"
    for name in p.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# source fixture\n")
    for name in ("model_final.pt", "motion.npz", "robot.usda", ".ssh/id_ed25519", "private_credentials.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not enter allowlisted snapshot")
    return root


def test_exact_allowlist_archive_and_manifest_hashes(tmp_path):
    root = source_fixture(tmp_path)
    original = base.FILES
    archive = tmp_path / "source_only.tar.gz"
    result = p.package(archive, root=root)
    assert base.FILES is original
    assert result["source_files"] == len(p.FILES) == 34
    assert len(p.FILES) == len(set(p.FILES))
    with tarfile.open(archive, "r:gz") as stream:
        assert set(stream.getnames()) == set(p.FILES) | {"snapshot_manifest.json"}
        manifest_bytes = stream.extractfile("snapshot_manifest.json").read()
        assert manifest_bytes == archive.with_suffix(".manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        assert manifest["kind"] == "ppo_ground_support_pure_source_snapshot"
        assert manifest["packaging_launches_jobs"] is False
        for entry in manifest["files"]:
            data = stream.extractfile(entry["path"]).read()
            assert len(data) == entry["size"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"] == p.sha256(root / entry["path"])
        assert all(item.isfile() and not item.issym() for item in stream.getmembers())
    assert result["archive_sha256"] == p.sha256(archive)


def test_existing_archive_or_manifest_never_overwritten(tmp_path):
    root = source_fixture(tmp_path)
    output = tmp_path / "snapshot.tar.gz"
    output.write_bytes(b"old archive")
    with pytest.raises(FileExistsError): p.package(output, root=root)
    assert output.read_bytes() == b"old archive"
    other = tmp_path / "another.tar.gz"
    metadata = other.with_suffix(".manifest.json")
    metadata.write_text("old metadata")
    with pytest.raises(FileExistsError): p.package(other, root=root)
    assert metadata.read_text() == "old metadata" and not other.exists()


@pytest.mark.parametrize("mode", ["missing", "symlink", "binary", "invalid_python"])
def test_invalid_source_rejected_before_archive_creation(tmp_path, mode):
    root = source_fixture(tmp_path)
    path = root / p.FILES[0]
    if mode == "missing": path.unlink()
    if mode == "symlink":
        path.unlink(); path.symlink_to(root / p.FILES[1])
    if mode == "binary": path.write_bytes(b"\x80\x00\xff")
    if mode == "invalid_python": path.write_text("def broken(:\n")
    output = tmp_path / "invalid.tar.gz"
    with pytest.raises(ValueError): p.package(output, root=root)
    assert not output.exists() and not output.with_suffix(".manifest.json").exists()


def test_invalid_json_rejected(tmp_path):
    root = source_fixture(tmp_path)
    (root / "config/ppo_ground_support_v1.json").write_text("not json")
    with pytest.raises(ValueError, match="Python/JSON source"):
        p.package(tmp_path / "invalid.tar.gz", root=root)


@pytest.mark.parametrize("entry", ["../credentials.py", "/tmp/secret.py", "model.pt"])
def test_allowlist_path_and_type_guard(tmp_path, monkeypatch, entry):
    root = source_fixture(tmp_path)
    monkeypatch.setattr(p, "FILES", (*p.FILES, entry))
    with pytest.raises(ValueError, match="relative Python/JSON"):
        p.package(tmp_path / "invalid.tar.gz", root=root)


def test_duplicate_entries_and_broken_output_symlink_rejected(tmp_path, monkeypatch):
    root = source_fixture(tmp_path)
    output = tmp_path / "broken.tar.gz"
    output.symlink_to(tmp_path / "nonexistent")
    with pytest.raises(FileExistsError): p.package(output, root=root)
    monkeypatch.setattr(p, "FILES", (*p.FILES, p.FILES[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        p.package(tmp_path / "duplicate.tar.gz", root=root)


def test_allowlist_preserves_original19_and_only_new_required_method_files():
    assert p.FILES[:len(base.FILES)] == base.FILES
    assert set(p.FILES) - set(base.FILES) == {
        "scripts/ppo_leg_torque_smoothness.py", "scripts/ppo_substep_ground_support.py",
        "scripts/train_ppo_ground_support.py", "scripts/run_ppo_ground_support.py",
        "scripts/analyze_ppo_contact_stability.py", "scripts/summarize_ppo_smoothness.py",
        "scripts/package_ppo_ground_support.py", "config/ppo_ground_support_v1.json",
        "tests/test_ppo_leg_torque_smoothness.py", "tests/test_ppo_substep_ground_support.py",
        "tests/test_train_ppo_ground_support.py", "tests/test_run_ppo_ground_support.py",
        "tests/test_analyze_ppo_contact_stability.py", "tests/test_summarize_ppo_smoothness.py",
        "tests/test_package_ppo_ground_support.py"}


def test_actual_workspace_source_package_if_dispatcher_ready(tmp_path):
    if not all((ROOT / name).is_file() for name in p.FILES):
        pytest.skip("Prepared dispatcher source/tests not ready yet")
    before = {name: p.sha256(ROOT / name) for name in p.FILES}
    result = p.package(tmp_path / "actual_sources.tar.gz")
    assert result["source_files"] == 34
    assert before == {name: p.sha256(ROOT / name) for name in p.FILES}

"""Pure-source v2 allowlist/archive tests; no imports of simulator or trainer."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import package_ppo_torque_control_v2 as p


def fixture_source(tmp_path):
    root = tmp_path / "source"
    for name in p.FILES:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# source fixture\n")
    for name in ("model_final.pt", "motion.npz", "robot.usda", ".ssh/id_ed25519",
                 "scripts/ppo_substep_ground_support.py", "scripts/train_ppo_torque_smoothness.py",
                 "config/ppo_leg_torque_smoothness_v1.json", "config/ppo_ground_support_v1.json"):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must be excluded")
    return root


def test_allowlist_is_minimal_unique_and_excludes_other_protocols():
    assert len(p.FILES) == len(set(p.FILES)) == 24
    assert "scripts/ppo_leg_torque_smoothness.py" in p.FILES
    assert not any("ground_support" in name or "train_ppo_torque_smoothness" in name or
                   "ppo_leg_torque_smoothness_v1.json" in name for name in p.FILES)
    assert all(Path(name).suffix in (".py", ".json") and name.split("/")[0] in ("scripts", "tests", "config") for name in p.FILES)


def test_exact_captured_source_bytes_and_manifest_hashes(tmp_path):
    root = fixture_source(tmp_path)
    output = tmp_path / "v2.tar.gz"
    result = p.package(output, root=root)
    assert result["source_files"] == 24 and result["archive_sha256"] == p.sha256(output)
    with tarfile.open(output) as stream:
        assert set(stream.getnames()) == set(p.FILES) | {"snapshot_manifest.json"}
        rendered = stream.extractfile("snapshot_manifest.json").read()
        assert rendered == output.with_suffix(".manifest.json").read_bytes()
        manifest = json.loads(rendered)
        assert manifest["kind"] == "ppo_torque_control_v2_pure_source_snapshot"
        assert manifest["packaging_launches_jobs"] is False
        assert all(item.isfile() and not item.issym() for item in stream.getmembers())
        for entry in manifest["files"]:
            data = stream.extractfile(entry["path"]).read()
            assert len(data) == entry["size"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"] == p.sha256(root / entry["path"])


@pytest.mark.parametrize("existing", ["archive", "manifest", "broken_symlink"])
def test_existing_evidence_never_overwritten(tmp_path, existing):
    root = fixture_source(tmp_path); output = tmp_path / "v2.tar.gz"
    if existing == "archive": output.write_bytes(b"old archive")
    if existing == "manifest": output.with_suffix(".manifest.json").write_text("old manifest")
    if existing == "broken_symlink": output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError): p.package(output, root=root)
    if existing == "archive": assert output.read_bytes() == b"old archive"
    if existing == "manifest": assert output.with_suffix(".manifest.json").read_text() == "old manifest"


@pytest.mark.parametrize("invalid", ["missing", "symlink", "binary", "bad_python", "bad_json", "directory_symlink"])
def test_invalid_or_nonregular_source_fails_before_archive(tmp_path, invalid):
    root = fixture_source(tmp_path); path = root / p.FILES[0]
    if invalid == "missing": path.unlink()
    if invalid == "symlink":
        path.unlink(); path.symlink_to(root / p.FILES[1])
    if invalid == "binary": path.write_bytes(b"\xff\x00")
    if invalid == "bad_python": path.write_text("def broken(:")
    if invalid == "bad_json": (root / "config/ppo_torque_control_v2.json").write_text("not json")
    if invalid == "directory_symlink":
        (root / "scripts").rename(tmp_path / "real_scripts")
        (root / "scripts").symlink_to(tmp_path / "real_scripts", target_is_directory=True)
    output = tmp_path / "invalid.tar.gz"
    with pytest.raises(ValueError): p.package(output, root=root)
    assert not output.exists() and not output.with_suffix(".manifest.json").exists()


@pytest.mark.parametrize("entry", ["../private.py", "/tmp/private.json", "model.pt"])
def test_no_traversal_absolute_or_binary_allowlist_entry(tmp_path, monkeypatch, entry):
    root = fixture_source(tmp_path)
    monkeypatch.setattr(p, "FILES", (*p.FILES, entry))
    with pytest.raises(ValueError, match="relative Python/JSON"):
        p.package(tmp_path / "bad.tar.gz", root=root)


def test_duplicate_allowlist_entry_rejected(tmp_path, monkeypatch):
    root = fixture_source(tmp_path)
    monkeypatch.setattr(p, "FILES", (*p.FILES, p.FILES[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        p.package(tmp_path / "bad.tar.gz", root=root)


def test_actual_source_package_preserves_all_workspace_inputs(tmp_path):
    if not all((ROOT / name).is_file() for name in p.FILES): pytest.skip("Prepared v2 source/tests incomplete")
    before = {name: p.sha256(ROOT / name) for name in p.FILES}
    result = p.package(tmp_path / "actual_v2.tar.gz")
    assert result["source_files"] == 24
    assert before == {name: p.sha256(ROOT / name) for name in p.FILES}

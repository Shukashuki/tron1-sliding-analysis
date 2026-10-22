import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("backend400_packet", ROOT/"scripts/package_ppo_backend_400hz.py")
packet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packet)


def fixture_root(tmp_path):
    root = tmp_path/"source"
    names = (["scripts/"+n+".py" for n in packet.CORE_SCRIPTS] +
        ["tests/"+n+".py" for n in packet.CORE_TESTS] + ["config/"+n+".json" for n in packet.CONFIGS] +
        ["native/ppo_batch.cpp", "scripts/run_ppo_backend_400hz_pair.py", "tests/test_run_ppo_backend_400hz_pair.py"] +
        ["upstream/tron1-learning-from-human/training/"+n+".py" for n in packet.UPSTREAM_TRAINING] +
        ["upstream/tron1-learning-from-human/scripts/"+n+".py" for n in ("eval_tracking", "eval_tracking_mujoco", "train_tracking")] +
        ["upstream/tron1-learning-from-human/notices/"+n for n in packet.NOTICES])
    for name in names:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# fixture source\n")
    (root/"private_checkpoint.pt").write_bytes(b"never included")
    (root/"motion.npz").write_bytes(b"never included")
    return root


@pytest.mark.parametrize("calibration", [True, False])
def test_immutable_explicit_packet_and_new_cohort_manifest(tmp_path, calibration):
    root = fixture_root(tmp_path)
    output = tmp_path/"source.tar.gz"
    result = packet.package(output, root=root, calibration_only=calibration)
    expected = packet.build_file_list(root, calibration_only=calibration)
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == set(expected) | {"snapshot_manifest.json"}
        manifest = json.load(archive.extractfile("snapshot_manifest.json"))
    assert manifest["kind"] == "matched_ppo_backend_400hz_pure_source_snapshot"
    assert manifest["total_files"] == len(expected) == result["source_files"]
    assert manifest["upstream_root_license_file_present"] is False
    assert not any(n.endswith((".pt", ".npz", ".so")) for n in expected)
    assert ("scripts/run_ppo_backend_400hz_pair.py" in expected) is not calibration
    assert "native/ppo_batch.cpp" in expected
    with pytest.raises(FileExistsError):
        packet.package(output, root=root, calibration_only=calibration)


def test_refuse_unreviewed_upstream_training_addition(tmp_path):
    root = fixture_root(tmp_path)
    (root/"upstream/tron1-learning-from-human/training/unreviewed.py").write_text("# new\n")
    with pytest.raises(ValueError, match="training file set changed"):
        packet.build_file_list(root)


def test_refuse_file_symlink_or_missing_required_transitive_helper(tmp_path):
    root = fixture_root(tmp_path)
    path = root/"scripts/eval_ppo_contact_damping.py"; path.unlink()
    with pytest.raises(ValueError, match="regular allowlisted"):
        packet.build_file_list(root)
    path.symlink_to(root/"scripts/package_ppo_backend.py")
    with pytest.raises(ValueError, match="regular allowlisted"):
        packet.build_file_list(root)


def test_existing_manifest_prevents_archive_creation(tmp_path):
    root = fixture_root(tmp_path)
    output = tmp_path/"source.tar.gz"
    manifest = output.with_suffix(".manifest.json"); manifest.write_text("existing")
    with pytest.raises(FileExistsError):
        packet.package(output, root=root)
    assert manifest.read_text() == "existing" and not output.exists()

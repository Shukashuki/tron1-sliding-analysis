import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import package_ppo_backend_400hz_matrix as packet


def fixture_root(tmp_path):
    root = tmp_path/"source"
    names = (*packet.training_packet.build_file_list(ROOT), *packet.EXTRA_FILES)
    for name in names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# fixture source\n")
    (root/"private_key").write_bytes(b"must not include")
    (root/"policy.pt").write_bytes(b"must not include")
    return root


def test_explicit_safe_packet_and_no_overwrite(tmp_path):
    root = fixture_root(tmp_path)
    path = tmp_path/"matrix.tar.gz"
    result = packet.package(path, root)
    with tarfile.open(path) as bundle:
        manifest = json.load(bundle.extractfile("snapshot_manifest.json"))
        assert manifest["kind"] == "ppo_backend_400hz_validation_matrix_source_snapshot"
        assert manifest["training_snapshot_modified"] is False
        assert set(bundle.getnames()) == set(packet.build_file_list(root))|{"snapshot_manifest.json"}
        assert all(member.isfile() for member in bundle.getmembers())
    assert result["source_files"] == len(packet.build_file_list(root))
    assert not any(entry["path"].endswith((".pt", ".npz", ".so")) for entry in manifest["files"])
    with pytest.raises(FileExistsError):
        packet.package(path, root)


def test_refuse_missing_transitive_import(tmp_path):
    root = fixture_root(tmp_path)
    (root/"scripts/summarize_ppo_smoothness.py").unlink()
    with pytest.raises(ValueError, match="allowlisted validation source"):
        packet.build_file_list(root)


def test_refuse_linked_new_adapter(tmp_path):
    root = fixture_root(tmp_path)
    path = root/"scripts/eval_ppo_backend_400hz_isaac.py"
    path.unlink(); path.symlink_to(root/"scripts/train_ppo_backend.py")
    with pytest.raises(ValueError, match="allowlisted validation source"):
        packet.build_file_list(root)

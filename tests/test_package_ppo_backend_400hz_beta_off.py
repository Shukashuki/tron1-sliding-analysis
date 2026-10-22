import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import package_ppo_backend_400hz_beta_off as packet


def fixture_root(tmp_path):
    root = tmp_path / "source"
    for name in (*packet.baseline_packet.build_file_list(ROOT), *packet.EXTRA_FILES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# fixture only\n")
    (root / "id_ed25519").write_text("excluded fixture")
    (root / "policy.pt").write_text("excluded fixture")
    return root


def test_safe_explicit_allowlist_and_no_overwrite(tmp_path):
    root = fixture_root(tmp_path)
    output = tmp_path / "source.tar.gz"
    result = packet.package(output, root)
    with tarfile.open(output) as bundle:
        manifest = json.load(bundle.extractfile("snapshot_manifest.json"))
        assert manifest["kind"] == "ppo_backend_400hz_beta_off_source_snapshot"
        assert manifest["baseline_source_files_edited"] is False
        assert set(bundle.getnames()) == set(packet.build_file_list(root)) | {"snapshot_manifest.json"}
        assert all(member.isfile() for member in bundle.getmembers())
        assert not any(name.endswith((".pt", ".npz", ".so")) for name in bundle.getnames())
    assert result["source_files"] == len(packet.build_file_list(root))
    with pytest.raises(FileExistsError):
        packet.package(output, root)


def test_missing_transitive_source_rejected(tmp_path):
    root = fixture_root(tmp_path)
    (root / "scripts/eval_ppo_backend_400hz_beta_off_mujoco.py").unlink()
    with pytest.raises(ValueError, match="allowlisted beta-off"):
        packet.build_file_list(root)


def test_symlinked_source_rejected(tmp_path):
    root = fixture_root(tmp_path)
    path = root / "scripts/run_ppo_backend_400hz_beta_off.py"
    path.unlink()
    path.symlink_to(root / "scripts/train_ppo_backend.py")
    with pytest.raises(ValueError, match="allowlisted beta-off"):
        packet.build_file_list(root)

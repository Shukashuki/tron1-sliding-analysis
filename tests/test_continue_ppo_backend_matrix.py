"""Evidence reuse must preserve bytes and refuse redirects/overwrites."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("continuation", Path(__file__).resolve().parents[1] / "scripts/continue_ppo_backend_400hz_matrix.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_copy_preserves_evidence_and_refuses_overwrite(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    (source / "raw").mkdir()
    (source / "raw/trace.bin").write_bytes(bytes(range(256)))
    (source / "report.json").write_text('{"verified": true}\n')
    target = tmp_path / "new"
    hashes = module.copy_verified_tree(source, target)
    assert len(hashes) == 2
    assert (target / "raw/trace.bin").read_bytes() == bytes(range(256))
    with pytest.raises(ValueError, match="fresh destination"):
        module.copy_verified_tree(source, target)
    assert (source / "report.json").read_text() == '{"verified": true}\n'


def test_redirected_evidence_is_rejected_before_copy(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    outside = tmp_path / "unrelated"
    outside.write_text("untouched")
    (source / "report.json").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        module.copy_verified_tree(source, tmp_path / "new")
    assert not (tmp_path / "new").exists()
    assert outside.read_text() == "untouched"


def test_busy_process_prevents_launch():
    observations = []
    with pytest.raises(TimeoutError):
        module.wait_idle(lambda _: dict(idle=True, physical_index=3,
            pci_bus_id_cuda_ordinal_verified=True,
            all_reported_compute_and_graphics_processes=[{"pid": 123}]), 3, 0, observations.append)
    assert len(observations) == 1

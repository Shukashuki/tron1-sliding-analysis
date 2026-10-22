import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from package_ppo_backend_400hz_nominal_results import add_file, digest, write_archive


def test_explicit_data_archive_and_no_overwrite(tmp_path):
    source = tmp_path/"source.json"; source.write_text("{}")
    payload = {}; add_file(payload, "cases/a/report.json", source)
    manifest = {"cases": ["test"], "files": [{"path": "cases/a/report.json", "sha256": digest(b"{}")}]}
    output = tmp_path/"result.tar.gz"
    result = write_archive(output, payload, manifest)
    with tarfile.open(output) as archive:
        assert archive.getnames() == ["cases/a/report.json", "manifest.json"]
        assert all(member.isfile() for member in archive.getmembers())
        assert json.load(archive.extractfile("manifest.json")) == manifest
    assert result["nominal_cases"] == 1 and result["sha256"] == digest(output.read_bytes())
    with pytest.raises(FileExistsError):
        write_archive(output, payload, manifest)


@pytest.mark.parametrize("name", ["../report.json", "/report.json", "actor.pt", "model.xml", "id_ed25519"])
def test_disallow_paths_and_non_data_suffixes(tmp_path, name):
    source = tmp_path/"source.json"; source.write_text("{}")
    with pytest.raises(ValueError):
        add_file({}, name, source)


def test_reject_symlink_and_duplicate(tmp_path):
    source = tmp_path/"source.json"; source.write_text("{}")
    link = tmp_path/"link.json"; link.symlink_to(source)
    with pytest.raises(ValueError):
        add_file({}, "link.json", link)
    payload = {}; add_file(payload, "report.json", source)
    with pytest.raises(ValueError):
        add_file(payload, "report.json", source)

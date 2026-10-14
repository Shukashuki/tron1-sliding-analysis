import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "registry", Path(__file__).parents[1] / "scripts/discover_re3data_human_sources.py")
registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry)

INDEX = b'<repositories><repository><id>r3d100011968</id><name>Motion Capture Database HDM05</name><doi>10.17616/R38P9K</doi></repository></repositories>'
RECORD = b'''<r:re3data xmlns:r="urn:test"><r:repository>
<r:re3data.orgIdentifier>r3d100011968</r:re3data.orgIdentifier>
<r:repositoryName>Motion Capture Database HDM05</r:repositoryName>
<r:repositoryURL>http://example.org/HDM05/index.html</r:repositoryURL>
<r:dataAccessType>open</r:dataAccessType>
<r:dataLicenseURL>https://creativecommons.org/licenses/by-sa/3.0/</r:dataLicenseURL>
</r:repository></r:re3data>'''


def test_index_namespaces():
    assert registry.parse_index(INDEX)[0]["id"] == registry.HDM05_ID
    assert registry.parse_index(INDEX.replace(b'<repositories>', b'<repositories xmlns="urn:test">')) == registry.parse_index(INDEX)


def test_record_not_a_motion_download():
    record = registry.parse_record(RECORD, registry.HDM05_ID)
    assert record["data_access_type"] == ["open"]
    assert record["motion_file_links"] == []
    assert record["record_sha256"] == hashlib.sha256(RECORD).hexdigest()
    with pytest.raises(ValueError, match="identifier"):
        registry.parse_record(RECORD, "r3d123")


def test_offline_discovery_preserves_raw_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "fetch", lambda url: INDEX if url.endswith("/repositories") else RECORD)
    report = registry.discover(tmp_path, [registry.HDM05_ID, registry.HDM05_ID])
    assert len(report["selected_records"]) == 1
    assert (tmp_path / (registry.HDM05_ID + ".xml")).read_bytes() == RECORD
    assert json.loads((tmp_path / "re3data_sources.json").read_text()) == report
    assert "excludes running and skating" in report["selection"]
    with pytest.raises(FileExistsError):
        registry.discover(tmp_path, [registry.HDM05_ID])

"""Asset-free checks that do not require local Torch, Warp, or a CUDA device."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preflight_mjlab_runtime.py"
SPEC = importlib.util.spec_from_file_location("mjlab_runtime_preflight", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_missing_source_is_reported_without_importing_simulators(tmp_path, capsys):
    destination = tmp_path / "diagnostic.json"
    status = PREFLIGHT.main(["--mjlab-source", str(tmp_path / "missing"),
                             "--output", str(destination)])
    assert status == 1
    data = json.loads(destination.read_text())
    assert data["passed"] is False
    assert data["error_type"] == "FileNotFoundError"
    assert set(PREFLIGHT.PACKAGES) == set(data["packages"])
    assert json.loads(capsys.readouterr().out) == data


def test_too_short_test_would_not_validate_ground_contact(tmp_path):
    with pytest.raises(ValueError, match="At least 40"):
        PREFLIGHT.run_preflight(tmp_path, steps=1)


def test_existing_result_is_not_overwritten(tmp_path):
    destination = tmp_path / "diagnostic.json"
    destination.write_text("original evidence\n")
    with pytest.raises(FileExistsError):
        PREFLIGHT.main(["--mjlab-source", str(tmp_path / "missing"),
                        "--output", str(destination)])
    assert destination.read_text() == "original evidence\n"


def test_package_inventory_is_json_serializable():
    versions = PREFLIGHT.package_versions()
    assert set(versions) == set(PREFLIGHT.PACKAGES)
    assert all(value is None or isinstance(value, str) for value in versions.values())
    json.dumps(versions)

"""Incomplete or smoke evidence must never become a four-arm formal result."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_ppo_k_native_slices9 as analysis
import collect_ppo_k_native_slices9 as collection


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_live_continuation_cannot_be_analyzed_as_completed(tmp_path):
    write(tmp_path/"nominal_v1/status.json", {"status": "running_slices9"})
    with pytest.raises(ValueError, match="Both verified nominal arms are required"):
        analysis.verify_new_training(tmp_path, tmp_path)


def test_genuine_completed_smoke_is_rejected_as_formal_training(tmp_path):
    smoke = ROOT/"results/ppo_k_native_slices9_20261025_v1/evidence/smoke_v1_strict_verification.json"
    write(tmp_path/"nominal_v1/status.json", {"status": "both_nominal_arms_completed_require_analysis"})
    write(tmp_path/"nominal_v1/formal_training_strict_verification.json", json.loads(smoke.read_text()))
    with pytest.raises(ValueError, match="smoke or incomplete training"):
        analysis.verify_new_training(tmp_path, tmp_path)


def test_child_without_strict_verification_is_rejected(tmp_path):
    write(tmp_path/"model_evaluation_status.json", {})
    write(tmp_path/"report.json", {})
    child = {"id": "slices9", "status": "running", "returncode": None}
    with pytest.raises(ValueError, match="did not complete strict verification"):
        analysis.verify_new_case(tmp_path, child, {}, {}, "slices9", tmp_path)


def test_collector_leaves_live_training_and_weights_untouched(tmp_path):
    write(tmp_path/"nominal_v1/status.json", {"status": "waiting_for_exact_formal_training"})
    weights = tmp_path/"formal_v1/model_final.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"must not be read or packaged")
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(AssertionError, match="comparison is still incomplete"):
        collection.collect(tmp_path)
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after

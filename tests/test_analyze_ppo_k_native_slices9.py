"""Incomplete or smoke evidence must never become a four-arm formal result."""
import json
from pathlib import Path
import sys

import numpy as np
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


def native_fixture():
    trace = dict(time_s=np.arange(2793)*.0025, floor_normal_force=np.zeros((2793, 2, 3)),
        position=np.zeros((2793, 3)), com_velocity=np.zeros((2793, 3)),
        angular_velocity=np.zeros((2793, 3)), joint_vel=np.zeros((2793, 8)))
    trace["floor_normal_force"][:, 0, 2] = 6.
    raw = dict(action_was_applied=np.r_[False, np.ones(2792, dtype=bool)][:, None],
        reference_frame_for_action=np.r_[0, np.repeat(np.arange(349), 8)][:, None],
        actions_clipped=np.zeros((2793, 1, 8)), applied_joint_torque_nm=np.zeros((2793, 1, 8)))
    return raw, trace


def test_gap_threshold_allows_single_wheel_and_does_not_gate_height():
    raw, trace = native_fixture()
    trace["position"][:, 2] = 20.
    trace["com_velocity"][:, 2] = 100.
    assert analysis.support_gap_diagnostics(raw, trace)["interruption_count"] == 0
    trace["floor_normal_force"][100, :, 2] = 5.
    result = analysis.support_gap_diagnostics(raw, trace)
    assert result["unsupported_native_samples"] == 1
    gap = result["interruptions"][0]
    assert gap["start_sample"] == gap["end_sample"] == 100
    assert gap["unsupported_sample_duration_s"] == .0025
    assert gap["first_to_last_sample_span_s"] == 0.
    assert gap["root_height_m_at_gap_start"] == 20.


def test_gap_intervals_use_native_samples_and_exclude_outside_contact_window():
    raw, trace = native_fixture()
    trace["floor_normal_force"][[0, 2792], :, 2] = 0.
    trace["floor_normal_force"][100:103, :, 2] = 0.
    trace["floor_normal_force"][110, :, 2] = 0.
    result = analysis.support_gap_diagnostics(raw, trace)
    assert result["unsupported_native_samples"] == 4
    assert [row["unsupported_native_samples"] for row in result["interruptions"]] == [3, 1]
    assert result["interruptions"][0]["unsupported_sample_duration_s"] == .0075


def test_action_context_requires_exact_original_eight_sample_hold():
    raw, trace = native_fixture()
    trace["floor_normal_force"][80, :, 2] = 0.
    raw["actions_clipped"][81:89, 0, 0] = .5
    result = analysis.support_gap_diagnostics(raw, trace)
    assert result["interruptions"][0]["context_policy_action_jump_l2_peak"] == .5
    raw["actions_clipped"][82, 0, 0] = .7
    with pytest.raises(AssertionError):
        analysis.support_gap_diagnostics(raw, trace)

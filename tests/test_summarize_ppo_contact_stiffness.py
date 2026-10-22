"""CPU fixtures only: copied baseline arrays are not new stiffness experiments."""
from pathlib import Path
import copy
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import summarize_ppo_contact_stiffness as stiffness


def candidate_report():
    report = json.loads((stiffness.DEFAULT_BASELINE/"report.json").read_text())
    audit = report["contact_stability_diagnostic"]
    for key in ("deployment_leg_kd", "trained_leg_kd", "unchanged_leg_kp"):
        audit.pop(key)
    audit.update(kind="fixed_ppo_nominal_contact_stiffness_diagnostic", trained_leg_kp=500.,
        deployment_leg_kp=250., unchanged_leg_kd=10., only_changed_control_field="leg_kp",
        wrapper_sha256=stiffness.DEFAULT_WRAPPER_SHA256, provenance_adapter_sha256=stiffness.BASE_WRAPPER_SHA256)
    report["evaluator_source_sha256"] = stiffness.DEFAULT_WRAPPER_SHA256
    report["policy_export"]["contact_stability_diagnostic"] = copy.deepcopy(audit)
    report["policy_export"]["leg_kp"] = 250.
    report["policy_export"]["task_profile"]["leg_kp"] = 250.
    report["contact_material_audit"]["leg_pd_gain_readback"]["kp_range"] = [250., 250.]
    return report


@pytest.fixture
def candidate(tmp_path):
    folder = tmp_path/"kp250"; folder.mkdir()
    (folder/"report.json").write_text(json.dumps(candidate_report()))
    (folder/"physics_trajectory.npz").symlink_to(stiffness.DEFAULT_BASELINE/"physics_trajectory.npz")
    return folder


def test_independent_kinds_and_original_baseline_are_preserved(candidate):
    raw, summary = stiffness.load_comparison(candidate=candidate)
    assert summary["case_order"] == ["kp500", "kp250"]
    assert summary["deployed_leg_kp"] == [500., 250.] and summary["deployed_leg_kd"] == 10
    assert summary["reused_baseline_episode"] is True and summary["new_candidate_episodes"] == 1
    assert summary["cases"]["kp500"]["diagnostic_contract"]["kind"] == "fixed_ppo_nominal_contact_damping_diagnostic"
    assert summary["cases"]["kp250"]["diagnostic_contract"]["kind"] == "fixed_ppo_nominal_contact_stiffness_diagnostic"
    assert summary["cases"]["kp500"]["input_sha256"]["report.json"] == stiffness.BASE_REPORT_SHA256
    assert summary["cases"]["kp500"]["windows"]["core"]["tracking"]["xy_rmse_m"] == pytest.approx(.056221388688688805)
    assert summary["cases"]["kp250"]["windows"]["core"]["samples"] == 513
    assert summary["kp250_minus_kp500"]["core"]["both_contact_duty_percentage_point_change"] == 0
    assert np.array_equal(raw["kp500"]["root_position_m"], raw["kp250"]["root_position_m"])
    assert summary["new_training"] is summary["stability_success_claimed"] is False
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("field,value", [("deployment_leg_kp", 500.), ("trained_leg_kp", 250.), ("unchanged_leg_kd", 5.),
    ("kind", "fixed_ppo_nominal_contact_damping_diagnostic"), ("only_changed_control_field", "leg_kd"),
    ("additional_training_updates", 1), ("reward_changes", True), ("reset_changes", True),
    ("force_filter_changes", True), ("training_at_deployment_gain_claimed", True),
    ("provenance_adapter_sha256", "0"*64), ("wrapper_sha256", "0"*64)])
def test_incorrect_or_misleading_diagnostic_fields_rejected(field, value):
    report = candidate_report(); report["contact_stability_diagnostic"][field] = value
    with pytest.raises(ValueError):
        stiffness.audit_stiffness_report(report)


def test_inherited_damping_only_fields_rejected():
    report = candidate_report(); report["contact_stability_diagnostic"]["deployment_leg_kd"] = 10.
    with pytest.raises(ValueError, match="Misleading"):
        stiffness.audit_stiffness_report(report)


def test_actual_gains_and_material_are_verified():
    for mutate in (lambda r: r["contact_material_audit"]["leg_pd_gain_readback"].update(kd_range=[5., 5.]),
        lambda r: r["contact_material_audit"].update(robot_runtime_friction_range=[.5, .5]),
        lambda r: r["contact_stability_diagnostic"]["input_sha256"].update(checkpoint="0"*64)):
        report = candidate_report(); mutate(report)
        with pytest.raises(ValueError):
            stiffness.audit_stiffness_report(report)


def test_other_control_changes_cannot_hide_behind_kp(candidate):
    path = candidate/"report.json"; report = json.loads(path.read_text())
    report["policy_export"]["leg_position_scale"] = .8; path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="beyond leg Kp"):
        stiffness.load_comparison(candidate=candidate)


def test_actor_export_roundoff_is_validated_not_a_control_difference(candidate):
    path = candidate/"report.json"; report = json.loads(path.read_text())
    report["policy_export"]["export_validation_max_abs_error"] = 2e-7; path.write_text(json.dumps(report))
    stiffness.load_comparison(candidate=candidate)
    report["policy_export"]["export_validation_max_abs_error"] = float("nan")
    with pytest.raises(ValueError, match="validation error"):
        stiffness.audit_stiffness_report(report)


def test_raw_binding_must_hold(candidate):
    path = candidate/"report.json"; report = json.loads(path.read_text())
    report["physics_trajectory_sha256"] = "0"*64; path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="trajectory SHA"):
        stiffness.load_comparison(candidate=candidate)


def test_no_overwrite_before_loading(tmp_path):
    args = stiffness.parser().parse_args(["--candidate", "missing", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Preserve existing"):
        stiffness.run(args)


def test_packaged_arrays_provenance(candidate, tmp_path, monkeypatch):
    monkeypatch.setattr(stiffness, "render_time_figure", lambda raw, summary, path: path.write_bytes(b"fixture"))
    output = tmp_path/"new"
    args = stiffness.parser().parse_args(["--candidate", str(candidate), "--output-dir", str(output)])
    summary = stiffness.run(args)
    assert summary["artifacts"]["recorded_states.npz"]["sha256"] == stiffness.sha256(output/"recorded_states.npz")
    with np.load(output/"recorded_states.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["kp250_time_s"], np.arange(1397)*.005)

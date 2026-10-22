"""CPU-only metadata/clock tests; synthetic diagnostic reports are not experiments."""
from pathlib import Path
import copy
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import summarize_ppo_contact_damping as damping

SOURCE = damping.ROOT/"outputs/ppo_smoothness_20261021_v2/action_rate_0p5/validation/mu0p700/isaac"


def report_fixture(kd):
    report = json.loads((SOURCE/"report.json").read_text())
    audit = {"schema_version": 1, "kind": "fixed_ppo_nominal_contact_damping_diagnostic", "diagnostic_only": True,
        "deployment_leg_kd": kd, "trained_leg_kd": 10., "unchanged_leg_kp": 500.,
        "checkpoint_sha256": damping.PINS["checkpoint"], "additional_training_updates": 0,
        "physics_dt_s": .005, "policy_dt_s": .02, "evaluation_contact_friction": .7,
        "trained_action_rate_weight": -.5, "training_at_deployment_gain_claimed": False,
        "reward_changes": False, "reset_changes": False, "force_filter_changes": False,
        "only_changed_control_field": "leg_kd" if kd == 5 else None,
        "legacy_evaluator_sha256": damping.BASE_EVALUATOR_SHA256,
        "training_dependency_sha256": dict(damping.TRAINING_DEPENDENCIES),
        "evaluation_dependency_sha256": dict(damping.EVALUATION_DEPENDENCIES),
        "training_evaluation_dependencies_identical": False,
        "wrapper_sha256": damping.DEFAULT_WRAPPER_SHA256,
        "input_sha256": {**damping.PINS, "run_status": "a"*64},
        "generated_evaluator_sha256": "b"*64,
        "hardware_ready": False, "statistical_robustness_claimed": False}
    report["contact_stability_diagnostic"] = audit
    report["evaluator_source_sha256"] = damping.DEFAULT_WRAPPER_SHA256
    report["policy_export"]["contact_stability_diagnostic"] = copy.deepcopy(audit)
    report["policy_export"]["leg_kd"] = kd
    if kd == 5:
        report["policy_export"]["task_profile"]["leg_kd"] = kd
    report["contact_material_audit"]["leg_pd_gain_readback"]["kd_range"] = [kd, kd]
    for current in (report["evaluation_contact_friction_override"], report["policy_export"]["evaluation_contact_friction_override"]):
        current.pop("only_changed_physics_field", None)
        current["only_changed_material_field"] = "contact_friction"
    return report


@pytest.fixture
def pair(tmp_path):
    for label, kd in (("kd10", 10.), ("kd5", 5.)):
        folder = tmp_path/label; folder.mkdir()
        (folder/"physics_trajectory.npz").symlink_to(SOURCE/"physics_trajectory.npz")
        (folder/"report.json").write_text(json.dumps(report_fixture(kd)))
    return tmp_path


def test_pair_metadata_and_original_native_metrics(pair):
    raw, summary = damping.load_comparison(pair)
    assert summary["checkpoint_sha256"] == damping.PINS["checkpoint"]
    assert summary["deployed_leg_kd"] == [10., 5.]
    assert summary["trained_leg_kd"] == 10 and summary["trained_action_rate_weight"] == -.5
    assert summary["new_training"] is False and summary["stability_success_claimed"] is False
    assert np.array_equal(raw["kd10"]["time_s"], np.arange(1397)*.005)
    core = summary["cases"]["kd5"]["windows"]["core"]
    assert core["samples"] == 513
    assert core["contact"]["measured_force_bearing_contact"]["both"]["sample_duty_fraction"] == pytest.approx(.341130604288499)
    assert core["tracking"]["xy_rmse_m"] == pytest.approx(.056221388688688805)
    assert summary["kd5_minus_kd10"]["core"]["both_contact_duty_percentage_point_change"] == 0
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("field,value", [("deployment_leg_kd", 10.), ("trained_leg_kd", 5.), ("trained_action_rate_weight", -.1),
    ("additional_training_updates", 1), ("reward_changes", True), ("reset_changes", True), ("force_filter_changes", True),
    ("training_at_deployment_gain_claimed", True), ("wrapper_sha256", "0"*64)])
def test_false_diagnostic_declarations_rejected(field, value):
    report = report_fixture(5.)
    report["contact_stability_diagnostic"][field] = value
    with pytest.raises(ValueError):
        damping.audit_report(report, 5., damping.DEFAULT_WRAPPER_SHA256)


def test_actual_gain_readback_not_configuration_only():
    report = report_fixture(5.)
    report["contact_material_audit"]["leg_pd_gain_readback"]["kd_range"] = [10., 10.]
    with pytest.raises(ValueError, match="readback"):
        damping.audit_report(report, 5., damping.DEFAULT_WRAPPER_SHA256)


def test_material_and_source_mutations_rejected():
    for mutate in (
        lambda r: r["contact_material_audit"].update(robot_runtime_friction_range=[.8, .8]),
        lambda r: r["contact_stability_diagnostic"]["input_sha256"].update(checkpoint="0"*64),
        lambda r: r.update(drift_module_sha256="0"*64)):
        report = report_fixture(5.); mutate(report)
        with pytest.raises(ValueError):
            damping.audit_report(report, 5., damping.DEFAULT_WRAPPER_SHA256)


def test_other_controller_field_cannot_hide_behind_kd_change(pair):
    path = pair/"kd5/report.json"; report = json.loads(path.read_text())
    report["policy_export"]["leg_position_scale"] = .8
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="beyond leg Kd"):
        damping.load_comparison(pair)


def test_actor_export_error_is_bounded_not_false_control_change(pair):
    path = pair/"kd5/report.json"; report = json.loads(path.read_text())
    report["policy_export"]["export_validation_max_abs_error"] = 2e-7
    path.write_text(json.dumps(report))
    _, summary = damping.load_comparison(pair)
    assert summary["cases"]["kd5"]["actor_export_validation_max_abs_error"] == 2e-7
    report["policy_export"]["export_validation_max_abs_error"] = 1e-2
    with pytest.raises(ValueError, match="validation error"):
        damping.audit_report(report, 5., damping.DEFAULT_WRAPPER_SHA256)


def test_bound_raw_hash_rejected_before_analysis(pair):
    path = pair/"kd5/report.json"; report = json.loads(path.read_text()); report["physics_trajectory_sha256"] = "0"*64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="trajectory SHA"):
        damping.load_comparison(pair)


def test_output_freshness_precedes_input_load(tmp_path):
    args = damping.parser().parse_args(["--input-root", "missing", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Preserve existing"):
        damping.run(args)


def test_packaging_preserves_recordings_and_hashes(pair, tmp_path, monkeypatch):
    monkeypatch.setattr(damping, "render_time_figure", lambda raw, summary, output: output.write_bytes(b"fixture"))
    output = tmp_path/"new"
    args = damping.parser().parse_args(["--input-root", str(pair), "--output-dir", str(output)])
    summary = damping.run(args)
    assert summary["artifacts"]["recorded_states.npz"]["sha256"] == damping.sha256(output/"recorded_states.npz")
    with np.load(output/"recorded_states.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["kd10_root_position_m"], archive["kd5_root_position_m"])
    assert (output/"summary.json").is_file()

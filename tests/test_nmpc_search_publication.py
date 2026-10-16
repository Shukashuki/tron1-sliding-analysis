"""Replay five published MuJoCo-only variants without assets, FK or physics."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from analyze_controller_pair import posture_diagnostic
from analyze_nmpc_nominal_development import configuration_sha256, contact_diagnostic
from controller_metrics import heading_signals, summarize_turn
from controller_references import get_reference
from nmpc_tracking_objective import JOINT_NAMES


FOLDER = ROOT/"results/nmpc_nominal_search_development_20261016_v1"
CASES = (
    ("residual_h15", 15, False, .11537216667873322,
     "30a9502f69caeb76b1b2cd1bbd2e8d76534229944a48f06c0aeef34756ef828f"),
    ("residual_h30", 30, False, .09163851640152391,
     "b5e03d9223703b8e35811862db92b955794477f4ba328c01a2a2a9e651b9fc81"),
    ("modelplan_h15", 15, True, .20316468036119653,
     "376f794b8785f703e533941e7a837e1b6ada682c3eda3b8ac02fc22775757847"),
    ("modelplan_h30", 30, True, .15374212335056991,
     "b00fb4ff81f74e87a76657b49de6d288d79352d5e1ceb0857da8ad36e99fcbd4"),
    ("low_leg_std", 30, False, .2058103989142875,
     "8905bd07f4e434d901987f1d619388cfea142adfa953172a41b5c84a9e80f9e0"),
)
LABELS = ("reference",) + tuple(case[0] for case in CASES)
STATE_FIELDS = ("time_s", "physics_dt_s", "joint_names", "position", "quaternion", "com_velocity",
                "angular_velocity", "joint_pos", "joint_vel", "wheel_pos")
GEOMETRY_FIELDS = ("wheel_quaternion_wxyz", "wheel_center_velocity_w")
SIGNAL_FIELDS = ("speed_m_s", "wheel_center_speed_m_s", "course_rad", "body_heading_rad",
    "wheel_rolling_plane_heading_rad", "root_directed_beta_rad", "root_course_to_wheel_gamma_rad",
    "wheel_center_course_to_wheel_gamma_rad", "root_beta_valid", "root_course_to_wheel_valid",
    "wheel_center_course_to_wheel_valid", "speed_threshold_m_s")
MEASURED_FIELDS = ("contact_count", "contact_force", "contact_slip", "nonwheel_ground_contact_count",
                   "nonwheel_ground_contact_force_norm_n", "held_action")
PRIVATE = ("/home/", "/data/", "/private/", "/tmp/", "zengxiong", "14.103.233.29", "117.50.226.213")


@pytest.fixture(scope="module")
def publication():
    summary = json.loads((FOLDER/"summary.json").read_text())
    with np.load(FOLDER/"development_trace.npz", allow_pickle=False) as archive:
        compact = {name: archive[name].copy() for name in archive.files}
    return summary, compact


def trace(compact, label):
    fields = STATE_FIELDS + (() if label == "reference" else MEASURED_FIELDS)
    return {field: compact[f"{label}_{field}"] for field in fields}


def signals(compact, label):
    return heading_signals(compact[f"{label}_quaternion"], compact[f"{label}_com_velocity"],
        compact[f"{label}_wheel_quaternion_wxyz"], compact[f"{label}_wheel_center_velocity_w"],
        speed_threshold_m_s=float(compact[f"{label}_speed_threshold_m_s"]))


def test_five_variant_bundle_hash_allowlist_and_portable_derived_payload(publication):
    summary, compact = publication
    assert {path.name for path in FOLDER.iterdir()} == {"summary.json", "development_trace.npz", "topdown.png"}
    assert summary["run_order"] == list(LABELS[1:]) and set(summary["runs"]) == set(LABELS[1:])
    assert summary["reference_spec"] == get_reference("drift_drl_traj4")
    assert summary["engine"] == "MuJoCo" and summary["fixed_mu"] == .7
    assert summary["status"] == "development_analysis_only"
    for key in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking",
                "controlled_execution_success_claimed", "pooled_success_statistics", "dynamics_stepping",
                "learned_inference", "height_or_flight_failure_gate", "heading_offset_is_mismatch"):
        assert summary[key] is False
    for name, key, digest in (
        ("development_trace.npz", "compact_trace_sha256", "79882971c384c2d6898a84c7537b7a2af8e1bea28e1ee26734a4dd00e8a262a2"),
        ("topdown.png", "topdown_sha256", "09fb76a70eae573f393d73594c02c2c729813c690b3c09edd60c89ef3c7e93b7")):
        assert hashlib.sha256((FOLDER/name).read_bytes()).hexdigest() == summary["artifacts"][key] == digest
    expected = {"schema_version", "wheel_order"}
    for label in LABELS:
        fields = STATE_FIELDS + GEOMETRY_FIELDS + SIGNAL_FIELDS + (() if label == "reference" else MEASURED_FIELDS)
        expected.update(f"{label}_{field}" for field in fields)
    assert set(compact) == expected
    assert int(compact["schema_version"]) == 1 and compact["wheel_order"].tolist() == ["left", "right"]
    for value in compact.values():
        assert not value.dtype.hasobject
        if value.dtype.kind in "biufc": assert np.isfinite(value).all()
        else: assert not any(token in str(value.tolist()) for token in PRIVATE)
    assert not any(token in (FOLDER/"summary.json").read_text() for token in PRIVATE)
    assert not any(name.startswith("isaac_") or name.startswith("reference_contact_") for name in compact)
    assert summary["input_sha256"]["motion_file"] == get_reference()["motion_npz_sha256"]
    assert summary["input_sha256"]["reference_file"] == get_reference()["robot_reference_npz_sha256"]
    for mapping in (summary["input_sha256"], summary["analysis_source_sha256"]):
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in mapping.values())


@pytest.mark.parametrize("label,horizon,enabled,rmse,source_trace_sha", CASES)
def test_full_native_clock_349_four_substep_holds_and_report_bindings(publication, label, horizon, enabled, rmse, source_trace_sha):
    summary, compact = publication
    time = compact[f"{label}_time_s"]
    assert len(time) == 1397 and float(compact[f"{label}_physics_dt_s"]) == .005
    np.testing.assert_allclose(time, np.arange(1397)*.005, atol=1e-9, rtol=0)
    held = compact[f"{label}_held_action"]
    assert held.shape == (1397, 8) and np.max(np.abs(held)) <= 1.
    np.testing.assert_array_equal(held[0], np.zeros(8))
    blocks = held[1:].reshape(349, 4, 8)
    np.testing.assert_array_equal(blocks, np.repeat(blocks[:, :1], 4, axis=1))
    np.testing.assert_array_equal(held, held.astype(np.float32).astype(float))
    # The compact exports held actions, not original solver/reward-phase rows.
    assert f"{label}_reference_frame_for_action" not in compact
    np.testing.assert_array_equal(compact[f"{label}_joint_names"], JOINT_NAMES)
    assert compact[f"{label}_position"].shape == (1397, 3)
    assert compact[f"{label}_joint_vel"].shape == (1397, 8)
    run = summary["runs"][label]
    assert set(run["input_sha256"]) == {"report.json", "rollout.npz", "solver_log.json"}
    assert run["input_sha256"]["rollout.npz"] == source_trace_sha
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in run["input_sha256"].values())
    diagnostic = run["solver_diagnostics"]
    assert diagnostic["decision_samples"] == diagnostic["timing"]["sample_count"] == 349
    assert diagnostic["counts"]["candidate_evaluations"] == diagnostic["counts"]["finite_candidates"]
    assert diagnostic["counts"]["candidate_evaluations"] == (68403 if enabled else 68054)
    for key in ("failed_candidates", "failed_scenarios", "failed_batch_prediction_calls", "optimization_failed", "fallback_used"):
        assert diagnostic["counts"][key] == 0
    assert not diagnostic["timing"]["real_time_verified"] and not diagnostic["timing"]["actuation_delay_injected"]


@pytest.mark.parametrize("label", LABELS)
def test_exact_heading_window_posture_and_measured_contact_replay(publication, label):
    summary, compact = publication
    measured, geometric = trace(compact, label), signals(compact, label)
    assert set(geometric) == set(SIGNAL_FIELDS)
    for name, value in geometric.items():
        np.testing.assert_array_equal(value, compact[f"{label}_{name}"])
    reference = trace(compact, "reference")
    np.testing.assert_allclose(reference["time_s"], np.arange(349)*.02, atol=1e-12, rtol=0)
    contacts = {} if label == "reference" else dict(wheel_ground_contact_count=measured["contact_count"],
        wheel_ground_force_w=measured["contact_force"], wheel_contact_lateral_slip_m_s=measured["contact_slip"])
    replay = summarize_turn(measured["time_s"], measured["position"], geometric,
        reference["time_s"], reference["position"][:, :2], signals(compact, "reference")["speed_m_s"],
        summary["reference_spec"]["evaluation_core_interval_s"], **contacts)
    assert replay == summary["windows"]["core"][label]
    assert replay["heading_offset_is_mismatch"] is False and replay["height_or_flight_failure_gate"] is False
    if label == "reference":
        assert len(reference["time_s"]) == 349 and reference["time_s"][-1] == 6.96
        assert replay["ground_contact_heading_geometry"] is None and replay["ground_contact_lateral_slip_m_s"] is None
    else:
        assert posture_diagnostic(measured) == summary["posture_diagnostics"][label]
        assert contact_diagnostic(measured, [1.44, 4.]) == summary["contact_diagnostics"][label]
        assert replay["samples_in_window"] == 513 and replay["coverage_complete"]
        assert replay["ground_contact_heading_geometry"]["measured_contact_gate_applied"]
        assert replay["heading_geometry"]["root_directed_beta"]["mean_absolute_deg"] > 1.


def test_effective_search_std_changes_are_retained_without_rewriting_reported_settings(publication):
    summary, _ = publication
    runs = summary["runs"]
    for label, horizon, enabled, _, _ in CASES:
        run, config = runs[label], runs[label]["effective_cem_config"]
        assert config["horizon_steps"] == horizon and config["control_blocks"] == 5
        assert (config["candidates"], config["iterations"], config["elite_count"], config["seed"]) == (64, 3, 8, 42)
        assert config["parameterization"] == "warm_start_residual_blocks"
        assert config["initial_std"] == ([.02]*6 if label == "low_leg_std" else [.04]*6) + [.35, .35]
        assert config["min_std"] == [.005]*6+[.03, .03]
        assert run["effective_cem_config_sha256"] == configuration_sha256(config)
        assert run["reported_settings_sha256"] == configuration_sha256(run["reported_settings"])
        assert run["prediction_horizon_s"] == pytest.approx(horizon*.02)
        assert run["control_block_durations_s"] == pytest.approx([horizon*.02/5]*5)
    base, changed = dict(runs["residual_h30"]["effective_cem_config"]), dict(runs["low_leg_std"]["effective_cem_config"])
    del base["initial_std"]; del changed["initial_std"]
    assert base == changed
    # The low-std episode uses a separately recorded settings file, not a
    # relabeled baseline; preserve both its original and effective values.
    assert runs["low_leg_std"]["reported_settings"]["nmpc"]["initial_std"][:6] == [.02]*6
    assert runs["residual_h30"]["reported_settings"]["nmpc"]["initial_std"][:6] == [.04]*6


def test_model_only_donor_identity_and_separate_offline_budget_are_not_lost(publication):
    summary, _ = publication
    runs = summary["runs"]
    for label in ("residual_h15", "residual_h30", "low_leg_std"):
        assert runs[label]["model_generated_nominal_plan"] == {"enabled": False}
    donor = runs["modelplan_h15"]["model_generated_nominal_plan"]
    assert donor == runs["modelplan_h30"]["model_generated_nominal_plan"]
    assert donor["enabled"] and donor["artifact_sha256"] == "dd94769db93707eee2cc78e9f3fb7ff943b9df6b81415c60b0d87b08fdfb9c2e"
    metadata, budget = donor["contract"], donor["offline_generation_budget"]
    assert budget == metadata["offline_generation_budget"]
    assert budget["source_control_decisions"] == 349 and budget["source_candidate_evaluations"] == 68054
    assert budget["source_predicted_physics_substeps"] == 8166480 == 68054*30*4
    assert "antecedent" in budget["scope"] and "additional" in budget["scope"]
    assert metadata["identity"]["reference_spec"] == summary["reference_spec"]
    assert metadata["identity"]["objective_contract"] == summary["objective_contract"]
    assert metadata["source_generation"]["nominal_plant_mu"] == .7
    assert metadata["source_generation"]["prediction_mu_scenarios"] == [.7]
    assert all(value is False for value in metadata["source_generation"]["declarations"].values())
    assert metadata["source_generation"]["source_artifact_sha256"]["trace"] == runs["residual_h30"]["input_sha256"]["rollout.npz"]
    assert "unverified" in metadata["action_contract"]["tail_rule"]


def test_correct_xy_screens_do_not_qualify_failing_or_mujoco_only_variants(publication):
    summary, _ = publication
    failed = set()
    for label, _, _, expected_rmse, _ in CASES:
        core = summary["windows"]["core"][label]
        assert core["reference_xy_rmse_m"] == pytest.approx(expected_rmse, abs=1e-14)
        screen = core["reference_xy_rmse_m"] > .15 or core["reference_xy_peak_m"] > .30
        assert core["large_xy_exploratory_screen"] is screen
        if screen: failed.add(label)
        assert not core["heading_offset_is_mismatch"] and not core["height_or_flight_failure_gate"]
    assert failed == {"modelplan_h15", "modelplan_h30", "low_leg_std"}
    assert not summary["frozen_benchmark"] and not summary["eligible_for_controller_ranking"]
    assert not summary["controlled_execution_success_claimed"] and not summary["sim2sim_comparison"]
    # A core XY pass does not silently certify the rest of an episode.
    assert summary["posture_diagnostics"]["modelplan_h30"]["max_base_tilt_deg"] > 90.
    assert summary["contact_diagnostics"]["modelplan_h30"]["first_nonwheel_ground_time_s"] == pytest.approx(4.24)

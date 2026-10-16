"""Asset-free replay of the published MuJoCo TVLQR pilot, not raw-action replay.

Clipping/timing/design budgets are source-bound aggregate declarations; their
private raw command/design archives are deliberately not published here.
"""
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
from analyze_nmpc_nominal_development import contact_diagnostic
from controller_metrics import heading_signals, summarize_turn
from controller_references import DEFAULT_CATALOG, get_reference

FOLDER = ROOT/"results/tvlqr_pilot_20261016_v1"
LABELS = ("reference", "NMPC_donor", "TVLQR")
STATE = ("time_s", "physics_dt_s", "joint_names", "position", "quaternion", "com_velocity", "angular_velocity", "wheel_pos")
MEASURED = ("contact_count", "contact_force", "contact_slip", "nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n")
GEOMETRY = ("wheel_quaternion_wxyz", "wheel_center_velocity_w")
SIGNALS = ("speed_m_s", "wheel_center_speed_m_s", "course_rad", "body_heading_rad", "wheel_rolling_plane_heading_rad",
    "root_directed_beta_rad", "root_course_to_wheel_gamma_rad", "wheel_center_course_to_wheel_gamma_rad",
    "root_beta_valid", "root_course_to_wheel_valid", "wheel_center_course_to_wheel_valid", "speed_threshold_m_s")
PRIVATE = ("/home/", "/data/", "/private/", "/tmp/", "file://", "zengxiong", "14.103.233.29", "117.50.226.213")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def publication():
    summary = json.loads((FOLDER/"summary.json").read_text())
    with np.load(FOLDER/"development_trace.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    return summary, data


def trace(data, label):
    return {key: data[f"{label}_{key}"] for key in STATE + (() if label == "reference" else MEASURED)}


def replay_signals(data, label):
    return heading_signals(data[f"{label}_quaternion"], data[f"{label}_com_velocity"],
        data[f"{label}_wheel_quaternion_wxyz"], data[f"{label}_wheel_center_velocity_w"],
        speed_threshold_m_s=float(data[f"{label}_speed_threshold_m_s"]))


def assert_tree(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected: assert_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): assert_tree(a, b)
    elif type(expected) in (float, int):
        assert actual == pytest.approx(expected, rel=1e-11, abs=1e-10)
    else:
        assert actual == expected


def test_exact_three_file_hashes_geometry_allowlist_and_privacy(publication):
    summary, data = publication
    hashes = {"summary.json": "c2929f95820af97d77935afded3b3fc1feaa9ebc70cd56faa484b03fffb8376d",
        "development_trace.npz": "acdbf487288d0dfb685ff50481f7ff397a959b25eedc2e8db2caa0f5472fc396",
        "topdown.png": "74558a235b8ac36a359dd1bae6743c49d4f51e2ec40c66e4f791423edbb3022c"}
    assert {path.name for path in FOLDER.iterdir()} == set(hashes)
    for name, digest in hashes.items(): assert sha(FOLDER/name) == digest
    for item in summary["artifacts"].values(): assert hashes[item["file"]] == item["sha256"]
    assert (FOLDER/"topdown.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    expected = {"schema_version", "wheel_order"}
    for label in LABELS:
        expected.update(f"{label}_{key}" for key in STATE + GEOMETRY + SIGNALS + (() if label == "reference" else MEASURED))
    assert set(data) == expected and data["wheel_order"].tolist() == ["left", "right"]
    assert not any(name.endswith(("qpos", "qvel", "actions", "held_action", "K", "feed")) for name in data)
    assert not any(token in (FOLDER/"summary.json").read_text() for token in PRIVATE)
    for array in data.values():
        assert not array.dtype.hasobject
        if array.dtype.kind in "biufc": assert np.isfinite(array).all()
        else: assert not any(token in str(array.tolist()) for token in PRIVATE)


def test_model_reference_design_donor_sources_and_complete_offline_budgets(publication):
    summary, _ = publication
    spec = get_reference()
    assert summary["reference_spec"] == spec
    assert summary["input_sha256"] == {"motion_file": spec["motion_npz_sha256"], "reference_file": spec["robot_reference_npz_sha256"],
        "model": "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"}
    assert summary["reference_catalog_sha256"] == sha(DEFAULT_CATALOG)
    for name, digest in summary["analysis_source_sha256"].items(): assert sha(ROOT/"scripts"/name) == digest
    donor, feedback = summary["runs"]["NMPC_donor"], summary["runs"]["TVLQR"]
    assert feedback["donor_identity"]["archive_sha256"] == donor["input_sha256"] == {
        "report.json": "8d0a3dc07e5871a8e707f19b5903a1fb1b56d2878d193ee03eb6d1864b89e459",
        "rollout.npz": "f0429603d0e2ed553cad3d07cbf946eeec65de9116fdb99dadb4fce53e4f018b",
        "solver_log.json": "cce00b34ee528ea0447d08672a0399362b4a300a93ff804caf91d00d5183060a"}
    assert feedback["design_sha256"] == "f8797156cf536bfee34ea5225dac96e329371ec86ca70588e735d40212c8e718"
    assert feedback["design_report_sha256"] == "013f382512fd52fcc6a1eed6344736532f2088528d1e8c8fea82879c8ccdee25"
    assert feedback["actuator_contract_sha256"] == donor["actuator_identity"]["policy_contract_sha256"]
    counts = donor["solver_diagnostics"]["counts"]
    assert counts["scenario_evaluations"] == counts["finite_candidates"] == 135062
    assert counts["failed_candidates"] == counts["optimization_failed"] == 0
    assert feedback["donor_identity"]["generation_budget"] == {"scenario_evaluations": 135062,
        "predicted_physics_substeps": 16207440, "plant_physics_substeps": 1396, "wall_s": 706.9858978170087}
    assert feedback["offline_design_budget"] == {"attempted_transition_calls": 25477, "completed_transition_calls": 25477,
        "predicted_physics_substeps": 101908, "wall_s": 18.036000271036755}
    assert 135062*30*4 == 16207440 and 349*73*4 == 101908
    sources = {item["source_name"]: item["sha256"] for item in feedback["reported_source_provenance"]["reported_source_hashes"]}
    for name in ("run_tvlqr_tracking.py", "design_tvlqr_nominal.py", "tvlqr_core.py"):
        assert sha(ROOT/"scripts"/name) == sources[name]
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values())


def test_native_clocks_and_heading_geometry_replay_without_assets_or_contact_invention(publication):
    _, data = publication
    for label in LABELS:
        n, dt = (349, .02) if label == "reference" else (1397, .005)
        np.testing.assert_allclose(data[f"{label}_time_s"], np.arange(n)*dt, atol=1e-10, rtol=0)
        assert float(data[f"{label}_physics_dt_s"]) == dt
        assert data[f"{label}_position"].shape == (n, 3) and data[f"{label}_wheel_pos"].shape == (n, 2, 3)
        signals = replay_signals(data, label)
        assert set(signals) == set(SIGNALS)
        for key, expected in signals.items():
            actual = data[f"{label}_{key}"]
            if actual.dtype.kind == "b": np.testing.assert_array_equal(actual, expected)
            else: np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert "reference_contact_count" not in data and "reference_contact_force" not in data


def test_core_xy_speed_and_grounded_sideslip_replay_from_public_geometry(publication):
    summary, data = publication
    reference = trace(data, "reference"); reference_speed = replay_signals(data, "reference")["speed_m_s"]
    for label in LABELS:
        measured = trace(data, label)
        contact = {} if label == "reference" else dict(wheel_ground_contact_count=measured["contact_count"],
            wheel_ground_force_w=measured["contact_force"], wheel_contact_lateral_slip_m_s=measured["contact_slip"])
        replay = summarize_turn(measured["time_s"], measured["position"], replay_signals(data, label), reference["time_s"],
            reference["position"][:, :2], reference_speed, summary["reference_spec"]["evaluation_core_interval_s"], **contact)
        assert_tree(replay, summary["windows"]["core"][label])
    core = summary["windows"]["core"]
    assert core["TVLQR"]["reference_xy_rmse_m"] == pytest.approx(.43853105159540506)
    assert core["TVLQR"]["reference_xy_peak_m"] == pytest.approx(.9061286500642337)
    assert core["TVLQR"]["large_xy_exploratory_screen"] is True
    assert core["NMPC_donor"]["reference_xy_rmse_m"] == pytest.approx(.05206128270397581)
    assert core["NMPC_donor"]["reference_xy_peak_m"] == pytest.approx(.09667200965191625)
    assert core["NMPC_donor"]["large_xy_exploratory_screen"] is False


def test_posture_contact_failure_and_source_bound_clipping_are_not_ranked_or_raw_action_replay(publication):
    summary, data = publication
    assert summary["run_order"] == ["NMPC_donor", "TVLQR"] and summary["engine"] == "MuJoCo"
    for name in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking", "matched_design_cost",
        "independent_trajectory_design", "controlled_execution_success_claimed", "height_or_flight_failure_gate",
        "heading_offset_is_mismatch", "dynamics_stepping", "learned_inference"):
        assert summary[name] is False
    for label in LABELS[1:]:
        measured = trace(data, label)
        assert_tree(posture_diagnostic(measured), summary["posture_diagnostics"][label])
        assert_tree(contact_diagnostic(measured, summary["reference_spec"]["evaluation_core_interval_s"]), summary["contact_diagnostics"][label])
    assert summary["posture_diagnostics"]["TVLQR"]["max_base_tilt_deg"] == pytest.approx(118.53586602922397)
    assert summary["contact_diagnostics"]["TVLQR"]["first_nonwheel_ground_time_s"] == pytest.approx(2.635)
    feedback = summary["runs"]["TVLQR"]
    # Source-bound aggregates only: public geometry cannot reconstruct commands
    # or independently verify the original clipping count or runtime latency.
    assert feedback["input_sha256"]["feedback_log.json"] == "400be05ec037afb07cece1b40e9308c236e241460a2a0180ac23cc59dc90151e"
    assert feedback["feedback_diagnostics"]["clipped_action_component_fraction"] == pytest.approx(.4326647564469914)
    assert feedback["design_cost_equals_15_term_objective"] is False
    assert feedback["donor_identity"]["learned_policy_used"] is False
    assert feedback["design_objective_contract"]["absolute_height_tracking"] is False
    assert feedback["design_objective_contract"]["original_reward_matched"] is False

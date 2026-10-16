"""Replay the public failed-I6 diagnostic without assets, physics or inference.

This case is intentionally separate from normal nominal-publication CASES.
Complete finite traces and generic file/config audits are not qualification
when repeated prediction failures and fallback actions dominate execution.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_controller_pair import posture_diagnostic, summarize_pair
from controller_metrics import heading_signals
from controller_references import get_reference

DIRECTORY = "controller_primary_nmpc_residual_h30_b5_i6_diagnostic_pair_v1"
NOTE = "2026-10-16-nmpc-progress-and-horizon.md"
HASHES = {
    "summary.json": "bb1cca505bf1adca94bf45150358ca373d767150b1d3f69a658962bdd336c537",
    "pair_trace.npz": "bb42f635c5386262d5678e9bbffe3cb6fe2e2d547a09136c9398c95aef545ce1",
    "topdown.png": "c57dcf0f686893920b50027d1894fec0d4fb9de3a7ce86cad67cc2f0e169f90b",
}


def publication():
    folder = ROOT / "results" / DIRECTORY
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "pair_trace.npz", allow_pickle=False) as archive:
        compact = {key: archive[key] for key in archive.files}
    return folder, summary, compact


def test_only_three_sanitized_derived_artifacts_and_exact_recorded_hashes():
    folder, summary, compact = publication()
    assert {path.name for path in folder.iterdir()} == set(HASHES)
    for name, expected in HASHES.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    assert summary["artifacts"]["compact_trace_sha256"] == HASHES["pair_trace.npz"]
    assert summary["artifacts"]["topdown_sha256"] == HASHES["topdown.png"]
    for path in (folder / "summary.json", ROOT / "results" / NOTE):
        for private in ("/home/", "/data/", "zengxiong", "14.103.233.29", "117.50.226.213"):
            assert private not in path.read_text()
    assert all(value.dtype.kind != "O" for value in compact.values())
    assert not any(name.startswith("reference_contact_") for name in compact)


def test_complete_original_clocks_named_states_and_no_synthetic_contact_labels():
    _, summary, compact = publication()
    assert summary["algorithm"] == "NMPC" and summary["fixed_mu"] == .7
    assert summary["reference_spec"] == get_reference("drift_drl_traj4")
    assert summary["core_coverage_complete"] and not summary["integration_only"]
    assert compact["wheel_order"].tolist() == ["left", "right"]
    for label, samples, dt in (("reference", 349, .02), ("isaac", 1397, .005), ("mujoco", 1397, .005)):
        time = compact[f"{label}_time_s"]
        assert len(time) == samples and time[0] == 0.
        np.testing.assert_allclose(np.diff(time), dt, atol=1e-9, rtol=0)
        assert time[-1] == pytest.approx((samples - 1) * dt, abs=1e-9)
        assert compact[f"{label}_position"].shape == (samples, 3)
        assert compact[f"{label}_joint_pos"].shape == (samples, 8)
        assert compact[f"{label}_wheel_pos"].shape == (samples, 2, 3)
        assert len(compact[f"{label}_joint_names"]) == 8
    assert summary["reference_contact_semantics"].startswith("Kinematic only")


def test_heading_xy_contact_masks_and_posture_replay_exactly():
    _, summary, compact = publication()
    traces, reconstructions = {}, {}
    for label in ("reference", "isaac", "mujoco"):
        keys = ("time_s", "position", "quaternion")
        if label != "reference": keys += ("contact_count", "contact_force", "contact_slip")
        traces[label] = {name: compact[f"{label}_{name}"] for name in keys}
        signals = heading_signals(compact[f"{label}_quaternion"], compact[f"{label}_com_velocity"],
            compact[f"{label}_wheel_quaternion_wxyz"], compact[f"{label}_wheel_center_velocity_w"],
            speed_threshold_m_s=float(compact[f"{label}_speed_threshold_m_s"]))
        for name, value in signals.items(): np.testing.assert_array_equal(value, compact[f"{label}_{name}"])
        reconstructions[label] = {"signals": signals}
    assert summarize_pair(summary["reference_spec"], traces, reconstructions) == summary["windows"]
    assert {label: posture_diagnostic(traces[label]) for label in ("isaac", "mujoco")} == summary["posture_diagnostics"]
    core = summary["windows"]["core"]
    for label in ("isaac", "mujoco"):
        assert core[label]["samples_in_window"] == 513 and core[label]["coverage_complete"]
        assert core[label]["ground_contact_heading_geometry"]["measured_contact_gate_applied"]
        assert not core[label]["heading_offset_is_mismatch"] and not core[label]["height_or_flight_failure_gate"]
        assert core[label]["large_xy_exploratory_screen"] == (
            core[label]["reference_xy_rmse_m"] > .15 or core[label]["reference_xy_peak_m"] > .30)
    assert core["reference"]["ground_contact_heading_geometry"] is None
    assert core["reference"]["ground_contact_lateral_slip_m_s"] is None


def test_i6_configuration_outcomes_and_original_input_source_pins_are_preserved():
    _, summary, _ = publication()
    config = summary["audit"]["nmpc_controller_identity"]["effective_cem_config"]
    assert {key: config[key] for key in ("horizon_steps", "control_blocks", "candidates", "iterations", "elite_count", "seed")} == {
        "horizon_steps": 30, "control_blocks": 5, "candidates": 64, "iterations": 6, "elite_count": 8, "seed": 42}
    assert config["parameterization"] == "warm_start_residual_blocks"
    assert config["initial_std"] == [.04] * 6 + [.35, .35]
    assert summary["audit"]["nmpc_controller_identity"]["model_generated_nominal_plan"] == {"enabled": False}
    assert summary["input_sha256"] == {
        "motion_file": "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254",
        "reference_file": "afc22f19b813cf10b3e3a90affb0f26f5a7743074f98b262394c69de786e9f6e",
        "model": "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513",
        "isaac_trace": "5f7a251fe47e694a03e477a327298ea7bae02b520d0c7e531e8e59e869f764c4",
        "isaac_report": "88c8cde0734687209ffb0487e055a16acd305a1c824e1de4a3e8dd43e3a4a01a",
        "mujoco_trace": "89b9ce395b9ec7115ff879f95f6092b1d9ba9ec2869c073eac39b5b56b6c7824",
        "mujoco_report": "9bfb16dc1a5da567d919d802b32792f4c0f0610e9396069cd264f3497758514b"}
    for engine, rmse, peak, speed, tilt, cem_sha in (
        ("isaac", .1829577456481854, .4807837824283452, .8905041846621282, 107.81026947521276,
         "8d2ac598d9e0e2765f9566954ba3fd0f443d939d2a5c6566125cca18668b35bd"),
        ("mujoco", .067312481937738, .11812278367690882, .9782487461750395, 16.453275511641447,
         "d7d50588af454dcd50d27e8e00bdcfd9b62a4803cf843dccac40f1a187d7a4ae")):
        core = summary["windows"]["core"][engine]
        assert core["reference_xy_rmse_m"] == pytest.approx(rmse, abs=1e-14)
        assert core["reference_xy_peak_m"] == pytest.approx(peak, abs=1e-14)
        assert core["mean_speed_retention_ratio"] == pytest.approx(speed, abs=1e-14)
        assert summary["posture_diagnostics"][engine]["max_base_tilt_deg"] == pytest.approx(tilt, abs=1e-12)
        sources = {item["source_name"]: item["sha256"] for item in
                   summary["reported_plant_source_provenance"][engine]["reported_source_hashes"]}
        assert sources["nmpc_shooting.py"] == cem_sha
        assert sources["nmpc_tracking_objective.py"] == "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860"
        assert sources["nmpc_native_batch.py"] == "475ef6fe1712783ea11af2cae64b5ff0d61321e6edee0a596f9977a7a69da8a1"
        assert sources["nmpc_batch.cpp"] == "deb3afe3367e11c28295e00e4a081e84968818070dcfb7044f06466761875ed2"


def test_runtime_failure_unknown_exit_and_invalid_comparison_cannot_be_hidden():
    _, summary, _ = publication()
    development = summary["development_status"]
    assert development["role"] == "nominal_I6_runtime_failure_diagnostic_not_a_valid_controller_comparison"
    for key in ("controller_frozen", "qualified_for_friction_grid", "algorithm_ranking_claimed", "plant_friction_sweep",
                "valid_nominal_controller_trial", "exact_controller_source_identity", "new_remote_source_upload_performed"):
        assert development[key] is False
    assert summary["eligible_for_controller_ranking"] is False
    assert "repeated failed predictions" in summary["eligibility_override_reason"]
    for key in ("controlled_execution_success_claimed", "hardware_ready", "pooled_success_statistics",
                "height_or_flight_failure_gate", "heading_offset_is_mismatch"):
        assert summary[key] is False
    failure = development["runtime_failure"]
    assert failure["isaac_actual_app_exit_code"] is None
    for key in ("isaac_actual_app_exit_code_observed", "underlying_exception_recorded", "failure_cause_established",
                "normal_controller_timing_comparison_valid"):
        assert failure[key] is False
    assert failure["isaac_original_launch_status"] == "running_status_preserved_after_supervisor_transport_interruption"
    assert {name: failure[name] for name in ("first_failed_batch_frame", "last_clean_decision_frame", "first_fallback_frame",
        "last_fallback_frame", "fallback_decisions", "candidate_evaluations", "finite_candidates", "failed_candidates",
        "batch_prediction_calls", "failed_batch_prediction_calls")} == {
        "first_failed_batch_frame": 137, "last_clean_decision_frame": 136, "first_fallback_frame": 138,
        "last_fallback_frame": 348, "fallback_decisions": 211, "candidate_evaluations": 134852,
        "finite_candidates": 53341, "failed_candidates": 81511, "batch_prediction_calls": 2930, "failed_batch_prediction_calls": 1690}
    assert failure["post_run_snapshot_and_runtime_hashes_unchanged"] is True
    assert failure["recovery_provenance_sha256"] == "4e3df37b56890a1d6bb9cabca704d121fe6171aef0d5100b0d6f4cb6aa48a8f2"
    assert failure["solver_log_sha256"] == {
        "isaac": "89dbb8b6408124d43c8a3cf426cd1b17d3215931a49bb48fe7f531b776eae120",
        "mujoco": "74dd4fa8b06744c9438f9fb4dce2d4f6228b0efc9e16e00a21f80454f711a9dd"}
    assert "Unknown" in failure["failure_cause"] and "do not infer" in failure["failure_cause"]
    assert "not evidence of faster or real-time" in failure["timing_caveat"]


def test_failure_stays_separate_from_normal_nominal_cases_and_docs_are_current():
    from test_primary_nominal_publications import CASES
    assert DIRECTORY not in {directory for _, directory, _ in CASES}
    note = (ROOT / "results" / NOTE).read_text()
    readme = (ROOT / "README.md").read_text()
    assert "H30/B5 check remains in progress" not in note
    assert "211" in note and "unknown" in note.lower()
    assert "not normal six-iteration timing" in note
    assert "It has not been deployed" in note
    assert f"({DIRECTORY}/topdown.png)" in note
    assert "six-iteration H30/B5 diagnostic" in readme

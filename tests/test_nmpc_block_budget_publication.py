"""Asset-free replay of the four published nominal MuJoCo block/budget cases."""
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


FOLDER = ROOT/"results/nmpc_nominal_block_budget_20261016_v1"
NOTE = ROOT/"results/2026-10-16-nmpc-block-budget.md"
CASES = (
    ("b5_i3", 5, 3, .09163851640152391, .12520633892938793,
     "b5e03d9223703b8e35811862db92b955794477f4ba328c01a2a2a9e651b9fc81"),
    ("b10_i3", 10, 3, .170030215967865, .30637733265429923,
     "8408c44fb3ac04971f8ff2dce81ec648614b30b393fbb5eb1c16c3ee2aae713f"),
    ("b5_i6", 5, 6, .067312481937738, .11812278367690882,
     "89b9ce395b9ec7115ff879f95f6092b1d9ba9ec2869c073eac39b5b56b6c7824"),
    ("b10_i6", 10, 6, .05206128270397581, .09667200965191625,
     "f0429603d0e2ed553cad3d07cbf946eeec65de9116fdb99dadb4fce53e4f018b"),
)
LABELS = ("reference",) + tuple(case[0] for case in CASES)
STATE = ("time_s", "physics_dt_s", "joint_names", "position", "quaternion", "com_velocity",
         "angular_velocity", "joint_pos", "joint_vel", "wheel_pos")
GEOMETRY = ("wheel_quaternion_wxyz", "wheel_center_velocity_w")
SIGNALS = ("speed_m_s", "wheel_center_speed_m_s", "course_rad", "body_heading_rad",
           "wheel_rolling_plane_heading_rad", "root_directed_beta_rad", "root_course_to_wheel_gamma_rad",
           "wheel_center_course_to_wheel_gamma_rad", "root_beta_valid", "root_course_to_wheel_valid",
           "wheel_center_course_to_wheel_valid", "speed_threshold_m_s")
MEASURED = ("contact_count", "contact_force", "contact_slip", "nonwheel_ground_contact_count",
            "nonwheel_ground_contact_force_norm_n", "held_action")
PRIVATE = ("/home/", "/data/", "/private/", "/tmp/", "file://", "zengxiong",
           "14.103.233.29", "117.50.226.213")


@pytest.fixture(scope="module")
def publication():
    summary = json.loads((FOLDER/"summary.json").read_text())
    with np.load(FOLDER/"development_trace.npz", allow_pickle=False) as archive:
        compact = {name: archive[name].copy() for name in archive.files}
    return summary, compact


def trace(compact, label):
    return {field: compact[f"{label}_{field}"] for field in STATE + (() if label == "reference" else MEASURED)}


def signals(compact, label):
    return heading_signals(compact[f"{label}_quaternion"], compact[f"{label}_com_velocity"],
        compact[f"{label}_wheel_quaternion_wxyz"], compact[f"{label}_wheel_center_velocity_w"],
        speed_threshold_m_s=float(compact[f"{label}_speed_threshold_m_s"]))


def test_three_file_bundle_hashes_strict_derived_field_allowlist_and_privacy(publication):
    summary, compact = publication
    assert {path.name for path in FOLDER.iterdir()} == {"summary.json", "development_trace.npz", "topdown.png"}
    for name, key, digest in (
        ("summary.json", None, "3df107c1bf22518fb408d9e44ef6abcec977831858a400d6385dbd9af472dd7c"),
        ("development_trace.npz", "compact_trace_sha256", "04ee3083df5e59d7a60add6e1ecb1e42d0d23f7788aa0356c08410636513b4bd"),
        ("topdown.png", "topdown_sha256", "1ffc2068c93d678bef014a507df7b5d676d6a3623ddbeb2d9db411798c2a124a"),
    ):
        assert hashlib.sha256((FOLDER/name).read_bytes()).hexdigest() == digest
        if key is not None: assert summary["artifacts"][key] == digest
    assert summary["run_order"] == list(LABELS[1:]) and set(summary["runs"]) == set(LABELS[1:])
    assert summary["reference_spec"] == get_reference("drift_drl_traj4")
    assert summary["engine"] == "MuJoCo" and summary["fixed_mu"] == .7
    assert summary["status"] == "development_analysis_only"
    expected = {"schema_version", "wheel_order"}
    for label in LABELS:
        expected.update(f"{label}_{field}" for field in STATE + GEOMETRY + SIGNALS
                        + (() if label == "reference" else MEASURED))
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


@pytest.mark.parametrize("label,blocks,iterations,rmse,peak,source_trace", CASES)
def test_full_native_clock_actions_input_hashes_and_finite_search_budget(publication, label, blocks, iterations, rmse, peak, source_trace):
    summary, compact = publication
    measured = trace(compact, label)
    np.testing.assert_allclose(measured["time_s"], np.arange(1397)*.005, atol=1e-9, rtol=0)
    assert float(measured["physics_dt_s"]) == .005
    assert measured["position"].shape == (1397, 3) and measured["joint_vel"].shape == (1397, 8)
    np.testing.assert_array_equal(measured["joint_names"], JOINT_NAMES)
    held = measured["held_action"]
    assert held.shape == (1397, 8) and np.max(np.abs(held)) <= 1.
    np.testing.assert_array_equal(held[0], np.zeros(8))
    holds = held[1:].reshape(349, 4, 8)
    np.testing.assert_array_equal(holds, np.repeat(holds[:, :1], 4, axis=1))
    np.testing.assert_array_equal(held, held.astype(np.float32).astype(float))
    run = summary["runs"][label]
    assert set(run["input_sha256"]) == {"report.json", "rollout.npz", "solver_log.json"}
    assert run["input_sha256"]["rollout.npz"] == source_trace
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in run["input_sha256"].values())
    diagnostic = run["solver_diagnostics"]; counts = diagnostic["counts"]
    assert diagnostic["decision_samples"] == diagnostic["timing"]["sample_count"] == 349
    budget = 68054 if iterations == 3 else 135062
    assert counts["candidate_evaluations"] == counts["finite_candidates"] == counts["scenario_evaluations"] == budget
    assert budget == 349*(3+64*iterations)-1
    assert budget*30*4 == (8166480 if iterations == 3 else 16207440)
    assert counts["batch_prediction_calls"] == (2093 if iterations == 3 else 3140)
    for key in ("failed_candidates", "failed_scenarios", "failed_batch_prediction_calls", "optimization_failed",
                "fallback_used", "initial_candidate_generation_failed"):
        assert counts[key] == 0
    assert counts["warm_start_used"] == 348
    assert diagnostic["residual_fixed_baseline_flag"] == {"present_steps": 349, "true_steps": 349, "missing_steps": 0}
    assert not diagnostic["timing"]["real_time_verified"] and not diagnostic["timing"]["actuation_delay_injected"]
    assert diagnostic["timing"]["all_calls_above_20ms"] and not diagnostic["timing"]["deadline_enforced"]


@pytest.mark.parametrize("label", LABELS)
def test_exact_heading_core_metrics_posture_and_contact_replay(publication, label):
    summary, compact = publication
    measured, geometric = trace(compact, label), signals(compact, label)
    assert set(geometric) == set(SIGNALS)
    for name, value in geometric.items():
        np.testing.assert_array_equal(value, compact[f"{label}_{name}"])
    reference = trace(compact, "reference")
    np.testing.assert_allclose(reference["time_s"], np.arange(349)*.02, atol=1e-12, rtol=0)
    assert len(reference["time_s"]) == 349 and reference["time_s"][-1] == 6.96
    contacts = {} if label == "reference" else dict(wheel_ground_contact_count=measured["contact_count"],
        wheel_ground_force_w=measured["contact_force"], wheel_contact_lateral_slip_m_s=measured["contact_slip"])
    replay = summarize_turn(measured["time_s"], measured["position"], geometric,
        reference["time_s"], reference["position"][:, :2], signals(compact, "reference")["speed_m_s"],
        [1.44, 4.], **contacts)
    assert replay == summary["windows"]["core"][label]
    assert not replay["heading_offset_is_mismatch"] and not replay["height_or_flight_failure_gate"]
    assert replay["coverage_complete"] and replay["samples_in_window"] == (129 if label == "reference" else 513)
    if label == "reference":
        assert replay["ground_contact_heading_geometry"] is None and replay["ground_contact_lateral_slip_m_s"] is None
    else:
        assert posture_diagnostic(measured) == summary["posture_diagnostics"][label]
        assert contact_diagnostic(measured, [1.44, 4.]) == summary["contact_diagnostics"][label]
        assert replay["ground_contact_heading_geometry"]["measured_contact_gate_applied"]
        assert replay["heading_geometry"]["root_directed_beta"]["mean_absolute_deg"] > 1.
        assert summary["posture_diagnostics"][label]["max_base_tilt_deg"] < 20.
        assert summary["contact_diagnostics"][label]["full_clip_nonwheel_ground_fraction"] == 0.


def test_complete_two_by_two_effective_knobs_and_no_extra_donor_budget(publication):
    summary, _ = publication
    normalized = []
    for label, blocks, iterations, _, _, _ in CASES:
        run = summary["runs"][label]; config = dict(run["effective_cem_config"])
        assert (config["horizon_steps"], config["control_blocks"], config["iterations"]) == (30, blocks, iterations)
        assert (config["candidates"], config["elite_count"], config["seed"]) == (64, 8, 42)
        assert config["parameterization"] == "warm_start_residual_blocks"
        assert config["initial_std"] == [.04]*6+[.35, .35]
        assert config["min_std"] == [.005]*6+[.03, .03]
        assert config["reference_convention"] == "held_pre_action" and config["smoothing"] == 0.
        assert config["scenario_reduction"] == "worst" and config["fallback_action"] == "previous"
        assert run["effective_cem_config_sha256"] == configuration_sha256(config)
        assert run["reported_settings_sha256"] == configuration_sha256(run["reported_settings"])
        assert run["reported_settings"]["prediction_mu_scenarios"] == [.7]
        # Preserve source settings defaults; CLI overrides are the effective knobs.
        assert run["reported_settings"]["nmpc"]["horizon_steps"] == 15
        assert run["prediction_horizon_s"] == .6
        assert run["control_block_durations_s"] == [.6/blocks]*blocks
        assert run["model_generated_nominal_plan"] == {"enabled": False}
        assert run["actuator_identity"]["control_dt_s"] == .02
        assert run["actuator_identity"]["physics_dt_s"] == .005
        del config["control_blocks"]; del config["iterations"]; normalized.append(config)
    assert all(config == normalized[0] for config in normalized)
    objective = summary["objective_contract"]
    assert len(objective["weights"]) == 15 and not objective["absolute_height_target"]
    assert not objective["reference_vertical_velocity_target"]
    assert objective["height_regularization"]["mode"] == "vertical_acceleration_only"
    assert objective["sideslip_target"] == "held reference body-frame course angle, NOT zero"


def test_actual_controller_source_versions_are_separately_pinned_not_identical(publication):
    summary, _ = publication
    expected = {
        "b5_i3": ("8d2ac598d9e0e2765f9566954ba3fd0f443d939d2a5c6566125cca18668b35bd",
                  "475ef6fe1712783ea11af2cae64b5ff0d61321e6edee0a596f9977a7a69da8a1"),
        "b10_i3": ("d7d50588af454dcd50d27e8e00bdcfd9b62a4803cf843dccac40f1a187d7a4ae",
                   "475ef6fe1712783ea11af2cae64b5ff0d61321e6edee0a596f9977a7a69da8a1"),
        "b5_i6": ("d7d50588af454dcd50d27e8e00bdcfd9b62a4803cf843dccac40f1a187d7a4ae",
                  "475ef6fe1712783ea11af2cae64b5ff0d61321e6edee0a596f9977a7a69da8a1"),
        "b10_i6": ("7c667ef0cbd32c58ce53882302e75f5d33487bcfea51c2f6ec3435d454a480ce",
                   "bb19aeea51cf03eedd87c0fb9e4561c11bc8ab5569cabf31c5465287a815fb76"),
    }
    for label, pair in expected.items():
        provenance = summary["runs"][label]["reported_plant_source_provenance"]
        sources = {row["source_name"]: row["sha256"] for row in provenance["reported_source_hashes"]}
        assert provenance["mujoco_version"] == "3.8.1"
        assert (sources["nmpc_shooting.py"], sources["nmpc_native_batch.py"]) == pair
        assert sources["run_nmpc_tracking.py"] == (
            "d121350b553c8a5622e6759a9e69b33e946e9193e01f442ebe3d333541743293" if label == "b5_i3"
            else "ccc14be5cca439e43b9211b2d98e4faf2e931202979b8ac832238c38d299d46f")
        assert sources["nmpc_tracking_objective.py"] == "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860"
        assert sources["nmpc_batch_objective.py"] == "1c9da90ae66778a71ceb2e6bb9d0dcf7f0a2259fafee39b37cbe958b79178fae"
        assert sources["nmpc_batch.cpp"] == "deb3afe3367e11c28295e00e4a081e84968818070dcfb7044f06466761875ed2"
    assert len({pair[0] for pair in expected.values()}) == 3


def test_xy_only_screens_do_not_merge_different_controllers_or_qualify_mujoco_only(publication):
    summary, _ = publication
    for label, _, _, rmse, peak, _ in CASES:
        core = summary["windows"]["core"][label]
        assert core["reference_xy_rmse_m"] == rmse and core["reference_xy_peak_m"] == peak
        assert core["xy_screen_limits_m"] == {"rmse": .15, "peak": .30}
        assert core["large_xy_exploratory_screen"] is (label == "b10_i3")
        assert core["large_xy_exploratory_screen"] is (rmse > .15 or peak > .30)
    for key in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking",
                "controlled_execution_success_claimed", "pooled_success_statistics", "dynamics_stepping",
                "learned_inference", "height_or_flight_failure_gate", "heading_offset_is_mismatch"):
        assert summary[key] is False
    assert summary["windows"]["core"]["reference"]["mean_speed_m_s"] == pytest.approx(1.0285870794256273)
    note = NOTE.read_text()
    assert not any(token in note for token in PRIVATE)
    for phrase in ("MuJoCo-only", "not frozen", "not a ranking", "different controllers", "upload remains pending",
                   "not identical source", "1.03 m/s", "not real-time", "no absolute-height"):
        assert phrase in note
    for target in re.findall(r"\]\(([^)]+)\)", note):
        assert not target.startswith(("http:", "https:")) and (NOTE.parent/target).is_file()

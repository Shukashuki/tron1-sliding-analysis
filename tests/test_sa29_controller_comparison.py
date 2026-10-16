"""Synthetic clock/score rejection tests; golden MJ recordings are never rerun."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_sa29_controller_comparison as a
from nmpc_tracking_objective import HeightSmoothTrackingObjective, JOINT_NAMES


@pytest.fixture(scope="module")
def bundle():
    directory = ROOT/"results/nmpc_sa29_friction_development_20261016_v1"
    return directory, a.read_json(directory/"summary.json"), a.load_arrays(directory/"development_trace.npz")


def test_existing_mujoco_bundle_identity_and_metrics_replay(bundle):
    directory, summary, data = bundle
    reference, _ = a.compact_case(data, "reference")
    actual = a.load_mujoco_bundle(directory, summary["reference_spec"], summary["input_sha256"], reference)
    assert len(actual[-2]) == len(actual[-1]) == 17
    assert actual[1] == a.MUJOCO_BUNDLE_SHA256


@pytest.mark.parametrize("prefix", ["reference", "mu0p700"])
def test_compact_clocks_are_sa29_not_primary(bundle, prefix):
    _, _, data = bundle
    trace, geometry = a.compact_case(data, prefix)
    assert len(trace["time_s"]) == (258 if prefix == "reference" else 1033)
    assert geometry["signals"]["speed_threshold_m_s"] == .25


@pytest.mark.parametrize("field", ["time_s", "position", "quaternion", "com_velocity", "angular_velocity", "wheel_pos",
    "contact_count", "contact_force", "contact_slip", "nonwheel_ground_contact_count"])
def test_compact_rejects_partial_fields(bundle, field):
    _, _, original = bundle
    data = dict(original); key = "mu0p700_"+field
    data[key] = data[key][:-1]
    with pytest.raises((ValueError, IndexError)):
        a.compact_case(data, "mu0p700")


@pytest.mark.parametrize("field", ["speed_m_s", "course_rad", "root_directed_beta_rad", "wheel_rolling_plane_heading_rad"])
def test_compact_signals_must_replay_from_measured_velocity_and_planes(bundle, field):
    _, _, original = bundle
    data = dict(original); key = "mu0p700_"+field
    data[key] = data[key].copy()+.001
    with pytest.raises(ValueError, match="Compact speed/heading"):
        a.compact_case(data, "mu0p700")


def test_published_bundle_is_not_replaced_with_recomputed_self_hashes(bundle, monkeypatch):
    directory, summary, data = bundle
    reference, _ = a.compact_case(data, "reference")
    monkeypatch.setattr(a, "sha256", lambda _: "0"*64)
    with pytest.raises(ValueError, match="byte-for-byte"):
        a.load_mujoco_bundle(directory, summary["reference_spec"], summary["input_sha256"], reference)


@pytest.fixture
def isaac_synthetic():
    n, f = a.SAMPLES, a.FRAMES
    q = np.tile([1.,0.,0.,0.], (n,1)); pos = np.zeros((n,3)); pos[:,0] = np.arange(n)*.005
    raw = {"time_s": np.arange(n)*.005, "physics_dt_s": np.array(.005), "policy_dt_s": np.array(.02),
        "root_pos": pos, "root_quaternion_wxyz": q,
        "base_com_velocity_w": np.zeros((n,3)), "base_angular_velocity_w": np.zeros((n,3)),
        "joint_pos": np.zeros((n,8)), "joint_vel": np.zeros((n,8)), "wheel_pos": np.zeros((n,2,3)),
        "action_joint_names": np.array(JOINT_NAMES), "wheel_ground_contact_count": np.zeros((n,2)),
        "wheel_ground_contact_force_w_n": np.zeros((n,2,3)), "wheel_lateral_contact_slip_m_s": np.zeros((n,2)),
        "nonwheel_ground_contact_count": np.zeros(n), "nonwheel_ground_contact_force_norm_n": np.zeros(n),
        "simulation_step_counter": np.arange(n)+123, "actions": np.zeros((f,8)), "compute_s": np.full(f,.03)}
    packed = np.zeros((f,111)); packed[:,:3] = pos[::4][:-1]; packed[:,3] = 1.
    raw["packed_controller_states"] = packed
    task = a.read_json(ROOT/"config/car_drift_height_smooth_task.json")
    contract = HeightSmoothTrackingObjective(task).contract(); weights = contract["weights"]
    upper = sum(v for v in weights.values() if v > 0)
    info = {"candidate_evaluations": 32, "finite_candidates": 32, "failed_candidates": 0,
        "scenario_evaluations": 32, "failed_scenarios": 0, "batch_prediction_calls": 1,
        "failed_batch_prediction_calls": 0, "optimization_failed": False, "fallback_used": False,
        "warm_start_used": False, "initial_candidate_generation_failed": False,
        "control_deadline_enforced": False, "real_time_verified": False, "timing_scope": "synthetic unit test"}
    score = {"raw_terms": dict.fromkeys(weights,0.), "weighted_terms": dict.fromkeys(weights,0.),
        "total_reward": 0., "reward_upper_bound": upper, "stage_cost": .02*upper, "base_com_vz": 0.}
    log = [{"frame": k, "reference_frame_for_reward": k, "action": [0.]*8,
        "raw_cem_action": [0.]*8, "compute_s": .03, "solver": copy.deepcopy(info),
        "actual_isaac_objective": copy.deepcopy(score), "same_state_dt_reward_abs_error": 0.,
        "coordinate_conversion_max_velocity_error": 0., "actual_dt_scaled_reward": 0.} for k in range(f)]
    report = {"objective_contract": contract, "compute_ms_median_p95_max": [30.]*3,
        "compute_over_control_dt_fraction": 1., "solver_failures": 0,
        "maximum_same_state_dt_reward_abs_error": 0., "maximum_coordinate_conversion_error": 0.,
        "real_time_verified": False, "elapsed_s": 100.}
    return report,raw,log


def test_synthetic_complete_isaac_diagnostics(isaac_synthetic):
    trace, result = a.audit_isaac_trace(*isaac_synthetic)
    assert len(trace["time_s"]) == 1033
    assert result["decision_samples"] == 258
    assert result["counts"]["optimization_failed"] == 0


@pytest.mark.parametrize("mutation", ["counter_gap", "counter_reset", "packed_pose", "packed_nan", "raw_action",
    "action_phase", "timing", "solver_count", "fallback", "weighted_score", "stage_cost", "action_memory",
    "absolute_height_replaces_acceleration", "conversion", "reward_error", "real_time", "missing_decision"])
def test_synthetic_isaac_rejects_invalid_evidence(isaac_synthetic, mutation):
    report,raw,log = isaac_synthetic
    if mutation == "counter_gap": raw["simulation_step_counter"][10:] += 1
    elif mutation == "counter_reset": raw["simulation_step_counter"][10:] -= 10
    elif mutation == "packed_pose": raw["packed_controller_states"][0,0] += 1
    elif mutation == "packed_nan": raw["packed_controller_states"][0,10] = np.nan
    elif mutation == "raw_action": log[0]["raw_cem_action"][0] = .1
    elif mutation == "action_phase": log[0]["reference_frame_for_reward"] = 1
    elif mutation == "timing": raw["compute_s"][0] = .01
    elif mutation == "solver_count": log[0]["solver"]["finite_candidates"] -= 1
    elif mutation == "fallback": log[0]["solver"]["fallback_used"] = True
    elif mutation == "weighted_score": log[0]["actual_isaac_objective"]["weighted_terms"]["action_rate_l2"] = 1.
    elif mutation == "stage_cost": log[0]["actual_isaac_objective"]["stage_cost"] += 1.
    elif mutation in ("action_memory", "absolute_height_replaces_acceleration"):
        name = "action_rate_l2" if mutation == "action_memory" else "height_vertical_acceleration"
        score = log[0]["actual_isaac_objective"]; score["raw_terms"][name] = .1
        score["weighted_terms"][name] = .1*report["objective_contract"]["weights"][name]
        score["total_reward"] = sum(score["weighted_terms"].values())
        score["stage_cost"] = .02*(score["reward_upper_bound"]-score["total_reward"])
    elif mutation == "conversion": log[0]["coordinate_conversion_max_velocity_error"] = .001
    elif mutation == "reward_error": log[0]["same_state_dt_reward_abs_error"] = .001
    elif mutation == "real_time": report["real_time_verified"] = True
    elif mutation == "missing_decision": log.pop()
    with pytest.raises(ValueError): a.audit_isaac_trace(report,raw,log)

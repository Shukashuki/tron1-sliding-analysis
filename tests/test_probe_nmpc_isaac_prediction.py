"""Pure measured-log audit tests; no controller, native worker or Isaac app."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from probe_nmpc_isaac_prediction import (
    FIXED_FRAMES, EXPECTED_SELECTED_FRAMES, EXPECTED_ARCHIVE, validate_trace,
    select_intervals, actual_sensor_history, measurement_at, course_error,
    archived_source_hash, analyze,
)
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES


def synthetic_trace():
    """Nonzero reset sensor memory and distinct substep forces catch zero fills."""
    n = 1397
    trace = {"time_s": np.arange(n)*.005, "simulation_step_counter": np.arange(12, 12+n),
             "root_pos": np.zeros((n, 3)), "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (n, 1)),
             "base_com_velocity_w": np.tile([1., .2, .3], (n, 1)), "base_angular_velocity_w": np.zeros((n, 3)),
             "joint_pos": np.zeros((n, 8)), "joint_vel": np.zeros((n, 8)), "wheel_pos": np.zeros((n, 2, 3)),
             "body_net_forces_w": np.arange(n*27, dtype=float).reshape(n, 9, 3),
             "applied_joint_torque_nm": np.zeros((n, 8)), "wheel_ground_contact_count": np.ones((n, 2), dtype=int),
             "actions": np.zeros((349, 8)), "packed_controller_states": np.zeros((349, 111)),
             "held_action": np.zeros((n, 8)), "reference_frame_for_action": np.r_[0, np.repeat(np.arange(349), 4)],
             "action_was_applied": np.r_[False, np.ones(n-1, dtype=bool)],
             "body_names": np.array(BODY_NAMES), "action_joint_names": np.array(JOINT_NAMES),
             "physics_dt_s": np.array(.005), "policy_dt_s": np.array(.02)}
    trace["packed_controller_states"][0, -1] = 1.
    trace["packed_controller_states"][0, 29:110] = 42.+np.arange(81)
    trace["body_net_forces_w"][0] = trace["packed_controller_states"][0, 29:56].reshape(9, 3)
    for k in range(1, 349):
        trace["packed_controller_states"][k, 29:110] = trace["body_net_forces_w"][4*k-np.arange(3)].reshape(-1)
    for sample in (57, 293, 469, 648):
        trace["wheel_ground_contact_count"][sample:, 0] = 1-trace["wheel_ground_contact_count"][sample:, 0]
    solver = [{"frame": k, "reference_frame_for_reward": k, "action": [0.]*8} for k in range(349)]
    return trace, solver


def test_selection_depends_only_on_fixed_clock_and_measured_contact_changes():
    trace, solver = synthetic_trace()
    validate_trace(trace, solver)
    selected = select_intervals(trace)
    assert tuple(row["policy_frame"] for row in selected) == EXPECTED_SELECTED_FRAMES
    assert len(set(row["policy_frame"] for row in selected)) == 16
    assert set(FIXED_FRAMES) <= set(row["policy_frame"] for row in selected)
    trace["xy_error_m"] = np.full(1397, np.nan)  # selection MUST NOT read errors
    assert select_intervals(trace) == selected


def test_boundary_contact_event_belongs_to_preceding_interval_and_no_replacement_if_missing():
    trace, _ = synthetic_trace()
    event = next(row for row in select_intervals(trace) if row.get("event_sample") == 648)
    assert event["policy_frame"] == 161  # 3.24 s is the END of action 161
    trace["wheel_ground_contact_count"][:] = 1
    with pytest.raises(ValueError, match="Missing unused"):
        select_intervals(trace)
    trace["wheel_ground_contact_count"][10, 0] = -1
    with pytest.raises(ValueError, match="nonnegative integers"):
        select_intervals(trace)


def test_history_keeps_actual_nonzero_reset_memory_and_native_order_is_not_imposed():
    trace, _ = synthetic_trace()
    initial = trace["packed_controller_states"][0, 29:110].reshape(3, 9, 3)
    np.testing.assert_array_equal(actual_sensor_history(trace, 0), initial)
    assert np.max(actual_sensor_history(trace, 0)) > 0
    one = actual_sensor_history(trace, 1)
    np.testing.assert_array_equal(one[0], trace["body_net_forces_w"][1])
    np.testing.assert_array_equal(one[1:], initial[:2])
    np.testing.assert_array_equal(actual_sensor_history(trace, 3), trace["body_net_forces_w"][[3, 2, 1]])
    np.testing.assert_array_equal(actual_sensor_history(trace, 4), trace["body_net_forces_w"][[4, 3, 2]])
    np.testing.assert_array_equal(actual_sensor_history(trace, 1396), trace["body_net_forces_w"][[1396, 1395, 1394]])
    one[:] = 0
    np.testing.assert_array_equal(actual_sensor_history(trace, 0), initial)
    for sample in (-1, True, 1397, 1.5):
        with pytest.raises(ValueError, match="sample"):
            actual_sensor_history(trace, sample)


@pytest.mark.parametrize("problem,match", [
    ("counter", "counter"), ("clock", "clock"), ("nan", "finite"),
    ("raw_action", "float32"), ("phase", "phase"), ("history_order", "newest-first"),
    ("history_zeroed", "sensor-history"), ("reset_flag", "first recorded"),
    ("quaternion", "unit wxyz"), ("missing", "Missing actual"),
])
def test_trace_refuses_invalid_clock_missing_channels_or_synthetic_history(problem, match):
    trace, solver = synthetic_trace()
    if problem == "counter": trace["simulation_step_counter"][100:] += 1
    elif problem == "clock": trace["time_s"][100] += .0001
    elif problem == "nan": trace["base_com_velocity_w"][100, 0] = np.nan
    elif problem == "raw_action": trace["actions"][1, 0] = .1234567890123
    elif problem == "phase": trace["reference_frame_for_action"][10] += 1
    elif problem == "history_order":
        h = trace["packed_controller_states"][10, 29:110].reshape(3, 9, 3)
        trace["packed_controller_states"][10, 29:110] = h[::-1].reshape(-1)
    elif problem == "history_zeroed": trace["packed_controller_states"][1, 29:110] = 0
    elif problem == "reset_flag": trace["packed_controller_states"][10, -1] = 1
    elif problem == "quaternion": trace["root_quaternion_wxyz"][10] = [0., 0., 0., 0.]
    elif problem == "missing": del trace["base_com_velocity_w"]
    with pytest.raises(ValueError, match=match):
        validate_trace(trace, solver)


def test_measured_root_link_position_and_COM_world_twist_are_not_transport_approximated():
    trace, _ = synthetic_trace()
    trace["root_pos"][3] = [2., -1., .75]
    trace["root_quaternion_wxyz"][3] = [-1., 0., 0., 0.]
    trace["base_angular_velocity_w"][3] = [.1, .2, .3]
    measurement = measurement_at(trace, 3)
    np.testing.assert_array_equal(measurement["root_pos"], [2., -1., .75])
    np.testing.assert_array_equal(measurement["root_com_lin_vel_w"], [1., .2, .3])
    np.testing.assert_array_equal(measurement["root_ang_vel_w"], [.1, .2, .3])
    np.testing.assert_array_equal(measurement["root_quat_wxyz"], [-1., 0., 0., 0.])
    measurement["root_pos"][:] = 0
    np.testing.assert_array_equal(trace["root_pos"][3], [2., -1., .75])


def test_course_error_wraps_directed_angle_and_masks_low_speed_without_zero_target():
    p = np.array([[-1., -.01, 0.], [0., 1., 0.], [.1, 0., 0.], [1., 1., 0.]])
    a = np.array([[-1., .01, 0.], [1., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    result = course_error(p, a)
    assert result["valid"] == [True, True, False, True]
    assert result["predicted_minus_measured_rad"][0] == pytest.approx(2*np.arctan(.01))
    assert result["predicted_minus_measured_rad"][1] == pytest.approx(np.pi/2)
    assert result["predicted_minus_measured_rad"][2] is None
    assert result["predicted_minus_measured_rad"][3] == 0.


def test_archived_source_roles_are_unique_not_basename_or_current_path_guesses():
    report = {"source_sha256": {"/remote/snapshot/scripts/target.py": "abc"}}
    assert archived_source_hash(report, "scripts/target.py") == "abc"
    with pytest.raises(ValueError, match="Missing/ambiguous"):
        archived_source_hash(report, "training/target.py")
    report["source_sha256"]["/other/scripts/target.py"] = "abc"
    with pytest.raises(ValueError, match="Missing/ambiguous"):
        archived_source_hash(report, "scripts/target.py")


def test_probe_preserves_existing_output_before_loading_private_inputs(tmp_path):
    pytest.importorskip("mujoco")
    output = tmp_path/"existing.json"
    output.write_text("original")
    with pytest.raises(FileExistsError, match="Preserving"):
        analyze(SimpleNamespace(output=output))
    assert output.read_text() == "original"


def test_actual_archived_log_channels_and_selection_if_available():
    episode = ROOT/"outputs/nmpc_isaac_primary_nominal_residual_20261016_v1"
    if not all((episode/name).is_file() for name in EXPECTED_ARCHIVE):
        pytest.skip("Private existing Isaac episode absent")
    with np.load(episode/"rollout.npz", allow_pickle=False) as z:
        trace = {name: z[name].copy() for name in z.files}
    solver = json.loads((episode/"solver_log.json").read_text())
    validate_trace(trace, solver)
    assert tuple(row["policy_frame"] for row in select_intervals(trace)) == EXPECTED_SELECTED_FRAMES
    assert np.max(np.abs(actual_sensor_history(trace, 0))) > 0

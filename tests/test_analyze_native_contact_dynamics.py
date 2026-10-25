"""Known rolling/sliding velocities, rotation invariance and stance-side logic."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_native_contact_dynamics as analysis


def fixture():
    n = 17
    time = np.arange(n+1)*.0025
    normal = np.zeros((n, 2)); normal[:, 0] = 200.
    raw_force = np.zeros((n+1, 1, 2, 3)); raw_force[:-1, 0, :, 2] = normal; raw_force[-1] = raw_force[-2]
    raw = {"time_s": time, "simulation_step_counter": np.arange(n+1),
        "wheel_ground_contact_force_w_n": raw_force,
        "root_linear_velocity_w_m_s": np.zeros((n+1, 1, 3)),
        "joint_names": np.array(analysis.JOINTS), "actions_clipped": np.zeros((n+1, 1, 8))}
    pos = np.broadcast_to([0., 0., .127], (n, 2, 3)).copy()
    point = np.zeros_like(pos); point[:, 1] = np.nan
    velocity = np.broadcast_to([1.27, .3, 0.], (n, 2, 3)).copy()
    omega = np.broadcast_to([0., 10., 0.], (n, 2, 3)).copy()
    q = np.broadcast_to([1., 0., 0., 0.], (n, 2, 4)).copy()
    ft = np.zeros_like(pos); ft[:, 0, 1] = -70.
    detail = {"time_before_step_s": time[:-1], "sim_counter": np.arange(1, n+1),
        "preceding_normal_load_n": normal, "wheel_quaternion_wxyz": q,
        "COM_velocity_w_m_s": velocity, "angular_velocity_w_rad_s": omega,
        "COM_position_w_m": pos, "preceding_contact_position_w_m": point,
        "native_friction_force_w_n": ft, "native_sum_patch_tangent_force_norm_n": np.linalg.norm(ft, axis=-1),
        "hub_side_angle_rad": np.zeros((n, 2)), "applied_tangent_force_w_n": np.zeros_like(pos),
        "applied_tangent_moment_COM_w_nm": np.zeros_like(pos)}
    return detail, raw


def test_wheel_spin_cancels_longitudinal_rolling_but_not_lateral_sliding():
    detail, raw = fixture(); result = analysis.contact_channels(detail, raw)
    np.testing.assert_allclose(result["contact_longitudinal_slip_m_s"][:, 0], 0., atol=1e-12)
    np.testing.assert_allclose(result["contact_lateral_slip_m_s"][:, 0], .3)
    np.testing.assert_allclose(result["aggregate_force_utilization"][:, 0], .5)
    assert np.isnan(result["aggregate_force_utilization"][:, 1]).all()
    assert np.isnan(result["contact_velocity_w_m_s"][:, 1]).all()
    detail["angular_velocity_w_rad_s"] *= 0.
    np.testing.assert_allclose(analysis.contact_channels(detail, raw)["contact_longitudinal_slip_m_s"][:, 0], 1.27)


def test_wheel_frame_components_invariant_under_world_yaw_rotation():
    detail, raw = fixture(); expected = analysis.contact_channels(detail, raw)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for name in ("COM_velocity_w_m_s", "angular_velocity_w_rad_s", "COM_position_w_m",
                 "preceding_contact_position_w_m", "native_friction_force_w_n"):
        detail[name] = detail[name] @ rotation.T
    detail["wheel_quaternion_wxyz"][:] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
    actual = analysis.contact_channels(detail, raw)
    for name in ("contact_longitudinal_slip_m_s", "contact_lateral_slip_m_s", "aggregate_force_utilization"):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("corruption", ["clock", "counter", "normal_force", "external_force"])
def test_wrong_time_force_or_non_native_law_refused(corruption):
    detail, raw = fixture()
    if corruption == "clock": detail["time_before_step_s"] = detail["time_before_step_s"]+.0025
    if corruption == "counter": detail["sim_counter"] += 1
    if corruption == "normal_force": detail["preceding_normal_load_n"][3, 0] += 1.
    if corruption == "external_force": detail["applied_tangent_force_w_n"][2, 0, 0] = .1
    with pytest.raises(AssertionError): analysis.contact_channels(detail, raw)


def test_allow_airborne_side_target_change_and_detect_loaded_side_command_jump():
    _, raw = fixture(); action = raw["actions_clipped"][:, 0]
    action[1:, [1, 3, 5]] = .8  # The right wheel is never carrying load.
    summary, values = analysis.command_transitions(raw, (0., .0425))
    np.testing.assert_array_equal(values["minimum_supporting_leg_target_jump_rms_rad"], np.zeros(3))
    assert summary["core_gap_onsets"] == 0
    action[9:17, [0, 2, 4]] = .3
    raw["wheel_ground_contact_force_w_n"][9, 0, 0, 2] = 0.
    summary, values = analysis.command_transitions(raw, (0., .0425))
    assert summary["gap_onset_physics_phase_0_through7"] == [0, 1, 0, 0, 0, 0, 0, 0]
    assert summary["immediate_gap_after_previously_supported_control_updates"] == 1
    assert values["minimum_supporting_leg_target_jump_rms_rad"][1] == pytest.approx(.3)


def test_auc_ties_missing_class_and_held_action_violation():
    assert analysis.empirical_auc([True, False], [1., 1.]) == .5
    assert analysis.empirical_auc([True, False], [2., 1.]) == 1.
    assert analysis.empirical_auc([True, True], [2., 1.]) is None
    _, raw = fixture(); raw["actions_clipped"][4, 0, 0] = .5
    with pytest.raises(AssertionError): analysis.command_transitions(raw, (0., .0425))

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("retime_hdm05", ROOT / "scripts/retime_hdm05_gmr.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
SETTINGS = json.loads((ROOT / "config/hdm05_rate_sweep.json").read_text())
GMR = json.loads((ROOT / "config/hdm05_gmr_experiment.json").read_text())


def fixture():
    time = np.linspace(0, 2, 241)
    n = len(time)
    qpos = np.zeros((n, 15))
    qpos[:, 1], qpos[:, 2], qpos[:, 3] = .4 * time, .8, 1
    leg_qids = np.array([7, 8, 9, 11, 12, 13])
    leg_vids = np.array([6, 7, 8, 10, 11, 12])
    qpos[:, leg_qids] = .1 * np.sin(time[:, None] * np.arange(1, 7)[None, :])
    qvel = np.zeros((n - 1, 14))
    qvel[:, :3] = np.diff(qpos[:, :3], axis=0) / np.diff(time)[:, None]
    qvel[:, leg_vids] = np.diff(qpos[:, leg_qids], axis=0) / np.diff(time)[:, None]
    human = np.repeat(qpos[:, None, :3], 5, axis=1)
    human[:, 1] += [0, .2, -.7]
    human[:, 2] += [0, -.2, -.7]
    human[:, 3] = human[:, 1] + [.1, 0, 0]
    human[:, 4] = human[:, 2] + [.1, 0, 0]
    sites = np.repeat(qpos[:, None, :3], 7, axis=1)
    sites[:, 5:7] = human[:, 1:3]
    source = {"time_s": time, "qpos": qpos, "qvel_intervals": qvel,
              "human_joint_names": np.array(["hip", "lfoot", "rfoot", "ltoes", "rtoes"]),
              "human_positions_robot_scale_m": human, "robot_site_positions_m": sites,
              "root_target_m": qpos[:, :3], "robot_wheel_heading_deg": np.zeros((n, 2)),
              "robot_wheel_bottom_m": np.zeros((n, 2))}
    source.update(mod.heading_features(time, qpos[:, :3], qpos[:, :3], human[:, 1:3], human[:, 3:5],
                                      source["robot_wheel_heading_deg"], source["robot_wheel_bottom_m"], GMR,
                                      wheel_centers=sites[:, 5:7]))
    geometry = {"com_position_m": qpos[:, :3].copy(), "mass_kg": np.array(22.273),
                "leg_position_indices": leg_qids, "leg_velocity_indices": leg_vids,
                "wheel_velocity_indices": np.array([9, 13])}
    return source, geometry


def test_native_physical_time_velocity_acceleration_energy_laws():
    motor_bounds = mod.load_motor_envelope(SETTINGS)
    source, geometry = fixture()
    report1, one = mod.retime_case(source, geometry, 1, GMR, SETTINGS, motor_bounds)
    report3, three = mod.retime_case(source, geometry, 3, GMR, SETTINGS, motor_bounds)
    np.testing.assert_allclose(three["time_s"], one["time_s"] / 3)
    for key in ("qvel_intervals", "com_velocity_m_s", "com_xy_momentum_kg_m_s"):
        np.testing.assert_allclose(three[key], 3 * one[key], atol=1e-11)
    for key in ("qacc_interior", "com_acceleration_m_s2", "com_xy_translation_energy_J"):
        np.testing.assert_allclose(three[key], 9 * one[key], atol=1e-9)
    assert not one["robot_valid_common"].any()
    assert three["robot_valid_common"].all()
    assert report3["duration_s"] == pytest.approx(report1["duration_s"] / 3)
    assert report3["fixed_phase_core"]["entry_native_frame_indices"] == report1["fixed_phase_core"]["entry_native_frame_indices"]
    assert report3["fixed_phase_core"]["exit_native_frame_indices"] == report1["fixed_phase_core"]["exit_native_frame_indices"]
    np.testing.assert_array_equal(source["qpos"], fixture()[0]["qpos"])
    assert not report3["physical_feasibility_assessed"]


def test_exact_pinned_motor_envelope_beyond_no_load_is_not_a_velocity_clip():
    bounds = mod.load_motor_envelope(SETTINGS)
    velocity = np.array([0., 15., 20., -20.])
    original = velocity.copy()
    lower, upper = bounds(velocity, 80, 80, 15)
    np.testing.assert_allclose(lower, [-80, -80, -80, 80 / 3])
    np.testing.assert_allclose(upper, [80, 0, -80 / 3, 80])
    np.testing.assert_array_equal(velocity, original)


def test_fixed_source_phase_turn_and_retention_do_not_change_with_rate():
    time = np.linspace(0, 2, 241)
    velocity = np.column_stack((np.cos(time), np.sin(time), np.zeros(len(time))))
    one = mod.fixed_phase_core(time, velocity, SETTINGS)
    three = mod.fixed_phase_core(time / 3, velocity * 3, SETTINGS)
    assert three["net_course_change_deg"] == pytest.approx(one["net_course_change_deg"])
    assert three["core_speed_p10_over_entry"] == pytest.approx(one["core_speed_p10_over_entry"])
    assert three["entry_speed_median_m_s"] == pytest.approx(3 * one["entry_speed_median_m_s"])


def test_no_bridging_masked_gaps_or_counting_single_sample_as_duration():
    time = np.arange(6) * .1
    windows = mod.intervals(time, [True, True, False, True, False, True])
    assert [row["duration_s"] for row in windows] == pytest.approx([.1, 0, 0])
    assert mod.max_window(time, [False] * 6) is None


def test_invalid_rate_and_short_duration_do_not_pass():
    source, geometry = fixture()
    bounds = mod.load_motor_envelope(SETTINGS)
    with pytest.raises(ValueError, match="positive"):
        mod.retime_case(source, geometry, 0, GMR, SETTINGS, bounds)
    report, _ = mod.retime_case(source, geometry, 6, GMR, SETTINGS, bounds)
    assert report["com_xy_speed_m_s"]["p95"] > 2
    assert not report["exploratory_screens"]["fast_high_angle_duration"]


def test_same_wall_clock_uses_retimed_states_and_holds_finished_reference():
    from render_hdm05_rate_sweep import sample_at_wall_time
    native = np.arange(5) * .5
    assert [sample_at_wall_time(native / factor, .5)[0] for factor in (1, 2, 3)] == [1, 2, 3]
    index, active = sample_at_wall_time(native / 3, .8)
    assert index == len(native) - 1
    assert not active
    with pytest.raises(ValueError):
        sample_at_wall_time(native, -.1)
    with pytest.raises(ValueError):
        sample_at_wall_time(native, float("nan"))

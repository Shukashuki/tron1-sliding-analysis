"""Synthetic, asset-free checks of planar controller-comparison diagnostics."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from controller_metrics import axial, heading_signals, summarize_turn, wrap


def yaw_quaternion(degrees):
    half = np.radians(np.asarray(degrees, dtype=float)) / 2.
    zero = np.zeros_like(half)
    return np.stack((np.cos(half), zero, zero, np.sin(half)), axis=-1)


def yaw_and_local_wheel_spin(yaw_deg, spin_deg):
    """Independent closed form for Rz(yaw) @ Ry(local axle spin)."""
    yaw, spin = np.radians(yaw_deg)/2., np.radians(spin_deg)/2.
    cy, sy, cs, ss = np.cos(yaw), np.sin(yaw), np.cos(spin), np.sin(spin)
    return np.stack((cy*cs, -sy*ss, cy*ss, sy*cs), axis=-1)


def planar_velocity(course_deg, speed):
    course, speed = np.broadcast_arrays(np.radians(course_deg), speed)
    return np.stack((speed*np.cos(course), speed*np.sin(course), np.zeros_like(speed)), axis=-1)


def make_signals(course_deg, *, body_yaw_deg=0., wheel_yaw_deg=(0., 0.),
                 wheel_course_deg=None, speed=1., wheel_speed=1.):
    course_deg = np.asarray(course_deg, dtype=float)
    n = len(course_deg)
    root_q = yaw_quaternion(np.broadcast_to(body_yaw_deg, (n,)))
    wheel_q = yaw_quaternion(np.broadcast_to(wheel_yaw_deg, (n, 2)))
    root_v = planar_velocity(course_deg, np.broadcast_to(speed, (n,)))
    wheel_course_deg = np.broadcast_to(course_deg[:, None] if wheel_course_deg is None else wheel_course_deg, (n, 2))
    wheel_v = planar_velocity(wheel_course_deg, np.broadcast_to(wheel_speed, (n, 2)))
    return heading_signals(root_q, root_v, wheel_q, wheel_v)


def simple_case(*, samples=11, duration=1.):
    time = np.linspace(0., duration, samples)
    position = np.column_stack((time, np.zeros(samples), np.ones(samples)))
    signals = make_signals(np.zeros(samples))
    return dict(time_s=time, root_position_w=position, signals=signals,
                reference_time_s=time.copy(), reference_xy=position[:, :2].copy(),
                reference_speed_m_s=np.ones(samples), interval_s=(0., duration))


def test_root_beta_directed_and_each_wheel_gamma_axial_use_distinct_velocities():
    signals = make_signals([35., 35.], body_yaw_deg=-15.,
                           wheel_yaw_deg=(10., -20.), wheel_course_deg=(45., -50.))
    np.testing.assert_allclose(np.degrees(signals["root_directed_beta_rad"]), 50.)
    np.testing.assert_allclose(np.degrees(signals["root_course_to_wheel_gamma_rad"]), [[25., 55.]]*2)
    np.testing.assert_allclose(np.degrees(signals["wheel_center_course_to_wheel_gamma_rad"]), [[35., -30.]]*2)
    assert signals["root_beta_valid"].all()
    assert signals["root_course_to_wheel_valid"].all()
    assert signals["wheel_center_course_to_wheel_valid"].all()


def test_reverse_heading_is_180_degree_root_beta_but_zero_axial_wheel_gamma():
    signals = make_signals([0., 180.], body_yaw_deg=[180., 0.], wheel_yaw_deg=((180., 0.), (0., 180.)))
    np.testing.assert_allclose(np.abs(np.degrees(signals["root_directed_beta_rad"])), 180.)
    np.testing.assert_allclose(signals["root_course_to_wheel_gamma_rad"], 0., atol=1e-14)
    np.testing.assert_allclose(signals["wheel_center_course_to_wheel_gamma_rad"], 0., atol=1e-14)


def test_wheel_spin_and_arbitrary_quaternion_sign_do_not_change_axial_signals():
    n = 5
    root_q = yaw_quaternion(np.linspace(-60., 60., n))
    wheel_yaw = np.array([[15., -35.]]*n)
    wheel_q = yaw_quaternion(wheel_yaw)
    root_v = planar_velocity(np.linspace(-25., 40., n), np.ones(n))
    wheel_v = planar_velocity(np.column_stack((np.linspace(-10., 30., n), np.linspace(35., -40., n))), 1.)
    expected = heading_signals(root_q, root_v, wheel_q, wheel_v)
    spins = np.array([[0., 37.], [180., -90.], [360., 720.], [-225., 47.], [1000., -1000.]])
    spun = yaw_and_local_wheel_spin(wheel_yaw, spins)
    signs = np.array([[1., -1.], [-1., 1.], [-1., -1.], [1., 1.], [-1., 1.]])
    actual = heading_signals(-root_q, root_v, spun*signs[..., None], wheel_v)
    for name in expected:
        if name == "wheel_rolling_plane_heading_rad":
            # Oriented axle may reverse, but this comparison is axial.
            np.testing.assert_allclose(axial(actual[name]-expected[name]), 0., atol=1e-14)
        else:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-14)


def test_degenerate_axes_and_low_xy_speeds_have_separate_explicit_validity_masks():
    n = 4
    root_q = yaw_quaternion(np.zeros(n)); wheel_q = np.tile([1., 0., 0., 0.], (n, 2, 1))
    # Body +X becomes vertical; left wheel +Y becomes vertical at the next row.
    root_q[0] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
    wheel_q[1, 0] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
    root_v = planar_velocity(np.zeros(n), [1., 1., .25, .249])
    root_v[3, 2] = 99.  # Large vertical speed does not repair low planar speed.
    wheel_v = planar_velocity(np.zeros((n, 2)), [[1., 1.], [1., 0.], [.25, .249], [1., 1.]])
    signals = heading_signals(root_q, root_v, wheel_q, wheel_v)
    np.testing.assert_array_equal(signals["root_beta_valid"], [False, True, True, False])
    np.testing.assert_array_equal(signals["root_course_to_wheel_valid"],
                                  [[True, True], [False, True], [True, True], [False, False]])
    np.testing.assert_array_equal(signals["wheel_center_course_to_wheel_valid"],
                                  [[True, True], [False, False], [True, False], [True, True]])
    assert np.isfinite(signals["root_directed_beta_rad"]).all()  # storage only, not a valid angle


@pytest.mark.parametrize("angles, axial_angle, expected_abs_mean", [
    ([179., -179.], False, 180.), ([89., -89.], True, 90.),
])
def test_signed_circular_mean_uses_correct_directed_or_axial_wrap(angles, axial_angle, expected_abs_mean):
    case = simple_case(samples=2)
    case["signals"] = make_signals(angles)
    result = summarize_turn(**case)
    if axial_angle:
        summary = result["heading_geometry"]["per_wheel"]["left"]["root_course_to_wheel_gamma"]
    else:
        summary = result["heading_geometry"]["root_directed_beta"]
    assert abs(summary["signed_circular_mean_deg"]) == pytest.approx(expected_abs_mean)
    assert summary["mean_absolute_deg"] == pytest.approx(abs(angles[0]))


def test_circular_mean_reports_none_when_resultant_is_undefined():
    case = simple_case(samples=2); case["signals"] = make_signals([0., 180.])
    result = summarize_turn(**case)
    assert result["heading_geometry"]["root_directed_beta"]["signed_circular_mean_deg"] is None
    case["signals"] = make_signals([0., 90.])
    result = summarize_turn(**case)
    assert result["heading_geometry"]["per_wheel"]["left"]["root_course_to_wheel_gamma"]["signed_circular_mean_deg"] is None


def test_signed_means_preserve_negative_turn_offsets_instead_of_absolute_values():
    case = simple_case(samples=2); case["signals"] = make_signals([-40., -20.])
    result = summarize_turn(**case)
    root = result["heading_geometry"]["root_directed_beta"]
    wheel = result["heading_geometry"]["per_wheel"]["left"]["root_course_to_wheel_gamma"]
    assert root["signed_circular_mean_deg"] == pytest.approx(-30.)
    assert wheel["signed_circular_mean_deg"] == pytest.approx(-30.)
    assert root["mean_absolute_deg"] == pytest.approx(30.)


def test_wrappers_keep_directional_and_axial_domains_distinct():
    np.testing.assert_allclose(wrap(np.radians([190., -190., 370.])), np.radians([-170., 170., 10.]))
    np.testing.assert_allclose(axial(np.radians([190., -190., 100., -100.])), np.radians([10., -10., -80., 80.]))


@pytest.mark.parametrize("interval, observed_end, complete", [
    ((0., 6.), 6., True), ((3., 5.), 6., True), ((0., 6.), 5.98, False),
    ((3., 5.), 4.98, False), ((0., 6.), 2., False),
])
def test_primary_and_supplemental_windows_do_not_turn_partial_coverage_into_pass(interval, observed_end, complete):
    case = simple_case(samples=301, duration=6.)
    selected = case["time_s"] <= observed_end + 1e-10
    for field in ("time_s", "root_position_w"):
        case[field] = case[field][selected]
    case["signals"] = make_signals(np.zeros(selected.sum()))
    case["interval_s"] = interval
    result = summarize_turn(**case)
    assert result["coverage_complete"] is complete
    if complete:
        assert result["reference_xy_rmse_m"] == pytest.approx(0.)
        assert result["large_xy_exploratory_screen"] is False
    else:
        for field in ("reference_xy_rmse_m", "reference_xy_peak_m", "large_xy_exploratory_screen",
                      "mean_speed_m_s", "net_course_change_deg", "heading_geometry"):
            assert result[field] is None


def test_one_native_sample_inside_bracketed_window_is_censored_not_zero_course():
    case = simple_case(samples=3)
    case["interval_s"] = (.4, .6)  # Native times 0,.5,1 bracket it, but only .5 is inside.
    result = summarize_turn(**case)
    assert result["samples_in_window"] == 1
    assert result["coverage_complete"] is False
    assert result["net_course_change_deg"] is None
    assert result["reference_xy_rmse_m"] is None
    assert result["large_xy_exploratory_screen"] is None


def test_xy_screen_ignores_height_flight_and_large_heading_offset():
    case = simple_case()
    case["root_position_w"][:, 2] = np.linspace(-1000., 1000., len(case["time_s"]))
    case["signals"] = make_signals(np.zeros(len(case["time_s"])), body_yaw_deg=170., wheel_yaw_deg=(80., -80.))
    case["wheel_ground_contact_count"] = np.zeros((len(case["time_s"]), 2))
    case["wheel_ground_force_w"] = np.zeros((len(case["time_s"]), 2, 3))
    result = summarize_turn(**case)
    assert result["large_xy_exploratory_screen"] is False
    assert result["heading_offset_is_mismatch"] is False
    assert result["height_or_flight_failure_gate"] is False
    assert result["heading_geometry"]["root_directed_beta"]["mean_absolute_deg"] == pytest.approx(170.)
    assert result["ground_contact_heading_geometry"]["contact_fraction_left_right"] == [0., 0.]
    # Opposite direction: perfect beta/gamma and any height cannot rescue XY drift.
    case["signals"] = make_signals(np.zeros(len(case["time_s"])))
    case["root_position_w"][:, 1] = .151
    assert summarize_turn(**case)["large_xy_exploratory_screen"] is True


def test_xy_peak_and_rmse_screen_thresholds_are_strict_and_independent():
    case = simple_case(samples=11)
    case["root_position_w"][:, 1] = .15
    result = summarize_turn(**case)
    assert result["reference_xy_rmse_m"] == pytest.approx(.15)
    assert result["large_xy_exploratory_screen"] is False
    case["root_position_w"][:, 1] = 0.
    case["root_position_w"][0, 1] = .301
    result = summarize_turn(**case)
    assert result["reference_xy_rmse_m"] < .15
    assert result["reference_xy_peak_m"] > .30
    assert result["large_xy_exploratory_screen"] is True


def test_contact_gate_requires_ground_count_and_positive_upward_force_per_wheel():
    case = simple_case(samples=6)
    case["signals"] = make_signals(np.arange(6)*10.)
    counts = np.array([[1, 0], [0, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    force = np.zeros((6, 2, 3)); force[:, :, 2] = [[10., 10.], [10., 20.], [5., -20.], [6., 4.], [0., 8.], [7., 5.]]
    force[4, 0, :2] = [1e6, 1e6]  # Horizontal force is not an upward-force gate.
    slip = np.array([[3., 999.], [999., 4.], [999., 999.], [4., 999.], [999., 3.], [0., 999.]])
    result = summarize_turn(**case, wheel_ground_contact_count=counts,
                            wheel_ground_force_w=force, wheel_contact_lateral_slip_m_s=slip)
    contact = result["ground_contact_heading_geometry"]
    assert contact["measured_contact_gate_applied"] is True
    assert result["heading_geometry"]["measured_contact_gate_applied"] is False
    assert contact["contact_fraction_left_right"] == pytest.approx([3/6, 2/6])
    left = contact["per_wheel"]["left"]["root_course_to_wheel_gamma"]
    right = contact["per_wheel"]["right"]["root_course_to_wheel_gamma"]
    assert left["valid_samples"] == 3 and right["valid_samples"] == 2
    assert left["mean_absolute_deg"] == pytest.approx((0.+30.+50.)/3)
    assert right["mean_absolute_deg"] == pytest.approx((10.+40.)/2)
    assert result["ground_contact_lateral_slip_m_s"] == pytest.approx({"left": np.sqrt(25/3), "right": np.sqrt(25/2)})


def test_contact_heading_masks_also_require_each_geometric_speed_mask():
    case = simple_case(samples=3)
    case["signals"] = make_signals([10., 20., 30.], speed=[1., 0., 1.], wheel_speed=[[1., 0.], [1., 1.], [0., 1.]])
    counts = np.ones((3, 2)); force = np.zeros((3, 2, 3)); force[:, :, 2] = 10.
    result = summarize_turn(**case, wheel_ground_contact_count=counts, wheel_ground_force_w=force)
    per_wheel = result["ground_contact_heading_geometry"]["per_wheel"]
    assert per_wheel["left"]["root_course_to_wheel_gamma"]["valid_samples"] == 2
    assert per_wheel["left"]["wheel_center_course_to_wheel_gamma"]["valid_samples"] == 2
    assert per_wheel["right"]["root_course_to_wheel_gamma"]["valid_samples"] == 2
    assert per_wheel["right"]["wheel_center_course_to_wheel_gamma"]["valid_samples"] == 2


def test_no_contact_slip_is_none_not_zero_and_missing_slip_stays_unavailable():
    case = simple_case(samples=3)
    counts = np.array([[1., 0.]]*3); force = np.zeros((3, 2, 3)); force[:, 0, 2] = 10.
    result = summarize_turn(**case, wheel_ground_contact_count=counts, wheel_ground_force_w=force,
                            wheel_contact_lateral_slip_m_s=np.zeros((3, 2)))
    assert result["ground_contact_lateral_slip_m_s"] == {"left": 0., "right": None}
    summary = result["ground_contact_heading_geometry"]["per_wheel"]["right"]["root_course_to_wheel_gamma"]
    assert summary["valid_samples"] == 0
    assert summary["signed_circular_mean_deg"] is None and summary["mean_absolute_deg"] is None
    result = summarize_turn(**case, wheel_ground_contact_count=counts, wheel_ground_force_w=force)
    assert result["ground_contact_lateral_slip_m_s"] is None
    result = summarize_turn(**case)
    assert result["ground_contact_heading_geometry"] is None


@pytest.mark.parametrize("censored", [False, True])
@pytest.mark.parametrize("kind", ["counts_only", "forces_only", "slip_without_gate"])
def test_missing_measured_contact_gate_is_never_silently_accepted(censored, kind):
    case = simple_case(samples=3)
    if censored:
        case["time_s"] = case["time_s"]*.5
    options = {"counts_only": dict(wheel_ground_contact_count=np.ones((3, 2))),
               "forces_only": dict(wheel_ground_force_w=np.zeros((3, 2, 3))),
               "slip_without_gate": dict(wheel_contact_lateral_slip_m_s=np.zeros((3, 2)))}[kind]
    with pytest.raises(ValueError, match="contact|gate"):
        summarize_turn(**case, **options)


@pytest.mark.parametrize("censored", [False, True])
@pytest.mark.parametrize("field", ["root_beta_valid", "root_course_to_wheel_valid", "wheel_center_course_to_wheel_valid"])
@pytest.mark.parametrize("bad_kind", ["integer", "nan"])
def test_validity_masks_must_be_real_booleans_even_for_censored_windows(censored, field, bad_kind):
    case = simple_case(samples=3)
    if censored:
        case["time_s"] *= .5
    mask = case["signals"][field]
    case["signals"][field] = np.ones(mask.shape, dtype=int) if bad_kind == "integer" else np.full(mask.shape, np.nan)
    with pytest.raises(ValueError, match="bool"):
        summarize_turn(**case)


@pytest.mark.parametrize("censored", [False, True])
@pytest.mark.parametrize("kind", ["negative_speed", "nan_angle", "bad_counts", "bad_force", "negative_slip", "nan_slip"])
def test_malformed_measurements_cannot_hide_behind_censoring(censored, kind):
    case = simple_case(samples=3)
    if censored:
        case["time_s"] *= .5
    options = dict(wheel_ground_contact_count=np.ones((3, 2)),
                   wheel_ground_force_w=np.zeros((3, 2, 3)),
                   wheel_contact_lateral_slip_m_s=np.zeros((3, 2)))
    if kind == "negative_speed":
        case["signals"]["speed_m_s"][0] = -1.
    elif kind == "nan_angle":
        case["signals"]["root_directed_beta_rad"][0] = np.nan
    elif kind == "bad_counts":
        options["wheel_ground_contact_count"][0, 0] = .5
    elif kind == "bad_force":
        options["wheel_ground_force_w"] = np.zeros((3, 2))
    elif kind == "negative_slip":
        options["wheel_contact_lateral_slip_m_s"][0, 0] = -1.
    else:
        options["wheel_contact_lateral_slip_m_s"][0, 0] = np.nan
    with pytest.raises(ValueError):
        summarize_turn(**case, **options)


@pytest.mark.parametrize("counts", [np.full((3, 2), -.5), np.full((3, 2), .5)])
def test_ground_contact_counts_must_be_nonnegative_integers(counts):
    with pytest.raises(ValueError, match="nonnegative integers"):
        summarize_turn(**simple_case(samples=3), wheel_ground_contact_count=counts,
                       wheel_ground_force_w=np.zeros((3, 2, 3)))


@pytest.mark.parametrize("direction", [-1., 1.])
def test_native_course_unwrap_is_valid_for_fast_turn_but_not_across_low_speed_break(direction):
    case = simple_case(samples=4)
    courses = direction*np.array([170., 179., -179., -170.])
    case["signals"] = make_signals(courses)
    assert summarize_turn(**case)["net_course_change_deg"] == pytest.approx(direction*20.)
    case["signals"] = make_signals(courses, speed=[1., 0., 1., 1.])
    assert summarize_turn(**case)["net_course_change_deg"] is None


def test_speed_retention_is_measured_without_becoming_a_failure_gate():
    case = simple_case(samples=3)
    case["reference_speed_m_s"][:] = 2.
    case["signals"] = make_signals([0., 0., 0.], speed=[1., 2., 0.])
    result = summarize_turn(**case)
    assert result["mean_speed_m_s"] == pytest.approx(1.)
    assert result["mean_speed_retention_ratio"] == pytest.approx(.5)
    assert result["fraction_below_80pct_reference_speed"] == pytest.approx(2/3)
    assert result["large_xy_exploratory_screen"] is False
    case["reference_speed_m_s"][:] = 0.
    assert summarize_turn(**case)["mean_speed_retention_ratio"] is None


@pytest.mark.parametrize("field", ["time_s", "reference_time_s"])
def test_invalid_native_or_reference_clock_is_rejected(field):
    case = simple_case(samples=3); case[field][1] = case[field][0]
    with pytest.raises(ValueError, match="strictly increasing"):
        summarize_turn(**case)


@pytest.mark.parametrize("threshold", [0., -1., np.nan])
def test_heading_threshold_and_nonunit_quaternions_are_rejected(threshold):
    q = np.tile([1., 0., 0., 0.], (2, 1)); wq = np.tile([1., 0., 0., 0.], (2, 2, 1))
    v = np.ones((2, 3)); wv = np.ones((2, 2, 3))
    with pytest.raises(ValueError, match="positive speed threshold"):
        heading_signals(q, v, wq, wv, speed_threshold_m_s=threshold)
    q[0] *= 2.
    with pytest.raises(ValueError, match="unit wxyz"):
        heading_signals(q, v, wq, wv)

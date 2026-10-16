"""Controller-independent planar turning diagnostics on native physical clocks.

Heading separation is a feature, NOT a mismatch or a zero-slip target. Wheel
rolling-plane headings are axial (modulo pi); root forward direction is
directed (modulo 2*pi). Contact-point slip is a different measured quantity.
Reference geometry has no measured contact gate and is never called grounded.
"""
from __future__ import annotations

import numpy as np

from nmpc_tracking_objective import quaternion_rotate


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def axial(angle):
    return .5 * np.arctan2(np.sin(2. * angle), np.cos(2. * angle))


def _array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite shape {shape}")
    return value


def _quaternions(value, shape, name):
    value = _array(value, shape, name)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.allclose(norm, 1., atol=1e-5, rtol=0):
        raise ValueError(f"{name}: expected unit wxyz quaternions")
    return value / norm


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind != "b":
        raise ValueError(f"{name}: expected boolean mask of shape {shape}")
    return value


def heading_signals(root_quaternion_wxyz, root_com_velocity_w,
                    wheel_quaternion_wxyz, wheel_center_velocity_w,
                    *, speed_threshold_m_s=.25):
    """Root and per-wheel geometric angles; no contact or success inference.

    Wheel local +Y is the TRON1 axle. Its cross product with world +Z is
    projected into XY to define the rolling plane; wheel spin and quaternion
    sign therefore do not change the axial heading. Nearly vertical axles or
    body forward axes are marked unavailable instead of fabricating a heading.
    """
    root_velocity = np.asarray(root_com_velocity_w, dtype=float)
    if root_velocity.ndim != 2 or root_velocity.shape[1] != 3:
        raise ValueError("root_com_velocity_w must have shape (samples,3)")
    n = len(root_velocity)
    if n < 2 or not np.isfinite(speed_threshold_m_s) or speed_threshold_m_s <= 0:
        raise ValueError("Need at least two samples and a positive speed threshold")
    velocity = _array(root_velocity, (n, 3), "root_com_velocity_w")
    wheel_velocity = _array(wheel_center_velocity_w, (n, 2, 3), "wheel_center_velocity_w")
    root_q = _quaternions(root_quaternion_wxyz, (n, 4), "root_quaternion_wxyz")
    wheel_q = _quaternions(wheel_quaternion_wxyz, (n, 2, 4), "wheel_quaternion_wxyz")
    root_forward = quaternion_rotate(root_q, np.array([1., 0., 0.]))
    axles = quaternion_rotate(wheel_q, np.array([0., 1., 0.]))
    rolling = np.cross(axles, np.array([0., 0., 1.]))
    root_axis_valid = np.linalg.norm(root_forward[:, :2], axis=-1) > 1e-6
    wheel_axis_valid = np.linalg.norm(rolling[:, :, :2], axis=-1) > 1e-6
    speed = np.linalg.norm(velocity[:, :2], axis=-1)
    wheel_speed = np.linalg.norm(wheel_velocity[:, :, :2], axis=-1)
    course = np.arctan2(velocity[:, 1], velocity[:, 0])
    wheel_course = np.arctan2(wheel_velocity[:, :, 1], wheel_velocity[:, :, 0])
    body_heading = np.arctan2(root_forward[:, 1], root_forward[:, 0])
    wheel_heading = np.arctan2(rolling[:, :, 1], rolling[:, :, 0])
    root_valid = (speed >= speed_threshold_m_s) & root_axis_valid
    root_wheel_valid = (speed[:, None] >= speed_threshold_m_s) & wheel_axis_valid
    wheel_valid = (wheel_speed >= speed_threshold_m_s) & wheel_axis_valid
    # Values at invalid samples are merely storage; always consume their masks.
    return {"speed_m_s": speed, "wheel_center_speed_m_s": wheel_speed,
            "course_rad": course, "body_heading_rad": body_heading,
            "wheel_rolling_plane_heading_rad": wheel_heading,
            "root_directed_beta_rad": wrap(course - body_heading),
            "root_course_to_wheel_gamma_rad": axial(course[:, None] - wheel_heading),
            "wheel_center_course_to_wheel_gamma_rad": axial(wheel_course - wheel_heading),
            "root_beta_valid": root_valid,
            "root_course_to_wheel_valid": root_wheel_valid,
            "wheel_center_course_to_wheel_valid": wheel_valid,
            "speed_threshold_m_s": float(speed_threshold_m_s)}


def _angle_summary(values, mask, *, axial_angle):
    valid = np.asarray(mask, dtype=bool)
    angles = np.asarray(values)[valid]
    result = {"valid_samples": int(valid.sum()), "eligible_samples": int(valid.size),
              "valid_fraction": float(valid.mean()) if valid.size else None,
              "signed_circular_mean_deg": None, "mean_absolute_deg": None,
              "absolute_p95_deg": None}
    if not angles.size:
        return result
    factor = 2. if axial_angle else 1.
    resultant = np.mean(np.exp(1j * factor * angles))
    # An undefined circular mean must not be reported as an arbitrary zero.
    if abs(resultant) > 1e-10:
        result["signed_circular_mean_deg"] = float(np.degrees(np.angle(resultant) / factor))
    result.update(mean_absolute_deg=float(np.degrees(np.abs(angles)).mean()),
                  absolute_p95_deg=float(np.degrees(np.quantile(np.abs(angles), .95))))
    return result


def summarize_turn(time_s, root_position_w, signals, reference_time_s,
                   reference_xy, reference_speed_m_s, interval_s, *,
                   wheel_ground_contact_count=None, wheel_ground_force_w=None,
                   wheel_contact_lateral_slip_m_s=None, contact_threshold_n=5.):
    """Score one predeclared window; partial coverage is censored, not a pass.

    Optional ground-filtered counts AND upward net force gate measured contact.
    In their absence, only kinematic heading geometry is summarized. Never use
    height, flight, beta or gamma as an automatic tracking failure criterion.
    """
    time = np.asarray(time_s, dtype=float)
    reference_t = np.asarray(reference_time_s, dtype=float)
    if (time.ndim != 1 or reference_t.ndim != 1 or min(len(time), len(reference_t)) < 2
            or not np.isfinite(time).all() or not np.isfinite(reference_t).all()
            or np.any(np.diff(time) <= 0) or np.any(np.diff(reference_t) <= 0)):
        raise ValueError("Expected finite strictly increasing native/reference clocks")
    n = len(time)
    position = _array(root_position_w, (n, 3), "root_position_w")
    ref_xy = _array(reference_xy, (len(reference_t), 2), "reference_xy")
    ref_speed = _array(reference_speed_m_s, (len(reference_t),), "reference_speed_m_s")
    if np.any(ref_speed < 0) or not np.isfinite(contact_threshold_n) or contact_threshold_n < 0:
        raise ValueError("Invalid reference speed or contact threshold")
    interval = _array(interval_s, (2,), "interval_s")
    start, end = interval
    if not reference_t[0] <= start < end <= reference_t[-1]:
        raise ValueError("Evaluation interval must lie within the real reference clock")
    speed = _array(signals["speed_m_s"], (n,), "speed_m_s")
    if np.any(speed < 0):
        raise ValueError("Speed cannot be negative")
    root_valid = _mask(signals["root_beta_valid"], (n,), "root_beta_valid")
    root_wheel_valid = _mask(signals["root_course_to_wheel_valid"], (n, 2), "root_course_to_wheel_valid")
    wheel_valid = _mask(signals["wheel_center_course_to_wheel_valid"], (n, 2), "wheel_center_course_to_wheel_valid")
    course = _array(signals["course_rad"], (n,), "course_rad")
    beta = _array(signals["root_directed_beta_rad"], (n,), "root_directed_beta_rad")
    root_gamma = _array(signals["root_course_to_wheel_gamma_rad"], (n, 2), "root_course_to_wheel_gamma_rad")
    wheel_gamma = _array(signals["wheel_center_course_to_wheel_gamma_rad"], (n, 2), "wheel_center_course_to_wheel_gamma_rad")
    supplied = (wheel_ground_contact_count is not None, wheel_ground_force_w is not None)
    if supplied[0] != supplied[1]:
        raise ValueError("Measured contact requires both ground-filtered counts and upward forces")
    gate = slip = None
    if supplied[0]:
        counts = _array(wheel_ground_contact_count, (n, 2), "wheel_ground_contact_count")
        force = _array(wheel_ground_force_w, (n, 2, 3), "wheel_ground_force_w")
        if np.any(counts < 0) or not np.array_equal(counts, np.round(counts)):
            raise ValueError("Ground contact counts must be nonnegative integers")
        gate = (counts > 0) & (force[:, :, 2] > contact_threshold_n)
        if wheel_contact_lateral_slip_m_s is not None:
            slip = _array(wheel_contact_lateral_slip_m_s, (n, 2), "wheel_contact_lateral_slip_m_s")
            if np.any(slip < 0):
                raise ValueError("Expected nonnegative force-weighted lateral-slip RMS")
    elif wheel_contact_lateral_slip_m_s is not None:
        raise ValueError("Cannot interpret contact-slip zeros without a measured contact gate")
    selected = (time >= start - 1e-9) & (time <= end + 1e-9)
    covered = bool(time[0] <= start + 1e-9 and time[-1] >= end - 1e-9 and selected.sum() >= 2)
    result = {"window_s": interval.tolist(), "coverage_complete": covered,
              "samples_in_window": int(selected.sum()), "reference_xy_rmse_m": None,
              "reference_xy_peak_m": None, "large_xy_exploratory_screen": None,
              "mean_speed_m_s": None, "reference_mean_speed_m_s": None,
              "mean_speed_retention_ratio": None, "fraction_below_80pct_reference_speed": None,
              "net_course_change_deg": None, "heading_geometry": None,
              "ground_contact_heading_geometry": None, "ground_contact_lateral_slip_m_s": None,
              "heading_offset_is_mismatch": False, "height_or_flight_failure_gate": False}
    result["xy_screen_limits_m"] = {"rmse": .15, "peak": .30}
    if not covered:
        return result
    query = time[selected]
    target_xy = np.column_stack([np.interp(query, reference_t, ref_xy[:, j]) for j in range(2)])
    target_speed = np.interp(query, reference_t, ref_speed)
    error = np.linalg.norm(position[selected, :2] - target_xy, axis=-1)
    rmse, peak = float(np.sqrt(np.mean(error ** 2))), float(error.max())
    denominator = float(target_speed.mean())
    result.update(reference_xy_rmse_m=rmse, reference_xy_peak_m=peak,
                  large_xy_exploratory_screen=rmse > .15 or peak > .30,
                  mean_speed_m_s=float(speed[selected].mean()), reference_mean_speed_m_s=denominator,
                  mean_speed_retention_ratio=float(speed[selected].mean() / denominator) if denominator > 1e-12 else None,
                  fraction_below_80pct_reference_speed=float(np.mean(speed[selected] < .8 * target_speed)))
    if root_valid[selected].all():
        result["net_course_change_deg"] = float(np.degrees(np.diff(np.unwrap(course[selected])[[0, -1]])[0]))
    def wheel_summaries(gate):
        return {side: {"root_course_to_wheel_gamma": _angle_summary(root_gamma[selected, j], (root_wheel_valid & gate)[selected, j], axial_angle=True),
                       "wheel_center_course_to_wheel_gamma": _angle_summary(wheel_gamma[selected, j], (wheel_valid & gate)[selected, j], axial_angle=True)}
                for j, side in enumerate(("left", "right"))}
    result["heading_geometry"] = {"root_directed_beta": _angle_summary(beta[selected], root_valid[selected], axial_angle=False),
                                  "per_wheel": wheel_summaries(np.ones((n, 2), dtype=bool)),
                                  "measured_contact_gate_applied": False}
    if gate is not None:
        result["ground_contact_heading_geometry"] = {
            "threshold_upward_force_n": float(contact_threshold_n), "per_wheel": wheel_summaries(gate),
            "measured_contact_gate_applied": True,
            "contact_fraction_left_right": gate[selected].mean(axis=0).tolist()}
        if slip is not None:
            result["ground_contact_lateral_slip_m_s"] = {
                side: float(np.sqrt(np.mean(slip[selected, j][gate[selected, j]] ** 2))) if gate[selected, j].any() else None
                for j, side in enumerate(("left", "right"))}
    return result

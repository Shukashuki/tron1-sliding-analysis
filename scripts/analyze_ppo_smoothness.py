"""Offline native-clock PPO smoothness diagnostics and unvalidated screening.

Only NumPy and the standard library are required. No simulation, policy,
filtering, trajectory correction, or replacement of recorded states occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CORE_INTERVAL = (1.44, 4.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(value):
    array = np.asarray(value, dtype=float)
    return float(np.sqrt(np.mean(array ** 2))) if array.size else None


def validate_clock(time_s, *, name="physical"):
    time = np.asarray(time_s, dtype=float)
    require(time.ndim == 1 and len(time) >= 4 and np.isfinite(time).all(),
            name + " clock must have at least four finite samples")
    spacing = np.diff(time)
    require(np.all(spacing > 0), name + " clock must be strictly increasing")
    dt = float(np.median(spacing))
    require(np.allclose(spacing, dt, rtol=1e-7, atol=1e-10),
            name + " clock must be uniform; irregular clocks cannot use FFT")
    return time, dt


def spectral_metrics(values, dt, high_frequency_hz=10.0):
    """Hann-windowed, linear-detrended one-sided power, averaged by channel.

    Integrated powers are mean squares in signal units squared, not raw FFT
    magnitudes. The HF RMS is the square root of the integrated HF mean square.
    It is window-weighted/detrended, not the unwindowed raw-signal RMS.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    require(array.ndim == 2 and len(array) >= 4 and array.shape[1] > 0 and np.isfinite(array).all(),
            "Spectrum needs at least four finite samples and one channel")
    require(np.isfinite(dt) and dt > 0 and np.isfinite(high_frequency_hz)
            and 0 < high_frequency_hz < .5 / dt, "HF cutoff must be between zero and Nyquist")
    n = len(array)
    design = np.column_stack((np.ones(n), np.arange(n, dtype=float)))
    detrended = array - design @ np.linalg.lstsq(design, array, rcond=None)[0]
    window = np.hanning(n)
    transform = np.fft.rfft(detrended * window[:, None], axis=0)
    power = np.mean(np.abs(transform) ** 2, axis=1) / (n * np.sum(window ** 2))
    if n % 2:
        power[1:] *= 2.
    else:
        power[1:-1] *= 2.
    power[0] = 0.  # DC is not an oscillation or a useful peak frequency.
    frequency = np.fft.rfftfreq(n, dt)
    total = float(power.sum())
    high = float(power[frequency > high_frequency_hz].sum())
    negligible = total < 1e-24
    return {
        "samples": n, "sample_hz": 1. / dt, "frequency_resolution_hz": 1. / (n * dt),
        "cutoff_hz_exclusive": float(high_frequency_hz),
        "dominant_frequency_hz": None if negligible else float(frequency[np.argmax(power)]),
        "total_mean_square": 0. if negligible else total,
        "high_frequency_mean_square": 0. if negligible else high,
        "high_frequency_rms": 0. if negligible else float(np.sqrt(high)),
        "high_frequency_power_fraction": 0. if negligible else high / total,
        "normalization": "one-sided Parseval normalization; linear detrend; Hann; channel-mean power; DC excluded",
    }


def _field(raw, name, count, width=None, *, optional=False):
    if name not in raw:
        require(optional, "Missing required field: " + name)
        return None
    value = np.asarray(raw[name])
    require(value.ndim >= 1, name + ": expected a sample axis")
    if width is None:
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        require(value.shape[0] >= count and value.ndim == 1, name + ": expected one environment scalar per sample")
    else:
        if value.ndim == 3 and value.shape[1] == 1:
            value = value[:, 0]
        require(value.shape[0] >= count and value.ndim == 2 and value.shape[1] == width,
                name + ": expected one environment with width " + str(width))
    value = value[:count]
    require(value.dtype.kind in "biuf" and np.isfinite(value).all(), name + ": nonfinite or nonnumeric data")
    return value


def deduplicate_actions(time_s, actions, applied, policy_dt_s, *, frames=None, counters=None):
    """Use decision identity/time, never a count of repeated applied flags.

    A PhysX physics recorder repeats each held command across four substeps;
    ``action_was_applied`` is true at all four, not just at the decision boundary.
    """
    time, dt = validate_clock(time_s)
    actions = np.asarray(actions, dtype=float)
    applied = np.asarray(applied)
    require(actions.shape == (len(time), 8) and np.isfinite(actions).all(), "Actions must be finite (samples,8)")
    require(np.max(np.abs(actions)) <= 1.+1e-7, "Recorded clipped actions exceed the [-1,1] contract")
    require(applied.shape == time.shape and applied.dtype.kind == "b", "Applied-action flags must be boolean")
    require(np.isfinite(policy_dt_s) and policy_dt_s > 0, "Policy dt must be positive")
    ratio = policy_dt_s / dt
    decimation = int(round(ratio))
    require(decimation >= 1 and np.isclose(ratio, decimation, rtol=0, atol=1e-7), "Policy dt must be an integer physics decimation")
    indices = np.flatnonzero(applied)
    require(len(indices) >= decimation and np.all(np.diff(indices) == 1),
            "Applied-action records must be a contiguous first-episode segment")
    require(indices[0] == 1 and not applied[0], "Need the initial no-action state followed by post-step applied records")
    if counters is not None:
        counter = np.asarray(counters)
        require(counter.shape == time.shape and np.isfinite(counter).all()
                and np.array_equal(counter, np.round(counter)) and np.all(np.diff(counter) == 1),
                "Physical-step counters must advance one per sample")
    if frames is not None:
        frame = np.asarray(frames)
        require(frame.shape == time.shape and np.isfinite(frame).all()
                and np.array_equal(frame, np.round(frame)), "Decision frame IDs must be finite integers")
        selected = frame[indices]
        require(np.all(np.diff(selected) >= 0), "Decision frame IDs reset or move backwards")
        offsets = np.r_[0, np.flatnonzero(np.diff(selected) != 0) + 1]
        require(np.all(np.diff(selected[offsets]) == 1), "Decision frame IDs must advance one per control interval")
        starts = indices[offsets]
    else:
        require(counters is not None, "Need decision frame IDs or physical-step counters, not held-action flags alone")
        starts = indices[np.arange(len(indices)) % decimation == 0]
    ends = np.r_[starts[1:], indices[-1] + 1]
    lengths = ends - starts
    require(np.all(lengths[:-1] == decimation) and 0 < lengths[-1] <= decimation,
            "Decision blocks do not match the recorded policy decimation")
    require(len(starts) >= 2 and np.allclose(np.diff(time[starts]), policy_dt_s, rtol=0, atol=1e-9),
            "Decision starts must follow the policy clock")
    for start, end in zip(starts, ends):
        require(np.array_equal(actions[start:end], np.repeat(actions[start:start+1], end-start, axis=0)),
                "Actions change inside a held decision block")
    return {"time_s": time[starts] - dt, "actions": actions[starts].copy(), "record_indices": starts,
            "decimation": decimation, "policy_dt_s": float(policy_dt_s),
            "held_physical_records": int(len(indices)), "decisions": len(starts)}


def _load_reference(path):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    if "body_pos_w" in values:
        names = tuple(values["body_names"].tolist())
        root = names.index("base_Link")
        time = np.asarray(values["time_s"], dtype=float)
        position = np.asarray(values["body_pos_w"][:, root], dtype=float)
        velocity = np.asarray(values["body_lin_vel_w"][:, root], dtype=float)
        quaternion = np.asarray(values["body_quat_w"][:, root], dtype=float)
    else:
        time = np.asarray(values["reference_time_s"], dtype=float)
        position = np.asarray(values["reference_position"], dtype=float)
        velocity = np.asarray(values["reference_com_velocity"], dtype=float)
        quaternion = np.asarray(values["reference_quaternion"], dtype=float)
    time, _ = validate_clock(time, name="reference")
    require(position.shape == velocity.shape == (len(time), 3) and quaternion.shape == (len(time), 4)
            and all(np.isfinite(value).all() for value in (position, velocity, quaternion)), "Invalid reference state arrays")
    return {"time_s": time, "position": position, "velocity": velocity,
            "quaternion": quaternion, "sha256": sha256(path)}


def _orientation(quaternion):
    value = np.asarray(quaternion, dtype=float)
    require(value.ndim == 2 and value.shape[1] == 4 and np.isfinite(value).all(), "Quaternion must be finite wxyz")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    require(np.allclose(norm, 1., rtol=0, atol=1e-5), "Quaternion must be unit wxyz")
    w, x, y, z = (value / norm).T
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    heading = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.column_stack((roll, pitch)), heading


def _world_to_body(quaternion, vectors):
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    imaginary = -quaternion[:, 1:]
    twice_cross = 2.*np.cross(imaginary, vectors)
    return vectors + quaternion[:, :1]*twice_cross + np.cross(imaginary, twice_cross)


def _tracking(time, position, velocity, quaternion, reference, interval):
    if reference is None:
        return {"available": False, "reason": "No reference supplied; XY, speed retention and beta error are not inferred from jitter"}
    start, end = interval
    complete = bool(time[0] <= start+1e-9 and time[-1] >= end-1e-9
                    and reference["time_s"][0] <= start+1e-9 and reference["time_s"][-1] >= end-1e-9)
    if not complete:
        return {"available": False, "coverage_complete": False,
                "reason": "Physical/reference clock does not fully cover this window; no extrapolation or endpoint hold"}
    selected = (time >= start-1e-9) & (time <= end+1e-9)
    query = time[selected]
    interp = lambda array: np.column_stack([np.interp(query, reference["time_s"], array[:, j])
                                           for j in range(array.shape[1])])
    target = interp(reference["position"][:, :2])
    error = np.linalg.norm(position[selected, :2] - target, axis=1)
    speed = np.linalg.norm(velocity[selected, :2], axis=1)
    reference_speed = np.interp(query, reference["time_s"], np.linalg.norm(reference["velocity"][:, :2], axis=1))
    denominator = float(reference_speed.mean())
    _, heading = _orientation(quaternion)
    _, target_heading = _orientation(reference["quaternion"])
    beta = np.arctan2(velocity[:, 1], velocity[:, 0]) - heading
    reference_beta = np.arctan2(reference["velocity"][:, 1], reference["velocity"][:, 0]) - target_heading
    target_beta = np.interp(query, reference["time_s"], np.unwrap(reference_beta))
    delta_beta = beta[selected] - target_beta
    delta_beta = np.arctan2(np.sin(delta_beta), np.cos(delta_beta))
    valid = (speed >= .25) & (reference_speed >= .25)
    return {"available": True, "coverage_complete": True, "samples": int(selected.sum()),
            "xy_rmse_m": rms(error), "xy_peak_m": float(error.max()),
            "mean_speed_m_s": float(speed.mean()), "reference_mean_speed_m_s": denominator,
            "speed_retention": float(speed.mean()/denominator) if denominator > 1e-12 else None,
            "beta_rmse_deg": float(np.rad2deg(rms(delta_beta[valid]))) if valid.any() else None,
            "beta_valid_samples": int(valid.sum()),
            "beta_definition": "world-XY COM course minus projected body +X heading; error against intended nonzero reference, not zero slip"}


def analyze_trajectory(trajectory, *, reference=None, report=None, high_frequency_hz=10.0,
                       core_interval=CORE_INTERVAL, leg_action_scale_rad=1.0):
    """Analyze one original first-episode recording without changing samples."""
    with np.load(trajectory, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    require("time_s" in raw, "Missing required field: time_s")
    require(np.asarray(raw["time_s"]).ndim == 1, "Physical clock must be one-dimensional")
    physical_count = len(raw["time_s"])
    if "valid_mask" in raw:
        valid = _field(raw, "valid_mask", physical_count)
        require(valid.dtype.kind == "b" and valid[0], "Validity must begin with a valid first-episode sample")
        first_invalid = np.flatnonzero(~valid)
        count = int(first_invalid[0]) if len(first_invalid) else physical_count
        require(not valid[count:].any(), "Validity mask splices episodes")
    else:
        count = physical_count
    time, dt = validate_clock(np.asarray(raw["time_s"])[:count])
    require(abs(time[0]) <= 1e-9, "Expected relative first-episode clock starting at zero")
    if "physics_dt_s" in raw:
        require(np.isclose(float(raw["physics_dt_s"]), dt, rtol=1e-7, atol=1e-10), "Recorded physics dt disagrees with native clock")
    require("policy_dt_s" in raw, "Missing required field: policy_dt_s")
    policy_dt = float(raw["policy_dt_s"])
    position = _field(raw, "root_position_m", count, 3).astype(float)
    quaternion = _field(raw, "root_quaternion_wxyz", count, 4).astype(float)
    velocity = _field(raw, "root_linear_velocity_w_m_s", count, 3).astype(float)
    angular = _field(raw, "root_angular_velocity_w_rad_s", count, 3).astype(float)
    joint_position = _field(raw, "joint_position_rad", count, 8).astype(float)
    joint_velocity = _field(raw, "joint_velocity_rad_s", count, 8).astype(float)
    torque = _field(raw, "applied_joint_torque_nm", count, 8).astype(float)
    actions = _field(raw, "actions_clipped", count, 8).astype(float)
    raw_actions = _field(raw, "actions_raw", count, 8, optional=True)
    applied = _field(raw, "action_was_applied", count)
    frames = _field(raw, "reference_frame_for_action", count, optional=True)
    counters = _field(raw, "simulation_step_counter", count, optional=True)
    require("joint_names" in raw, "Missing required field: joint_names")
    names = tuple(raw["joint_names"].tolist())
    require(len(names) == 8 and len(set(names)) == 8, "Need eight unique joint names")
    wheels = [index for index, name in enumerate(names) if name in ("wheel_L_Joint", "wheel_R_Joint")]
    legs = [index for index in range(8) if index not in wheels]
    require(len(wheels) == 2 and len(legs) == 6, "Need six leg joints and both named wheel joints")
    require(np.isfinite(leg_action_scale_rad) and leg_action_scale_rad > 0, "Leg action scale must be positive")
    decisions = deduplicate_actions(time, actions, applied, policy_dt, frames=frames, counters=counters)
    roll_pitch, _ = _orientation(quaternion)
    angular_body = _world_to_body(quaternion, angular)
    reference_data = _load_reference(reference)
    evaluation = json.loads(Path(report).read_text()) if report is not None else {}
    bound_sha = evaluation.get("physics_trajectory_sha256")
    if bound_sha is not None:
        require(bound_sha == sha256(trajectory), "Report-to-trajectory SHA binding mismatch")
    coefficient = evaluation.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction")
    require(coefficient is None or (np.isfinite(coefficient) and coefficient >= 0), "Invalid reported friction coefficient")
    windows = {}
    require(len(core_interval) == 2 and 0 <= core_interval[0] < core_interval[1], "Invalid core interval")
    for name, interval in (("full", (float(time[0]), float(time[-1]))), ("core", tuple(core_interval))):
        lo, hi = interval
        mask = (time >= lo-1e-9) & (time <= hi+1e-9)
        action_mask = (decisions["time_s"] >= lo-1e-9) & (decisions["time_s"] <= hi+1e-9)
        complete = bool(time[0] <= lo+1e-9 and time[-1] >= hi-1e-9)
        if mask.sum() < 4 or action_mask.sum() < 4:
            windows[name] = {"interval_s": list(interval), "coverage_complete": False,
                             "reason": "Fewer than four physical/control samples in window"}
            continue
        selected_actions = decisions["actions"][action_mask]
        delta = np.diff(selected_actions, axis=0)
        selected_torque = torque[mask]
        native_indices = np.arange(0, count, decisions["decimation"])
        video_indices = native_indices[(time[native_indices] >= lo-1e-9) & (time[native_indices] <= hi+1e-9)]
        pose_delta = np.diff(roll_pitch[video_indices], axis=0)
        selected_raw = raw_actions[decisions["record_indices"]][action_mask] if raw_actions is not None else None
        windows[name] = {
            "interval_s": list(interval), "coverage_complete": complete,
            "physical_samples": int(mask.sum()), "control_decisions": int(action_mask.sum()),
            "action": {"leg_delta_rms": rms(delta[:, legs]), "wheel_delta_rms": rms(delta[:, wheels]),
                "leg_target_delta_rms_rad": rms(delta[:, legs])*leg_action_scale_rad,
                "leg_delta_abs_peak": float(np.abs(delta[:, legs]).max()),
                "leg_raw_outside_clip_fraction": float((np.abs(selected_raw[:, legs]) > 1.).mean()) if selected_raw is not None else None,
                "wheel_raw_outside_clip_fraction": float((np.abs(selected_raw[:, wheels]) > 1.).mean()) if selected_raw is not None else None,
                "leg_spectrum": spectral_metrics(selected_actions[:, legs], policy_dt, high_frequency_hz)},
            "legs": {"joint_position_step_rms_rad": rms(np.diff(joint_position[mask][:, legs], axis=0)),
                "joint_velocity_rms_rad_s": rms(joint_velocity[mask][:, legs]),
                "joint_velocity_component_rms_rad_s": [rms(joint_velocity[mask, index]) for index in legs],
                "joint_velocity_abs_peak_rad_s": float(np.abs(joint_velocity[mask][:, legs]).max()),
                "finite_difference_joint_acceleration_rms_rad_s2": rms(np.diff(joint_velocity[mask][:, legs], axis=0)/dt),
                "acceleration_note": "Finite difference of recorded joint velocity; not a measured accelerometer or separate actuator state"},
            "torque": {"leg_rms_nm": rms(selected_torque[:, legs]),
                "wheel_rms_nm": rms(selected_torque[:, wheels]),
                "leg_step_rms_nm": rms(np.diff(selected_torque[:, legs], axis=0)),
                "leg_abs_peak_nm": float(np.abs(selected_torque[:, legs]).max()),
                "leg_near_80nm_fraction": float((np.abs(selected_torque[:, legs]) >= 79.2).mean()),
                "near_limit_definition": "abs(applied leg torque)>=79.2 Nm; sample fraction, NOT inferred motor saturation on a speed-torque curve",
                "leg_spectrum": spectral_metrics(selected_torque[:, legs], dt, high_frequency_hz)},
            "body": {"angular_xy_rms_rad_s": rms(angular[mask, :2]),
                "angular_component_rms_rad_s": [rms(angular[mask, index]) for index in range(3)],
                "angular_xy_abs_peak_rad_s": float(np.abs(angular[mask, :2]).max()),
                "angular_xy_spectrum": spectral_metrics(angular[mask, :2], dt, high_frequency_hz),
                "angular_body_xy_rms_rad_s": rms(angular_body[mask, :2]),
                "angular_body_component_rms_rad_s": [rms(angular_body[mask, index]) for index in range(3)],
                "angular_body_xy_spectrum": spectral_metrics(angular_body[mask, :2], dt, high_frequency_hz),
                "vertical_com_velocity_rms_m_s": rms(velocity[mask, 2]),
                "com_xy_velocity_step_rms_m_s": rms(np.diff(velocity[mask, :2], axis=0)),
                "angular_velocity_definition": "Recorded world-frame XY components, not differentiated quaternions or exact body roll/pitch rates"},
            "pose_at_policy_fps": {"sample_hz": 1./policy_dt, "samples": len(video_indices),
                "roll_pitch_step_rms_deg": float(np.rad2deg(rms(pose_delta))) if len(pose_delta) else None,
                "roll_pitch_step_abs_peak_deg": float(np.rad2deg(np.abs(pose_delta).max())) if len(pose_delta) else None,
                "note": "Original native poses subsampled at policy rate; frequencies above its Nyquist can alias. No interpolation or smoothing."},
            "tracking": _tracking(time, position, velocity, quaternion, reference_data, interval)}
    return {"trajectory_file_name": Path(trajectory).name, "trajectory_sha256": sha256(trajectory),
            "report_sha256": sha256(report) if report is not None else None,
            "report_to_trajectory_sha_verified": bound_sha is not None,
            "reference_sha256": reference_data["sha256"] if reference_data else None,
            "checkpoint_sha256": evaluation.get("checkpoint_sha256"), "evaluation_mu": coefficient,
            "high_frequency_hz": float(high_frequency_hz),
            "physical_dt_s": dt, "policy_dt_s": policy_dt, "physical_samples": count,
            "recorded_samples": physical_count, "excluded_invalid_tail_samples": physical_count-count,
            "control_decisions": decisions["decisions"], "held_action_records": decisions["held_physical_records"],
            "joint_names": list(names), "leg_action_scale_rad": leg_action_scale_rad,
            "action_record_semantics": "Recorded actions_clipped only; if filtering exists, logger must specify whether these are pre/post-filter. No unrecorded actuator target is invented.",
            "windows": windows, "new_simulation": False, "new_inference": False}


def compare_candidate(candidate, baseline, *, reduction_fraction=.5, xy_max_rmse_m=.15,
                      xy_extra_rmse_m=.03, speed_retention_min=.9,
                      beta_extra_rmse_deg=None, window="core"):
    """Descriptive engineering screen, NOT a validated safety/success gate."""
    thresholds = [reduction_fraction, xy_max_rmse_m, xy_extra_rmse_m, speed_retention_min]
    if beta_extra_rmse_deg is not None:
        thresholds.append(beta_extra_rmse_deg)
    require(np.isfinite(thresholds).all() and 0 <= reduction_fraction < 1 and xy_max_rmse_m > 0 and xy_extra_rmse_m >= 0
            and speed_retention_min >= 0 and (beta_extra_rmse_deg is None or beta_extra_rmse_deg >= 0),
            "Invalid configurable screening thresholds")
    current, original = candidate["windows"].get(window, {}), baseline["windows"].get(window, {})
    criteria = {}

    def add(name, value, limit, *, evaluable=True, note=None):
        criteria[name] = {"value": value, "maximum": limit,
                          "evaluable": bool(evaluable), "passes": bool(value <= limit) if evaluable else None}
        if note:
            criteria[name]["note"] = note

    complete = current.get("coverage_complete") is True and original.get("coverage_complete") is True
    require(candidate.get("high_frequency_hz") == baseline.get("high_frequency_hz"),
            "Candidate and baseline high-frequency cutoffs differ")
    require(np.isclose(candidate.get("physical_dt_s", np.nan), baseline.get("physical_dt_s", np.nan), rtol=1e-7, atol=1e-10)
            and np.isclose(candidate.get("policy_dt_s", np.nan), baseline.get("policy_dt_s", np.nan), rtol=1e-7, atol=1e-10),
            "Candidate and baseline physical/control clocks differ")
    for label, getter in (
        ("absolute_hf_body_angular_xy_rms_ratio", lambda item: item["body"]["angular_xy_spectrum"]["high_frequency_rms"]),
        ("leg_action_delta_rms_ratio", lambda item: item["action"]["leg_delta_rms"])):
        try:
            current_value, base_value = getter(current), getter(original)
            evaluable = complete and np.isfinite(current_value) and np.isfinite(base_value) and base_value > 1e-12
        except (KeyError, TypeError):
            evaluable, current_value, base_value = False, None, None
        add(label, current_value/base_value if evaluable else None, 1.-reduction_fraction,
            evaluable=evaluable, note="Absolute band RMS, not just fraction of spectral power" if label.startswith("absolute") else None)
    ctrack, btrack = current.get("tracking", {}), original.get("tracking", {})
    tracking_available = complete and ctrack.get("available") is True and btrack.get("available") is True
    xy_limit = min(xy_max_rmse_m, btrack["xy_rmse_m"]+xy_extra_rmse_m) if tracking_available else xy_max_rmse_m
    add("xy_rmse_m", ctrack.get("xy_rmse_m"), xy_limit, evaluable=tracking_available)
    speed = ctrack.get("speed_retention")
    criteria["nominal_speed_retention"] = {
        "value": speed, "minimum": speed_retention_min,
        "evaluable": bool(tracking_available and speed is not None),
        "passes": bool(speed >= speed_retention_min) if tracking_available and speed is not None else None,
        "note": "Intended for nominal-condition screening; low-friction robustness must be evaluated separately"}
    if beta_extra_rmse_deg is not None:
        c_beta, b_beta = ctrack.get("beta_rmse_deg"), btrack.get("beta_rmse_deg")
        evaluable = tracking_available and c_beta is not None and b_beta is not None
        add("beta_rmse_deg", c_beta, b_beta+beta_extra_rmse_deg if evaluable else None, evaluable=evaluable,
            note="Error against intended reference beta; nonzero sideslip is not itself a mismatch")
    same_mu = candidate.get("evaluation_mu") == baseline.get("evaluation_mu")
    same_reference = candidate.get("reference_sha256") == baseline.get("reference_sha256")
    require(same_mu, "Candidate and baseline evaluation friction differ")
    require(same_reference, "Candidate and baseline reference identities differ")
    evaluable = all(item["evaluable"] for item in criteria.values())
    passed = evaluable and all(item["passes"] for item in criteria.values())
    return {"kind": "unvalidated_smoothing_engineering_screen", "window": window,
            "status": "passes_unvalidated_screen" if passed else "fails_unvalidated_screen" if evaluable else "insufficient_evidence",
            "passes": bool(passed), "criteria": criteria,
            "baseline_trajectory_sha256": baseline["trajectory_sha256"],
            "candidate_trajectory_sha256": candidate["trajectory_sha256"],
            "statistical_success_claimed": False, "safety_qualification_claimed": False,
            "causal_improvement_claimed": False,
            "thresholds": {"required_reduction_fraction": reduction_fraction,
                "xy_max_rmse_m": xy_max_rmse_m, "xy_allowed_increase_m": xy_extra_rmse_m,
                "nominal_speed_retention_min": speed_retention_min, "beta_allowed_increase_deg": beta_extra_rmse_deg}}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trajectory", type=Path, action="append", required=True,
                        help="Repeat for the paired historical mu=.70/.15 baselines")
    result.add_argument("--report", type=Path, action="append", default=[], help="Optional matching Isaac report per trajectory")
    result.add_argument("--reference", type=Path, help="Motion NPZ or published reference_* archive; no extrapolation")
    result.add_argument("--baseline", type=Path, help="Baseline NPZ or this analyzer's previous JSON")
    result.add_argument("--baseline-report", type=Path)
    result.add_argument("--output", type=Path, required=True, help="Fresh JSON filename; never overwrite")
    result.add_argument("--high-frequency-hz", type=float, default=10.)
    result.add_argument("--required-reduction", type=float, default=.5)
    result.add_argument("--xy-max-rmse-m", type=float, default=.15)
    result.add_argument("--xy-extra-rmse-m", type=float, default=.03)
    result.add_argument("--speed-retention-min", type=float, default=.9)
    result.add_argument("--beta-extra-rmse-deg", type=float)
    return result


def run(args):
    output = Path(args.output)
    require(not output.exists(), "Preserve existing analysis: choose a fresh output file")
    require(len(args.report) in (0, len(args.trajectory)), "Supply either no reports or one report per trajectory")
    reports = args.report or [None]*len(args.trajectory)
    runs = [analyze_trajectory(path, reference=args.reference, report=report,
                              high_frequency_hz=args.high_frequency_hz)
            for path, report in zip(args.trajectory, reports)]
    comparisons = []
    if args.baseline:
        if Path(args.baseline).suffix.lower() == ".npz":
            baselines = [analyze_trajectory(args.baseline, reference=args.reference, report=args.baseline_report,
                                           high_frequency_hz=args.high_frequency_hz)]
        else:
            payload = json.loads(Path(args.baseline).read_text())
            baselines = payload["runs"] if "runs" in payload else [payload]
        for candidate in runs:
            matches = [item for item in baselines if item.get("evaluation_mu") == candidate.get("evaluation_mu")]
            require(len(matches) == 1, "Baseline is ambiguous or has no matching friction coefficient")
            comparisons.append(compare_candidate(candidate, matches[0], reduction_fraction=args.required_reduction,
                xy_max_rmse_m=args.xy_max_rmse_m, xy_extra_rmse_m=args.xy_extra_rmse_m,
                speed_retention_min=args.speed_retention_min, beta_extra_rmse_deg=args.beta_extra_rmse_deg))
    result = {"schema_version": 1, "kind": "native_physx_ppo_smoothness_diagnostics",
              "analysis_source_sha256": sha256(__file__), "runs": runs, "comparisons": comparisons,
              "high_frequency_hz": args.high_frequency_hz,
              "limitations": ["One recorded episode per condition; not statistical or safety qualification.",
                  "Low XY error does not certify smooth execution; height, flight and intended sideslip are not new failure gates.",
                  "Near-80-Nm sample fraction is not a motor saturation measurement.",
                  "FFT band RMS is Hann-windowed and linear-detrended; ratios are descriptive, not a causal diagnosis.",
                  "Reference-free recordings cannot pass an XY/speed/beta screen; absence of data is not success."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    print(json.dumps({"runs": len(outcome["runs"]), "comparisons": [item["status"] for item in outcome["comparisons"]]}))

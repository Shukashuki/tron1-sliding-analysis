"""Same-time root-origin XY error decomposition from published state vectors.

Pure NumPy geometry: no FK, physics, nearest-path matching, time warping or
spatial registration. A negative tangent projection means behind the reference
at that timestamp; neither projection measures contact slip or causal delay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from controller_references import get_reference
from nmpc_tracking_objective import quaternion_rotate


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
COM_OFFSET_M = np.array([.04576, .00014, -.16398])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite shape {shape}")
    return value


def _clock(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or len(value) < 3 or not np.isfinite(value).all() or np.any(np.diff(value) <= 0):
        raise ValueError("Expected at least three finite increasing timestamps")
    return value


def _quat(value, n):
    value = _array(value, (n, 4), "quaternion_wxyz")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.allclose(norm, 1., atol=1e-5, rtol=0):
        raise ValueError("Expected normalized wxyz quaternions")
    return value / norm


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def root_origin_velocity(com_velocity, angular_velocity, quaternion):
    """Rigid-body point transport; uses the already-audited original base COM."""
    n = len(com_velocity)
    velocity = _array(com_velocity, (n, 3), "original_base_COM_velocity")
    omega = _array(angular_velocity, (n, 3), "world_angular_velocity")
    return velocity - np.cross(omega, quaternion_rotate(_quat(quaternion, n), COM_OFFSET_M))


def prepare_reference(time_s, position_w, quaternion_wxyz, interval_s):
    """Differentiate demonstrated positions only, before any interpolation.

    Three-point centered differences internally and second-order one-sided
    differences at source endpoints; never use a synthetic recovery/tail point.
    No temporal filter, fit, extrapolation or reference-state mutation.
    """
    time = _clock(time_s)
    position = _array(position_w, (len(time), 3), "reference_origin_position")
    quaternion = _quat(quaternion_wxyz, len(time))
    start, end = _array(interval_s, (2,), "source_interval")
    if not time[0] <= start < end <= time[-1]:
        raise ValueError("Source interval must lie within reference samples")
    selected = (time >= start-1e-9) & (time <= end+1e-9)
    t = _clock(time[selected])
    if not np.allclose(t[[0, -1]], [start, end], atol=1e-9, rtol=0):
        raise ValueError("Source endpoints must coincide with recorded reference samples")
    xy = position[selected, :2].copy()
    velocity = np.gradient(xy, t, axis=0, edge_order=2)
    forward = quaternion_rotate(quaternion[selected], np.array([1., 0., 0.]))
    if np.any(np.linalg.norm(forward[:, :2], axis=1) <= 1e-6):
        raise ValueError("Reference yaw is undefined at a vertical forward axis")
    yaw = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    return {"time_s": t.copy(), "xy_m": xy, "velocity_xy_m_s": velocity,
            "yaw_rad": yaw, "original_sample_indices": np.flatnonzero(selected)}


def decompose(reference, trace, *, speed_threshold_m_s=.25):
    time = _clock(trace["time_s"])
    dt = float(trace["physics_dt_s"])
    if not np.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(time), dt, atol=1e-9, rtol=0):
        raise ValueError("Native time must be consecutive physical sampling")
    if not np.isfinite(speed_threshold_m_s) or speed_threshold_m_s <= 0:
        raise ValueError("Speed validity threshold must be positive")
    n = len(time)
    position = _array(trace["position"], (n, 3), "measured_origin_position")
    quaternion = _quat(trace["quaternion"], n)
    origin_velocity = root_origin_velocity(trace["com_velocity"], trace["angular_velocity"], quaternion)
    rt = reference["time_s"]
    supported = (time >= rt[0]-1e-9) & (time <= rt[-1]+1e-9)
    indices = np.flatnonzero(supported)
    query = _clock(time[supported])
    # Tolerance covers accumulated float64 native clocks, never true extrapolation.
    query_ref = np.clip(query, rt[0], rt[-1])
    interpolate = lambda value: np.column_stack([np.interp(query_ref, rt, value[:, j]) for j in range(2)])
    target = interpolate(reference["xy_m"])
    velocity = interpolate(reference["velocity_xy_m_s"])
    target_speed = np.linalg.norm(velocity, axis=1)
    valid = target_speed >= speed_threshold_m_s
    tangent = np.full_like(velocity, np.nan)
    tangent[valid] = velocity[valid]/target_speed[valid, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    error = position[supported, :2] - target
    along, cross = np.sum(error*tangent, axis=1), np.sum(error*normal, axis=1)
    actual_velocity = origin_velocity[supported, :2]
    actual_speed = np.linalg.norm(actual_velocity, axis=1)
    course_valid = actual_speed >= speed_threshold_m_s
    reference_course = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    actual_course = reference_course + _wrap(np.arctan2(actual_velocity[:, 1], actual_velocity[:, 0])-reference_course)
    reference_yaw = np.interp(query_ref, rt, reference["yaw_rad"])
    forward = quaternion_rotate(quaternion[supported], np.array([1., 0., 0.]))
    yaw_valid = np.linalg.norm(forward[:, :2], axis=1) > 1e-6
    actual_yaw = reference_yaw + _wrap(np.arctan2(forward[:, 1], forward[:, 0])-reference_yaw)
    reference_course[~valid] = np.nan
    actual_course[~course_valid] = np.nan
    actual_yaw[~yaw_valid] = np.nan
    equivalent_time = np.full(len(query), np.nan)
    equivalent_time[valid] = along[valid]/target_speed[valid]
    return {"time_s": query, "native_sample_indices": indices, "native_source_interval_mask": supported,
            "physics_dt_s": np.array(dt), "reference_xy_m": target, "actual_xy_m": position[supported, :2].copy(),
            "error_xy_m": error, "along_m": along, "normal_m": cross,
            "total_error_m": np.linalg.norm(error, axis=1), "projection_valid": valid,
            "reference_origin_velocity_xy_m_s": velocity, "actual_origin_velocity_xy_m_s": actual_velocity,
            "reference_origin_speed_m_s": target_speed, "actual_origin_speed_m_s": actual_speed,
            "reference_course_rad": reference_course, "actual_course_rad": actual_course,
            "actual_course_valid": course_valid, "reference_yaw_rad": reference_yaw, "actual_yaw_rad": actual_yaw,
            "actual_yaw_valid": yaw_valid, "along_displacement_equivalent_time_s": equivalent_time,
            "course_error_rad": _wrap(actual_course-reference_course), "yaw_error_rad": _wrap(actual_yaw-reference_yaw)}


def persistent_onset(time, condition, window, *, duration_s=.1):
    """First retrospectively confirmed run spanning >=duration_s in a window."""
    time = _clock(time)
    condition = np.asarray(condition)
    if condition.shape != time.shape or condition.dtype.kind != "b" or not np.isfinite(duration_s) or duration_s < 0:
        raise ValueError("Expected boolean onset condition and nonnegative duration")
    selected = (time >= window[0]-1e-9) & (time <= window[1]+1e-9)
    eligible = np.flatnonzero(selected)
    start = None
    for i in eligible:
        if not condition[i]:
            start = None
        elif start is None:
            start = i
        if start is not None and time[i]-time[start] >= duration_s-1e-9:
            return {"time_s": float(time[start]), "confirmation_time_s": float(time[i]),
                    "already_active_at_window_start": bool(start == eligible[0])}
    return None


def _stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {"samples": int(len(values)), "signed_mean": float(values.mean()),
            "rmse": float(np.sqrt(np.mean(values**2))), "absolute_peak": float(np.max(np.abs(values))),
            "absolute_p95": float(np.quantile(np.abs(values), .95))}


def component_statistics(data, selected):
    """All component/total energy comparisons use identical tangent-valid samples."""
    valid = selected & data["projection_valid"]
    total = data["total_error_m"][valid]
    along, normal = data["along_m"][valid], data["normal_m"][valid]
    denominator = float(np.sum(total**2))
    return {
        "projection_valid_samples": int(valid.sum()),
        "total_error_m_all_samples": _stats(data["total_error_m"][selected]),
        "total_error_m_projection_valid": _stats(total),
        "along_m": _stats(along), "normal_m": _stats(normal),
        "along_energy_fraction": float(np.sum(along**2)/denominator) if denominator else None,
        "normal_energy_fraction": float(np.sum(normal**2)/denominator) if denominator else None,
        "maximum_squared_decomposition_residual_m2": float(np.max(np.abs(total**2-along**2-normal**2))) if len(total) else None,
        "behind_fraction": float(np.mean(along < 0)) if len(along) else None,
        "along_displacement_equivalent_time_s": _stats(data["along_displacement_equivalent_time_s"][valid]),
    }


def summarize(data, interval_s, *, duration_s=.1):
    time = data["time_s"]
    selected = (time >= interval_s[0]-1e-9) & (time <= interval_s[1]+1e-9)
    coverage = bool(time[0] <= interval_s[0]+1e-9 and time[-1] >= interval_s[1]-1e-9)
    result = {"interval_s": list(interval_s), "coverage_complete": coverage, "native_samples": int(selected.sum())}
    if not coverage or selected.sum() < 2:
        result["decomposition"] = None
        return result
    result["decomposition"] = component_statistics(data, selected)
    onset_conditions = {f"{name}_{threshold:g}m": condition for threshold in (.15, .30)
                        for name, condition in (("total_above", data["total_error_m"] > threshold),
                                                ("behind_beyond", data["along_m"] < -threshold),
                                                ("absolute_normal_above", np.abs(data["normal_m"]) > threshold))}
    result["onsets"] = {name: {"source_interval": persistent_onset(time, condition, [time[0], time[-1]], duration_s=duration_s),
                               "core": persistent_onset(time, condition, interval_s, duration_s=duration_s)}
                         for name, condition in onset_conditions.items()}
    result["origin_speed_timing"] = {}
    for label in ("reference", "actual"):
        speed = data[f"{label}_origin_speed_m_s"][selected]
        t = time[selected]
        result["origin_speed_timing"][label] = {"mean_m_s": float(speed.mean()), "minimum_m_s": float(speed.min()),
            "minimum_time_s": float(t[np.argmin(speed)]), "maximum_m_s": float(speed.max()), "maximum_time_s": float(t[np.argmax(speed)])}
    result["origin_speed_timing"]["fraction_below_80pct_same_time_reference"] = float(np.mean(
        data["actual_origin_speed_m_s"][selected] < .8*data["reference_origin_speed_m_s"][selected]))
    result["angular_timing"] = {}
    for kind in ("course", "yaw"):
        reference = data[f"reference_{kind}_rad"]
        anchor = float(np.interp(interval_s[0], time, reference))
        events = []
        for change in ((15., 30., 45.) if np.isfinite(anchor) else ()):
            threshold = anchor + np.radians(change)
            reference_event = persistent_onset(time, np.isfinite(reference) & (reference >= threshold), interval_s, duration_s=duration_s)
            actual = data[f"actual_{kind}_rad"]
            actual_event = persistent_onset(time, np.isfinite(actual) & (actual >= threshold), interval_s, duration_s=duration_s)
            events.append({"threshold_deg": float(np.degrees(threshold)), "change_from_reference_core_start_deg": change,
                           "reference": reference_event, "actual": actual_event,
                           "actual_minus_reference_crossing_time_s": (actual_event["time_s"]-reference_event["time_s"]
                               if reference_event and actual_event else None)})
        result["angular_timing"][kind] = {"reference_core_start_deg": float(np.degrees(anchor)) if np.isfinite(anchor) else None,
                                          "wrapped_error_deg": _stats(np.degrees(data[f"{kind}_error_rad"][selected])),
                                          "first_sustained_upward_crossings": events}
    return result


def load_pair(directory):
    directory = Path(directory)
    summary_path, trace_path = directory/"summary.json", directory/"pair_trace.npz"
    summary = json.loads(summary_path.read_text())
    spec = get_reference(summary["reference_id"])
    if (summary.get("algorithm") != "NMPC" or summary.get("reference_id") != "drift_drl_traj4" or summary.get("reference_spec") != spec
            or summary.get("fixed_mu") != .7 or summary.get("input_sha256", {}).get("model") != MODEL_SHA256
            or sha256(trace_path) != summary.get("artifacts", {}).get("compact_trace_sha256")):
        raise ValueError("Require audited selected-reference nominal NMPC compact pair with original COM model and matching byte hash")
    with np.load(trace_path, allow_pickle=False) as archive:
        compact = {name: archive[name].copy() for name in archive.files}
    reference_time = _clock(compact["reference_time_s"])
    if (len(reference_time) != spec["frames"] or abs(reference_time[0]) > 1e-9
            or not np.allclose(np.diff(reference_time), 1/spec["fps"], atol=1e-9, rtol=0)
            or abs(reference_time[-1]-spec["sample_span_s"]) > 1e-9):
        raise ValueError("Reference clock does not match the hash-locked catalog metadata")
    reference = prepare_reference(compact["reference_time_s"], compact["reference_position"], compact["reference_quaternion"], spec["source_demo_interval_s"])
    fields = ("time_s", "physics_dt_s", "position", "quaternion", "com_velocity", "angular_velocity")
    data = {engine: decompose(reference, {name: compact[f"{engine}_{name}"] for name in fields}) for engine in ("isaac", "mujoco")}
    return summary, compact, data, {"summary_sha256": sha256(summary_path), "compact_trace_sha256": sha256(trace_path)}


def render(cases, spec, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    for column, engine in enumerate(("isaac", "mujoco")):
        reference = cases["residual"][engine]
        time = reference["time_s"]
        for row, quantity in ((1, "origin_speed_m_s"), (2, "course_rad"), (3, "yaw_rad")):
            value = reference[f"reference_{quantity}"]
            axes[row, column].plot(time, np.degrees(value) if row > 1 else value, color="black", label="reference root origin")
        for label, color in (("baseline", "#D55E00"), ("residual", "#0072B2")):
            data = cases[label][engine]; time = data["time_s"]
            axes[0, column].plot(time, data["along_m"], color=color, label=f"{label} along")
            axes[0, column].plot(time, data["normal_m"], color=color, ls="--", label=f"{label} left normal")
            axes[1, column].plot(time, data["actual_origin_speed_m_s"], color=color, label=label)
            axes[2, column].plot(time, np.degrees(data["actual_course_rad"]), color=color, label=label)
            axes[3, column].plot(time, np.degrees(data["actual_yaw_rad"]), color=color, label=label)
        axes[0, column].set_title(engine.title())
        for row in range(4):
            axes[row, column].axvspan(*spec["evaluation_core_interval_s"], color=".5", alpha=.08)
            axes[row, column].grid(alpha=.2)
            axes[row, column].set_xlim(*spec["source_demo_interval_s"])
        axes[3, column].set_xlabel("Same physical time [s]")
    for row, label in enumerate(("Signed projection [m]", "Origin XY speed [m/s]", "Origin course [deg]", "Body heading/yaw [deg]")):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].legend(fontsize=8, ncol=2 if row == 0 else 1)
    fig.suptitle("Nominal NMPC: same-time root-origin geometry; not measured lateral slip or delay")
    fig.savefig(destination, dpi=160); plt.close(fig)


def run(baseline_dir, residual_dir, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Preserving existing XY component diagnostic output")
    cases, inputs, spec = {}, {}, None
    frozen_reference = None
    for name, directory in (("baseline", baseline_dir), ("residual", residual_dir)):
        summary, compact, cases[name], inputs[name] = load_pair(directory)
        current_reference = tuple(compact[f"reference_{key}"] for key in ("time_s", "position", "quaternion"))
        if frozen_reference is not None and not all(np.array_equal(a, b) for a, b in zip(frozen_reference, current_reference)):
            raise ValueError("Paired diagnostics must use exactly identical frozen reference vectors")
        frozen_reference, spec = current_reference, summary["reference_spec"]
    result = {"schema_version": 1, "kind": "same_time_root_origin_xy_components", "reference_id": spec["reference_id"],
              "source_demo_interval_s": spec["source_demo_interval_s"], "core_interval_s": spec["evaluation_core_interval_s"],
              "definitions": {"error": "actual root link-origin XY minus same-time reference root-origin XY",
                  "along": "dot(error, reference origin unit tangent); negative is behind",
                  "normal": "dot(error, left normal [-tangent_y,tangent_x]); not nearest-path distance or contact slip",
                  "derivative": "np.gradient(reference origin XY, recorded 50Hz time, edge_order=2), demonstrated-only; linearly interpolate XY and derivative independently onto native physical timestamps",
                  "endpoints": "Source endpoint derivatives are one-sided three-point estimates; recovery/tail samples never enter derivative; no extrapolation beyond demonstrated interval",
                  "measured_velocity": "original base COM velocity minus world omega cross rotated original COM offset; pure rigid-body point transport, no FK",
                  "original_com_offset_m": COM_OFFSET_M.tolist(), "speed_validity_threshold_m_s": .25,
                  "onset_duration_s": .1, "onsets": "Retrospectively confirmed first persistent threshold exceedance, not task failure times",
                  "angular_timing": "Exploratory first persistent upward +15/+30/+45deg crossings from reference core-start angle; crossing differences are not isolated controller latency",
                  "equivalent_time": "along displacement / same-time reference origin speed, not measured delay or alignment",
                  "angle_branch": "Measured course/yaw represented on the nearest reference angle branch; no robot-state rotation or alignment"},
              "registration_or_time_warp": False, "physics_or_fk_used": False, "height_or_flight_failure_gate": False,
              "heading_offset_zero_target": False, "causal_contact_engine_diagnosis": False,
              "input_sha256": inputs, "source_sha256": {"analyze_xy_error_components.py": sha256(__file__),
                  "nmpc_tracking_objective.py": sha256(ROOT/"scripts/nmpc_tracking_objective.py")},
              "cases": {name: {engine: summarize(data, spec["evaluation_core_interval_s"]) for engine, data in engines.items()}
                        for name, engines in cases.items()}}
    result["definitions"]["tilt_caveat"] = "Projected heading/course under large roll or pitch are geometric diagnostics, not causal angular delay or contact slip; a degenerate projected body forward axis has unavailable yaw"
    result["definitions"]["subwindows"] = "Closed descriptive intervals; shared endpoints appear in adjacent summaries and statistics are never pooled"
    for name, engines in cases.items():
        for engine, data in engines.items():
            result["cases"][name][engine]["core_subwindows"] = [
                {"interval_s": [start, end], "decomposition": component_statistics(data,
                    (data["time_s"] >= start-1e-9) & (data["time_s"] <= end+1e-9))}
                for start, end in ((1.44, 2.), (2., 3.), (3., 4.))]
    output.mkdir(parents=True)
    arrays = {f"{case}_{engine}_{field}": value for case, engines in cases.items() for engine, data in engines.items() for field, value in data.items()}
    np.savez_compressed(output/"time_series.npz", **arrays)
    render(cases, spec, output/"components_and_timing.png")
    result["artifacts"] = {name: sha256(output/name) for name in ("time_series.npz", "components_and_timing.png")}
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline_dir, args.residual_dir, args.output_dir)
    print(json.dumps({"status": "analyzed", "output_dir": str(args.output_dir)}))

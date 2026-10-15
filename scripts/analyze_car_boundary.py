"""Replay car root-XY errors and sustained onsets, without inferring tire slip.

Uses compact native-clock traces. Height is neither loaded nor used as a gate.
The existing core RMSE/peak verdict is unchanged; onset is an extra diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

import analyze_saved_xy as xy
from analyze_friction_grid import summarize_new_grid

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"reference": "#b78600", "isaac": "#0574b6", "mujoco": "#d55e00"}


def sustained_onset(time, error, threshold=.30, duration=.20):
    time, error = np.asarray(time, float), np.asarray(error, float)
    if (time.ndim != 1 or error.shape != time.shape or len(time) < 2
            or not np.isfinite(time).all() or not np.isfinite(error).all()
            or not np.all(np.diff(time) > 0) or not np.isfinite([threshold, duration]).all()
            or threshold <= 0 or duration <= 0):
        raise ValueError("Expected finite aligned increasing native samples and positive limits")
    high = error > threshold
    edges = np.flatnonzero(np.diff(np.r_[False, high, False]))
    first = np.flatnonzero(high)
    window = next(({"start_s": float(time[a]), "end_s": float(time[b-1]),
                    "observed_duration_s": float(time[b-1] - time[a]), "samples": int(b-a)}
                   for a, b in edges.reshape(-1, 2)
                   if time[b-1] - time[a] >= duration - 1e-9), None)
    return {"first_above_threshold_s": float(time[first[0]]) if len(first) else None,
            "first_sustained_window": window,
            "right_censored_last_high_run": bool(high[-1])}


def transitions(rows, engine):
    """Every adjacent sampled verdict change, never an assumed monotone threshold."""
    ordered = sorted(rows, key=lambda x: x["evaluation_mu_static_dynamic_equal"])
    output = []
    for left, right in zip(ordered, ordered[1:]):
        a, b = (item[engine]["large_reference_xy_screen"] for item in (left, right))
        if a is not None and b is not None and a != b:
            output.append({"mu_left": left["evaluation_mu_static_dynamic_equal"],
                           "mu_right": right["evaluation_mu_static_dynamic_equal"],
                           "left_large_xy": a, "right_large_xy": b})
    return output


def onset_phase(onset, core):
    window = onset["first_sustained_window"]
    if window is None:
        return None
    time = window["start_s"]
    return "before_turn_core" if time < core[0] else "within_turn_core" if time <= core[1] else "after_turn_core"


def curves(trace, case, engine):
    t, p = trace[f"{case}_{engine}_t"], trace[f"{case}_{engine}_xy"]
    xy.validate_path(t, p, engine)
    rt, rp = trace["reference_t"], trace["reference_xy"]
    keep = (t >= rt[0] - 1e-9) & (t <= rt[-1] + 1e-9)
    t, p = t[keep], p[keep]
    return t, np.linalg.norm(p - xy.interpolate(rt, rp, t), axis=1)


def compare_controls(trace, provenance, previous_path, selected):
    old = xy.arrays(previous_path)
    previous = json.loads(previous_path.with_suffix(".provenance.json").read_text())
    for key in ("checkpoint_sha256", "motion_sha256", "reference_sha256", "full_horizon_diagnostic"):
        if provenance.get(key) is None or provenance[key] != previous.get(key):
            raise ValueError("Repeated controls changed locked input/mode: " + key)
    rows = []
    for case, item in provenance["cases"].items():
        if not any(np.isclose(item["mu"], mu, atol=1e-9, rtol=0) for mu in selected):
            continue
        match = [name for name, old_item in previous["cases"].items()
                 if np.isclose(item["mu"], old_item["mu"], atol=1e-9, rtol=0)]
        if len(match) != 1:
            raise ValueError("Repeated control lacks one previous case: " + case)
        row = {"mu": item["mu"]}
        for engine in ("isaac", "mujoco"):
            t, p = trace[f"{case}_{engine}_t"], trace[f"{case}_{engine}_xy"]
            ot, op = old[f"{match[0]}_{engine}_t"], old[f"{match[0]}_{engine}_xy"]
            keep = (t >= ot[0] - 1e-9) & (t <= ot[-1] + 1e-9)
            delta = np.linalg.norm(p[keep] - xy.interpolate(ot, op, t[keep]), axis=1)
            row[engine] = {"observed_overlap_start_s": float(t[keep][0]),
                           "observed_overlap_end_s": float(t[keep][-1]),
                           "repeat_xy_rmse_m": float(np.sqrt(np.mean(delta**2))),
                           "repeat_xy_peak_m": float(delta.max())}
        rows.append(row)
    return {"previous_compact_trace_sha256": xy.sha(previous_path), "cases": rows}


def render_errors(trace, cases, output, threshold, duration):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
    if len(cases) > 6:
        raise ValueError("Boundary report supports at most six cases")
    for ax, (case, item) in zip(axes.flat, cases.items()):
        for engine in ("isaac", "mujoco"):
            t, error = curves(trace, case, engine)
            ax.plot(t, error, color=COLORS[engine], label=engine)
            onset = sustained_onset(t, error, threshold, duration)["first_sustained_window"]
            if onset:
                ax.axvline(onset["start_s"], color=COLORS[engine], ls=":", alpha=.7)
        ax.axhline(threshold, color="gray", ls="--", lw=.8)
        ax.axvspan(1.44, 4., color="gray", alpha=.10)
        ax.set(title=f"mu={item['mu']:.2f}", xlabel="time (s)", ylabel="reference root XY error (m)")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("Fixed actor; shaded turn core; dotted = first sustained XY offset")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def render_replay(trace, cases, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False, dpi=90)
    artists = []
    for ax, (case, item) in zip(axes.flat, cases.items()):
        paths = {"reference": (trace["reference_t"], trace["reference_xy"]),
                 **{e: (trace[f"{case}_{e}_t"], trace[f"{case}_{e}_xy"]) for e in ("isaac", "mujoco")}}
        all_xy = np.concatenate([p for _, p in paths.values()])
        center = .5 * (all_xy.max(axis=0) + all_xy.min(axis=0))
        half = max(float(np.ptp(all_xy, axis=0).max()) * .56, .5)
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
               xlabel="world X (m)", ylabel="world Y (m)", title=f"mu={item['mu']:.2f}")
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
        for engine, (t, p) in paths.items():
            ax.plot(p[:, 0], p[:, 1], color=COLORS[engine], alpha=.12, lw=1)
            line, = ax.plot([], [], color=COLORS[engine], lw=1.5, label=engine)
            dot, = ax.plot([], [], "o", color=COLORS[engine], ms=4)
            artists.append((t, p, line, dot))
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    label = fig.suptitle("Root-XY replay only; not a mesh/contact or wheel-heading rendering")
    fig.tight_layout(rect=(0, 0, 1, .95))
    end = float(trace["reference_t"][-1])
    times = np.r_[np.arange(0, end, .1), end]
    writer = PillowWriter(fps=10)
    with writer.saving(fig, str(output), dpi=90):
        for time in times:
            for t, p, line, dot in artists:
                # Interpolate only inside the recorded horizon. Completed paths
                # visibly hold their last state; never splice resets or extrapolate.
                current = np.clip(time, t[0], t[-1])
                point = xy.interpolate(t, p, np.array([current]))[0]
                history = np.r_[p[t < current], point[None]]
                line.set_data(history[:, 0], history[:, 1])
                dot.set_data([point[0]], [point[1]])
            label.set_text(f"Root-XY replay, t={time:.2f}s / {end:.2f}s; not wheel/contact rendering")
            writer.grab_frame()
    plt.close(fig)
    return {"sample_times_s": times.tolist(), "gif_fps": 10,
            "presentation_duration_s": len(times) / 10.,
            "note": "GIF clock quantized to 0.1s; numeric analysis uses native physics samples."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--previous-trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/car_boundary_refinement.json").read_text())
    trace = xy.arrays(args.trace)
    provenance = json.loads(args.trace.with_suffix(".provenance.json").read_text())
    if provenance.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint differs from boundary experiment")
    cases = provenance["cases"]
    result = summarize_new_grid(trace, provenance, {k: (v["mu"], "") for k, v in cases.items()})
    spec = config["additional_onset_diagnostic"]
    threshold, duration = spec["xy_error_above_m"], spec["minimum_contiguous_duration_s"]
    result["native_clock_xy_onsets"] = {case: {engine: sustained_onset(*curves(trace, case, engine), threshold, duration)
                                                for engine in ("isaac", "mujoco")} for case in cases}
    for engines in result["native_clock_xy_onsets"].values():
        for onset in engines.values():
            onset["first_sustained_onset_phase"] = onset_phase(onset, result["core_interval_s"])
    result["sampled_verdict_changes"] = {e: transitions(result["rows"], e) for e in ("isaac", "mujoco")}
    result["onset_specification"] = spec
    result["compact_trace_sha256"] = xy.sha(args.trace)
    result["analysis_source_sha256"] = xy.sha(Path(__file__).resolve())
    result["experiment_config_sha256"] = xy.sha(ROOT / "config/car_boundary_refinement.json")
    if args.previous_trace:
        result["control_repeats"] = compare_controls(trace, provenance, args.previous_trace, config["repeat_control_mu"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    render_errors(trace, cases, args.output_dir / "xy_error_time.png", threshold, duration)
    if args.gif:
        result["root_xy_replay"] = render_replay(trace, cases, args.output_dir / "root_xy_replay.gif")
        result["root_xy_replay"]["gif_sha256"] = xy.sha(args.output_dir / "root_xy_replay.gif")
    (args.output_dir / "boundary_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(cases), "sampled_verdict_changes": result["sampled_verdict_changes"]}))


if __name__ == "__main__":
    main()

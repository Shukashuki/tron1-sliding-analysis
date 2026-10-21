"""Render the four audited, measured CEM predictor-by-plant recordings.

This is a visualization-only replay: no simulation, interpolation, synthetic
episodes, alignment, or policy inference is performed. Native 200 Hz samples
are selected for a synchronized 25 fps, seven-second top-down video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np

from controller_metrics import heading_signals
from nmpc_tracking_objective import quaternion_rotate


PREDICTORS = {"mujoco": "CEM MuJoCo CPU predictor", "physx": "CEM PhysX predictor"}
PLANTS = {"isaac": "PhysX plant", "mujoco": "MuJoCo plant"}
TRACE_NAME = "prediction_transfer_trace.npz"
TRACE_FIELDS = ("time_s", "physics_dt_s", "position", "quaternion", "com_velocity",
                "wheel_pos", "wheel_quaternion_wxyz", "wheel_center_velocity_w")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_trace(arrays, prefix, *, measured):
    """Recheck a complete native recording and exactly replay its angle arrays."""
    names = TRACE_FIELDS + (("contact_count", "contact_force") if measured else ())
    require(all(prefix + name in arrays for name in names), "Incomplete trace: " + prefix)
    trace = {name: arrays[prefix + name] for name in names}
    n, dt = (1397, .005) if measured else (349, .02)
    shapes = {"time_s": (n,), "physics_dt_s": (), "position": (n, 3),
              "quaternion": (n, 4), "com_velocity": (n, 3), "wheel_pos": (n, 2, 3),
              "wheel_quaternion_wxyz": (n, 2, 4), "wheel_center_velocity_w": (n, 2, 3)}
    if measured:
        shapes.update(contact_count=(n, 2), contact_force=(n, 2, 3))
    for name, shape in shapes.items():
        value = np.asarray(trace[name])
        require(value.shape == shape and value.dtype.kind in "fiu" and np.isfinite(value).all(),
                "Invalid measured array: " + prefix + name)
    require(float(trace["physics_dt_s"]) == dt
            and np.allclose(trace["time_s"], np.arange(n)*dt, rtol=0, atol=1e-9),
            "Incomplete or resampled native clock: " + prefix)
    if measured:
        counts = trace["contact_count"]
        require(np.all(counts >= 0) and np.array_equal(counts, np.floor(counts)),
                "Invalid measured contact counts: " + prefix)
    signals = heading_signals(trace["quaternion"], trace["com_velocity"],
                              trace["wheel_quaternion_wxyz"], trace["wheel_center_velocity_w"])
    require(all(prefix + name in arrays and np.array_equal(value, arrays[prefix + name])
                for name, value in signals.items()), "Exact heading/speed replay failed: " + prefix)
    return trace, signals


def load_recordings(source_dir):
    summary = json.loads((source_dir / "summary.json").read_text())
    require(summary.get("kind") == "measured_cem_predictor_by_plant_nominal_2x2"
            and summary.get("plant_mu") == .7
            and summary.get("height_or_flight_failure_gate") is False
            and summary.get("heading_offset_is_mismatch") is False,
            "Require the measured, nominal, height-free CEM predictor-by-plant comparison")
    require(set(summary.get("conditions", {})) == set(PREDICTORS)
            and all(set(summary["conditions"][name]) == set(PLANTS) for name in PREDICTORS),
            "Require all four measured predictor-by-plant conditions")
    artifact = summary.get("artifacts", {}).get(TRACE_NAME, {})
    require(artifact.get("file") == TRACE_NAME
            and artifact.get("sha256") == sha256(source_dir / TRACE_NAME),
            "Compact recording SHA-256 differs from the audited summary")
    with np.load(source_dir / TRACE_NAME, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    reference = validate_trace(arrays, "reference_", measured=False)
    cases = {(predictor, plant): validate_trace(arrays, predictor + "_predictor_" + plant + "_", measured=True)
             for predictor in PREDICTORS for plant in PLANTS}
    return summary, reference, cases


def find_encoder():
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        try:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as error:
            raise RuntimeError("ffmpeg is required; install ffmpeg or imageio-ffmpeg") from error
    require(Path(encoder).is_file(), "The ffmpeg encoder is unavailable")
    return encoder


def render_video(summary, reference, cases, output_file):
    require(not output_file.exists(), "Preserving existing output; choose a new --output-file")
    encoder = find_encoder()
    matplotlib.rcParams["animation.ffmpeg_path"] = encoder
    reference_trace, _ = reference
    points = [reference_trace["position"][:, :2]]
    for trace, _ in cases.values():
        points.extend((trace["position"][:, :2], trace["wheel_pos"][:, :, :2].reshape(-1, 2)))
    points = np.concatenate(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.
    half_span = max(float(np.ptp(points, axis=0).max()) / 2. + .25, .5)
    low, high = center-half_span, center+half_span
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained", sharex=True, sharey=True)
    artists, valid = {}, {}
    for row, predictor in enumerate(PREDICTORS):
        for col, plant in enumerate(PLANTS):
            ax = axes[row, col]
            trace, signals = cases[(predictor, plant)]
            ax.plot(reference_trace["position"][:, 0], reference_trace["position"][:, 1],
                    "--", color=".65", lw=1.)
            path_line, = ax.plot([], [], color="#2672ae", lw=1.8)
            body = ax.quiver([0.], [0.], [.19], [0.], color="black", angles="xy",
                             scale_units="xy", scale=1., width=.006)
            course = ax.quiver([0.], [0.], [.16], [0.], color="#24936a", angles="xy",
                               scale_units="xy", scale=1., width=.004)
            wheels = [ax.plot([], [], color=color, lw=2.2)[0] for color in ("#d54843", "#8a54aa")]
            dot, = ax.plot([], [], "o", color="#2672ae", ms=4.)
            artists[(predictor, plant)] = (path_line, body, course, wheels, dot)
            body_axis = quaternion_rotate(trace["quaternion"], np.array([1., 0., 0.]))
            wheel_axles = quaternion_rotate(trace["wheel_quaternion_wxyz"], np.array([0., 1., 0.]))
            valid[(predictor, plant)] = {
                "body": np.linalg.norm(body_axis[:, :2], axis=-1) > 1e-6,
                "course": signals["speed_m_s"] >= signals["speed_threshold_m_s"],
                "wheel": np.linalg.norm(wheel_axles[:, :, :2], axis=-1) > 1e-6,
            }
            metrics = summary["conditions"][predictor][plant]
            ax.set(title=f"{PREDICTORS[predictor]} → {PLANTS[plant]}\n"
                         f"Core XY RMSE {metrics['reference_xy_rmse_m']:.3f} m | "
                         f"speed {100*metrics['mean_speed_retention_ratio']:.1f}%",
                   xlabel="World X [m]", ylabel="World Y [m]",
                   xlim=(low[0], high[0]), ylim=(low[1], high[1]))
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2)
    subtitle = ("Black: body | green: COM velocity | red/purple: wheel planes\n"
                "Faint wheels: measured contact gate not met; offsets/flight are not failure labels")
    title = fig.suptitle("traj_4 · μ=0.70 · physical time 0.00 s\n" + subtitle, fontsize=10)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    writer = FFMpegWriter(fps=25, codec="libx264", extra_args=["-crf", "23", "-pix_fmt", "yuv420p"],
                         metadata={"title": "Measured CEM predictor-by-plant transfer at mu=0.7",
                                   "comment": "Four measured native-clock episodes; visualization-only replay"})
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # xb reserves the requested new artifact without ever overwriting a prior one.
    with output_file.open("xb"):
        pass
    try:
        with writer.saving(fig, str(output_file), dpi=100):
            for time in np.arange(175) / 25.:
                for key, (trace, signals) in cases.items():
                    current = int(np.argmin(np.abs(trace["time_s"] - time)))
                    xy = trace["position"][:, :2]
                    path_line, body, course, wheels, dot = artists[key]
                    path_line.set_data(xy[:current+1, 0], xy[:current+1, 1])
                    dot.set_data([xy[current, 0]], [xy[current, 1]])
                    for arrow, name, length, mask in (
                            (body, "body_heading_rad", .19, "body"),
                            (course, "course_rad", .16, "course")):
                        angle = signals[name][current]
                        arrow.set_offsets(xy[current][None, :])
                        arrow.set_UVC([length*np.cos(angle)], [length*np.sin(angle)])
                        arrow.set_visible(bool(valid[key][mask][current]))
                    for side, line in enumerate(wheels):
                        angle = signals["wheel_rolling_plane_heading_rad"][current, side]
                        delta = .085*np.array([np.cos(angle), np.sin(angle)])
                        segment = trace["wheel_pos"][current, side, :2] + np.array([-delta, delta])
                        line.set_data(segment[:, 0], segment[:, 1])
                        contact = trace["contact_count"][current, side] > 0 and trace["contact_force"][current, side, 2] > 5.
                        line.set_alpha(1. if contact else .25)
                        line.set_visible(bool(valid[key]["wheel"][current, side]))
                title.set_text(f"traj_4 · μ=0.70 · physical time {time:.2f} s\n" + subtitle)
                writer.grab_frame()
    finally:
        plt.close(fig)
    return {"output_file": str(output_file), "sha256": sha256(output_file),
            "frames": 175, "fps": 25, "duration_s": 7., "native_sample_interpolation": False}


def run(args):
    require(not args.output_file.exists(), "Preserving existing output; choose a new --output-file")
    summary, reference, cases = load_recordings(args.source_dir)
    result = render_video(summary, reference, cases, args.output_file)
    print(json.dumps(result, indent=2))
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--output-file", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())

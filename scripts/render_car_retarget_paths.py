"""Compare the locked source car path and existing TRON1 retarget in 2-D.

Offline array replay only. No new retargeting, IK, policy, dynamics or contacts.
The source is a CARLA recording with an estimated, not original, clock.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from controller_references import PRIMARY_REFERENCE_ID, verify_reference_pair
from nmpc_tracking_objective import quaternion_rotate
from prepare_controller_reference import (
    DEFAULT_SOURCE_ROOT, PINNED_SOURCE_FILES, _source_modules,
    sha256, verify_source_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/robot_reference.npz")
DEFAULT_MOTION = DEFAULT_REFERENCE.with_name("motion.npz")
COLORS = {"path": "#0072B2", "body": "#161616", "velocity": "#009E73",
          "left": "#D55E00", "right": "#8B4BA6"}
ARROW_LENGTH_M = .18


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _bounds(*positions):
    xy = np.concatenate([np.asarray(value).reshape(-1, 2) for value in positions])
    center = .5*(xy.min(axis=0)+xy.max(axis=0))
    half = max(.5, .5*float(np.ptp(xy, axis=0).max())+.3)
    return {"xlim": [float(center[0]-half), float(center[0]+half)],
            "ylim": [float(center[1]-half), float(center[1]+half)]}


def load_data(reference_file=DEFAULT_REFERENCE, motion_file=DEFAULT_MOTION,
              source_root=DEFAULT_SOURCE_ROOT):
    """Verify locked archives and reconstruct only the historical affine mapping."""
    spec = verify_reference_pair(motion_file, reference_file, PRIMARY_REFERENCE_ID)
    source_inputs = verify_source_inputs(PRIMARY_REFERENCE_ID, source_root=source_root)
    config = source_inputs["configuration"]
    require(config["spatial_scale"] == .08 and config["time_scale"] == 2., "Unexpected historical scales")
    with _source_modules(source_inputs["source_root"]) as modules:
        decoded = modules.turn.decode_csv(source_inputs["payload"])
        selected, selection_evidence = modules.turn.selected_source(decoded, config)
    robot, motion = _load(reference_file), _load(motion_file)
    phase = np.asarray(robot["source_phase"])
    mask = phase == 0
    require(mask.sum() == 249 and np.array_equal(np.flatnonzero(mask), np.arange(249)),
            "Require only the 249 demonstrated source frames, no synthetic stop/tail")
    time = robot["time_s"][mask]
    require(np.array_equal(time, np.arange(249)/50.) and
            np.array_equal(time, motion["time_s"][mask]), "Locked 50 Hz clock differs")
    ids = selected["unique_pose_indices"]
    native_t = selected["time_s_estimated"][ids]
    query = time/2.
    yaw0 = float(selected["yaw_rad"][ids[0]])
    rotation = np.array([[np.cos(yaw0), np.sin(yaw0)], [-np.sin(yaw0), np.cos(yaw0)]])
    raw_xy = selected["position_xy_m"][ids]
    raw_velocity = selected["velocity_xy_m_s"][ids]
    interpolate = lambda value: np.column_stack([np.interp(query, native_t, value[:, j]) for j in range(value.shape[1])])
    car_xy = interpolate(.08*(raw_xy-raw_xy[0])@rotation.T)
    car_native_xy = interpolate(raw_xy)
    car_velocity = interpolate(.04*raw_velocity@rotation.T)
    car_yaw = np.interp(query, native_t, selected["yaw_rad"][ids])-yaw0
    root_target = robot["root_target_position_m"][mask, :2]
    error64 = float(np.max(np.abs(car_xy-root_target)))
    require(error64 < 1e-12, "Affine car mapping differs from locked float64 root target")
    bodies = tuple(motion["body_names"].tolist())
    root_id = bodies.index("base_Link")
    wheel_ids = [bodies.index(name) for name in ("wheel_L_Link", "wheel_R_Link")]
    robot_xy = motion["body_pos_w"][mask, root_id, :2].astype(float)
    error32 = float(np.max(np.abs(car_xy-robot_xy)))
    require(error32 <= 2e-7, "Existing float32 motion differs beyond locked export precision")
    root_quaternion = motion["body_quat_w"][mask, root_id].astype(float)
    wheel_quaternion = motion["body_quat_w"][mask][:, wheel_ids].astype(float)
    root_quaternion /= np.linalg.norm(root_quaternion, axis=-1, keepdims=True)
    wheel_quaternion /= np.linalg.norm(wheel_quaternion, axis=-1, keepdims=True)
    forward = quaternion_rotate(root_quaternion, np.array([1., 0., 0.]))
    axle = quaternion_rotate(wheel_quaternion, np.array([0., 1., 0.]))
    rolling = np.cross(axle, np.array([0., 0., 1.]))
    robot_velocity = motion["body_lin_vel_w"][mask, root_id].astype(float)
    arrays = {"time_s": time, "source_time_s": query,
        "car_common_xy_m": car_xy, "car_native_xy_m": car_native_xy,
        "car_body_heading_rad": car_yaw,
        "car_native_body_heading_rad": car_yaw+yaw0,
        "car_velocity_common_m_s": car_velocity,
        "car_velocity_native_m_s": interpolate(raw_velocity),
        "car_course_rad": np.arctan2(car_velocity[:, 1], car_velocity[:, 0]),
        "car_native_course_rad": np.arctan2(interpolate(raw_velocity)[:, 1], interpolate(raw_velocity)[:, 0]),
        "car_velocity_valid": np.linalg.norm(car_velocity, axis=1) >= .25,
        "robot_xy_m": robot_xy,
        "robot_body_heading_rad": np.arctan2(forward[:, 1], forward[:, 0]),
        "robot_body_valid": np.linalg.norm(forward[:, :2], axis=-1) > 1e-6,
        "robot_com_velocity_w_m_s": robot_velocity,
        "robot_course_rad": np.arctan2(robot_velocity[:, 1], robot_velocity[:, 0]),
        "robot_velocity_valid": np.linalg.norm(robot_velocity[:, :2], axis=-1) >= .25,
        "robot_wheel_xy_m": motion["body_pos_w"][mask][:, wheel_ids, :2].astype(float),
        "robot_wheel_heading_rad": np.arctan2(rolling[:, :, 1], rolling[:, :, 0]),
        "robot_wheel_valid": np.linalg.norm(rolling[:, :, :2], axis=-1) > 1e-6,
        "selected_source_time_s_estimated": selected["time_s_estimated"],
        "selected_source_position_xy_m": selected["position_xy_m"],
        "selected_source_velocity_xy_m_s": selected["velocity_xy_m_s"],
        "selected_source_yaw_rad": selected["yaw_rad"],
        "selected_source_unique_pose_indices": ids,
        "source_to_common_rotation": rotation,
        "wheel_order": np.array(["left", "right"])}
    common_bounds = _bounds(car_xy, robot_xy, arrays["robot_wheel_xy_m"])
    summary = {"schema_version": 1, "kind": "locked_source_car_vs_existing_retarget_2d_paths",
        "reference_id": PRIMARY_REFERENCE_ID, "frames": 249, "fps": 50,
        "display_interval_s": [0., 4.96], "evaluation_core_interval_s": [1.44, 4.],
        "excluded_synthetic_recovery_and_tail_frames": 100,
        "source_row_interval_inclusive": [1535, 1595],
        "source_repository": spec["source"]["repository"], "source_commit": source_inputs["head"],
        "source_clock": "estimated from displacement/velocity, not original timestamps",
        "source_is_physical_car_capture": False,
        "source_selection_evidence": selection_evidence,
        "input_sha256": {"robot_reference.npz": sha256(reference_file), "motion.npz": sha256(motion_file),
            "source_csv": sha256(source_inputs["csv_file"]), "source_config": sha256(source_inputs["configuration_file"]),
            "source_license": sha256(source_inputs["license_file"])},
        "pinned_decoder_source_sha256": dict(PINNED_SOURCE_FILES),
        "mapping": {"source_handedness": "CARLA (x,y) -> Z-up planar (x,-y); yaw -> -yaw",
            "spatial_scale": .08, "time_scale": 2., "velocity_scale": .04,
            "initial_position_subtracted_m": raw_xy[0].tolist(),
            "initial_source_yaw_rad": yaw0, "initial_heading_rotation_matrix": rotation.tolist(),
            "query_source_time": "task_time / 2", "additional_alignment_or_time_warp": False},
        "mapping_audit": {"source_common_vs_locked_root_target_max_abs_m": error64,
            "source_common_vs_export_float32_max_abs_m": error32,
            "float64_tolerance_m": 1e-12, "float32_tolerance_m": 2e-7},
        "plot_settings": {"colors": dict(COLORS), "figure_size_inches": [12., 5.5],
            "common_arrow_length_m": ARROW_LENGTH_M, "native_arrow_length_axis_span_fraction": .04,
            "wheel_half_length_m": .065, "aspect": "equal", "grid_alpha": .18,
            "common_axes_shared": True, "shared_common_bounds": common_bounds,
            "native_axes_shared": False, "native_car_bounds": _bounds(car_native_xy),
            "native_robot_bounds": _bounds(robot_xy, arrays["robot_wheel_xy_m"])},
        "car_wheel_orientation_available": False, "source_steer_is_normalized_control_not_wheel_angle": True,
        "robot_wheel_heading_definition": "world wheel +Y axle crossed with world +Z, independent of axle spin",
        "robot_velocity_definition": "Existing exported original-link COM world XY velocity; not root-position differentiation",
        "no_new_retarget": True, "ik_solver_called": False, "dynamics_stepping": False,
        "new_training": False, "new_policy_inference": False, "contact_measurement_available": False,
        "heading_offset_is_mismatch": False,
        "limitations": ["The source car is a CARLA simulation; clock reconstruction is estimated.",
            "Car wheel headings/contact patches are absent: normalized steer is not converted to tire angle.",
            "Main panels share robot-scale world axes; source translation/rotation, spatial x0.08 and time x2 are the existing locked mapping, not a new alignment.",
            "Native-scale panels intentionally use separate axis limits and units remain source meters versus robot meters.",
            "The same source-root XY path is preserved by design; this is kinematic retarget playback, not evidence of physical tracking or contact slip."]}
    return arrays, summary


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _figure(arrays, summary, native=False):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=not native, sharey=not native)
    settings = summary["plot_settings"]
    for side, ax in enumerate(axes):
        bounds = (settings["native_car_bounds"] if side == 0 else settings["native_robot_bounds"]) if native else settings["shared_common_bounds"]
        ax.set_xlim(bounds["xlim"]); ax.set_ylim(bounds["ylim"])
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=settings["grid_alpha"])
        ax.set_xlabel("Source world X [m]" if native and side == 0 else "Robot-scale X [m]")
        ax.set_ylabel("Source world Y, Z-up [m]" if native and side == 0 else "Robot-scale Y [m]")
        ax.set_title("Original car source" + (" | native scale" if native else " | mapped to robot scale") if side == 0 else "Existing TRON1 retarget | body + separate wheels", fontsize=10)
    caption = ("Native scales: source meters vs robot meters; axis limits NOT shared" if native else
               "Shared axes / scale / colors / arrow lengths | historical mapping: spatial x0.08, time x2")
    fig.suptitle("Original car vs existing TRON1 retarget | source-only interval 0-4.96 s\n" + caption, fontsize=11, y=.97)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COLORS["path"], label="Root path; thick = core"),
        Line2D([], [], color=COLORS["body"], label="Body heading"),
        Line2D([], [], color=COLORS["velocity"], label="Velocity direction"),
        Line2D([], [], color=COLORS["left"], label="Left wheel plane"),
        Line2D([], [], color=COLORS["right"], label="Right wheel plane")], loc="lower center", ncol=5, fontsize=8)
    fig.text(.5, .055, "Car wheel orientation unavailable; no steer-to-angle conversion. Kinematic replay only; no contact inference.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.08, right=.97, bottom=.20, top=.80, wspace=.22)
    return fig, axes


def render_comparison(arrays, summary, output, native=False):
    plt = _plt(); fig, axes = _figure(arrays, summary, native)
    core = (arrays["time_s"] >= 1.44-1e-9) & (arrays["time_s"] <= 4.+1e-9)
    indices = np.arange(0, 249, 20)
    for side, ax in enumerate(axes):
        robot = side == 1
        xy = arrays["robot_xy_m" if robot else "car_native_xy_m" if native else "car_common_xy_m"]
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS["path"], lw=1, alpha=.55)
        ax.plot(xy[core, 0], xy[core, 1], color=COLORS["path"], lw=2.5)
        length = .04*float(np.diff(ax.get_xlim())[0]) if native else ARROW_LENGTH_M
        for name, color, mask in (("robot_body_heading_rad" if robot else "car_native_body_heading_rad" if native else "car_body_heading_rad", COLORS["body"], arrays["robot_body_valid"] if robot else np.ones(249, dtype=bool)),
                                  ("robot_course_rad" if robot else "car_native_course_rad" if native else "car_course_rad", COLORS["velocity"], arrays["robot_velocity_valid"] if robot else arrays["car_velocity_valid"])):
            selected = indices[mask[indices]]; angle = arrays[name][selected]
            ax.quiver(xy[selected, 0], xy[selected, 1], length*np.cos(angle), length*np.sin(angle),
                      angles="xy", scale_units="xy", scale=1., color=color, width=.004)
        if robot:
            for index in indices:
                for wheel, color in enumerate((COLORS["left"], COLORS["right"])):
                    if arrays["robot_wheel_valid"][index, wheel]:
                        heading = arrays["robot_wheel_heading_rad"][index, wheel]
                        delta = .065*np.array([np.cos(heading), np.sin(heading)])
                        segment = arrays["robot_wheel_xy_m"][index, wheel]+np.array([-delta, delta])
                        ax.plot(segment[:, 0], segment[:, 1], color=color, lw=1.6)
    fig.savefig(output, dpi=150); plt.close(fig)


def render_video(arrays, summary, output):
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    plt = _plt()
    if not FFMpegWriter.isAvailable():
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    fig, axes = _figure(arrays, summary)
    artists = []
    for side, ax in enumerate(axes):
        path, = ax.plot([], [], color=COLORS["path"], lw=1, alpha=.6)
        core, = ax.plot([], [], color=COLORS["path"], lw=2.6)
        body = ax.quiver([0.], [0.], [.18], [0.], angles="xy", scale_units="xy", scale=1., color=COLORS["body"], width=.006)
        velocity = ax.quiver([0.], [0.], [.18], [0.], angles="xy", scale_units="xy", scale=1., color=COLORS["velocity"], width=.004)
        wheels = [ax.plot([], [], color=color, lw=2.5)[0] for color in (COLORS["left"], COLORS["right"])] if side else []
        clock = ax.text(.03, .96, "", transform=ax.transAxes, va="top", fontsize=9,
                        bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        artists.append((path, core, body, velocity, wheels, clock))
    writer = FFMpegWriter(fps=50, codec="libx264", bitrate=2200, extra_args=["-pix_fmt", "yuv420p"],
                         metadata={"title": "Locked source car and existing TRON1 retarget paths", "comment": "249 phase-0 frames; no new retarget, physics or policy"})
    with writer.saving(fig, str(output), dpi=100):
        for index, time in enumerate(arrays["time_s"]):
            for side, (path, core, body, velocity, wheels, clock) in enumerate(artists):
                tag = "robot" if side else "car"
                xy = arrays["robot_xy_m" if side else "car_common_xy_m"]
                path.set_data(xy[:index+1, 0], xy[:index+1, 1])
                selected = (arrays["time_s"][:index+1] >= 1.44-1e-9) & (arrays["time_s"][:index+1] <= 4.+1e-9)
                core.set_data(xy[:index+1, 0][selected], xy[:index+1, 1][selected])
                for arrow, key, mask in ((body, tag+"_body_heading_rad", arrays["robot_body_valid"] if side else np.ones(249, dtype=bool)),
                                         (velocity, tag+"_course_rad", arrays[tag+"_velocity_valid"])):
                    angle = arrays[key][index]; arrow.set_offsets(xy[index][None, :])
                    arrow.set_UVC([.18*np.cos(angle)], [.18*np.sin(angle)]); arrow.set_visible(bool(mask[index]))
                for wheel, line in enumerate(wheels):
                    angle = arrays["robot_wheel_heading_rad"][index, wheel]
                    delta = .065*np.array([np.cos(angle), np.sin(angle)])
                    segment = arrays["robot_wheel_xy_m"][index, wheel]+np.array([-delta, delta])
                    line.set_data(segment[:, 0], segment[:, 1]); line.set_visible(bool(arrays["robot_wheel_valid"][index, wheel]))
                clock.set_text(f"Task t={time:.2f} s | source estimated t={arrays['source_time_s'][index]:.2f} s")
            writer.grab_frame()
    plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh output directory")
    arrays, summary = load_data(args.reference_file, args.motion_file, args.source_root)
    output.mkdir(parents=True)
    np.savez_compressed(output/"derived_paths.npz", **arrays)
    render_comparison(arrays, summary, output/"comparison.png")
    render_comparison(arrays, summary, output/"native_scale.png", native=True)
    files = ["derived_paths.npz", "comparison.png", "native_scale.png"]
    if not args.skip_video:
        render_video(arrays, summary, output/"comparison.mp4"); files.append("comparison.mp4")
    summary["video"] = {"fps": 50, "frames": 0 if args.skip_video else 249,
                        "last_sample_time_s": 4.96, "encoded_duration_s": None if args.skip_video else 4.98}
    summary["analysis_source_sha256"] = sha256(__file__)
    summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in files}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    result.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": "rendered_locked_source_and_existing_retarget", "frames": result["frames"],
                      "new_retarget": False, "mapping_audit": result["mapping_audit"]}))

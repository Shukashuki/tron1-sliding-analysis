"""Render source-keypoint versus real TRON1-mesh FK replay, top-down only.

The optional model-free mode renders the recorded robot FK sites schematically.
All replay is kinematic: no mj_step, action replay, physics or contact forces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image


COLORS = ("#14885b", "#8152bd")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def bounds(trace):
    points = np.concatenate((trace["human_positions_robot_scale_m"][..., :2].reshape(-1, 2),
                             trace["robot_site_positions_m"][..., :2].reshape(-1, 2)))
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    half = max(.7, float(np.max(upper - lower)) / 2 + .3)
    return center, half


def plot_angles(experiment, traces, output):
    fig, axes = plt.subplots(len(traces), 2, figsize=(12, 2.1 * len(traces)), squeeze=False)
    for row, (report, trace) in enumerate(zip(experiment["clips"], traces)):
        t = trace["time_s"]
        for side, axis in enumerate(axes[row]):
            for kind, style in (("human", "-"), ("robot", "--")):
                angle = trace[("human_course_foot_axial_deg" if kind == "human" else "robot_course_wheel_axial_deg")][:, side]
                mask = trace[kind + "_valid_slow_sensitivity"][:, side]
                axis.plot(t, np.where(mask, angle, np.nan), style, color=COLORS[side],
                          alpha=1 if kind == "human" else .6, lw=1.3,
                          label="Human ankle-to-toe" if kind == "human" else "GMR wheel")
            axis.axhline(20, color="gray", lw=.6, ls=":")
            axis.axhline(-20, color="gray", lw=.6, ls=":")
            axis.set(ylim=(-95, 95), xlim=(t[0], t[-1]), ylabel="axial offset (deg)")
            if row == 0:
                axis.legend(loc="lower left", fontsize=7)
            counts = [int(trace[kind + "_valid_common"][:, side].sum()) for kind in ("human", "robot")]
            action = report["clip"].split("_", 2)[2].rsplit("_", 2)[0]
            axis.set_title(f"{action} | {'Left' if side == 0 else 'Right'} | 0.8 m/s valid H/R: {counts[0]}/{counts[1]}", fontsize=8)
            axis.grid(alpha=.15)
    axes[-1, 0].set_xlabel("source time (s)")
    axes[-1, 1].set_xlabel("source time (s)")
    fig.suptitle("Near-ground heading features at >=0.2 m/s (slow-speed sensitivity only)\n"
                 "Human: robot-leg scale, geometry proxy; robot: cylinder-bottom proxy. Not measured sliding.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(output / "near_ground_offsets.png", dpi=135)
    plt.close(fig)


def plot_topdowns(experiment, traces, output):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.5))
    for axis, report, trace in zip(axes.flat, experiment["clips"], traces):
        axis.plot(*trace["root_target_m"][:, :2].T, color="orange", lw=2, label="Root target")
        axis.plot(*trace["qpos"][:, :2].T, "k--", lw=1, label="GMR root")
        for side in range(2):
            points = trace["robot_site_positions_m"][:, 5 + side, :2]
            axis.plot(*points.T, color=COLORS[side], alpha=.6, lw=1)
            stride = max(1, len(points) // 12)
            for i in range(0, len(points), stride):
                angle = np.radians(trace["robot_wheel_heading_deg"][i, side])
                direction = .11 * np.array([np.cos(angle), np.sin(angle)])
                axis.plot(*(points[i, None] + np.array([-1, 1])[:, None] * direction).T,
                          color=COLORS[side], lw=.9)
        axis.set_title(report["clip"].split("_", 2)[2].rsplit("_", 2)[0], fontsize=9)
        axis.set_aspect("equal")
        axis.grid(alpha=.2)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("TRON1 GMR kinematic reference: root XY and wheel rolling-plane axes\nNo dynamics, rolling constraint, contact forces or policy", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output / "topdown_overview.png", dpi=140)
    plt.close(fig)


def render_clip(report, trace, model_path, output, fps, schematic=False):
    clip = Path(report["clip"]).stem
    times = trace["time_s"]
    source = trace["human_positions_robot_scale_m"]
    parents = trace["human_parent_indices"]
    names = list(trace["human_joint_names"])
    ankles = [names.index(name) for name in ("lfoot", "rfoot")]
    toes = [names.index(name) for name in ("ltoes", "rtoes")]
    center, half = bounds(trace)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=90)
    fig.subplots_adjust(top=.82, bottom=.23, wspace=.17, left=.07, right=.97)
    for axis in axes:
        axis.set(xlim=(center[0] - half, center[0] + half), ylim=(center[1] - half, center[1] + half))
        axis.set_aspect("equal")
        axis.grid(alpha=.2)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
    axes[0].set_title("HDM05 keypoints (robot-leg scale)", fontsize=10)
    axes[1].set_title("TRON1 GMR: " + ("recorded FK schematic" if schematic else "mesh / qpos kinematic playback"), fontsize=10)
    human_lines = []
    for child, parent in enumerate(parents):
        if parent >= 0:
            human_lines.append((child, int(parent), axes[0].plot([], [], color="#92999d", alpha=.65, lw=1)[0]))
    pelvis = axes[0].scatter([], [], c="orange", s=35, zorder=4)
    ankle_marks = [axes[0].scatter([], [], c=color, s=32, zorder=4) for color in COLORS]
    foot_lines = [axes[0].plot([], [], color=color, lw=3)[0] for color in COLORS]
    human_path = axes[0].plot([], [], color="orange", alpha=.6, lw=1)[0]
    labels = [fig.text(.08 + .48 * side, .10, "", fontsize=8) for side in range(2)]
    clock = fig.text(.5, .88, "", ha="center", fontsize=11)
    fig.suptitle(clip.replace("HDM_bd_", ""), fontsize=13, y=.98)
    fig.text(.5, .015, "Green=Left, purple=Right. Gamma: signed course minus heading; '*' = near-ground + speed >=0.2 m/s.\n"
             "BVH human-skeleton-standardized source. No dynamics/contact-force validation. Wheel spin frozen.",
             ha="center", va="bottom", fontsize=7.5)
    renderer = None
    model = data = camera = None
    if not schematic:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        model.vis.global_.orthographic = 1
        model.vis.global_.fovy = 2 * half
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.orthographic = True
        camera.lookat[:] = [center[0], center[1], 0]
        camera.distance, camera.azimuth, camera.elevation = 5., 90., -90.
        renderer = mujoco.Renderer(model, height=440, width=440)
        robot_image = axes[1].imshow(np.ones((440, 440, 3)),
                                    extent=(center[0] - half, center[0] + half, center[1] - half, center[1] + half))
        wheel_marks = [axes[1].scatter([], [], c=color, s=23, zorder=4) for color in COLORS]
        wheel_axes = [axes[1].plot([], [], color=color, lw=2.5, zorder=3)[0] for color in COLORS]
    else:
        robot_lines = [axes[1].plot([], [], "o-", color=color, lw=3, markersize=4)[0] for color in COLORS]
        wheel_lines = [axes[1].plot([], [], color=color, lw=5)[0] for color in COLORS]
    robot_path = axes[1].plot([], [], color="orange", lw=1, alpha=.8)[0]

    def draw(frame):
        for child, parent, line in human_lines:
            line.set_data(source[frame, [parent, child], :2].T)
        pelvis.set_offsets(source[frame, 0, :2])
        human_path.set_data(source[:frame + 1, 0, :2].T)
        robot_path.set_data(trace["qpos"][:frame + 1, :2].T)
        for side in range(2):
            ankle_marks[side].set_offsets(source[frame, ankles[side], :2])
            foot_lines[side].set_data(source[frame, [ankles[side], toes[side]], :2].T)
            gamma_h = trace["human_course_foot_offset_deg"][frame, side]
            gamma_r = trace["robot_course_wheel_offset_deg"][frame, side]
            hvalid = trace["human_valid_slow_sensitivity"][frame, side]
            rvalid = trace["robot_valid_slow_sensitivity"][frame, side]
            htext = f"{gamma_h:+.1f}*" if hvalid else "masked"
            rtext = f"{gamma_r:+.1f}*" if rvalid else "masked"
            labels[side].set_text(f"{'Left' if side == 0 else 'Right'} gamma H/R: "
                                  f"{htext} / {rtext} deg\n"
                                  f"Wheel bottom: {trace['robot_wheel_bottom_m'][frame, side]:+.3f} m (geometry only)")
        clock.set_text(f"t={times[frame]:.2f} s | root speed H/R "
                       f"{trace['human_root_speed_robot_scale_m_s'][frame]:.2f}/{trace['robot_root_speed_m_s'][frame]:.2f} m/s")
        if renderer is not None:
            import mujoco
            data.qpos[:] = trace["qpos"][frame]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            if (not renderer.scene.camera[0].orthographic
                    or not np.isclose(renderer.scene.camera[0].frustum_top, half, atol=1e-6)):
                raise ValueError("Mesh camera does not share the declared metric XY bounds")
            robot_image.set_data(renderer.render())
            for side in range(2):
                position = trace["robot_site_positions_m"][frame, 5 + side, :2]
                wheel_marks[side].set_offsets(position)
                angle = np.radians(trace["robot_wheel_heading_deg"][frame, side])
                direction = .127 * np.array([np.cos(angle), np.sin(angle)])
                wheel_axes[side].set_data((position + np.array([-1, 1])[:, None] * direction).T)
        else:
            sites = trace["robot_site_positions_m"][frame]
            for side in range(2):
                robot_lines[side].set_data(sites[[0, 1 + side, 3 + side, 5 + side], :2].T)
                angle = np.radians(trace["robot_wheel_heading_deg"][frame, side])
                direction = .127 * np.array([np.cos(angle), np.sin(angle)])
                wheel_lines[side].set_data((sites[5 + side, :2] + np.array([-1, 1])[:, None] * direction).T)
        return []

    frames = np.unique(np.r_[np.searchsorted(times, np.arange(0, times[-1], 1 / fps)), len(times) - 1])
    middle = int(frames[len(frames) // 2])
    draw(middle)
    fig.savefig(output / (clip + ".png"), dpi=120)
    try:
        animation = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
        animation.save(output / (clip + ".gif"), writer=PillowWriter(fps=fps))
    finally:
        if renderer is not None:
            renderer.close()
        plt.close(fig)
    return {"clip": report["clip"], "frames_rendered": len(frames), "gif_fps": fps,
            "mode": "recorded_FK_schematic" if schematic else "MuJoCo_mj_forward_mesh_only",
            "gif_sha256": sha(output / (clip + ".gif")), "dynamics_stepped": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--schematic", action="store_true", help="Replay recorded FK sites without licensed model/GL")
    parser.add_argument("--clip", nargs="+", help="Optional report clip names to render animations for")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Preserving existing render directory")
    if not args.schematic and args.model is None:
        parser.error("Mesh replay requires --model, or select --schematic")
    if not np.isfinite(args.fps) or not 1 <= args.fps <= 30:
        parser.error("Expected rendering FPS in [1, 30]")
    experiment = json.loads((args.experiment_dir / "experiment.json").read_text())
    if not args.schematic and sha(args.model) != experiment["model_sha256"]:
        raise ValueError("Renderer model differs from GMR model")
    traces = []
    for report in experiment["clips"]:
        trace_path = args.experiment_dir / Path(report["clip"]).stem / "retarget_trace.npz"
        if sha(trace_path) != report["trace_sha256"]:
            raise ValueError("Trace content differs from experiment record")
        traces.append(load(trace_path))
    output.mkdir(parents=True, exist_ok=True)
    plot_angles(experiment, traces, output)
    plot_topdowns(experiment, traces, output)
    records = []
    for report, trace in zip(experiment["clips"], traces):
        if args.clip and report["clip"] not in args.clip:
            continue
        records.append(render_clip(report, trace, args.model, output, args.fps, args.schematic))
        print("Rendered " + report["clip"], flush=True)
    (output / "render_audit.json").write_text(json.dumps({"renders": records,
        "renderer_sha256": sha(Path(__file__)), "experiment_sha256": sha(args.experiment_dir / "experiment.json"),
        "note": "GIF sample display uses nearest following source frame at requested presentation FPS. Scientific metrics use native BVH clock, not GIF timing."}, indent=2) + "\n")


if __name__ == "__main__":
    main()

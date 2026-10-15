"""Wall-clock synchronized 1x/2x/3x human-keypoint and TRON1 mesh replay.

Each reference uses its changed physical time array, not altered GIF FPS.
Finished references hold a clearly labeled final pose without looping or
claiming further observed motion. No simulation steps or policy actions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import mujoco
import numpy as np
from PIL import Image

from hdm05_gmr import ROOT, sha
from retime_hdm05_gmr import read_npz
from render_hdm05_gmr import COLORS, bounds


def sample_at_wall_time(times, wall_time):
    if not np.isfinite(wall_time):
        raise ValueError("Expected a finite presentation clock")
    if wall_time < times[0]:
        raise ValueError("Cannot render before the reference starts")
    active = wall_time <= times[-1] + 1e-10
    index = min(int(np.searchsorted(times, wall_time)), len(times) - 1)
    return index, active


def render_one(clip, source, cases, model, output, fps):
    stem = Path(clip).stem
    human = source["human_positions_robot_scale_m"]
    names = list(source["human_joint_names"])
    ankles, toes = [[names.index(name) for name in nodes] for nodes in
                    (("lfoot", "rfoot"), ("ltoes", "rtoes"))]
    center, half = bounds(source)
    factors = [float(case["speed_factor"]) for case in cases]
    if factors != [1., 2., 3.]:
        raise ValueError("This comparison layout expects the predeclared 1x/2x/3x sweep")
    final_time = max(case["time_s"][-1] for case in cases)
    timeline = np.r_[np.arange(0, final_time, 1 / fps), final_time]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=90)
    fig.subplots_adjust(left=.06, right=.98, bottom=.20, top=.85, hspace=.4, wspace=.25)
    clock = fig.text(.5, .935, "", ha="center", fontsize=11)
    fig.suptitle(stem.replace("HDM_bd_", "") + " | same GMR path, different reference clocks", fontsize=13, y=.98)
    fig.text(.5, .01, "Top: human FK keypoints at robot scale. Bottom: TRON1 mesh with imposed qpos, not dynamics.\n"
             "Green=Left; purple=Right; red=CoM. Gamma masked below 0.8 m/s; fast analysis gate is 2 m/s.\n"
             "Faster references freeze after ending; no looping or extrapolated motion.",
             fontsize=8, ha="center", va="bottom")
    data = mujoco.MjData(model)
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = 2 * half
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.orthographic = True
    camera.lookat[:] = [center[0], center[1], 0]
    camera.distance, camera.azimuth, camera.elevation = 5., 90., -90.
    renderer = mujoco.Renderer(model, height=320, width=320)
    artists = []
    labels = []
    for col, factor in enumerate(factors):
        for row in range(2):
            axis = axes[row, col]
            axis.set(xlim=(center[0] - half, center[0] + half), ylim=(center[1] - half, center[1] + half), xlabel="X (m)", ylabel="Y (m)")
            axis.set_aspect("equal")
            axis.grid(alpha=.15)
        lines = [(child, int(parent), axes[0, col].plot([], [], color="#979ca1", lw=1)[0])
                 for child, parent in enumerate(source["human_parent_indices"]) if parent >= 0]
        foot_lines = [axes[0, col].plot([], [], color=color, lw=3)[0] for color in COLORS]
        keypoints = axes[0, col].scatter(np.zeros(3), np.zeros(3), c=["orange", *COLORS], s=23, zorder=4)
        mesh = axes[1, col].imshow(np.ones((320, 320, 3)), extent=(center[0] - half, center[0] + half, center[1] - half, center[1] + half))
        wheel_lines = [axes[1, col].plot([], [], color=color, lw=2)[0] for color in COLORS]
        wheel_points = [axes[1, col].scatter([], [], c=color, s=15, zorder=4) for color in COLORS]
        com_point = axes[1, col].scatter([], [], c="#d33d3d", s=22, zorder=5)
        path = axes[1, col].plot([], [], color="#d33d3d", lw=1, alpha=.7)[0]
        labels.append(fig.text(.19 + .32 * col, .08, "", ha="center", fontsize=8))
        artists.append({"lines": lines, "foot_lines": foot_lines, "keypoints": keypoints,
                        "mesh": mesh, "wheel_lines": wheel_lines, "wheel_points": wheel_points,
                        "com_point": com_point, "path": path, "last_rendered": None})

    sampled = np.empty((len(timeline), len(cases)), dtype=int)
    active_flags = np.empty_like(sampled, dtype=bool)
    for frame, wall in enumerate(timeline):
        for col, case in enumerate(cases):
            sampled[frame, col], active_flags[frame, col] = sample_at_wall_time(case["time_s"], wall)

    def draw(frame):
        wall = timeline[frame]
        clock.set_text(f"Shared reference wall clock: {wall:.2f} s | imposed reference states only")
        for col, (case, factor, artist) in enumerate(zip(cases, factors, artists)):
            index, active = sampled[frame, col], bool(active_flags[frame, col])
            for child, parent, line in artist["lines"]:
                line.set_data(human[index, [parent, child], :2].T)
            artist["keypoints"].set_offsets(human[index, [0, *ankles], :2])
            for side in range(2):
                artist["foot_lines"][side].set_data(human[index, [ankles[side], toes[side]], :2].T)
                position = source["robot_site_positions_m"][index, 5 + side, :2]
                artist["wheel_points"][side].set_offsets(position)
                angle = np.radians(source["robot_wheel_heading_deg"][index, side])
                direction = .127 * np.array([np.cos(angle), np.sin(angle)])
                artist["wheel_lines"][side].set_data((position + np.array([-1, 1])[:, None] * direction).T)
            artist["com_point"].set_offsets(case["com_position_m"][index, :2])
            artist["path"].set_data(case["com_position_m"][:index + 1, :2].T)
            if artist["last_rendered"] != index:
                data.qpos[:] = source["qpos"][index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                if not renderer.scene.camera[0].orthographic or not np.isclose(renderer.scene.camera[0].frustum_top, half, atol=1e-6):
                    raise ValueError("Mesh XY bounds do not match keypoint panels")
                artist["mesh"].set_data(renderer.render())
                artist["last_rendered"] = index
            axes[0, col].set_title(f"{factor:g}x | {'ACTIVE' if active else 'ENDED / held pose'}\n"
                                   f"native source t={source['time_s'][index]:.2f} s", fontsize=10,
                                   color="black" if active else "gray")
            axes[1, col].set_title(f"{factor:g}x TRON1 | reference duration {case['time_s'][-1]:.2f} s", fontsize=10)
            if active:
                gamma_text = (f"{case['com_course_wheel_axial_deg'][index, 0]:+.1f}/{case['com_course_wheel_axial_deg'][index, 1]:+.1f} deg"
                              if case['com_xy_speed_m_s'][index] >= .8 else "masked below 0.8 m/s")
                torque_available = case["directional_motoring_torque_available_nm"][min(index, len(case["directional_motoring_torque_available_nm"]) - 1)].min()
                labels[col].set_text(f"CoM speed {case['com_xy_speed_m_s'][index]:.2f} m/s | translation E {case['com_xy_translation_energy_J'][index]:.1f} J\n"
                                     f"CoM-to-wheel gamma L/R: {gamma_text}\n"
                                     f"Geometry near-ground L/R: {source['robot_near_ground_proxy'][index].astype(int).tolist()}\n"
                                     f"Min leg motoring torque available: {torque_available:.1f} Nm (not demand)")
            else:
                labels[col].set_text("REFERENCE ENDED\nFinal pose held only for comparison\nNo subsequent motion or velocity is claimed")
        return []

    draw(min(len(timeline) - 1, int(np.searchsorted(timeline, final_time / 6))))
    fig.savefig(output / (stem + ".png"), dpi=115)
    try:
        animation = FuncAnimation(fig, draw, frames=range(len(timeline)), interval=1000 / fps, blit=False)
        animation.save(output / (stem + ".gif"), writer=PillowWriter(fps=fps))
    finally:
        renderer.close()
        plt.close(fig)
    np.savez_compressed(output / (stem + "_timeline.npz"), reference_wall_time_s=timeline,
                        native_frame_indices=sampled, active_reference_mask=active_flags,
                        speed_factors=np.asarray(factors))
    with Image.open(output / (stem + ".gif")) as gif:
        duration_ms = 0
        gif_frames = gif.n_frames
        for frame in range(gif.n_frames):
            gif.seek(frame)
            duration_ms += gif.info.get("duration", 0)
    return {"clip": clip, "rates": factors, "gif_sha256": sha(output / (stem + ".gif")),
            "gif_frames": gif_frames, "gif_presentation_duration_s": duration_ms / 1000,
            "longest_reference_duration_s": float(final_time), "presentation_fps": fps,
            "timeline_sha256": sha(output / (stem + "_timeline.npz")),
            "dynamics_stepped": False, "finish_behavior": "hold final pose and mark ended, no loop",
            "clock_note": "GIF uses fixed presentation FPS; sample indices follow each retimed physical reference clock. End-frame display adds <=2 presentation frames; scientific metrics use native arrays."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment", type=Path, default=ROOT / "results/hdm05_gmr_bd")
    parser.add_argument("--rate-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip", nargs="+")
    parser.add_argument("--fps", type=int, choices=(5, 10, 20, 25), default=10)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Preserving existing render directory")
    overview = json.loads((args.rate_dir / "summary.json").read_text())
    if sha(args.base_experiment / "experiment.json") != overview["base_experiment_sha256"] or sha(args.model) != overview["model_sha256"]:
        raise ValueError("Base reference or model provenance mismatch")
    clips = args.clip or overview["settings"]["render_clips"]
    if set(clips) - {row["clip"] for row in overview["cases"]}:
        parser.error("Unknown clip")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in clips:
        rows = [row for row in overview["cases"] if row["clip"] == clip]
        source_path = args.base_experiment / Path(clip).stem / "retarget_trace.npz"
        if any(sha(source_path) != row["base_trace_sha256"] or sha(args.rate_dir / row["delta_file"]) != row["delta_sha256"] for row in rows):
            raise ValueError("Base or retiming trace mismatch")
        record = render_one(clip, read_npz(source_path), [read_npz(args.rate_dir / row["delta_file"]) for row in rows],
                            model, output, args.fps)
        records.append(record)
        print("Rendered shared-clock comparison " + clip, flush=True)
    (output / "render_audit.json").write_text(json.dumps({"renders": records,
        "renderer_source_sha256": sha(Path(__file__)), "summary_sha256": sha(args.rate_dir / "summary.json")}, indent=2) + "\n")


if __name__ == "__main__":
    main()

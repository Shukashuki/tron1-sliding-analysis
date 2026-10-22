"""Replay the verified nominal four cells with a constant-offset follow camera.

Only graphics change. The previous immutable nominal render and physical
recordings remain untouched; fixed-world maps continue to show XY drift.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import render_ppo_backend_400hz_nominal as original
from ppo_backend_400hz_contract import require, sha256, write_json
from render_physx_ppo_fullbody import _pose, build_qpos, create_runtime, mesh_provenance
from render_ppo_fixed_follow import apply_camera, prepare_camera

ORIGINAL_SHA256 = "e0dc27b3bef1c7ab84ea3fa9b0839aa314b45cb4081e8839d2f75d7a228b7411"
ORDER, FRAMES, FPS = original.ORDER, original.FRAMES, original.FPS


def render_frame(runtime, traces, qposes, reference, bounds, plan, frame):
    from PIL import Image, ImageDraw
    require(isinstance(frame, int) and 0 <= frame < FRAMES, "Invalid original replay frame")
    index = int(original.native_indices()[frame])
    canvas = Image.new("RGB", original.CANVAS_SIZE, "#fafafa")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 5), f"400 Hz PPO | μ=0.7 | t={index*.0025:.2f} s | FIXED follow camera: distance {plan['distance_m']:.3f} m, same angles/scale in all panels; no zoom", font=runtime.font, fill="#111111")
    for column, key in enumerate(ORDER):
        trace = traces[key]
        _pose(runtime.model, runtime.data, qposes[key][index])
        camera = apply_camera(runtime, plan, key, frame)
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        x = column*640
        canvas.paste(Image.fromarray(runtime.renderer.render().copy()), (x, 58))
        draw.rectangle((x, 30, x+640, 57), fill="#f0f2f4")
        draw.text((x+9, 34), original.TITLES[key], font=runtime.font, fill="#111111")
        value = original.diagnostics(trace, reference, index)
        xy = f"{value['xy_error_m']:.3f} m" if value["xy_error_m"] is not None else "n/a (ref ends 6.96 s)"
        draw.text((x+9, 424), f"XY {xy} | COM speed {value['speed_m_s']:.2f} m/s | tilt {value['tilt_deg']:.1f} deg", font=runtime.font, fill="#111111")
        draw.text((x+9, 447), f"Original height {value['height_m']:.3f} m | native 400 Hz / video 50 Hz", font=runtime.font, fill="#111111")
        draw.text((x+9, 470), f"L/R ground count {int(value['count'][0])}/{int(value['count'][1])} | Fz {value['fz'][0]:.0f}/{value['fz'][1]:.0f} N", font=runtime.font, fill="#111111")
        original._map_panel(draw, trace, reference, index, (x+5, 500, x+635, 672), bounds, runtime.small)
    draw.text((12, 683), "Camera follows each original root by translation only; fixed world orientation, distance and field of view. World-XY maps preserve path offsets; no pose correction.", font=runtime.small, fill="#333333")
    draw.text((12, 702), "Four nominal cells only, N=1. Full friction grid remains partial 46/68. Native contact telemetry is diagnostic, not a flight/failure label. No new simulation or training.", font=runtime.small, fill="#333333")
    return canvas


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts; choose a fresh directory")
    require(sha256(original.__file__) == ORIGINAL_SHA256, "Frozen nominal data/graphics helper differs")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    require(os.environ["MUJOCO_GL"] == "osmesa", "Offline replay uses CPU OSMesa")
    traces, reference, summary = original.load_data(args.input_root, args.reference, args.model,
        args.legacy_mujoco_evaluator, args.cohort_protocol)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes, arrays, audits = {}, {}, {}
        for key in ORDER:
            trace = traces[key]
            qposes[key], mapping = build_qpos(runtime.model, trace)
            name = key[0]+"_to_"+key[1]
            audits[name] = {**original.replay_fk_audit(runtime.model, trace, qposes[key]),
                "named_joint_qpos_addresses": mapping.tolist()}
            body, wheel = original.heading_geometry_masks(trace)
            trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = body, wheel
            arrays.update({name+"_qpos": qposes[key], name+"_native_time_s": trace["time_s"],
                name+"_selected_native_indices": original.native_indices(),
                name+"_recorded_ground_count": trace["contact_count"],
                name+"_recorded_ground_force_w_n": trace["contact_force"]})
        plan = prepare_camera(runtime, traces, qposes, ORDER, original.native_indices())
        bounds = [v.tolist() for v in original.world_bounds(traces, reference)]
        for key in ORDER:
            arrays[key[0]+"_to_"+key[1]+"_camera_lookat_m"] = plan["lookat_by_case"][key]
        arrays["fixed_camera_distance_m"] = plan["distance_m"]
        summary.update(kind="recorded_nominal_400hz_fixed_relative_follow_replay",
            renderer_source_sha256=sha256(__file__), frozen_nominal_helper_sha256=ORIGINAL_SHA256,
            camera_mode="fixed_distance_world_orientation_root_relative_translation_follow",
            camera_distance_m=plan["distance_m"], camera_distance_range_m=[plan["distance_m"]]*2,
            camera_same_distance_across_panels=True, camera_same_world_orientation_across_panels=True,
            camera_same_target_across_panels=False, camera_individual_root_translation_follow=True,
            camera_follows_body_yaw=False, camera_zoom=False, camera_fit_audit=plan["fit_audit"],
            camera_settings={k:plan[k] for k in ("azimuth_deg", "elevation_deg", "vertical_fov_deg")},
            fk_validation_by_panel=audits, world_bounds=bounds, mesh_provenance=mesh_provenance(args.model),
            rendering_limitations=["Camera-only revision; original native states, reference and metrics are unchanged.",
                "One camera distance is selected once across all frames/cases; target is each original root plus fixed offset, with fixed world orientation and FOV.",
                "Independent translation following is requested for visual readability. Compare XY drift in the fixed-world maps, not robot position within a following viewport.",
                "Only four nominal mu=0.7 cells; full 17-mu matrix is partial, stopped at 46/68.",
                "Native CPU MuJoCo deployment is not GPU MJWarp validation. N=1 per cell; 50 Hz video does not resolve continuous 400 Hz contact.",
                "No filtering, interpolation, height correction, dynamics stepping, policy inference or additional training."])
        output.mkdir(parents=True)
        np.savez_compressed(output/"replay_states.npz", **arrays)
        render_frame(runtime, traces, qposes, reference, bounds, plan, 150).save(output/"comparison.png")
        files = ["replay_states.npz", "comparison.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output/"fullbody_comparison.mp4"), original.CANVAS_SIZE,
                fps=FPS, codec="libx264", quality=7, macro_block_size=2, output_params=["-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(FRAMES):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, bounds, plan, frame)))
            finally:
                stream.close()
            files.append("fullbody_comparison.mp4")
        summary["video_generated"] = not args.skip_video
        summary["render_helpers_sha256"] = {name:sha256(Path(__file__).with_name(name)) for name in
            ("render_ppo_fixed_follow.py", "render_ppo_backend_400hz_nominal.py", "render_physx_ppo_fullbody.py",
             "summarize_ppo_backend_400hz_nominal.py", "controller_trace.py")}
        summary["artifacts"] = {name:sha256(output/name) for name in files}
        write_json(output/"summary.json", summary)
        return summary
    finally:
        runtime.renderer.close()


if __name__ == "__main__":
    run(original.parser().parse_args())

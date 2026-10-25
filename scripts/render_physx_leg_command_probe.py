"""Two genuine command-diagnostic arms, opaque native poses, fixed camera."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

import analyze_physx_leg_command_probe as audit
import render_ppo_ground_contact_v2 as graphics

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    assert not args.output_dir.exists() and os.environ["MUJOCO_GL"] == "osmesa"
    summary = audit.read(args.analysis)
    assert summary["status"] == "completed_analysis" and summary["analyzer_sha256"] == audit.sha(audit.__file__)
    reference = graphics.metrics.reference_data(args.reference)
    assert audit.sha(args.reference) == summary["reference_sha256"]
    rows, traces, _ = audit.load_cases(args.collection, reference)
    assert rows == summary["rows"]
    order = list(audit.MODES)
    kinematics, source = graphics.metrics.load_fk(args.model, args.legacy_evaluator)
    cases = {}
    for row in rows:
        trace = traces[row["id"]]
        fk = kinematics.reconstruct(trace)
        trace.update(signals=fk["signals"], wheel_quaternion_wxyz=fk["wheel_quaternion_wxyz"], wheel_center_velocity_w=fk["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = graphics.metrics.heading_geometry_masks(trace)
        cases[row["id"]] = dict(variant="command_probe", engine="PhysX", analysis=row["metrics"],
            title="Original leg target hold" if row["mode"] == "passthrough" else "Leg target ramp over8 physical steps",
            new_nominal_screen=row["nominal_screen"])
    runtime = graphics.create_runtime(args.model, width=640, height=360)
    try:
        qposes, audits = {}, {}
        for key in order:
            qposes[key], mapping = graphics.build_qpos(runtime.model, traces[key])
            audits[key] = dict(graphics.replay_fk_audit(runtime.model, traces[key], qposes[key]), named_joint_qpos_addresses=mapping.tolist())
        camera = graphics.prepare_camera(runtime, traces, qposes, order, graphics.native_indices())
        bounds = [v.tolist() for v in graphics.world_bounds(traces, reference)]
        args.output_dir.mkdir(parents=True)
        import imageio_ffmpeg
        from PIL import ImageDraw
        writer = imageio_ffmpeg.write_frames(str(args.output_dir/"fullbody_comparison.mp4"), (1280, 850), fps=50,
            codec="libx264", quality=7, macro_block_size=2, output_params=["-movflags", "+faststart"])
        writer.send(None)
        try:
            for frame in range(350):
                canvas = graphics.render_frame(runtime, traces, qposes, reference, cases, order, camera, bounds, frame)
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 1280, 29), fill="#fafafa")
                draw.text((12, 4), f"Same frozen K PPO | PhysX mu=.7 | leg-command execution diagnostic | beta ON | t={frame*.02:.2f}s", font=runtime.small, fill="#111111")
                draw.rectangle((0, 763, 1280, 849), fill="#fafafa")
                draw.text((12, 765), "Two actual completed arms | command execution changed at inference; no PPO training or parameter search", font=runtime.small, fill="#8f2929")
                draw.text((12, 787), "Native400Hz poses; eight contact samples per50Hz frame. Wheel efforts and raw actor mapping unchanged.", font=runtime.small, fill="#333333")
                draw.text((12, 809), f"Opaque TRON1, fixed relative camera {camera['distance_m']:.3f}m; fixed world angles, translation only; no zoom/yaw follow.", font=runtime.small, fill="#333333")
                draw.text((12, 831), "Neither arm has continuous any-wheel support. N=1; no pose smoothing/correction; task completion not claimed.", font=runtime.small, fill="#333333")
                writer.send(np.asarray(canvas))
                if frame in (112, 150, 200, 348):
                    canvas.save(args.output_dir/(f"frame_{frame:03d}.png"))
                if frame == 150:
                    canvas.save(args.output_dir/"comparison.png")
                if frame % 100 == 0:
                    print(f"Leg-command native replay frame={frame}/350", flush=True)
        finally:
            writer.close()
        arrays = {"native_indices": graphics.native_indices()}
        for key in order:
            arrays[key+"__qpos"], arrays[key+"__lookat"] = qposes[key], camera["lookat_by_case"][key]
        np.savez_compressed(args.output_dir/"replay_states.npz", **arrays)
        names = [p.name for p in args.output_dir.iterdir() if p.is_file()]
        result = dict(kind="two_actual_frozen_actor_command_execution_native_replay", case_order=order,
            analysis_sha256=audit.sha(args.analysis), renderer_sha256=audit.sha(__file__), graphics_sha256=audit.sha(graphics.__file__),
            video_frames=350, video_fps=50, video_size_pixels=[1280, 850], FK_source=source, FK_audits=audits,
            fixed_camera_distance_m=camera["distance_m"], camera_same_distance_all_cases_frames=True, camera_zoom=False, camera_yaw_follow=False,
            world_bounds=bounds, robot_mesh_alpha=1., pose_smoothing_interpolation_alignment_height_correction=False,
            new_physics_policy_inference_or_training=False, actuator_command_interpolation_recorded_as_experimental_intervention=True,
            task_success_claimed=False, artifacts={name:audit.sha(args.output_dir/name) for name in names})
        (args.output_dir/"provenance.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    finally:
        runtime.renderer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("collection", "analysis", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--reference", type=Path, default=graphics.assets.paired.DEFAULT_REFERENCE)
    p.add_argument("--model", type=Path, default=graphics.assets.paired.DEFAULT_MODEL)
    p.add_argument("--legacy-evaluator", type=Path, default=graphics.assets.paired.DEFAULT_LEGACY)
    run(p.parse_args())

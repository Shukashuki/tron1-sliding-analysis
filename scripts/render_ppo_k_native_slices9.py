"""Four horizontal opaque recorded-pose panels with a fixed relative camera."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

import analyze_ppo_k_native_slices9 as analysis
import render_ppo_ground_contact_v2 as graphics

ROOT = Path(__file__).resolve().parents[1]
GRAPHICS_SHA = "0289c7da1764758cb79c184a7fbf9f5f0e9afd4adab23ef1ea700411be997efa"


def run(args):
    analysis.require(os.environ["MUJOCO_GL"] == "osmesa" and not args.output_dir.exists(),
        "CPU OSMesa and a fresh render directory are required")
    analysis.require(analysis.sha(graphics.__file__) == GRAPHICS_SHA, "Frozen replay/camera helper differs")
    summary = analysis.read(args.analysis)
    analysis.require(summary["kind"] == "PPO_training_geometry_by_validation_geometry_PhysX_nominal"
        and summary["status"] == "completed_analysis" and summary["analyzer_sha256"] == analysis.sha(analysis.__file__),
        "Completed model comparison analysis is required")
    reference = graphics.metrics.reference_data(args.reference)
    reference["source_path"] = str(args.reference.resolve())
    analysis.require(analysis.sha(args.reference) == summary["reference_sha256"], "Reference differs")
    rows, traces, _, _, proof = analysis.load_cases(args.collection, args.preflight, args.previous_result, reference)
    analysis.require(rows == summary["rows"], "Native recomputation differs from published analysis")
    kinematics, fk_source = graphics.metrics.load_fk(args.model, args.legacy_evaluator)
    cases = {}
    for row in rows:
        trace = traces[row["id"]]
        fk = kinematics.reconstruct(trace)
        trace.update(signals=fk["signals"], wheel_quaternion_wxyz=fk["wheel_quaternion_wxyz"],
            wheel_center_velocity_w=fk["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = graphics.metrics.heading_geometry_masks(trace)
        short = lambda name: "cylinder" if name == "authored" else "9 slices"
        cases[row["id"]] = dict(variant="model_comparison", engine="PhysX", analysis=row["metrics"],
            title=short(row["training_geometry"])+" trained -> "+short(row["validation_geometry"]),
            new_nominal_screen=row["nominal_screen"])
    runtime = graphics.create_runtime(args.model, width=640, height=360)
    try:
        qposes, fk_audits = {}, {}
        for key in analysis.ORDER:
            qposes[key], mapping = graphics.build_qpos(runtime.model, traces[key])
            fk_audits[key] = dict(graphics.replay_fk_audit(runtime.model, traces[key], qposes[key]),
                named_joint_qpos_addresses=mapping.tolist())
        plan = graphics.prepare_camera(runtime, traces, qposes, list(analysis.ORDER), graphics.native_indices())
        bounds = [value.tolist() for value in graphics.world_bounds(traces, reference)]
        args.output_dir.mkdir(parents=True)
        from PIL import ImageDraw
        import imageio_ffmpeg
        stream = imageio_ffmpeg.write_frames(str(args.output_dir/"fullbody_comparison.mp4"), (2560, 850), fps=50,
            codec="libx264", quality=7, macro_block_size=2, output_params=["-movflags", "+faststart"])
        stream.send(None)
        try:
            for frame in range(350):
                canvas = graphics.render_frame(runtime, traces, qposes, reference, cases,
                    list(analysis.ORDER), plan, bounds, frame)
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 2560, 29), fill="#fafafa")
                draw.text((12, 4), f"PPO training geometry x validation geometry | PhysX mu=.7 | same K reward/DR/budget | beta ON | t={frame*.02:.2f}s", font=runtime.small, fill="#111111")
                draw.rectangle((0, 763, 2560, 786), fill="#fafafa")
                caption = " | ".join(cases[key]["title"]+": "+str(len(cases[key]["new_nominal_screen"]["failed_checks"]))+" failed checks" for key in analysis.ORDER)
                draw.text((12, 765), caption+" | task completion not claimed", font=runtime.small, fill="#8f2929")
                stream.send(np.asarray(canvas))
                if frame in (112, 150, 200, 348):
                    canvas.save(args.output_dir/("frame_"+str(frame).zfill(3)+".png"))
                if frame == 150:
                    canvas.save(args.output_dir/"comparison.png")
                if frame % 100 == 0:
                    print(f"Geometry-trained PPO native-pose replay frame={frame}/350", flush=True)
        finally:
            stream.close()
        arrays = {"native_indices": graphics.native_indices(), "fixed_camera_distance_m": np.asarray(plan["distance_m"])}
        for key in analysis.ORDER:
            arrays[key+"__qpos"] = qposes[key]
            arrays[key+"__lookat"] = plan["lookat_by_case"][key]
            arrays[key+"__floor_normal_force_w"] = traces[key]["contact_force"]
        np.savez_compressed(args.output_dir/"replay_states.npz", **arrays)
        names = ("fullbody_comparison.mp4", "comparison.png", "replay_states.npz", "frame_112.png", "frame_150.png", "frame_200.png", "frame_348.png")
        provenance = dict(kind="four_panel_training_geometry_by_validation_geometry_recorded_replay",
            case_order=list(analysis.ORDER), cases=cases, analysis_sha256=analysis.sha(args.analysis),
            analyzer_sha256=analysis.sha(analysis.__file__), renderer_sha256=analysis.sha(__file__), graphics_sha256=GRAPHICS_SHA,
            new_actor_sha256=proof["actor_sha256"], original_actor_sha256=analysis.previous.ACTOR,
            video_frames=350, video_fps=50, video_size_pixels=[2560, 850], native_states_per_case=2793,
            FK_source=fk_source, FK_readback_audits=fk_audits,
            source_model_sha256=analysis.sha(args.model), reference_sha256=analysis.sha(args.reference),
            world_bounds=bounds, fixed_camera_distance_m=plan["distance_m"],
            camera_same_distance_all_cases_frames=True, camera_yaw_follow=False, camera_zoom=False,
            smoothing_interpolation_alignment_height_correction=False, original_visual_meshes_retained=True,
            collision_slices_not_drawn=True, robot_mesh_alpha=1., new_physics_policy_inference_training=False,
            single_wheel_support_allowed=True, contact_threshold_upward_n_strict=5.,
            visual_review_complete=False, task_success_claimed=False,
            artifacts={name: analysis.sha(args.output_dir/name) for name in names})
        (args.output_dir/"provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
        return provenance
    finally:
        runtime.renderer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("collection", "analysis", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--preflight", type=Path, default=ROOT/"results/ppo_k_native_slices9_20261025_v1/evidence")
    p.add_argument("--previous-result", type=Path, default=ROOT/"results/physx_tron1_wheel_models_20261025_v1")
    p.add_argument("--reference", type=Path, default=graphics.assets.paired.DEFAULT_REFERENCE)
    p.add_argument("--model", type=Path, default=graphics.assets.paired.DEFAULT_MODEL)
    p.add_argument("--legacy-evaluator", type=Path, default=graphics.assets.paired.DEFAULT_LEGACY)
    run(p.parse_args())

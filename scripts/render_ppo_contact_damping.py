"""Replay two recorded nominal PhysX episodes of the same actor at Kd10/Kd5.

MuJoCo is used only for mesh kinematics and graphics, never for physics replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from controller_trace import native_trace
from nmpc_tracking_objective import quaternion_rotate
from render_physx_ppo_fullbody import DEFAULT_MODEL, build_qpos, create_runtime, mesh_provenance, validate_fk, _pose
from render_ppo_smoothness_comparison import MODEL_SHA256, load_reference, common_camera_target
from summarize_ppo_contact_damping import DEFAULT_INPUT, DEFAULT_REFERENCE, DEFAULT_WRAPPER_SHA256, ORDER, ROOT, load_comparison, require, sha256


def render_frame(runtime, traces, qposes, reference, summary, index):
    from PIL import Image, ImageDraw
    w, h = runtime.width, runtime.height
    image = Image.new("RGB", (2*w, h+145), "#fafafa"); draw = ImageDraw.Draw(image)
    time = float(traces[ORDER[0]]["time_s"][index])
    draw.text((12, 6), f"Same frozen PPO (-0.5 action-rate) | PhysX mu=0.7 | original t={time:.2f}s", font=runtime.font, fill="black")
    camera = runtime.cameras[0]; camera.lookat[:] = common_camera_target(reference, time)
    for column, label in enumerate(ORDER):
        trace = traces[label]; x, y = column*w, 35
        _pose(runtime.model, runtime.data, qposes[label][index])
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        image.paste(Image.fromarray(runtime.renderer.render().copy()), (x, y))
        draw.rectangle((x, y, x+w, y+27), fill="#f5f5f5")
        kd = summary["cases"][label]["deployed_leg_kd"]
        draw.text((x+10, y+3), f"Kp500 / deployed Kd{kd:g} | trained Kd10 | no retraining", font=runtime.font, fill="black")
        position = trace["position"][index]
        target = common_camera_target(reference, time)+np.array([0., 0., .24])
        error = float(np.linalg.norm(position[:2]-target[:2])); speed = float(np.linalg.norm(trace["com_velocity"][index, :2]))
        up = quaternion_rotate(trace["quaternion"][index], np.array([0., 0., 1.]))
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1., 1.))))
        counts, fz = trace["contact_count"][index], trace["contact_force"][index, :, 2]
        draw.text((x+10, y+h+5), f"XY error {error:.3f}m | COM speed {speed:.2f}m/s | tilt {tilt:.2f}deg", font=runtime.font, fill="#111111")
        draw.text((x+10, y+h+31), f"Original height {position[2]:.3f}m | L/R count {int(counts[0])}/{int(counts[1])}", font=runtime.font, fill="#111111")
        draw.text((x+10, y+h+57), f"Recorded upward Fz L/R {fz[0]:.0f}/{fz[1]:.0f}N (zero count is not a flight label)", font=runtime.small, fill="#333333")
    draw.text((12, image.height-22), "Original PhysX poses; same reference-follow camera; opaque meshes; MuJoCo graphics only; no filtering or pose correction.", font=runtime.small, fill="#333333")
    return image


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh output directory")
    raw, summary = load_comparison(args.input_root, args.reference, args.expected_wrapper_sha256)
    require(sha256(args.model) == MODEL_SHA256, "Original display model SHA differs")
    reference = load_reference(args.reference)
    traces = {label: native_trace(current, "policy_isaac") for label, current in raw.items()}
    require(np.array_equal(traces["kd10"]["time_s"], traces["kd5"]["time_s"]), "Original clocks differ; no alignment allowed")
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes = {}
        for label in ORDER:
            qposes[label], mapping = build_qpos(runtime.model, traces[label])
            summary["cases"][label]["fk_validation"] = validate_fk(runtime.model, traces[label], qposes[label])
            summary["cases"][label]["named_joint_qpos_addresses"] = mapping.tolist()
        summary.update({"kind": "fixed_actor_zero_shot_contact_damping_fullbody_replay", "graphics_engine": "MuJoCo FK only",
            "model_sha256": MODEL_SHA256, "mesh_provenance": mesh_provenance(args.model),
            "video_fps": 50, "video_frames": 350, "native_sample_stride": 4, "first_native_time_s": 0., "last_native_time_s": 6.98,
            "encoded_duration_s": 7., "dynamics_stepping": False, "contact_solver_called": False,
            "root_height_corrected": False, "pose_alignment": False, "time_alignment": False,
            "camera": {"azimuth_deg": 135., "elevation_deg": -24., "distance_m": 2.15,
                "lookat": "Same locked reference root position + [0,0,-.24] for both panels; hold final reference after6.96s",
                "follows_actual_robot": False, "follows_actual_orientation": False}, "robot_mesh_alpha": 1.})
        output.mkdir(parents=True)
        arrays = {"time_s": traces["kd10"]["time_s"]}
        for label in ORDER:
            arrays[label+"_qpos"] = qposes[label]
            arrays[label+"_original_root_position_m"] = traces[label]["position"]
        np.savez_compressed(output/"replay_states.npz", **arrays)
        render_frame(runtime, traces, qposes, reference, summary, 480).save(output/"comparison.png")
        artifacts = ["replay_states.npz", "comparison.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            first = render_frame(runtime, traces, qposes, reference, summary, 0)
            stream = imageio_ffmpeg.write_frames(str(output/"fullbodycomparison.mp4"), first.size, fps=50,
                codec="libx264", quality=7, macro_block_size=2, output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for index in range(0, 1397, 4):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, summary, index)))
            finally:
                stream.close()
            artifacts.append("fullbodycomparison.mp4")
        summary["video_generated"] = not args.skip_video
        summary["analysis_source_sha256"] = sha256(__file__)
        summary["helper_source_sha256"] = {name: sha256(ROOT/"scripts"/name) for name in
            ("summarize_ppo_contact_damping.py", "analyze_ppo_contact_stability.py", "analyze_ppo_smoothness.py", "summarize_ppo_smoothness.py", "render_physx_ppo_fullbody.py", "render_ppo_smoothness_comparison.py", "eval_ppo_contact_damping.py")}
        summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in artifacts}
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--expected-wrapper-sha256", default=DEFAULT_WRAPPER_SHA256)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    summary = run(parser().parse_args())
    print(json.dumps({"status": "rendered_recorded_damping_comparison", "dynamics_stepping": False, "case_order": summary["case_order"]}))

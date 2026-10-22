"""Recorded PhysX PPO replay of an independent leg-torque-rate candidate.

Both policies keep action-rate -0.5 and the original contact reward +0.4. Only
the candidate was trained with the added leg torque-rate penalty -0.05. Opaque
mesh replay uses original states and FK/graphics only, not new physics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ppo_smoothness import sha256
from controller_trace import native_trace
from render_physx_ppo_fullbody import DEFAULT_MODEL, build_qpos, create_runtime, mesh_provenance, validate_fk, _pose
from render_ppo_smoothness_comparison import MODEL_SHA256, load_reference, common_camera_target
from render_ppo_ground_support_comparison import frame_diagnostics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT/"outputs/ppo_smoothness_20261021_v2/action_rate_0p5/validation/mu0p700/isaac"
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
BASELINE_CHECKPOINT_SHA256 = "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a"
ORDER = ("baseline", "candidate")
CANVAS_SIZE = (1280, 506)
TITLES = {"baseline": "Baseline: no added leg torque-rate penalty",
    "candidate": "Candidate: leg torque-rate -0.05 (original contact +0.4)"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(candidate_dir, candidate_training_dir, reference=DEFAULT_REFERENCE, baseline_dir=DEFAULT_BASELINE,
              *, candidate_status=None, model=DEFAULT_MODEL):
    from analyze_ppo_torque_control_v2 import load_comparison
    raw, summary = load_comparison(candidate_dir, candidate_training_dir, reference,
        baseline_dir=baseline_dir, candidate_status=candidate_status)
    require(set(raw) == set(ORDER) and summary["case_order"] == list(ORDER), "Require audited baseline/new-policy pair")
    checkpoints = {label: summary["cases"][label]["checkpoint_sha256"] for label in ORDER}
    require(checkpoints["baseline"] == BASELINE_CHECKPOINT_SHA256 and checkpoints["candidate"] != checkpoints["baseline"],
        "Independent baseline/candidate checkpoint identities differ from the declared cohort")
    require(all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        for value in checkpoints.values()), "Invalid audited checkpoint SHA")
    require(sha256(model) == MODEL_SHA256, "Original opaque display-model SHA differs")
    traces = {label: native_trace(current, "policy_isaac") for label, current in raw.items()}
    for label in ORDER:
        require(np.array_equal(traces[label]["time_s"], np.arange(1397)*.005)
            and raw[label]["valid_mask"].shape == (1397, 1) and raw[label]["valid_mask"].dtype.kind == "b"
            and raw[label]["valid_mask"].all(), "Require original full200Hz first episode; no reset splice/alignment")
    reference_arrays = load_reference(reference)
    summary["audited_analysis_kind"] = summary["kind"]
    summary.update({"kind": "independently_trained_ppo_leg_torque_control_v2_fullbody_replay",
        "checkpoint_sha256_by_panel": checkpoints, "same_actor_comparison": False,
        "graphics_engine": "MuJoCo FK and graphics only", "model_sha256": MODEL_SHA256,
        "native_physical_hz": 200, "video_fps": 50, "video_frames": 350, "native_sample_stride": 4,
        "first_native_time_s": 0., "last_native_time_s": 6.98, "encoded_duration_s": 7.,
        "video_width": CANVAS_SIZE[0], "video_height": CANVAS_SIZE[1],
        "camera": {"azimuth_deg": 135., "elevation_deg": -24., "distance_m": 2.15,
            "lookat": "Same locked reference root position+[0,0,-.24] for both panels; hold last knot after6.96s",
            "follows_actual_robot": False, "follows_actual_orientation": False},
        "panel_titles": TITLES, "renderer_reward_labels": {"both": "action-rate -0.5, original contact reward +0.4",
            "candidate_addition": "leg applied-torque-rate penalty -0.05; not a contact reward replacement"},
        "robot_mesh_alpha": 1., "root_height_corrected": False, "pose_alignment": False, "time_alignment": False,
        "state_filtering": False, "dynamics_stepping": False, "contact_solver_called": False, "new_policy_inference": False,
        "continuous_contact_claimed_by_renderer": False,
        "rendering_limitations": ["Video50Hz subsamples native200Hz states; it cannot prove uninterrupted contact between frames.",
            "Count is measured ground points above threshold. Zero count is not proof of flight; eligibility=count>0 AND upward Fz>5N.",
            "Both independent policies retain original world poses and are not individually recentered if outside the common view.",
            "Original contact +0.4 is retained; this is not the earlier failed ground-support-replacement policy.",
            "Audited training/runtime/checkpoint provenance remains attached; replay alone does not certify improvement or causality."]})
    return traces, reference_arrays, summary


def render_frame(runtime, traces, qposes, reference, summary, index):
    from PIL import Image, ImageDraw
    w, h = runtime.width, runtime.height
    require((2*w, h+146) == CANVAS_SIZE, "Exact even video geometry required; no auto-resize")
    image = Image.new("RGB", CANVAS_SIZE, "#fafafa"); draw = ImageDraw.Draw(image)
    time = float(traces["baseline"]["time_s"][index])
    draw.text((12, 6), f"PhysX PPO | mu=0.7 | action-rate -0.5 | original contact +0.4 | Kp500/Kd10 | t={time:.2f}s", font=runtime.font, fill="black")
    camera = runtime.cameras[0]; camera.lookat[:] = common_camera_target(reference, time)
    for column, label in enumerate(ORDER):
        trace = traces[label]; x, y = column*w, 35
        _pose(runtime.model, runtime.data, qposes[label][index])
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        image.paste(Image.fromarray(runtime.renderer.render().copy()), (x, y))
        draw.rectangle((x, y, x+w, y+27), fill="#f5f5f5")
        draw.text((x+10, y+3), TITLES[label], font=runtime.font, fill="black")
        value = frame_diagnostics(trace, reference, index)
        draw.text((x+10, y+h+4), f"XY error {value['xy_error_m']:.3f}m | COM speed {value['com_speed_m_s']:.2f}m/s | tilt {value['tilt_deg']:.2f}deg", font=runtime.font, fill="#111111")
        draw.text((x+10, y+h+26), f"Original height {value['height_m']:.3f}m | L/R count {int(value['count'][0])}/{int(value['count'][1])}", font=runtime.font, fill="#111111")
        draw.text((x+10, y+h+48), f"Ground Fz L/R {value['force_z_n'][0]:.0f}/{value['force_z_n'][1]:.0f}N | eligible L/R {int(value['eligible'][0])}/{int(value['eligible'][1])}", font=runtime.font, fill="#111111")
        draw.text((x+10, y+h+70), f"Checkpoint {summary['checkpoint_sha256_by_panel'][label][:12]}... | count>0 AND Fz>5N eligibility", font=runtime.small, fill="#333333")
    draw.text((12, image.height-20), "Recorded200Hz / video50Hz; opaque meshes; MuJoCo graphics only. Zero contact count is not a flight label. No pose correction.", font=runtime.small, fill="#333333")
    return image


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh output directory")
    traces, reference, summary = load_data(args.candidate_dir, args.candidate_training_dir, args.reference, args.baseline_dir,
        candidate_status=args.candidate_status, model=args.model)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes = {}
        for label in ORDER:
            qposes[label], mapping = build_qpos(runtime.model, traces[label])
            summary["cases"][label]["fk_validation"] = validate_fk(runtime.model, traces[label], qposes[label])
            summary["cases"][label]["named_joint_qpos_addresses"] = mapping.tolist()
        summary["mesh_provenance"] = mesh_provenance(args.model); output.mkdir(parents=True)
        arrays = {"time_s": traces["baseline"]["time_s"]}
        for label in ORDER:
            arrays.update({label+"_qpos": qposes[label], label+"_original_root_position_m": traces[label]["position"],
                label+"_recorded_contact_count": traces[label]["contact_count"], label+"_recorded_ground_force_w_n": traces[label]["contact_force"]})
        np.savez_compressed(output/"replay_states.npz", **arrays)
        render_frame(runtime, traces, qposes, reference, summary, 480).save(output/"comparison.png")
        artifacts = ["replay_states.npz", "comparison.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            first = render_frame(runtime, traces, qposes, reference, summary, 0)
            require(first.size == CANVAS_SIZE, "Unexpected video canvas")
            stream = imageio_ffmpeg.write_frames(str(output/"fullbodycomparison.mp4"), first.size, fps=50,
                codec="libx264", quality=7, macro_block_size=2, output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for index in range(0, 1397, 4):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, summary, index)))
            finally:
                stream.close()
            artifacts.append("fullbodycomparison.mp4")
        summary["video_generated"] = not args.skip_video; summary["renderer_source_sha256"] = sha256(__file__)
        summary["rendering_helper_source_sha256"] = {name: sha256(ROOT/"scripts"/name) for name in
            ("analyze_ppo_torque_control_v2.py", "render_ppo_ground_support_comparison.py", "render_physx_ppo_fullbody.py", "render_ppo_smoothness_comparison.py", "controller_trace.py", "nmpc_tracking_objective.py")}
        summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in artifacts}
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-dir", type=Path, required=True)
    result.add_argument("--candidate-training-dir", type=Path, required=True)
    result.add_argument("--candidate-status", type=Path)
    result.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
    print(json.dumps({"status": "rendered_existing_torque_control_v2_comparison", "same_actor": False, "dynamics_stepping": False}))

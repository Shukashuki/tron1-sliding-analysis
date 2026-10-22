"""Opaque FK replay: one fixed PPO actor, alpha1 versus six-leg alpha.5 filter.

The filter was applied to processed position targets during the recorded
zero-shot diagnostic. This renderer does not filter states, execute a policy,
advance dynamics or call a contact solver. Wheels and actor action history
retain their original requested-action semantics in both recorded episodes.
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
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
CHECKPOINT_SHA256 = "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a"
ORDER = ("baseline", "candidate")
CANVAS_SIZE = (1280, 506)
TITLES = {"baseline": "alpha1: exact target bypass | same frozen actor",
          "candidate": "alpha0.5: six-leg target low-pass | zero-shot delay"}
ALPHAS = {"baseline": 1., "candidate": .5}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def core_applied_target_step_rms(summary, label):
    """Audited sidecar metric, not an estimate from raw requested actions."""
    value = summary["cases"][label]["windows"]["core"]["target_control"]["filtered_applied_target_step_rms_rad"]
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0,
            "Require finite audited applied-target step RMS")
    return float(value)


def load_data(input_root, reference=DEFAULT_REFERENCE, *, model=DEFAULT_MODEL):
    from analyze_ppo_leg_target_filter import load_comparison
    raw, summary = load_comparison(input_root, reference)
    require(set(raw) == set(ORDER) and summary["case_order"] == list(ORDER), "Require audited alpha1/alpha.5 pair")
    checkpoints = {label: summary["cases"][label]["checkpoint_sha256"] for label in ORDER}
    require(all(value == CHECKPOINT_SHA256 for value in checkpoints.values()), "Both panels must use the same original8ba actor")
    require(all(summary["cases"][label]["alpha"] == ALPHAS[label] for label in ORDER), "Audited alpha labels/order differ")
    require(sha256(model) == MODEL_SHA256, "Original opaque display-model SHA differs")
    traces = {label: native_trace(current, "policy_isaac") for label, current in raw.items()}
    for label in ORDER:
        require(np.array_equal(traces[label]["time_s"], np.arange(1397)*.005)
                and raw[label]["valid_mask"].shape == (1397, 1)
                and raw[label]["valid_mask"].dtype.kind == "b" and raw[label]["valid_mask"].all(),
                "Require original complete200Hz first episode; no reset splice or alignment")
        core_applied_target_step_rms(summary, label)
    reference_arrays = load_reference(reference)
    summary["audited_analysis_kind"] = summary["kind"]
    summary.update({"kind": "fixed_actor_ppo_leg_target_lowpass_fullbody_replay",
        "checkpoint_sha256_by_panel": checkpoints, "same_actor_comparison": True,
        "graphics_engine": "MuJoCo FK and graphics only", "model_sha256": MODEL_SHA256,
        "native_physical_hz": 200, "video_fps": 50, "video_frames": 350, "native_sample_stride": 4,
        "first_native_time_s": 0., "last_native_time_s": 6.98, "encoded_duration_s": 7.,
        "video_width": CANVAS_SIZE[0], "video_height": CANVAS_SIZE[1],
        "camera": {"azimuth_deg": 135., "elevation_deg": -24., "distance_m": 2.15,
            "lookat": "Same locked reference root position+[0,0,-.24] for both panels; hold last knot after6.96s",
            "follows_actual_robot": False, "follows_actual_orientation": False},
        "panel_titles": TITLES, "robot_mesh_alpha": 1., "root_height_corrected": False,
        "pose_alignment": False, "time_alignment": False, "state_filtering": False,
        "dynamics_stepping": False, "contact_solver_called": False, "new_policy_inference": False,
        "additional_training_updates": 0, "filter_was_trained": False, "actor_last_action_is_requested": True,
        "wheel_outputs_filtered": False, "continuous_contact_claimed_by_renderer": False,
        "target_step_hud_source": "Audited349-decision processed/applied-target sidecar; core1.44-4.00s; NOT raw requested actions",
        "rendering_limitations": ["Video50Hz subsamples native200Hz recorded states and cannot prove continuous contact between displayed frames.",
            "Counts are measured ground points. Zero count is not proof of flight; eligibility=count>0 AND upward Fz>5N is diagnostic only.",
            "Both panels use the same frozen actor/checkpoint, original action-rate-.5 reward and PD500/10. Only six processed leg targets are filtered in alpha.5.",
            "Wheels are not filtered and actor action history remains requested. Closed-loop wheel outputs can differ across episodes because physical feedback differs.",
            "Alpha.5 adds unobserved actuator state/delay to an unretrained policy. Replay is not contact-stability or safety certification.",
            "Original world pose, height and timing are retained; the common camera follows reference, not either robot."]})
    return traces, reference_arrays, summary


def render_frame(runtime, traces, qposes, reference, summary, index):
    from PIL import Image, ImageDraw
    w, h = runtime.width, runtime.height
    require((2*w, h+146) == CANVAS_SIZE, "Exact even video geometry required; no auto-resize")
    require(isinstance(index, int) and 0 <= index < 1397, "Invalid original native frame index")
    image = Image.new("RGB", CANVAS_SIZE, "#fafafa"); draw = ImageDraw.Draw(image)
    time = float(traces["baseline"]["time_s"][index])
    draw.text((12, 6), f"Same PhysX PPO actor8ba4178471c7 | mu=0.7 | Kp500/Kd10 | original t={time:.2f}s", font=runtime.font, fill="black")
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
        delta = core_applied_target_step_rms(summary, label)
        draw.text((x+10, y+h+70), f"Core applied-target step RMS {delta:.4f}rad (50Hz) | wheels/history unfiltered", font=runtime.small, fill="#333333")
    draw.text((12, image.height-20), "Recorded200Hz / video50Hz; opaque FK only. Same actor, zero-shot target delay (not retrained). No pose correction or flight label.", font=runtime.small, fill="#333333")
    return image


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh output directory")
    traces, reference, summary = load_data(args.input_root, args.reference, model=args.model)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes = {}
        for label in ORDER:
            qposes[label], mapping = build_qpos(runtime.model, traces[label])
            summary["cases"][label]["fk_validation"] = validate_fk(runtime.model, traces[label], qposes[label])
            summary["cases"][label]["named_joint_qpos_addresses"] = mapping.tolist()
        summary["mesh_provenance"] = mesh_provenance(args.model)
        output.mkdir(parents=True, exist_ok=False)
        arrays = {"time_s": traces["baseline"]["time_s"], "video_native_indices": np.arange(0, 1397, 4)}
        for label in ORDER:
            arrays.update({label+"_qpos": qposes[label], label+"_original_root_position_m": traces[label]["position"],
                label+"_recorded_contact_count": traces[label]["contact_count"],
                label+"_recorded_ground_force_w_n": traces[label]["contact_force"]})
        np.savez_compressed(output/"replay_states.npz", **arrays)
        render_frame(runtime, traces, qposes, reference, summary, 480).save(output/"comparison.png")
        artifacts = ["replay_states.npz", "comparison.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            first = render_frame(runtime, traces, qposes, reference, summary, 0)
            require(first.size == CANVAS_SIZE, "Unexpected video canvas")
            stream = imageio_ffmpeg.write_frames(str(output/"fullbodycomparison.mp4"), first.size, fps=50,
                codec="libx264", quality=7, macro_block_size=2,
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for index in range(0, 1397, 4):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, summary, index)))
            finally:
                stream.close()
            artifacts.append("fullbodycomparison.mp4")
        summary["video_generated"] = not args.skip_video
        summary["renderer_source_sha256"] = sha256(__file__)
        summary["rendering_helper_source_sha256"] = {name: sha256(ROOT/"scripts"/name) for name in
            ("analyze_ppo_leg_target_filter.py", "analyze_ppo_smoothness.py", "render_ppo_ground_support_comparison.py",
             "render_physx_ppo_fullbody.py", "render_ppo_smoothness_comparison.py", "controller_trace.py", "nmpc_tracking_objective.py")}
        summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in artifacts}
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, required=True)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
    print(json.dumps({"status": "rendered_existing_fixed_actor_leg_target_filter_comparison", "same_actor": True, "dynamics_stepping": False}))

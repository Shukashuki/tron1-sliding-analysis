"""Four staged PPO recipes, replayed with one fixed-distance follow camera.

All cells are existing nominal PhysX recordings. This is NOT an isolated
timestep ablation: actors, action-rate regularization and training DR differ.
The original grounded reward +0.4 remains; the dense-contact trial is excluded.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import analyze_ppo_beta_jitter as beta
import analyze_ppo_contact_timestep as timestep
from analyze_ppo_backend_400hz_matrix import COLORS, heading_geometry_masks, load_fk, reference_data
from controller_trace import native_trace
from ppo_backend_400hz_contract import DEFAULT_PROTOCOL, INPUT_PINS, read_json, require, sha256, write_json
from render_physx_ppo_fullbody import DEFAULT_MODEL, build_qpos, create_runtime, mesh_provenance, _pose
from render_ppo_backend_400hz_nominal import CANVAS_SIZE, diagnostics, replay_fk_audit
from render_ppo_beta_nominal import DEFAULT_LEGACY, validate_native, world_bounds
from render_ppo_fixed_follow import prepare_camera, apply_camera
from run_ppo_contact_damping import validate_output as validate_damping
from summarize_ppo_backend_400hz_nominal import load_nominal

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("original200", "smooth200", "deploy400", "retrain400")
TITLES = {"original200": "A | original 200 Hz | action-rate −0.1",
    "smooth200": "B | retrained 200 Hz | action-rate −0.5",
    "deploy400": "C | same B actor, deployed at 400 Hz",
    "retrain400": "D | new 400 Hz training | action-rate −0.5"}
DEFAULT_ORIGINAL = beta.DEFAULT_ON
DEFAULT_SMOOTH = ROOT / "outputs/contact_damping_20261021_v2/kd10"
DEFAULT_TIMESTEP = ROOT / "outputs/ppo_contact_timestep_20261022_v1"
DEFAULT_NOMINAL = ROOT / "outputs/ppo_backend_400hz_nominal_20261022_v1"
DEFAULT_SMOOTH_TRAINING = ROOT / "outputs/ppo_smoothness_20261021_v2/action_rate_0p5/training"
CHECKPOINTS = {"original200": beta.IDENTITIES["beta_on"]["checkpoint"],
    "smooth200": timestep.CHECKPOINT_SHA256, "deploy400": timestep.CHECKPOINT_SHA256,
    "retrain400": "c508dc3ba50db81f56f538bf23c550aca12f3279ddd15749d60582a487d2d8c7"}
DT = {"original200": .005, "smooth200": .005, "deploy400": .0025, "retrain400": .0025}
SMOOTH_RAW_SHA = "54e01f57b01da11c5b0736c572f8b2ebc81ebe2cac8f06c9b3b5be29b1637780"
SMOOTH_REPORT_SHA = "fdf3881b765276832dbbdfe0528446055a4af91a6147ca3520dc847f6906dc77"
DRIFT_SHA = "5fee46238c825a2bcf474e969e93af65834aa1302ce857aa1c553e8a9fbaefaa"
SMOOTH_TRAINING_PINS = {"manifest.json": "066ddaeb5a23ff44dfca79e785520c57d8786cdc3a2d7b30d118e3a955a94d63",
    "ablation_manifest.json": "4cf22f9c4e6b5f28aa1711f2e093516522046d259e0322b740e7cad5da05905e"}


def native_indices(stage):
    require(stage in ORDER, "Unknown staged recipe")
    return np.arange(350, dtype=np.int64) * int(round(.02 / DT[stage]))


def read_raw(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def verify_smooth_training(folder):
    folder = Path(folder)
    require(all(sha256(folder/name) == digest for name, digest in SMOOTH_TRAINING_PINS.items()),
            "B/C actual training manifest/action-rate ablation source differs")
    manifest, ablation = (read_json(folder/name) for name in ("manifest.json", "ablation_manifest.json"))
    require(ablation["status"] == "training_completed" and ablation["smoke_only"] is False
            and ablation["initialization"] == "from_scratch_no_checkpoint_resume"
            and ablation["checkpoint_sha256"] == CHECKPOINTS["smooth200"]
            and ablation["actual_environment_transitions"] == 12288000
            and ablation["final_training_manifest_sha256"] == SMOOTH_TRAINING_PINS["manifest.json"]
            and ablation["reward_override"]["term"] == "action_rate_l2"
            and ablation["reward_override"]["baseline_weight"] == -.1
            and ablation["reward_override"]["weight"] == -.5
            and manifest["training_reward_overrides"] == {"action_rate_l2": -.5},
            "B/C completed from-scratch training identity differs")
    return {"directory": str(folder.resolve()), "sha256": SMOOTH_TRAINING_PINS.copy(),
        "reward_override": ablation["reward_override"], "training_reward_overrides": manifest["training_reward_overrides"],
        "checkpoint_sha256": ablation["checkpoint_sha256"], "environment_transitions": 12288000,
        "profile_alone_does_not_include_action_rate_override": True}


def validate_stage(stage, trace, report):
    dt = DT[stage]; n = 349 * int(round(.02 / dt)) + 1
    require(trace["physics_dt_s"] == dt and trace["time_s"].shape == (n,)
            and np.array_equal(trace["time_s"], np.arange(n) * dt), "Stage native clock differs: " + stage)
    require(report.get("status") == "evaluated" and report.get("simulator") == "IsaacLab/PhysX"
            and report.get("checkpoint_sha256") == CHECKPOINTS[stage]
            and report.get("motion_file_sha256") == beta.MOTION_SHA
            and report.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") == .7
            and report.get("drift_module_sha256") == DRIFT_SHA, "Stage actor/reference/material/reward source differs: " + stage)
    contract = report["policy_export"]; profile = contract["task_profile"]
    require(contract["physics_dt"] == dt and contract["decimation"] == int(round(.02 / dt))
            and contract["policy_fps"] == 50. and contract["leg_kp"] == 500. and contract["leg_kd"] == 10.
            and profile["reward_weights"]["grounded"] == .4
            and profile["reward_weights"]["sideslip"] == 2.
            and profile["height_regularization"]["mode"] == "vertical_acceleration_only",
            "Stage clock/PD/original grounded/beta/free-height recipe differs: " + stage)
    require(np.allclose(trace["time_s"][native_indices(stage)], np.arange(350)*.02, rtol=0, atol=1e-12),
            "Exact native selections do not share the 50 Hz display clock")


def load_stages(reference=beta.DEFAULT_REFERENCE, model=DEFAULT_MODEL, legacy_evaluator=DEFAULT_LEGACY,
                *, original_dir=DEFAULT_ORIGINAL, smooth_dir=DEFAULT_SMOOTH,
                timestep_root=DEFAULT_TIMESTEP, nominal_root=DEFAULT_NOMINAL, protocol=DEFAULT_PROTOCOL,
                smooth_training_dir=DEFAULT_SMOOTH_TRAINING):
    """Strictly load existing A/B/C/D recordings, retaining their native clocks."""
    require(sha256(model) == INPUT_PINS["model"] and sha256(reference) == beta.MOTION_SHA,
            "Locked opaque display model/reference differs")
    original_dir, smooth_dir, timestep_root, nominal_root = map(Path,
        (original_dir, smooth_dir, timestep_root, nominal_root))
    smooth_training = verify_smooth_training(smooth_training_dir)
    original_report = read_json(original_dir / "report.json")
    off_report_path = beta.DEFAULT_OFF / "report.json"
    require(sha256(original_dir / "report.json") == beta.IDENTITIES["beta_on"]["report"]
            and sha256(original_dir / "physics_trajectory.npz") == beta.IDENTITIES["beta_on"]["trajectory"]
            and sha256(off_report_path) == beta.IDENTITIES["beta_off"]["report"], "Pinned original historical inputs differ")
    historical_audit = beta.audit_reports(original_report, read_json(off_report_path))
    require(original_report["physics_trajectory_sha256"] == beta.IDENTITIES["beta_on"]["trajectory"],
            "Original report/native recording binding differs")
    raw_a = read_raw(original_dir / "physics_trajectory.npz")
    trace_a = validate_native(raw_a, "physx")
    damping_proof = validate_damping(smooth_dir, 10.)
    require(damping_proof["recording_sha256"] == SMOOTH_RAW_SHA
            and damping_proof["report_sha256"] == SMOOTH_REPORT_SHA, "Specified B kd10 artifact differs")
    paired_raw, paired_summary = timestep.load_comparison(timestep_root, reference)
    require(paired_summary["cases"]["baseline"]["input_sha256"]["physics_trajectory.npz"] == SMOOTH_RAW_SHA,
            "B is not the exact timestep baseline recording")
    b_report = read_json(smooth_dir / "report.json")
    require(b_report["contact_stability_diagnostic"]["trained_action_rate_weight"] == -.5,
            "B training action-rate override differs")
    nominal, nominal_reference, nominal_summary = load_nominal(nominal_root, reference, model, legacy_evaluator, protocol)
    require(nominal_summary["nominal_only"] is True and nominal_summary["full_matrix_complete"] is False,
            "D must come from the verified nominal partial-matrix retrieval")
    d_rows = [row for row in nominal_summary["rows"] if row["id"] == "physx_to_physx"]
    require(len(d_rows) == 1 and d_rows[0]["checkpoint_sha256"] == CHECKPOINTS["retrain400"],
            "New native400Hz PhysX-trained actor differs")
    folders = {"original200": original_dir, "smooth200": smooth_dir,
        "deploy400": timestep_root / "dt0p0025", "retrain400": Path(d_rows[0]["directory"])}
    reports = {stage: read_json(folder / "report.json") for stage, folder in folders.items()}
    require(reports["retrain400"]["policy_export"]["trained_task_profile"]["action_rate_weight"] == -.5,
            "D training action-rate recipe differs")
    traces = {"original200": trace_a, "smooth200": native_trace(read_raw(smooth_dir / "physics_trajectory.npz"), "policy_isaac"),
        "deploy400": native_trace(paired_raw["candidate"], "policy_isaac"), "retrain400": nominal[("physx", "physx")]}
    fk, fk_source = load_fk(model, legacy_evaluator)
    rows = []
    for stage in ORDER:
        trace, report, folder = traces[stage], reports[stage], folders[stage]
        validate_stage(stage, trace, report)
        reconstructed = fk.reconstruct(trace)
        trace["signals"], trace["wheel_quaternion_wxyz"] = reconstructed["signals"], reconstructed["wheel_quaternion_wxyz"]
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = heading_geometry_masks(trace)
        rows.append({"id": stage, "title": TITLES[stage], "directory": str(folder.resolve()),
            "checkpoint_sha256": CHECKPOINTS[stage], "report_sha256": sha256(folder / "report.json"),
            "recording_sha256": sha256(folder / "physics_trajectory.npz"),
            "training_physics_hz": 400 if stage == "retrain400" else 200,
            "evaluation_physics_hz": int(round(1./DT[stage])), "physics_dt_s": DT[stage],
            "policy_hz": 50, "action_rate_weight": -.1 if stage == "original200" else -.5,
            "training_DR_recipe": "shared mjlab-style backend-comparison DR" if stage == "retrain400" else "historical BeyondMimic-style DR",
            "beta_reward_weight": 2., "sideslip_reward_weight": 2., "grounded_reward_weight": .4,
            "native_samples": len(trace["time_s"]), "native_sample_stride": int(round(.02/DT[stage])),
            "parent_verification": historical_audit if stage == "original200" else damping_proof if stage == "smooth200"
                else paired_summary["cases"]["candidate"]["parent_verification"] if stage == "deploy400" else d_rows[0]["remote_verification"],
            "wheel_FK_audit": reconstructed["audit"]})
    reference_values = reference_data(reference)
    require(np.array_equal(reference_values["time_s"], nominal_reference["time_s"])
            and np.array_equal(reference_values["position"], nominal_reference["position"]), "Shared reference decoder differs")
    return traces, reference_values, {"schema_version": 1, "kind": "recorded_ppo_staged_training_recipe_fullbody_comparison",
        "stage_order": list(ORDER), "rows": rows, "mu": .7, "training_backend": "PhysX", "validation_backend": "PhysX",
        "reference_sha256": sha256(reference), "model_sha256": sha256(model), "FK_provenance": fk_source,
        "historical_auxiliary_off_report_sha256": sha256(off_report_path),
        "new400_retrieval_manifest_sha256": nominal_summary["retrieval_manifest_sha256"],
        "new400_protocol_sha256": nominal_summary["protocol_sha256"],
        "B_C_actual_training_override_provenance": smooth_training,
        "grounded_reward_weight_all": .4, "grounded_reward_function": "original last-policy-substep drift_grounded",
        "all_main_stages_sideslip_reward_weight": 2.,
        "reward_implementation_evaluation_sha256_all": DRIFT_SHA,
        "dense_ground_support_reward_used": False, "additional_torque_reward_used": False, "leg_target_filter_used": False,
        "B_C_same_checkpoint": True, "D_from_scratch_retrained": True, "causal_hz_only_ablation_claimed": False,
        "sample_size_per_stage": 1, "video_frames": 350, "video_fps": 50, "encoded_duration_s": 7.,
        "last_original_time_s": 6.98, "reference_real_last_time_s": 6.96,
        "actor_weights_loaded": False, "new_simulation": False, "new_training": False, "new_policy_inference": False,
        "state_interpolation": False, "pose_alignment": False, "root_height_corrected": False,
        "height_or_flight_failure_gate": False, "stable_contact_claimed": False, "visual_reviewed": False,
        "limitations": ["Staged recipes, not a pure timestep causal comparison. A/B use different independently trained actors and action-rate weights.",
            "B→C is the same frozen actor: both integration and explicit PD update rate change200→400Hz; policy remains50Hz.",
            "D is a new from-scratch400Hz cohort with a shared mjlab-style DR recipe, unlike the historical BeyondMimic-style DR in A/B/C.",
            "Original grounded+0.4 is unchanged in all four stages. No dense-contact reward, target low-pass filter or extra torque-rate reward is included.",
            "One seed/episode per stage; no statistical or hardware-stability claim.50Hz video cannot resolve all native200/400Hz jitter.",
            "Native poses/heights/falls are unchanged. Native contact counts/Fz are diagnostics; zero does not establish geometric flight.",
            "One camera distance for all350frames/allfourcases, fixed root-relative offset and world orientation. Fixed world maps retain XY drift.",
            "Reference ends6.96s; XY tracking is unavailable beyond it, with no endpoint extrapolation."]}


def selected_views(traces, qposes):
    """Exact native selections, used only for graphics and camera fitting."""
    selected_traces, selected_poses = {}, {}
    for stage in ORDER:
        indices = native_indices(stage); trace = traces[stage]
        selected_traces[stage] = {name: value[indices] if isinstance(value, np.ndarray) and value.ndim > 0
            and len(value) == len(trace["time_s"]) else value for name, value in trace.items() if name != "signals"}
        selected_traces[stage]["signals"] = {name: value[indices] if isinstance(value, np.ndarray) and value.ndim > 0
            and len(value) == len(trace["time_s"]) else value for name, value in trace["signals"].items()}
        selected_poses[stage] = qposes[stage][indices]
    return selected_traces, selected_poses


def _map_panel(draw, trace, reference, index, box, bounds, font):
    """Same opaque axial glyphs as the frozen map; retain EVERY50Hz path point."""
    low, high = np.asarray(bounds, dtype=float)
    left, top, right, bottom = box
    scale = min((right-left-20)/(high[0]-low[0]), (bottom-top-28)/(high[1]-low[1]))
    center = (low+high)/2.
    pixel = lambda point: ((left+right)/2.+(point[0]-center[0])*scale,
                           (top+bottom)/2.-(point[1]-center[1])*scale)
    draw.rectangle(box, fill="#f7f9fa", outline="#c9cdd1")
    draw.text((left+8, top+3), "Fixed world-XY: ref grey | actual green | body blue | L red / R purple ↔", font=font, fill="#333333")
    draw.text((left+8, bottom-18), f"X→ [{low[0]:.2f},{high[0]:.2f}]m | Y↑ [{low[1]:.2f},{high[1]:.2f}]m | same bounds/scale in all panels", font=font, fill="#555555")
    draw.line([pixel(point) for point in reference["position"][:, :2]], fill=COLORS["reference"], width=2)
    if index >= 1:
        draw.line([pixel(point) for point in trace["position"][:index+1, :2]], fill=COLORS["path"], width=2)
    root = trace["position"][index, :2]; x, y = pixel(root)
    draw.ellipse((x-3, y-3, x+3, y+3), fill=COLORS["path"])
    signals = trace["signals"]
    arrows = [(root, signals["body_heading_rad"][index], "body", trace["body_heading_geometry_valid"][index], False)]
    for wheel, color in enumerate(("left", "right")):
        arrows.append((trace["wheel_pos"][index, wheel, :2], signals["wheel_rolling_plane_heading_rad"][index, wheel], color,
                       trace["wheel_heading_geometry_valid"][index, wheel], True))
    for point, heading, color, valid, axial in arrows:
        if not valid:
            continue
        direction = .18*np.array([np.cos(heading), np.sin(heading)])
        start = point-direction/2. if axial else point
        stop = point+direction/2. if axial else point+direction
        draw.line((pixel(start), pixel(stop)), fill=COLORS[color], width=3)
        for tip, sign in ((stop, 1.), (start, -1.)) if axial else ((stop, 1.),):
            tip_px = np.asarray(pixel(tip)); axis = np.array([np.cos(heading), -np.sin(heading)])*sign
            normal = np.array([-axis[1], axis[0]])
            draw.polygon([tuple(tip_px), tuple(tip_px-axis*6.+normal*3.), tuple(tip_px-axis*6.-normal*3.)], fill=COLORS[color])


def render_frame(runtime, traces, qposes, display_traces, reference, summary, plan, frame):
    from PIL import Image, ImageDraw
    require(type(frame) is int and 0 <= frame < 350, "Invalid staged video frame")
    image = Image.new("RGB", CANVAS_SIZE, "#fafafa"); draw = ImageDraw.Draw(image)
    draw.text((12, 5), f"PPO staged recipes | PhysX → PhysX | μ=0.7, β reward=2 | t={frame*.02:.2f} s | fixed-distance follow camera, no zoom", font=runtime.font, fill="#111111")
    for column, stage in enumerate(ORDER):
        index = int(native_indices(stage)[frame]); camera = apply_camera(runtime, plan, stage, frame)
        _pose(runtime.model, runtime.data, qposes[stage][index])
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        image.paste(Image.fromarray(runtime.renderer.render().copy()), (column*640, 58))
        x = column*640
        draw.rectangle((x, 30, x+640, 57), fill="#f0f2f4")
        draw.text((x+9, 34), TITLES[stage], font=runtime.font, fill="#111111")
        value = diagnostics(traces[stage], reference, index)
        xy = f"{value['xy_error_m']:.3f} m" if value["xy_error_m"] is not None else "n/a (ref ends 6.96 s)"
        draw.text((x+9, 424), f"XY {xy} | speed {value['speed_m_s']:.2f} m/s | tilt {value['tilt_deg']:.1f} deg", font=runtime.font, fill="#111111")
        draw.text((x+9, 447), f"Original height {value['height_m']:.3f} m | native {int(1./DT[stage])} Hz / video 50 Hz", font=runtime.font, fill="#111111")
        draw.text((x+9, 470), f"L/R count {int(value['count'][0])}/{int(value['count'][1])} | Fz {value['fz'][0]:.0f}/{value['fz'][1]:.0f} N", font=runtime.font, fill="#111111")
        _map_panel(draw, display_traces[stage], reference, frame, (x+5, 500, x+635, 672), summary["world_bounds"], runtime.small)
    draw.text((12, 683), "Original grounded +0.4 in ALL stages; no dense-contact reward, extra torque-rate reward or target filter. Staged recipes, NOT a causal Hz-only ablation; training DR also differs.", font=runtime.small, fill="#333333")
    draw.text((12, 702), "N=1/stage. Opaque original poses; no new physics/inference/alignment. Constant-distance following; shared world-XY maps keep drift. Contact zeros are not flight labels.", font=runtime.small, fill="#333333")
    return image


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve previous evidence; use a fresh output directory")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    require(os.environ["MUJOCO_GL"] == "osmesa", "Require local CPU OSMesa graphics")
    traces, reference, summary = load_stages(args.reference, args.model, args.legacy_mujoco_evaluator,
        original_dir=args.original_dir, smooth_dir=args.smooth_dir, timestep_root=args.timestep_root,
        nominal_root=args.nominal_root, protocol=args.protocol, smooth_training_dir=args.smooth_training_dir)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes, fk_audits = {}, {}
        for stage in ORDER:
            qposes[stage], ids = build_qpos(runtime.model, traces[stage])
            fk_audits[stage] = {**replay_fk_audit(runtime.model, traces[stage], qposes[stage]), "named_joint_qpos_addresses": ids.tolist()}
        display_traces, display_poses = selected_views(traces, qposes)
        plan = prepare_camera(runtime, display_traces, display_poses, ORDER, np.arange(350, dtype=np.int64))
        summary.update(world_bounds=[value.tolist() for value in world_bounds(traces, reference)],
            fixed_follow_camera={key: value for key, value in plan.items() if key != "lookat_by_case"},
            camera_fit_input="exact original native stride4/4/8/8 selections; helper indices are display-frame indices, not shared native indices",
            mesh_provenance=mesh_provenance(args.model), fk_audits=fk_audits, renderer_sha256=sha256(__file__))
        output.mkdir(parents=True, exist_ok=False)
        arrays = {}
        for stage in ORDER:
            arrays.update({stage+"_native_time_s": traces[stage]["time_s"], stage+"_qpos": qposes[stage],
                stage+"_selected_native_indices": native_indices(stage), stage+"_camera_lookat_m": plan["lookat_by_case"][stage]})
        np.savez_compressed(output / "replay_states.npz", **arrays, fixed_camera_distance_m=np.array(plan["distance_m"]))
        render_frame(runtime, traces, qposes, display_traces, reference, summary, plan, 150).save(output / "comparison.png")
        files = ["replay_states.npz", "comparison.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output / "training_recipe_fullbody_comparison.mp4"), CANVAS_SIZE, fps=50,
                codec="libx264", quality=7, macro_block_size=2, output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(350):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, display_traces, reference, summary, plan, frame)))
            finally:
                stream.close()
            files.append("training_recipe_fullbody_comparison.mp4")
        summary["video_generated"] = not args.skip_video
        summary["helpers_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in
            ("render_ppo_fixed_follow.py", "render_ppo_backend_400hz_nominal.py", "render_physx_ppo_fullbody.py",
             "render_ppo_beta_nominal.py", "analyze_ppo_beta_jitter.py", "analyze_ppo_contact_timestep.py",
             "run_ppo_contact_damping.py", "run_ppo_contact_timestep.py", "summarize_ppo_backend_400hz_nominal.py", "controller_trace.py")}
        summary["artifacts"] = {name: sha256(output / name) for name in files}
        write_json(output / "summary.json", summary)
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL)
    result.add_argument("--smooth-dir", type=Path, default=DEFAULT_SMOOTH)
    result.add_argument("--timestep-root", type=Path, default=DEFAULT_TIMESTEP)
    result.add_argument("--nominal-root", type=Path, default=DEFAULT_NOMINAL)
    result.add_argument("--smooth-training-dir", type=Path, default=DEFAULT_SMOOTH_TRAINING)
    result.add_argument("--reference", type=Path, default=beta.DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--legacy-mujoco-evaluator", type=Path, default=DEFAULT_LEGACY)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())

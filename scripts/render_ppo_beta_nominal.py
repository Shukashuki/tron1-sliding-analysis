"""Historical200Hz PhysX-trained PPO beta ON/OFF, four horizontal replays.

This is NOT the new400Hz matched training cohort or a causal beta-only trial.
No actor is loaded; original poses are replayed through opaque graphics/FK.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import numpy as np

import analyze_ppo_beta_jitter as beta
import analyze_ppo_smoothness as smooth
from analyze_ppo_backend_400hz_matrix import heading_geometry_masks, load_fk, reference_data
from analyze_ppo_contact_timestep import band_spectrum
from controller_trace import native_trace
from ppo_backend_400hz_contract import INPUT_PINS, read_json, require, sha256, write_json
from render_physx_ppo_fullbody import DEFAULT_MODEL, build_qpos, create_runtime, mesh_provenance, _pose
from render_ppo_backend_400hz_nominal import CANVAS_SIZE, diagnostics, _map_panel, replay_fk_audit
from render_ppo_fixed_follow import prepare_camera, apply_camera

ROOT = Path(__file__).resolve().parents[1]
ORDER = (("beta_on", "physx"), ("beta_off", "physx"), ("beta_on", "mujoco"), ("beta_off", "mujoco"))
TITLES = {("beta_on", "physx"): "β reward ON (2) → PhysX",
    ("beta_off", "physx"): "β reward OFF (0) → PhysX",
    ("beta_on", "mujoco"): "β reward ON (2) → native CPU MuJoCo",
    ("beta_off", "mujoco"): "β reward OFF (0) → native CPU MuJoCo"}
DEFAULT_ON = ROOT / "outputs/controller_primary_ppo_mu070_20261015_v1/mu0p700"
DEFAULT_OFF = ROOT / "outputs/no_sideslip_ppo_mu0p7_validation_20261016_v1"
DEFAULT_LEGACY = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_mujoco.py"
MJ_PINS = {
    "beta_on": {"report": "5f79ac1f8296277b5c65d028a344a5ad5a4bdbed920bbf56848ee6f8e6c7910b",
        "trajectory": "6db6516fd0d2bc5446a38c7787793ba6ff86fa3238a3b354543905149a3cd027",
        "actor": "a20e298d937f0812349f8b84112efe93101b4ab4aa9923d57a4bd5accc8c4753"},
    "beta_off": {"report": "517d6bd04172b03024247d0fe1e1ad019a64a4b418a3b0de91078f37c205834e",
        "trajectory": "a7c9e6ab2081f5ec5f6fe194ae0bf9eb6ee191f39669adaee0b1c0d4e4b88da5",
        "actor": "ad9c2e31b3cd2616219d81a7c91b44e9625c45e08a942b6e25ee8f0e8a59b931"}}


def native_indices():
    return np.arange(350, dtype=np.int64) * 4


def validate_native(raw, engine):
    trace = native_trace(raw, "policy_isaac" if engine == "physx" else "policy_mujoco")
    require(len(trace["time_s"]) == 1397 and trace["physics_dt_s"] == .005
            and np.allclose(trace["time_s"], np.arange(1397)*.005, rtol=0, atol=1e-10)
            and float(raw["policy_dt_s"]) == .02, "Require original200Hz/50Hz complete native clock")
    if engine == "physx":
        require(raw["valid_mask"].all() and raw["action_was_applied"].shape == (1397, 1)
                and not raw["action_was_applied"][0, 0] and raw["action_was_applied"][1:].all(), "PhysX native decision coverage differs")
        held = raw["actions_clipped"][1:, 0].reshape(349, 4, 8)
        require(np.array_equal(held, np.repeat(held[:, :1], 4, axis=1)), "PhysX actions changed within held interval")
    else:
        require(raw["actions_clipped"].shape == (349, 8) and raw["action_observation"].shape == (349, 51)
                and np.allclose(raw["policy_time_s"], np.arange(349)*.02, rtol=0, atol=1e-10), "MuJoCo native decision coverage differs")
        require(np.array_equal(raw["previous_clipped_actions"][0], np.zeros(8))
                and np.array_equal(raw["previous_clipped_actions"][1:], raw["actions_clipped"][:-1]), "Previous action history differs")
    require(np.array_equal(raw["actions_clipped"], np.clip(raw["actions_raw"], -1., 1.)), "Historical action clipping differs")
    return trace


def load_historical(on_root=DEFAULT_ON, off_root=DEFAULT_OFF, reference=beta.DEFAULT_REFERENCE,
                    model=DEFAULT_MODEL, legacy_evaluator=DEFAULT_LEGACY):
    require(sha256(model) == INPUT_PINS["model"], "Original display model differs")
    reference_values = reference_data(reference)
    roots = {"beta_on": Path(on_root), "beta_off": Path(off_root)}
    physx_reports = {label: read_json(path / "isaac/report.json") for label, path in roots.items()}
    parity = beta.audit_reports(physx_reports["beta_on"], physx_reports["beta_off"])
    fk, fk_provenance = load_fk(model, legacy_evaluator)
    traces, rows, raw_by_key = {}, [], {}
    for label, engine in ORDER:
        folder = roots[label] / ("isaac" if engine == "physx" else "mujoco")
        recording = folder / ("physics_trajectory.npz" if engine == "physx" else "rollout.npz")
        report_file = folder / "report.json"
        report = physx_reports[label] if engine == "physx" else read_json(report_file)
        pin = beta.IDENTITIES[label] if engine == "physx" else MJ_PINS[label]
        require(sha256(report_file) == pin["report"] and sha256(recording) == pin["trajectory"], "Pinned historical report/recording differs")
        contract = report["policy_export"] if engine == "physx" else report["contract"]
        require(contract == physx_reports[label]["policy_export"] and contract["checkpoint_sha256"] == beta.IDENTITIES[label]["checkpoint"],
                "Historical cross-engine actor/deployment contract differs")
        require(contract["physics_dt"] == .005 and contract["decimation"] == 4 and contract["leg_kp"] == 500.
                and contract["leg_kd"] == 10. and contract["task_profile"].get("action_rate_weight", -.1) == -.1,
                "Historical200Hz/action-rate/controller contract differs")
        if engine == "mujoco":
            require(report["status"] == "completed" and report["full_horizon_diagnostic"] is True
                    and report["completed_full_reference"] is True and report["root_state_writes"] == 1 and report["hidden_resets"] == 0
                    and report["start_reference_frame"] == 0 and report["motion_sha256"] == beta.MOTION_SHA
                    and report["model_sha256"] == INPUT_PINS["model"] and report["policy_sha256"] == pin["actor"]
                    and report["domain_parameters_resampled_during_episode"] is False
                    and report["contact_material_audit"]["compiled_friction_range"] == [.7, .7], "Historical native MuJoCo evaluation differs")
        with np.load(recording, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        trace = validate_native(raw, engine)
        reconstructed = fk.reconstruct(trace)
        trace["signals"], trace["wheel_quaternion_wxyz"] = reconstructed["signals"], reconstructed["wheel_quaternion_wxyz"]
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = heading_geometry_masks(trace)
        tracking = smooth._tracking(trace["time_s"], trace["position"], trace["com_velocity"], trace["quaternion"], reference_values, (1.44, 4.))
        selected = (trace["time_s"] >= 1.44-1e-9) & (trace["time_s"] <= 4.+1e-9)
        require(int(selected.sum()) == 513 and tracking["available"], "Historical core incomplete")
        rows.append({"beta_reward": label, "validation_engine": "PhysX" if engine == "physx" else "native CPU MuJoCo",
            "checkpoint_sha256": beta.IDENTITIES[label]["checkpoint"], "recording_sha256": pin["trajectory"],
            "report_sha256": pin["report"], "directory": str(folder), "core_tracking": tracking,
            "core_world_angular_xy_common10_100": band_spectrum(trace["angular_velocity"][selected, :2], .005),
            "wheel_FK_audit": reconstructed["audit"]})
        traces[(label, engine)], raw_by_key[(label, engine)] = trace, raw
    for engine in ("physx", "mujoco"):
        for name in ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel", "wheel_pos"):
            require(np.array_equal(traces[("beta_on", engine)][name][0], traces[("beta_off", engine)][name][0]),
                    "Historical beta ON/OFF initial state differs: " + name)
    require(np.array_equal(raw_by_key[("beta_on", "mujoco")]["initial_qpos"], raw_by_key[("beta_off", "mujoco")]["initial_qpos"])
            and np.array_equal(raw_by_key[("beta_on", "mujoco")]["initial_qvel"], raw_by_key[("beta_off", "mujoco")]["initial_qvel"]),
            "Historical native initial qpos/qvel differs")
    return traces, reference_values, {"kind": "historical200hz_physx_trained_beta_nominal_four_horizontal_replay",
        "rows": rows, "training_backend": "PhysX", "training_physics_hz": 200, "validation_physics_hz": 200,
        "policy_hz": 50, "video_fps": 50, "video_frames": 350, "native_sample_stride": 4,
        "action_rate_weight": -.1, "beta_reward_weights": {"beta_on": 2., "beta_off": 0.}, "mu": .7,
        "reference_sha256": sha256(reference), "model_sha256": sha256(model), "FK_provenance": fk_provenance,
        "evaluation_profile_parity": parity, "same_initial_state_ON_OFF_per_engine": True,
        "new400hz_cohort": False, "causal_beta_only_intervention_claimed": False, "sample_size_per_cell": 1,
        "setup_smoke_steps": {"beta_on": 0, "beta_off": 8}, "training_GPU": {"beta_on": "cuda:2", "beta_off": "cuda:7"},
        "actor_weights_loaded": False, "new_simulation": False, "new_training": False, "visual_reviewed": False,
        "limitations": ["Historical200Hz cohort, action-rate-0.1, not the new400Hz/action-rate-0.5 comparison.",
            "ON/OFF training setup differed:0vs8 zero-action smoke steps and GPUs2vs7; not a strict causal beta-only intervention.",
            "Same full native evaluation/source profiles except beta reward, but one seed/actor per condition is descriptive only.",
            "Nonzero sideslip remains a valid intended motion feature. No zero-beta, height, contact or flight success gate is imposed.",
            "The moving camera has a fixed root-relative translation offset, fixed world orientation and ONE distance for all frames/cases.",
            "Fixed world-XY maps retain absolute drift; camera following does not mean trajectory error was removed."]}


def world_bounds(traces, reference):
    points = np.concatenate([reference["position"][:, :2], *[trace["position"][:, :2] for trace in traces.values()],
                             *[trace["wheel_pos"][:, :, :2].reshape(-1, 2) for trace in traces.values()]])
    return points.min(axis=0)-.25, points.max(axis=0)+.25


def render_frame(runtime, traces, qposes, reference, summary, plan, frame):
    from PIL import Image, ImageDraw
    require(0 <= frame < 350, "Invalid historical video frame")
    index = int(native_indices()[frame])
    image = Image.new("RGB", CANVAS_SIZE, "#fafafa"); draw = ImageDraw.Draw(image)
    draw.text((12, 5), f"HISTORICAL 200 Hz PhysX-trained PPO | μ=0.7 | t={traces[ORDER[0]]['time_s'][index]:.2f} s | β reward ON/OFF | fixed-distance follow camera, no zoom", font=runtime.font, fill="#111111")
    for column, key in enumerate(ORDER):
        camera = apply_camera(runtime, plan, key, frame)
        _pose(runtime.model, runtime.data, qposes[key][index])
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        image.paste(Image.fromarray(runtime.renderer.render().copy()), (column*640, 58))
        x = column*640
        draw.rectangle((x, 30, x+640, 57), fill="#f0f2f4")
        draw.text((x+9, 34), TITLES[key], font=runtime.font, fill="#111111")
        value = diagnostics(traces[key], reference, index)
        xy = f"{value['xy_error_m']:.3f} m" if value["xy_error_m"] is not None else "n/a (ref ends 6.96 s)"
        draw.text((x+9, 424), f"XY {xy} | speed {value['speed_m_s']:.2f} m/s | tilt {value['tilt_deg']:.1f} deg", font=runtime.font, fill="#111111")
        draw.text((x+9, 447), f"Original height {value['height_m']:.3f} m | native 200 Hz / video 50 Hz", font=runtime.font, fill="#111111")
        draw.text((x+9, 470), f"L/R count {int(value['count'][0])}/{int(value['count'][1])} | Fz {value['fz'][0]:.0f}/{value['fz'][1]:.0f} N", font=runtime.font, fill="#111111")
        _map_panel(draw, traces[key], reference, index, (x+5, 500, x+635, 672), summary["world_bounds"], runtime.small)
    draw.text((12, 683), "N=1 historical pair; setup smoke ON=0/OFF=8, GPUs differ. Descriptive comparison, NOT a matched causal beta-only experiment or the new 400 Hz cohort.", font=runtime.small, fill="#333333")
    draw.text((12, 702), "Opaque recorded poses only; no new physics/inference/pose correction. Fixed-distance following; world-XY maps keep original drift. Contact zeros are not flight labels.", font=runtime.small, fill="#333333")
    return image


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve previous historical replay; use a fresh output directory")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    require(os.environ["MUJOCO_GL"] == "osmesa", "Require local CPU OSMesa graphics")
    traces, reference, summary = load_historical(args.on_root, args.off_root, args.reference, args.model, args.legacy_mujoco_evaluator)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes, fk_audits = {}, {}
        for key in ORDER:
            qposes[key], ids = build_qpos(runtime.model, traces[key])
            fk_audits[key[0]+"_"+key[1]] = {**replay_fk_audit(runtime.model, traces[key], qposes[key]), "named_joint_qpos_addresses": ids.tolist()}
        plan = prepare_camera(runtime, traces, qposes, ORDER, native_indices())
        summary.update(world_bounds=[value.tolist() for value in world_bounds(traces, reference)],
            fixed_follow_camera={key: value for key, value in plan.items() if key != "lookat_by_case"},
            mesh_provenance=mesh_provenance(args.model), fk_audits=fk_audits, renderer_sha256=sha256(__file__))
        output.mkdir(parents=True)
        arrays = {}
        for key in ORDER:
            name = key[0]+"_"+key[1]
            arrays.update({name+"_native_time_s": traces[key]["time_s"], name+"_qpos": qposes[key],
                name+"_selected_native_indices": native_indices(), name+"_camera_lookat_m": plan["lookat_by_case"][key]})
        np.savez_compressed(output / "replay_states.npz", **arrays, fixed_camera_distance_m=np.array(plan["distance_m"]))
        render_frame(runtime, traces, qposes, reference, summary, plan, 150).save(output / "comparison.png")
        files = ["replay_states.npz", "comparison.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output / "beta_fullbody_comparison.mp4"), CANVAS_SIZE, fps=50,
                codec="libx264", quality=7, macro_block_size=2, output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(350):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, summary, plan, frame)))
            finally:
                stream.close()
            files.append("beta_fullbody_comparison.mp4")
        summary["video_generated"] = not args.skip_video
        summary["helpers_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in
            ("render_ppo_fixed_follow.py", "render_ppo_backend_400hz_nominal.py", "render_physx_ppo_fullbody.py", "analyze_ppo_beta_jitter.py", "controller_trace.py")}
        summary["artifacts"] = {name: sha256(output / name) for name in files}
        write_json(output / "summary.json", summary)
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--on-root", type=Path, default=DEFAULT_ON)
    result.add_argument("--off-root", type=Path, default=DEFAULT_OFF)
    result.add_argument("--reference", type=Path, default=beta.DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--legacy-mujoco-evaluator", type=Path, default=DEFAULT_LEGACY)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())

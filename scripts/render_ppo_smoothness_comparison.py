"""Replay baseline/candidate PhysX PPO meshes with one shared reference camera.

Rows are mu=.70/.15; columns are frozen baseline/action-rate candidate. Rendering
uses recorded states and kinematics only, with no physics or policy inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from controller_trace import native_trace
from nmpc_tracking_objective import quaternion_rotate
from render_physx_ppo_fullbody import (
    DEFAULT_MODEL, DEFAULT_PUBLIC_SUMMARY, build_qpos, create_runtime,
    mesh_provenance, sha256, validate_fk, _pose,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT/"outputs/policy_primary_grid_raw_20261016_v1/PPO"
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
BASELINE_CHECKPOINT = "6ec411efeaea37fc61389285978b377aa7c5be80e326ce7f18581efc9acdf27b"
MOTION_SHA256 = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
REFERENCE_SHA256 = "afc22f19b813cf10b3e3a90affb0f26f5a7743074f98b262394c69de786e9f6e"
MODEL_SHA256 = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
MU = (.7, .15)
ORDER = ("mu0p700_baseline", "mu0p700_candidate", "mu0p150_baseline", "mu0p150_candidate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def case_folder(root, mu):
    case = Path(root)/("mu"+f"{mu:.3f}".replace(".", "p"))
    paths = [path for path in (case/"isaac", case) if (path/"physics_trajectory.npz").is_file()]
    require(len(paths) == 1, "Require one unambiguous PhysX recording: " + str(case))
    return paths[0]


def load_case(folder, *, mu, checkpoint_sha256):
    folder = Path(folder)
    report_path, trace_path = folder/"report.json", folder/"physics_trajectory.npz"
    report = read_json(report_path)
    digest = sha256(trace_path)
    require(report["physics_trajectory_sha256"] == digest, "Report-bound physical recording SHA mismatch")
    require(report["checkpoint_sha256"] == checkpoint_sha256, "Policy checkpoint differs from requested cohort")
    require(report["simulator"] == "IsaacLab/PhysX" and report["status"] == "evaluated", "Require completed PhysX evaluation")
    require(report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == mu,
            "Recorded friction differs from case")
    require(report["motion_file_sha256"] == MOTION_SHA256 and report["reference_frames"] == 349,
            "Locked motion/control horizon differs")
    require(report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False,
            "Comparison requires deterministic fixed-mu evaluation without DR/noise")
    with np.load(trace_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    require(raw["valid_mask"].shape == (1397, 1) and raw["valid_mask"].all(), "Require full original first episode")
    require(raw["action_was_applied"].shape == (1397, 1) and int(raw["action_was_applied"].sum()) == 349*4,
            "Require exactly 349 held-action control intervals")
    trace = native_trace(raw, "policy_isaac")
    require(trace["time_s"].shape == (1397,) and
            np.allclose(trace["time_s"], np.arange(1397)*.005, atol=1e-9, rtol=0),
            "Require unchanged complete native 0--6.98 s physical clock")
    require(float(raw["policy_dt_s"]) == .02, "Policy interval differs")
    return trace, {"mu": mu, "checkpoint_sha256": checkpoint_sha256,
        "recording_sha256": digest, "report_sha256": sha256(report_path),
        "evaluator_source_sha256": report["evaluator_source_sha256"],
        "reported_training_manifest_sha256": report.get("training_manifest_sha256"),
        "drift_module_sha256": report["drift_module_sha256"]}


def load_reference(path):
    path = Path(path); digest = sha256(path)
    require(digest in (MOTION_SHA256, REFERENCE_SHA256), "Reference archive is not the locked original pair")
    with np.load(path, allow_pickle=False) as archive:
        time = archive["time_s"].copy()
        if digest == MOTION_SHA256:
            index = tuple(archive["body_names"].tolist()).index("base_Link")
            position = archive["body_pos_w"][:, index].astype(float)
        else:
            position = archive["root_target_position_m"].astype(float)
    require(time.shape == (349,) and np.array_equal(time, np.arange(349)/50.), "Locked reference clock mismatch")
    return {"time_s": time, "position": position, "sha256": digest}


def common_camera_target(reference, time_s):
    """No actual-robot position or orientation can affect this camera target."""
    target = np.array([np.interp(time_s, reference["time_s"], reference["position"][:, j]) for j in range(3)])
    return target+np.array([0., 0., -.24])


def load_data(baseline_root, candidate_root, candidate_training, reference=DEFAULT_REFERENCE,
              model=DEFAULT_MODEL, public_summary=DEFAULT_PUBLIC_SUMMARY):
    """Audit frozen baseline and completed from-scratch candidate, without GL."""
    training = Path(candidate_training)
    status, ablation = read_json(training/"run_status.json"), read_json(training/"ablation_manifest.json")
    require(status["status"] == "training_completed" and status["trained_policy_claimed"] is True,
            "Candidate must be a completed formal training run")
    require(ablation["status"] == "training_completed" and ablation["smoke_only"] is False and
            ablation["initialization"] == "from_scratch_no_checkpoint_resume", "Candidate ablation provenance differs")
    require(status["environment_transitions"] == ablation["actual_environment_transitions"] == 12288000,
            "Candidate formal budget differs")
    require(ablation["final_training_manifest_sha256"] == sha256(training/"manifest.json"), "Candidate training manifest SHA mismatch")
    override = ablation["reward_override"]
    require(override["term"] == "action_rate_l2" and override["baseline_weight"] == -.1 and
            np.isfinite(override["weight"]) and override["weight"] < -.1,
            "Candidate reward override is not the isolated action-rate ablation")
    training_manifest = read_json(training/"manifest.json")
    require(training_manifest["training_reward_overrides"] == {"action_rate_l2": override["weight"]},
            "Candidate training manifest and ablation weights differ")
    candidate_checkpoint = ablation["checkpoint_sha256"]
    checkpoint_bytes_verified = (training/"model_final.pt").is_file()
    if checkpoint_bytes_verified:
        require(sha256(training/"model_final.pt") == candidate_checkpoint, "Candidate checkpoint bytes differ")
    require(sha256(model) == MODEL_SHA256, "Original display-model SHA mismatch")
    public = read_json(public_summary)
    cases, audits = {}, {}
    for mu in MU:
        name = "mu"+f"{mu:.3f}".replace(".", "p")
        for label, directory, checkpoint in (("baseline", baseline_root, BASELINE_CHECKPOINT),
                                              ("candidate", candidate_root, candidate_checkpoint)):
            key = name+"_"+label
            trace, audit = load_case(case_folder(directory, mu), mu=mu, checkpoint_sha256=checkpoint)
            if label == "baseline":
                expected = public["runs"]["PPO_"+name]["input_sha256"]
                require(audit["recording_sha256"] == expected["isaac_trace"] and
                        audit["report_sha256"] == expected["isaac_report"], "Historical baseline recording differs")
            cases[key] = trace; audits[key] = audit
        for field in ("evaluator_source_sha256", "drift_module_sha256"):
            require(audits[name+"_baseline"][field] == audits[name+"_candidate"][field],
                    "Baseline and candidate must use the same frozen evaluator sources")
    clocks = [cases[key]["time_s"] for key in ORDER]
    require(all(np.array_equal(clock, clocks[0]) for clock in clocks), "Four native clocks differ; no alignment permitted")
    reference_arrays = load_reference(reference)
    summary = {"schema_version": 1, "kind": "physx_ppo_action_rate_fullbody_recorded_comparison",
        "training_engine": "PhysX", "evaluation_engine": "PhysX", "graphics_engine": "MuJoCo FK only",
        "baseline_action_rate_weight": -.1, "candidate_action_rate_weight": ablation["reward_override"]["weight"],
        "baseline_checkpoint_sha256": BASELINE_CHECKPOINT, "candidate_checkpoint_sha256": candidate_checkpoint,
        "candidate_checkpoint_bytes_verified_locally": checkpoint_bytes_verified,
        "candidate_ablation_manifest_sha256": sha256(training/"ablation_manifest.json"),
        "candidate_training_status_sha256": sha256(training/"run_status.json"),
        "candidate_training_manifest_sha256": sha256(training/"manifest.json"),
        "reference_sha256": reference_arrays["sha256"], "model_sha256": MODEL_SHA256,
        "cases": audits, "case_order": list(ORDER), "row_mu": list(MU),
        "column_order": ["baseline", "candidate"], "native_samples": 1397,
        "video_fps": 50, "video_frames": 350, "native_sample_stride": 4,
        "first_native_time_s": 0., "last_native_time_s": 6.98, "encoded_duration_s": 7.,
        "camera": {"azimuth_deg": 135., "elevation_deg": -24., "distance_m": 2.15,
            "lookat": "Identical locked reference root position + [0,0,-0.24] for all four panels",
            "follows_actual_robot": False, "follows_actual_orientation": False,
            "reference_after_6p96s": "Hold last reference knot until original physical t=6.98 s"},
        "robot_mesh_alpha": 1., "root_height_corrected": False, "pose_alignment": False,
        "time_alignment": False, "dynamics_stepping": False, "contact_solver_called": False,
        "new_policy_inference": False, "new_training": False,
        "jitter_reduction_claimed_by_renderer": False,
        "limitations": ["Rendering is recorded-state replay, not another simulator validation.",
            "Each panel keeps its original world pose; a robot leaving the common view is not recentered.",
            "Only one checkpoint/episode per configuration and coefficient is shown; no statistical success rate.",
            "Visual comparison alone does not establish the cause or elimination of physical oscillation."]}
    return cases, reference_arrays, summary


def render_frame(runtime, cases, qposes, reference, summary, index):
    from PIL import Image, ImageDraw
    w, h = runtime.width, runtime.height
    row_height = h+100
    image = Image.new("RGB", (2*w, 2*row_height+50), "#fafafa")
    draw = ImageDraw.Draw(image)
    time = float(cases[ORDER[0]]["time_s"][index])
    draw.text((12, 5), f"PhysX PPO recorded comparison | physical t={time:.2f} s | identical reference-following camera", font=runtime.font, fill="black")
    camera = runtime.cameras[0]
    camera.lookat[:] = common_camera_target(reference, time)
    for number, key in enumerate(ORDER):
        row, column = divmod(number, 2)
        trace = cases[key]
        _pose(runtime.model, runtime.data, qposes[key][index])
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        pixels = runtime.renderer.render().copy()
        x, y = column*w, 30+row*row_height
        image.paste(Image.fromarray(pixels), (x, y))
        draw.rectangle((x, y, x+w, y+27), fill="#f5f5f5")
        weight = summary["baseline_action_rate_weight"] if column == 0 else summary["candidate_action_rate_weight"]
        draw.text((x+10, y+3), f"{'Baseline' if column == 0 else 'Candidate'} | action-rate {weight:g} | mu={MU[row]:g}", font=runtime.font, fill="black")
        position = trace["position"][index]
        target = common_camera_target(reference, time)-np.array([0., 0., -.24])
        error = float(np.linalg.norm(position[:2]-target[:2]))
        speed = float(np.linalg.norm(trace["com_velocity"][index, :2]))
        up = quaternion_rotate(trace["quaternion"][index], np.array([0., 0., 1.]))
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1., 1.))))
        draw.text((x+10, y+h+6), f"XY error {error:.3f} m | COM speed {speed:.2f} m/s", font=runtime.font, fill="#111111")
        draw.text((x+10, y+h+31), f"Original height {position[2]:.3f} m | tilt {tilt:.2f} deg", font=runtime.font, fill="#111111")
        counts, fz = trace["contact_count"][index], trace["contact_force"][index, :, 2]
        draw.text((x+10, y+h+58), f"Recorded L/R contact count {int(counts[0])}/{int(counts[1])} | Fz {fz[0]:.0f}/{fz[1]:.0f} N", font=runtime.small, fill="#333333")
    draw.text((12, image.height-21), "Original PhysX states; opaque meshes; MuJoCo graphics only. No pose correction, new physics or inferred contact.", font=runtime.small, fill="#333333")
    return image


def run(args):
    from PIL import Image
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh output directory")
    cases, reference, summary = load_data(args.baseline_root, args.candidate_root, args.candidate_training,
                                         args.reference, args.model, args.public_summary)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes = {}
        for key in ORDER:
            qposes[key], mapping = build_qpos(runtime.model, cases[key])
            summary["cases"][key]["fk_validation"] = validate_fk(runtime.model, cases[key], qposes[key])
            summary["cases"][key]["named_joint_qpos_addresses"] = mapping.tolist()
        output.mkdir(parents=True)
        summary["mesh_provenance"] = mesh_provenance(args.model)
        arrays = {"time_s": cases[ORDER[0]]["time_s"]}
        for key in ORDER:
            arrays[key+"_qpos"] = qposes[key]
            arrays[key+"_original_root_position_m"] = cases[key]["position"]
        np.savez_compressed(output/"replay_states.npz", **arrays)
        index = int(np.argmin(np.abs(cases[ORDER[0]]["time_s"]-2.4)))
        render_frame(runtime, cases, qposes, reference, summary, index).save(output/"comparison.png")
        files = ["replay_states.npz", "comparison.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            first = render_frame(runtime, cases, qposes, reference, summary, 0)
            stream = imageio_ffmpeg.write_frames(str(output/"fullbodycomparison.mp4"), first.size,
                fps=50, codec="libx264", quality=7, macro_block_size=2,
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for index in range(0, 1397, 4):
                    stream.send(np.asarray(render_frame(runtime, cases, qposes, reference, summary, index)))
            finally:
                stream.close()
            files.append("fullbodycomparison.mp4")
        summary["video_generated"] = not args.skip_video
        summary["analysis_source_sha256"] = sha256(__file__)
        summary["renderer_helper_sha256"] = sha256(ROOT/"scripts/render_physx_ppo_fullbody.py")
        summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in files}
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--candidate-root", type=Path, required=True)
    result.add_argument("--candidate-training", type=Path, required=True)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--public-summary", type=Path, default=DEFAULT_PUBLIC_SUMMARY)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps({"status": "rendered_recorded_comparison", "dynamics_stepping": False,
                      "candidate_action_rate_weight": report["candidate_action_rate_weight"]}))

"""Actual CEM-only Original/A/C/D recorded-pose replay; B deliberately omitted.

Strict terminal inputs and their frozen D numerical analysis are verified before
graphics. Reuses frozen A pose/FK/camera helpers, not its PPO identity/footer.
No actors, dynamics, interpolation, alignment, height correction or tuning.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

import analyze_cem_ground_contact_v2d as audit
import render_ppo_ground_contact_v2 as graphics
from controller_trace import native_trace
from ppo_backend_400hz_contract import require, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
GRAPHICS_SHA256 = "0289c7da1764758cb79c184a7fbf9f5f0e9afd4adab23ef1ea700411be997efa"
ANALYZER_SHA256 = "7008d5d9612c295596cd2e263f6bd1099147776d656d627bee6eddcf9f2792c8"
ANALYSIS_SHA256 = "c4d6a40bbba688b45ea7155e7eca92a9f62211074f13e3afe071f1cd05ce51ab"
DEFAULT_ANALYSIS = ROOT/"outputs/ground_contact_v2d_20261023_v1/cem_actual_analysis_v1/summary.json"
DEFAULT_FOLDERS = {"original": audit.base.BASELINE_FOLDER,
    "v2a": ROOT/"outputs/ground_contact_v2_20261023_v1/cem_mu0p7",
    "v2c": ROOT/"outputs/ground_contact_v2c_20261023_v1/cem_mu0p7",
    "v2d": ROOT/"outputs/ground_contact_v2d_20261023_v1/cem_mu0p7"}
ORDER = ("original", "v2a", "v2c", "v2d")
TITLES = {"original": "Original CEM | beta ON", "v2a": "A CEM | .75mean8 + .25min8",
          "v2c": "C CEM | min8 - dense deficit", "v2d": "D CEM | mean8 - dense deficit"}
FRAMES, FPS, HEIGHT, COLORS = graphics.FRAMES, graphics.FPS, graphics.HEIGHT, graphics.COLORS
native_indices, contact_window = graphics.native_indices, graphics.contact_window
heading_glyphs, case_contact_caption, draw_map = graphics.heading_glyphs, graphics.case_contact_caption, graphics.draw_map


def validate_episode(raw, report, reference_sha, model_sha):
    """Common complete native clock/material/budget checks, including original."""
    require(report.get("status") == "completed" and not report.get("smoke_only", False)
            and report.get("truncated_smoke") is False and report.get("completed_policy_steps") == 349
            and report.get("native_samples") == 2793, "Require terminal349-step actual CEM episode, not pending/smoke")
    require(report.get("condition") == "beta_on" and report.get("plant_mu") == .7
            and report.get("prediction_mu_scenarios") == [.7]
            and report.get("prediction_friction_known") is True
            and report.get("motion_sha256") == reference_sha and report.get("model_sha256") == model_sha,
            "Wrong CEM reference/model/beta/known material")
    require(raw["qpos"].shape == (2793, 15) and raw["qvel"].shape == (2793, 14)
            and raw["actions"].shape == (349, 8), "Incomplete native states/actions")
    trace = native_trace(raw, "nmpc")
    require(trace["physics_dt_s"] == .0025 and np.allclose(trace["time_s"], np.arange(2793)*.0025, rtol=0, atol=1e-9),
            "Require unspliced consecutive400Hz clock0-6.98s")
    force_key = "ground_normal_force_w" if "ground_normal_force_w" in raw else "wheel_ground_contact_force_w_n"
    force = np.asarray(raw[force_key], dtype=float)
    require(force.shape == (2793, 2, 3) and np.isfinite(force).all() and np.min(force[..., 2]) >= -1e-4,
            "Require actual finite upward floor-normal force, not net body force")
    trace["contact_force"] = force.copy()
    trace["floor_normal_force"] = force.copy()
    return trace, force_key


def assert_analysis(summary, metrics, hashes, reward_audits, screen):
    require(summary.get("kind") == "CEM_ground_contact_V2d_ACTUAL_nominal_readonly_audit"
            and summary.get("status") == "completed_analysis"
            and summary.get("analyzer_sha256") == ANALYZER_SHA256
            and summary.get("shared_metric_source_sha256") == audit.METRICS_SOURCE_SHA256
            and summary.get("protocol_sha256") == audit.PROTOCOL_SHA256,
            "Require frozen completed ACTUAL D analysis, not pending/PPO/relabelled inputs")
    require(summary["actual_V2d"] == metrics["v2d"] and summary["input_sha256"] == hashes["v2d"]
            and summary["actual_D_reward_audit"] == reward_audits["v2d"] and summary["nominal_screen"] == screen,
            "D actual trace/reward/screen differs from frozen analysis")
    require(summary["original_grounded_baseline"] == metrics["original"]
            and summary["baseline_input_sha256"] == hashes["original"], "Original recorded metrics/bindings differ")
    for key, field in (("v2a", "optional_prior_actual_V2a"), ("v2c", "optional_prior_actual_V2c")):
        old = summary[field]
        require(old is not None and old["metrics"] == metrics[key] and old["input_sha256"] == hashes[key]
                and old["reward_audit"] == reward_audits[key], "Prior ACTUAL "+key+" metric/reward/binding differs")


def load_cases(args, reference):
    require(sha256(graphics.__file__) == GRAPHICS_SHA256, "Frozen A recorded-pose graphics changed")
    require(sha256(audit.__file__) == ANALYZER_SHA256
            and sha256(audit.prior_c.__file__) == audit.V2C_ANALYZER_SHA256, "Frozen CEM analyzers changed")
    require(sha256(args.analysis_summary) == ANALYSIS_SHA256, "Frozen actual CEM analysis summary changed")
    protocol = audit.load_protocol(args.protocol)
    summary = read_json(args.analysis_summary)
    raw_d, report_d, hash_d, reward_d = audit.load_completed(args.v2d_dir, args.protocol)
    raw_a, report_a, hash_a, reward_a = audit.base.load_completed(args.v2a_dir)
    raw_c, report_c, hash_c, reward_c = audit.prior_c.load_completed(args.v2c_dir)
    original_values, original_hash = audit.load_original(reference)
    original_folder = audit.base.BASELINE_FOLDER
    with np.load(original_folder/"rollout.npz", allow_pickle=False) as archive:
        raw_original = {name: archive[name].copy() for name in archive.files}
    report_original = read_json(original_folder/"report.json")
    decisions = json.loads((original_folder/"solver_log.json").read_text())
    require(isinstance(decisions, list) and len(decisions) == 349, "Original CEM solver log must contain349 actual decisions")
    require(all(hash_a[name] == digest for name, digest in audit.V2A_PINS.items())
            and all(hash_c[name] == digest for name, digest in audit.V2C_PINS.items()), "Prior actual A/C SHA differs")
    raws = {"original": raw_original, "v2a": raw_a, "v2c": raw_c, "v2d": raw_d}
    reports = {"original": report_original, "v2a": report_a, "v2c": report_c, "v2d": report_d}
    hashes = {"original": original_hash, "v2a": hash_a, "v2c": hash_c, "v2d": hash_d}
    rewards = {"v2a": reward_a, "v2c": reward_c, "v2d": reward_d}
    traces, cases, evidence, values = {}, {}, {}, {}
    reference_sha, model_sha = sha256(args.reference), sha256(args.model)
    for key in ORDER:
        raw, report = raws[key], reports[key]
        trace, force_key = validate_episode(raw, report, reference_sha, model_sha)
        require(np.array_equal(raw["qpos"][0], raw_d["qpos"][0]) and np.array_equal(raw["qvel"][0], raw_d["qvel"][0]),
                "Observable initial state differs: "+key)
        require(report["CEM_config"] == report_d["CEM_config"] and report["discount_factor"] == .99,
                "Frozen CEM budget/discount differs: "+key)
        if key != "original":
            for field in ("plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
                require(report[field] == report_d[field], "Actual material/normalization differs: "+key+"/"+field)
        values[key] = audit.base.recorded_metrics(raw, reference)
        traces[key] = trace
        evidence[key] = {"input_directory": str((original_folder if key == "original" else getattr(args, key+"_dir")).resolve()),
            "input_sha256": hashes[key], "floor_force_channel": force_key,
            "actual_initial_qpos": raw["qpos"][0].tolist(), "actual_initial_qvel": raw["qvel"][0].tolist(),
            "actual_reward_audit": rewards.get(key), "executed_source_sha256": report.get("source_sha256"),
            "original_solver_log_sha256": sha256(original_folder/"solver_log.json") if key == "original" else None}
    require(values["original"] == original_values, "Original recorded metrics changed during loading")
    screens = {key: audit.base.nominal_screen(values[key], values["original"], protocol) for key in ORDER if key != "original"}
    assert_analysis(summary, values, hashes, rewards, screens["v2d"])
    for key in ORDER:
        cases[key] = {"id": key, "engine": "native CPU MuJoCo", "variant": key, "title": TITLES[key],
            "analysis": values[key], "new_nominal_screen": screens.get(key),
            "baseline_screen_not_applied_to_itself": key == "original", "sample_size": 1}
    binding = {"path": str(Path(args.analysis_summary).resolve()), "sha256": ANALYSIS_SHA256,
        "all_four_metrics_recomputed_and_bound": True, "CEM_config": report_d["CEM_config"],
        "discount_factor": .99, "prediction_horizon_s": .6, "B_excluded_from_display": True}
    return traces, cases, evidence, list(ORDER), binding


def outcome_caption(cases, order):
    pieces = ["Original = historical baseline"]
    for key in order:
        if key == "original": continue
        screen = cases[key]["new_nominal_screen"]
        pieces.append(key.upper()+": "+("PASS numerical screen" if screen["numeric_nominal_episode_pass"]
            else f"FAIL {len(screen['failed_checks'])} numerical checks"))
    return "CEM ONLY | "+" | ".join(pieces)+" | B omitted; overall task completion NOT claimed."


def render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame):
    from PIL import ImageDraw
    canvas = graphics.render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame)
    draw = ImageDraw.Draw(canvas)
    for low, high in ((0, 29), (763, 786), (829, 849)):
        draw.rectangle((0, low, canvas.width, high), fill="#fafafa")
    draw.text((12, 4), f"CEM ONLY Original/A/C/D | native MuJoCo | known mu=.7 | beta ON | native400Hz / replay50Hz | t={int(native_indices()[frame])*.0025:.2f}s | B excluded",
              font=runtime.small, fill="#111111")
    draw.text((12, 765), outcome_caption(cases, order), font=runtime.small, fill="#8f2929")
    draw.text((12, 831), "Frozen offline CEM: horizon30 / blocks10 / candidates64 x3 / seed42 / gamma.99; N=1 each. Native0-6.98s retained; reference ends6.96s. No policy result claimed.",
              font=runtime.small, fill="#333333")
    return canvas


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve CEM replay; fresh output directory required")
    require(os.environ.get("MUJOCO_GL") == "osmesa", "CPU OSMesa required")
    reference = graphics.metrics.reference_data(args.reference)
    traces, cases, evidence, order, binding = load_cases(args, reference)
    sources = {str(Path(__file__).resolve()): sha256(__file__), str(Path(graphics.__file__).resolve()): GRAPHICS_SHA256,
        str(Path(audit.__file__).resolve()): ANALYZER_SHA256, str(Path(audit.base.__file__).resolve()): audit.METRICS_SOURCE_SHA256,
        str(Path(audit.prior_c.__file__).resolve()): audit.V2C_ANALYZER_SHA256}
    kinematics, fk_source = graphics.metrics.load_fk(args.model, args.legacy_evaluator)
    for trace in traces.values():
        reconstruction = kinematics.reconstruct(trace)
        trace.update(signals=reconstruction["signals"], wheel_quaternion_wxyz=reconstruction["wheel_quaternion_wxyz"],
                     wheel_center_velocity_w=reconstruction["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = graphics.metrics.heading_geometry_masks(trace)
    runtime = graphics.create_runtime(args.model, width=640, height=360)
    try:
        qposes, audits = {}, {}
        for key in order:
            qposes[key], mapping = graphics.build_qpos(runtime.model, traces[key])
            audits[key] = {**graphics.replay_fk_audit(runtime.model, traces[key], qposes[key]),
                           "named_joint_qpos_addresses": mapping.tolist()}
        plan = graphics.prepare_camera(runtime, traces, qposes, order, native_indices())
        bounds = [value.tolist() for value in graphics.world_bounds(traces, reference)]
        output.mkdir(parents=True, exist_ok=False)
        replay = {"native_indices": native_indices(), "fixed_camera_distance_m": np.asarray(plan["distance_m"])}
        for key in order:
            replay.update({key+"__qpos": qposes[key], key+"__native_time_s": traces[key]["time_s"],
                key+"__lookat": plan["lookat_by_case"][key], key+"__floor_normal_force_w": traces[key]["contact_force"]})
        np.savez_compressed(output/"replay_states.npz", **replay)
        render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, args.poster_frame).save(output/"comparison.png")
        files = ["comparison.png", "replay_states.npz"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output/"fullbody_comparison.mp4"), (2560, HEIGHT),
                fps=FPS, codec="libx264", quality=7, macro_block_size=2, output_params=["-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(FRAMES):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame)))
                    if frame%100 == 0: print(f"CEM Original/A/C/D recorded-only OSMesa frame{frame}/{FRAMES}", flush=True)
            finally: stream.close()
            files.append("fullbody_comparison.mp4")
        require(all(sha256(path) == digest for path, digest in sources.items())
                and sha256(args.analysis_summary) == ANALYSIS_SHA256, "Frozen source/analysis changed while replaying")
        for current in evidence.values():
            folder = Path(current["input_directory"])
            require(all(sha256(folder/name) == digest for name, digest in current["input_sha256"].items()), "Actual input changed while replaying")
        result = {"kind": "CEM_Original_A_C_D_actual_case_caption_recorded_replay", "case_order": order, "cases": cases,
            "B_excluded_from_display": True, "B_is_present_in_separate_numeric_analysis": True, "CEM_only_no_PPO_result": True,
            "engine": "native CPU MuJoCo", "mu": .7, "prediction_mu_known": True, "nonzero_beta_intended": True,
            "input_evidence": evidence, "frozen_actual_analysis": binding, "source_sha256": sources,
            "native_states_per_case": 2793, "native_dt_s": .0025, "selected_stride": 8,
            "video_frames": FRAMES, "video_fps": FPS, "video_generated": not args.skip_video,
            "video_size_pixels": [2560, HEIGHT], "poster_frame": args.poster_frame,
            "poster_native_time_s": float(native_indices()[args.poster_frame]*.0025),
            "camera": {key: value for key, value in plan.items() if key != "lookat_by_case"},
            "camera_same_distance_all_cases_frames": True, "camera_yaw_follow": False, "camera_zoom": False,
            "world_bounds": bounds, "shared_world_bounds_all_panels": True, "FK_source": fk_source, "replay_FK_audit": audits,
            "mesh_provenance": graphics.mesh_provenance(args.model), "model_sha256": sha256(args.model),
            "reference_sha256": sha256(args.reference), "protocol_sha256": sha256(args.protocol),
            "renderer_sha256": sha256(__file__), "frozen_A_graphics_sha256": GRAPHICS_SHA256,
            "A_B_C_D_sources_or_results_changed": False,
            "contact_caption_source": "Each case actual recomputed full_contact0.02-6.96s metrics; not core or hardcoded historical footer",
            "contact_gate": "any wheel actual upward floor-normal Fz>5N; NO point-count or net-body gate",
            "contact_dot_native_samples_per_video_frame": 8, "single_wheel_support_allowed": True, "robot_mesh_alpha": 1.,
            "visual_reviewed": False, "goal_completion_claimed": False, "new_physics_policy_inference_training": False,
            "smoothing_interpolation_alignment_height_correction": False,
            "artifacts": {name: sha256(output/name) for name in files},
            "limitations": ["N=1 each; all panels are actual CEM episodes, not PPO or counterfactual reward trajectories.",
                "Different contact reward definitions: episode returns are not compared or ranked.",
                "Frozen offline CEM budget is not a real-time performance claim.",
                "B is deliberately omitted from four-panel graphics, not from the separate numerical analysis.",
                "All recorded heights/poses retained; force absence is not a geometric flight label.",
                "Reference ends6.96s; final6.98s pose is retained without extrapolated XY error."]}
        write_json(output/"provenance.json", result)
        return result
    finally: runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ORDER[1:]: result.add_argument("--"+key+"-dir", type=Path, default=DEFAULT_FOLDERS[key])
    result.add_argument("--analysis-summary", type=Path, default=DEFAULT_ANALYSIS)
    result.add_argument("--protocol", type=Path, default=audit.DEFAULT_PROTOCOL)
    result.add_argument("--reference", type=Path, default=audit.runner.paired.DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=graphics.assets.paired.DEFAULT_MODEL)
    result.add_argument("--legacy-evaluator", type=Path, default=graphics.assets.paired.DEFAULT_LEGACY)
    result.add_argument("--poster-frame", type=int, choices=range(FRAMES), default=348)
    result.add_argument("--skip-video", action="store_true")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__": run(parser().parse_args())

"""Actual CEM-only Original/A/G/H recorded-pose replay; B/C/D/E/F deliberately omitted.

Strict terminal inputs and their frozen H numerical analysis are verified before
graphics. Reuses frozen A pose/FK/camera helpers, not its PPO identity/footer.
No actors, dynamics, interpolation, alignment, height correction or tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

import analyze_cem_ground_contact_v2h as audit
import render_cem_ground_contact_v2d as frozen_d
import render_ppo_ground_contact_v2 as graphics
from ppo_backend_400hz_contract import require, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
GRAPHICS_SHA256 = "0289c7da1764758cb79c184a7fbf9f5f0e9afd4adab23ef1ea700411be997efa"
ANALYZER_SHA256 = "843c546a2545e7ed66b07bd57fa6326903037fd57fe77d026922def62ae573db"
D_RENDERER_SHA256 = "7c65f7c7f0f6dc0b06e095d18dbd6f1e96baa1dc5007f1349fd2bb1978ac5878"
DEFAULT_ANALYSIS = ROOT/"outputs/ground_contact_v2h_20261024_v1/cem_actual_analysis_v1/summary.json"
DEFAULT_FOLDERS = {"original": audit.base.BASELINE_FOLDER,
    "v2a": ROOT/"outputs/ground_contact_v2_20261023_v1/cem_mu0p7",
    "v2b": ROOT/"outputs/ground_contact_v2b_20261023_v1/cem_mu0p7",
    "v2c": ROOT/"outputs/ground_contact_v2c_20261023_v1/cem_mu0p7",
    "v2f": ROOT/"outputs/ground_contact_v2f_20261023_v1/cem_mu0p7",
    "v2g": ROOT/"outputs/ground_contact_v2g_20261023_v1/cem_mu0p7",
    "v2d": ROOT/"outputs/ground_contact_v2d_20261023_v1/cem_mu0p7",
    "v2e": ROOT/"outputs/ground_contact_v2e_20261023_v1/cem_mu0p7",
    "v2h": ROOT/"outputs/ground_contact_v2h_20261024_v1/cem_mu0p7"}
ORDER = ("original", "v2a", "v2g", "v2h")
TITLES = {"original": "Original CEM | beta ON", "v2a": "A CEM | .75mean8 + .25min8",
          "v2g": "G CEM | beta-gated bonus - gaps", "v2h": "H CEM | G support/gaps - loaded COM-vZ"}
FRAMES, FPS, HEIGHT, COLORS = graphics.FRAMES, graphics.FPS, graphics.HEIGHT, graphics.COLORS
native_indices, contact_window = graphics.native_indices, graphics.contact_window
heading_glyphs, case_contact_caption, draw_map = graphics.heading_glyphs, graphics.case_contact_caption, graphics.draw_map
validate_episode = frozen_d.validate_episode


def analysis_digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Explicit --analysis-sha256 required for the actual completed H summary")
    return value


def load_original_recording(reference):
    metrics, hashes = audit.load_original(reference)
    folder = audit.base.BASELINE_FOLDER
    with np.load(folder/"rollout.npz", allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    report = read_json(folder/"report.json")
    decisions = json.loads((folder/"solver_log.json").read_text())
    require(isinstance(decisions, list) and len(decisions) == 349, "Original CEM solver log must contain349 actual decisions")
    return raw, report, hashes, metrics, sha256(folder/"solver_log.json")


def assert_analysis(summary, metrics, hashes, reward_audits, screen):
    require(summary.get("kind") == "CEM_ground_contact_V2h_ACTUAL_nominal_readonly_audit"
            and summary.get("status") == "completed_analysis"
            and summary.get("analyzer_sha256") == ANALYZER_SHA256
            and summary.get("shared_metric_source_sha256") == audit.METRICS_SOURCE_SHA256
            and summary.get("protocol_sha256") == audit.PROTOCOL_SHA256
            and summary.get("executed_source_sha256") == audit.SOURCE_PINS,
            "Require frozen completed ACTUAL H analysis, not pending/PPO/relabelled inputs")
    require(summary["actual_V2h"] == metrics["v2h"] and summary["input_sha256"] == hashes["v2h"]
            and summary["actual_H_reward_audit"] == reward_audits["v2h"] and summary["nominal_screen"] == screen,
            "H actual trace/reward/screen differs from frozen analysis")
    require(summary["original_grounded_baseline"] == metrics["original"]
            and summary["baseline_input_sha256"] == hashes["original"], "Original recorded metrics/bindings differ")
    for key, field in (("v2a", "optional_prior_actual_V2a"), ("v2g", "optional_prior_actual_V2g")):
        old = summary[field]
        require(old is not None and old["metrics"] == metrics[key] and old["input_sha256"] == hashes[key]
                and old["reward_audit"] == reward_audits[key], "Prior ACTUAL "+key+" metric/reward/binding differs")


def load_cases(args, reference):
    require(sha256(frozen_d.__file__) == D_RENDERER_SHA256, "Frozen D CEM rendering foundation changed")
    require(sha256(graphics.__file__) == GRAPHICS_SHA256, "Frozen A recorded-pose graphics changed")
    require(sha256(audit.__file__) == ANALYZER_SHA256
            and sha256(audit.prior_g.__file__) == audit.V2G_ANALYZER_SHA256, "Frozen CEM analyzers changed")
    summary_sha = analysis_digest(getattr(args, "analysis_sha256", None))
    require(sha256(args.analysis_summary) == summary_sha, "SHA-bound actual CEM analysis summary changed")
    protocol = audit.load_protocol(args.protocol)
    summary = read_json(args.analysis_summary)
    raw_e, report_e, hash_e, reward_e = audit.load_completed(args.v2h_dir, args.protocol)
    raw_a, report_a, hash_a, reward_a = audit.base.load_completed(args.v2a_dir)
    raw_c, report_c, hash_c, reward_c = audit.prior_g.load_completed(args.v2g_dir)
    raw_original, report_original, original_hash, original_values, original_solver_sha = load_original_recording(reference)
    original_folder = audit.base.BASELINE_FOLDER
    require(all(hash_a[name] == digest for name, digest in audit.V2A_PINS.items())
            and all(hash_c[name] == digest for name, digest in audit.V2G_PINS.items()), "Prior actual A/C SHA differs")
    raws = {"original": raw_original, "v2a": raw_a, "v2g": raw_c, "v2h": raw_e}
    reports = {"original": report_original, "v2a": report_a, "v2g": report_c, "v2h": report_e}
    hashes = {"original": original_hash, "v2a": hash_a, "v2g": hash_c, "v2h": hash_e}
    rewards = {"v2a": reward_a, "v2g": reward_c, "v2h": reward_e}
    traces, cases, evidence, values = {}, {}, {}, {}
    reference_sha, model_sha = sha256(args.reference), sha256(args.model)
    for key in ORDER:
        raw, report = raws[key], reports[key]
        trace, force_key = validate_episode(raw, report, reference_sha, model_sha)
        require(np.array_equal(raw["qpos"][0], raw_e["qpos"][0]) and np.array_equal(raw["qvel"][0], raw_e["qvel"][0]),
                "Observable initial state differs: "+key)
        require(report["CEM_config"] == report_e["CEM_config"] and report["discount_factor"] == .99,
                "Frozen CEM budget/discount differs: "+key)
        if key != "original":
            for field in ("plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
                require(report[field] == report_e[field], "Actual material/normalization differs: "+key+"/"+field)
        values[key] = audit.base.recorded_metrics(raw, reference)
        traces[key] = trace
        evidence[key] = {"input_directory": str((original_folder if key == "original" else getattr(args, key+"_dir")).resolve()),
            "input_sha256": hashes[key], "floor_force_channel": force_key,
            "actual_initial_qpos": raw["qpos"][0].tolist(), "actual_initial_qvel": raw["qvel"][0].tolist(),
            "actual_reward_audit": rewards.get(key), "executed_source_sha256": report.get("source_sha256"),
            "original_solver_log_sha256": original_solver_sha if key == "original" else None}
    require(values["original"] == original_values, "Original recorded metrics changed during loading")
    screens = {key: audit.base.nominal_screen(values[key], values["original"], protocol) for key in ORDER if key != "original"}
    assert_analysis(summary, values, hashes, rewards, screens["v2h"])
    for key in ORDER:
        cases[key] = {"id": key, "engine": "native CPU MuJoCo", "variant": key, "title": TITLES[key],
            "analysis": values[key], "new_nominal_screen": screens.get(key),
            "baseline_screen_not_applied_to_itself": key == "original", "sample_size": 1}
    numeric = {key: {"metrics": values[key], "nominal_screen": screens.get(key),
        "input_sha256": hashes[key], "reward_audit": rewards.get(key), "displayed": True} for key in ORDER}
    for key, field, loader in (("v2b", "optional_prior_actual_V2b", audit.load_v2b),
                               ("v2c", "optional_prior_actual_V2c", audit.load_v2c),
                               ("v2d", "optional_prior_actual_V2d", audit.load_v2d),
                               ("v2e", "optional_prior_actual_V2e", audit.load_v2e),
                               ("v2f", "optional_prior_actual_V2f", audit.load_v2f)):
        if summary.get(field) is None: continue
        record = loader(getattr(args, key+"_dir"), reference, raw_e, report_e)
        require(record == summary[field], "Numeric-only actual "+key+" differs from SHA-bound H analysis")
        numeric[key] = {"metrics": record["metrics"], "input_sha256": record["input_sha256"],
            "reward_audit": record["reward_audit"], "displayed": False,
            "nominal_screen": audit.base.nominal_screen(record["metrics"], values["original"], protocol)}
    numeric_order = [key for key in ("original", "v2a", "v2b", "v2c", "v2d", "v2e", "v2f", "v2g", "v2h") if key in numeric]
    binding = {"path": str(Path(args.analysis_summary).resolve()), "sha256": summary_sha,
        "all_four_metrics_recomputed_and_bound": True, "CEM_config": report_e["CEM_config"],
        "discount_factor": .99, "prediction_horizon_s": .6, "B_C_D_E_F_excluded_from_display": True,
        "numeric_case_order": numeric_order, "numeric_cases": numeric,
        "numeric_optional_cases_strict_loaded_not_copied_unchecked": True,
        "episode_returns_compared_or_ranked": False}
    return traces, cases, evidence, list(ORDER), binding


def outcome_caption(cases, order):
    pieces = ["Original = historical baseline"]
    for key in order:
        if key == "original": continue
        screen = cases[key]["new_nominal_screen"]
        pieces.append(key.upper()+": "+("PASS numerical screen" if screen["numeric_nominal_episode_pass"]
            else f"FAIL {len(screen['failed_checks'])} numerical checks"))
    return "CEM ONLY | "+" | ".join(pieces)+" | B/C/D/E/F omitted; overall task completion NOT claimed."


def render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame):
    from PIL import ImageDraw
    canvas = graphics.render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame)
    draw = ImageDraw.Draw(canvas)
    for low, high in ((0, 29), (763, 786), (829, 849)):
        draw.rectangle((0, low, canvas.width, high), fill="#fafafa")
    draw.text((12, 4), f"CEM ONLY Original/A/G/H | native MuJoCo | known mu=.7 | beta ON | native400Hz / replay50Hz | t={int(native_indices()[frame])*.0025:.2f}s | B/C/D/E/F excluded",
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
        str(Path(frozen_d.__file__).resolve()): D_RENDERER_SHA256,
        str(Path(audit.__file__).resolve()): ANALYZER_SHA256, str(Path(audit.base.__file__).resolve()): audit.METRICS_SOURCE_SHA256,
        str(Path(audit.prior_g.__file__).resolve()): audit.V2G_ANALYZER_SHA256}
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
        write_json(output/"numeric_comparison.json", {"kind": "CEM_actual_numeric_cases_not_episode_return_ranking",
            "case_order": binding["numeric_case_order"], "cases": binding["numeric_cases"],
            "analysis_summary_sha256": binding["sha256"], "optional_cases_strictly_verified": True,
            "graphics_case_order": order, "no_PPO_result_claimed": True, "overall_task_completion_claimed": False})
        files = ["comparison.png", "replay_states.npz", "numeric_comparison.json"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output/"fullbody_comparison.mp4"), (2560, HEIGHT),
                fps=FPS, codec="libx264", quality=7, macro_block_size=2, output_params=["-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(FRAMES):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame)))
                    if frame%100 == 0: print(f"CEM Original/A/G/H recorded-only OSMesa frame{frame}/{FRAMES}", flush=True)
            finally: stream.close()
            files.append("fullbody_comparison.mp4")
        require(all(sha256(path) == digest for path, digest in sources.items())
                and sha256(args.analysis_summary) == binding["sha256"], "Frozen source/analysis changed while replaying")
        for current in evidence.values():
            folder = Path(current["input_directory"])
            require(all(sha256(folder/name) == digest for name, digest in current["input_sha256"].items()), "Actual input changed while replaying")
        for key, current in binding["numeric_cases"].items():
            if current["displayed"]: continue
            folder = Path(getattr(args, key+"_dir"))
            require(all(sha256(folder/name) == digest for name, digest in current["input_sha256"].items()), "Numeric-only input changed while replaying")
        result = {"kind": "CEM_Original_A_G_H_loaded_COM_vz_actual_case_caption_recorded_replay", "case_order": order, "cases": cases,
            "B_C_D_E_F_excluded_from_display": True, "numeric_case_order": binding["numeric_case_order"],
            "CEM_only_no_PPO_result": True, "H_support_gap_weight": 1., "H_support_force_threshold_n": 5.,
            "H_positive_bonus": "G support/gaps retained; subtract bounded loaded COM-vZ cost .5*mean8 with .3m/s deadband/scale; no absolute-height or bilateral target", "external_grounded_weight": .4,
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
            "frozen_D_CEM_renderer_sha256": D_RENDERER_SHA256, "A_B_C_D_E_F_sources_or_results_changed": False,
            "contact_caption_source": "Each case actual recomputed full_contact0.02-6.96s metrics; not core or hardcoded historical footer",
            "contact_gate": "any wheel actual upward floor-normal Fz>5N; NO point-count or net-body gate",
            "contact_dot_native_samples_per_video_frame": 8, "single_wheel_support_allowed": True, "robot_mesh_alpha": 1.,
            "visual_reviewed": False, "goal_completion_claimed": False, "new_physics_policy_inference_training": False,
            "smoothing_interpolation_alignment_height_correction": False,
            "artifacts": {name: sha256(output/name) for name in files},
            "limitations": ["N=1 each; all panels are actual CEM episodes, not PPO or counterfactual reward trajectories.",
                "Different contact reward definitions: episode returns are not compared or ranked.",
                "Frozen offline CEM budget is not a real-time performance claim.",
                "B/C/D/E/F are omitted from four-panel graphics; their numeric rows are included only when provided and independently verified.",
                "No H result may be rendered until strict actual completed349-step inputs and their explicit summary SHA pass verification.",
                "All recorded heights/poses retained; force absence is not a geometric flight label.",
                "Reference ends6.96s; final6.98s pose is retained without extrapolated XY error."]}
        write_json(output/"provenance.json", result)
        return result
    finally: runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("v2a", "v2b", "v2c", "v2d", "v2e", "v2f", "v2g", "v2h"):
        result.add_argument("--"+key+"-dir", type=Path, default=DEFAULT_FOLDERS[key])
    result.add_argument("--analysis-summary", type=Path, default=DEFAULT_ANALYSIS)
    result.add_argument("--analysis-sha256", type=analysis_digest, required=True,
        help="Explicit digest of the actual completed H analyzer summary; no invented or pending-result pin")
    result.add_argument("--protocol", type=Path, default=audit.DEFAULT_PROTOCOL)
    result.add_argument("--reference", type=Path, default=audit.runner.paired.DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=graphics.assets.paired.DEFAULT_MODEL)
    result.add_argument("--legacy-evaluator", type=Path, default=graphics.assets.paired.DEFAULT_LEGACY)
    result.add_argument("--poster-frame", type=int, choices=range(FRAMES), default=348)
    result.add_argument("--skip-video", action="store_true")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__": run(parser().parse_args())




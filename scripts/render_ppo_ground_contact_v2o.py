"""Proof-gated O(K plus pre-load-conditioned stance-command cost) recorded replay; no speculative O results.

Four horizontal panels: original/O PhysX, original/O native CPU MuJoCo.
Requires an explicit completed O training audit bound to both evaluation proofs.
Frozen A graphics supply exact native poses/FK, translation-only camera, opaque
meshes, force-only contact dots and headings; O identity captions differ.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

import analyze_ppo_ground_contact_v2o as audit
import render_ppo_ground_contact_v2 as graphics
from ppo_backend_400hz_contract import require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
GRAPHICS_SHA256 = "0289c7da1764758cb79c184a7fbf9f5f0e9afd4adab23ef1ea700411be997efa"
ANALYZER_SHA256 = "2dc3f6deafcad507506a57ae431c8e59b998bf5a458deb635941f0077d2c7501"
FRAMES, FPS, HEIGHT, COLORS = graphics.FRAMES, graphics.FPS, graphics.HEIGHT, graphics.COLORS
native_indices = graphics.native_indices
contact_window = graphics.contact_window
heading_glyphs = graphics.heading_glyphs
case_contact_caption = graphics.case_contact_caption
draw_map = graphics.draw_map


def resolve_directories(args):
    result = copy.copy(args)
    if not any(getattr(result, engine+"_dir", None) is not None for engine in audit.BACKENDS):
        root = Path(getattr(result, "input_root", None) or audit.DEFAULT_INPUT_ROOT)
        for engine in audit.BACKENDS: setattr(result, engine+"_dir", root/engine)
    return result


def selected_engines(args):
    return graphics.selected_engines(resolve_directories(args))


def load_cases(args, reference):
    require(sha256(graphics.__file__) == GRAPHICS_SHA256, "Frozen A recorded-pose graphics changed")
    require(sha256(audit.__file__) == ANALYZER_SHA256, "Frozen O completed-proof analyzer changed")
    require(getattr(args, "training_audit", None) is not None, "Explicit --training-audit required for O rendering")
    args = resolve_directories(args)
    engines = selected_engines(args)
    protocol = audit.load_protocol(args.protocol)
    trained, training = audit.load_training_audit(args.training_audit, args.protocol)
    traces, cases, evidence = {}, {}, {}
    for engine in engines:
        raw, proof = audit.load_completed(getattr(args, engine+"_dir"), engine, args.protocol,
                                         training_audit=args.training_audit)
        require(proof["formal_training_source_evidence"]["sha256"] == training["sha256"]
                and proof["actor_sha256"] == trained["exported_policy_sha256"]
                and proof["checkpoint_sha256"] == trained["checkpoint_sha256"],
                "O render inputs are not bound to explicit completed training audit")
        old_raw, old_proof = audit.load_baseline(engine, args.baseline_root, args.baseline_summary)
        audit._compare_profiles(proof["contract"], old_proof["contract"])
        values, trace = audit.recorded_metrics(raw, engine, reference)
        old_values, old_trace = audit.recorded_metrics(old_raw, engine, reference)
        screen = audit.nominal_screen(values, old_values, protocol, engine)
        for variant, current, analysis, current_proof in (("baseline", old_trace, old_values, old_proof),
                                                        ("v2o", trace, values, proof)):
            key = variant+"_"+engine
            current["contact_force"] = np.asarray(current["floor_normal_force"]).copy()
            traces[key], evidence[key] = current, current_proof
            cases[key] = {"id": key, "engine": engine, "variant": variant,
                "title": ("Original PPO" if variant == "baseline" else "O contact-V2o K margin + stance-command cost PPO")
                    +" | "+audit.BACKENDS[engine]+" | beta ON",
                "analysis": analysis, "new_nominal_screen": screen if variant == "v2o" else None,
                "O_cohort_id": audit.evaluator.COHORT_ID if variant == "v2o" else None,
                "A_B_C_D_E_F_reward_policy_substituted": False}
    if len(engines) == 2:
        first, second = evidence["v2o_physx"], evidence["v2o_mujoco"]
        require(first["actor_sha256"] == second["actor_sha256"]
                and first["checkpoint_sha256"] == second["checkpoint_sha256"],
                "Both engines must evaluate the SAME O trained normalized actor")
    order = [variant+"_"+engine for engine in engines for variant in ("baseline", "v2o")]
    return traces, cases, evidence, order, training


def outcome_caption(cases, order):
    pieces = []
    for key in order:
        case = cases[key]
        if case["variant"] != "v2o": continue
        screen = case["new_nominal_screen"]
        verdict = "PASS numerical screen" if screen["numeric_nominal_episode_pass"] else f"FAIL {len(screen['failed_checks'])} numerical checks"
        pieces.append(case["engine"]+": "+verdict)
    missing = [engine for engine in audit.BACKENDS if "v2o_"+engine not in cases]
    if missing: pieces.append("NOT DISPLAYED: "+"/".join(missing))
    return "O PPO(K margin + stance-command cost) | "+" | ".join(pieces)+" | Overall task completion NOT claimed; qualitative review separate."


def render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame):
    """Frozen A body/map/contact rendering, with explicitly O header/footer."""
    from PIL import ImageDraw
    canvas = graphics.render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 29), fill="#fafafa")
    subset = "EXPLICIT TWO-PANEL SUBSET" if len(order) == 2 else "FOUR ACTUAL COMPLETED CASES"
    draw.text((12, 4), f"PPO original vs O contactV2o(K margin + stance-command cost) | mu=.7 | native400Hz / replay50Hz | t={int(native_indices()[frame])*.0025:.2f}s | {subset}",
              font=runtime.small, fill="#111111")
    draw.rectangle((0, 763, canvas.width, 786), fill="#fafafa")
    draw.text((12, 765), outcome_caption(cases, order), font=runtime.small, fill="#8f2929")
    return canvas


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve O render; fresh output directory required")
    args = resolve_directories(args)
    selected_engines(args)
    require(os.environ.get("MUJOCO_GL") == "osmesa", "CPU OSMesa required")
    reference = graphics.metrics.reference_data(args.reference)
    traces, cases, evidence, order, training = load_cases(args, reference)
    source_hashes = {str(Path(__file__).resolve()): sha256(__file__),
        str(Path(graphics.__file__).resolve()): GRAPHICS_SHA256,
        str(Path(audit.__file__).resolve()): sha256(audit.__file__)}
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
            stream = imageio_ffmpeg.write_frames(str(output/"fullbody_comparison.mp4"), (640*len(order), HEIGHT),
                fps=FPS, codec="libx264", quality=7, macro_block_size=2, output_params=["-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(FRAMES):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, cases, order, plan, bounds, frame)))
                    if frame%100 == 0: print(f"O PPO recorded-only OSMesa frame{frame}/{FRAMES}", flush=True)
            finally: stream.close()
            files.append("fullbody_comparison.mp4")
        require(all(sha256(name) == digest for name, digest in source_hashes.items())
                and sha256(args.training_audit) == training["sha256"], "O source/training proof changed while replaying")
        result = {"kind": "PPO_contact_V2o_stance_command_actual_case_caption_recorded_replay", "cohort_id": audit.evaluator.COHORT_ID,
            "reward_definition_version": audit.evaluator.REWARD_VERSION,
            "support_aggregation": "K physical load-margin cost and reference-beta support bonus retained; bounded minimum previously bearing leg clipped-target change cost, sigma0.1rad. Other leg free; prior force is before the command; no root velocity or height input",
            "support_gap_weight": 1., "stance_command_delta_std_rad": .1, "stance_command_penalty_weight": 1.,
            "support_force_threshold_n": 5., "external_grounded_weight": .4,
            "case_order": order, "cases": cases, "input_evidence": evidence,
            "explicit_training_audit": training, "training_audit_required_and_bound": True,
            "explicit_subset": args.subset, "full_four_panel_comparison": len(order) == 4,
            "native_states_per_case": 2793, "native_dt_s": .0025, "selected_stride": 8,
            "video_frames": FRAMES, "video_fps": FPS, "video_generated": not args.skip_video,
            "video_size_pixels": [640*len(order), HEIGHT],
            "camera": {key: value for key, value in plan.items() if key != "lookat_by_case"},
            "camera_same_distance_all_cases_frames": True, "camera_yaw_follow": False, "camera_zoom": False,
            "world_bounds": bounds, "FK_source": fk_source, "replay_FK_audit": audits,
            "mesh_provenance": graphics.mesh_provenance(args.model), "model_sha256": sha256(args.model),
            "reference_sha256": sha256(args.reference), "protocol_sha256": sha256(args.protocol),
            "renderer_sha256": sha256(__file__), "source_sha256": source_hashes,
            "frozen_A_graphics_sha256": GRAPHICS_SHA256, "A_B_C_D_E_F_formal_results_changed_or_substituted": False,
            "contact_caption_source": "actual F/baseline per-case analyzer metrics, never A/CEM outcome constants",
            "contact_gate": "any wheel upward actual floor-normal Fz>5N, no pointcount gate",
            "contact_dot_native_samples_per_video_frame": 8, "nonzero_beta_intended": True,
            "single_wheel_support_allowed": True, "robot_mesh_alpha": 1., "visual_reviewed": False,
            "goal_completion_claimed": False, "new_physics_policy_inference_training": False,
            "smoothing_interpolation_alignment_height_correction": False,
            "artifacts": {name: sha256(output/name) for name in files},
            "limitations": ["No O result exists until actual strict completed training/episode proofs pass this loader.",
                "N=1 per engine; numerical protocol screen does not establish broad task completion.",
                "A/B/C/D bundles remain separate; old original PPO is the pinned historical baseline, not either contact-policy actor.",
                "All original poses/heights retained. No dynamics, actors, smoothing, interpolation or state correction.",
                "Contact uses actual normal force, not contact count or geometric flight classification.",
                "Reference ends6.96s; final6.98s XY error unavailable without extrapolation."]}
        write_json(output/"provenance.json", result)
        return result
    finally: runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--physx-dir", type=Path)
    result.add_argument("--mujoco-dir", type=Path)
    result.add_argument("--input-root", type=Path, default=audit.DEFAULT_INPUT_ROOT)
    result.add_argument("--training-audit", type=Path, required=True,
                        help="Actual copied completed O formal training audit; not A/B/C/D or smoke metadata")
    result.add_argument("--subset", choices=("physx", "mujoco"))
    result.add_argument("--protocol", type=Path, default=audit.DEFAULT_PROTOCOL)
    result.add_argument("--reference", type=Path, default=audit.DEFAULT_REFERENCE)
    result.add_argument("--baseline-root", type=Path, default=audit.BASELINE_ROOT)
    result.add_argument("--baseline-summary", type=Path, default=audit.BASELINE_SUMMARY)
    result.add_argument("--model", type=Path, default=graphics.assets.paired.DEFAULT_MODEL)
    result.add_argument("--legacy-evaluator", type=Path, default=graphics.assets.paired.DEFAULT_LEGACY)
    result.add_argument("--poster-frame", type=int, choices=range(FRAMES), default=150)
    result.add_argument("--skip-video", action="store_true")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__": run(parser().parse_args())

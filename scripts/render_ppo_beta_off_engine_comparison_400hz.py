"""One completed PhysX-trained400Hz beta-OFF actor in two validation engines.

This is a PARTIAL backend comparison: no MJWarp-trained beta-OFF policy is
substituted or implied. Original full native poses are replayed, never stepped.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path

import numpy as np

import render_ppo_beta_physx_to_mujoco_400hz as previous
from controller_metrics import heading_signals
from controller_trace import native_trace
from ppo_backend_400hz_contract import INPUT_PINS, read_json, require, sha256, verify_recording, write_json
from render_physx_ppo_fullbody import build_qpos, create_runtime, mesh_provenance, _pose
from render_ppo_backend_400hz_nominal import diagnostics, replay_fk_audit
from render_ppo_beta_nominal import world_bounds
from render_ppo_fixed_follow import prepare_camera, apply_camera
from render_ppo_training_recipe_comparison import _map_panel
from summarize_ppo_backend_400hz_nominal import local_file, verify_files

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHYSX = ROOT/"outputs/ppo_beta_off_engine_comparison_400hz_20261022_v1"
ORDER = ("physx", "mujoco")
LABELS = {"physx": "PhysX", "mujoco": "native CPU MuJoCo"}
CANVAS_SIZE = (1280, 750)
KIND = "verified_beta_off_physx_trained_two_engine_partial_replay_inputs"
PHYSX_PINS = {"recording_sha256": "9bc8b9e689f05d7109151ec27e70e869c82f1c24dda3e7dfdbdcf91364637c4b",
    "report_sha256": "acdb578a8b2c52cd6b8700248f03194e8fe0bc1814b4ae39e146821cbb901471"}


def validate_physx(root, manifest, ledger, metadata):
    cases = manifest["cases"]
    require(len(cases) == 1, "Require exactly one new completed PhysX validation cell")
    case = cases[0]; folder = Path(case["directory"])
    require(case.get("case_id") == "physx_to_physx_beta_off" and case.get("training_backend") == "physx" and case.get("engine_tag") == "physx"
            and case.get("validation_backend") == "PhysX" and case.get("mu") == .7
            and case.get("checkpoint_sha256") == previous.OFF["checkpoint"]
            and case.get("actor_sha256") == previous.OFF["actor"]
            and case.get("raw_file") == "physics_trajectory.npz" and case.get("report_file") == "report.json",
            "New cell is not the SAME completed400Hz beta-OFF PhysX-trained actor")
    recording = local_file(root, str(folder/"physics_trajectory.npz"))
    report_file = local_file(root, str(folder/"report.json"))
    for path, key in ((recording, "recording_sha256"), (report_file, "report_sha256")):
        require(sha256(path) == case[key] == ledger[str(path.relative_to(root))] == PHYSX_PINS[key],
                "Pinned PhysX report/recording SHA binding differs")
    proof = read_json(local_file(root, str(folder/"remote_verification.json")))
    launch = read_json(local_file(root, str(folder/"launch_status.json")))
    require(case["verification"] == proof == launch["verification"] and launch["status"] == "completed_verified"
            and launch["child_returncode"] == 0, "PhysX completed launch/remote proof differs")
    expected_proof = {"status": "verified", "native_states": 2793, "policy_decisions": 349,
        "duration_s": 6.98, "physics_dt_s": .0025, "policy_dt_s": .02,
        "recording_sha256": sha256(recording), "report_sha256": sha256(report_file),
        "validation_backend": "PhysX", "training_backend": "physx", "smoke_only": False,
        "checkpoint_sha256": previous.OFF["checkpoint"], "evaluation_contact_friction": .7}
    for name, value in expected_proof.items():
        require(proof.get(name) == value, "PhysX remote proof differs: "+name)
    for name, value in verify_recording(recording, "PhysX").items():
        require(proof.get(name) == value, "Local native PhysX verification differs: "+name)
    report = read_json(report_file); audit = report["backend_400hz_cohort_evaluation"]
    expected_audit = {"status": "verified", "training_backend": "physx", "validation_backend": "PhysX",
        "smoke_only": False, "trained_policy_claimed": True, "run_kind": "formal", "additional_training_updates": 0,
        "checkpoint_sha256": previous.OFF["checkpoint"], "normalized_actor_sha256": previous.OFF["actor"],
        "protocol_sha256": sha256(metadata["protocol_file"]), "physics_dt_s": .0025, "decimation": 8,
        "policy_dt_s": .02, "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
        "inference_algorithm": "PPO", "normalizer_included_in_actor": True, "outer_normalizer": "Identity",
        "sideslip_reward_weight": 0., "old_200hz_actor_used": False, "absolute_height_gate": False, "airborne_failure": False}
    for name, value in expected_audit.items():
        require(audit.get(name) == value, "PhysX beta-OFF cohort/source audit differs: "+name)
    require(audit["wrapper_sha256"] == sha256(Path(__file__).with_name("eval_ppo_backend_400hz_beta_off_isaac.py"))
            and audit["beta_ablation"] == metadata["contract"]["beta_ablation"], "PhysX beta-OFF wrapper/ablation identity differs")
    require(audit["runner_config_sha256"] == sha256(local_file(root, str(folder/"runner_config.json"))),
            "PhysX actual runner configuration SHA differs")
    for key, path in metadata["paths"].items():
        require(audit["cohort_input_sha256"][path.name] == sha256(path), "PhysX report is not bound to SAME completed training metadata: "+key)
    require(audit["cohort_input_sha256"]["model_final.pt"] == previous.OFF["checkpoint"]
            and audit["cohort_input_sha256"]["policy.pt"] == previous.OFF["actor"], "PhysX source checkpoint/export differs")
    deployment = audit["deployment_actor_sha256"]
    require(isinstance(deployment, str) and len(deployment) == 64 and set(deployment) <= set("0123456789abcdef"),
            "Missing remotely verified deployed normalized actor SHA")
    require(report["status"] == "evaluated" and report["simulator"] == "IsaacLab/PhysX"
            and report["checkpoint_sha256"] == previous.OFF["checkpoint"]
            and report["motion_file_sha256"] == INPUT_PINS["motion"] and report["asset_root_file_sha256"] == INPUT_PINS["asset"]
            and report["deterministic_start_frame"] == 0 and report["seed"] == 42
            and report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False
            and report["full_horizon_diagnostic"] is True and report["physics_trajectory_sha256"] == sha256(recording),
            "PhysX deterministic full-native lifecycle differs")
    device = report["runtime_device_audit"]
    require(device["status"] == "verified" and device["requested_device"] == f"cuda:{audit['physical_gpu_index_requested']}",
            "PhysX native unmasked GPU identity is unverified")
    clock = report["physics_clock_readback"]
    require(clock == {"status": "verified", "decimation": 8, "simulator_physics_dt_s": .0025,
        "environment_physics_dt_s": .0025, "policy_dt_s": .02, "net_contact_sensor_update_period_s": .0025},
        "PhysX actual native clock/sensor readback differs")
    contract = report["policy_export"]
    require(contract["backend_400hz_cohort_evaluation"] == audit and contract["physics_dt"] == .0025 and contract["decimation"] == 8
            and contract["trained_task_profile"] == metadata["contract"]["trained_task_profile"], "PhysX final trained/exported contract differs")
    expected_profile = copy.deepcopy(metadata["contract"]["trained_task_profile"]); expected_profile["contact_friction"] = .7
    require(contract["task_profile"] == expected_profile and contract["training_domain_randomization"] == metadata["contract"]["training_domain_randomization"]
            and contract["leg_kp"] == 500. and contract["leg_kd"] == 10. and contract["actor_dim"] == 51,
            "PhysX evaluation task/PD/DR mapping differs")
    material = report["contact_material_audit"]; wheel = material["recipe"]["wheel_contact_friction"]
    require(material["live_physx_materials_verified"] is True and material["ground_material_usd_verified"] is True
            and material["recipe"]["training"] is False and wheel["range"] == [.7, .7]
            and wheel["physx_ground_friction"] == 1. and wheel["physx_ground_friction_combine_mode"] == "multiply"
            and material["recipe"]["nonwheel_friction"] == .7 and material["encoder_bias_range_rad"] == [0., 0.]
            and material["leg_pd_gain_readback"] == {"kp_range": [500., 500.], "kd_range": [10., 10.]},
            "PhysX fixed effective-pair friction/actuation readback differs")
    require(bool(material["native_plant_readback"]["readbacks"]), "Missing native plant nuisance readbacks")
    for name in ("policy_contract.json", "evaluation_inputs/evaluation_source_audit.json"):
        local_file(root, str(folder/name))
    require(read_json(local_file(root, str(folder/"policy_contract.json"))) == contract
            and read_json(local_file(root, str(folder/"evaluation_inputs/evaluation_source_audit.json"))) == audit,
            "PhysX copied final contract/source audit differs")
    return recording, report_file, report, proof


def load_comparison(physx_root=DEFAULT_PHYSX, mujoco_root=previous.DEFAULT_OFF, on_root=previous.DEFAULT_ON,
                    reference=previous.DEFAULT_REFERENCE, model=previous.DEFAULT_MODEL,
                    legacy_evaluator=previous.DEFAULT_LEGACY, protocol=previous.DEFAULT_PROTOCOL):
    root = Path(physx_root).resolve(); manifest_path = local_file(root, "manifest.json")
    manifest_sha = sha256(manifest_path); manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == 1 and manifest.get("kind") == KIND and manifest.get("nominal_mu") == .7
            and manifest.get("training_backend") == "physx"
            and manifest.get("reference_sha256") == sha256(reference) == INPUT_PINS["motion"]
            and manifest.get("model_sha256") == sha256(model) == INPUT_PINS["model"]
            and manifest.get("training_backend_comparison_complete") is False,
            "Require partial SAME-actor two-validation-engine retrieval, not a completed training-backend matrix")
    ledger = verify_files(root, manifest)
    mujoco_manifest_path = local_file(Path(mujoco_root).resolve(), "manifest.json")
    require(manifest["mujoco_retrieval_manifest_sha256"] == sha256(mujoco_manifest_path), "Existing strict beta-OFF MuJoCo bundle differs")
    paired, reference_values, previous_summary = previous.load_comparison(mujoco_root, on_root, reference, model, legacy_evaluator, protocol)
    old_manifest = read_json(mujoco_manifest_path)
    metadata = previous.copied_off_metadata(Path(mujoco_root).resolve(), old_manifest, verify_files(Path(mujoco_root).resolve(), old_manifest))
    recording, report_file, report, proof = validate_physx(root, manifest, ledger, metadata)
    trace = native_trace(previous.read_raw(recording), "policy_isaac")
    fk, fk_source = previous.metrics.load_fk(model, legacy_evaluator)
    reconstructed = fk.reconstruct(trace)
    trace.update(signals=reconstructed["signals"], wheel_quaternion_wxyz=reconstructed["wheel_quaternion_wxyz"],
                 wheel_center_velocity_w=reconstructed["wheel_center_velocity_w"])
    trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = previous.metrics.heading_geometry_masks(trace)
    traces = {"physx": trace, "mujoco": paired["beta_off"]}
    old_row = next(row for row in previous_summary["rows"] if row["id"] == "beta_off")
    rows = [{"id": "physx", "validation_backend": "PhysX", "beta_reward_weight": 0.,
        "directory": str(recording.parent), "checkpoint_sha256": previous.OFF["checkpoint"], "actor_sha256": previous.OFF["actor"],
        "recording_sha256": sha256(recording), "report_sha256": sha256(report_file), "verification": proof,
        "windows": previous.trace_metrics(trace, reference_values), "wheel_FK_audit": reconstructed["audit"]},
        {**copy.deepcopy(old_row), "id": "mujoco", "validation_backend": "native CPU MuJoCo"}]
    initial = {name: float(np.max(np.abs(traces["physx"][name][0]-traces["mujoco"][name][0])))
        for name in ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel", "wheel_pos")}
    require(sha256(manifest_path) == manifest_sha, "PhysX retrieval manifest changed during read")
    verify_files(root, manifest)
    return traces, reference_values, {"schema_version": 1, "kind": "new400hz_beta_off_same_physx_trained_actor_two_validation_engines",
        "case_order": list(ORDER), "rows": rows, "training_backend": "PhysX", "mu": .7, "beta_reward_weight": 0.,
        "checkpoint_sha256": previous.OFF["checkpoint"], "actor_sha256": previous.OFF["actor"],
        "training_backend_comparison_complete": False, "MJWarp_trained_actor_included": False,
        "scope": "Same completed PhysX-trained beta-OFF actor: in-engine PhysX versus transfer to native CPU MuJoCo",
        "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "action_rate_weight": -.5,
        "original_grounded_reward_weight": .4, "dense_ground_support_reward_used": False,
        "reference_sha256": sha256(reference), "model_sha256": sha256(model), "FK_provenance": fk_source,
        "physx_retrieval_manifest_sha256": manifest_sha, "physx_input_ledger_sha256": ledger,
        "mujoco_retrieval_manifest_sha256": sha256(mujoco_manifest_path),
        "on_retrieval_manifest_sha256_for_strict_existing_MJ_loader": previous_summary["on_retrieval_manifest_sha256"],
        "same_actual_source_actor": True, "weights_or_inference_loaded_locally": False,
        "PhysX_deployment_actor_SHA_remotely_verified_not_locally_rehashed": report["backend_400hz_cohort_evaluation"]["deployment_actor_sha256"],
        "initial_recorded_state_max_absolute_difference": initial,
        "materials_by_engine": {"physx": report["contact_material_audit"], "mujoco": read_json(Path(old_row["directory"])/"report.json")["contact_material_audit"]},
        "sample_size_per_engine": 1, "new_simulation": False, "new_policy_inference": False,
        "state_interpolation": False, "pose_alignment": False, "height_or_airborne_failure_gate": False,
        "lower_HF_alone_establishes_stability": False, "visual_reviewed": False,
        "limitations": ["Only one completed PhysX-trained beta-OFF actor in two validation engines; NOT a complete four-cell training-backend comparison.",
            "No MJWarp-trained smoke/intermediate/200Hz actor is used. Missing training-backend cells are not placeholders.",
            "Same400Hz/50Hz actor/interface and nominal effective pair mu=.7; contact laws, model representations and native telemetry operators still differ.",
            "Initial recorded state differences are reported, not aligned or forced equal. USD/MJ native inertias/contact readbacks are retained as nuisance evidence.",
            "Core1.44–4s alone can hide late loss of balance. Reference-overlap0–6.96s and fullnative0–6.98s are both reported.",
            "LowerHF is not stability; thresholded count/Fz is not a flight label. No absoluteheight/contact/tilt failure gate is introduced.",
            "One deterministic episode per engine is descriptive, not a statistical backend ranking or hardware claim.",
            "Training/export hashes are bound to copied completed metadata and strict remote proofs; no weights are downloaded or executed locally."]}


def compact_arrays(traces, reference):
    result = {"reference_"+name: np.asarray(value).copy() for name, value in reference.items() if isinstance(value, np.ndarray)}
    for engine in ORDER:
        result.update({engine+"_"+name: np.asarray(traces[engine][name]).copy() for name in previous.TRACE_FIELDS})
    return result


def recompute_compact(arrays):
    reference = {name[len("reference_"):]: value for name, value in arrays.items() if name.startswith("reference_")}
    result = {}
    for engine in ORDER:
        trace = {name: arrays[engine+"_"+name] for name in previous.TRACE_FIELDS}
        trace["signals"] = heading_signals(trace["quaternion"], trace["com_velocity"], trace["wheel_quaternion_wxyz"], trace["wheel_center_velocity_w"])
        result[engine] = previous.trace_metrics(trace, reference)
    return result


def render_frame(runtime, traces, qposes, reference, summary, plan, frame):
    from PIL import Image, ImageDraw
    require(type(frame) is int and 0 <= frame < 350, "Invalid50Hz original replay frame")
    index = int(previous.native_indices()[frame]); image = Image.new("RGB", CANVAS_SIZE, "#fafafa"); draw = ImageDraw.Draw(image)
    draw.text((12, 5), f"NEW 400 Hz β-OFF PPO | SAME PhysX-trained actor | μ=0.7 | t={frame*.02:.2f} s", font=runtime.font, fill="#111111")
    for column, engine in enumerate(ORDER):
        camera = apply_camera(runtime, plan, engine, frame)
        _pose(runtime.model, runtime.data, qposes[engine][index])
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        image.paste(Image.fromarray(runtime.renderer.render().copy()), (column*640, 58))
        x = column*640; draw.rectangle((x, 30, x+640, 57), fill="#f0f2f4")
        draw.text((x+9, 34), "PhysX-trained → "+LABELS[engine]+" | β reward OFF (0)", font=runtime.font, fill="#111111")
        value = diagnostics(traces[engine], reference, index)
        xy = f"{value['xy_error_m']:.3f} m" if value["xy_error_m"] is not None else "n/a (ref ends 6.96 s)"
        draw.text((x+9, 424), f"XY {xy} | speed {value['speed_m_s']:.2f} m/s | tilt {value['tilt_deg']:.1f} deg", font=runtime.font, fill="#111111")
        draw.text((x+9, 447), f"Original height {value['height_m']:.3f} m | native 400 Hz / video 50 Hz", font=runtime.font, fill="#111111")
        draw.text((x+9, 470), f"L/R count {int(value['count'][0])}/{int(value['count'][1])} | Fz {value['fz'][0]:.0f}/{value['fz'][1]:.0f} N", font=runtime.font, fill="#111111")
        display = {name: value[previous.native_indices()] if isinstance(value, np.ndarray) and value.ndim and len(value) == 2793 else value
            for name, value in traces[engine].items() if name != "signals"}
        display["signals"] = {name: value[previous.native_indices()] if isinstance(value, np.ndarray) and value.ndim and len(value) == 2793 else value
            for name, value in traces[engine]["signals"].items()}
        _map_panel(draw, display, reference, frame, (x+5, 500, x+635, 672), summary["world_bounds"], runtime.small)
    draw.text((12, 683), "PARTIAL: same PhysX-trained actor, two validation engines; MJWarp-trained β-OFF is not included.", font=runtime.small, fill="#333333")
    draw.text((12, 703), "Fixed distance/orientation for both; original root-follow translation and shared world-XY maps retain drift/falls.", font=runtime.small, fill="#333333")
    draw.text((12, 723), "N=1/engine. Read full/overlap results, not core HF alone. Native contact telemetry differs; zero is not a flight label.", font=runtime.small, fill="#333333")
    return image


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve previous evidence; use a fresh output directory")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    require(os.environ["MUJOCO_GL"] == "osmesa", "Require CPU OSMesa graphics")
    traces, reference, summary = load_comparison(args.physx_root, args.mujoco_root, args.on_root, args.reference,
        args.model, args.legacy_mujoco_evaluator, args.protocol)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes, audits = {}, {}
        for engine in ORDER:
            qposes[engine], ids = build_qpos(runtime.model, traces[engine])
            audits[engine] = {**replay_fk_audit(runtime.model, traces[engine], qposes[engine]), "named_joint_qpos_addresses": ids.tolist()}
        plan = prepare_camera(runtime, traces, qposes, ORDER, previous.native_indices())
        summary.update(world_bounds=[value.tolist() for value in world_bounds(traces, reference)],
            fixed_follow_camera={key: value for key, value in plan.items() if key != "lookat_by_case"},
            mesh_provenance=mesh_provenance(args.model), fk_audits=audits, renderer_sha256=sha256(__file__),
            video_frames=350, video_fps=50, video_size_pixels=list(CANVAS_SIZE), native_sample_stride=8,
            preview_original_time_s=5., robot_mesh_alpha=1.)
        output.mkdir(parents=True, exist_ok=False)
        compact = compact_arrays(traces, reference); recomputed = recompute_compact(compact)
        require(all(recomputed[row["id"]] == row["windows"] for row in summary["rows"]), "Compact native metrics do not reproduce all windows exactly")
        np.savez_compressed(output/"native_comparison_signals.npz", **compact)
        replay = {"fixed_camera_distance_m": np.array(plan["distance_m"])}
        for engine in ORDER:
            replay.update({engine+"_qpos": qposes[engine], engine+"_selected_native_indices": previous.native_indices(),
                engine+"_native_time_s": traces[engine]["time_s"], engine+"_camera_lookat_m": plan["lookat_by_case"][engine]})
        np.savez_compressed(output/"replay_states.npz", **replay)
        render_frame(runtime, traces, qposes, reference, summary, plan, 250).save(output/"comparison.png")
        render_frame(runtime, traces, qposes, reference, summary, plan, 150).save(output/"comparison_core_3s.png")
        from plot_ppo_beta_off_engine_comparison_400hz import draw_diagnostics
        summary["diagnostic_plot_metadata"] = draw_diagnostics(traces, reference, output/"turn_diagnostics.png")
        rows = previous.overview_rows(summary)
        with (output/"metric_table.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        write_json(output/"overview.json", {"kind": summary["kind"], "training_backend_comparison_complete": False,
            "same_actor": True, "beta_reward_weight": 0., "mu": .7, "sample_size_per_engine": 1, "visual_reviewed": False, "rows": rows})
        files = ["native_comparison_signals.npz", "replay_states.npz", "comparison.png", "comparison_core_3s.png", "turn_diagnostics.png", "metric_table.csv", "overview.json"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output/"beta_off_physx_vs_mujoco_400hz.mp4"), CANVAS_SIZE, fps=50,
                codec="libx264", quality=7, macro_block_size=2, output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(350):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, summary, plan, frame)))
            finally:
                stream.close()
            files.append("beta_off_physx_vs_mujoco_400hz.mp4")
        summary.update(video_generated=not args.skip_video, compact_native_recomputed_all_windows_exactly=True,
            artifacts={name: sha256(output/name) for name in files},
            helpers_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                ("render_ppo_beta_physx_to_mujoco_400hz.py", "render_ppo_fixed_follow.py", "render_physx_ppo_fullbody.py",
                 "plot_ppo_beta_off_engine_comparison_400hz.py", "render_ppo_training_recipe_comparison.py", "controller_trace.py")})
        write_json(output/"summary.json", summary)
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--physx-root", type=Path, default=DEFAULT_PHYSX)
    result.add_argument("--mujoco-root", type=Path, default=previous.DEFAULT_OFF)
    result.add_argument("--on-root", type=Path, default=previous.DEFAULT_ON)
    result.add_argument("--reference", type=Path, default=previous.DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=previous.DEFAULT_MODEL)
    result.add_argument("--legacy-mujoco-evaluator", type=Path, default=previous.DEFAULT_LEGACY)
    result.add_argument("--protocol", type=Path, default=previous.DEFAULT_PROTOCOL)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())

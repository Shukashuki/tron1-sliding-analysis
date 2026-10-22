"""NEW 400 Hz PhysX-trained PPO beta ON/OFF -> native CPU MuJoCo.

Existing full native recordings only. The two-panel replay preserves the
post-core fall; lower core high-frequency energy is NOT a stability claim.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path

import numpy as np

import analyze_ppo_backend_400hz_matrix as metrics
from controller_metrics import heading_signals
from controller_trace import native_trace
from ppo_backend_400hz_contract import DEFAULT_PROTOCOL, INPUT_PINS, read_json, require, sha256, verify_recording, write_json
from render_physx_ppo_fullbody import DEFAULT_MODEL, build_qpos, create_runtime, mesh_provenance, _pose
from render_ppo_backend_400hz_nominal import diagnostics, replay_fk_audit
from render_ppo_beta_nominal import DEFAULT_LEGACY, world_bounds
from render_ppo_fixed_follow import prepare_camera, apply_camera
from render_ppo_training_recipe_comparison import _map_panel
from summarize_ppo_backend_400hz_nominal import load_nominal, local_file, metadata_for, verify_files

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFF = ROOT / "outputs/ppo_beta_physx_to_mujoco_400hz_20261022_v1"
DEFAULT_ON = ROOT / "outputs/ppo_backend_400hz_nominal_20261022_v1"
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
ORDER = ("beta_on", "beta_off")
TITLES = {"beta_on": "β reward ON (2) | PhysX-trained → native CPU MuJoCo",
          "beta_off": "β reward OFF (0) | PhysX-trained → native CPU MuJoCo"}
CANVAS_SIZE = (1280, 750)
OFF = {"checkpoint": "fd9bb85554b7d25b31fbce57af63d33f73323e0530cf04990e95a7ac468f0688",
    "actor": "3e32748317244d4e1dc3f3df8fd6ec1052796fe1a44327ee7948c3601bbebdd5",
    "recording": "19c8781e4e94f012d017c9a52682126dc565188a3908084f3709f9ab29e72a60",
    "report": "09b93c9beabf4267b99a18fff5f8f99d2628bac66870fe0ac90a5f9e477dfc71"}
ON_CHECKPOINT = "c508dc3ba50db81f56f538bf23c550aca12f3279ddd15749d60582a487d2d8c7"
ON_ACTOR = "ab3c486ef779deb0cff52bfb33415f5c3a779381740a2c12898506e4e80f08bc"
INITIAL_SHA = "7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d"
MANIFEST_KIND = "verified_beta_off_physx_to_mujoco_400hz_single_nominal_replay_inputs"


def native_indices():
    return np.arange(350, dtype=np.int64)*8


def read_raw(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def check_profile_pair(on, off):
    expected = copy.deepcopy(on)
    require(expected["reward_weights"]["sideslip"] == 2., "ON sideslip reward differs")
    expected["reward_weights"]["sideslip"] = 0.
    require(expected == off and off["action_rate_weight"] == -.5
            and off["reward_weights"]["grounded"] == .4
            and off["height_regularization"]["mode"] == "vertical_acceleration_only",
            "Trained task profiles differ beyond beta2->0")


def copied_off_metadata(root, manifest, ledger):
    """Metadata-only audit; remote actor hashes are disclosed, not loaded locally."""
    import train_ppo_backend_400hz_beta_off as trainer
    protocol_file = local_file(root, "config/ppo_backend_400hz_beta_off_v1.json")
    require(sha256(protocol_file) == sha256(trainer.DEFAULT_PROTOCOL), "Copied beta-OFF protocol differs")
    protocol = trainer.validate_protocol(read_json(protocol_file))
    adapted = {"training_metadata": {"physx": manifest["training_metadata"]}, "cohort_id": protocol["cohort_id"]}
    data = metadata_for(root, "physx", adapted, ledger, protocol_file)
    paths = data["paths"]
    require(paths["final_policy_contract"].read_bytes() == paths["policy_contract"].read_bytes(),
            "OFF canonical and compatibility contracts differ")
    training_manifest, startup = read_json(paths["manifest"]), read_json(paths["startup_audit"])
    completion, contract, status = data["completion"], data["contract"], data["status"]
    require(completion["checkpoint_sha256"] == OFF["checkpoint"]
            and completion["exported_policy_sha256"] == OFF["actor"]
            and completion["policy_contract_sha256"] == sha256(paths["policy_contract"])
            and completion["initial_policy_state_sha256"] == INITIAL_SHA,
            "OFF completed checkpoint/export/initial network differs")
    for item in (contract, completion, training_manifest, training_manifest["cohort"], startup, status):
        require(item.get("beta_ablation") == trainer.ablation_identity(), "Missing beta-only training identity")
    require(training_manifest["contract"] == contract
            and startup["runtime"] == training_manifest["cohort"]["runtime"]
            and startup["wrapper_sha256"] == sha256(trainer.__file__)
            and startup["initial_policy_state_sha256"] == INITIAL_SHA,
            "OFF startup/final contract/source crosslinks differ")
    trainer.validate_runtime_fields(startup["runtime"]["fields"])
    trainer.validate_task_profile(contract["trained_task_profile"], protocol)
    require(completion["sideslip_reward_weight"] == contract["sideslip_reward_weight"] == 0.
            and completion["listed_reward_term_count"] == 15 and completion["nonzero_reward_term_count"] == 14
            and completion["effective_sideslip_reward_enabled"] is False,
            "OFF must truly disable beta reward, not use epsilon or relabel an ON actor")
    data.update(protocol_file=protocol_file, protocol=protocol, manifest=training_manifest, startup=startup)
    return data


def check_off_case(root, manifest, ledger, metadata):
    cases = manifest["cases"]
    require(len(cases) == 1, "Require exactly one completed OFF native MuJoCo cell")
    case = cases[0]
    require(case.get("case_id") == "beta_off_physx_to_mujoco" and case.get("training_backend") == "physx"
            and case.get("engine_tag") == "mujoco" and case.get("validation_backend") == "native CPU MuJoCo"
            and case.get("mu") == .7 and case.get("raw_file") == "rollout.npz" and case.get("report_file") == "report.json",
            "OFF case is not native400Hz PhysX->MuJoCo nominal")
    folder = Path(case["directory"])
    recording = local_file(root, str(folder/"rollout.npz")); report_file = local_file(root, str(folder/"report.json"))
    for name, digest, path in (("recording", OFF["recording"], recording), ("report", OFF["report"], report_file)):
        require(case[name+"_sha256"] == sha256(path) == digest
                and ledger[str(path.relative_to(root))] == digest, "Pinned OFF "+name+" differs")
    proof = read_json(local_file(root, str(folder/"remote_verification.json")))
    launch = read_json(local_file(root, str(folder/"launch_status.json")))
    require(proof == case["verification"] == launch["verification"] and launch["status"] == "completed_verified"
            and launch["child_returncode"] == 0 and launch["physical_gpu_allocation"] is False
            and launch["cuda_visible_devices"] == "", "OFF completed CPU launch/remote proof differs")
    expected = {"status": "verified", "training_backend": "physx", "validation_backend": "native CPU MuJoCo",
        "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
        "physics_dt_s": .0025, "policy_dt_s": .02, "smoke_only": False,
        "checkpoint_sha256": OFF["checkpoint"], "recording_sha256": OFF["recording"],
        "report_sha256": OFF["report"], "evaluation_contact_friction": .7}
    for key, value in expected.items():
        require(proof.get(key) == value, "OFF remote proof differs: "+key)
    require(case["checkpoint_sha256"] == OFF["checkpoint"] and case["actor_sha256"] == OFF["actor"],
            "OFF case actor identity differs")
    for key, value in verify_recording(recording, "native CPU MuJoCo").items():
        require(proof.get(key) == value, "Local native clock/action verification differs: "+key)
    report = read_json(report_file); audit = report["backend_400hz_cohort_evaluation"]
    expected_audit = {"status": "verified", "training_backend": "physx", "validation_backend": "native CPU MuJoCo",
        "run_kind": "formal", "smoke_only": False, "trained_policy_claimed": True,
        "checkpoint_sha256": OFF["checkpoint"], "normalized_actor_sha256": OFF["actor"],
        "protocol_sha256": sha256(metadata["protocol_file"]), "physics_dt_s": .0025, "decimation": 8,
        "policy_dt_s": .02, "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
        "old_200hz_actor_used": False, "additional_training_updates": 0, "normalizer_included_in_actor": True,
        "outer_normalizer": "Identity", "normalization_applications_per_inference": 1,
        "extra_normalization": False, "sideslip_reward_weight": 0., "recording_sha256": OFF["recording"]}
    for key, value in expected_audit.items():
        require(audit.get(key) == value, "OFF report source/cohort audit differs: "+key)
    require(audit == read_json(local_file(root, str(folder/"evaluation_inputs/evaluation_source_audit.json")))
            and report["contract"]["backend_400hz_cohort_evaluation"] == audit,
            "OFF exported report/source audit crosslink differs")
    mapping = {"completion_audit": "cohort_completion_audit.json", "final_policy_contract": "final_policy_contract.json",
        "policy_contract": "policy_contract.json", "manifest": "manifest.json", "startup_audit": "cohort_startup_audit.json", "run_status": "run_status.json"}
    for key, name in mapping.items():
        require(audit["cohort_input_sha256"][name] == launch["cohort_input_sha256"][name] == sha256(metadata["paths"][key]),
                "OFF report is not bound to completed copied metadata: "+name)
    require(audit["cohort_input_sha256"]["model_final.pt"] == OFF["checkpoint"]
            and audit["cohort_input_sha256"]["policy.pt"] == OFF["actor"]
            and audit["wrapper_sha256"] == sha256(Path(__file__).with_name("eval_ppo_backend_400hz_beta_off_mujoco.py")),
            "OFF normalized actor/source identity differs")
    exported = copy.deepcopy(report["contract"]); exported.pop("backend_400hz_cohort_evaluation")
    declared_profile = read_json(local_file(root, str(folder/"evaluation_inputs/task_profile.json")))
    expected_evaluation_profile = copy.deepcopy(declared_profile); expected_evaluation_profile["contact_friction"] = .7
    require(exported == read_json(local_file(root, str(folder/"evaluation_inputs/contract.json")))
            and exported["trained_task_profile"] == metadata["contract"]["trained_task_profile"]
            and declared_profile == exported["trained_task_profile"]
            and exported["task_profile"] == expected_evaluation_profile,
            "OFF declared/trained/evaluation material-profile mapping differs")
    require(report["status"] == "completed" and report["completed_full_reference"] is True
            and report["hidden_resets"] == 0 and report["root_state_writes"] == 1
            and report["start_reference_frame"] == 0 and report["domain_parameters_resampled_during_episode"] is False
            and report["motion_sha256"] == INPUT_PINS["motion"] and report["model_sha256"] == INPUT_PINS["model"]
            and report["policy_sha256"] == OFF["actor"], "OFF full-horizon/native plant lifecycle differs")
    return recording, report_file, report, proof


def trace_metrics(trace, reference):
    return {"core": metrics.window_metrics(trace, reference, trace["signals"], (1.44, 4.)),
        "full": metrics.window_metrics(trace, reference, trace["signals"], (0., 6.98)),
        "reference_overlap": metrics.window_metrics(trace, reference, trace["signals"], (0., 6.96)),
        "post_core": metrics.window_metrics(trace, reference, trace["signals"], (4., 6.98))}


def load_comparison(off_root=DEFAULT_OFF, on_root=DEFAULT_ON, reference=DEFAULT_REFERENCE,
                    model=DEFAULT_MODEL, legacy_evaluator=DEFAULT_LEGACY, protocol=DEFAULT_PROTOCOL):
    """Only the two actual completed PhysX-trained400Hz -> native-MuJoCo cells."""
    root = Path(off_root).resolve(); manifest_path = local_file(root, "manifest.json")
    manifest_sha = sha256(manifest_path); manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == 1 and manifest.get("kind") == MANIFEST_KIND
            and manifest.get("nominal_mu") == .7 and manifest.get("training_backend") == "physx"
            and manifest.get("validation_backend") == "native CPU MuJoCo" and manifest.get("beta_reward_weight") == 0.
            and manifest.get("reference_sha256") == sha256(reference) == INPUT_PINS["motion"]
            and manifest.get("model_sha256") == sha256(model) == INPUT_PINS["model"], "Wrong OFF nominal retrieval identity")
    ledger = verify_files(root, manifest)
    off_metadata = copied_off_metadata(root, manifest, ledger)
    off_recording, off_report_file, off_report, off_proof = check_off_case(root, manifest, ledger, off_metadata)
    nominal, reference_values, nominal_summary = load_nominal(on_root, reference, model, legacy_evaluator, protocol)
    on_row = next(row for row in nominal_summary["rows"] if row["id"] == "physx_to_mujoco")
    require(on_row["checkpoint_sha256"] == ON_CHECKPOINT and on_row["actor_sha256"] == ON_ACTOR,
            "Wrong NEW400Hz ON PhysX-trained actor")
    on_folder = Path(on_row["directory"]); on_report = read_json(on_folder/"report.json")
    on_training = read_json(Path(on_root)/"training/physx/manifest.json")
    on_contract = read_json(Path(on_root)/"training/physx/final_policy_contract.json")
    check_profile_pair(on_contract["trained_task_profile"], off_metadata["contract"]["trained_task_profile"])
    for key in ("training_domain_randomization", "domain_randomization", "default_joint_position_rad",
                "physics_dt", "decimation", "leg_kp", "leg_kd", "actor_dim", "critic_dim", "action_dim"):
        require(on_contract[key] == off_metadata["contract"][key], "Paired training/actuation contract differs: "+key)
    for key in ("algorithm", "seed", "num_envs", "iterations", "transitions_requested", "smoke_steps", "task_implementation"):
        require(on_training[key] == off_metadata["manifest"][key], "Paired training budget/setup differs: "+key)
    require(on_training["initial_policy_state_sha256"] == off_metadata["manifest"]["initial_policy_state_sha256"] == INITIAL_SHA,
            "Paired initial network differs")
    require(on_report["contact_material_audit"] == off_report["contact_material_audit"]
            and on_report["backend_400hz_cohort_evaluation"]["generated_source_sha256"] == off_report["backend_400hz_cohort_evaluation"]["generated_source_sha256"],
            "Paired compiled native plant/material or generated evaluator differs")
    raw = {"beta_on": read_raw(on_folder/"rollout.npz"), "beta_off": read_raw(off_recording)}
    for key in ("initial_qpos", "initial_qvel", "action_observation", "previous_clipped_actions"):
        first = raw["beta_on"][key] if key.startswith("initial_") else raw["beta_on"][key][0]
        second = raw["beta_off"][key] if key.startswith("initial_") else raw["beta_off"][key][0]
        require(np.array_equal(first, second), "Paired original initial state/51D observation differs: "+key)
    fk, fk_source = metrics.load_fk(model, legacy_evaluator)
    traces, rows = {}, []
    for label in ORDER:
        trace = native_trace(raw[label], "policy_mujoco")
        require(trace["time_s"].shape == (2793,) and trace["physics_dt_s"] == .0025
                and np.allclose(trace["time_s"], np.arange(2793)*.0025, rtol=0, atol=1e-10),
                "Require complete original native400Hz clock")
        reconstructed = fk.reconstruct(trace)
        trace.update(signals=reconstructed["signals"], wheel_quaternion_wxyz=reconstructed["wheel_quaternion_wxyz"],
                     wheel_center_velocity_w=reconstructed["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = metrics.heading_geometry_masks(trace)
        traces[label] = trace
        is_on = label == "beta_on"
        rows.append({"id": label, "title": TITLES[label], "beta_reward_weight": 2. if is_on else 0.,
            "checkpoint_sha256": ON_CHECKPOINT if is_on else OFF["checkpoint"], "actor_sha256": ON_ACTOR if is_on else OFF["actor"],
            "directory": str(on_folder if is_on else off_recording.parent),
            "recording_sha256": on_row["input_hashes"]["recording"] if is_on else OFF["recording"],
            "report_sha256": on_row["input_hashes"]["report"] if is_on else OFF["report"],
            "verification": on_row["remote_verification"] if is_on else off_proof,
            "windows": trace_metrics(trace, reference_values), "wheel_FK_audit": reconstructed["audit"]})
    for name in ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel", "wheel_pos"):
        require(np.array_equal(traces["beta_on"][name][0], traces["beta_off"][name][0]), "Initial measured state differs: "+name)
    require(sha256(manifest_path) == manifest_sha, "OFF manifest changed during offline reading")
    verify_files(root, manifest)
    return traces, reference_values, {"schema_version": 1, "kind": "new400hz_physx_trained_beta_on_off_native_cpu_mujoco_comparison",
        "case_order": list(ORDER), "rows": rows, "training_backend": "PhysX", "validation_backend": "native CPU MuJoCo",
        "mu": .7, "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "native_samples_per_case": 2793,
        "policy_decisions_per_case": 349, "action_rate_weight": -.5, "original_grounded_reward_weight": .4,
        "dense_ground_support_reward_used": False, "sample_size_per_case": 1,
        "reference_sha256": sha256(reference), "model_sha256": sha256(model), "FK_provenance": fk_source,
        "off_retrieval_manifest_sha256": manifest_sha, "off_input_ledger_sha256": ledger,
        "on_retrieval_manifest_sha256": nominal_summary["retrieval_manifest_sha256"],
        "off_metadata_hashes": {key: sha256(path) for key, path in off_metadata["paths"].items()},
        "training_pair_audit": {"same_initial_network_sha256": INITIAL_SHA, "same_training_budget": 12288000,
            "same_setup_smoke_steps": on_training["smoke_steps"], "same_training_DR": True,
            "task_profile_only_change": "reward_weights.sideslip:2->0", "same_native_plant_and_material_readback": True,
            "same_initial_physical_state_and_first_observation": True, "same_generated_native_evaluator": True,
            "training_physical_GPU_by_condition": {"beta_on": 4, "beta_off": 6},
            "weights_locally_loaded_or_reverified": False},
        "full_horizon_preserved": True, "state_interpolation": False, "pose_alignment": False,
        "height_or_airborne_failure_gate": False, "statistical_causal_effect_claimed": False,
        "lower_HF_alone_establishes_stability": False, "new_training": False, "new_simulation": False,
        "new_policy_inference": False, "visual_reviewed": False,
        "limitations": ["Two actual NEW400Hz PhysX-trained actors evaluated in native CPU MuJoCo, not MJWarp and not a200Hz substitute.",
            "Only explicit body/COM beta reward changes2->0; nonzero sideslip remains a valid motion feature. Wheel-beta reward is not introduced.",
            "One seed/episode per condition, with training GPU4 versus GPU6; matched configuration/initialization does not establish a statistical causal effect or hardware robustness.",
            "Core1.44–4s can hide later loss of balance. Also inspect reference-overlap0–6.96s and fullnative0–6.98s; lowerHF alone is not improvement.",
            "Persistent tilt30/45/60deg for0.1s is a diagnostic with onset/confirmation/censoring, not a termination or absoluteheight gate.",
            "Native counts/Fz eligibility is not a flight label. All poses, heights and the entire post-core trajectory are retained.",
            "Reference ends6.96s; full6.98s tracking is unavailable without extrapolation. Contact/tilt/HF still use the full native horizon.",
            "Copied completed metadata and remote strict proofs bind actor hashes; no checkpoint/actor weights are downloaded or executed locally."]}


TRACE_FIELDS = ("time_s", "position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel",
    "wheel_pos", "contact_count", "contact_force", "contact_slip", "wheel_quaternion_wxyz", "wheel_center_velocity_w")


def compact_arrays(traces, reference):
    result = {"reference_"+name: np.asarray(value).copy() for name, value in reference.items() if isinstance(value, np.ndarray)}
    for label in ORDER:
        result.update({label+"_"+name: np.asarray(traces[label][name]).copy() for name in TRACE_FIELDS})
    return result


def recompute_compact(arrays):
    reference = {name[len("reference_"):]: value for name, value in arrays.items() if name.startswith("reference_")}
    result = {}
    for label in ORDER:
        trace = {name: np.asarray(arrays[label+"_"+name]) for name in TRACE_FIELDS}
        trace["signals"] = heading_signals(trace["quaternion"], trace["com_velocity"], trace["wheel_quaternion_wxyz"], trace["wheel_center_velocity_w"])
        result[label] = trace_metrics(trace, reference)
    return result


def overview_rows(summary):
    rows = []
    for case in summary["rows"]:
        for window in ("core", "reference_overlap", "full", "post_core"):
            value = case["windows"][window]; tracking = value["tracking"]
            event = value["persistent_tilt_diagnostics"]["30"]["first_persistent_event"]
            rows.append({"id": case["id"], "window": window, "beta_reward_weight": case["beta_reward_weight"],
                "xy_rmse_m": tracking.get("xy_rmse_m"), "xy_peak_m": tracking.get("xy_peak_m"),
                "speed_retention": tracking.get("speed_retention"), "beta_rmse_deg": tracking.get("beta_rmse_deg"),
                "beta_valid_samples": tracking.get("beta_valid_samples"),
                "world_angular_xy_common10_100_rms_rad_s": value["world_angular_xy_common10_100"]["common_band_rms"],
                "maximum_tilt_deg": value["maximum_tilt_deg"],
                "tilt30_persistent_onset_s": event["onset_s"] if event else None})
    return rows


def render_frame(runtime, traces, qposes, reference, summary, plan, frame):
    from PIL import Image, ImageDraw
    require(type(frame) is int and 0 <= frame < 350, "Invalid50Hz replay frame")
    index = int(native_indices()[frame]); image = Image.new("RGB", CANVAS_SIZE, "#fafafa"); draw = ImageDraw.Draw(image)
    draw.text((12, 5), f"NEW 400 Hz PPO | PhysX → native CPU MuJoCo | μ=0.7 | t={frame*.02:.2f} s", font=runtime.font, fill="#111111")
    for column, label in enumerate(ORDER):
        camera = apply_camera(runtime, plan, label, frame)
        _pose(runtime.model, runtime.data, qposes[label][index])
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        image.paste(Image.fromarray(runtime.renderer.render().copy()), (column*640, 58))
        x = column*640; draw.rectangle((x, 30, x+640, 57), fill="#f0f2f4")
        draw.text((x+9, 34), TITLES[label], font=runtime.font, fill="#111111")
        value = diagnostics(traces[label], reference, index)
        xy = f"{value['xy_error_m']:.3f} m" if value["xy_error_m"] is not None else "n/a (ref ends 6.96 s)"
        draw.text((x+9, 424), f"XY {xy} | speed {value['speed_m_s']:.2f} m/s | tilt {value['tilt_deg']:.1f} deg", font=runtime.font, fill="#111111")
        draw.text((x+9, 447), f"Original height {value['height_m']:.3f} m | native 400 Hz / video 50 Hz", font=runtime.font, fill="#111111")
        draw.text((x+9, 470), f"L/R count {int(value['count'][0])}/{int(value['count'][1])} | Fz {value['fz'][0]:.0f}/{value['fz'][1]:.0f} N", font=runtime.font, fill="#111111")
        # The map owns every50Hz exact native point; no interpolation or coarse line.
        display = {name: value[native_indices()] if isinstance(value, np.ndarray) and value.ndim and len(value) == 2793 else value
            for name, value in traces[label].items() if name != "signals"}
        display["signals"] = {name: value[native_indices()] if isinstance(value, np.ndarray) and value.ndim and len(value) == 2793 else value
            for name, value in traces[label]["signals"].items()}
        _map_panel(draw, display, reference, frame, (x+5, 500, x+635, 672), summary["world_bounds"], runtime.small)
    draw.text((12, 683), "Core ends 4.00 s. Read full/overlap trajectories too: lower core HF alone is NOT stability.", font=runtime.small, fill="#333333")
    draw.text((12, 703), "Fixed distance/orientation for both panels; translation follows original root. World-XY maps keep drift.", font=runtime.small, fill="#333333")
    draw.text((12, 723), "N=1. Original grounded +0.4; action-rate −0.5. No pose edits/inference/physics. Contact zeros are not flight labels.", font=runtime.small, fill="#333333")
    return image


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve prior evidence; use a fresh output directory")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    require(os.environ["MUJOCO_GL"] == "osmesa", "Require local CPU OSMesa graphics")
    traces, reference, summary = load_comparison(args.off_root, args.on_root, args.reference, args.model,
        args.legacy_mujoco_evaluator, args.protocol)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        qposes, fk_audits = {}, {}
        for label in ORDER:
            qposes[label], ids = build_qpos(runtime.model, traces[label])
            fk_audits[label] = {**replay_fk_audit(runtime.model, traces[label], qposes[label]), "named_joint_qpos_addresses": ids.tolist()}
        plan = prepare_camera(runtime, traces, qposes, ORDER, native_indices())
        summary.update(world_bounds=[value.tolist() for value in world_bounds(traces, reference)],
            fixed_follow_camera={key: value for key, value in plan.items() if key != "lookat_by_case"},
            mesh_provenance=mesh_provenance(args.model), fk_audits=fk_audits, renderer_sha256=sha256(__file__),
            video_frames=350, video_fps=50, video_size_pixels=list(CANVAS_SIZE), native_sample_stride=8,
            preview_original_time_s=5., robot_mesh_alpha=1.)
        output.mkdir(parents=True, exist_ok=False)
        compact = compact_arrays(traces, reference)
        recomputed = recompute_compact(compact)
        require(all(recomputed[row["id"]] == row["windows"] for row in summary["rows"]),
                "Compact native bundle does not exactly reproduce all reported windows")
        np.savez_compressed(output/"native_comparison_signals.npz", **compact)
        replay = {"fixed_camera_distance_m": np.array(plan["distance_m"])}
        for label in ORDER:
            replay.update({label+"_qpos": qposes[label], label+"_selected_native_indices": native_indices(),
                label+"_native_time_s": traces[label]["time_s"], label+"_camera_lookat_m": plan["lookat_by_case"][label]})
        np.savez_compressed(output/"replay_states.npz", **replay)
        render_frame(runtime, traces, qposes, reference, summary, plan, 250).save(output/"comparison.png")
        render_frame(runtime, traces, qposes, reference, summary, plan, 150).save(output/"comparison_core_3s.png")
        from plot_ppo_beta_physx_to_mujoco_400hz import draw_diagnostics
        draw_diagnostics(traces, reference, output/"turn_failure_diagnostics.png")
        rows = overview_rows(summary)
        with (output/"metric_table.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        write_json(output/"overview.json", {"kind": summary["kind"], "mu": .7, "sample_size_per_case": 1,
            "visual_reviewed": False, "lower_HF_alone_establishes_stability": False, "rows": rows})
        files = ["native_comparison_signals.npz", "replay_states.npz", "comparison.png", "comparison_core_3s.png",
            "turn_failure_diagnostics.png", "metric_table.csv", "overview.json"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output/"beta_physx_to_mujoco_400hz.mp4"), CANVAS_SIZE, fps=50,
                codec="libx264", quality=7, macro_block_size=2, output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(350):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, summary, plan, frame)))
            finally:
                stream.close()
            files.append("beta_physx_to_mujoco_400hz.mp4")
        summary["video_generated"] = not args.skip_video
        summary["compact_native_recomputed_all_windows_exactly"] = True
        summary["helpers_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in
            ("render_ppo_fixed_follow.py", "render_physx_ppo_fullbody.py", "render_ppo_training_recipe_comparison.py",
             "summarize_ppo_backend_400hz_nominal.py", "analyze_ppo_backend_400hz_matrix.py", "controller_trace.py",
             "controller_metrics.py", "train_ppo_backend_400hz_beta_off.py", "eval_ppo_backend_400hz_beta_off_mujoco.py",
             "plot_ppo_beta_physx_to_mujoco_400hz.py")}
        summary["artifacts"] = {name: sha256(output/name) for name in files}
        write_json(output/"summary.json", summary)
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--off-root", type=Path, default=DEFAULT_OFF)
    result.add_argument("--on-root", type=Path, default=DEFAULT_ON)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--legacy-mujoco-evaluator", type=Path, default=DEFAULT_LEGACY)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())

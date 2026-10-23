"""Offline nominal audits of the NEW formal any-wheel-contact PPO only.

No actors, checkpoint pickles, physics, FK models or rendering are loaded. A
copied parent verification proof binds each completed report/recording; formal
training artifacts were verified on the execution host, not rehashed locally.
The old same-engine PhysX-trained beta-ON baselines are separately SHA-pinned.
Numerical episode screens never claim the broader task or visual review done.
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

import analyze_ppo_backend_400hz_matrix as metrics
from analyze_cem_ground_contact_v2 import support_summary
from analyze_ppo_ground_support_failure import persistent_events
from controller_metrics import wrap
from controller_trace import native_trace
import eval_ppo_ground_contact_v2 as evaluator
import ground_contact_reward_v2 as definition
from nmpc_tracking_objective import quaternion_rotate
from ppo_backend_400hz_contract import (INPUT_PINS, read_json, require, sha256,
                                       verify_protocol, verify_recording, write_json)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "config/ppo_ground_contact_v2_400hz_v1.json"
PROTOCOL_SHA256 = "1d34a2a71b85f678e8bc305f926f4b58dd43e66658dacdad20bb8e3722735260"
DEFAULT_REFERENCE = ROOT.parent / "tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz"
BASELINE_ROOT = ROOT / "outputs/ppo_backend_400hz_nominal_20261022_v1"
BASELINE_SUMMARY = ROOT / "results/ppo_backend_400hz_nominal_20261022_v1/summary.json"
BASELINE_SUMMARY_SHA256 = "6dcc65fb2b863d856190e452bcea744dfeb2bdd253d1d31389cb51146fcac5ac"
BASELINE_CHECKPOINT = "c508dc3ba50db81f56f538bf23c550aca12f3279ddd15749d60582a487d2d8c7"
BASELINE_ACTOR = "ab3c486ef779deb0cff52bfb33415f5c3a779381740a2c12898506e4e80f08bc"
BACKENDS = {"physx": "PhysX", "mujoco": "native CPU MuJoCo"}
WINDOWS = {"core": (1.44, 4.), "whole_reference_overlap": (0., 6.96), "full_native": (0., 6.98)}
TRACE_FIELDS = ("time_s", "position", "quaternion", "com_velocity", "angular_velocity",
                "joint_pos", "joint_vel", "wheel_pos", "contact_count", "contact_force", "contact_slip")


def load_protocol(path=DEFAULT_PROTOCOL):
    require(sha256(path) == PROTOCOL_SHA256, "Frozen V2a protocol/acceptance bytes changed")
    protocol = verify_protocol(path)
    require(protocol["cohort_id"] == evaluator.COHORT_ID
            and protocol["reward_definition_version"] == evaluator.REWARD_VERSION,
            "Wrong declared new any-wheel cohort")
    definition.validate_settings(protocol["ground_contact_reward_v2"])
    return protocol


def _digest(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value), "Invalid SHA256: " + label)
    return value


def _file(folder, name):
    path = Path(folder).resolve() / name
    require(path.is_file() and not path.is_symlink(), "Missing/indirect completed artifact: " + str(path))
    return path


def _clock(trace):
    time = np.asarray(trace["time_s"])
    require(time.shape == (2793,) and trace["physics_dt_s"] == .0025
            and np.allclose(time, np.arange(2793)*.0025, atol=1e-10, rtol=0),
            "Require consecutive 400 Hz native states, 0–6.98s, not a partial/reset-spliced trace")
    count = np.asarray(trace["contact_count"])
    require((count >= 0).all() and np.array_equal(count, np.round(count)), "Invalid native contact counts")


def _root_signals(trace):
    """Only measured root geometry: no fabricated wheel headings or FK needed."""
    quaternion = np.asarray(trace["quaternion"], dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    forward = quaternion_rotate(quaternion, np.array([1., 0., 0.]))
    velocity = np.asarray(trace["com_velocity"], dtype=float)
    speed = np.linalg.norm(velocity[:, :2], axis=-1)
    course = np.arctan2(velocity[:, 1], velocity[:, 0])
    heading = np.arctan2(forward[:, 1], forward[:, 0])
    return {"root_beta_valid": (speed >= .25) & (np.linalg.norm(forward[:, :2], axis=-1) > 1e-6),
            "root_directed_beta_rad": wrap(course-heading)}


def analyze_trace(trace, reference, *, floor_normal_force=None, nonwheel_ground_force=None):
    """Pure native-signal arithmetic; root beta is relative to nonzero reference."""
    _clock(trace)
    signals = _root_signals(trace)
    windows = {name: metrics.window_metrics(trace, reference, signals, interval)
               for name, interval in WINDOWS.items()}
    force = np.asarray(trace["contact_force"] if floor_normal_force is None else floor_normal_force, dtype=float)
    require(force.shape == (2793, 2, 3) and np.isfinite(force).all()
            and (force[..., 2] >= -1e-4).all(), "Invalid upward ground-normal force array")
    # No count gate: weak individual points can sum to more than 5 N per wheel.
    support = {"core": support_summary(trace["time_s"], force[..., 2], (1.44, 4.)),
               "full_contact": support_summary(trace["time_s"], force[..., 2], (.02, 6.96))}
    events = {}
    for name, interval in (("core", (1.44, 4.)), ("whole_reference_overlap", (0., 6.96))):
        time = trace["time_s"]
        mask = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
        query = time[mask]
        target = np.column_stack([np.interp(query, reference["time_s"], reference["position"][:, j]) for j in (0, 1)])
        error = np.linalg.norm(trace["position"][mask, :2]-target, axis=-1)
        events[name] = persistent_events(query, error > .30, .1)
    nonwheel = {"available": False, "reason": "No measured nonwheel-ground normal force supplied"}
    if nonwheel_ground_force is not None:
        value = np.asarray(nonwheel_ground_force, dtype=float)
        require(value.shape == (2793,) and np.isfinite(value).all() and (value >= 0).all(),
                "Invalid measured nonwheel-ground normal-force norm")
        nonwheel = {"available": True, "threshold_n_strict": 1., "maximum_n": float(value.max()),
                    "sample_fraction_over_threshold": float(np.mean(value > 1.)),
                    "persistent_events": persistent_events(trace["time_s"], value > 1., .1),
                    "used_as_automatic_failure_gate": False}
    return {"windows": windows, "any_wheel_ground_support": support,
            "persistent_xy_departure_gt0p30m": events, "nonwheel_ground_contact_diagnostic": nonwheel,
            "nonzero_beta_is_intended": True, "bilateral_support_required": False,
            "height_contact_tilt_or_beta_used_as_XY_failure_gate": False,
            "trajectory_aligned_smoothed_or_extrapolated": False,
            "root_heading_and_reference_beta_only_no_wheel_FK_inferred": True,
            "whole_tracking_reference_end_s": 6.96, "physical_recording_end_s": 6.98}


def recorded_metrics(raw, engine, reference):
    require(engine in BACKENDS, "Select physx or mujoco explicitly")
    trace = native_trace(raw, "policy_isaac" if engine == "physx" else "policy_mujoco")
    _clock(trace)
    source = "ground_normal_force_w" if "ground_normal_force_w" in raw else "wheel_ground_contact_force_w_n"
    force = np.asarray(raw[source], dtype=float)
    if engine == "physx":
        require(force.shape == (2793, 1, 2, 3), "Require one PhysX environment for actual floor normal forces")
        force = force[:, 0]
    nonwheel = raw.get("nonwheel_ground_contact_force_norm_n")
    if nonwheel is not None:
        nonwheel = np.asarray(nonwheel)
        if engine == "physx":
            require(nonwheel.shape == (2793, 1), "Require one native nonwheel-ground-force environment")
            nonwheel = nonwheel[:, 0]
    result = analyze_trace(trace, reference, floor_normal_force=force, nonwheel_ground_force=nonwheel)
    result["support_force_source"] = source
    trace["floor_normal_force"] = force.copy()
    if nonwheel is not None: trace["nonwheel_ground_force"] = nonwheel.copy()
    return result, trace


def nominal_screen(candidate, baseline, protocol, engine):
    """Apply the PRE-training protocol exactly; never soften failed criteria."""
    require(engine in BACKENDS, "Unknown baseline engine")
    criteria = protocol["nominal_acceptance_before_training"]
    core = candidate["windows"]["core"]["tracking"]
    whole = candidate["windows"]["whole_reference_overlap"]["tracking"]
    old_hf = baseline["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"]
    new_hf = candidate["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"]
    ratio = new_hf / old_hf if old_hf > 0 else None
    contact = candidate["any_wheel_ground_support"]["full_contact"]
    checks = {
        "core_xy_rmse": core["xy_rmse_m"] <= criteria["core_and_whole_xy_rmse_m_max"],
        "core_xy_peak": core["xy_peak_m"] <= criteria["core_and_whole_xy_peak_m_max"],
        "whole_xy_rmse": whole["xy_rmse_m"] <= criteria["core_and_whole_xy_rmse_m_max"],
        "whole_xy_peak": whole["xy_peak_m"] <= criteria["core_and_whole_xy_peak_m_max"],
        "core_speed_retention": core["speed_retention"] is not None
            and criteria["core_speed_ratio_range"][0] <= core["speed_retention"] <= criteria["core_speed_ratio_range"][1],
        "core_reference_beta": core["beta_rmse_deg"] is not None
            and core["beta_rmse_deg"] <= criteria["core_reference_beta_rmse_deg_max"],
        "any_wheel_every_physical_sample": contact["every_physical_sample_supported"]
            and contact["any_wheel_support_sample_fraction"] >= criteria["any_wheel_support_fraction_min"],
        "longest_no_support_gap": contact["longest_no_support_observed_s"] <= criteria["longest_no_support_interval_s_max"],
        "core_HF_reduction": ratio is not None
            and ratio <= criteria["core_10_to_100hz_angular_xy_rms_ratio_to_original_baseline_max"],
    }
    return {"criteria": copy.deepcopy(criteria), "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "numeric_nominal_episode_pass": bool(all(checks.values())),
            "core_HF_ratio_to_original_same_engine_PPO": ratio,
            "baseline": "Original PhysX-trained400Hz beta-ON policy evaluated in " + BACKENDS[engine],
            "core_beta_valid_samples": core["beta_valid_samples"], "core_native_samples": core["samples"],
            "qualitative_whole_body_review_complete": False,
            "overall_task_goal_complete_claimed": False}


def _check_new_report(report, proof, engine, protocol_path, raw_sha, report_sha):
    protocol = load_protocol(protocol_path)
    audit = report.get("ground_contact_v2_evaluation", {})
    expected = {"kind": evaluator.KIND, "cohort_id": evaluator.COHORT_ID,
        "reward_definition_version": evaluator.REWARD_VERSION, "training_backend": "physx",
        "smoke_only": False, "trained_policy_claimed": True, "additional_training_updates": 0,
        "status": "verified", "validation_backend": BACKENDS[engine], "protocol_sha256": sha256(protocol_path),
        "wrapper_sha256": sha256(evaluator.__file__), "normalizer_included_in_actor": True,
        "normalization_applications_per_inference": 1, "extra_normalization": False,
        "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
        "recording_sha256": raw_sha, "single_wheel_support_allowed": True,
        "bilateral_support_required": False, "absolute_height_gate": False, "airborne_failure": False}
    for key, value in expected.items():
        require(audit.get(key) == value, "NEW formal evaluator provenance differs: " + key)
    actor = _digest(audit.get("normalized_actor_sha256"), "new normalized actor")
    checkpoint = _digest(audit.get("checkpoint_sha256"), "new checkpoint")
    require(actor != BASELINE_ACTOR and checkpoint != BASELINE_CHECKPOINT,
            "Do not relabel the old baseline actor as the NEW contact experiment")
    for key in ("legacy_evaluator_sha256", "generated_source_sha256"):
        _digest(audit.get(key), key)
    ledger = audit.get("cohort_input_sha256", {})
    require(isinstance(ledger, dict) and ledger.get("model_final.pt") == checkpoint
            and ledger.get("policy.pt") == actor
            and {"ground_contact_v2_training_audit.json", "cohort_completion_audit.json", "manifest.json",
                 "policy_contract.json", "final_policy_contract.json", "run_status.json"} <= ledger.keys(),
            "Missing remotely verified formal training/export metadata bindings")
    for name, value in ledger.items(): _digest(value, name)
    expected_proof = {"status": "verified", "native_states": 2793, "policy_decisions": 349,
        "duration_s": 6.98, "physics_dt_s": .0025, "policy_dt_s": .02,
        "recording_sha256": raw_sha, "validation_backend": BACKENDS[engine],
        "checkpoint_sha256": checkpoint, "report_sha256": report_sha, "evaluation_mu": .7}
    for key, value in expected_proof.items():
        require(proof.get(key) == value, "Copied parent verification proof differs: " + key)
    contract = report.get("policy_export" if engine == "physx" else "contract", {})
    for key, value in {"physics_dt": .0025, "decimation": 8, "policy_fps": 50.,
        "actor_dim": 51, "critic_dim": 120, "action_dim": 8, "leg_kp": 500., "leg_kd": 10.,
        "full_horizon_diagnostic": True, "actor_includes_observation_normalizer": True,
        "training_environment_transitions": 12288000, "reward_definition_version": evaluator.REWARD_VERSION}.items():
        require(contract.get(key) == value, "New formal inference contract differs: " + key)
    profile, trained = contract.get("task_profile", {}), contract.get("trained_task_profile", {})
    compare = copy.deepcopy(profile); compare["contact_friction"] = 1.
    require(compare == trained and profile.get("contact_friction") == .7
            and profile.get("ground_contact_reward_v2") == protocol["ground_contact_reward_v2"]
            and profile.get("reward_weights", {}).get("sideslip") == 2.
            and profile.get("reward_weights", {}).get("grounded") == .4
            and profile.get("action_rate_weight") == -.5
            and profile.get("height_regularization", {}).get("mode") == "vertical_acceleration_only"
            and contract.get("training_domain_randomization") == protocol["training_randomization"],
            "New deployed/trained reward, DR, beta or height profile differs")
    require(contract.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") == .7,
            "Wrong nominal evaluation coefficient")
    material = report.get("contact_material_audit", {})
    if engine == "physx":
        gpu = audit.get("physical_gpu_index_requested")
        require(report.get("status") == "evaluated" and report.get("deterministic_start_frame") == 0
                and report.get("domain_randomization_enabled") is False and report.get("observation_noise_enabled") is False
                and report.get("terminal_states_captured_before_auto_reset") is True
                and report.get("runtime_device_audit", {}).get("status") == "verified"
                and type(gpu) is int and gpu in (3, 4, 5)
                and report["runtime_device_audit"].get("requested_device") == "cuda:"+str(gpu)
                and report.get("motion_file_sha256") == INPUT_PINS["motion"], "PhysX lifecycle/reference differs")
        _digest(audit.get("deployment_actor_sha256"), "remotely verified PhysX deployment actor")
        require(material.get("live_physx_materials_verified") is True and material.get("ground_material_usd_verified") is True
                and material.get("recipe", {}).get("training") is False
                and material.get("recipe", {}).get("wheel_contact_friction", {}).get("range") == [.7, .7]
                and material["recipe"]["wheel_contact_friction"].get("physx_ground_friction") == 1.
                and material.get("encoder_bias_range_rad") == [0., 0.], "PhysX fixed native materials differ")
    else:
        require(report.get("status") == "completed" and report.get("completed_full_reference") is True
                and report.get("policy_steps") == 349 and report.get("physics_steps") == 2792
                and report.get("hidden_resets") == 0 and report.get("root_state_writes") == 1
                and report.get("start_reference_frame") == 0
                and report.get("domain_parameters_resampled_during_episode") is False
                and report.get("policy_sha256") == actor and report.get("motion_sha256") == INPUT_PINS["motion"]
                and report.get("model_sha256") == INPUT_PINS["model"], "MuJoCo lifecycle/reference/model differs")
        require(material.get("native_mapping_verified") is True and material.get("explicit_contact_pair_count") == 0
                and material.get("pair_coefficient") == .7 and material.get("compiled_native_timestep_s") == .0025,
                "MuJoCo native material coefficient/clock differs")
        friction = np.asarray(material.get("geom_friction"), dtype=float)
        require(friction.ndim == 2 and friction.shape[1] == 3 and len(friction) > 0
                and np.isfinite(friction).all() and np.all(friction[:, 0] == .7), "MuJoCo compiled sliding coefficients differ")
        priorities = dict(zip(material.get("colliding_geom_ids", []), material.get("geom_priority", [])))
        wheels = material.get("wheel_geom_ids", [])
        require(priorities and wheels and all(priorities.get(index) == 1 for index in wheels)
                and all(value == (1 if index in wheels else 0) for index, value in priorities.items()),
                "MuJoCo effective wheel material priority differs")
    return contract, audit


def load_completed(folder, engine, protocol_path=DEFAULT_PROTOCOL):
    """Only a completed NEW formal episode; smoke/old/missing proofs fail closed."""
    require(engine in BACKENDS, "Select physx or mujoco explicitly")
    raw_name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
    paths = {name: _file(folder, name) for name in ("report.json", raw_name, "remote_verification.json")}
    hashes = {name: sha256(path) for name, path in paths.items()}
    report, proof = read_json(paths["report.json"]), read_json(paths["remote_verification.json"])
    contract, audit = _check_new_report(report, proof, engine, protocol_path,
        hashes[raw_name], hashes["report.json"])
    verify_recording(paths[raw_name], BACKENDS[engine])
    with np.load(paths[raw_name], allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    native_trace(raw, "policy_isaac" if engine == "physx" else "policy_mujoco")
    require(all(sha256(paths[name]) == value for name, value in hashes.items()),
            "Completed candidate files changed during offline analysis")
    return raw, {"input_directory": str(Path(folder).resolve()), "input_sha256": hashes,
        "checkpoint_sha256": audit["checkpoint_sha256"], "actor_sha256": audit["normalized_actor_sha256"],
        "contract": contract, "evaluation_proof": proof,
        "training_artifacts_verified_on_execution_host_not_locally_rehashed": True}


def load_baseline(engine, root=BASELINE_ROOT, summary_path=BASELINE_SUMMARY):
    require(engine in BACKENDS, "Select baseline engine explicitly")
    require(sha256(summary_path) == BASELINE_SUMMARY_SHA256, "Published original PPO baseline summary bytes differ")
    published = read_json(summary_path)
    row = next(item for item in published["rows"] if item["id"] == "physx_to_" + engine)
    require(row["training_backend"] == "physx" and row["validation_backend"] == BACKENDS[engine]
            and row["mu"] == .7 and row["checkpoint_sha256"] == BASELINE_CHECKPOINT
            and row["actor_sha256"] == BASELINE_ACTOR, "Wrong baseline training origin, beta actor or engine")
    folder = Path(root) / "cases/physx" / engine
    raw_name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
    hashes = {name: sha256(_file(folder, name)) for name in ("report.json", raw_name)}
    require(hashes["report.json"] == row["input_hashes"]["report"]
            and hashes[raw_name] == row["input_hashes"]["recording"],
            "Baseline report/native recording differs from published SHA bindings")
    verify_recording(folder/raw_name, BACKENDS[engine])
    report = read_json(folder/"report.json")
    contract = report["policy_export" if engine == "physx" else "contract"]
    require(contract["task_profile"]["reward_weights"]["sideslip"] == 2.
            and "ground_contact_reward_v2" not in contract["task_profile"], "Baseline is not original beta-ON contact reward")
    with np.load(folder/raw_name, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    return raw, {"published_summary_sha256": BASELINE_SUMMARY_SHA256, "published_row": row,
        "input_directory": str(folder.resolve()), "input_sha256": hashes, "contract": contract}


def _compare_profiles(new, baseline):
    candidate = copy.deepcopy(new["trained_task_profile"])
    candidate.pop("ground_contact_reward_v2", None)
    candidate["description"] = baseline["trained_task_profile"]["description"]
    require(candidate == baseline["trained_task_profile"], "Other existing PPO reward/profile parameters changed")
    for key in ("actor_dim", "critic_dim", "action_dim", "leg_kp", "leg_kd", "physics_dt", "decimation",
                "leg_position_scale", "wheel_torque_scale_nm", "raw_action_clip", "training_domain_randomization"):
        require(new[key] == baseline[key], "Baseline/new deployment ABI or DR differs: " + key)


def compact_arrays(traces, reference):
    result = {"reference_"+key: value.copy() for key, value in reference.items() if isinstance(value, np.ndarray)}
    result["case_ids"] = np.asarray(list(traces))
    for case, trace in traces.items():
        for name in (*TRACE_FIELDS, "floor_normal_force", "nonwheel_ground_force"):
            if name in trace: result[case+"__"+name] = np.asarray(trace[name]).copy()
    return result


def recompute_compact(arrays):
    reference = {key[len("reference_"):]: np.asarray(value) for key, value in arrays.items() if key.startswith("reference_")}
    result = {}
    for case in np.asarray(arrays["case_ids"]).tolist():
        trace = {name: np.asarray(arrays[case+"__"+name]) for name in TRACE_FIELDS}
        trace["physics_dt_s"] = .0025
        result[case] = analyze_trace(trace, reference, floor_normal_force=arrays[case+"__floor_normal_force"],
            nonwheel_ground_force=arrays.get(case+"__nonwheel_ground_force"))
    return result


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing analysis; use a fresh output directory")
    protocol = load_protocol(args.protocol)
    reference = metrics.reference_data(args.reference)
    folders = {engine: getattr(args, engine+"_dir", None) for engine in BACKENDS}
    require(any(value is not None for value in folders.values()), "At least one verified NEW completed PPO evaluation is required")
    rows, traces = [], {}
    for engine, folder in folders.items():
        if folder is None: continue
        raw, evidence = load_completed(folder, engine, args.protocol)
        old_raw, baseline_evidence = load_baseline(engine, args.baseline_root, args.baseline_summary)
        _compare_profiles(evidence["contract"], baseline_evidence["contract"])
        candidate, trace = recorded_metrics(raw, engine, reference)
        baseline, old_trace = recorded_metrics(old_raw, engine, reference)
        for name, original in (("core", "core"), ("whole_reference_overlap", "reference_overlap"), ("full_native", "full")):
            require(baseline["windows"][name] == baseline_evidence["published_row"]["windows"][original],
                    "Recomputed original same-engine baseline metrics differ from published result")
        initial = {key: float(np.max(np.abs(trace[key][0]-old_trace[key][0])))
                   for key in ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel", "wheel_pos")}
        traces["new_"+engine], traces["baseline_"+engine] = trace, old_trace
        rows.append({"id": "physx_trained_to_"+engine, "validation_backend": BACKENDS[engine], "sample_size": 1,
            "new_V2": candidate, "original_grounded_baseline": baseline,
            "nominal_screen": nominal_screen(candidate, baseline, protocol, engine),
            "input_evidence": evidence, "baseline_evidence": baseline_evidence,
            "initial_recorded_state_max_abs_difference": initial})
    if len(rows) == 2:
        require(rows[0]["input_evidence"]["actor_sha256"] == rows[1]["input_evidence"]["actor_sha256"]
                and rows[0]["input_evidence"]["checkpoint_sha256"] == rows[1]["input_evidence"]["checkpoint_sha256"],
                "Both physics backends must evaluate the SAME newly trained PPO actor")
    summary = {"schema_version": 1, "kind": "PPO_ground_contact_V2_nominal_offline_audit",
        "status": "completed_analysis", "protocol_sha256": sha256(args.protocol), "reference_sha256": sha256(args.reference),
        "analyzer_sha256": sha256(__file__), "rows": rows, "expected_validation_engines": list(BACKENDS),
        "completed_validation_engines": [engine for engine, folder in folders.items() if folder is not None],
        "both_PPO_validation_backends_complete": len(rows) == 2, "sample_size_per_case": 1,
        "physics_dt_s": .0025, "policy_dt_s": .02, "native_states_per_case": 2793,
        "new_dynamics_inference_training_or_FK": False, "height_or_flight_failure_gate": False,
        "bilateral_support_required": False, "qualitative_whole_body_review_complete": False,
        "overall_task_goal_complete_claimed": False,
        "limitations": ["N=1 per actor/engine: not statistical, hardware, or universal-friction qualification.",
            "Formal new training/export evidence is remotely parent-verified; weights are not fetched or executed here.",
            "Any-wheel support uses actual summed floor-normal Fz>5N, not total body net force or legacy thresholded point count.",
            "A single-wheel lift and alternating loaded wheels are allowed; no bilateral criterion is added.",
            "All physical samples stay on their original400Hz clocks; no alignment, smoothing, resets or state correction.",
            "The reference ends6.96s. Tracking over full native0–6.98s remains unavailable; no endpoint extrapolation.",
            "Reference-relative beta errors disclose valid samples; nonzero beta is intended, not a mismatch or zero-slip target.",
            "Contact, tilt and nonwheel-contact diagnostics do not change the independent XY failure metric.",
            "Passing a numerical episode screen does not complete the broader PPO/CEM comparison or qualitative whole-body review."]}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_states.npz", **compact_arrays(traces, reference))
    fields = ("id", "core_xy_rmse_m", "whole_xy_rmse_m", "whole_xy_peak_m", "core_speed_retention",
              "core_beta_rmse_deg", "core_beta_valid_samples", "any_support_sample_fraction",
              "unsupported_samples", "longest_gap_s", "core_HF_ratio", "numeric_episode_pass")
    with (output/"metric_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows:
            core = row["new_V2"]["windows"]["core"]["tracking"]
            whole = row["new_V2"]["windows"]["whole_reference_overlap"]["tracking"]
            support = row["new_V2"]["any_wheel_ground_support"]["full_contact"]
            writer.writerow(dict(zip(fields, (row["id"], core["xy_rmse_m"], whole["xy_rmse_m"], whole["xy_peak_m"],
                core["speed_retention"], core["beta_rmse_deg"], core["beta_valid_samples"],
                support["any_wheel_support_sample_fraction"], support["unsupported_physical_samples"],
                support["longest_no_support_observed_s"], row["nominal_screen"]["core_HF_ratio_to_original_same_engine_PPO"],
                row["nominal_screen"]["numeric_nominal_episode_pass"]))))
    summary["artifact_sha256"] = {name: sha256(output/name) for name in ("native_states.npz", "metric_table.csv")}
    write_json(output/"summary.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--physx-dir", type=Path)
    result.add_argument("--mujoco-dir", type=Path)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    result.add_argument("--baseline-summary", type=Path, default=BASELINE_SUMMARY)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())

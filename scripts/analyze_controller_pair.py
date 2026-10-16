"""Audit one fixed-mu, two-engine controller pair without stepping physics.

Heading offsets are planar geometric features, not mismatch or failure labels.
Only the predeclared XY screen is applied; coverage, posture and measured
contact-patch slip are separate diagnostics. No model assets are redistributed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT, verify_reference_pair
from controller_trace import WheelKinematics, native_trace
from nmpc_tracking_objective import JOINT_NAMES, quaternion_rotate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_mu(values, mu, label):
    values = np.asarray(values, dtype=float)
    _require(values.size > 0 and np.isfinite(values).all()
             and np.allclose(values, mu, atol=1e-6, rtol=0), f"{label}: fixed-mu readback mismatch")


def _hash_binding(report, key, actual, *, explicit=None):
    recorded = report.get(key)
    if recorded is not None:
        _require(recorded == actual, f"{key}: trace SHA-256 mismatch")
        if explicit is not None:
            _require(explicit == actual, "Explicit capture SHA-256 mismatch")
        return {"sha256": actual, "hash_source": "report", "report_key": key}
    _require(explicit is not None and explicit == actual,
             f"Report lacks {key}; supply its independently captured trace SHA-256 explicitly")
    return {"sha256": actual, "hash_source": "user_supplied_capture_sha256",
            "report_key": None, "scope": "File identity binding, not independent proof of measured physics"}


def _height_free(contract):
    return (isinstance(contract, dict) and contract.get("mode") == "vertical_acceleration_only"
            and contract.get("absolute_height_reward") is False
            and contract.get("height_termination") is False
            and contract.get("reference_vertical_velocity_reward") is False
            and contract.get("position_and_linear_velocity_tracking_axes") == ["x", "y"])


def audit_reports(algorithm, spec, isaac, mujoco, *, isaac_trace_sha256,
                  mujoco_trace_sha256, model_sha256, integration_only=False,
                  mujoco_capture_sha256=None):
    """Fail closed on trace identity, native-plant evidence and fixed material.

    Legacy MuJoCo RL reports did not hash their own rollout. Their Isaac
    observation-validation hash is NOT a MuJoCo trace hash. An explicit capture
    hash can bind these existing artifacts, with that narrower scope recorded.
    """
    _require(algorithm in ("PPO", "SAC", "NMPC"), "Unsupported algorithm")
    motion_hash = spec["motion_npz_sha256"]
    learned = algorithm != "NMPC"
    _require(isaac.get("simulator" if learned else "engine") in ("IsaacLab/PhysX", "Isaac/PhysX")
             and mujoco.get("engine") == "MuJoCo", "Reports must declare Isaac/PhysX and MuJoCo engines respectively")
    ia_contract = isaac.get("policy_export", {}) if learned else isaac.get("objective_contract", {})
    mj_contract = mujoco.get("contract", {}) if learned else mujoco.get("objective_contract", {})
    bindings = {
        "isaac": _hash_binding(isaac, "physics_trajectory_sha256" if learned else "rollout_sha256", isaac_trace_sha256),
        "mujoco": _hash_binding(mujoco, "rollout_sha256", mujoco_trace_sha256, explicit=mujoco_capture_sha256),
    }
    _require(isaac.get("motion_file_sha256" if learned else "motion_sha256") == motion_hash
             and mujoco.get("motion_sha256") == motion_hash, "Engine motion hashes differ from the selected locked motion")
    _require(mujoco.get("model_sha256") == model_sha256, "MuJoCo model hash differs from supplied FK model")
    if learned:
        _require(isaac.get("status") == "evaluated" and mujoco.get("status") == "completed", "Incomplete or failed learned-policy evaluation")
        _require(isaac.get("reference_state_initialization_during_evaluation") is False
                 and isaac.get("domain_randomization_enabled") is False
                 and isaac.get("observation_noise_enabled") is False
                 and isaac.get("terminal_states_captured_before_auto_reset") is True
                 and isaac.get("episodes_per_environment") == 1
                 and isaac.get("deterministic_start_frame") == 0,
                 "Isaac evaluation lacks deterministic first-episode/no-reset evidence")
        recording = isaac.get("physics_recording", {})
        _require(recording.get("terminal_states_before_reset") is True
                 and recording.get("same_counter_reset_and_forward_updates_excluded") is True
                 and recording.get("first_episode_only_valid_mask") is True,
                 "Isaac physical trace could include reset/forward samples")
        _require(mujoco.get("physics_stepped") is True and mujoco.get("root_state_writes") == 1
                 and mujoco.get("hidden_resets") == 0
                 and mujoco.get("domain_parameters_resampled_during_episode") is False,
                 "MuJoCo requires physical stepping, exactly one initial root write and no hidden resets")
        for engine, contract in (("Isaac", ia_contract), ("MuJoCo", mj_contract)):
            _require(contract.get("root_prescribed_during_steps") is False
                     and contract.get("motion_end_hidden_teleport") is False,
                     f"{engine}: root prescription or hidden teleport is unsupported")
            declared = contract.get("policy_algorithm", "PPO")
            _require(declared == algorithm, f"{engine}: policy algorithm mismatch")
        _require(ia_contract == mj_contract, "Engines must share the exact frozen policy observation/action/physics contract")
        for engine, report in (("Isaac", isaac), ("MuJoCo", mujoco)):
            _require(report.get("wheel_lateral_contact_slip_evidence_kind") == "actual_ground_contact_points"
                     and report.get("contact_slip_measurement", {}).get("ground_is_stationary") is True,
                     f"{engine}: actual ground-contact-point slip evidence is required, not inferred center velocity")
        checkpoint = isaac.get("checkpoint_sha256")
        _require(isinstance(checkpoint, str) and len(checkpoint) == 64
                 and ia_contract.get("checkpoint_sha256") == checkpoint
                 and mj_contract.get("checkpoint_sha256") == checkpoint,
                 "Engines must use the same frozen checkpoint identity")
        ia_mat, mj_mat = isaac.get("contact_material_audit", {}), mujoco.get("contact_material_audit", {})
        mu = ia_mat.get("expected_friction")
        _require(isinstance(mu, (int, float)) and np.isfinite(mu) and mu >= 0, "Invalid fixed mu")
        _same_mu(isaac.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction", []), mu, "Isaac override")
        _same_mu(ia_mat.get("robot_runtime_friction_range", []), mu, "Isaac robot")
        ground = ia_mat.get("ground_colliders", [])
        _require(bool(ground), "Missing Isaac ground material readback")
        for collider in ground:
            _same_mu([collider.get("static_friction", np.nan), collider.get("dynamic_friction", np.nan)], mu, "Isaac ground")
        _same_mu(mj_mat.get("expected_friction", []), mu, "MuJoCo expected material")
        _same_mu(mj_mat.get("compiled_friction_range", []), mu, "MuJoCo compiled material")
        height_free = (_height_free(ia_contract.get("height_regularization"))
                       and _height_free(mj_contract.get("height_regularization"))
                       and isaac.get("full_horizon_diagnostic") is True
                       and mujoco.get("full_horizon_diagnostic") is True)
        _require(integration_only or height_free, "Formal comparison requires height-free training and diagnostic evaluation")
    else:
        for engine, report in (("Isaac", isaac), ("MuJoCo", mujoco)):
            _require(report.get("status") in ("completed", "smoke_completed"), f"{engine}: incomplete NMPC rollout")
            _require(report.get("learned_policy_used") is False and report.get("tvlqr_used") is False
                     and report.get("root_state_prescribed_after_reset") is False
                     and report.get("external_root_forces_used") is False
                     and report.get("prediction_receives_plant_mu") is False,
                     f"{engine}: missing NMPC no-replay/no-prescription/no-oracle evidence")
            _require(report.get("action_replay", False) is False, f"{engine}: action replay is not NMPC")
            _require(report.get("reference_id") == spec["reference_id"], f"{engine}: reference_id mismatch")
            _require(integration_only or report.get("truncated_smoke") is False,
                     "Truncated NMPC smoke needs explicit --integration-only; it is not a full comparison")
        _require(isaac.get("mujoco_model_sha256") == model_sha256, "Isaac predictor model hash differs from FK model")
        _require(isaac.get("fixed_evaluation", {}).get("all_termination_terms_removed") is True,
                 "Isaac NMPC requires explicit removal of reset-triggering termination terms")
        for engine, report in (("Isaac", isaac), ("MuJoCo", mujoco)):
            _require(isinstance(report.get("contact_slip_definition"), str)
                     and "contact" in report["contact_slip_definition"].lower(),
                     f"{engine}: missing measured contact-slip definition")
        scenarios = np.asarray(isaac.get("prediction_mu_scenarios", []), dtype=float)
        _require(scenarios.ndim == 1 and scenarios.size > 0 and np.isfinite(scenarios).all()
                 and np.all(scenarios >= 0)
                 and np.array_equal(scenarios, np.asarray(mujoco.get("prediction_mu_scenarios", []))),
                 "Prediction scenarios must be frozen and equal across engines, not plant-mu oracle inputs")
        mu = isaac.get("plant_mu")
        _require(isinstance(mu, (int, float)) and np.isfinite(mu) and mu >= 0, "Invalid fixed mu")
        _same_mu(mujoco.get("plant_mu", []), mu, "MuJoCo plant")
        material = isaac.get("material_audit", {})
        _same_mu(material.get("expected_mu", []), mu, "Isaac expected material")
        _same_mu(material.get("robot_static_dynamic_range", []), mu, "Isaac robot")
        ground = material.get("ground_bound_materials", [])
        _require(bool(ground), "Missing Isaac ground material readback")
        for collider in ground:
            _same_mu(collider.get("coefficients", [])[:2], mu, "Isaac ground")
        _same_mu(mujoco.get("plant_material_readback", {}).get("compiled_sliding_friction_range", []), mu, "MuJoCo compiled material")
        height_free = all(c.get("absolute_height_target") is False
                          and c.get("reference_vertical_velocity_target") is False
                          and c.get("height_regularization", {}).get("mode") == "vertical_acceleration_only"
                          for c in (ia_contract, mj_contract))
        _require(height_free and isaac.get("pose_height_failure_gate") is False
                 and mujoco.get("pose_height_failure_gate") is False, "NMPC objective must exclude absolute-height targets/failure gates")
        for key in ("weights", "stds", "height_regularization", "dt_s", "reference_order"):
            _require(ia_contract.get(key) is not None and ia_contract.get(key) == mj_contract.get(key),
                     f"NMPC objective contract differs across engines: {key}")
    return {"fixed_mu": float(mu), "trace_bindings": bindings,
            "height_free_objective_and_evaluation": height_free,
            "scope": "Reported provenance and file-identity checks; not independent proof of simulator internals",
            "checkpoint_sha256": isaac.get("checkpoint_sha256") if learned else None,
            "integration_only": bool(integration_only), "eligible_for_controller_ranking": not integration_only}


def audit_training(manifest_path, algorithm, spec, isaac, mujoco, *, integration_only=False, checkpoint_path=None):
    """Hash actual checkpoint bytes (never deserialize) for a formal RL pair."""
    if algorithm == "NMPC":
        _require(manifest_path is None, "NMPC must not be initialized or certified by an RL training manifest")
        return None
    if manifest_path is None:
        _require(integration_only, "Formal PPO/SAC comparison requires --training-manifest")
        return {"verified_formal_training": False, "reason": "integration_only_without_training_manifest"}
    manifest_path = Path(manifest_path)
    training, status = read_json(manifest_path), read_json(manifest_path.with_name("run_status.json"))
    _require(training.get("algorithm") == algorithm, "Training algorithm differs from selected controller")
    _require(training.get("motion_sha256") == spec["motion_npz_sha256"], "Training motion differs from selected reference")
    if not integration_only:
        _require(status.get("status") == "training_completed" and status.get("trained_policy_claimed") is True
                 and training.get("smoke_only") is False, "Formal comparison requires completed, non-smoke training")
        _require(training.get("reference_id") == spec["reference_id"]
                 and training.get("reference_catalog_sha256") == sha256(DEFAULT_CATALOG)
                 and training.get("pooled_success_statistics") is False, "Missing selected-reference training provenance")
        _require(training.get("task_profile_sha256") == sha256(ROOT / "config/car_drift_height_smooth_task.json")
                 and training.get("task_profile") == read_json(ROOT / "config/car_drift_height_smooth_task.json")
                 and training.get("height_adapter_sha256") == sha256(ROOT / "scripts/height_smoothness.py")
                 and _height_free(training.get("height_regularization")), "Training profile/height adapter provenance mismatch")
        recipe = training.get("recipe", {})
        _require(recipe.get("friction_static") == [.3, 1.6]
                 and recipe.get("friction_dynamic") == [.3, 1.2]
                 and recipe.get("material_buckets") == 64, "Training lacks the shared BeyondMimic friction randomization recipe")
        for contract in (isaac.get("policy_export", {}), mujoco.get("contract", {})):
            _require(contract.get("trained_task_profile") == training.get("task_profile")
                     and contract.get("height_regularization") == training.get("height_regularization")
                     and contract.get("training_domain_randomization") == recipe,
                     "Evaluation contract differs from the height-free trained task/DR recipe")
            provenance = contract.get("scripted_actor_provenance")
            if provenance is not None:
                _require(provenance.get("algorithm") == algorithm
                         and provenance.get("manifest_sha256") == sha256(manifest_path)
                         and provenance.get("run_status_sha256") == sha256(manifest_path.with_name("run_status.json"))
                         and provenance.get("height_adapter_sha256") == training.get("height_adapter_sha256")
                         and provenance.get("smoke_only") is False
                         and provenance.get("trained_policy_claimed") is True,
                         "Scripted actor provenance differs from formal training artifacts")
    candidates = [Path(checkpoint_path)] if checkpoint_path is not None else [
        manifest_path.with_name("model_final.pt"), Path(isaac.get("checkpoint", "__missing_checkpoint__"))]
    checkpoint = next((path for path in candidates if path.is_file()), None)
    checkpoint_hash = sha256(checkpoint) if checkpoint is not None else None
    expected = isaac.get("checkpoint_sha256")
    _require(integration_only or checkpoint is not None, "Formal audit requires actual checkpoint bytes adjacent to manifest or --checkpoint")
    if checkpoint_hash is not None:
        _require(checkpoint_hash == expected, "Actual checkpoint SHA-256 differs from evaluated checkpoint")
    if status.get("checkpoint_sha256") is not None:
        _require(status["checkpoint_sha256"] == expected, "Completed status checkpoint SHA-256 differs from evaluated checkpoint")
    return {"verified_formal_training": not integration_only,
            "manifest_sha256": sha256(manifest_path), "run_status_sha256": sha256(manifest_path.with_name("run_status.json")),
            "status": status.get("status"), "algorithm": algorithm,
            "checkpoint_sha256": expected, "checkpoint_bytes_verified": checkpoint_hash is not None,
            "task_profile_sha256": training.get("task_profile_sha256"),
            "height_adapter_sha256": training.get("height_adapter_sha256"),
            "reference_catalog_sha256": training.get("reference_catalog_sha256")}


def reference_trace(motion):
    """Use named locked link poses, original COM velocities and all eight dq."""
    bodies = tuple(np.asarray(motion["body_names"]).tolist())
    names = tuple(np.asarray(motion["joint_names"]).tolist())
    _require(len(names) == 8 and set(names) == set(JOINT_NAMES), "Reference must name eight unique joints")
    base, wheels = bodies.index("base_Link"), [bodies.index(name) for name in ("wheel_L_Link", "wheel_R_Link")]
    order = [names.index(name) for name in JOINT_NAMES]
    return {"time_s": np.asarray(motion["time_s"], dtype=float), "physics_dt_s": 1. / float(np.asarray(motion["fps"]).reshape(-1)[0]),
            "joint_names": JOINT_NAMES, "position": motion["body_pos_w"][:, base],
            "quaternion": motion["body_quat_w"][:, base], "com_velocity": motion["body_lin_vel_w"][:, base],
            "angular_velocity": motion["body_ang_vel_w"][:, base], "joint_pos": motion["joint_pos"][:, order],
            "joint_vel": motion["joint_vel"][:, order], "wheel_pos": motion["body_pos_w"][:, wheels]}


@contextmanager
def source_model(source_root, model_path):
    """Import the caller's real evaluator and use its mesh-free model loader."""
    source_root = Path(source_root).resolve()
    relatives = ("scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py",
                 "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py")
    sources = {name: sha256(source_root / name) for name in relatives}
    saved_path = sys.path[:]
    prefixes = ("training", "eval_tracking", "_controller_pair_source_evaluator")
    saved = {key: module for key, module in sys.modules.items()
             if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)}
    for key in saved:
        del sys.modules[key]
    try:
        sys.path[:0] = [str(source_root / "scripts"), str(source_root)]
        module_spec = importlib.util.spec_from_file_location("_controller_pair_source_evaluator", source_root / relatives[0])
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model, removed = module.load_model(Path(model_path), no_visual_mesh=True)
        yield model, {"source_helpers_sha256": sources, "visual_geometries_removed": removed,
                      "loader": "eval_tracking_mujoco.load_model(no_visual_mesh=True)", "dynamics_stepping": False}
    finally:
        for key in list(sys.modules):
            if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = saved_path


def pair_xy_gap(isaac, mujoco, interval_s):
    """Unaligned world-XY difference, interpolated only on common native time."""
    start, end = interval_s
    covered = all(trace["time_s"][0] <= start + 1e-9 and trace["time_s"][-1] >= end - 1e-9 for trace in (isaac, mujoco))
    query = isaac["time_s"][(isaac["time_s"] >= start - 1e-9) & (isaac["time_s"] <= end + 1e-9)]
    result = {"window_s": list(interval_s), "coverage_complete": bool(covered and len(query) >= 2),
              "xy_rmse_m": None, "xy_peak_m": None, "samples": int(len(query)),
              "alignment": "same physical time/world frame; no translation, rotation, time warp or reference-error subtraction"}
    if result["coverage_complete"]:
        ip = np.column_stack([np.interp(query, isaac["time_s"], isaac["position"][:, j]) for j in range(2)])
        mp = np.column_stack([np.interp(query, mujoco["time_s"], mujoco["position"][:, j]) for j in range(2)])
        error = np.linalg.norm(ip-mp, axis=1)
        result.update(xy_rmse_m=float(np.sqrt(np.mean(error**2))), xy_peak_m=float(error.max()))
    return result


def posture_diagnostic(trace):
    quaternion = np.array(trace["quaternion"], dtype=float, copy=True)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    up = quaternion_rotate(quaternion, np.array([0., 0., 1.]))
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1., 1.)))
    return {"max_base_tilt_deg": float(tilt.max()), "fraction_tilt_over_60_deg": float(np.mean(tilt > 60.)),
            "interval": "entire recorded first episode, not only the declared core",
            "role": "Posture diagnostic, not a height/flight failure gate or controlled-execution certificate",
            "heading_geometry_caveat": "World-XY projected beta/gamma are not reward body-frame beta under pitch/roll; interpret large-tilt geometry cautiously"}


def reference_course_point_audit(motion, robot_reference, spec):
    """Explain, numerically, COM-course versus link-origin path descriptors.

    The catalog selection statistic is preserved. The comparison metrics use
    original unmerged base COM velocity, consistently across both engines.
    Transport is a diagnostic of the frozen arrays, not a new retarget.
    """
    trace = reference_trace(motion)
    time = trace["time_s"]
    reference_t = np.asarray(robot_reference["time_s"], dtype=float)
    _require(np.array_equal(time, reference_t), "Reference point audit requires identical locked sample clocks")
    gradient = np.gradient(np.asarray(robot_reference["root_target_position_m"], dtype=float), reference_t, axis=0)
    quaternion = np.array(trace["quaternion"], dtype=float, copy=True)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    world_offset = quaternion_rotate(quaternion, WheelKinematics.ROOT_COM)
    rotational_velocity = np.cross(trace["angular_velocity"], world_offset)
    transported = trace["com_velocity"] - rotational_velocity
    start, end = spec["evaluation_core_interval_s"]
    indices = np.flatnonzero((time >= start-1e-9) & (time <= end+1e-9))
    _require(len(indices) >= 2, "Reference point audit core lacks samples")
    endpoints = indices[[0, -1]]
    def course(value):
        angle = np.unwrap(np.arctan2(value[:, 1], value[:, 0]))
        return {"endpoint_course_deg": np.degrees(angle[endpoints]).tolist(),
                "net_course_change_deg": float(np.degrees(angle[endpoints[1]]-angle[endpoints[0]]))}
    return {"window_s": [start, end], "endpoint_frames": endpoints.tolist(),
            "endpoint_time_s": time[endpoints].tolist(),
            "metric_point": "Original unmerged base_Link COM; stored motion.body_lin_vel_w at named base_Link",
            "metric_com_course": course(trace["com_velocity"]),
            "selection_descriptor_point": "base_Link origin; np.gradient(robot_reference.root_target_position_m, robot_reference.time_s)",
            "root_origin_gradient_course": course(gradient),
            "com_transported_to_origin_course": course(transported),
            "transport_identity": "v_origin = v_original_link_COM - omega_world cross (R_link_to_world @ original_link_COM_offset)",
            "original_link_com_offset_m": WheelKinematics.ROOT_COM.tolist(),
            "transported_origin_vs_position_gradient_max_abs_component_m_s": float(np.max(np.abs(transported-gradient))),
            "catalog_core_course_turn_deg_unchanged": spec.get("core_course_turn_deg"),
            "interpretation": "Point definitions differ; catalog selection descriptor remains unchanged. COM-course is the controller metric, not a replacement of the frozen reference."}


def plant_source_provenance(report):
    """Keep reported implementation hashes without copying private source paths."""
    result = {key: report[key] for key in ("evaluator_source_sha256", "task_source_sha256",
              "drift_module_sha256", "contact_slip_helper_sha256", "engine_version", "mujoco_version") if key in report}
    if isinstance(report.get("source_sha256"), dict):
        result["reported_source_hashes"] = [
            {"source_name": Path(name).name, "sha256": digest}
            for name, digest in sorted(report["source_sha256"].items())]
    return result


def summarize_pair(spec, traces, reconstructions):
    reference = traces["reference"]
    speed = reconstructions["reference"]["signals"]["speed_m_s"]
    windows = [("core", spec["evaluation_core_interval_s"])] + [
        (f"supplementary_event_{i+1}", interval) for i, interval in enumerate(spec["supplementary_event_windows_s"])]
    result = {}
    for label, interval in windows:
        item = {"window_role": "predeclared_core" if label == "core" else "supplementary_descriptive_event",
                "reference": None, "isaac": None, "mujoco": None}
        for name, trace in traces.items():
            contacts = {} if name == "reference" else dict(
                wheel_ground_contact_count=trace["contact_count"], wheel_ground_force_w=trace["contact_force"],
                wheel_contact_lateral_slip_m_s=trace["contact_slip"])
            item[name] = summarize_turn(trace["time_s"], trace["position"], reconstructions[name]["signals"],
                                        reference["time_s"], reference["position"][:, :2], speed, interval, **contacts)
        item["sim2sim_xy_gap"] = pair_xy_gap(traces["isaac"], traces["mujoco"], interval)
        result[label] = item
    return result


def compact_arrays(traces, reconstructions):
    result = {"schema_version": np.array(1), "wheel_order": np.array(["left", "right"])}
    for label, trace in traces.items():
        for name, value in trace.items():
            result[f"{label}_{name}"] = np.asarray(value)
        for name in ("wheel_quaternion_wxyz", "wheel_center_velocity_w"):
            result[f"{label}_{name}"] = reconstructions[label][name]
        for name, value in reconstructions[label]["signals"].items():
            result[f"{label}_{name}"] = np.asarray(value)
    return result


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    interval = summary["reference_spec"]["evaluation_core_interval_s"]
    # Draw the same demonstrated source interval, never synthetic recovery/tail.
    span = summary["reference_spec"]["source_demo_interval_s"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True)
    all_xy = []
    rt = compact["reference_time_s"]
    ref_selected = (rt >= span[0]-1e-9) & (rt <= span[1]+1e-9)
    ref_xy = compact["reference_position"][ref_selected, :2]
    for ax, label, color in zip(axes, ("reference", "isaac", "mujoco"), ("#555555", "#0072B2", "#D55E00")):
        time, position = compact[f"{label}_time_s"], compact[f"{label}_position"]
        selected = (time >= span[0]-1e-9) & (time <= span[1]+1e-9)
        core = selected & (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
        ax.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color=".75", lw=1)
        ax.plot(position[selected, 0], position[selected, 1], color=color, lw=1, alpha=.5)
        ax.plot(position[core, 0], position[core, 1], color=color, lw=2.2)
        wheels = compact[f"{label}_wheel_pos"]
        heading = compact[f"{label}_wheel_rolling_plane_heading_rad"]
        valid = compact[f"{label}_root_course_to_wheel_valid"]
        indices = np.flatnonzero(selected)[::max(1, int(round(.4/float(compact[f"{label}_physics_dt_s"]))))]
        for side in range(2):
            ax.plot(wheels[selected, side, 0], wheels[selected, side, 1], color=color, lw=.55, alpha=.35)
            for index in indices:
                if not valid[index, side]:
                    continue
                contact = label == "reference" or (compact[f"{label}_contact_count"][index, side] > 0
                           and compact[f"{label}_contact_force"][index, side, 2] > 5.)
                delta = .055*np.array([np.cos(heading[index, side]), np.sin(heading[index, side])])
                segment = wheels[index, side, :2] + np.array([-delta, delta])
                ax.plot(segment[:, 0], segment[:, 1], color=color, alpha=.9 if contact else .15, lw=1.2)
        arrow_indices = indices[compact[f"{label}_root_beta_valid"][indices]]
        course = compact[f"{label}_course_rad"][arrow_indices]
        ax.quiver(position[arrow_indices, 0], position[arrow_indices, 1], np.cos(course), np.sin(course),
                  color=color, angles="xy", scale_units="xy", scale=7, width=.003)
        all_xy.extend((position[selected, :2], wheels[selected, :, :2].reshape(-1, 2), ref_xy))
        coverage = summary["windows"]["core"][label]["coverage_complete"]
        ax.set_title(("Kinematic reference" if label == "reference" else label.title()) + (" [core censored]" if not coverage else ""))
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.set_xlabel("World X [m]")
    xy = np.concatenate(all_xy)
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    margin = max(.15, .06*float(np.max(upper-lower)))
    axes[0].set_xlim(lower[0]-margin, upper[0]+margin); axes[0].set_ylim(lower[1]-margin, upper[1]+margin)
    axes[0].set_ylabel("World Y [m]")
    fig.suptitle(f"{summary['algorithm']} · {summary['reference_id']} · fixed μ={summary['fixed_mu']:g}"
                 + (" · integration only, not ranking" if summary["integration_only"] else " · single pair, not a success-rate claim"))
    fig.legend(handles=[Line2D([], [], color=".3", lw=2.2, label="Thick: declared core"),
                        Line2D([], [], color=".3", lw=1, label="Ticks: wheel rolling planes; arrows: XY velocity"),
                        Line2D([], [], color=".7", lw=1, label="Pale measured ticks: no gated wheel-ground contact")],
               loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94)); fig.savefig(output, dpi=160); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    _require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Preserving existing nonempty analysis output")
    spec = verify_reference_pair(args.motion_file, args.reference_file, args.reference_id)
    isaac, mujoco = read_json(args.isaac_report), read_json(args.mujoco_report)
    audit = audit_reports(args.algorithm, spec, isaac, mujoco,
                          isaac_trace_sha256=sha256(args.isaac_trace), mujoco_trace_sha256=sha256(args.mujoco_trace),
                          model_sha256=sha256(args.model), integration_only=args.integration_only,
                          mujoco_capture_sha256=args.mujoco_trace_sha256)
    training = audit_training(args.training_manifest, args.algorithm, spec, isaac, mujoco,
                              integration_only=args.integration_only, checkpoint_path=args.checkpoint)
    kinds = ("nmpc", "nmpc") if args.algorithm == "NMPC" else ("policy_isaac", "policy_mujoco")
    try:
        motion = load_arrays(args.motion_file)
        traces = {"reference": reference_trace(motion),
                  "isaac": native_trace(load_arrays(args.isaac_trace), kinds[0]),
                  "mujoco": native_trace(load_arrays(args.mujoco_trace), kinds[1])}
    except KeyError as error:
        raise ValueError(f"Unsupported old or incomplete physical trace; required measured field missing: {error}") from error
    with source_model(args.source_root, args.model) as (model, source_audit):
        fk = WheelKinematics(model)
        reconstructions = {name: fk.reconstruct(trace) for name, trace in traces.items()}
    windows = summarize_pair(spec, traces, reconstructions)
    core_complete = all(windows["core"][name]["coverage_complete"] for name in ("isaac", "mujoco"))
    summary = {"schema_version": 1, "kind": "offline_single_controller_fixed_mu_engine_pair",
               "algorithm": args.algorithm, "reference_id": spec["reference_id"], "reference_spec": spec,
               "fixed_mu": audit["fixed_mu"], "integration_only": args.integration_only,
               "eligible_for_controller_ranking": bool(not args.integration_only and core_complete),
               "core_coverage_complete": core_complete, "controlled_execution_success_claimed": False,
               "pooled_success_statistics": False, "hardware_ready": False,
               "height_or_flight_failure_gate": False, "heading_offset_is_mismatch": False,
               "heading_geometry_definition": "World-XY projected directed root beta and axial modulo-pi wheel gamma; not reward inverse-quaternion body-frame beta when pitched/rolled",
               "wheel_velocity_definition": "Analytic wheel link-origin center velocity from measured original base COM twist and all eight named joint velocities; not patch slip",
               "patch_slip_definition": "Recorded force-weighted ground-contact-point lateral-speed RMS; summarize only where contact count>0 and upward net wheel force>5 N",
               "reference_contact_semantics": "Kinematic only; no measured contact gate, no grounded or contact-slip inference",
               "reference_course_point_audit": reference_course_point_audit(motion, load_arrays(args.reference_file), spec),
               "audit": audit, "training_provenance": training, "model_loader_audit": source_audit,
               "reported_plant_source_provenance": {"isaac": plant_source_provenance(isaac), "mujoco": plant_source_provenance(mujoco)},
               "fk_audit": {name: result["audit"] for name, result in reconstructions.items()},
               "posture_diagnostics": {name: posture_diagnostic(trace) for name, trace in traces.items() if name != "reference"},
               "windows": windows, "input_sha256": {
                   name: sha256(getattr(args, name)) for name in ("motion_file", "reference_file", "model", "isaac_trace", "isaac_report", "mujoco_trace", "mujoco_report")},
               "analysis_source_sha256": {name: sha256(ROOT / "scripts" / name) for name in
                   ("analyze_controller_pair.py", "controller_metrics.py", "controller_trace.py", "controller_references.py", "nmpc_tracking_objective.py")},
               "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
               "limitations": ["One deterministic coefficient/algorithm/reference pair; no pooled success rates, controller ranking conclusion or calibrated tire/hardware claim.",
                               "Equal numeric friction is not identical contact physics across engines; contact force/slip comes from each engine, never reconstructed by FK.",
                               "Core coverage or small XY error alone does not demonstrate controlled upright execution. Large posture tilt remains a separate diagnostic.",
                               "The 0.15 m RMSE / 0.30 m peak XY limits are exploratory screens, not hardware tolerances; beta/gamma offsets are not mismatch gates."]}
    compact = compact_arrays(traces, reconstructions)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "pair_trace.npz", **compact)
    summary["artifacts"] = {"compact_trace": "pair_trace.npz", "compact_trace_sha256": sha256(output / "pair_trace.npz"),
                            "topdown": "topdown.png", "replay": "NPZ contains native clocks, named states, FK wheel geometry, validity masks and original measured contacts; plot with render_topdown"}
    render_topdown(compact, summary, output / "topdown.png")
    summary["artifacts"]["topdown_sha256"] = sha256(output / "topdown.png")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--algorithm", required=True, choices=("PPO", "SAC", "NMPC"))
    result.add_argument("--reference-id", default=PRIMARY_REFERENCE_ID)
    for name in ("motion-file", "reference-file", "model", "source-root", "isaac-trace", "isaac-report", "mujoco-trace", "mujoco-report", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--training-manifest", type=Path, help="Required for formal PPO/SAC; sibling run_status.json and actual checkpoint bytes are audited")
    result.add_argument("--checkpoint", type=Path, help="Optional explicit checkpoint path; default manifest-adjacent model_final.pt, then reported path")
    result.add_argument("--mujoco-trace-sha256", help="Independently captured hash for legacy reports lacking rollout_sha256; never use Isaac observation-validation hash")
    result.add_argument("--integration-only", action="store_true", help="Explicit smoke/partial opt-in; never eligible for controller ranking")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = run(arguments)
    print(json.dumps({"status": "analyzed", "reference_id": result["reference_id"], "core_coverage_complete": result["core_coverage_complete"], "output_dir": str(arguments.output_dir)}))

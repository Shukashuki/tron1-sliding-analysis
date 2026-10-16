"""Offline, separate SA29 fixed-NMPC nominal DEVELOPMENT pilot.

Audit one complete original 258-decision MuJoCo recording. Reconstruct wheel
geometry with FK only; retain measured velocities/contacts and original clocks.
No physics, IK, retargeting, optimization, policy inference or qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_controller_pair import (load_arrays, plant_source_provenance,
    posture_diagnostic, read_json, reference_trace, sha256, source_model)
from analyze_nmpc_friction_development import (PILOT_CEM,
    audit_header as audit_fixed_header)
from analyze_nmpc_nominal_development import (configuration_sha256,
    contact_diagnostic, require, safe_configuration, source_digest)
from analyze_tvlqr_pilot import public_compact
from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, ROOT, get_reference, verify_reference_pair
from controller_trace import WheelKinematics, native_trace
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, HeightSmoothTrackingObjective
from nmpc_validation_contract import episode_evidence


REFERENCE_ID = "doa_sa29_scale2"
FRAMES, SAMPLES = 258, 1033
PREDECLARATION = ROOT/"config/nmpc_sa29_nominal_development.json"
PREDECLARATION_SHA256 = "3346fddd93e3312535d9b24899370f657e06ef1938cf9484e29e4eacd3579aa8"
CORE = [0., 3.14]
EVENTS = [[.78, 1.54], [1.88, 3.04]]
FK_HELPERS = ("scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py",
    "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py")


def audit_predeclaration(manifest=None):
    """Require the original one-case predeclaration, not a new tuning contract."""
    require(sha256(PREDECLARATION) == PREDECLARATION_SHA256,
            "SA29 predeclaration byte identity changed")
    locked = read_json(PREDECLARATION)
    manifest = locked if manifest is None else manifest
    require(manifest == locked, "SA29 predeclaration differs from the locked manifest")
    require(manifest["reference_id"] == REFERENCE_ID and manifest["role"] == "separate_supplement"
            and manifest["stratum"] == "high_sideslip_s_avoidance"
            and manifest["intended_local_episodes"] == 1 and len(manifest["source_sha256"]) == 21,
            "Require the separate, one-episode SA29 scope")
    evaluation = manifest["evaluation"]
    require(evaluation["frames"] == FRAMES and evaluation["physical_samples"] == SAMPLES
            and evaluation["duration_s"] == 5.16 and evaluation["control_dt_s"] == .02
            and evaluation["physics_dt_s"] == .005 and evaluation["core_window_s"] == CORE
            and evaluation["supplementary_event_windows_s"] == EVENTS
            and evaluation["events_are_descriptive_not_gates"] is True,
            "SA29 predeclared clock/window contract differs")
    return manifest


def audit_header(report, spec, *, model_sha256, trace_sha256, solver_sha256, declaration=None):
    """Hash-bind SA29 reference, unchanged runtime and common 15-term objective."""
    declaration = audit_predeclaration(declaration)
    require(spec == get_reference(REFERENCE_ID) and spec["frames"] == FRAMES,
            "Only the locked SA29 supplement may be analyzed; primary relabeling is forbidden")
    header = audit_fixed_header(report, .7, spec, model_sha256)
    require(report.get("status") == "completed"
            and report.get("role") == spec["role"] and report.get("stratum") == spec["stratum"]
            and report.get("pooled_success_statistics") is False,
            "Require a completed, separately labeled supplementary recording")
    require(report.get("rollout_sha256") == trace_sha256 and report.get("solver_log_sha256") == solver_sha256,
            "Original trace/solver-log hash binding differs")
    require(all(report.get(name) == FRAMES for name in (
        "completed_policy_steps", "evaluated_policy_steps", "total_reference_policy_steps"))
            and np.isclose(report.get("duration_s", np.nan), 5.16, atol=1e-10, rtol=0),
            "Require all 258 decisions through 5.16 s; partial cases are not substituted")
    require(report.get("action_replay", False) is False and report.get("isaac_result") is False,
            "No replay or fabricated Isaac evidence is permitted")
    require(model_sha256 == declaration["input_sha256"]["model"]
            and report["motion_sha256"] == declaration["input_sha256"]["motion"]
            and spec["robot_reference_npz_sha256"] == declaration["input_sha256"]["robot_reference"]
            and configuration_sha256(report["effective_cem_config"])
                == declaration["effective_cem_configuration_sha256"],
            "SA29 input or unchanged fixed-controller identity differs")
    require(report.get("settings") == read_json(ROOT/declaration["settings"]),
            "Original settings content differs; H30/B10 are explicit fixed overrides, not a rewritten settings file")
    parameterization = report.get("action_parameterization", {})
    require(parameterization.get("mode") == "warm_start_residual_blocks"
            and parameterization.get("baseline_copied_and_fixed_per_solve") is True
            and parameterization.get("learned_baseline") is False
            and report.get("contact_threshold_n") == 5.
            and report.get("wheel_lateral_contact_slip_evidence_kind") == "actual_ground_contact_points",
            "Residual decoder or measured contact-point contract differs")
    expected_sources = declaration["source_sha256"]
    for relative, expected in expected_sources.items():
        require(sha256(ROOT/relative) == expected and source_digest(report, relative) == expected,
                "Changed predeclared on-disk/reported runtime source: "+relative)
    actuator = declaration["input_sha256"]["actuator_contract"]
    require(source_digest(report, "policy_contract.json") == actuator
            and sorted(report["source_sha256"].values()) == sorted([*expected_sources.values(), actuator]),
            "Original 22-source runtime/actuator bindings differ")
    task = read_json(ROOT/declaration["task_profile"])
    objective = HeightSmoothTrackingObjective(task).contract()
    require(report.get("objective_contract") == objective and len(objective["weights"]) == 15,
            "The exact common height-smooth 15-term objective differs")
    require(report.get("packed_state_contract", {}).get("dimension") == 111
            and report["packed_state_contract"].get("qpos_dimension") == 15
            and report["packed_state_contract"].get("qvel_dimension") == 14
            and report["packed_state_contract"].get("contact_force_history_shape") == [3, 9, 3],
            "Recorded physical/contact-memory state contract differs")
    require(header["native_identity"]["native_library_sha256"]
            == declaration["input_sha256"]["native_library"], "Different unchanged native predictor")
    return {**header, "runtime_source_sha256": dict(expected_sources),
        "actuator_contract_sha256": actuator, "actual_plant_mu": .7,
        "plant_material_readback": safe_configuration(report["plant_material_readback"]),
        "prediction_mu_scenarios": [.7], "complete_physical_evidence": True,
        "censored": False, "no_retuning": True}


def _finite(raw, name, shape):
    value = np.asarray(raw[name])
    require(value.shape == shape and value.dtype.kind in "biuf" and np.isfinite(value).all(),
            "Missing/invalid complete SA29 field: "+name)
    return value


def audit_trace(raw, solver, spec):
    """SA29-specific native clock/actions/state audit; never forge 349 samples."""
    require(spec == get_reference(REFERENCE_ID), "Require the locked SA29 reference spec")
    require(isinstance(solver, list) and len(solver) == FRAMES, "Require all 258 original decisions")
    require(all(value.dtype.kind not in "biufc" or np.isfinite(value).all()
                for value in raw.values()), "Nonfinite numeric physical evidence")
    trace = native_trace(raw, "nmpc")
    require(len(trace["time_s"]) == SAMPLES and trace["physics_dt_s"] == .005
            and np.asarray(raw["policy_dt_s"]).shape == () and float(raw["policy_dt_s"]) == .02
            and np.allclose(trace["time_s"], np.arange(SAMPLES)*.005, atol=1e-10, rtol=0),
            "Require the consecutive 1033-sample 0..5.16 s native clock")
    require(tuple(np.asarray(raw["body_names"]).tolist()) == BODY_NAMES,
            "Physical body/contact-history order differs")
    for field, expected in (("reference_id", REFERENCE_ID), ("reference_role", spec["role"]),
        ("reference_stratum", spec["stratum"])):
        require(np.asarray(raw[field]).shape == () and str(raw[field]) == expected,
                "Native SA29 reference label differs: "+field)
    require(np.array_equal(raw["reference_evaluation_core_interval_s"], CORE)
            and np.array_equal(raw["reference_supplementary_event_windows_s"], EVENTS),
            "Native SA29 descriptive windows differ")
    require(np.allclose(_finite(raw, "reference_time_s", (FRAMES,)), np.arange(FRAMES)*.02,
                        atol=1e-10, rtol=0), "Saved reference clock differs")
    _finite(raw, "reference_root_xy_m", (FRAMES, 2))
    actions = _finite(raw, "actions", (FRAMES, 8))
    require(np.max(np.abs(actions)) <= 1. and np.array_equal(actions, actions.astype(np.float32).astype(float)),
            "Require the actual bounded float32 action interface")
    held = _finite(raw, "held_action", (SAMPLES, 8))
    require(np.array_equal(held, np.vstack((np.zeros((1, 8)), np.repeat(actions, 4, axis=0))))
            and np.array_equal(_finite(raw, "reference_frame_for_action", (SAMPLES,)),
                np.r_[0, np.repeat(np.arange(FRAMES), 4)]),
            "Held-action/reference phase must match four physical substeps per decision")
    applied = _finite(raw, "action_was_applied", (SAMPLES,))
    require(applied.dtype.kind == "b" and np.array_equal(applied, np.r_[False, np.ones(SAMPLES-1, dtype=bool)]),
            "The initial sample alone must have no applied action")
    for k, row in enumerate(solver):
        require(row.get("frame") == k and row.get("reference_frame_for_reward") == k
                and np.isclose(row.get("time_s", np.nan), (k+1)*.02, atol=1e-10, rtol=0)
                and np.array_equal(row.get("action"), actions[k]),
                "Original decision action/held-pre-action target/clock differs")
    for name, width in (("qpos", 15), ("qvel", 14)):
        _finite(raw, name, (SAMPLES, width))
    require(np.allclose(raw["qpos"][:, :3], trace["position"], atol=1e-10, rtol=0)
            and np.allclose(np.abs(np.sum(raw["qpos"][:, 3:7]*trace["quaternion"], axis=1)),
                            1., atol=1e-5, rtol=0), "Physical qpos and measured root pose differ")
    names = tuple(np.asarray(raw["qpos_joint_names"]).tolist())
    require(len(names) == 8 and set(names) == set(JOINT_NAMES), "Physical joint coordinate names differ")
    order = np.array([names.index(name) for name in JOINT_NAMES])
    require(np.allclose(raw["qpos"][:, 7+order], trace["joint_pos"], atol=1e-10, rtol=0)
            and np.allclose(raw["qvel"][:, 6+order], trace["joint_vel"], atol=1e-10, rtol=0),
            "Physical qpos/qvel and named measured joints differ")
    require(np.all(trace["contact_count"] >= 0)
            and np.array_equal(trace["contact_count"], np.round(trace["contact_count"]))
            and np.all(trace["contact_slip"] >= 0), "Invalid measured wheel contact/slip evidence")
    for name in ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"):
        value = _finite(raw, name, (SAMPLES,))
        require(np.all(value >= 0) and (not name.endswith("count") or np.array_equal(value, np.round(value))),
                "Invalid measured nonwheel contact evidence")
        trace[name] = value.copy()
    trace["held_action"] = held.copy()
    return trace


def solver_diagnostic(report, raw, solver):
    """Audit original 258 solve timings/counters and all 15 logged score terms."""
    require(isinstance(solver, list) and len(solver) == FRAMES, "Require 258 original solver records")
    compute = _finite(raw, "compute_s", (FRAMES,))
    require(np.all(compute >= 0) and np.array_equal(compute, [row.get("compute_s") for row in solver])
            and np.allclose(report.get("compute_ms_median_p95_max", []),
                1000*np.quantile(compute, [.5, .95, 1.]), atol=1e-8, rtol=1e-12)
            and report.get("compute_over_control_dt_fraction") == float(np.mean(compute > .02))
            and report.get("real_time_verified") is False, "Original solve timing trace/log/report differs")
    wall = report.get("wall_s")
    require(type(wall) in (int, float) and np.isfinite(wall) and wall >= 0, "Invalid recorded rollout wall time")
    count_names = ("candidate_evaluations", "finite_candidates", "failed_candidates", "scenario_evaluations",
        "failed_scenarios", "batch_prediction_calls", "failed_batch_prediction_calls")
    flag_names = ("optimization_failed", "fallback_used", "warm_start_used", "initial_candidate_generation_failed")
    totals = {name: 0 for name in (*count_names, *flag_names)}
    scopes = set(); weights = report["objective_contract"]["weights"]
    require(len(weights) == 15 and all(type(value) in (int, float) and np.isfinite(value) for value in weights.values()),
            "Require the finite common 15-term weights")
    upper = sum(value for value in weights.values() if value > 0)
    stage = _finite(raw, "stage_cost", (FRAMES,)); previous = np.zeros(8)
    for k, row in enumerate(solver):
        info, score = row.get("solver", {}), row.get("objective", {})
        for name in count_names:
            value = info.get(name)
            require(type(value) is int and value >= 0, "Invalid original solver counter: "+name)
            totals[name] += value
        for name in flag_names:
            require(type(info.get(name)) is bool, "Invalid original solver flag: "+name)
            totals[name] += int(info[name])
        require(info["finite_candidates"]+info["failed_candidates"] == info["candidate_evaluations"]
                == info["scenario_evaluations"] and info["failed_scenarios"] == info["failed_candidates"]
                and info["failed_batch_prediction_calls"] <= info["batch_prediction_calls"]
                and info["fallback_used"] == info["optimization_failed"], "Original solver counts/fallbacks differ")
        require(info.get("optimizer") == "bounded_cem_shooting" and info.get("horizon_steps") == 30
                and info.get("control_blocks") == 10 and info.get("prediction_mu_scenarios") == [.7]
                and info.get("action_parameterization") == "warm_start_residual_blocks"
                and info.get("residual_baseline_fixed_per_solve") is True
                and info.get("reference_convention") == "held_pre_action"
                and info.get("population_evaluation_backend") == "predictor_rollout_batch"
                and info.get("control_deadline_enforced") is False and info.get("real_time_verified") is False
                and info.get("robustness_certificate") is False and info.get("learned_policy_fallback") is False,
                "Original solve controller/latency contract differs")
        require(isinstance(info.get("timing_scope"), str), "Missing original solve timing scope")
        scopes.add(safe_configuration(info["timing_scope"]))
        command = np.asarray(row.get("raw_cem_action", []), dtype=float)
        action = np.asarray(raw["actions"][k])
        require(command.shape == (8,) and np.isfinite(command).all() and np.max(np.abs(command)) <= 1.+1e-9
                and np.array_equal(command.astype(np.float32).astype(float), action),
                "Raw CEM command does not reproduce the applied float32 action")
        if info["fallback_used"]:
            require(np.array_equal(action, previous), "Previous-action fallback does not match physical history")
        terms, weighted = score.get("raw_terms", {}), score.get("weighted_terms", {})
        require(set(terms) == set(weighted) == set(weights)
                and all(type(value) in (int, float) and np.isfinite(value)
                        for value in (*terms.values(), *weighted.values())), "Missing/nonfinite 15-term score logs")
        require(all(np.isclose(weighted[name], weights[name]*terms[name], atol=1e-12, rtol=1e-12) for name in weights)
                and np.isclose(score.get("total_reward", np.nan), sum(weighted.values()), atol=1e-12, rtol=1e-12)
                and score.get("reward_upper_bound") == upper
                and stage[k] == score.get("stage_cost")
                and np.isclose(stage[k], .02*(upper-score["total_reward"]), atol=1e-12, rtol=1e-12),
                "Logged 15-term arithmetic/stage-cost differs")
        require(np.isclose(terms["action_rate_l2"], np.sum((action-previous)**2), atol=1e-12, rtol=1e-12)
                and score.get("base_com_vz") == float(raw["base_com_velocity_w"][(k+1)*4, 2]),
                "Score action-rate or measured vertical-velocity memory differs")
        scale = report["objective_contract"]["height_regularization"]["acceleration_scale_m_s2"]
        vz = raw["base_com_velocity_w"][:, 2]
        acceleration = 0. if k == 0 else float(-np.expm1(-((vz[(k+1)*4]-vz[k*4])/(.02*scale))**2))
        require(np.isclose(terms["height_vertical_acceleration"], acceleration, atol=1e-12, rtol=1e-12),
                "Height smoothing must use measured acceleration, not absolute height")
        previous = action
    require(totals["optimization_failed"] == report.get("solver_failures")
            and np.isclose(report.get("total_stage_cost", np.nan), stage.sum(), atol=1e-9, rtol=1e-12),
            "Original solver-failure/stage-cost aggregate differs")
    return {"decision_samples": FRAMES, "counts": totals, "common_score_terms": 15,
        "common_score_arithmetic_verified": True, "raw_float64_to_applied_float32_verified": True,
        "finite_fallbacks_retained_as_descriptive_outcomes": True,
        "timing": {"sample_count": FRAMES, "compute_ms_median_p95_max": (1000*np.quantile(compute, [.5,.95,1.])).tolist(),
            "fraction_above_20ms": float(np.mean(compute > .02)), "reported_rollout_wall_s": wall,
            "reported_solver_timing_scopes": sorted(scopes), "deadline_enforced": False,
            "real_time_verified": False, "actuation_delay_injected": False,
            "scope": "Saved controller.action wall times, not matched hardware or real-time evidence; plant always advances four 5ms substeps independent of solve wall time."}}


def summarize_windows(traces, recon, spec):
    require(spec == get_reference(REFERENCE_ID) and set(traces) == set(recon) == {"reference", "NMPC"},
            "SA29 contains only one kinematic reference and one physical NMPC recording")
    reference = traces["reference"]
    def metrics(interval):
        result = {}
        for label, trace in traces.items():
            contact = {} if label == "reference" else dict(wheel_ground_contact_count=trace["contact_count"],
                wheel_ground_force_w=trace["contact_force"], wheel_contact_lateral_slip_m_s=trace["contact_slip"])
            result[label] = summarize_turn(trace["time_s"], trace["position"], recon[label]["signals"],
                reference["time_s"], reference["position"][:, :2], recon["reference"]["signals"]["speed_m_s"], interval, **contact)
        return result
    return {"core": metrics(CORE), "supplementary_events": [
        {"interval_s": interval, "descriptive_not_gate": True, "metrics": metrics(interval)} for interval in EVENTS]}


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(12., 5.4), sharex=True, sharey=True)
    reference = compact["reference_position"][:, :2]; bounds = [reference]
    for axis, label, color in zip(axes, ("reference", "NMPC"), (".35", "#2ca02c")):
        time, pos, wheels = [compact[label+"_"+name] for name in ("time_s", "position", "wheel_pos")]
        core = (time >= CORE[0]-1e-9) & (time <= CORE[1]+1e-9)
        axis.plot(reference[:, 0], reference[:, 1], "--", color=".75", lw=1.)
        axis.plot(pos[:, 0], pos[:, 1], color=color, lw=.8, alpha=.35)
        axis.plot(pos[core, 0], pos[core, 1], color=color, lw=2.)
        indices = np.arange(0, len(time), max(1, round(.4/float(compact[label+"_physics_dt_s"]))))
        for side in range(2):
            axis.plot(wheels[:, side, 0], wheels[:, side, 1], color=color, lw=.5, alpha=.2)
            for index in indices:
                if not compact[label+"_root_course_to_wheel_valid"][index, side]: continue
                angle = compact[label+"_wheel_rolling_plane_heading_rad"][index, side]
                delta = .055*np.array([np.cos(angle), np.sin(angle)])
                segment = wheels[index, side, :2]+np.array([-delta, delta])
                grounded = label == "reference" or (compact[label+"_contact_count"][index, side] > 0
                    and compact[label+"_contact_force"][index, side, 2] > 5.)
                axis.plot(segment[:, 0], segment[:, 1], color=color, lw=1.2, alpha=.9 if grounded else .15)
        arrows = indices[compact[label+"_root_beta_valid"][indices]]
        velocity = compact[label+"_com_velocity"][arrows, :2]
        axis.quiver(pos[arrows, 0], pos[arrows, 1], velocity[:, 0], velocity[:, 1], color=color,
            angles="xy", scale_units="xy", scale=7, width=.003)
        bounds.extend((pos[:, :2], wheels[:, :, :2].reshape(-1, 2)))
        axis.set(title="Kinematic reference · contacts UNMEASURED" if label == "reference" else "NMPC · MuJoCo μ=0.7 · measured contacts",
            xlabel="World X [m]", ylabel="World Y [m]")
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
    xy = np.concatenate(bounds); low, high = xy.min(axis=0), xy.max(axis=0)
    margin = max(.15, .06*float(np.max(high-low)))
    axes[0].set_xlim(low[0]-margin, high[0]+margin); axes[0].set_ylim(low[1]-margin, high[1]+margin)
    fig.suptitle("SA29 S-avoidance · separate nominal DEVELOPMENT pilot · not a large-angle-turn replicate", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=".3", lw=2., label="Thick: core 0–3.14 s; thin: full playback"),
        Line2D([], [], color=".3", lw=1., label="COM velocity arrows; wheel-plane ticks (pale: no eligible contact)")],
        loc="lower center", ncol=1, fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, .94)); fig.savefig(output, dpi=160); plt.close(fig)


def run(args):
    output, directory = Path(args.output_dir), Path(args.run_dir)
    require(not output.exists(), "Preserve existing output; choose a NEW directory")
    declaration = audit_predeclaration()
    spec = verify_reference_pair(args.motion_file, args.reference_file, REFERENCE_ID)
    hashes = {name: sha256(directory/name) for name in ("report.json", "rollout.npz", "solver_log.json")}
    report, raw = read_json(directory/"report.json"), load_arrays(directory/"rollout.npz")
    solver = json.loads((directory/"solver_log.json").read_text())
    audit = audit_header(report, spec, model_sha256=sha256(args.model),
        trace_sha256=hashes["rollout.npz"], solver_sha256=hashes["solver_log.json"], declaration=declaration)
    evidence = episode_evidence(directory, "mujoco", spec,
        {"effective_cem_config": PILOT_CEM, "prediction_mu_scenarios": [.7]}, .7)
    trace = audit_trace(raw, solver, spec); diagnostic = solver_diagnostic(report, raw, solver)
    motion = load_arrays(args.motion_file); reference = reference_trace(motion)
    require(np.array_equal(raw["reference_root_xy_m"], reference["position"][:, :2]),
            "Recorded reference XY differs from the locked SA29 motion")
    traces = {"reference": reference, "NMPC": trace}
    with source_model(args.source_root, args.model) as (model, loader):
        require(loader["source_helpers_sha256"] == audit["fk_source_helpers_sha256"],
                "Actual FK loader bytes differ from the original recording helpers")
        fk = WheelKinematics(model); recon = {label: fk.reconstruct(value) for label, value in traces.items()}
    windows = summarize_windows(traces, recon, spec)
    summary = {"schema_version": 1, "kind": "separate_sa29_mujoco_fixed_nmpc_nominal_development",
        "status": "development_analysis_only", "engine": "MuJoCo", "reference_id": REFERENCE_ID,
        "reference_spec": spec, "role": spec["role"], "stratum": spec["stratum"],
        "predeclaration_sha256": PREDECLARATION_SHA256, "intended_local_episodes": 1,
        "completed_physical_episodes": 1, "frames": FRAMES, "physical_samples": SAMPLES,
        "duration_s": 5.16, "physics_dt_s": .005, "control_dt_s": .02,
        "actual_plant_mu": .7, "prediction_mu_scenarios": [.7],
        "effective_cem_config": report["effective_cem_config"],
        "effective_cem_config_sha256": configuration_sha256(report["effective_cem_config"]),
        "run": {**audit, "input_sha256": evidence["hashes"], "solver_diagnostics": diagnostic,
            "reported_source_provenance": plant_source_provenance(report)},
        "windows": windows, "events_are_descriptive_not_success_gates": True,
        "posture_diagnostics": {"NMPC": posture_diagnostic(trace)},
        "posture_scope": "Full 5.16 s playback; not a core-only fall-time assertion",
        "contact_diagnostics": {"core": contact_diagnostic(trace, CORE), "supplementary_events": [
            {"interval_s": interval, "descriptive_not_gate": True, "diagnostic": contact_diagnostic(trace, interval)} for interval in EVENTS]},
        "objective_contract": safe_configuration(report["objective_contract"]),
        "model_loader_audit": loader, "fk_audit": {label: value["audit"] for label, value in recon.items()},
        "simulation_budget": {"actual_plant_physics_substeps": SAMPLES-1,
            "requested_prediction_substeps": diagnostic["counts"]["scenario_evaluations"]*30*4,
            "failed_prediction_requests_prove_completed_steps": False},
        "input_sha256": {"motion": sha256(args.motion_file), "robot_reference": sha256(args.reference_file), "model": sha256(args.model)},
        "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
        **{name: False for name in ("pooled_success_statistics", "primary_task_replicate", "sim2sim_comparison",
            "isaac_evidence_available", "supplementary_policy_evidence_available", "friction_grid_complete",
            "two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed", "frozen_benchmark",
            "eligible_for_controller_ranking", "robustness_certificate", "real_time_verified", "controlled_execution_success_claimed",
            "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "inverse_kinematics",
            "retargeting", "learned_inference", "actuation_delay_injected", "controller_retuned")},
        "missing_evidence": declaration["missing_evidence"],
        "limitations": ["One fixed local nominal SA29 recording, not a friction grid, paired Isaac validation or trained supplementary policy.",
            "SA29 is CARLA S-avoidance with reconstructed source timing; not physical-car capture or a second large-angle-turn trial.",
            "Original clocks and world axes retained; no alignment/time warp. COM speed is measured velocity, not finite-difference root speed.",
            "Reference contacts are UNMEASURED. Height, flight, tilt, heading separation and contact slip are descriptive, not added failure gates.",
            "Full playback has synthetic recovery/stationary tail; only the locked 0..3.14 s source-demo core is the XY screen.",
            "No robust-stability, hardware, matched-compute, real-time or controller-ranking claim."],
        "analysis_source_sha256": {name: sha256(ROOT/"scripts"/name) for name in (
            "analyze_nmpc_sa29_pilot.py", "analyze_nmpc_friction_development.py", "analyze_nmpc_nominal_development.py",
            "analyze_tvlqr_pilot.py", "analyze_controller_pair.py", "controller_metrics.py", "controller_trace.py",
            "controller_references.py", "nmpc_validation_contract.py", "nmpc_tracking_objective.py")}}
    compact = public_compact(traces, recon); output.mkdir(parents=True)
    np.savez_compressed(output/"pilot_trace.npz", **compact); render_topdown(compact, summary, output/"topdown.png")
    summary["artifacts"] = {name: {"file": filename, "sha256": sha256(output/filename)}
        for name, filename in (("compact_trace", "pilot_trace.npz"), ("topdown", "topdown.png"))}
    safe_configuration(summary)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--motion", "--motion-file", dest="motion_file", type=Path, required=True)
    for name in ("reference-file", "model", "source-root", "run-dir", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "reference_id": result["reference_id"],
        "core_xy_rmse_m": result["windows"]["core"]["NMPC"]["reference_xy_rmse_m"]}))

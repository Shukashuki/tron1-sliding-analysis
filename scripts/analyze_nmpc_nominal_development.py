"""Offline nominal MuJoCo-only DEVELOPMENT comparison, not sim2sim or frozen.

Consume 2+ completed primary NMPC episodes; never run physics, inference,
optimization, retargeting or alignment. Reuse canonical measured traces, FK,
heading/contact masks and XY screens from the common controller analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from analyze_controller_pair import (
    compact_arrays, load_arrays, plant_source_provenance, posture_diagnostic,
    read_json, reference_trace, sha256, source_model, _model_generated_nominal_plan_identity,
)
from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT, verify_reference_pair
from controller_trace import WheelKinematics, native_trace
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES


LEGACY_CEM_SHA256 = "6b5d25f6985e700043ed47dbfdb58c244a2cb708a15d0e9d35f5ddf57d7449b4"
PARAMETERIZATIONS = ("absolute_blocks", "warm_start_residual_blocks")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_configuration(value):
    """Copy declared knobs, rejecting nonportable paths/nonfinite JSON values."""
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "Configuration keys must be strings")
        return {key: safe_configuration(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_configuration(item) for item in value]
    if isinstance(value, str):
        require(not value.startswith(("/", "~", "file:", "\\"))
                and not any(token in value for token in ("/home/", "/data/", "C:\\", "D:\\")),
                "Configuration contains a private/absolute source path")
        return value
    require(value is None or isinstance(value, (bool, int, float)), "Unsupported configuration value")
    if isinstance(value, float):
        require(np.isfinite(value), "Configuration contains nonfinite values")
    return value


def configuration_sha256(value):
    encoded = json.dumps(safe_configuration(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def source_digest(report, suffix):
    values = [value for name, value in report.get("source_sha256", {}).items()
              if name == suffix or name.endswith("/" + suffix)]
    require(len(values) == 1 and isinstance(values[0], str)
            and re.fullmatch(r"[0-9a-f]{64}", values[0]) is not None,
            "Missing/ambiguous reported source identity: " + suffix)
    return values[0]


def parameterization_audit(report):
    """Do not infer a new decoder from a label or silently relabel legacy CEM."""
    config = report["effective_cem_config"]
    declarations = {name: safe_configuration(value) for name, value in (
        ("report.parameterization", report.get("parameterization")),
        ("report.action_parameterization", report.get("action_parameterization")),
        ("effective_cem_config.parameterization", config.get("parameterization")),
    ) if value is not None}
    if declarations:
        require(all(not isinstance(value, dict) or value.get("learned_baseline", False) is False
                    for value in declarations.values()), "Learned baseline is outside nominal development scope")
        modes = [value.get("mode", value.get("parameterization")) if isinstance(value, dict) else value
                 for value in declarations.values()]
        require(all(mode in PARAMETERIZATIONS for mode in modes) and len(set(modes)) == 1,
                "Missing, conflicting or unsupported explicit parameterization")
        return {"mode": modes[0], "source": "explicit_reported_configuration", "declarations": declarations}
    digest = source_digest(report, "scripts/nmpc_shooting.py")
    require(digest == LEGACY_CEM_SHA256,
            "Absent parameterization is supported only for the known legacy absolute-block CEM SHA-256")
    return {"mode": "absolute_blocks", "source": "known_legacy_reported_cem_source_sha256",
            "reported_cem_source_sha256": digest,
            "scope": "Reported source identity allowlist, not independent proof of original execution"}


def parse_runs(values):
    require(len(values) >= 2, "Provide at least two --run LABEL=DIR arguments")
    result = {}
    for value in values:
        label, separator, directory = value.partition("=")
        require(separator and directory and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", label),
                "Run syntax must be portable LABEL=DIR, with an alphanumeric/hyphen/underscore label")
        require(label.lower() != "reference" and label not in result, "Duplicate/reserved run label")
        path = Path(directory)
        require(path.is_dir(), "Run directory does not exist: " + label)
        result[label] = path
    require(len({path.resolve() for path in result.values()}) == len(result), "Run directories must be distinct")
    return result


def audit_report(report, spec, *, model_sha256, trace_sha256, solver_sha256):
    require(report.get("kind") == "full_body_height_smooth_cem_nmpc"
            and report.get("engine") == "MuJoCo" and report.get("status") == "completed",
            "Require a completed full-body MuJoCo NMPC episode")
    require(report.get("reference_id") == PRIMARY_REFERENCE_ID == spec["reference_id"]
            and report.get("motion_sha256") == spec["motion_npz_sha256"]
            and report.get("model_sha256") == model_sha256
            and report.get("reference_catalog_sha256") == sha256(DEFAULT_CATALOG), "Model/motion/reference identity mismatch")
    require(report.get("rollout_sha256") == trace_sha256 and report.get("solver_log_sha256") == solver_sha256,
            "Report-bound trace/solver-log SHA-256 mismatch")
    require(report.get("plant_mu") == .7 and report.get("prediction_mu_scenarios") == [.7],
            "Development analysis is nominal mu=.7 with prediction scenarios [.7] ONLY")
    material = np.asarray(report.get("plant_material_readback", {}).get("compiled_sliding_friction_range", []), dtype=float)
    require(material.shape == (2,) and np.allclose(material, [.7, .7], atol=1e-12, rtol=0),
            "Missing/incorrect compiled nominal plant material readback")
    for name in ("learned_policy_used", "tvlqr_used", "old_qr_used", "root_state_prescribed_after_reset",
                 "external_root_forces_used", "prediction_receives_plant_mu", "truncated_smoke", "pose_height_failure_gate"):
        require(report.get(name) is False, "Unsupported or missing development contract: " + name)
    require(report.get("action_replay", False) is False, "Action replay is not a controller ablation")
    require(report.get("completed_policy_steps") == report.get("evaluated_policy_steps") == report.get("total_reference_policy_steps") == 349
            and np.isclose(report.get("duration_s", np.nan), 6.98, atol=1e-10, rtol=0)
            and report.get("physics_dt_s") == .005 and report.get("control_dt_s") == .02,
            "Require all 349 decisions and original .005/.02 s clocks")
    objective = report.get("objective_contract", {})
    require(objective.get("absolute_height_target") is False
            and objective.get("reference_vertical_velocity_target") is False
            and objective.get("height_regularization", {}).get("mode") == "vertical_acceleration_only"
            and objective.get("dt_s") == .02
            and objective.get("body_names") == list(BODY_NAMES) and objective.get("joint_names") == list(JOINT_NAMES),
            "Missing height-free named full-body objective contract")
    require(isinstance(objective.get("weights"), dict) and len(objective["weights"]) == 15
            and isinstance(objective.get("stds"), dict) and isinstance(objective.get("reference_order"), str),
            "Require the complete common 15-term held-reference objective")
    require(isinstance(report.get("contact_slip_definition"), str)
            and "contact" in report["contact_slip_definition"].lower(), "Missing measured contact-slip contract")
    packed = report.get("packed_state_contract", {})
    require(packed.get("dimension") == 111 and packed.get("qpos_dimension") == 15
            and packed.get("qvel_dimension") == 14 and packed.get("contact_force_history_shape") == [3, 9, 3],
            "Unsupported physical/contact-memory predictor state contract")
    config, settings = safe_configuration(report.get("effective_cem_config")), safe_configuration(report.get("settings"))
    require(isinstance(config, dict) and isinstance(settings, dict), "Missing effective CEM settings")
    for name in ("horizon_steps", "control_blocks", "candidates", "iterations", "elite_count"):
        require(type(config.get(name)) is int and config[name] > 0, "Invalid effective CEM knob: " + name)
    require(config["control_blocks"] <= config["horizon_steps"] and 1 <= config["elite_count"] <= config["candidates"]
            and config["candidates"] >= 2 and config.get("reference_convention") == "held_pre_action",
            "Invalid blocking/population/reference convention")
    require(settings.get("physics_dt_s") == .005 and settings.get("policy_dt_s") == .02,
            "Settings clocks differ from actuator/control clocks")
    assignment = np.arange(config["horizon_steps"])*config["control_blocks"]//config["horizon_steps"]
    return {"effective_cem_config": config, "effective_cem_config_sha256": configuration_sha256(config),
            "reported_settings": settings, "reported_settings_sha256": configuration_sha256(settings),
            "configuration_hash_definition": "SHA-256 of sorted compact UTF-8 JSON; not a settings-file byte hash",
            "parameterization": parameterization_audit(report),
            "model_generated_nominal_plan": safe_configuration(
                _model_generated_nominal_plan_identity(report, report)),
            "prediction_horizon_s": config["horizon_steps"]*.02,
            "control_block_durations_s": (np.bincount(assignment)*.02).tolist(),
            "actuator_identity": {"policy_contract_sha256": source_digest(report, "policy_contract.json"),
                                  "actuator_helper_sha256": source_digest(report, "scripts/classical_tracking.py"),
                                  "actuator_setup_helper_sha256": source_digest(report, "scripts/eval_tracking_mujoco.py"),
                                  "physics_dt_s": .005, "control_dt_s": .02},
            "packed_state_contract": safe_configuration(packed),
            "evidence_scope": "Hash-bound recorded reports/traces; source declarations are not independent proof of original simulator internals"}


def audit_trace(raw, solver, spec):
    trace = native_trace(raw, "nmpc")
    require(len(trace["time_s"]) == 1397 and float(raw["policy_dt_s"]) == .02 and trace["physics_dt_s"] == .005,
            "Require exactly 1397 original physical samples")
    require(np.allclose(trace["time_s"], np.arange(1397)*.005, rtol=0, atol=1e-10),
            "Native clock must run continuously from zero through 6.98 s")
    require(tuple(np.asarray(raw["body_names"]).tolist()) == BODY_NAMES, "Physical body/history order mismatch")
    require(len(solver) == 349 and np.asarray(raw["actions"]).shape == (349, 8), "Missing full decision/action log")
    for k, decision in enumerate(solver):
        require(decision.get("frame") == k and decision.get("reference_frame_for_reward") == k
                and np.isclose(decision.get("time_s", np.nan), (k+1)*.02, rtol=0, atol=1e-10),
                "Reward target must be held at reference[k], never a reset-spliced/advanced phase")
    actions = np.asarray(raw["actions"], dtype=float)
    require(np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.
            and np.array_equal(actions, actions.astype(np.float32).astype(float)), "Require applied float32 normalized actions")
    held = np.asarray(raw["held_action"])
    require(held.shape == (1397, 8) and np.isfinite(held).all() and np.array_equal(held[0], np.zeros(8))
            and np.array_equal(held[1:], np.repeat(actions, 4, axis=0)),
            "Held actions must match four physical substeps per decision")
    require(all(np.array_equal(decision.get("action"), action) for decision, action in zip(solver, actions)),
            "Decision-log action differs from physically held action")
    applied = np.asarray(raw["action_was_applied"])
    require(np.array_equal(raw["reference_frame_for_action"][1:], np.repeat(np.arange(349), 4))
            and applied.dtype.kind == "b" and np.array_equal(applied, np.r_[False, np.ones(1396, dtype=bool)]),
            "Physical action/phase sampling mismatch")
    for name in ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"):
        value = np.asarray(raw[name], dtype=float)
        require(value.shape == (1397,) and np.isfinite(value).all() and np.all(value >= 0),
                "Missing measured nonwheel-ground evidence: " + name)
        if name.endswith("count"):
            require(np.array_equal(value, np.round(value)), "Nonwheel contact counts must be integers")
        trace[name] = value.copy()
    trace["held_action"] = held.copy()
    return trace


def contact_diagnostic(trace, interval):
    time = trace["time_s"]
    core = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
    active = trace["nonwheel_ground_contact_count"] > 0
    return {"role": "Measured ground-contact diagnostic, NOT a height/flight failure gate",
            "full_clip_nonwheel_ground_fraction": float(active.mean()),
            "core_nonwheel_ground_fraction": float(active[core].mean()),
            "first_nonwheel_ground_time_s": float(time[np.flatnonzero(active)[0]]) if active.any() else None,
            "maximum_nonwheel_ground_force_norm_n": float(trace["nonwheel_ground_contact_force_norm_n"].max())}


def solver_diagnostic(report, raw, solver, parameterization):
    """Validate saved solve timing and aggregate bounded original counters only."""
    compute = np.asarray(raw["compute_s"], dtype=float)
    reported = np.asarray(report.get("compute_ms_median_p95_max", []), dtype=float)
    require(compute.shape == (349,) and np.isfinite(compute).all() and np.all(compute >= 0)
            and reported.shape == (3,) and np.isfinite(reported).all()
            and np.allclose(reported, 1000*np.quantile(compute, [.5, .95, 1.]), atol=1e-8, rtol=1e-12),
            "Saved report timing quantiles differ from original 349 solve samples")
    require(np.array_equal(compute, np.asarray([row.get("compute_s", np.nan) for row in solver])),
            "Physical trace and solver-log compute times differ")
    above = float(np.mean(compute > .02))
    require(report.get("compute_over_control_dt_fraction") == above and report.get("real_time_verified") is False,
            "Saved deadline fraction/real-time declaration mismatch")
    wall = report.get("wall_s")
    require(type(wall) in (int, float) and np.isfinite(wall) and wall >= 0, "Invalid recorded rollout wall time")
    count_names = ("candidate_evaluations", "finite_candidates", "failed_candidates", "scenario_evaluations",
                   "failed_scenarios", "batch_prediction_calls", "failed_batch_prediction_calls")
    flag_names = ("optimization_failed", "fallback_used", "warm_start_used", "initial_candidate_generation_failed")
    totals = {name: 0 for name in (*count_names, *flag_names)}
    scopes = set()
    for row in solver:
        info = row.get("solver", {})
        for name in count_names:
            value = info.get(name)
            require(type(value) is int and value >= 0, "Missing/invalid solver counter: " + name)
            totals[name] += value
        for name in flag_names:
            require(type(info.get(name)) is bool, "Missing/invalid solver flag: " + name)
            totals[name] += int(info[name])
        require(info["finite_candidates"]+info["failed_candidates"] == info["candidate_evaluations"]
                and info["scenario_evaluations"] == info["candidate_evaluations"]
                and info["failed_scenarios"] <= info["scenario_evaluations"]
                and info["failed_batch_prediction_calls"] <= info["batch_prediction_calls"],
                "Inconsistent nominal candidate/scenario/batch counters")
        require(info.get("control_deadline_enforced") is False and info.get("real_time_verified") is False,
                "Solver timing must not be a deadline-enforced/real-time execution claim")
        require(isinstance(info.get("timing_scope"), str), "Missing original solver timing scope")
        scopes.add(safe_configuration(info["timing_scope"]))
        if "action_parameterization" in info:
            require(info["action_parameterization"] == parameterization["mode"], "Per-step parameterization differs from declared mode")
    require(totals["optimization_failed"] == report.get("solver_failures"), "Saved solver-failure total differs from decision log")
    optional = {}
    for name in ("initial_baseline_block_projection_max_abs_error", "residual_parameter_clip_fraction",
                 "residual_action_decode_clip_fraction", "residual_selected_parameter_linf"):
        present = [row["solver"][name] for row in solver if name in row["solver"]]
        values = [value for value in present if value is not None]
        require(all(type(value) in (int, float) and np.isfinite(value) and value >= 0 for value in values),
                "Invalid optional residual diagnostic: " + name)
        if name.endswith("fraction"):
            require(all(value <= 1 for value in values), "Residual clip fraction outside [0,1]")
        optional[name] = {"present_steps": len(present), "finite_value_steps": len(values),
                          "null_steps": len(present)-len(values), "missing_steps": len(solver)-len(present),
                          "mean": float(np.mean(values)) if values else None,
                          "median_p95_max": np.quantile(values, [.5, .95, 1.]).tolist() if values else None}
    flags = [row["solver"]["residual_baseline_fixed_per_solve"] for row in solver
             if "residual_baseline_fixed_per_solve" in row["solver"]]
    require(all(type(value) is bool and value == (parameterization["mode"] == "warm_start_residual_blocks") for value in flags),
            "Residual fixed-baseline flag differs from declared mode")
    return {"decision_samples": 349, "counts": totals,
            "optional_residual_diagnostics": optional,
            "optional_diagnostics_scope": "Original logs only; projection is of the winning INITIAL baseline, not final selected plan. Missing/null values are never imputed as zero.",
            "residual_fixed_baseline_flag": {"present_steps": len(flags), "true_steps": sum(flags), "missing_steps": 349-len(flags)},
            "timing": {"compute_ms_median_p95_max": reported.tolist(), "fraction_above_20ms": above,
                       "all_calls_above_20ms": bool(np.all(compute > .02)), "sample_count": 349,
                       "reported_solver_timing_scopes": sorted(scopes),
                       "scope": "Complete recorded controller.action including native worker IPC/solver bookkeeping; plant/logging/model/worker startup excluded",
                       "recorded_rollout_wall_s": wall, "real_time_verified": False,
                       "comparison_scope": "Saved original-run timings, not a controlled hardware microbenchmark",
                       "deadline_enforced": False, "actuation_delay_injected": False,
                       "delay_scope": "Physical trace advances exactly four 5ms samples per decision, irrespective of solve wall time; no injected wall-latency delay model"}}


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = ["reference", *summary["run_order"]]
    columns = 2 if len(labels) == 4 else min(3, len(labels))
    rows = math.ceil(len(labels)/columns)
    fig, grid = plt.subplots(rows, columns, figsize=(4.7*columns, 4.9*rows), sharex=True, sharey=True, squeeze=False)
    axes = grid.ravel(); spec = summary["reference_spec"]
    start, end = spec["source_demo_interval_s"]; lower_core, upper_core = spec["evaluation_core_interval_s"]
    rt = compact["reference_time_s"]
    reference = compact["reference_position"][(rt >= start-1e-9) & (rt <= end+1e-9), :2]
    bounds = [reference]
    for j, (axis, label) in enumerate(zip(axes, labels)):
        color = ".35" if label == "reference" else plt.get_cmap("tab10")((j-1) % 10)
        time, position, wheels = [compact[f"{label}_{field}"] for field in ("time_s", "position", "wheel_pos")]
        selected = (time >= start-1e-9) & (time <= end+1e-9)
        core = selected & (time >= lower_core-1e-9) & (time <= upper_core+1e-9)
        axis.plot(reference[:, 0], reference[:, 1], "--", color=".75", lw=1)
        axis.plot(position[selected, 0], position[selected, 1], color=color, lw=1, alpha=.5)
        axis.plot(position[core, 0], position[core, 1], color=color, lw=2.2)
        indices = np.flatnonzero(selected)[::max(1, round(.4/float(compact[f"{label}_physics_dt_s"])))]
        heading, valid = [compact[f"{label}_{field}"] for field in ("wheel_rolling_plane_heading_rad", "root_course_to_wheel_valid")]
        for side in range(2):
            axis.plot(wheels[selected, side, 0], wheels[selected, side, 1], color=color, lw=.55, alpha=.35)
            for index in indices:
                if not valid[index, side]:
                    continue
                contact = label == "reference" or (compact[f"{label}_contact_count"][index, side] > 0
                           and compact[f"{label}_contact_force"][index, side, 2] > 5.)
                delta = .055*np.array([np.cos(heading[index, side]), np.sin(heading[index, side])])
                segment = wheels[index, side, :2] + np.array([-delta, delta])
                axis.plot(segment[:, 0], segment[:, 1], color=color, alpha=.9 if contact else .15, lw=1.2)
        arrows = indices[compact[f"{label}_root_beta_valid"][indices]]
        course = compact[f"{label}_course_rad"][arrows]
        axis.quiver(position[arrows, 0], position[arrows, 1], np.cos(course), np.sin(course),
                    color=color, angles="xy", scale_units="xy", scale=7, width=.003)
        bounds.extend((position[selected, :2], wheels[selected, :, :2].reshape(-1, 2)))
        axis.set(title="Kinematic reference" if label == "reference" else "MuJoCo: " + label,
                 xlabel="World X [m]", ylabel="World Y [m]")
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
    xy = np.concatenate(bounds); lo, hi = xy.min(axis=0), xy.max(axis=0)
    margin = max(.15, .06*float(np.max(hi-lo)))
    axes[0].set_xlim(lo[0]-margin, hi[0]+margin); axes[0].set_ylim(lo[1]-margin, hi[1]+margin)
    for axis in axes[len(labels):]: axis.set_visible(False)
    fig.suptitle("MuJoCo-only DEVELOPMENT · nominal μ=0.7 · not sim2sim / not frozen", fontsize=12)
    fig.legend(handles=[Line2D([], [], color=".3", lw=2.2, label="Thick: declared core"),
                        Line2D([], [], color=".3", lw=1, label="Ticks: wheel rolling planes; arrows: XY velocity"),
                        Line2D([], [], color=".7", lw=1, label="Pale measured ticks: no gated wheel-ground contact")],
               loc="lower center", bbox_to_anchor=(.5, .015), ncol=1 if columns == 2 else 3, fontsize=8)
    fig.tight_layout(rect=(0, .12 if columns == 2 else .10, 1, .94))
    fig.savefig(output, dpi=160); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserving existing output; choose a NEW output directory")
    directories = parse_runs(args.run)
    spec = verify_reference_pair(args.motion_file, args.reference_file, PRIMARY_REFERENCE_ID)
    require(spec["frames"] == 349 and spec["fps"] == 50., "Require the full primary reference clock")
    traces = {"reference": reference_trace(load_arrays(args.motion_file))}
    audits, reports = {}, {}
    for label, directory in directories.items():
        files = {name: directory / name for name in ("report.json", "rollout.npz", "solver_log.json")}
        report = read_json(files["report.json"])
        digest = {name: sha256(path) for name, path in files.items()}
        audit = audit_report(report, spec, model_sha256=sha256(args.model),
                             trace_sha256=digest["rollout.npz"], solver_sha256=digest["solver_log.json"])
        solver = json.loads(files["solver_log.json"].read_text())
        require(isinstance(solver, list), "Solver log must contain a decision list")
        try:
            raw = load_arrays(files["rollout.npz"])
            trace = audit_trace(raw, solver, spec)
            diagnostics = solver_diagnostic(report, raw, solver, audit["parameterization"])
        except KeyError as error:
            raise ValueError(f"Incomplete original physical trace: required measured field {error}") from error
        for earlier in reports.values():
            require(report["objective_contract"] == earlier["objective_contract"], "Development variants must share the EXACT objective contract")
        if audits:
            require(audit["actuator_identity"] == next(iter(audits.values()))["actuator_identity"],
                    "Development variants must share actuator-source identity and clocks")
        traces[label], reports[label] = trace, report
        audits[label] = {**audit, "input_sha256": digest,
                         "solver_diagnostics": diagnostics,
                         "reported_plant_source_provenance": plant_source_provenance(report)}
    with source_model(args.source_root, args.model) as (model, loader_audit):
        fk = WheelKinematics(model)
        reconstructions = {label: fk.reconstruct(trace) for label, trace in traces.items()}
    reference, interval = traces["reference"], spec["evaluation_core_interval_s"]
    windows = {"core": {}}
    for label, trace in traces.items():
        contacts = {} if label == "reference" else dict(wheel_ground_contact_count=trace["contact_count"],
                    wheel_ground_force_w=trace["contact_force"], wheel_contact_lateral_slip_m_s=trace["contact_slip"])
        windows["core"][label] = summarize_turn(trace["time_s"], trace["position"], reconstructions[label]["signals"],
            reference["time_s"], reference["position"][:, :2], reconstructions["reference"]["signals"]["speed_m_s"], interval, **contacts)
    summary = {"schema_version": 1, "kind": "mujoco_only_nominal_nmpc_development_ablation",
               "status": "development_analysis_only", "engine": "MuJoCo", "fixed_mu": .7,
               "reference_id": PRIMARY_REFERENCE_ID, "reference_spec": spec, "run_order": list(directories),
               "sim2sim_comparison": False, "frozen_benchmark": False, "eligible_for_controller_ranking": False,
               "controlled_execution_success_claimed": False, "pooled_success_statistics": False,
               "height_or_flight_failure_gate": False, "heading_offset_is_mismatch": False,
               "dynamics_stepping": False, "learned_inference": False,
               "objective_contract": safe_configuration(next(iter(reports.values()))["objective_contract"]),
               "runs": audits, "windows": windows, "model_loader_audit": loader_audit,
               "fk_audit": {label: result["audit"] for label, result in reconstructions.items()},
               "posture_diagnostics": {label: posture_diagnostic(traces[label]) for label in directories},
               "contact_diagnostics": {label: contact_diagnostic(traces[label], interval) for label in directories},
               "input_sha256": {name: sha256(getattr(args, name)) for name in ("motion_file", "reference_file", "model")},
               "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
               "analysis_source_sha256": {name: sha256(ROOT / "scripts" / name) for name in (
                   "analyze_nmpc_nominal_development.py", "analyze_controller_pair.py", "controller_metrics.py",
                   "controller_trace.py", "controller_references.py", "nmpc_tracking_objective.py")},
               "limitations": ["MuJoCo-only DEVELOPMENT variants, not a sim2sim comparison, frozen controller, statistical success rate or hardware result.",
                   "Horizon and block duration may change together; parameterization/search knobs are preserved, not treated as a causal horizon-only experiment.",
                   "Full-clip tilt and measured nonwheel contacts are separate diagnostics. XY screens alone do not certify upright controlled execution.",
                   "Reference contacts are unmeasured; no grounded/patch-slip labels are inferred by FK. Heading offsets are features, not mismatch gates.",
                   "World axes and native clocks are unchanged; no alignment/time warp. The figure shows the demonstrated interval, not synthetic recovery/tail."]}
    compact = compact_arrays(traces, reconstructions)
    output.mkdir(parents=True)
    np.savez_compressed(output / "development_trace.npz", **compact)
    render_topdown(compact, summary, output / "topdown.png")
    summary["artifacts"] = {"compact_trace": "development_trace.npz", "compact_trace_sha256": sha256(output / "development_trace.npz"),
                            "topdown": "topdown.png", "topdown_sha256": sha256(output / "topdown.png"),
                            "replay": "Use render_topdown with development_trace.npz and this summary; arrays retain native named states, masks and measured contacts"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("motion-file", "reference-file", "model", "source-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR", help="Repeat for 2+ completed nominal MuJoCo development variants")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = run(arguments)
    print(json.dumps({"status": result["status"], "runs": result["run_order"], "sim2sim": False, "output_dir": str(arguments.output_dir)}))

"""Offline, hash-bound MuJoCo MPPI pilot and CEM reference comparison.

The common 15-term evaluation is unchanged, but MPPI adds a Gaussian-prior
importance correction: these are not matched optimization objectives. This
analysis never steps dynamics, optimizes controls, or aligns trajectories.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_controller_pair import (
    load_arrays, plant_source_provenance, posture_diagnostic, read_json,
    reference_trace, sha256, source_model,
)
from analyze_nmpc_nominal_development import (
    audit_report as audit_cem_report, audit_trace, configuration_sha256,
    contact_diagnostic, parse_runs, require, safe_configuration,
    solver_diagnostic, source_digest,
)
from analyze_tvlqr_pilot import audit_physical_state, digest, public_compact
from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT, verify_reference_pair
from controller_trace import WheelKinematics
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES


MPPI_KIND = "full_body_height_smooth_mppi"
CEM_KIND = "full_body_height_smooth_cem_nmpc"
COUNT_NAMES = ("candidate_evaluations", "finite_candidates", "failed_candidates",
    "scenario_evaluations", "failed_scenarios", "batch_prediction_calls", "failed_batch_prediction_calls")
FLAG_NAMES = ("optimization_failed", "fallback_used", "warm_start_used", "initial_candidate_generation_failed")
SHARED_SOURCES = ("scripts/run_nmpc_tracking.py", "scripts/controller_references.py",
    "scripts/nmpc_tracking_objective.py", "scripts/height_smoothness.py", "scripts/classical_tracking.py",
    "scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py", "training/tron1_domain_randomization.py",
    "training/tron1_contact_slip.py", "training/tron1_tracking.py", "training/tron1_drift.py",
    "mdp/rewards.py", "tracking/tracking_env_cfg.py", "scripts/nmpc_batch_objective.py",
    "scripts/nmpc_native_batch.py", "native/nmpc_batch.cpp")


def finite_number(value, *, lower=None, upper=None, name="number"):
    require(type(value) in (float, int) and np.isfinite(value), "Invalid finite " + name)
    require(lower is None or value >= lower, "Below lower bound: " + name)
    require(upper is None or value <= upper, "Above upper bound: " + name)
    return float(value)


def audit_mppi_report(report, spec, *, model_sha256, trace_sha256, solver_sha256):
    """Audit MPPI as MPPI, without inventing elite counts or a CEM report."""
    require(report.get("kind") == MPPI_KIND and report.get("engine") == "MuJoCo"
            and report.get("status") == "completed", "Require completed full-body MuJoCo MPPI")
    require(report.get("reference_id") == PRIMARY_REFERENCE_ID == spec["reference_id"]
            and report.get("motion_sha256") == spec["motion_npz_sha256"]
            and report.get("model_sha256") == model_sha256
            and report.get("reference_catalog_sha256") == sha256(DEFAULT_CATALOG),
            "MPPI model/motion/reference identity mismatch")
    require(report.get("rollout_sha256") == trace_sha256
            and report.get("solver_log_sha256") == solver_sha256,
            "MPPI report-bound trace/solver SHA-256 mismatch")
    require(report.get("plant_mu") == .7 and report.get("prediction_mu_scenarios") == [.7],
            "Initial MPPI pilot requires nominal plant and singleton prediction mu=.7")
    material = np.asarray(report.get("plant_material_readback", {}).get("compiled_sliding_friction_range", []), dtype=float)
    require(material.shape == (2,) and np.allclose(material, [.7, .7], rtol=0, atol=1e-12),
            "Incorrect compiled nominal material readback")
    for name in ("learned_policy_used", "tvlqr_used", "old_qr_used", "root_state_prescribed_after_reset",
                 "external_root_forces_used", "prediction_receives_plant_mu", "truncated_smoke",
                 "pose_height_failure_gate", "height_or_flight_failure_gate", "heading_offset_is_mismatch",
                 "adaptive_model_selection_used", "sim2sim_comparison", "isaac_result",
                 "real_time_verified", "robustness_certificate", "optimization_equals_pure_task_cost_cem"):
        require(report.get(name) is False, "Missing/unsafe MPPI declaration: " + name)
    require(report.get("action_replay", False) is False, "Action replay is not MPPI")
    require(report.get("completed_policy_steps") == report.get("evaluated_policy_steps")
            == report.get("total_reference_policy_steps") == 349
            and report.get("physics_dt_s") == .005 and report.get("control_dt_s") == .02
            and report.get("physics_samples") == 1397
            and np.isclose(report.get("duration_s", np.nan), 6.98, atol=1e-10, rtol=0),
            "Require all 349 decisions at original .005/.02 s clocks")
    objective = safe_configuration(report.get("objective_contract"))
    require(isinstance(objective, dict) and objective.get("absolute_height_target") is False
            and objective.get("reference_vertical_velocity_target") is False
            and objective.get("height_regularization", {}).get("mode") == "vertical_acceleration_only"
            and objective.get("dt_s") == .02 and objective.get("body_names") == list(BODY_NAMES)
            and objective.get("joint_names") == list(JOINT_NAMES)
            and len(objective.get("weights", {})) == 15 and isinstance(objective.get("stds"), dict)
            and isinstance(objective.get("reference_order"), str),
            "Require common height-free named 15-term evaluation objective")
    packed = safe_configuration(report.get("packed_state_contract"))
    require(isinstance(packed, dict) and packed.get("dimension") == 111
            and packed.get("qpos_dimension") == 15 and packed.get("qvel_dimension") == 14
            and packed.get("contact_force_history_shape") == [3, 9, 3], "Invalid packed predictor state contract")
    require(isinstance(report.get("contact_slip_definition"), str)
            and "contact" in report["contact_slip_definition"].lower(), "Missing measured contact-slip definition")
    config = safe_configuration(report.get("effective_mppi_config"))
    settings = safe_configuration(report.get("settings"))
    require(isinstance(config, dict) and isinstance(settings, dict), "Missing effective MPPI settings")
    for name, value in (("horizon_steps", 30), ("control_blocks", 10), ("samples", 63), ("iterations", 3), ("seed", 42)):
        require(type(config.get(name)) is int and config[name] == value, "Unexpected fixed MPPI pilot knob: " + name)
    require(type(config.get("temperature")) in (int, float) and config["temperature"] == 1.
            and config.get("noise_std") == [.04]*6+[.35]*2
            and config.get("reference_convention") == "held_pre_action"
            and config.get("fallback_action") == "previous", "Unexpected MPPI sampler/reference declaration")
    require(settings.get("physics_dt_s") == .005 and settings.get("policy_dt_s") == .02,
            "MPPI settings clocks differ from plant/control clocks")
    require(settings.get("mppi") == config and settings.get("prediction_mu_scenarios") == [.7],
            "Effective MPPI settings differ from declared configuration")
    contract = safe_configuration(report.get("algorithm_contract"))
    require(isinstance(contract, dict) and contract, "Missing MPPI importance-sampling algorithm contract")
    require(contract.get("optimizer") == "fixed_covariance_latent_residual_mppi"
            and contract.get("log_weights_up_to_constant") == "-task_rollout_cost/temperature - sum_blocks(mean*raw_noise/noise_std**2)"
            and contract.get("mean_update") == "mean + sum_samples(normalized_weights * raw_noise)"
            and contract.get("temperature") == config["temperature"] and contract.get("noise_std") == config["noise_std"]
            and contract.get("per_iteration_rollouts") == {"iid_gaussian_samples": 63, "updated_mean_check": 1},
            "Unexpected MPPI latent importance-sampling update")
    for name in ("temperature_adaptation", "covariance_fitting", "elite_selection",
                 "deterministic_initial_plans_in_weighted_population", "task_rollout_score_modified",
                 "controller_added_terminal_value", "predictor_terminal_cost_hook_used_on_scalar_path",
                 "plant_friction_oracle", "robustness_certificate", "real_time_verified",
                 "package_reimplementation_identity_claimed"):
        require(contract.get(name) is False, "Missing/unsafe MPPI algorithm declaration: " + name)
    require(contract.get("sampling_distribution") == "Q=N(mean,diag(noise_std**2)); IID Gaussian blocks"
            and contract.get("base_distribution") == "P=N(0,diag(noise_std**2)) in per-solve baseline residual coordinates"
            and contract.get("deployed_plan") == "FINAL weighted-mean plan, freshly scored; never best sample or best initial plan"
            and contract.get("implicit_regularization") == "Gaussian-base KL-to-initial-baseline prior; not identical overall optimization objective to CEM"
            and report.get("common_15_term_physical_rollout_score_unchanged") is True,
            "Missing Gaussian prior / weighted-mean / unmatched overall objective declaration")
    parameterization = safe_configuration(report.get("action_parameterization"))
    require(isinstance(parameterization, dict)
            and parameterization.get("mode") == "unbounded_gaussian_latent_residual_blocks"
            and parameterization.get("baseline_copied_and_fixed_per_solve") is True
            and parameterization.get("latent_clip") is False and parameterization.get("noise_clip") is False
            and parameterization.get("density_block_multiplicity") == 1
            and parameterization.get("learned_baseline") is False,
            "Require unclipped Gaussian latent-space importance/update coordinates")
    require(report.get("model_generated_nominal_plan") == {"enabled": False},
            "No offline model-generated action plan in fixed MPPI pilot")
    native = native_metadata_audit(report)
    shared_sources = {name: source_digest(report, name) for name in SHARED_SOURCES}
    for suffix in ("scripts/run_mppi_tracking.py", "scripts/mppi_shooting.py",
                   "scripts/nmpc_tracking_objective.py", "scripts/nmpc_batch_objective.py",
                   "scripts/nmpc_native_batch.py", "native/nmpc_batch.cpp"):
        source_digest(report, suffix)
    require("effective_cem_config" not in report, "MPPI report must not invent an effective CEM configuration")
    assignment = np.arange(30)*10//30
    return {"controller_kind": MPPI_KIND, "actual_plant_mu": .7,
        "effective_mppi_config": config, "effective_mppi_config_sha256": configuration_sha256(config),
        "reported_settings": settings, "reported_settings_sha256": configuration_sha256(settings),
        "configuration_hash_definition": "SHA-256 of sorted compact UTF-8 JSON; not settings-file byte hash",
        "mppi_algorithm_contract": contract, "action_parameterization": parameterization,
        "native_prediction_backend": native,
        "prediction_horizon_s": .6, "control_block_durations_s": (np.bincount(assignment)*.02).tolist(),
        "actuator_identity": {"policy_contract_sha256": source_digest(report, "policy_contract.json"),
            "actuator_helper_sha256": source_digest(report, "scripts/classical_tracking.py"),
            "actuator_setup_helper_sha256": source_digest(report, "scripts/eval_tracking_mujoco.py"),
            "physics_dt_s": .005, "control_dt_s": .02},
        "packed_state_contract": packed,
        "shared_prediction_plant_reward_source_sha256": shared_sources,
        "evidence_scope": "Hash-bound recorded reports/traces, not independent proof of simulator internals"}


def native_metadata_audit(report):
    """Project only portable native identities, never compiler paths or PIDs."""
    raw = report.get("native_prediction_backend")
    require(report.get("prediction_backend") == "native" and isinstance(raw, dict)
            and raw.get("kind") == "optional_native_batch_exact_objective_adapter"
            and raw.get("production_cem_wired") is False and raw.get("production_mppi_wired") is True
            and raw.get("runtime_mppi_population_path") == "callable predictor.rollout_batch",
            "Require genuine native MPPI wiring, not CEM relabeling")
    objective = raw.get("objective", {})
    require(objective.get("production_mppi_wired") is True and objective.get("production_cem_wired") is False
            and raw.get("plant_mu_used_for_prediction") is False and raw.get("learned_initialization") is False
            and raw.get("prediction_mu_scenarios") == [.7] and raw.get("terminal_cost") is None,
            "Unsafe native MPPI objective/model metadata")
    workers = raw.get("workers", {})
    require(set(workers) == {"0.7"} and isinstance(workers["0.7"], dict), "Require singleton native worker identity")
    worker = workers["0.7"]
    fields = ("source_sha256", "mujoco_header_sha256", "mujoco_library_sha256", "native_library_sha256", "model_snapshot_sha256")
    identities = {name: digest(worker.get(name)) for name in fields}
    require(worker.get("physics_dt_s") == .005 and worker.get("control_dt_s") == .02,
            "Native worker clock metadata differs")
    require(identities["source_sha256"] == source_digest(report, "native/nmpc_batch.cpp"),
            "Native build/source identity mismatch")
    return {"kind": safe_configuration(raw.get("kind")), "production_cem_wired": False,
        "production_mppi_wired": True, "runtime_mppi_population_path": raw["runtime_mppi_population_path"],
        "objective_production_cem_wired": False, "objective_production_mppi_wired": True,
        "worker_identities": {"0.7": identities},
        "scope": "Recorded native attachment/build identities only; compiler commands, absolute paths and worker PIDs omitted"}


def mppi_diagnostic(report, raw, solver):
    """Audit genuine saved weighted updates and commands, not fictitious elites.

    Per-sample controls/costs/weights are not retained; the saved aggregate
    importance summaries are checked for consistency, not independently replayed.
    """
    require(len(solver) == 349, "Require every MPPI decision")
    compute = np.asarray(raw["compute_s"], dtype=float)
    require(compute.shape == (349,) and np.isfinite(compute).all() and np.all(compute >= 0)
            and np.array_equal(compute, [row.get("compute_s") for row in solver])
            and np.allclose(report.get("compute_ms_median_p95_max", []), 1000*np.quantile(compute, [.5, .95, 1.]),
                            atol=1e-8, rtol=1e-12), "Saved MPPI timing differs between report/trace/log")
    above = float(np.mean(compute > .02))
    require(report.get("compute_over_control_dt_fraction") == above and report.get("real_time_verified") is False,
            "MPPI deadline fraction/real-time declaration mismatch")
    wall = finite_number(report.get("wall_s"), lower=0, name="recorded rollout wall time")
    totals = {name: 0 for name in (*COUNT_NAMES, *FLAG_NAMES)}
    metrics = {name: [] for name in ("ess", "max_weight", "update_l2", "population_decode_clip_fraction",
        "updated_mean_decode_clip_fraction", "raw_cost_min", "raw_cost_max", "importance_correction_min",
        "importance_correction_max", "updated_mean_model_cost")}
    scopes = set(); null_metrics = {name: 0 for name in metrics}; failed_iterations = 0
    for k, (row, applied) in enumerate(zip(solver, raw["actions"])):
        info = row.get("solver", {})
        for name in COUNT_NAMES:
            value = info.get(name)
            require(type(value) is int and value >= 0, "Invalid MPPI counter: " + name)
            totals[name] += value
        for name in FLAG_NAMES:
            require(type(info.get(name)) is bool, "Invalid MPPI flag: " + name)
            totals[name] += int(info[name])
        require(info["warm_start_used"] == (k > 0 and not solver[k-1]["solver"]["optimization_failed"]),
                "MPPI warm-start history differs from preceding successful plan")
        require(info["finite_candidates"]+info["failed_candidates"] == info["candidate_evaluations"]
                and info["scenario_evaluations"] == info["candidate_evaluations"]
                and info["failed_scenarios"] <= info["scenario_evaluations"]
                and info["failed_batch_prediction_calls"] <= info["batch_prediction_calls"],
                "Inconsistent MPPI candidate/scenario/batch counts")
        require(info.get("prediction_mu_scenarios") == [.7]
                and info.get("control_deadline_enforced") is False and info.get("real_time_verified") is False,
                "Unexpected MPPI singleton model or enforced deadline")
        require(info.get("optimizer") == "fixed_covariance_latent_residual_mppi"
                and info.get("reference_convention") == "held_pre_action"
                and info.get("population_evaluation_backend") == "predictor_rollout_batch"
                and info.get("action_parameterization") == "unbounded_gaussian_latent_residual_blocks"
                and info.get("residual_baseline_fixed_per_solve") is True
                and info.get("implicit_gaussian_kl_prior") is True
                and info.get("learned_policy_fallback") is False and info.get("robustness_certificate") is False,
                "Per-step solver is not declared fixed-covariance latent-prior MPPI")
        for name, value in (("horizon_steps", 30), ("control_blocks", 10), ("samples", 63),
                            ("iterations_requested", 3), ("temperature", 1.)):
            require(info.get(name) == value, "MPPI per-step knob differs: " + name)
        require(info.get("noise_std") == [.04]*6+[.35]*2, "MPPI per-step noise differs")
        require(isinstance(info.get("timing_scope"), str), "Missing MPPI timing scope")
        scopes.add(safe_configuration(info["timing_scope"]))
        finite_number(info.get("compute_time_s"), lower=0, upper=float(row["compute_s"])+1e-8, name="inner MPPI timing")
        command = np.asarray(row.get("raw_mppi_action", []), dtype=float)
        require(command.shape == (8,) and np.isfinite(command).all() and np.max(np.abs(command)) <= 1.+1e-9
                and np.array_equal(np.clip(command, -1., 1.).astype(np.float32).astype(float), applied),
                "Raw MPPI action differs from actual clipped float32 command")
        if info["fallback_used"]:
            previous = np.zeros(8) if k == 0 else np.asarray(raw["actions"])[k-1]
            require(info["optimization_failed"] and np.array_equal(applied, previous),
                    "MPPI failure fallback must hold the previous applied command")
        require(info["fallback_used"] == info["optimization_failed"], "MPPI fallback/failure flags differ")
        require(not any(name in info for name in ("elite_count", "elite_indices", "elite_costs", "iteration_best_costs")),
                "MPPI must not invent CEM elite diagnostics")
        iterations = info.get("mppi_iterations")
        require(isinstance(iterations, list) and len(iterations) <= 3
                and (info["optimization_failed"] or len(iterations) == 3), "Missing/incomplete MPPI weighted updates")
        require(info.get("iterations_attempted") == len(iterations)
                and info.get("iterations_completed") == sum(item.get("updated_mean_model_cost") is not None for item in iterations),
                "MPPI attempted/completed-iteration count differs")
        initial = info.get("initial_candidate_evaluations")
        require(isinstance(initial, list) and len(initial) == (3 if info["warm_start_used"] else 2)-int(info["initial_candidate_generation_failed"]),
                "MPPI initial-candidate budget differs from saved warm/generation flags")
        initial_costs = [item.get("model_cost") for item in initial]
        require(all(value is None or (type(value) in (int, float) and np.isfinite(value)) for value in initial_costs),
                "Invalid logged initial costs")
        finite_initial = [float(value) for value in initial_costs if value is not None]
        require(info.get("initial_model_cost") == (min(finite_initial) if finite_initial else None),
                "Initial selected MPPI baseline differs from saved initial costs")
        evaluated = len(initial); finite_count = len(finite_initial); calls = len(initial)
        for index, item in enumerate(iterations):
            require(isinstance(item, dict) and item.get("iteration") == index, "Broken MPPI iteration order")
            population_finite = item.get("finite_population_candidates")
            require(type(population_finite) is int and 0 <= population_finite <= 63, "Invalid finite population count")
            evaluated += 63; finite_count += population_finite; calls += 1
            population_clip = finite_number(item.get("population_decode_clip_fraction"), lower=0, upper=1, name="population decode clipping")
            if population_finite == 0:
                require(item.get("failure") == "all_population_predictions_failed"
                        and item.get("updated_mean_model_cost") is None and index == len(iterations)-1
                        and info["optimization_failed"], "Empty MPPI population must stop and hold previous command")
                metrics["population_decode_clip_fraction"].append(population_clip)
                for name in metrics:
                    if name != "population_decode_clip_fraction": null_metrics[name] += 1
                failed_iterations += 1
                continue
            evaluated += 1; calls += 1
            for name in metrics:
                if item.get(name) is None:
                    require(name in ("importance_correction_min", "importance_correction_max", "updated_mean_model_cost"),
                            "Unexpected null MPPI iteration statistic: " + name)
                    if name == "updated_mean_model_cost":
                        require(info["optimization_failed"] and index == len(iterations)-1
                                and info.get("optimization_failure_reason") == "updated_mean_prediction_failed",
                                "Null weighted-mean score must stop with explicit prediction failure")
                    null_metrics[name] += 1
                    continue
                lower, upper = (1., float(population_finite)+1e-10) if name == "ess" else ((1/population_finite, 1.) if name == "max_weight"
                    else ((0., 1.) if name.endswith("clip_fraction") else ((0., None) if name == "update_l2" else (None, None))))
                metrics[name].append(finite_number(item.get(name), lower=lower, upper=upper, name=name))
            finite_count += int(item.get("updated_mean_model_cost") is not None)
            require(item["raw_cost_min"] <= item["raw_cost_max"]
                    and (item["importance_correction_min"] is None or item["importance_correction_max"] is None
                         or item["importance_correction_min"] <= item["importance_correction_max"]),
                    "Inverted MPPI cost/correction ranges")
            if item.get("log_weight_max") is not None:
                finite_number(item["log_weight_max"], name="log_weight_max")
            normalizer = finite_number(item.get("shifted_weight_normalizer"), lower=1., upper=population_finite+1e-10, name="shifted_weight_normalizer")
            require(np.isclose(item["max_weight"], 1/normalizer, atol=1e-12, rtol=1e-12)
                    and item["ess"]+1e-10 >= 1/item["max_weight"]
                    and item["ess"] <= 1/item["max_weight"]**2+1e-10,
                    "Inconsistent MPPI normalized weight/ESS summaries")
        require(info["candidate_evaluations"] == evaluated and info["finite_candidates"] == finite_count
                and info["failed_candidates"] == evaluated-finite_count and info["failed_scenarios"] == evaluated-finite_count
                and info["batch_prediction_calls"] == calls, "MPPI counters differ from initial/population/weighted-mean logs")
        if not info["optimization_failed"]:
            selected = finite_number(info.get("selected_model_cost"), name="selected_model_cost")
            require(np.isclose(selected, iterations[-1]["updated_mean_model_cost"], atol=1e-12, rtol=1e-12),
                    "Selected MPPI cost must be the final evaluated weighted mean, not the cheapest sample")
            require(info.get("selected_scenario_costs") == [selected], "MPPI selected singleton score differs")
            initial_cost = info.get("initial_model_cost")
            require(info.get("predicted_cost_nonincreasing") == (selected <= initial_cost if initial_cost is not None else None),
                    "MPPI nonincreasing-cost diagnostic differs; no best-initial-plan assumption")
        else:
            require(info.get("selected_model_cost") is None and info.get("selected_scenario_costs") is None,
                    "Failed MPPI solve must not publish a selected feasible-plan cost")
    require(totals["optimization_failed"] == report.get("solver_failures"), "MPPI failure total differs from log")
    aggregate = {name: {"samples": len(values), "null_or_unavailable_samples": null_metrics[name],
                       "median_p95_max": np.quantile(values, [.5, .95, 1.]).tolist()
                       if values else None, "minimum": min(values) if values else None}
                 for name, values in metrics.items()}
    return {"decision_samples": 349, "counts": totals, "weighted_update_diagnostics": aggregate,
        "weighted_diagnostics_scope": "Saved aggregate logs only; original per-sample costs/weights/plans were not retained and are not independently replayed",
        "selected_plan_rule": "Final re-evaluated MPPI weighted mean, not best raw-cost population member",
        "all_population_failed_iterations": failed_iterations,
        "timing": {"compute_ms_median_p95_max": (1000*np.quantile(compute, [.5, .95, 1.])).tolist(),
            "fraction_above_20ms": above, "reported_solver_timing_scopes": sorted(scopes),
            "recorded_rollout_wall_s": wall, "real_time_verified": False, "deadline_enforced": False,
            "actuation_delay_injected": False,
            "comparison_scope": "Original uncontrolled wall-clock observations; simulator waits for controller; not matched hardware/real-time evidence"}}


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = ["reference", *summary["run_order"]]
    columns = min(3, len(labels)); rows = math.ceil(len(labels)/columns)
    fig, grid = plt.subplots(rows, columns, figsize=(4.7*columns, 5.*rows), sharex=True, sharey=True, squeeze=False)
    start, end = summary["reference_spec"]["source_demo_interval_s"]
    low_core, high_core = summary["reference_spec"]["evaluation_core_interval_s"]
    rt = compact["reference_time_s"]; ref = compact["reference_position"][(rt >= start-1e-9) & (rt <= end+1e-9), :2]
    bounds = [ref]
    for j, (axis, label) in enumerate(zip(grid.ravel(), labels)):
        color = ".35" if label == "reference" else plt.get_cmap("tab10")((j-1) % 10)
        t, pos, wheels = [compact[f"{label}_{name}"] for name in ("time_s", "position", "wheel_pos")]
        selected = (t >= start-1e-9) & (t <= end+1e-9)
        core = selected & (t >= low_core-1e-9) & (t <= high_core+1e-9)
        axis.plot(ref[:, 0], ref[:, 1], "--", color=".75", lw=1)
        axis.plot(pos[selected, 0], pos[selected, 1], color=color, lw=1, alpha=.5)
        axis.plot(pos[core, 0], pos[core, 1], color=color, lw=2.2)
        indices = np.flatnonzero(selected)[::max(1, round(.4/float(compact[f"{label}_physics_dt_s"])))]
        for side in range(2):
            axis.plot(wheels[selected, side, 0], wheels[selected, side, 1], color=color, lw=.55, alpha=.35)
            for index in indices:
                if not compact[f"{label}_root_course_to_wheel_valid"][index, side]: continue
                grounded = label == "reference" or (compact[f"{label}_contact_count"][index, side] > 0
                    and compact[f"{label}_contact_force"][index, side, 2] > 5.)
                heading = compact[f"{label}_wheel_rolling_plane_heading_rad"][index, side]
                delta = .055*np.array([np.cos(heading), np.sin(heading)])
                segment = wheels[index, side, :2]+np.array([-delta, delta])
                axis.plot(segment[:, 0], segment[:, 1], color=color, lw=1.2, alpha=.9 if grounded else .15)
        arrows = indices[compact[f"{label}_root_beta_valid"][indices]]
        course = compact[f"{label}_course_rad"][arrows]
        axis.quiver(pos[arrows, 0], pos[arrows, 1], np.cos(course), np.sin(course), color=color,
                    angles="xy", scale_units="xy", scale=7, width=.003)
        axis.set(title="Kinematic reference (contacts unmeasured)" if label == "reference" else label+" · MuJoCo μ=0.7",
                 xlabel="World X [m]", ylabel="World Y [m]")
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
        bounds.extend((pos[selected, :2], wheels[selected, :, :2].reshape(-1, 2)))
    xy = np.concatenate(bounds); low, high = xy.min(axis=0), xy.max(axis=0)
    margin = max(.15, .06*float(np.max(high-low)))
    grid[0, 0].set_xlim(low[0]-margin, high[0]+margin); grid[0, 0].set_ylim(low[1]-margin, high[1]+margin)
    for axis in grid.ravel()[len(labels):]: axis.set_visible(False)
    fig.suptitle("MPPI fixed pilot · MuJoCo only / not matched optimization cost / not ranked", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=".3", lw=2.2, label="Thick: core; arrows: XY velocity"),
        Line2D([], [], color=".3", lw=1, label="Ticks: wheel rolling planes; pale: no measured contact")],
        loc="lower center", ncol=2 if columns > 1 else 1, fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .92)); fig.savefig(output, dpi=160); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing output; choose a NEW output directory")
    directories = parse_runs(args.run)
    spec = verify_reference_pair(args.motion_file, args.reference_file, PRIMARY_REFERENCE_ID)
    require(spec["frames"] == 349 and spec["fps"] == 50., "Require full primary reference")
    traces = {"reference": reference_trace(load_arrays(args.motion_file))}; reports = {}; audits = {}
    for label, directory in directories.items():
        files = {name: directory/name for name in ("report.json", "rollout.npz", "solver_log.json")}
        hashes = {name: sha256(path) for name, path in files.items()}; report = read_json(files["report.json"])
        log = json.loads(files["solver_log.json"].read_text()); raw = load_arrays(files["rollout.npz"])
        require(isinstance(log, list), "Require genuine per-decision optimization log")
        if report.get("kind") == MPPI_KIND:
            audit = audit_mppi_report(report, spec, model_sha256=sha256(args.model), trace_sha256=hashes["rollout.npz"], solver_sha256=hashes["solver_log.json"])
            diagnostic = mppi_diagnostic(report, raw, log)
            horizon = audit["effective_mppi_config"]["horizon_steps"]
        else:
            audit = audit_cem_report(report, spec, model_sha256=sha256(args.model), trace_sha256=hashes["rollout.npz"], solver_sha256=hashes["solver_log.json"])
            audit.update(controller_kind=CEM_KIND, actual_plant_mu=.7)
            diagnostic = solver_diagnostic(report, raw, log, audit["parameterization"])
            horizon = audit["effective_cem_config"]["horizon_steps"]
        trace = audit_trace(raw, log, spec)
        audit_physical_state(raw, trace)
        stage = np.asarray(raw["stage_cost"], dtype=float)
        require(stage.shape == (349,) and np.isfinite(stage).all()
                and np.array_equal(stage, [row.get("objective", {}).get("stage_cost") for row in log])
                and np.isclose(report.get("total_stage_cost", np.nan), stage.sum(), atol=1e-9, rtol=1e-12),
                "Common 15-term evaluation stage costs differ across trace/log/report")
        require(not reports or report["objective_contract"] == next(iter(reports.values()))["objective_contract"],
                "Different common evaluation objective contracts")
        require(not audits or audit["actuator_identity"] == next(iter(audits.values()))["actuator_identity"],
                "Different actuator/helper identities")
        shared_sources = {name: source_digest(report, name) for name in SHARED_SOURCES}
        require(not audits or shared_sources == next(iter(audits.values()))["shared_prediction_plant_reward_source_sha256"],
                "Different imported predictor/physical/reward implementation identities")
        reports[label], traces[label] = report, trace
        audits[label] = {**audit, "input_sha256": hashes, "solver_diagnostics": diagnostic,
            "shared_prediction_plant_reward_source_sha256": shared_sources,
            "simulation_budget_diagnostic": {"planning_predicted_physics_substeps": diagnostic["counts"]["scenario_evaluations"]*horizon*4,
                "actual_plant_physics_substeps": 1396,
                "scope": "Requested budget from saved counters/horizon; failed requests are not proof of completed native steps"},
            "reported_source_provenance": plant_source_provenance(report)}
    require(any(value["controller_kind"] == MPPI_KIND for value in audits.values())
            and any(value["controller_kind"] == CEM_KIND for value in audits.values()),
            "Require both MPPI pilot and actual CEM baseline")
    with source_model(args.source_root, args.model) as (model, loader):
        fk = WheelKinematics(model); recon = {label: fk.reconstruct(trace) for label, trace in traces.items()}
    reference, interval = traces["reference"], spec["evaluation_core_interval_s"]
    metrics = {}
    for label, trace in traces.items():
        contacts = {} if label == "reference" else dict(wheel_ground_contact_count=trace["contact_count"],
            wheel_ground_force_w=trace["contact_force"], wheel_contact_lateral_slip_m_s=trace["contact_slip"])
        metrics[label] = summarize_turn(trace["time_s"], trace["position"], recon[label]["signals"],
            reference["time_s"], reference["position"][:, :2], recon["reference"]["signals"]["speed_m_s"], interval, **contacts)
    summary = {"schema_version": 1, "kind": "mujoco_only_mppi_fixed_pilot", "status": "development_analysis_only",
        "engine": "MuJoCo", "fixed_mu": .7, "reference_id": PRIMARY_REFERENCE_ID, "reference_spec": spec,
        "run_order": list(directories), "runs": audits, "windows": {"core": metrics},
        "objective_contract": safe_configuration(next(iter(reports.values()))["objective_contract"]),
        **{name: False for name in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking",
            "matched_optimization_objective", "robustness_certificate", "controlled_execution_success_claimed",
            "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "learned_inference")},
        "common_15_term_evaluation_objective": True, "model_loader_audit": loader,
        "fk_audit": {label: value["audit"] for label, value in recon.items()},
        "posture_diagnostics": {label: posture_diagnostic(traces[label]) for label in directories},
        "contact_diagnostics": {label: contact_diagnostic(traces[label], interval) for label in directories},
        "input_sha256": {name: sha256(getattr(args, name)) for name in ("motion_file", "reference_file", "model")},
        "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
        "analysis_source_sha256": {name: sha256(ROOT/"scripts"/name) for name in (
            "analyze_mppi_pilot.py", "analyze_nmpc_nominal_development.py", "analyze_tvlqr_pilot.py",
            "analyze_controller_pair.py", "controller_metrics.py", "controller_trace.py", "controller_references.py")},
        "limitations": ["Fixed MuJoCo-only MPPI pilot; no Isaac/sim2sim qualification, algorithm ranking, hardware, real-time or robustness certificate.",
            "The common 15-term evaluation is unchanged. MPPI adds a Gaussian-prior importance/KL correction, so total optimization objectives differ from CEM.",
            "MPPI selects its final re-evaluated weighted mean, not the lowest raw-cost sample. Sampler, initialization and source identities remain explicit.",
            "63 random samples plus one weighted-mean re-evaluation per iteration match the CEM population evaluation count, not optimization-objective or information equality.",
            "Saved aggregate importance diagnostics are not independently replayed per-sample weights/costs; original candidate arrays are not retained.",
            "Original clocks and world axes retained; no alignment/time warp. Reference contacts are unmeasured; height, flight, contact loss and heading offsets are descriptive, not failure gates.",
            "Timings are uncontrolled original observations. Simulator waits for optimization; no injected wall-latency actuation delay or real-time claim."]}
    compact = public_compact(traces, recon)
    output.mkdir(parents=True); np.savez_compressed(output/"development_trace.npz", **compact)
    render_topdown(compact, summary, output/"topdown.png")
    summary["artifacts"] = {name: {"file": filename, "sha256": sha256(output/filename)}
        for name, filename in (("compact_trace", "development_trace.npz"), ("topdown", "topdown.png"))}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--motion", "--motion-file", dest="motion_file", type=Path, required=True)
    for name in ("reference-file", "model", "source-root", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "runs": result["run_order"]}))

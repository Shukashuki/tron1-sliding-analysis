"""Partial-aware, recorded-state friction diagnostics for frozen400Hz PPO/CEM.

The old200Hz grid is never loaded. Completed cells must be SHA-bound and match
the completed mu=.7 controller/plant contracts except the declared sliding
coefficient. Missing, running and interrupted cells have no XY verdict. This
module does not train, infer actions, step dynamics, align poses or retune.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_backend_400hz_matrix as metrics
from analyze_ppo_ground_support_failure import persistent_events
from controller_metrics import heading_signals, summarize_turn
from controller_trace import native_trace
from ppo_backend_400hz_contract import read_json, require, sha256, write_json
import render_ppo_cem_matched_reward_400hz as nominal

GRID = (.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.)
ORDER = nominal.ORDER
LABELS = {"ppo_beta_on": "PPO + beta", "cem_beta_on": "CEM + beta",
          "ppo_beta_off": "PPO - beta", "cem_beta_off": "CEM - beta"}
COLORS = nominal.COLORS
COMPLETE = {"completed", "completed_verified", "reused_completed_nominal"}
PENDING = {"pending", "missing", "running", "failed", "interrupted_preserved", "smoke_completed"}
WINDOWS = {"core": (1.44, 4.), "reference_overlap": (0., 6.96)}


def group_id(controller, condition):
    require(controller in ("PPO", "MPC-CEM") and condition in ("beta_on", "beta_off"),
            "Select one of the four frozen PPO/CEM beta conditions")
    return ("ppo_" if controller == "PPO" else "cem_")+condition


def case_key(group, mu):
    require(group in ORDER and type(mu) in (int, float) and float(mu) in GRID,
            "Require one exact coefficient from the locked17-point grid")
    return group+"/mu"+f"{mu:.3f}".replace(".", "p")


def canonical_cases(cases):
    """Fill absent cells explicitly; duplicate or off-grid cells are errors."""
    require(isinstance(cases, list), "Manifest cases/jobs must be a list")
    found = {}
    for original in cases:
        require(isinstance(original, dict), "Each manifest cell must be an object")
        cell = copy.deepcopy(original)
        group = group_id(cell.get("controller"), cell.get("condition"))
        key = case_key(group, cell.get("mu"))
        require(key not in found, "Duplicate grid cell: "+key)
        require(cell.get("status") in COMPLETE | PENDING, "Unknown case status: "+key)
        cell.update(id=key, group=group, mu=float(cell["mu"]))
        found[key] = cell
    rows = []
    for group in ORDER:
        for mu in GRID:
            key = case_key(group, mu)
            rows.append(found.get(key, {"id": key, "group": group,
                "controller": "PPO" if group.startswith("ppo") else "MPC-CEM",
                "condition": "beta_on" if group.endswith("beta_on") else "beta_off",
                "mu": mu, "status": "missing", "reused_nominal": False}))
    return rows


def _digest(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value), "Missing/invalid SHA256: "+label)
    return value


def _local(folder, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "Require relative input filename")
    path = (folder/name).resolve()
    require(path.is_relative_to(folder) and path.is_file(), "Missing or escaped input file: "+str(path))
    return path


def read_completed(cell):
    """Do not inspect a live partial file, even if it happens to be present."""
    require(cell["status"] in COMPLETE, "Only completed cells have physical metrics")
    folder = Path(cell["directory"]).resolve()
    files = {"report": _local(folder, cell.get("report_file", "report.json")),
             "recording": _local(folder, cell.get("raw_file", "rollout.npz"))}
    expected = {name: _digest(cell.get(name+"_sha256"), name) for name in files}
    for name, path in files.items():
        require(sha256(path) == expected[name], "Completed input SHA differs: "+str(path))
    report = read_json(files["report"])
    with np.load(files["recording"], allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    for name, path in files.items():
        require(sha256(path) == expected[name], "Completed input changed while loading: "+str(path))
    return raw, report, {name: {"path": str(path), "sha256": expected[name]} for name, path in files.items()}


def check_material(material, baseline, mu):
    require(material.get("native_mapping_verified") is True
            and material.get("compiled_native_timestep_s") == .0025
            and material.get("explicit_contact_pair_count") == 0
            and material.get("pair_coefficient") == mu, "Wrong native pair coefficient/clock/mapping")
    for key in ("wheel_geom_ids", "colliding_geom_ids", "colliding_geom_names", "geom_priority",
                "geom_condim", "geom_solref", "geom_solimp", "native_plant_readback"):
        require(material.get(key) == baseline[key], "Frozen native contact/plant arrays differ: "+key)
    friction = np.asarray(material.get("geom_friction"), dtype=float)
    previous = np.asarray(baseline["geom_friction"], dtype=float)
    require(friction.shape == previous.shape and friction.ndim == 2 and friction.shape[1] == 3
            and np.isfinite(friction).all() and np.all(friction[:, 0] == mu)
            and np.array_equal(friction[:, 1:], previous[:, 1:]),
            "Only the declared sliding coefficient may change, not torsional/rolling friction")


def check_report(cell, report, baseline, reference_sha, model_sha):
    """Bind controller identity to the frozen completed nominal cohort."""
    mu, condition = cell["mu"], cell["condition"]
    require(report.get("status") == "completed" and report.get("motion_sha256") == reference_sha
            and report.get("model_sha256") == model_sha, "Incomplete or different-reference/model episode")
    if cell["controller"] == "PPO":
        contract = report["contract"]
        profile = copy.deepcopy(contract["task_profile"])
        require(profile["contact_friction"] == mu, "PPO evaluation coefficient differs")
        profile["contact_friction"] = .7
        require(profile == baseline["contract"]["task_profile"]
                and contract["physics_dt"] == .0025 and contract["decimation"] == 8,
                "PPO deployment/reward/clock differs from frozen400Hz cohort")
        def frozen_contract(value):
            result = copy.deepcopy(value)
            result["task_profile"]["contact_friction"] = .7
            for key in ("evaluation_contact_friction_override", "backend_400hz_cohort_evaluation"):
                result.pop(key, None)
            return result
        require(frozen_contract(contract) == frozen_contract(baseline["contract"]),
                "PPO actuator, observation, DR or non-friction deployment contract changed")
        proof, original = report["backend_400hz_cohort_evaluation"], baseline["backend_400hz_cohort_evaluation"]
        for key in ("checkpoint_sha256", "normalized_actor_sha256", "training_backend", "cohort_id",
                    "protocol_sha256", "generated_source_sha256"):
            require(proof[key] == original[key], "PPO actor/training/evaluator identity differs: "+key)
        require(proof["status"] == "verified" and proof["training_backend"] == "physx"
                and proof["smoke_only"] is False and proof["old_200hz_actor_used"] is False
                and proof["additional_training_updates"] == 0
                and proof["native_states"] == 2793 and proof["policy_decisions"] == 349
                and proof["recording_sha256"] == cell["recording_sha256"]
                and report["policy_steps"] == 349 and report["physics_steps"] == 2792
                and report["hidden_resets"] == 0 and report["root_state_writes"] == 1
                and report["completed_full_reference"] is True,
                "PPO completion/no-reset provenance differs")
        check_material(report["contact_material_audit"], baseline["contact_material_audit"], mu)
    else:
        require(report.get("kind") in ("400hz_CEM_with_completed_PPO_stage_reward",
                    "400hz_friction_sweep_CEM_with_completed_PPO_stage_reward")
                and report["condition"] == condition and report["plant_mu"] == mu
                and report["prediction_mu_scenarios"] == [mu]
                and report["prediction_friction_known"] is True
                and report["physics_dt_s"] == .0025 and report["policy_dt_s"] == .02
                and report["decimation"] == 8 and report["completed_policy_steps"] == 349
                and report["native_samples"] == 2793 and report["truncated_smoke"] is False,
                "CEM coefficient, fixed-horizon completion, clock or oracle disclosure differs")
        profile = copy.deepcopy(report["task_profile"])
        require(profile["contact_friction"] == mu, "CEM evaluation profile coefficient differs")
        profile["contact_friction"] = .7
        require(profile == baseline["task_profile"], "CEM reward/actuator profile changed")
        for key in ("CEM_config", "objective_contract", "discount_factor", "prediction_state_contract",
                    "action_parameterization", "soft_joint_pos_limits", "soft_joint_limits_source"):
            require(report[key] == baseline[key], "Frozen CEM controller differs: "+key)
        require(report["rollout_sha256"] == cell["recording_sha256"]
                and report["termination_or_reset_after_initialization"] is False
                and report["height_contact_tilt_failure_gate"] is False
                and report["PPO_actions_or_future_states_used_for_initial_plan"] is False
                and report["new_training_or_policy_inference"] is False
                and report["controller_tuned_on_this_episode"] is False,
                "CEM rollout identity or no-retuning/no-gate provenance differs")
        for name in ("plant_contact_material_audit", "prediction_contact_material_audit"):
            check_material(report[name], baseline[name], mu)


def clock_audit(trace):
    time = np.asarray(trace["time_s"])
    require(time.shape == (2793,) and trace["physics_dt_s"] == .0025
            and np.allclose(time, np.arange(2793)*.0025, atol=1e-10, rtol=0),
            "Require full original400Hz clock:2793 states,0–6.98s")
    counts = np.asarray(trace["contact_count"])
    require((counts >= 0).all() and np.array_equal(counts, np.round(counts)), "Invalid measured contact counts")


def along_cross_error(time, position, reference, interval, minimum_reference_speed=.3):
    """Time-synchronous tangent decomposition, not closest-path projection."""
    time = np.asarray(time, dtype=float)
    selection = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
    query = time[selection]
    require(len(query) >= 2 and query[0] >= reference["time_s"][0]-1e-9
            and query[-1] <= reference["time_s"][-1]+1e-9, "Tangent metrics cannot extrapolate the reference")
    target = np.column_stack([np.interp(query, reference["time_s"], reference["position"][:, j]) for j in (0, 1)])
    velocity = np.column_stack([np.interp(query, reference["time_s"], reference["velocity"][:, j]) for j in (0, 1)])
    speed = np.linalg.norm(velocity, axis=1)
    valid = speed > minimum_reference_speed
    tangent = np.zeros_like(velocity)
    tangent[valid] = velocity[valid]/speed[valid, None]
    error = np.asarray(position)[selection, :2]-target
    along = np.sum(error*tangent, axis=1)
    cross = tangent[:, 0]*error[:, 1]-tangent[:, 1]*error[:, 0]
    def summary(value):
        sample = value[valid]
        return {"rms_m": float(np.sqrt(np.mean(sample**2))) if len(sample) else None,
                "signed_mean_m": float(np.mean(sample)) if len(sample) else None,
                "absolute_p95_m": float(np.quantile(np.abs(sample), .95)) if len(sample) else None,
                "maximum_absolute_m": float(np.abs(sample).max()) if len(sample) else None}
    return {"definition": "Actual minus reference at the same timestamp; reference COM-velocity tangent; left normal positive.",
            "minimum_reference_speed_m_s_strict": minimum_reference_speed,
            "eligible_samples": int(len(query)), "valid_samples": int(valid.sum()),
            "valid_fraction": float(valid.mean()), "along_track": summary(along), "cross_track": summary(cross),
            "time_alignment_or_closest_path_projection": False}


def contiguous_course_changes(time, signals, interval):
    """Do not unwrap or bridge headings through zero-speed/invalid samples."""
    time = np.asarray(time)
    mask = ((time >= interval[0]-1e-9) & (time <= interval[1]+1e-9))
    valid = mask & np.asarray(signals["root_beta_valid"], dtype=bool)
    ids = np.flatnonzero(valid)
    pieces = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1) if len(ids) else []
    segments = []
    for indices in pieces:
        if len(indices) < 2:
            continue
        angle = np.unwrap(signals["course_rad"][indices])
        segments.append({"start_s": float(time[indices[0]]), "end_s": float(time[indices[-1]]),
            "samples": int(len(indices)), "signed_course_change_deg": float(np.degrees(angle[-1]-angle[0]))})
    complete = bool(mask.any() and valid[mask].all())
    return {"valid_samples": int(valid.sum()), "window_samples": int(mask.sum()),
            "complete_valid_window": complete, "contiguous_segments": segments,
            "net_course_change_deg": segments[0]["signed_course_change_deg"] if complete and len(segments) == 1 else None,
            "invalid_gap_unwrap_or_interpolation": False}


def failure_metrics(trace, reference):
    """Existing windows plus additional XY/timing and contact-gated geometry."""
    clock_audit(trace)
    windows = nominal.paired.trace_metrics(trace, reference)
    time = trace["time_s"]
    extras = {}
    speed_ref = np.linalg.norm(reference["velocity"][:, :2], axis=1)
    reference_signals = {"root_beta_valid": speed_ref >= .25,
        "course_rad": np.arctan2(reference["velocity"][:, 1], reference["velocity"][:, 0])}
    for name, interval in WINDOWS.items():
        selected = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
        query = time[selected]
        desired = np.column_stack([np.interp(query, reference["time_s"], reference["position"][:, j]) for j in (0, 1)])
        error = np.linalg.norm(trace["position"][selected, :2]-desired, axis=1)
        turn = summarize_turn(time, trace["position"], trace["signals"], reference["time_s"],
            reference["position"][:, :2], speed_ref, interval,
            wheel_ground_contact_count=trace["contact_count"], wheel_ground_force_w=trace["contact_force"],
            wheel_contact_lateral_slip_m_s=trace["contact_slip"])
        extras[name] = {"tangent_error": along_cross_error(time, trace["position"], reference, interval),
            "course": contiguous_course_changes(time, trace["signals"], interval),
            "reference_course": contiguous_course_changes(reference["time_s"], reference_signals, interval),
            "turn_and_ground_contact_heading_geometry": turn,
            "final_xy_error_m": float(error[-1]),
            "persistent_xy_gt0p30m": persistent_events(query, error > .30, .1),
            "large_xy_screen": bool(np.sqrt(np.mean(error**2)) > .15 or error.max() > .30)}
    return {"windows": windows, "failure_diagnostics": extras,
            "core_large_xy_screen": extras["core"]["large_xy_screen"],
            "overlap_large_xy_screen": extras["reference_overlap"]["large_xy_screen"],
            "height_airborne_tilt_or_beta_failure_gate": False}


def baseline_differences(row, baseline):
    """A baseline failure is not newly caused by changing friction."""
    if row.get("evaluated") is not True or baseline is None or baseline.get("evaluated") is not True:
        return {"available": False, "nominal_missing_or_incomplete": True}
    result = {"available": True, "nominal_mu": .7,
        "nominal_core_large_xy_screen": baseline["core_large_xy_screen"],
        "nominal_overlap_large_xy_screen": baseline["overlap_large_xy_screen"],
        "new_core_screen_exceedance_relative_to_nominal": bool(row["core_large_xy_screen"] and not baseline["core_large_xy_screen"]),
        "new_overlap_screen_exceedance_relative_to_nominal": bool(row["overlap_large_xy_screen"] and not baseline["overlap_large_xy_screen"]),
        "friction_causality_or_statistical_significance_claimed": False}
    for window in WINDOWS:
        result[window] = {"xy_rmse_change_m": row["windows"][window]["tracking"]["xy_rmse_m"]-baseline["windows"][window]["tracking"]["xy_rmse_m"],
            "xy_peak_change_m": row["windows"][window]["tracking"]["xy_peak_m"]-baseline["windows"][window]["tracking"]["xy_peak_m"]}
    return result


def analyze_cases(cases, reference=nominal.paired.DEFAULT_REFERENCE, model=nominal.paired.DEFAULT_MODEL,
                  legacy_evaluator=nominal.paired.DEFAULT_LEGACY, nominal_data=None):
    """Return (summary, native traces); partial grids are valid, forged cells not."""
    base_traces, reference_values, nominal_summary = (nominal.load_comparison(reference=reference, model=model,
        legacy_evaluator=legacy_evaluator) if nominal_data is None else nominal_data)
    reference_sha, model_sha = sha256(reference), sha256(model)
    require(nominal_summary["reference_sha256"] == reference_sha and nominal_summary["model_sha256"] == model_sha,
            "Nominal bindings/reference/model differ")
    baseline_rows = {row["id"]: row for row in nominal_summary["rows"]}
    baseline_reports = {group: read_json(Path(baseline_rows[group]["directory"])/"report.json") for group in ORDER}
    training_mu_range = (baseline_reports["ppo_beta_on"].get("contract", {})
        .get("training_domain_randomization", {}).get("sliding_friction", {}).get("range"))
    if training_mu_range is not None:
        require(training_mu_range == [.3, 1.2]
                and baseline_reports["ppo_beta_off"]["contract"]["training_domain_randomization"]["sliding_friction"]["range"] == training_mu_range,
                "PPO training friction-DR range differs from the frozen paired cohort")
    fk, provenance = metrics.load_fk(model, legacy_evaluator)
    traces, rows, ledger = {}, [], {}
    for cell in canonical_cases(cases):
        row = {key: cell[key] for key in ("id", "group", "controller", "condition", "mu", "status")}
        row.update(evaluated=False, core_large_xy_screen=None, overlap_large_xy_screen=None,
            reused_nominal=bool(cell.get("reused_nominal", False)), sample_size=0,
            PPO_coefficient_outside_training_DR_range=(bool(cell["mu"] < training_mu_range[0] or cell["mu"] > training_mu_range[1])
                if training_mu_range is not None and cell["controller"] == "PPO" else None))
        if cell["status"] not in COMPLETE:
            row["physical_metrics_unavailable_reason"] = "No SHA-verified completed full-horizon cell; not a tracking pass or failure."
            rows.append(row)
            continue
        raw, report, files = read_completed(cell)
        check_report(cell, report, baseline_reports[cell["group"]], reference_sha, model_sha)
        if cell["mu"] == .7:
            require(cell.get("reused_nominal") is True
                    and cell["recording_sha256"] == baseline_rows[cell["group"]]["recording_sha256"]
                    and cell["report_sha256"] == baseline_rows[cell["group"]]["report_sha256"],
                    "Nominal mu=.7 must reuse the exact completed files, not rerun/relabel")
        kind = "policy_mujoco" if cell["controller"] == "PPO" else "nmpc"
        trace = native_trace(raw, kind)
        clock_audit(trace)
        for name in ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel", "wheel_pos"):
            require(np.array_equal(trace[name][0], base_traces[cell["group"]][name][0]),
                    "Original recorded initial state differs: "+name)
        reconstruction = fk.reconstruct(trace)
        trace.update(signals=reconstruction["signals"], wheel_quaternion_wxyz=reconstruction["wheel_quaternion_wxyz"],
            wheel_center_velocity_w=reconstruction["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = metrics.heading_geometry_masks(trace)
        row.update(failure_metrics(trace, reference_values), evaluated=True, sample_size=1,
            wheel_FK_audit=reconstruction["audit"], input_files=files,
            prediction_friction_known=cell["controller"] == "MPC-CEM")
        if cell["controller"] == "MPC-CEM":
            row["solver_failures"] = report["solver_failures"]
            row["compute_ms_median_p95_max"] = report["compute_ms_median_p95_max"]
            row["real_time_verified"] = False
        traces[cell["id"]] = trace
        ledger[cell["id"]] = files
        rows.append(row)
    for group in ORDER:
        baseline = next((r for r in rows if r["group"] == group and r["mu"] == .7 and r["evaluated"]), None)
        for row in (r for r in rows if r["group"] == group):
            row["difference_from_own_nominal_mu0p7"] = baseline_differences(row, baseline)
    # A final pass catches mutation during expensive offline FK/plot preparation.
    for files in ledger.values():
        for value in files.values():
            require(sha256(value["path"]) == value["sha256"], "Completed source changed during offline analysis")
    evaluated = sum(row["evaluated"] for row in rows)
    return {"schema_version": 1, "kind": "partial_aware400Hz_matched_reward_PPO_CEM_known_friction_failure_diagnostics",
        "grid": list(GRID), "case_order": list(ORDER), "rows": rows, "expected_cells": 68,
        "completed_verified_cells": evaluated, "remaining_or_unavailable_cells": 68-evaluated,
        "grid_complete": evaluated == 68, "validation_backend": "native CPU MuJoCo",
        "PPO_training_backend": "PhysX", "CEM_prediction_backend": "native CPU MuJoCo",
        "PPO_training_sliding_friction_DR_range": training_mu_range,
        "PPO_training_DR_range_scope": "Scalar sliding-coefficient range only; not a claim that all states or native contact laws are in-distribution.",
        "validation_friction_intervention": "Set every colliding geom's sliding coefficient to the declared mu; wheel priority1/other0 selects the effective pair value, not mu squared. This is not a fixed-wheel/ground-only causal intervention.",
        "CEM_predictor_receives_exact_plant_mu": True, "PPO_receives_explicit_mu": False,
        "stage_reward_matched": True, "comparison_is_optimizer_only": False,
        "physics_dt_s": .0025, "policy_dt_s": .02, "native_states_per_completed_cell": 2793,
        "reference_interval_s": [0., 6.96], "full_native_interval_s": [0., 6.98], "core_interval_s": [1.44, 4.],
        "xy_screen_limits_m": {"rmse_strict": .15, "peak_strict": .30},
        "persistent_xy_event": {"error_strict_gt_m": .30, "minimum_duration_s": .1, "interval_s": [0., 6.96]},
        "reference_sha256": reference_sha, "model_sha256": model_sha, "FK_provenance": provenance,
        "nominal_evidence": {"rows": nominal_summary["rows"], "CEM_input_ledger_sha256": nominal_summary["CEM_input_ledger_sha256"]},
        "input_files": ledger, "new_simulation_training_or_policy_inference_during_analysis": False,
        "height_airborne_tilt_or_beta_failure_gate": False, "pose_or_time_alignment": False,
        "statistical_success_probability_or_causal_effect_claimed": False,
        "limitations": ["N=1 per tested coefficient/controller/beta condition; no calibrated universal friction threshold, monotonicity, statistical superiority or causal claim.",
            "The400Hz completed policies and15-term CEM stage reward are frozen. Historical200Hz grids are not pooled.",
            "PPO was trained with scalar sliding-friction DR[0.3,1.2]; validation below0.3 or above1.2 is outside that parameter range. Equal numeric mu does not equate engine contact laws.",
            "Training DR randomizes wheel contact coefficients; validation overrides all colliding sliding coefficients and disables observation noise/pushes. Shaded coefficient range is not an IID distribution-equivalence claim or a wheel-fixed/ground-only intervention.",
            "CEM has privileged full simulator state, exact plant coefficient and0.6s reference/model preview; PPO has no explicit mu input. This is not optimizer-only or information-matched.",
            "Same stage reward is not the same return optimization: CEM gamma=.99 with30steps and no learned terminal value; PPO learns long-horizon value.",
            "Core XY screening can hide later falls. Reference-overlap metrics and persistent departures are separate; no extrapolation from6.96 to6.98s.",
            "A nominal failure is retained, not attributed anew to friction. All differences are relative to each controller/beta condition's exact reused mu=.7 result.",
            "Beta and axial wheel-heading offsets are tracked features, not failure or zero-slip targets. Contact-count/Fz zeros are not flight labels.",
            "Heading/contact metrics disclose validity/coverage. Missing, runtime-failed, running and smoke cells are not physical tracking pass/fail cases.",
            "CEM remains offline relative to20ms. Predictor cold-resets hidden solver state each interval; validation plant integrates continuously."]}, traces


def metric_rows(summary):
    result = []
    for row in summary["rows"]:
        value = {key: row[key] for key in ("id", "controller", "condition", "mu", "status", "evaluated")}
        value.update(core_large_xy_screen=row["core_large_xy_screen"], overlap_large_xy_screen=row["overlap_large_xy_screen"])
        if row["evaluated"]:
            for window in WINDOWS:
                tracking = row["windows"][window]["tracking"]
                for name in ("xy_rmse_m", "xy_peak_m", "speed_retention", "beta_rmse_deg", "beta_valid_samples"):
                    value[window+"_"+name] = tracking.get(name)
                extra = row["failure_diagnostics"][window]
                value[window+"_along_track_rms_m"] = extra["tangent_error"]["along_track"]["rms_m"]
                value[window+"_cross_track_rms_m"] = extra["tangent_error"]["cross_track"]["rms_m"]
                value[window+"_course_change_deg"] = extra["course"]["net_course_change_deg"]
                value[window+"_reference_course_change_deg"] = extra["reference_course"]["net_course_change_deg"]
                value[window+"_fraction_below80pct_reference_speed"] = extra["turn_and_ground_contact_heading_geometry"]["fraction_below_80pct_reference_speed"]
            event = row["failure_diagnostics"]["reference_overlap"]["persistent_xy_gt0p30m"]["first_persistent_event"]
            value["persistent_xy_departure_onset_s"] = event["onset_s"] if event else None
            value["persistent_xy_departure_confirmation_s"] = event["confirmation_s"] if event else None
        result.append(value)
    return result


def render_curves(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    lookup = {(row["group"], row["mu"]): row for row in summary["rows"]}
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    fields = (("core", "xy_rmse_m", "Core XY RMSE (m)"),
              ("reference_overlap", "xy_rmse_m", "Reference-overlap XY RMSE (m)"),
              ("core", "speed_retention", "Core mean speed / reference"),
              ("core", "beta_rmse_deg", "Core beta RMSE (deg, eligible only)"))
    for group in ORDER:
        for axis, (window, name, title) in zip(axes, fields):
            values = [lookup[group, mu]["windows"][window]["tracking"].get(name, np.nan)
                      if lookup[group, mu]["evaluated"] else np.nan for mu in GRID]
            axis.plot(GRID, values, "--o" if group.startswith("cem") else "-o", color=COLORS[group],
                label=LABELS[group], markersize=3, linewidth=1.4)
            axis.set(xscale="log", xlabel="Declared native sliding coefficient mu", title=title)
            axis.grid(alpha=.2, which="both"); axis.axvline(.7, color=".6", linestyle=":", linewidth=.8)
    for axis in axes[:2]:
        axis.axhline(.15, color=".5", linestyle=":", linewidth=1.)
    training_range = summary.get("PPO_training_sliding_friction_DR_range")
    if training_range is not None:
        for axis in axes:
            axis.axvspan(*training_range, color="#4682a6", alpha=.07, label="PPO training mu DR")
    axes[2].axhline(1., color=".5", linestyle=":", linewidth=1.)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Frozen400Hz PPO/CEM matched stage rewards | native CPU MuJoCo | verified{summary['completed_verified_cells']}/68 cells | N=1/cell", fontsize=12)
    figure.text(.5, .02, "Missing/running/runtime-failed cells are gaps, NOT passes. CEM receives exact mu; PPO does not. Dashed: CEM. Core1.44–4s; reference-overlap0–6.96s.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0., .06, 1., .92)); figure.savefig(Path(destination)/"friction_curves.png", dpi=160); plt.close(figure)

    screen = np.zeros((4,17), dtype=int)
    for i, group in enumerate(ORDER):
        for j, mu in enumerate(GRID):
            row = lookup[group, mu]
            if row["evaluated"]:
                screen[i, j] = 2 if row["core_large_xy_screen"] else 1
    figure, axis = plt.subplots(figsize=(15, 4.))
    cmap = colors.ListedColormap(("#d9d9d9", "#8bc8a3", "#df9997"))
    axis.imshow(screen, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    axis.set_xticks(range(17), [f"{mu:g}" for mu in GRID]); axis.set_yticks(range(4), [LABELS[group] for group in ORDER])
    for i, group in enumerate(ORDER):
        for j, mu in enumerate(GRID):
            row = lookup[group, mu]
            axis.text(j, i, "L" if screen[i,j] == 2 else "P" if screen[i,j] == 1 else "—", ha="center", va="center", fontsize=9)
    axis.set(title="Core XY descriptive screen only: RMSE >0.15m OR peak >0.30m; not task completion", xlabel="Declared native sliding coefficient mu")
    figure.text(.5, .01, "P: below both limits in the core; L: large core XY error; gray: not evaluated. A core P can still fall after4s. No height/flight/beta failure gate.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0., .05, 1., 1.)); figure.savefig(Path(destination)/"xy_screen_matrix.png", dpi=160); plt.close(figure)


def render_topdown(summary, traces, reference, destination, selected=(.15, .3, .7, 1.2)):
    """One fixed absolute world frame across all selected cells; no follow zoom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["group"], row["mu"]): row for row in summary["rows"]}
    available = [traces[case_key(group, mu)] for mu in selected for group in ORDER if case_key(group, mu) in traces]
    positions = [reference["position"][:, :2]]+[item["position"][:, :2] for item in available]
    xy = np.concatenate(positions)
    center = .5*(xy.min(axis=0)+xy.max(axis=0)); half = max(.5, .55*float(np.ptp(xy, axis=0).max()))
    figure, axes = plt.subplots(len(selected), 4, figsize=(16, 4*len(selected)), squeeze=False)
    for i, mu in enumerate(selected):
        for j, group in enumerate(ORDER):
            axis = axes[i,j]
            axis.plot(reference["position"][:,0], reference["position"][:,1], "--", color=".65", linewidth=1., label="Reference")
            key, row = case_key(group, mu), lookup[group, mu]
            if key in traces:
                trace = traces[key]; color = COLORS[group]; signals = trace["signals"]
                axis.plot(trace["position"][:,0], trace["position"][:,1], color=color, linewidth=1.4)
                bearing = (trace["contact_count"] > 0) & (trace["contact_force"][:,:,2] > 5.)
                indices = np.arange(0,2793,160)
                for k in indices:
                    body_valid = bool(trace["body_heading_geometry_valid"][k])
                    if body_valid:
                        angle = signals["body_heading_rad"][k]
                        arrow = .09*np.array([np.cos(angle), np.sin(angle)])
                        axis.annotate("", xy=trace["position"][k,:2]+arrow, xytext=trace["position"][k,:2],
                            arrowprops={"arrowstyle": "->", "color": "#1464b4", "linewidth": .7})
                    for side, wheel_color in enumerate(("#d62728", "#9467bd")):
                        if not trace["wheel_heading_geometry_valid"][k,side]:
                            continue
                        angle = signals["wheel_rolling_plane_heading_rad"][k,side]
                        delta = .055*np.array([np.cos(angle), np.sin(angle)])
                        points = trace["wheel_pos"][k,side,:2]+np.array([-delta,delta])
                        axis.plot(points[:,0],points[:,1],color=wheel_color,linewidth=.8,
                            alpha=.85 if bearing[k,side] else .2)
                core_end = int(round(4./.0025))
                axis.plot(*trace["position"][core_end,:2], "o", color=color, markersize=3)
                text = f"core RMSE={row['windows']['core']['tracking']['xy_rmse_m']:.3f}m"
            else:
                text = row["status"]+"; not evaluated"
            axis.set(title=f"{LABELS[group]} | mu={mu:g}\n{text}", xlabel="World X (m)", ylabel="World Y (m)",
                xlim=(center[0]-half,center[0]+half), ylim=(center[1]-half,center[1]+half))
            axis.set_aspect("equal",adjustable="box"); axis.grid(alpha=.2)
    figure.suptitle("Original world-XY paths | fixed shared limits | blue arrows: body heading; red/purple ticks: axial wheel planes\nOpaque ticks: measured wheel count>0 and upward Fz>5N; faint is not a flight label. Dot:4s. No pose/time alignment.", fontsize=11)
    figure.tight_layout(rect=(0.,0.,1.,.95)); figure.savefig(Path(destination)/"selected_topdown.png",dpi=140); plt.close(figure)
    return {"selected_mu": list(selected), "all_selected_axes_world_xlim_m": [float(center[0]-half),float(center[0]+half)],
            "all_selected_axes_world_ylim_m": [float(center[1]-half),float(center[1]+half)],
            "poses_filtered_aligned_or_corrected": False, "wheel_contact_gate_force_n": 5.}


def compact_arrays(traces, reference):
    """Retain native metrics inputs/FK outcomes without meshes, weights or model."""
    fields = nominal.paired.TRACE_FIELDS
    result = {"reference_"+key: np.asarray(value).copy() for key, value in reference.items() if isinstance(value,np.ndarray)}
    result["case_ids"] = np.asarray(list(traces))
    result["physics_dt_s"] = np.asarray(.0025)
    for key, trace in traces.items():
        prefix = key.replace("/", "__")+"__"
        for name in fields:
            result[prefix+name] = np.asarray(trace[name]).copy()
    return result


def recompute_compact(arrays):
    """Recompute metrics from saved native arrays; no licensed/model asset needed."""
    require(float(arrays["physics_dt_s"]) == .0025, "Compact trace clock must remain400Hz")
    reference = {key[len("reference_"):]: value for key, value in arrays.items() if key.startswith("reference_")}
    result = {}
    for key in np.asarray(arrays["case_ids"]).tolist():
        prefix = key.replace("/", "__")+"__"
        trace = {name: np.asarray(arrays[prefix+name]).copy() for name in nominal.paired.TRACE_FIELDS}
        trace["physics_dt_s"] = .0025
        trace["signals"] = heading_signals(trace["quaternion"],trace["com_velocity"],trace["wheel_quaternion_wxyz"],trace["wheel_center_velocity_w"])
        result[key] = failure_metrics(trace,reference)
    return result


def manifest_snapshot(path):
    """Snapshot JSON bytes once; a live runner may later publish more cells."""
    path = Path(path).resolve()
    payload = path.read_bytes(); value = json.loads(payload)
    cases = value.get("cases", value.get("jobs"))
    require(isinstance(cases, list), "Manifest needs canonical cases[] or runner jobs[]")
    return cases, {"path": str(path), "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
                   "runner_status_at_snapshot": value.get("status"), "case_count_at_snapshot": len(cases)}


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing analysis; fresh output directory required")
    cases, manifests = [], []
    for path in args.manifest:
        values, evidence = manifest_snapshot(path); cases.extend(values); manifests.append(evidence)
    summary, traces = analyze_cases(cases,args.reference,args.model,args.legacy_evaluator)
    reference = metrics.reference_data(args.reference)
    summary["manifest_snapshots"] = manifests
    summary["analyzer_sha256"] = sha256(__file__)
    output.mkdir(parents=True,exist_ok=False)
    render_curves(summary,output)
    summary["selected_topdown"] = render_topdown(summary,traces,reference,output)
    np.savez_compressed(output/"native_failure_signals.npz",**compact_arrays(traces,reference))
    rows = metric_rows(summary)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output/"metric_table.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    summary["artifact_sha256"] = {name: sha256(output/name) for name in
        ("friction_curves.png","xy_screen_matrix.png","selected_topdown.png","metric_table.csv","native_failure_signals.npz")}
    write_json(output/"summary.json",summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest",action="append",type=Path,required=True,
        help="Repeat for separate PPO and CEM manifests; use SHA-bound cases[] or runner jobs[].")
    result.add_argument("--reference",type=Path,default=nominal.paired.DEFAULT_REFERENCE)
    result.add_argument("--model",type=Path,default=nominal.paired.DEFAULT_MODEL)
    result.add_argument("--legacy-evaluator",type=Path,default=nominal.paired.DEFAULT_LEGACY)
    result.add_argument("--output-dir",type=Path,required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())

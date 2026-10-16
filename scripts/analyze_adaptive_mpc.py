"""Offline, hash-bound MuJoCo adaptive-MPC development comparison.

This script never steps dynamics, runs optimization, or aligns trajectories.
It accepts completed adaptive episodes and singleton fixed-model baselines;
the finite identification bank is NOT a worst-case optimization scenario set.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np

from analyze_controller_pair import (
    compact_arrays, load_arrays, plant_source_provenance, posture_diagnostic,
    read_json, reference_trace, sha256, source_model,
)
from analyze_nmpc_nominal_development import (
    audit_report as audit_nominal_common_schema, audit_trace, contact_diagnostic,
    require, safe_configuration, solver_diagnostic,
)
from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT, verify_reference_pair
from controller_trace import WheelKinematics


ADAPTIVE_KIND = "full_body_height_smooth_adaptive_cem_mpc"
FIXED_KIND = "full_body_height_smooth_cem_nmpc"
MODEL_BANK = [.3, .7, 1.6]


def parse_runs(values):
    require(len(values) >= 1, "Provide at least one --run LABEL=DIR")
    result = {}
    for value in values:
        label, separator, directory = value.partition("=")
        require(separator and directory and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", label),
                "Use portable LABEL=DIR run arguments")
        require(label.lower() != "reference" and label not in result, "Duplicate/reserved run label")
        require(Path(directory).is_dir(), "Missing run directory: " + label)
        result[label] = Path(directory)
    require(len({path.resolve() for path in result.values()}) == len(result), "Run directories must be distinct")
    return result


def audit_report(report, spec, *, model_sha256, trace_sha256, solver_sha256):
    """Validate the real variant/material before reusing the unchanged common audit.

    The old nominal analyzer remains strict and unchanged. Only a PRIVATE copy
    of the three nominal-only discriminator fields is normalized for its common
    objective, clocks, hashes, actuator, configuration and state-schema checks.
    The actual engine, material and adaptive declarations are checked first and
    preserved in the result; the normalized copy is never published or reused.
    """
    kind = report.get("kind")
    require(kind in (ADAPTIVE_KIND, FIXED_KIND), "Unsupported adaptive/fixed full-body controller kind")
    mu = report.get("plant_mu")
    require(type(mu) in (float, int) and np.isfinite(mu) and mu >= 0, "Invalid actual plant friction")
    readback = np.asarray(report.get("plant_material_readback", {}).get("compiled_sliding_friction_range", []), dtype=float)
    require(readback.shape == (2,) and np.allclose(readback, [mu, mu], rtol=0, atol=1e-12),
            "Actual plant friction differs from compiled material readback")
    if kind == ADAPTIVE_KIND:
        require(report.get("prediction_mu_scenarios") is None,
                "Adaptive identification bank must not be declared as worst-case prediction scenarios")
        require(report.get("prediction_mu_model_bank") == MODEL_BANK, "Unexpected finite identification model bank")
        contract = safe_configuration(report.get("adaptation_contract"))
        require(isinstance(contract, dict) and contract, "Missing adaptation contract")
        require(contract.get("kind") == "causal_multiple_model_adaptive_prediction_selection"
                and contract.get("model_bank") == MODEL_BANK
                and contract.get("prediction_models_per_solve") == 1
                and contract.get("identification_model_substeps") == 4
                and contract.get("identification_interval_s") == .02,
                "Unsupported causal finite contact-model adaptation contract")
        for key in ("selected_mu_is_physical_friction_estimate", "plant_mu_used_for_prediction",
                    "future_plant_state_used", "reference_used_for_identification", "robustness_certificate"):
            require(contract.get(key) is False, "Missing/unsafe adaptation declaration: " + key)
        selector = contract.get("selector", {})
        require(selector.get("mu_hypotheses") == MODEL_BANK and selector.get("initial_mu") in MODEL_BANK
                and type(selector.get("window_size")) is int and selector["window_size"] > 0
                and type(selector.get("minimum_evidence")) is int
                and 0 < selector["minimum_evidence"] <= selector["window_size"], "Invalid selector contract")
        for key in ("minimum_spread", "maximum_fit_rms", "switch_improvement_ratio"):
            value = selector.get(key)
            require(type(value) in (int, float) and np.isfinite(value) and value > 0,
                    "Invalid selector threshold: " + key)
        require(selector["switch_improvement_ratio"] < 1, "Invalid selector switch hysteresis")
        for key in ("robustness_certificate", "posterior_confidence", "plant_coefficient_input", "reference_error_input"):
            require(selector.get(key) is False, "Missing/unsafe selector declaration: " + key)
    else:
        scenarios = report.get("prediction_mu_scenarios")
        require(isinstance(scenarios, list) and len(scenarios) == 1
                and type(scenarios[0]) in (int, float) and np.isfinite(scenarios[0]) and scenarios[0] >= 0,
                "Fixed baseline must use one finite nonnegative prediction model")
        contract = None
    common = copy.deepcopy(report)
    common.update(kind=FIXED_KIND, plant_mu=.7, prediction_mu_scenarios=[.7])
    common["plant_material_readback"]["compiled_sliding_friction_range"] = [.7, .7]
    audit = audit_nominal_common_schema(common, spec, model_sha256=model_sha256,
                                      trace_sha256=trace_sha256, solver_sha256=solver_sha256)
    require(audit["model_generated_nominal_plan"] == {"enabled": False},
            "No offline/model-generated action-plan initializer in this comparison")
    return {**audit, "controller_kind": kind, "actual_plant_mu": float(mu),
            "plant_friction_readback": readback.tolist(),
            "prediction_mu_scenarios": safe_configuration(report.get("prediction_mu_scenarios")),
            "prediction_mu_model_bank": MODEL_BANK.copy() if kind == ADAPTIVE_KIND else None,
            "adaptation_contract": contract,
            "common_schema_scope": "Unchanged nominal objective/clock/hash/actuator audit after separately validating actual variant and material; no nominal-plant claim"}


def adaptation_diagnostic(report, solver, raw):
    """Audit actual singleton solves, causal indices, contact skips and ID cost.

    Recorded feature scores are evidence declarations, not independently
    recomputed residuals: intermediate hypothesis states were not retained.
    """
    require(len(solver) == 349, "Require every adaptive/fixed decision")
    if report["kind"] == FIXED_KIND:
        require(all(row["solver"].get("prediction_mu_scenarios") == report["prediction_mu_scenarios"]
                    and "adaptive_contact" not in row["solver"] for row in solver),
                "Fixed baseline singleton differs from declared prediction model")
        return {"enabled": False, "prediction_mu_scenarios": report["prediction_mu_scenarios"]}
    selector = report["adaptation_contract"]["selector"]
    previous = selector["initial_mu"]
    selected, statuses, elapsed, calls, substeps, switches = [], [], [], [], [], []
    valid_statuses = {"initial_prior", "nonwheel_contact", "no_contact", "low_spread",
                      "unsupported", "insufficient_evidence", "held", "switched"}
    for k, decision in enumerate(solver):
        info = decision["solver"]
        observation = info.get("adaptive_contact")
        require(isinstance(observation, dict), "Missing per-decision adaptive observation")
        mu, status = observation.get("selected_mu"), observation.get("status")
        require(type(mu) in (float, int) and mu in MODEL_BANK and status in valid_statuses,
                "Invalid selected model/status")
        require(info.get("prediction_mu_scenarios") == [mu]
                and observation.get("model_bank") == MODEL_BANK
                and observation.get("prediction_models_per_solve") == 1
                and observation.get("selected_mu_is_physical_friction_estimate") is False,
                "Actual solve is not the logged singleton model")
        require(observation.get("previous_mu") == previous and type(observation.get("updated")) is bool
                and observation["updated"] == (mu != previous)
                and (status == "switched") == observation["updated"], "Broken model-switch history")
        count, steps = observation.get("identification_prediction_calls"), observation.get("identification_physics_substeps")
        cost = observation.get("estimation_compute_s")
        require(type(count) is int and count in (0, 3) and type(steps) is int and steps == 4*count,
                "Invalid identification call/substep budget")
        require(type(cost) in (float, int) and np.isfinite(cost) and cost >= 0,
                "Invalid identification compute time")
        base_compute = info.get("compute_time_s")
        require(type(base_compute) in (float, int) and np.isfinite(base_compute) and base_compute >= 0
                and cost+base_compute <= decision["compute_s"]+1e-8,
                "Outer controller timing must include identification and CEM")
        window, evidence = observation.get("window_steps"), observation.get("informative_steps")
        require(type(window) is int and window == min(k, selector["window_size"])
                and type(evidence) is int and 0 <= evidence <= window,
                "Invalid bounded causal evidence window")
        scores, spread = observation.get("scores"), observation.get("prediction_spread_rms")
        if k == 0:
            require(status == "initial_prior" and mu == selector["initial_mu"] and count == 0
                    and scores is None and spread is None and evidence == 0
                    and observation.get("source_interval_reference_index") is None
                    and observation.get("transition_duration_s") is None,
                    "First decision must use the declared prior without a transition")
        else:
            require(status != "initial_prior" and observation.get("source_interval_reference_index") == k-1
                    and observation.get("transition_duration_s") == .02,
                    "Identification must use exactly the already completed preceding interval")
            require(type(spread) in (float, int) and np.isfinite(spread) and spread >= 0,
                    "Invalid hypothesis spread")
            interval = slice(4*(k-1)+1, 4*k+1)
            wheel = bool(np.any(np.asarray(raw["wheel_ground_contact_count"])[interval] > 0))
            nonwheel = bool(np.any(np.asarray(raw["nonwheel_ground_contact_count"])[interval] > 0))
            eligible = wheel and not nonwheel
            require(count == (3 if eligible else 0), "Identification calls disagree with actual preceding contact samples")
            if nonwheel:
                require(status == "nonwheel_contact", "Measured nonwheel contact must skip identification")
            elif not wheel:
                require(status == "no_contact", "Measured contact-free interval must skip identification")
            else:
                require(status not in ("no_contact", "nonwheel_contact"), "Contact-skip status contradicts measured interval")
            if status == "low_spread":
                require(spread < selector["minimum_spread"], "Low-spread status contradicts declared threshold")
            if status == "insufficient_evidence":
                require(evidence < selector["minimum_evidence"], "Insufficient-evidence status contradicts threshold")
            if status in ("held", "switched"):
                require(evidence >= selector["minimum_evidence"], "Model decision lacks declared minimum evidence")
        if evidence:
            values = np.asarray(scores, dtype=float)
            require(values.shape == (3,) and np.isfinite(values).all() and np.all(values >= 0),
                    "Missing/invalid bank-aligned finite evidence scores")
            if status in ("held", "switched"):
                require(float(values.min()) <= selector["maximum_fit_rms"], "Unsupported scores must not select a model")
                old_index = MODEL_BANK.index(previous)
                best_index = old_index if values[old_index] == values.min() else int(np.argmin(values))
                new = previous
                if best_index != old_index and values[best_index] <= selector["switch_improvement_ratio"]*values[old_index]:
                    new = MODEL_BANK[best_index]
                require(mu == new, "Selected model contradicts recorded scores and hysteresis")
        else:
            require(scores is None, "Empty evidence window must not invent score evidence")
        selected.append(float(mu)); statuses.append(status); elapsed.append(float(cost))
        calls.append(count); substeps.append(steps); switches.append(int(observation["updated"]))
        previous = mu
    aggregate = {"decisions": len(solver), "status_counts": dict(Counter(statuses)),
                 "selected_model_counts": dict(Counter(str(value) for value in selected)),
                 "model_switches": sum(switches), "identification_prediction_calls": sum(calls),
                 "identification_predicted_physics_substeps": sum(substeps),
                 "identification_compute_ms_median_p95_max": (1000*np.quantile(elapsed, [.5, .95, 1.])).tolist(),
                 "identification_backend": "existing_python_scalar_cold_reset_transition",
                 "included_in_controller_compute_s": True, "physical_friction_estimate_claimed": False,
                 "robustness_certificate": False}
    declared = report.get("adaptation_summary")
    require(isinstance(declared, dict) and set(declared) == set(aggregate), "Missing/incomplete report-bound adaptation summary")
    for key, value in aggregate.items():
        if key == "identification_compute_ms_median_p95_max":
            require(np.allclose(declared[key], value, rtol=1e-12, atol=1e-8), "Identification timing summary differs from decisions")
        else:
            require(declared[key] == value, "Adaptation summary differs from decisions: "+key)
    return {"enabled": True, **aggregate, "model_bank": MODEL_BANK.copy(),
            "selected_model_per_decision": selected, "status_per_decision": statuses,
            "selection_time_s": (np.arange(349)*.02).tolist(),
            "selection_time_definition": "Pre-action solve k at k*0.02 s; identification uses measured completed interval k-1 only",
            "identification_compute_total_s": float(np.sum(elapsed)),
            "evidence_scope": "Report-bound logs and measured interval contact samples. Feature scores/hypothesis transitions are not independently recomputed.",
            "physical_friction_estimate_claimed": False, "robustness_certificate": False}


def render_topdown(compact, summary, output):
    """Reference and actual native XY traces; no invented contact labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = ["reference", *summary["run_order"]]
    columns = min(3, len(labels)); rows = math.ceil(len(labels)/columns)
    fig, grid = plt.subplots(rows, columns, figsize=(4.7*columns, 4.9*rows),
                             sharex=True, sharey=True, squeeze=False)
    axes = grid.ravel(); spec = summary["reference_spec"]
    start, end = spec["source_demo_interval_s"]; lo_core, hi_core = spec["evaluation_core_interval_s"]
    rt = compact["reference_time_s"]
    ref = compact["reference_position"][(rt >= start-1e-9) & (rt <= end+1e-9), :2]
    bounds = [ref]
    for j, (axis, label) in enumerate(zip(axes, labels)):
        color = ".35" if label == "reference" else plt.get_cmap("tab10")((j-1) % 10)
        t, pos, wheels = [compact[f"{label}_{name}"] for name in ("time_s", "position", "wheel_pos")]
        selected = (t >= start-1e-9) & (t <= end+1e-9)
        core = selected & (t >= lo_core-1e-9) & (t <= hi_core+1e-9)
        axis.plot(ref[:, 0], ref[:, 1], "--", color=".75", lw=1)
        axis.plot(pos[selected, 0], pos[selected, 1], color=color, lw=1, alpha=.5)
        axis.plot(pos[core, 0], pos[core, 1], color=color, lw=2.2)
        indices = np.flatnonzero(selected)[::max(1, round(.4/float(compact[f"{label}_physics_dt_s"])))]
        heading = compact[f"{label}_wheel_rolling_plane_heading_rad"]
        valid = compact[f"{label}_root_course_to_wheel_valid"]
        for side in range(2):
            axis.plot(wheels[selected, side, 0], wheels[selected, side, 1], color=color, lw=.55, alpha=.35)
            for index in indices:
                if not valid[index, side]:
                    continue
                grounded = label == "reference" or (compact[f"{label}_contact_count"][index, side] > 0
                            and compact[f"{label}_contact_force"][index, side, 2] > 5.)
                delta = .055*np.array([np.cos(heading[index, side]), np.sin(heading[index, side])])
                segment = wheels[index, side, :2] + np.array([-delta, delta])
                axis.plot(segment[:, 0], segment[:, 1], color=color, alpha=.9 if grounded else .15, lw=1.2)
        arrows = indices[compact[f"{label}_root_beta_valid"][indices]]
        course = compact[f"{label}_course_rad"][arrows]
        axis.quiver(pos[arrows, 0], pos[arrows, 1], np.cos(course), np.sin(course), color=color,
                    angles="xy", scale_units="xy", scale=7, width=.003)
        if label == "reference":
            title = "Kinematic reference (contacts unmeasured)"
        else:
            title = f"{label} · plant μ={summary['runs'][label]['actual_plant_mu']:g}"
        axis.set(title=title, xlabel="World X [m]", ylabel="World Y [m]")
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
        bounds.extend((pos[selected, :2], wheels[selected, :, :2].reshape(-1, 2)))
    xy = np.concatenate(bounds); lo, hi = xy.min(axis=0), xy.max(axis=0)
    margin = max(.15, .06*float(np.max(hi-lo)))
    axes[0].set_xlim(lo[0]-margin, hi[0]+margin); axes[0].set_ylim(lo[1]-margin, hi[1]+margin)
    for axis in axes[len(labels):]: axis.set_visible(False)
    fig.suptitle("MuJoCo-only adaptive MPC DEVELOPMENT · not sim2sim / not certified / not ranked", fontsize=12)
    fig.legend(handles=[Line2D([], [], color=".3", lw=2.2, label="Thick: declared core"),
                        Line2D([], [], color=".3", lw=1, label="Ticks: wheel rolling planes; arrows: XY velocity"),
                        Line2D([], [], color=".7", lw=1, label="Pale measured ticks: no gated wheel-ground contact")],
               loc="lower center", bbox_to_anchor=(.5, .01), ncol=1 if columns == 2 else 3, fontsize=8)
    fig.tight_layout(rect=(0, .12 if columns == 2 else .10, 1, .94))
    fig.savefig(output, dpi=160); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing output; choose a NEW output directory")
    directories = parse_runs(args.run)
    spec = verify_reference_pair(args.motion_file, args.reference_file, PRIMARY_REFERENCE_ID)
    require(spec["frames"] == 349 and spec["fps"] == 50., "Require complete primary reference")
    traces = {"reference": reference_trace(load_arrays(args.motion_file))}
    reports, audits = {}, {}
    for label, directory in directories.items():
        files = {name: directory/name for name in ("report.json", "rollout.npz", "solver_log.json")}
        digests = {name: sha256(path) for name, path in files.items()}
        report = read_json(files["report.json"])
        audit = audit_report(report, spec, model_sha256=sha256(args.model),
                             trace_sha256=digests["rollout.npz"], solver_sha256=digests["solver_log.json"])
        solver = json.loads(files["solver_log.json"].read_text())
        require(isinstance(solver, list), "Solver log must be a decision list")
        raw = load_arrays(files["rollout.npz"])
        trace = audit_trace(raw, solver, spec)
        diagnostic = solver_diagnostic(report, raw, solver, audit["parameterization"])
        adaptation = adaptation_diagnostic(report, solver, raw)
        planning_steps = diagnostic["counts"]["scenario_evaluations"]*audit["effective_cem_config"]["horizon_steps"]*4
        identification_steps = adaptation.get("identification_predicted_physics_substeps", 0)
        budget = {"planning_predicted_physics_substeps": planning_steps,
                  "identification_predicted_physics_substeps": identification_steps,
                  "total_predicted_physics_substeps": planning_steps+identification_steps,
                  "actual_plant_physics_substeps": 1396,
                  "scope": "Declared rollout budget from saved calls/horizon. Failed batches are counted as requested, not independently verified completed native steps."}
        for previous in reports.values():
            require(report["objective_contract"] == previous["objective_contract"], "Different objective contracts")
        if audits:
            require(audit["actuator_identity"] == next(iter(audits.values()))["actuator_identity"], "Different actuator contracts")
        traces[label], reports[label] = trace, report
        audits[label] = {**audit, "input_sha256": digests, "solver_diagnostics": diagnostic,
                         "adaptation_diagnostics": adaptation, "simulation_budget_diagnostic": budget,
                         "reported_plant_source_provenance": plant_source_provenance(report)}
    require(any(item["controller_kind"] == ADAPTIVE_KIND for item in audits.values()), "At least one adaptive episode required")
    with source_model(args.source_root, args.model) as (model, loader_audit):
        fk = WheelKinematics(model)
        reconstructions = {label: fk.reconstruct(trace) for label, trace in traces.items()}
    reference, interval = traces["reference"], spec["evaluation_core_interval_s"]
    core = {}
    for label, trace in traces.items():
        contact = {} if label == "reference" else dict(wheel_ground_contact_count=trace["contact_count"],
                  wheel_ground_force_w=trace["contact_force"], wheel_contact_lateral_slip_m_s=trace["contact_slip"])
        core[label] = summarize_turn(trace["time_s"], trace["position"], reconstructions[label]["signals"],
                      reference["time_s"], reference["position"][:, :2],
                      reconstructions["reference"]["signals"]["speed_m_s"], interval, **contact)
    summary = {"schema_version": 1, "kind": "mujoco_only_adaptive_mpc_development",
               "status": "development_analysis_only", "engine": "MuJoCo", "reference_id": PRIMARY_REFERENCE_ID,
               "reference_spec": spec, "run_order": list(directories), "runs": audits, "windows": {"core": core},
               "sim2sim_comparison": False, "frozen_benchmark": False, "eligible_for_controller_ranking": False,
               "continuous_uncertainty_robustness_certified": False, "controlled_execution_success_claimed": False,
               "pooled_success_statistics": False, "height_or_flight_failure_gate": False,
               "heading_offset_is_mismatch": False, "dynamics_stepping": False, "learned_inference": False,
               "identification_bank_is_worst_case_scenario_set": False,
               "objective_contract": safe_configuration(next(iter(reports.values()))["objective_contract"]),
               "model_loader_audit": loader_audit,
               "fk_audit": {label: result["audit"] for label, result in reconstructions.items()},
               "posture_diagnostics": {label: posture_diagnostic(traces[label]) for label in directories},
               "contact_diagnostics": {label: contact_diagnostic(traces[label], interval) for label in directories},
               "input_sha256": {name: sha256(getattr(args, name)) for name in ("motion_file", "reference_file", "model")},
               "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
               "analysis_source_sha256": {name: sha256(ROOT/"scripts"/name) for name in (
                   "analyze_adaptive_mpc.py", "analyze_nmpc_nominal_development.py", "analyze_controller_pair.py",
                   "controller_metrics.py", "controller_trace.py", "controller_references.py", "nmpc_tracking_objective.py")},
               "limitations": ["MuJoCo-only finite contact-model adaptation; no Isaac result, sim2sim qualification, ranking, hardware or robustness certificate.",
                   "Identification chooses one bank member using completed transitions, not an oracle plant-friction value or worst-case scenario reduction.",
                   "Timings are original uncontrolled wall-clock observations. Simulator waits for optimization; no actuation-delay model or real-time claim.",
                   "Height, flight, heading offsets and contact loss are descriptive, not additional failure gates. XY alone does not certify upright controlled execution.",
                   "Reference contacts are unmeasured. Native clocks and world axes are unchanged; no trajectory alignment or time warp.",
                   "Plots show source-demo interval only; metrics use the catalog core. Full playback includes synthetic recovery/tail."]}
    compact = compact_arrays(traces, reconstructions)
    output.mkdir(parents=True)
    np.savez_compressed(output/"development_trace.npz", **compact)
    render_topdown(compact, summary, output/"topdown.png")
    summary["artifacts"] = {"compact_trace": "development_trace.npz", "compact_trace_sha256": sha256(output/"development_trace.npz"),
                            "topdown": "topdown.png", "topdown_sha256": sha256(output/"topdown.png"),
                            "replay": "Use render_topdown with development_trace.npz and this summary; original native clock and measured contact masks retained"}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--motion-file", "--motion", dest="motion_file", type=Path, required=True)
    for name in ("reference-file", "model", "source-root", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = run(arguments)
    print(json.dumps({"status": result["status"], "runs": result["run_order"], "sim2sim": False}))

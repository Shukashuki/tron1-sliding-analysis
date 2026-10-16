"""Offline MuJoCo TVLQR pilot: reference, pure-model NMPC donor, and feedback.

The NMPC nominal is shared initialization, not independent trajectory design.
The TVLQR quadratic design cost is not the common 15-term evaluation reward.
No dynamics, policy inference, alignment, controller ranking, or height gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from analyze_controller_pair import (compact_arrays, load_arrays, plant_source_provenance,
    posture_diagnostic, read_json, reference_trace, sha256, source_model)
from analyze_nmpc_nominal_development import (audit_report as audit_donor_report, audit_trace,
    contact_diagnostic, parse_runs, require, safe_configuration, solver_diagnostic, source_digest)
from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT, verify_reference_pair
from controller_trace import WheelKinematics
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES

TVLQR_KIND = "model_nominal_initialized_height_smooth_tvlqr"
DONOR_KIND = "full_body_height_smooth_cem_nmpc"
PUBLIC_TRACE_FIELDS = ("time_s", "physics_dt_s", "joint_names", "position", "quaternion", "com_velocity",
    "angular_velocity", "wheel_pos", "contact_count", "contact_force", "contact_slip",
    "nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n")
PUBLIC_SIGNALS = ("speed_m_s", "wheel_center_speed_m_s", "course_rad", "body_heading_rad",
    "wheel_rolling_plane_heading_rad", "root_directed_beta_rad", "root_course_to_wheel_gamma_rad",
    "wheel_center_course_to_wheel_gamma_rad", "root_beta_valid", "root_course_to_wheel_valid",
    "wheel_center_course_to_wheel_valid", "speed_threshold_m_s")


def public_compact(traces, reconstructions):
    public = {label: {key: trace[key] for key in PUBLIC_TRACE_FIELDS if key in trace}
              for label, trace in traces.items()}
    geometry = {label: {"signals": {key: value["signals"][key] for key in PUBLIC_SIGNALS},
        **{key: value[key] for key in ("wheel_quaternion_wxyz", "wheel_center_velocity_w")}}
        for label, value in reconstructions.items()}
    return compact_arrays(public, geometry)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256 identity")
    return value


def audit_tvlqr_report(report, spec, *, model_sha256, trace_sha256, log_sha256):
    require(report.get("kind") == TVLQR_KIND and report.get("status") == "completed"
            and report.get("engine") == "MuJoCo", "Require completed MuJoCo TVLQR feedback")
    require(report.get("reference_id") == spec["reference_id"] == PRIMARY_REFERENCE_ID
            and report.get("motion_sha256") == spec["motion_npz_sha256"]
            and report.get("model_sha256") == model_sha256
            and report.get("reference_catalog_sha256") == sha256(DEFAULT_CATALOG), "Model/reference identity mismatch")
    require(report.get("rollout_sha256") == trace_sha256 and report.get("feedback_log_sha256") == log_sha256,
            "TVLQR report-bound trace/feedback SHA-256 mismatch")
    require(report.get("plant_mu") == report.get("nominal_prediction_mu") == .7
            and report.get("plant_material_readback", {}).get("compiled_sliding_friction_range") == [.7, .7],
            "Initial TVLQR pilot requires nominal plant and design mu=.7")
    require(report.get("completed_policy_steps") == report.get("evaluated_policy_steps")
            == report.get("total_reference_policy_steps") == 349 and report.get("physics_samples") == 1397
            and np.isclose(report.get("duration_s", np.nan), 6.98, rtol=0, atol=1e-10)
            and report.get("physics_dt_s") == .005 and report.get("control_dt_s") == .02,
            "Require complete 349-decision/1397-sample native clocks")
    for name in ("learned_policy_used", "action_replay", "root_state_prescribed_after_reset",
                 "external_root_forces_used", "prediction_receives_plant_mu", "truncated_smoke",
                 "pose_height_failure_gate", "height_or_flight_failure_gate", "heading_offset_is_mismatch",
                 "real_time_verified", "robustness_certificate", "sim2sim_comparison", "design_cost_equals_15_term_objective"):
        require(report.get(name) is False, "Missing/unsafe TVLQR declaration: " + name)
    objective = safe_configuration(report.get("objective_contract"))
    require(isinstance(objective, dict) and objective.get("absolute_height_target") is False
            and objective.get("reference_vertical_velocity_target") is False and objective.get("dt_s") == .02
            and objective.get("height_regularization", {}).get("mode") == "vertical_acceleration_only"
            and objective.get("body_names") == list(BODY_NAMES) and objective.get("joint_names") == list(JOINT_NAMES)
            and len(objective.get("weights", {})) == 15, "Require common height-free 15-term evaluation objective")
    design_cost = safe_configuration(report.get("design_objective_contract"))
    require(isinstance(design_cost, dict) and design_cost.get("kind") == "quadratic_model_generated_nominal_surrogate_not_original_15_term_reward"
            and all(design_cost.get(name) is False for name in ("absolute_height_tracking", "absolute_vz_tracking",
                "sideslip_zero_target", "flight_failure_gate", "original_reward_matched")), "Missing/unmatched quadratic design declaration")
    donor = report.get("donor_identity", {})
    archives = donor.get("archive_sha256", {})
    require(set(archives) == {"report.json", "rollout.npz", "solver_log.json"}
            and donor.get("learned_policy_used") is False, "Require explicitly model-only NMPC donor identity")
    for value in archives.values(): digest(value)
    budget = safe_configuration(donor.get("generation_budget"))
    require(isinstance(budget, dict) and all(type(budget.get(name)) is int and budget[name] >= 0
            for name in ("scenario_evaluations", "predicted_physics_substeps", "plant_physics_substeps")),
            "Missing charged offline NMPC donor budget")
    design_budget = safe_configuration(report.get("offline_design_budget"))
    require(isinstance(design_budget, dict) and all(type(design_budget.get(name)) is int
            and design_budget[name] >= 0 for name in ("attempted_transition_calls", "completed_transition_calls", "predicted_physics_substeps"))
            and design_budget["attempted_transition_calls"] == design_budget["completed_transition_calls"] == 349*73
            and design_budget["predicted_physics_substeps"] == 349*73*4
            and type(design_budget.get("wall_s")) in (int, float) and np.isfinite(design_budget["wall_s"])
            and design_budget["wall_s"] >= 0, "Missing/inconsistent charged offline TVLQR design budget")
    for source in ("scripts/run_tvlqr_tracking.py", "scripts/tvlqr_core.py", "scripts/design_tvlqr_nominal.py"):
        source_digest(report, source)
    return {"controller_kind": TVLQR_KIND, "actual_plant_mu": .7,
            "design_sha256": digest(report.get("design_sha256")),
            "design_report_sha256": digest(report.get("design_report_sha256")),
            "actuator_contract_sha256": digest(report.get("actuator_contract_sha256")),
            "design_objective_contract": design_cost,
            "design_cost_equals_15_term_objective": False,
            "donor_identity": {"archive_sha256": archives.copy(), "learned_policy_used": False,
                "generation_budget": budget, "source_provenance": plant_source_provenance(donor)},
            "offline_design_budget": design_budget,
            "classical_actuator_helper_sha256": source_digest(report, "scripts/classical_tracking.py"),
            "actuator_setup_helper_sha256": source_digest(report, "scripts/eval_tracking_mujoco.py")}


def audit_physical_state(raw, trace):
    for name, width in (("qpos", 15), ("qvel", 14)):
        value = np.asarray(raw[name])
        require(value.shape == (1397, width) and value.dtype.kind in "iuf" and np.isfinite(value).all(),
                "Require full finite recorded " + name)
    require(np.allclose(raw["qpos"][:, :3], trace["position"], atol=1e-10, rtol=0)
            and np.allclose(np.abs(np.sum(raw["qpos"][:, 3:7] * trace["quaternion"], axis=1)), 1., atol=1e-5, rtol=0),
            "Recorded qpos differs from measured root pose")
    names = tuple(np.asarray(raw["qpos_joint_names"]).tolist())
    require(len(names) == 8 and set(names) == set(JOINT_NAMES), "Recorded qpos joint names differ")
    order = np.array([names.index(name) for name in JOINT_NAMES])
    require(np.allclose(raw["qpos"][:, 7+order], trace["joint_pos"], atol=1e-10, rtol=0)
            and np.allclose(raw["qvel"][:, 6+order], trace["joint_vel"], atol=1e-10, rtol=0),
            "Recorded qpos/qvel differ from named joint samples")


def feedback_diagnostic(report, raw, log):
    compute = np.asarray(raw["compute_s"], dtype=float)
    require(compute.shape == (349,) and np.isfinite(compute).all() and np.all(compute >= 0)
            and np.array_equal(compute, [row.get("compute_s") for row in log])
            and np.allclose(report.get("compute_ms_median_p95_max", []), 1000*np.quantile(compute, [.5, .95, 1.]), atol=1e-8, rtol=1e-12),
            "Feedback timing differs from original trace/log")
    clipped = 0
    for row, applied in zip(log, raw["actions"]):
        info = row.get("feedback", {}); command = np.asarray(info.get("raw_action", []), dtype=float)
        require(command.shape == (8,) and np.isfinite(command).all()
                and np.array_equal(np.clip(command, -1., 1.).astype(np.float32).astype(float), applied)
                and info.get("online_optimization") is False and info.get("action_replay") is False,
                "Feedback raw action does not match actual clipped float32 command")
        count = int(np.count_nonzero(np.abs(command) > 1.))
        require(type(info.get("clipped_action_components")) is int and info["clipped_action_components"] == count,
                "Feedback clipping counter differs from raw command")
        clipped += count
    require(report.get("clipped_action_component_fraction") == clipped/(349*8), "Feedback clipping fraction mismatch")
    return {"decision_samples": 349, "compute_ms_median_p95_max": (1000*np.quantile(compute, [.5, .95, 1.])).tolist(),
            "clipped_action_component_fraction": clipped/(349*8), "online_optimization": False,
            "timing_scope": "Original feedback-only wall times, not matched hardware/total design compute or real-time evidence"}


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = ["reference", *summary["run_order"]]
    fig, grid = plt.subplots(1, len(labels), figsize=(4.7*len(labels), 5.), sharex=True, sharey=True, squeeze=False)
    start, end = summary["reference_spec"]["source_demo_interval_s"]
    lo_core, hi_core = summary["reference_spec"]["evaluation_core_interval_s"]
    rt = compact["reference_time_s"]; ref = compact["reference_position"][(rt >= start-1e-9) & (rt <= end+1e-9), :2]
    bounds = [ref]
    for j, (axis, label) in enumerate(zip(grid.ravel(), labels)):
        color = ".35" if label == "reference" else plt.get_cmap("tab10")(j-1)
        t, pos, wheels = [compact[f"{label}_{name}"] for name in ("time_s", "position", "wheel_pos")]
        selected = (t >= start-1e-9) & (t <= end+1e-9); core = selected & (t >= lo_core-1e-9) & (t <= hi_core+1e-9)
        axis.plot(ref[:, 0], ref[:, 1], "--", color=".75", lw=1)
        axis.plot(pos[selected, 0], pos[selected, 1], color=color, lw=1, alpha=.5)
        axis.plot(pos[core, 0], pos[core, 1], color=color, lw=2.2)
        indices = np.flatnonzero(selected)[::max(1, round(.4/float(compact[f"{label}_physics_dt_s"])))]
        for side in range(2):
            axis.plot(wheels[selected, side, 0], wheels[selected, side, 1], color=color, lw=.5, alpha=.35)
            for index in indices:
                if not compact[f"{label}_root_course_to_wheel_valid"][index, side]: continue
                grounded = label == "reference" or (compact[f"{label}_contact_count"][index, side] > 0
                    and compact[f"{label}_contact_force"][index, side, 2] > 5.)
                angle = compact[f"{label}_wheel_rolling_plane_heading_rad"][index, side]
                delta = .055*np.array([np.cos(angle), np.sin(angle)])
                segment = wheels[index, side, :2] + np.array([-delta, delta])
                axis.plot(segment[:, 0], segment[:, 1], color=color, lw=1.2, alpha=.9 if grounded else .15)
        arrows = indices[compact[f"{label}_root_beta_valid"][indices]]; course = compact[f"{label}_course_rad"][arrows]
        axis.quiver(pos[arrows, 0], pos[arrows, 1], np.cos(course), np.sin(course), color=color,
                    angles="xy", scale_units="xy", scale=7, width=.003)
        axis.set(title="Kinematic reference (contacts unmeasured)" if label == "reference" else label + " · MuJoCo μ=0.7",
                 xlabel="World X [m]", ylabel="World Y [m]")
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
        bounds.extend((pos[selected, :2], wheels[selected, :, :2].reshape(-1, 2)))
    xy = np.concatenate(bounds); low, high = xy.min(axis=0), xy.max(axis=0); margin = max(.15, .06*float(np.max(high-low)))
    grid[0, 0].set_xlim(low[0]-margin, high[0]+margin); grid[0, 0].set_ylim(low[1]-margin, high[1]+margin)
    fig.suptitle("NMPC-donor-initialized TVLQR DEVELOPMENT · MuJoCo only / not matched-cost / not ranked", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=".3", lw=2.2, label="Thick: core; arrows: XY velocity"),
        Line2D([], [], color=".3", lw=1, label="Ticks: wheel rolling planes; pale: no measured contact")], loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .92)); fig.savefig(output, dpi=160); plt.close(fig)


def run(args):
    output = Path(args.output_dir); require(not output.exists(), "Preserve existing output; choose a NEW directory")
    directories = parse_runs(args.run)
    spec = verify_reference_pair(args.motion_file, args.reference_file, PRIMARY_REFERENCE_ID)
    traces = {"reference": reference_trace(load_arrays(args.motion_file))}; reports = {}; audits = {}
    for label, directory in directories.items():
        report = read_json(directory/"report.json"); tvlqr = report.get("kind") == TVLQR_KIND
        log_name = "feedback_log.json" if tvlqr else "solver_log.json"
        hashes = {name: sha256(directory/name) for name in ("report.json", "rollout.npz", log_name)}
        raw, log = load_arrays(directory/"rollout.npz"), json.loads((directory/log_name).read_text())
        require(isinstance(log, list), "Require genuine per-decision feedback/solver log")
        if tvlqr:
            audit = audit_tvlqr_report(report, spec, model_sha256=sha256(args.model), trace_sha256=hashes["rollout.npz"], log_sha256=hashes[log_name])
            audit["feedback_diagnostics"] = feedback_diagnostic(report, raw, log)
        else:
            audit = audit_donor_report(report, spec, model_sha256=sha256(args.model), trace_sha256=hashes["rollout.npz"], solver_sha256=hashes[log_name])
            audit.update(controller_kind=DONOR_KIND, actual_plant_mu=.7, solver_diagnostics=solver_diagnostic(report, raw, log, audit["parameterization"]))
        trace = audit_trace(raw, log, spec); audit_physical_state(raw, trace)
        stage = np.asarray(raw["stage_cost"], dtype=float)
        require(stage.shape == (349,) and np.isfinite(stage).all()
                and np.array_equal(stage, [row.get("objective", {}).get("stage_cost") for row in log])
                and np.isclose(report.get("total_stage_cost", np.nan), stage.sum(), atol=1e-9, rtol=1e-12),
                "Common evaluation stage-cost log/trace/report mismatch")
        require(not reports or report["objective_contract"] == next(iter(reports.values()))["objective_contract"], "Different common evaluation objective contracts")
        traces[label], reports[label] = trace, report
        audits[label] = {**audit, "input_sha256": hashes, "reported_source_provenance": plant_source_provenance(report)}
    donors = [value for value in audits.values() if value["controller_kind"] == DONOR_KIND]
    feedback = [value for value in audits.values() if value["controller_kind"] == TVLQR_KIND]
    require(donors and feedback, "Require both an actual NMPC donor and a TVLQR feedback run")
    for item in feedback:
        matching = [donor for donor in donors if donor["input_sha256"] == item["donor_identity"]["archive_sha256"]]
        require(len(matching) == 1, "TVLQR nominal donor bytes do not match supplied NMPC episode")
        require(item["actuator_contract_sha256"] == matching[0]["actuator_identity"]["policy_contract_sha256"]
                and item["classical_actuator_helper_sha256"] == matching[0]["actuator_identity"]["actuator_helper_sha256"]
                and item["actuator_setup_helper_sha256"] == matching[0]["actuator_identity"]["actuator_setup_helper_sha256"],
                "Different shared actuator/helper identity")
        budget = item["donor_identity"]["generation_budget"]
        diagnostic = matching[0]["solver_diagnostics"]
        require(budget["scenario_evaluations"] == diagnostic["counts"]["scenario_evaluations"]
                and budget["predicted_physics_substeps"] == diagnostic["counts"]["scenario_evaluations"]*matching[0]["effective_cem_config"]["horizon_steps"]*4
                and budget["plant_physics_substeps"] == 1396
                and budget.get("wall_s") == diagnostic["timing"]["recorded_rollout_wall_s"],
                "Charged donor budget differs from actual supplied NMPC episode")
    with source_model(args.source_root, args.model) as (model, loader):
        fk = WheelKinematics(model); recon = {label: fk.reconstruct(trace) for label, trace in traces.items()}
    reference, interval = traces["reference"], spec["evaluation_core_interval_s"]
    metrics = {}
    for label, trace in traces.items():
        contact = {} if label == "reference" else dict(wheel_ground_contact_count=trace["contact_count"], wheel_ground_force_w=trace["contact_force"], wheel_contact_lateral_slip_m_s=trace["contact_slip"])
        metrics[label] = summarize_turn(trace["time_s"], trace["position"], recon[label]["signals"], reference["time_s"], reference["position"][:, :2], recon["reference"]["signals"]["speed_m_s"], interval, **contact)
    summary = {"schema_version": 1, "kind": "mujoco_only_model_nominal_initialized_tvlqr_pilot", "status": "development_analysis_only",
        "engine": "MuJoCo", "reference_id": PRIMARY_REFERENCE_ID, "reference_spec": spec, "run_order": list(directories), "runs": audits,
        "windows": {"core": metrics}, "objective_contract": safe_configuration(next(iter(reports.values()))["objective_contract"]),
        **{name: False for name in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking", "matched_design_cost", "independent_trajectory_design",
            "controlled_execution_success_claimed", "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "learned_inference")},
        "model_loader_audit": loader, "fk_audit": {label: value["audit"] for label, value in recon.items()},
        "posture_diagnostics": {label: posture_diagnostic(traces[label]) for label in directories},
        "contact_diagnostics": {label: contact_diagnostic(traces[label], interval) for label in directories},
        "input_sha256": {name: sha256(getattr(args, name)) for name in ("motion_file", "reference_file", "model")},
        "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
        "analysis_source_sha256": {name: sha256(ROOT/"scripts"/name) for name in ("analyze_tvlqr_pilot.py", "analyze_nmpc_nominal_development.py", "analyze_controller_pair.py", "controller_metrics.py", "controller_trace.py")},
        "limitations": ["TVLQR uses a pure-model NMPC donor, not an independent nominal or PPO initialization; quadratic design and 15-term evaluation costs differ.",
            "Offline donor generation and TVLQR linearization/design costs are additional to feedback timing; unavailable design-budget fields are null, not zero.",
            "MuJoCo-only exploratory result: no ranking, Isaac qualification, robust stability/constraints, hardware or real-time claim.",
            "Original clocks/world axes retained. Reference contacts are unmeasured; height, flight, contact loss and heading offsets are descriptive, not failure gates."]}
    compact = public_compact(traces, recon); output.mkdir(parents=True)
    np.savez_compressed(output/"development_trace.npz", **compact); render_topdown(compact, summary, output/"topdown.png")
    summary["artifacts"] = {name: {"file": filename, "sha256": sha256(output/filename)} for name, filename in (("compact_trace", "development_trace.npz"), ("topdown", "topdown.png"))}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--motion", "--motion-file", dest="motion_file", type=Path, required=True)
    for name in ("reference-file", "model", "source-root", "output-dir"): result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args()); print(json.dumps({"status": result["status"], "runs": result["run_order"]}))

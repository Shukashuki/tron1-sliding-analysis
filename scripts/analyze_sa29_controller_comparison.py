"""Offline SA29 two-engine controller DEVELOPMENT, separate from the primary turn.

Read fixed-NMPC recordings and complete SA29-specific policy recordings only.
No simulation, policy inference, registration, tuning or qualified freeze.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

from analyze_controller_pair import (load_arrays, pair_xy_gap, posture_diagnostic,
    read_json, reference_trace, sha256, source_model)
from analyze_nmpc_friction_pair_development import audit_isaac_header, summarize_trace
from analyze_nmpc_friction_development import PILOT_CEM
from analyze_nmpc_sa29_friction_development import (GRID, SELECTED_MU,
    PREDECLARATION_SHA256, audit_predeclaration, case_name, parse_runs)
from analyze_nmpc_sa29_pilot import CORE, EVENTS, FRAMES, SAMPLES, REFERENCE_ID
from analyze_nmpc_nominal_development import contact_diagnostic, require, safe_configuration
from analyze_tvlqr_pilot import PUBLIC_SIGNALS, PUBLIC_TRACE_FIELDS, public_compact
from controller_references import DEFAULT_CATALOG, ROOT, verify_reference_pair
from controller_metrics import heading_signals
from controller_trace import WheelKinematics, native_trace
from nmpc_validation_contract import episode_evidence, validate_case_against_nominal


MUJOCO_BUNDLE_SHA256 = {
    "summary.json": "386004dce4309a9d95114759f71b19486be38cc0b6498d6187b686846438669a",
    "development_trace.npz": "4d0cbe38eeea069eb773404abb1d966a91db49c627d350ed5a72fdefbc525388",
    "topdown.png": "1d3a7971a0eb7c03a340a7297390ec37cc670045b0c5262c443140baac5d37df",
}
ISAAC_ASSET_SHA256 = "327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e"


def compact_case(data, prefix):
    """Require exact complete SA29 clocks; never resample the primary bundle."""
    trace = {name: data[prefix+"_"+name].copy() for name in PUBLIC_TRACE_FIELDS
             if prefix+"_"+name in data}
    geometry = {"signals": {name: data[prefix+"_"+name].copy() for name in PUBLIC_SIGNALS},
        **{name: data[prefix+"_"+name].copy() for name in
           ("wheel_quaternion_wxyz", "wheel_center_velocity_w")}}
    count, dt = (FRAMES, .02) if prefix == "reference" else (SAMPLES, .005)
    require(trace.get("time_s", np.empty(0)).shape == (count,)
            and np.allclose(trace["time_s"], np.arange(count)*dt, atol=1e-9, rtol=0)
            and float(trace.get("physics_dt_s", np.nan)) == dt, "Incomplete SA29 compact clock")
    for name, width in (("position", (3,)), ("quaternion", (4,)), ("com_velocity", (3,)),
        ("angular_velocity", (3,)), ("wheel_pos", (2, 3))):
        value = trace.get(name, np.empty(0))
        require(value.shape == (count, *width) and np.isfinite(value).all(),
                "Invalid compact measured state: "+name)
    require(np.allclose(np.linalg.norm(trace["quaternion"], axis=1), 1., atol=1e-5, rtol=0),
            "Compact root quaternion is not normalized")
    for value in (*geometry["signals"].values(), geometry["wheel_quaternion_wxyz"],
                  geometry["wheel_center_velocity_w"]):
        require(value.dtype.kind in "biuf" and np.isfinite(value).all(), "Invalid compact wheel geometry")
    reconstructed = heading_signals(trace["quaternion"], trace["com_velocity"],
        geometry["wheel_quaternion_wxyz"], geometry["wheel_center_velocity_w"])
    require(all(np.array_equal(reconstructed[name], geometry["signals"][name]) for name in PUBLIC_SIGNALS),
            "Compact speed/heading must replay from actual recorded velocities and wheel planes")
    if prefix != "reference":
        for name, width in (("contact_count", (2,)), ("contact_force", (2, 3)),
                           ("contact_slip", (2,)), ("nonwheel_ground_contact_count", ()),
                           ("nonwheel_ground_contact_force_norm_n", ())):
            value = trace.get(name, np.empty(0))
            require(value.shape == (count, *width) and np.isfinite(value).all(),
                    "Missing measured compact contact: "+name)
        require(np.all(trace["contact_count"] >= 0)
                and np.array_equal(trace["contact_count"], np.round(trace["contact_count"]))
                and np.all(trace["contact_slip"] >= 0), "Invalid compact measured contact")
    return trace, geometry


def windows(pair, geometry, reference, reference_geometry):
    def at(interval):
        metrics = {name: summarize_trace(trace, geometry[name], reference, reference_geometry, interval)
                   for name, trace in pair.items()}
        if set(pair) == {"isaac", "mujoco"}:
            metrics["sim2sim_xy_gap"] = pair_xy_gap(pair["isaac"], pair["mujoco"], interval)
        return metrics
    return {"core": at(CORE), "supplementary_events": [
        {"interval_s": interval, "descriptive_not_gate": True, "metrics": at(interval)} for interval in EVENTS]}


def load_mujoco_bundle(directory, spec, inputs, reference):
    directory = Path(directory); summary = read_json(directory/"summary.json")
    require({name: sha256(directory/name) for name in MUJOCO_BUNDLE_SHA256} == MUJOCO_BUNDLE_SHA256,
            "The original published 17-case SA29 MuJoCo bundle must be reused byte-for-byte")
    require(summary.get("kind") == "separate_sa29_mujoco_fixed_nmpc_friction_development"
            and summary.get("status") == "development_analysis_only"
            and summary.get("reference_spec") == spec and summary.get("input_sha256") == inputs
            and summary.get("complete_predeclared_grid") is True
            and summary.get("evaluated_mu") == list(GRID)
            and summary.get("predeclaration_sha256") == PREDECLARATION_SHA256,
            "Require the complete separately declared SA29 MuJoCo bundle")
    declaration = audit_predeclaration()
    require(summary.get("effective_cem_config_sha256") == declaration["effective_cem_configuration_sha256"]
            and summary.get("effective_cem_config") == PILOT_CEM
            and summary.get("prediction_mu_scenarios") == [.7], "MuJoCo fixed-controller identity changed")
    for flag in ("pooled_success_statistics", "primary_task_replicate", "two_engine_nominal_qualified",
        "qualified_freeze_created", "eligible_for_controller_ranking", "height_or_flight_failure_gate",
        "heading_offset_is_mismatch", "controller_retuned"):
        require(summary.get(flag) is False, "Unexpected prior qualification/gate: "+flag)
    require(summary.get("analysis_source_sha256", {}).get("analyze_nmpc_sa29_friction_development.py")
            == sha256(ROOT/"scripts/analyze_nmpc_sa29_friction_development.py"), "Changed SA29 MuJoCo analyzer")
    hashes = {"summary.json": sha256(directory/"summary.json")}
    for role, name in (("compact_trace", "development_trace.npz"), ("topdown", "topdown.png")):
        hashes[name] = sha256(directory/name)
        require(summary.get("artifacts", {}).get(role) == {"file": name, "sha256": hashes[name]},
                "MuJoCo bundle artifact hash differs")
    data = load_arrays(directory/"development_trace.npz")
    saved_reference, ref_geometry = compact_case(data, "reference")
    for name in ("time_s", "position", "quaternion", "com_velocity", "angular_velocity", "wheel_pos"):
        require(np.array_equal(saved_reference[name], reference[name]), "Compact reference was altered/aligned")
    traces, geometries = {}, {}
    for mu in GRID:
        case = case_name(mu); trace, geometry = compact_case(data, case)
        require(summary["runs"][case].get("complete_physical_evidence") is True
                and summary["runs"][case].get("censored") is False, "Incomplete original MuJoCo recording")
        replay = windows({"mujoco": trace}, {"mujoco": geometry}, reference, ref_geometry)
        require(replay["core"]["mujoco"] == summary["windows"]["core"][case], "MuJoCo core fails exact replay")
        for actual, recorded in zip(replay["supplementary_events"], summary["windows"]["supplementary_events"]):
            require(actual["interval_s"] == recorded["interval_s"]
                    and actual["metrics"]["mujoco"] == recorded["metrics"][case], "MuJoCo event fails exact replay")
        require(posture_diagnostic(trace) == summary["posture_diagnostics"][case]
                and contact_diagnostic(trace, CORE) == summary["contact_diagnostics"][case]["core"],
                "MuJoCo posture/contact fails exact replay")
        traces[case], geometries[case] = trace, geometry
    return summary, hashes, data, ref_geometry, traces, geometries


def audit_isaac_trace(report, raw, log):
    """SA29-specific native state/action/score checks, without field relabeling."""
    trace = native_trace(raw, "nmpc")
    require(len(trace["time_s"]) == SAMPLES and trace["physics_dt_s"] == .005
            and np.allclose(trace["time_s"], np.arange(SAMPLES)*.005, atol=1e-9, rtol=0)
            and isinstance(log, list) and len(log) == FRAMES
            and float(raw["policy_dt_s"]) == .02, "Require all SA29 decisions and physical samples")
    counter = np.asarray(raw["simulation_step_counter"])
    require(counter.shape == (SAMPLES,) and counter.dtype.kind in "iu"
            and np.array_equal(counter-counter[0], np.arange(SAMPLES)), "Isaac physical clock skipped/reset")
    for name in ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"):
        value = np.asarray(raw[name])
        require(value.shape == (SAMPLES,) and np.isfinite(value).all() and np.all(value >= 0),
                "Invalid measured nonwheel-contact evidence")
        if name.endswith("count"):
            require(np.array_equal(value, np.round(value)), "Noninteger measured contact count")
        trace[name] = value.copy()
    packed = np.asarray(raw["packed_controller_states"])
    require(packed.shape == (FRAMES, 111) and np.isfinite(packed).all()
            and np.allclose(packed[:, :3], trace["position"][::4][:-1], atol=1e-9, rtol=0)
            and np.allclose(np.linalg.norm(packed[:, 3:7], axis=1), 1., atol=1e-5, rtol=0)
            and np.allclose(np.abs(np.sum(packed[:, 3:7]*trace["quaternion"][::4][:-1], axis=1)), 1., atol=1e-5, rtol=0),
            "Pre-action converted root state differs from actual measured state")
    commands = np.asarray([row.get("raw_cem_action", []) for row in log])
    require(commands.shape == (FRAMES, 8) and np.isfinite(commands).all()
            and np.max(np.abs(commands)) <= 1.+1e-9
            and np.array_equal(commands.astype(np.float32).astype(float), raw["actions"]),
            "Raw CEM command differs from actual float32 actuation")
    compute = np.asarray(raw["compute_s"])
    require(compute.shape == (FRAMES,) and np.isfinite(compute).all() and np.all(compute >= 0)
            and np.array_equal(compute, [row.get("compute_s") for row in log])
            and np.allclose(report.get("compute_ms_median_p95_max", []),
                1000*np.quantile(compute, [.5,.95,1.]), atol=1e-8, rtol=1e-12)
            and report.get("compute_over_control_dt_fraction") == float(np.mean(compute > .02)),
            "Original 258-solve timing trace/report differs")
    count_names = ("candidate_evaluations", "finite_candidates", "failed_candidates", "scenario_evaluations",
        "failed_scenarios", "batch_prediction_calls", "failed_batch_prediction_calls")
    flag_names = ("optimization_failed", "fallback_used", "warm_start_used", "initial_candidate_generation_failed")
    totals = {name: 0 for name in (*count_names, *flag_names)}
    weights = report["objective_contract"]["weights"]; upper = sum(v for v in weights.values() if v > 0)
    require(len(weights) == 15, "Require the same 15-term objective")
    reward_errors, conversion_errors, costs, scopes = [], [], [], set()
    for k, row in enumerate(log):
        info, score = row.get("solver", {}), row.get("actual_isaac_objective", {})
        require(row.get("frame") == k and row.get("reference_frame_for_reward") == k
                and np.array_equal(row.get("action"), raw["actions"][k]),
                "Decision action/reference phase differs from applied trace")
        for name in count_names:
            require(type(info.get(name)) is int and info[name] >= 0, "Invalid solver counter: "+name)
            totals[name] += info[name]
        for name in flag_names:
            require(type(info.get(name)) is bool, "Invalid solver flag: "+name)
            totals[name] += int(info[name])
        require(info["finite_candidates"]+info["failed_candidates"] == info["candidate_evaluations"]
                == info["scenario_evaluations"] and info["failed_scenarios"] == info["failed_candidates"]
                and info["failed_batch_prediction_calls"] <= info["batch_prediction_calls"]
                and info["fallback_used"] == info["optimization_failed"]
                and info.get("control_deadline_enforced") is False and info.get("real_time_verified") is False,
                "Invalid counter/fallback/deadline declarations")
        require(isinstance(info.get("timing_scope"), str), "Missing solver timing scope")
        scopes.add(info["timing_scope"])
        if info["fallback_used"]:
            previous = np.zeros(8) if k == 0 else raw["actions"][k-1]
            require(np.array_equal(raw["actions"][k], previous), "Fallback must hold previous applied command")
        terms, weighted = score.get("raw_terms", {}), score.get("weighted_terms", {})
        require(set(terms) == set(weighted) == set(weights)
                and all(type(v) in (int,float) and np.isfinite(v) for v in (*terms.values(), *weighted.values())),
                "Missing original finite 15-term Isaac score")
        require(all(np.isclose(weighted[n], weights[n]*terms[n], atol=1e-12, rtol=1e-12) for n in weights)
                and np.isclose(score.get("total_reward", np.nan), sum(weighted.values()), atol=1e-12, rtol=1e-12)
                and score.get("reward_upper_bound") == upper
                and np.isclose(score.get("stage_cost", np.nan), .02*(upper-score["total_reward"]), atol=1e-12, rtol=1e-12),
                "Original actual Isaac score arithmetic differs")
        previous = np.zeros(8) if k == 0 else raw["actions"][k-1]
        require(np.isclose(terms["action_rate_l2"], np.sum((raw["actions"][k]-previous)**2), atol=1e-12, rtol=1e-12)
                and score.get("base_com_vz") == float(trace["com_velocity"][(k+1)*4, 2]),
                "Actual score action memory or measured COM vertical velocity differs")
        scale = report["objective_contract"]["height_regularization"]["acceleration_scale_m_s2"]
        vz = trace["com_velocity"][:, 2]
        acceleration = 0. if k == 0 else float(-np.expm1(-((vz[(k+1)*4]-vz[k*4])/(.02*scale))**2))
        require(np.isclose(terms["height_vertical_acceleration"], acceleration, atol=1e-12, rtol=1e-12),
                "Height score must smooth measured acceleration, not track absolute height")
        error, conversion = row.get("same_state_dt_reward_abs_error"), row.get("coordinate_conversion_max_velocity_error")
        require(all(type(v) in (int,float) and np.isfinite(v) for v in
                    (error, conversion, row.get("actual_dt_scaled_reward")))
                and error >= 0 and 0 <= conversion <= 1e-8
                and np.isclose(error, abs(row["actual_dt_scaled_reward"]-.02*score["total_reward"]), atol=1e-12, rtol=1e-12),
                "Original reward/conversion reconstruction differs")
        reward_errors.append(error); conversion_errors.append(conversion); costs.append(score["stage_cost"])
    require(totals["optimization_failed"] == report.get("solver_failures")
            and max(reward_errors) == report.get("maximum_same_state_dt_reward_abs_error")
            and max(conversion_errors) == report.get("maximum_coordinate_conversion_error")
            and report.get("real_time_verified") is False, "Isaac diagnostic aggregate differs")
    elapsed = report.get("elapsed_s")
    require(type(elapsed) in (int,float) and np.isfinite(elapsed) and elapsed >= 0, "Invalid original elapsed time")
    return trace, {"decision_samples": FRAMES, "counts": totals, "common_score_arithmetic_verified": True,
        "actual_isaac_stage_cost_sum": float(np.sum(costs)),
        "maximum_dt_reward_reconstruction_abs_error": max(reward_errors),
        "maximum_coordinate_conversion_velocity_abs_error": max(conversion_errors),
        "timing": {"sample_count": FRAMES, "compute_ms_median_p95_max": (1000*np.quantile(compute,[.5,.95,1.])).tolist(),
            "fraction_above_20ms": float(np.mean(compute > .02)), "reported_episode_elapsed_s": elapsed,
            "reported_solver_timing_scopes": sorted(scopes), "deadline_enforced": False,
            "real_time_verified": False, "actuation_delay_injected": False,
            "scope": "Original uncontrolled solve wall times including native IPC; plant waits for each solve, not real-time execution evidence."}}


def audit_isaac_episode(directory, mu, spec, model_sha, mj_record):
    directory = Path(directory); report = read_json(directory/"report.json")
    require(report.get("status") == "completed", "Incomplete runtime is censored, not a physical verdict; complete-only publication")
    header = audit_isaac_header(report, mu, spec, model_sha, mj_record)
    require(header["asset_sha256"] == ISAAC_ASSET_SHA256, "Isaac USD differs from the approved existing asset")
    evidence = episode_evidence(directory, "isaac", spec,
        {"effective_cem_config": copy.deepcopy(PILOT_CEM),
         "prediction_mu_scenarios": [.7]}, mu)
    raw = load_arrays(directory/"rollout.npz"); log = json.loads((directory/"solver_log.json").read_text())
    trace, diagnostic = audit_isaac_trace(report, raw, log)
    return evidence, trace, {**header, "input_sha256": evidence["hashes"],
        "complete_physical_evidence": True, "censored": False, "solver_diagnostics": diagnostic,
        "simulation_budget": {"actual_plant_physics_substeps": SAMPLES-1,
            "requested_prediction_substeps": diagnostic["counts"]["scenario_evaluations"]*30*4,
            "failed_requests_prove_completed_native_steps": False}}


def append_policy_compact(compact, policy_data):
    """Retain cast-proof inputs without replacing the pinned MJ reference."""
    proof = {"reference_robot_time_s", "reference_robot_position_float64_m"}
    for name, value in policy_data.items():
        if not name.startswith("reference_") or name in proof:
            compact[name] = value.copy()


def run(args):
    output = Path(args.output_dir); require(not output.exists(), "Choose a fresh output directory")
    directories = parse_runs(args.run); require(list(directories) == list(GRID), "Complete-only comparison requires all17 Isaac cases")
    spec = verify_reference_pair(args.motion_file, args.reference_file, REFERENCE_ID)
    inputs = {"motion": sha256(args.motion_file), "robot_reference": sha256(args.reference_file), "model": sha256(args.model)}
    reference = reference_trace(load_arrays(args.motion_file))
    mj, mj_hashes, mj_data, ref_geometry, mj_traces, mj_geometry = load_mujoco_bundle(args.mujoco_bundle, spec, inputs, reference)
    traces, audits, first, header = {}, {}, None, None
    for mu, directory in directories.items():
        case = case_name(mu); evidence, trace, public = audit_isaac_episode(directory, mu, spec, inputs["model"], mj["runs"][case])
        identity = {n: public[n] for n in ("controller_contract", "native_identity", "asset_sha256", "fk_source_helpers_sha256")}
        require(header is None or header == identity, "Isaac source/controller/asset identity changed across mu")
        header = identity
        if first is None: first = evidence
        else: validate_case_against_nominal(evidence, first)
        raw = load_arrays(directory/"rollout.npz")
        require(np.array_equal(raw["reference_root_xy_m"], reference["position"][:, :2]), "Recorded reference changed/aligned")
        traces["isaac_"+case], audits[case] = trace, public
    with source_model(args.source_root, args.model) as (model, loader):
        require(loader["source_helpers_sha256"] == header["fk_source_helpers_sha256"], "FK helper bytes differ from actual runner")
        fk = WheelKinematics(model); geometry = {name: fk.reconstruct(trace) for name, trace in traces.items()}
    policy_summary = policy_data = policy_hashes = None
    if args.policy_dynamics is not None:
        from analyze_sa29_policy_friction_dynamics import load_public_bundle
        policy_summary, policy_data, policy_hashes = load_public_bundle(args.policy_dynamics, spec, inputs, reference)
    rows = []
    for mu in GRID:
        case = case_name(mu); ia = "isaac_"+case
        pair_windows = windows({"isaac": traces[ia], "mujoco": mj_traces[case]},
            {"isaac": geometry[ia], "mujoco": mj_geometry[case]}, reference, ref_geometry)
        row = {"mu": mu, "NMPC": pair_windows["core"], "NMPC_windows": pair_windows}
        if policy_summary is not None:
            for algorithm in ("PPO", "SAC"):
                row[algorithm] = policy_summary["runs"][algorithm+"_"+case]["windows"]["core"]
        rows.append(row)
    summary = {"schema_version": 1, "kind": "separate_sa29_two_engine_controller_development",
        "status": "development_analysis_only", "reference_id": REFERENCE_ID, "reference_spec": spec,
        "role": spec["role"], "stratum": spec["stratum"], "evaluation_mu": list(GRID), "rows": rows,
        "complete_nmpc_two_engine_grid": True, "complete_three_controller_two_engine_grid": policy_summary is not None,
        "frames_per_episode": FRAMES, "physical_samples_per_episode": SAMPLES, "input_sha256": inputs,
        "mujoco_bundle_sha256": mj_hashes, "policy_dynamics_bundle_sha256": policy_hashes,
        "policy_training_evidence": None if policy_summary is None else policy_summary["policies"],
        "reference_numeric_representation_audit": None if policy_summary is None else policy_summary["reference_numeric_representation_audit"],
        "isaac_runs": audits, "model_loader_audit": loader,
        "isaac_fk_audit": {n: g["audit"] for n,g in geometry.items()},
        "isaac_posture_diagnostics": {n: posture_diagnostic(t) for n,t in traces.items()},
        "isaac_contact_diagnostics": {n: {"core": contact_diagnostic(t, CORE), "events": [
            {"interval_s": event, "diagnostic": contact_diagnostic(t,event)} for event in EVENTS]} for n,t in traces.items()},
        "effective_cem_config": mj["effective_cem_config"], "effective_cem_config_sha256": mj["effective_cem_config_sha256"],
        "prediction_mu_scenarios": [.7], "events_are_descriptive_not_success_gates": True,
        "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
        **{name: False for name in ("pooled_success_statistics", "primary_task_replicate", "eligible_for_controller_ranking",
            "qualified_freeze_created", "two_engine_nominal_qualified", "robustness_certificate", "real_time_verified",
            "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "learned_inference",
            "controller_retuned", "trajectory_alignment", "actuation_delay_injected")},
        "analysis_source_sha256": {name: sha256(ROOT/"scripts"/name) for name in (
            Path(__file__).name, "analyze_nmpc_friction_pair_development.py", "analyze_nmpc_sa29_friction_development.py",
            "analyze_nmpc_sa29_pilot.py", "analyze_controller_pair.py", "controller_trace.py", "controller_metrics.py",
            "nmpc_validation_contract.py", "analyze_sa29_policy_friction_dynamics.py")},
        "limitations": ["Separate CARLA S-avoidance supplement; never pooled with the primary large-angle turn or described as physical-car capture.",
            "One seed per condition, descriptive development, not controller-class ranking or a robustness/stability certificate.",
            "Same numeric friction does not calibrate equivalent tire/contact physics across engines. Measured contact absence is not proof of flight.",
            "XY screen only; speed, posture, height and heading/slip are continuous descriptive signals, not added gates.",
            "PPO/SAC use learned 51-dimensional observations; NMPC uses privileged measured state and preview with different compute. No real-time actuation delay is modeled."]}
    safe_configuration(summary)
    compact = public_compact(traces, geometry)
    for prefix in ("reference", *[case_name(mu) for mu in GRID]):
        destination = prefix if prefix == "reference" else "mujoco_"+prefix
        for name in (*PUBLIC_TRACE_FIELDS, *PUBLIC_SIGNALS, "wheel_quaternion_wxyz", "wheel_center_velocity_w"):
            key = prefix+"_"+name
            if key in mj_data: compact[destination+"_"+name] = mj_data[key].copy()
    if policy_data is not None:
        append_policy_compact(compact, policy_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name+".partial-", dir=output.parent))
    np.savez_compressed(temporary/"development_trace.npz", **compact)
    render_topdown(compact, summary, temporary/"topdown.png")
    summary["artifacts"] = {role: {"file": name, "sha256": sha256(temporary/name)} for role,name in (
        ("compact_trace", "development_trace.npz"), ("topdown", "topdown.png"))}
    (temporary/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    require(not output.exists(), "Output appeared during analysis; preserving both")
    temporary.rename(output)
    return summary


def render_topdown(data, summary, output):
    """Fixed illustrative mu, same world axes, both engines; contacts not invented."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    algorithms = ("Reference", "PPO", "SAC", "NMPC") if summary["complete_three_controller_two_engine_grid"] else ("Reference", "NMPC")
    fig, axes = plt.subplots(len(SELECTED_MU), len(algorithms), figsize=(4.2*len(algorithms),17.), squeeze=False)
    ref = data["reference_position"][:, :2]
    for r, mu in enumerate(SELECTED_MU):
        bounds = [ref]; case = case_name(mu)
        for axis, algorithm in zip(axes[r], algorithms):
            axis.plot(ref[:,0],ref[:,1],"--",color=".7",lw=.8)
            labels = [("reference", ".3")] if algorithm == "Reference" else [
                ((engine+"_"+case if algorithm == "NMPC" else algorithm+"_"+case+"_"+engine), color)
                for engine,color in (("isaac","#d62728"),("mujoco","#1f77b4"))]
            for prefix,color in labels:
                time,pos,wheels = [data[prefix+"_"+n] for n in ("time_s","position","wheel_pos")]
                core = (time >= CORE[0]-1e-9)&(time <= CORE[1]+1e-9)
                axis.plot(pos[:,0],pos[:,1],color=color,lw=.7,alpha=.3)
                axis.plot(pos[core,0],pos[core,1],color=color,lw=1.8)
                ix = np.arange(0,len(time),max(1,round(.4/float(data[prefix+"_physics_dt_s"]))))
                for side in range(2):
                    for k in ix:
                        if not data[prefix+"_root_course_to_wheel_valid"][k,side]: continue
                        angle = data[prefix+"_wheel_rolling_plane_heading_rad"][k,side]
                        delta = .05*np.array([np.cos(angle),np.sin(angle)])
                        segment = wheels[k,side,:2]+np.array([-delta,delta])
                        contact = prefix == "reference" or (data[prefix+"_contact_count"][k,side] > 0
                            and data[prefix+"_contact_force"][k,side,2] > 5.)
                        axis.plot(segment[:,0],segment[:,1],color=color,lw=.8,alpha=.85 if contact else .12)
                ix = ix[data[prefix+"_root_beta_valid"][ix]]
                velocity = data[prefix+"_com_velocity"][ix,:2]
                axis.quiver(pos[ix,0],pos[ix,1],velocity[:,0],velocity[:,1],color=color,
                    angles="xy",scale_units="xy",scale=7,width=.003)
                bounds.extend((pos[:,:2],wheels[:,:,:2].reshape(-1,2)))
            axis.set(title=f"{algorithm} · μ={mu:g}"+("\nContacts UNMEASURED" if algorithm == "Reference" else ""),xlabel="World X [m]",ylabel="World Y [m]")
            axis.set_aspect("equal",adjustable="box"); axis.grid(alpha=.2)
        xy=np.concatenate(bounds); lo,hi=xy.min(0),xy.max(0); margin=max(.15,.06*float(np.max(hi-lo)))
        for axis in axes[r]: axis.set_xlim(lo[0]-margin,hi[0]+margin); axis.set_ylim(lo[1]-margin,hi[1]+margin)
    fig.suptitle("SA29 S-avoidance · separate DEVELOPMENT · original clocks/world axes · not ranked",fontsize=12)
    fig.legend(handles=[Line2D([],[],color="#d62728",label="Isaac"),Line2D([],[],color="#1f77b4",label="MuJoCo"),
        Line2D([],[],color=".3",label="Thick: core; COM velocity arrows; wheel-plane ticks (pale: no eligible contact)")],loc="lower center",ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.975));fig.savefig(output,dpi=150);plt.close(fig)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--motion", "--motion-file", dest="motion_file", type=Path, required=True)
    for name in ("reference-file","model","source-root","mujoco-bundle","output-dir"):
        result.add_argument("--"+name,type=Path,required=True)
    result.add_argument("--run",action="append",required=True,metavar="MU=ISAAC_DIR")
    result.add_argument("--policy-dynamics",type=Path)
    return result


if __name__ == "__main__":
    result=run(parser().parse_args())
    print(json.dumps({"status":result["status"],"complete_three_controller_two_engine_grid":result["complete_three_controller_two_engine_grid"]}))

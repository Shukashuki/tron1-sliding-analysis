"""Offline MuJoCo fixed-CEM friction DEVELOPMENT, not nominal-qualified.

Validate real plant coefficients and unchanged controller bytes, then compare
the recorded NMPC episodes with existing PPO/SAC MuJoCo XY-only grid evidence.
The qualified freeze/execute workflow remains untouched and is NOT passed by
this analysis. No dynamics, optimization, inference, alignment or time warping.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from analyze_controller_pair import (load_arrays, plant_source_provenance,
    posture_diagnostic, read_json, reference_trace, sha256, source_model)
from analyze_friction_grid import summarize_new_grid
from analyze_nmpc_nominal_development import (audit_trace, configuration_sha256,
    contact_diagnostic, require, safe_configuration, solver_diagnostic, source_digest)
from analyze_tvlqr_pilot import audit_physical_state, digest, public_compact
from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT, verify_reference_pair
from controller_trace import WheelKinematics
from nmpc_validation_contract import (
    _controller_contract, _native_identity, episode_evidence, validate_case_against_nominal,
)


EVALUATION = ROOT/"config/evaluation_friction.json"
PILOT_CEM = {"horizon_steps": 30, "control_blocks": 10, "candidates": 64,
    "iterations": 3, "elite_count": 8, "initial_std": [.04]*6+[.35]*2,
    "min_std": [.005]*6+[.03]*2, "smoothing": 0., "seed": 42,
    "scenario_reduction": "worst", "fallback_action": "previous",
    "reference_convention": "held_pre_action", "parameterization": "warm_start_residual_blocks"}
PILOT_CONFIG_SHA256 = "c30a83dffe39e167748d8305f02d43025dbb83af2d6539bd44b3de7ec4616a74"
PILOT_NATIVE_LIBRARY_SHA256 = "b13e74df3a532496449d2f6b34b6934784a6bfd4ea5f0574c90cf279048821b7"
PILOT_SOURCE_SHA256 = {
    "scripts/run_nmpc_tracking.py": "5a5f714db4d8165f5d39d98372d96f97dd36d8e87f82c53bef740a620c5c1980",
    "scripts/nmpc_shooting.py": "7c667ef0cbd32c58ce53882302e75f5d33487bcfea51c2f6ec3435d454a480ce",
    "scripts/nmpc_tracking_objective.py": "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860",
    "scripts/nmpc_batch_objective.py": "1c9da90ae66778a71ceb2e6bb9d0dcf7f0a2259fafee39b37cbe958b79178fae",
    "scripts/nmpc_native_batch.py": "bb19aeea51cf03eedd87c0fb9e4561c11bc8ab5569cabf31c5465287a815fb76",
    "native/nmpc_batch.cpp": "deb3afe3367e11c28295e00e4a081e84968818070dcfb7044f06466761875ed2",
}
SELECTED_MU = (.1, .3, .7, 1.6, 2.)
FK_HELPERS = ("scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py",
    "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py")


def expected_grid():
    return read_json(EVALUATION)["currently_supported_equal_static_dynamic_grid"]


def case_name(mu):
    return f"mu{mu:.3f}".replace(".", "p")


def parse_runs(values):
    require(len(values) >= 3, "Provide at least three --run MU=DIR episodes; partial coverage remains explicit")
    result = {}
    for value in values:
        number, separator, directory = value.partition("=")
        require(separator and directory, "Use MU=DIR run arguments")
        try:
            mu = float(number)
        except ValueError as error:
            raise ValueError("Run coefficient must be numeric") from error
        require(np.isfinite(mu) and mu in expected_grid() and mu not in result,
                "Duplicate coefficient or coefficient outside predeclared grid")
        path = Path(directory)
        require(path.is_dir(), "Missing original run directory: "+case_name(mu))
        result[mu] = path
    require(len({path.resolve() for path in result.values()}) == len(result), "Run directories must be distinct")
    return dict(sorted(result.items()))


def audit_header(report, mu, spec, model_sha256):
    """Check actual material and fixed identity before interpreting completion."""
    require(configuration_sha256(PILOT_CEM) == PILOT_CONFIG_SHA256, "Internal fixed pilot contract changed")
    require(report.get("kind") == "full_body_height_smooth_cem_nmpc" and report.get("engine") == "MuJoCo"
            and report.get("status") in ("completed", "failed", "running"), "Unsupported CEM episode status/kind")
    require(report.get("reference_spec") == spec and report.get("reference_id") == spec["reference_id"]
            and report.get("motion_sha256") == spec["motion_npz_sha256"]
            and report.get("reference_catalog_sha256") == sha256(DEFAULT_CATALOG), "Actual reference identity differs")
    require(type(report.get("plant_mu")) in (int, float) and report["plant_mu"] == mu
            and report.get("plant_material_readback", {}).get("compiled_sliding_friction_range") == [mu, mu],
            "Actual plant coefficient/material readback differs")
    require(report.get("effective_cem_config") == PILOT_CEM
            and configuration_sha256(report["effective_cem_config"]) == PILOT_CONFIG_SHA256
            and report.get("prediction_mu_scenarios") == [.7],
            "Only validation plant mu may vary; fixed CEM/prediction contract differs")
    require(report.get("model_sha256") == model_sha256, "Actual plant model SHA-256 differs")
    require(report.get("prediction_backend") == "native" and report.get("real_time_verified") is False
            and report.get("robustness_certificate") is False, "Unexpected backend or certificate claim")
    require(report.get("physics_dt_s") == .005 and report.get("control_dt_s") == .02,
            "Actual report clocks differ")
    for name in ("learned_policy_used", "tvlqr_used", "old_qr_used", "root_state_prescribed_after_reset",
                 "external_root_forces_used", "prediction_receives_plant_mu", "pose_height_failure_gate", "truncated_smoke"):
        require(report.get(name) is False, "Unsupported fixed-controller declaration: "+name)
    require(report.get("model_generated_nominal_plan") == {"enabled": False}
            and report.get("adaptation_contract") is None and report.get("prediction_mu_model_bank") is None,
            "Fixed CEM development excludes model plans and online model selection")
    for suffix, expected_sha in PILOT_SOURCE_SHA256.items():
        require(source_digest(report, suffix) == expected_sha, "Changed predeclared runtime source: "+suffix)
    require(source_digest(report, "policy_contract.json")
            == "6e105cbd6efe7d35117a895fa097e8ec64b6ac822a9de07c00acfaef4d09642f"
            and source_digest(report, "config/car_drift_height_smooth_task.json")
            == "31de29a61eec847e0e39783f3a0d9d8f2db551acd679a1b38dfde2b24e8ae7d6",
            "Changed common actuator/task profile input")
    native = report.get("native_prediction_backend", {})
    require(native.get("production_cem_wired") is True and native.get("plant_mu_used_for_prediction") is False
            and native.get("prediction_mu_scenarios") == [.7] and native.get("learned_initialization") is False
            and native.get("terminal_cost") is None and set(native.get("workers", {})) == {"0.7"},
            "Native CEM wiring/model metadata differs")
    require(native["workers"]["0.7"].get("native_library_sha256") == PILOT_NATIVE_LIBRARY_SHA256,
            "Different fixed native predictor binary")
    require(isinstance(report.get("contact_slip_definition"), str)
            and "contact" in report["contact_slip_definition"].lower(), "Missing measured contact-slip definition")
    settings = safe_configuration(report.get("settings"))
    require(settings.get("physics_dt_s") == .005 and settings.get("policy_dt_s") == .02,
            "Reported settings clocks differ")
    require(source_digest(report, "config/nmpc_height_smooth_residual.json")
            == "643262e037d766e924c5ffa165385f2f9a85d5af0257192f222d7106e2685266",
            "Different original settings file (H30/B10 remain explicit CLI overrides)")
    return {"controller_contract": safe_configuration(_controller_contract(report, "mujoco")),
            "native_identity": safe_configuration(_native_identity(report, "mujoco")),
            "reported_settings": settings,
            "fk_source_helpers_sha256": {name: source_digest(report, name) for name in FK_HELPERS}}


def audit_episode(directory, mu, spec, model_sha256):
    """Use the real-mu finite-episode validator, never fake a nominal report.

    episode_evidence is a pure physical-file check, not make_freeze. Large XY
    error and controller fallbacks are valid descriptive outcomes here.
    """
    report = read_json(Path(directory)/"report.json")
    header = audit_header(report, mu, spec, model_sha256)
    if report["status"] != "completed":
        completed = report.get("completed_policy_steps")
        require(completed is None or type(completed) is int and 0 <= completed <= 349,
                "Invalid reported incomplete decision count")
        public = {**header, "actual_plant_mu": float(mu), "prediction_mu_scenarios": [.7],
            "input_sha256": {"report": sha256(Path(directory)/"report.json")},
            "recorded_report_status": report["status"], "complete_physical_evidence": False,
            "censored": True, "physical_xy_failure_claimed": False,
            "reported_completed_policy_steps": report.get("completed_policy_steps"),
            "reported_source_provenance": plant_source_provenance(report),
            "scope": "No complete hash-bound physical episode. May be failed, interrupted or still running; no process-exit inference, XY pass/fail, or partial-trace substitution."}
        return None, None, public
    expected = {"effective_cem_config": copy.deepcopy(PILOT_CEM), "prediction_mu_scenarios": [.7]}
    evidence = episode_evidence(directory, "mujoco", spec, expected, mu)
    raw = load_arrays(Path(directory)/"rollout.npz")
    log = json.loads((Path(directory)/"solver_log.json").read_text())
    require(isinstance(log, list), "Original solver log must be a decision list")
    trace = audit_trace(raw, log, spec); audit_physical_state(raw, trace)
    diagnostic = solver_diagnostic(report, raw, log, {"mode": "warm_start_residual_blocks"})
    stage = np.asarray(raw["stage_cost"], dtype=float)
    require(stage.shape == (349,) and np.isfinite(stage).all()
            and np.array_equal(stage, [row.get("objective", {}).get("stage_cost") for row in log])
            and np.isclose(report.get("total_stage_cost", np.nan), stage.sum(), atol=1e-9, rtol=1e-12),
            "Common evaluation stage-cost trace/log/report mismatch")
    public = {**header, "actual_plant_mu": float(mu), "prediction_mu_scenarios": [.7],
        "input_sha256": evidence["hashes"], "complete_physical_evidence": True, "censored": False,
        "solver_diagnostics": diagnostic,
        "reported_source_provenance": plant_source_provenance(report),
        "simulation_budget": {"requested_prediction_substeps": diagnostic["counts"]["scenario_evaluations"]*30*4,
            "actual_plant_physics_substeps": 1396,
            "failed_prediction_requests_prove_completed_steps": False}}
    return evidence, trace, public


def load_policy_grid(directory, algorithm, spec):
    """Replay authoritative published policy XY evidence, not invented speed.

    Both original engines are validated/replayed to bind the published grid;
    this development comparison uses ONLY its recorded MuJoCo XY component.
    """
    directory = Path(directory)
    names = ("summary.json", "xy_trace.npz", "xy_trace.provenance.json", "training_metadata.json")
    hashes = {name: sha256(directory/name) for name in names}
    provenance, metadata = [read_json(directory/name) for name in names[2:]]
    published = metadata.get("published_artifacts_sha256", {})
    require(all(hashes[name] == digest(published.get(name)) for name in names[:3]),
            "Policy grid bytes differ from training metadata published-artifact bindings")
    trace = load_arrays(directory/"xy_trace.npz")
    require(metadata.get("algorithm") == algorithm and metadata.get("training_status") == "training_completed"
            and metadata.get("smoke_only") is False and metadata.get("training_seed") == 42,
            "Require completed seed-42 "+algorithm+" training evidence")
    require(metadata.get("reference_id") == spec["reference_id"]
            and metadata.get("motion_sha256") == spec["motion_npz_sha256"]
            and metadata.get("robot_reference_sha256") == spec["robot_reference_npz_sha256"]
            and provenance.get("reference_spec") == spec
            and provenance.get("motion_sha256") == spec["motion_npz_sha256"]
            and provenance.get("reference_sha256") == spec["robot_reference_npz_sha256"]
            and provenance.get("reference_catalog_sha256") == metadata.get("reference_catalog_sha256") == sha256(DEFAULT_CATALOG),
            "Policy grid/reference identities differ")
    for field in ("training_manifest_sha256", "checkpoint_sha256"):
        require(digest(provenance.get(field)) == digest(metadata.get(field)), "Policy grid/training binding differs: "+field)
    require(metadata.get("task_profile_sha256") == "31de29a61eec847e0e39783f3a0d9d8f2db551acd679a1b38dfde2b24e8ae7d6"
            and metadata.get("height_adapter_sha256")
                == "e133aca58f671fc22ae1dfc97a03461e7724e4072bf8b74f58a2c14281d198b8",
            "Different common height-smooth policy task")
    declared = provenance.get("cases", {})
    require(isinstance(declared, dict) and len(declared) == 17, "Require complete published 17-case policy grid")
    cases = {name: (item.get("mu"), "") for name, item in declared.items()}
    require(sorted(mu for mu, _ in cases.values()) == expected_grid()
            and all(name == case_name(mu) for name, (mu, _) in cases.items()), "Policy coefficient names/grid differ")
    replay = summarize_new_grid(trace, provenance, cases)
    require(replay == read_json(directory/"summary.json"), "Published policy XY metrics fail exact replay")
    rows = {float(row["evaluation_mu_static_dynamic_equal"]): row for row in replay["rows"]}
    for mu, row in rows.items():
        name = case_name(mu)
        for engine in ("isaac", "mujoco"):
            time, xy = trace[f"{name}_{engine}_t"], trace[f"{name}_{engine}_xy"]
            require(time.shape == (1397,) and xy.shape == (1397, 2)
                    and np.allclose(time, np.arange(1397)*.005, atol=1e-9, rtol=0)
                    and row[engine].get("core_complete") is True,
                    "Require full original 1397-sample policy clocks")
    public = {"algorithm": algorithm, "input_sha256": hashes,
        "training_manifest_sha256": provenance["training_manifest_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "training_budget": safe_configuration(metadata.get("training_budget")),
        "action_contract": safe_configuration(metadata.get("action_contract")),
        "evidence_scope": "Exactly replayed existing paired XY grid; only MuJoCo XY used here. No new policy evaluation or inferred contact/speed.",
        "measured_com_speed_available": False, "wheel_heading_available": False, "measured_contact_slip_available": False}
    return public, rows, trace


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    selected = SELECTED_MU
    fig, axes = plt.subplots(len(selected), 4, figsize=(16., 3.5*len(selected)), squeeze=False)
    start, end = summary["reference_spec"]["source_demo_interval_s"]
    low_core, high_core = summary["reference_spec"]["evaluation_core_interval_s"]
    rt, rp = compact["reference_time_s"], compact["reference_position"][:, :2]
    reference = rp[(rt >= start-1e-9) & (rt <= end+1e-9)]
    for row, mu in enumerate(selected):
        case = case_name(mu); bounds = [reference]
        for column, label in enumerate(("reference", "PPO", "SAC", "NMPC")):
            axis = axes[row, column]
            if label == "reference": time, pos = rt, rp
            elif label == "NMPC" and (case not in summary["runs"] or summary["runs"][case]["censored"]):
                axis.text(.5, .5, "Censored: no complete\nphysical episode", ha="center", va="center", transform=axis.transAxes)
                axis.plot(reference[:, 0], reference[:, 1], "--", color=".75", lw=.9)
                axis.set(title=f"NMPC · μ={mu:g} (unavailable)", xlabel="World X [m]", ylabel="World Y [m]")
                axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
                continue
            elif label == "NMPC": time, pos = compact[f"{case}_time_s"], compact[f"{case}_position"][:, :2]
            else: time, pos = compact[f"{label.lower()}_{case}_time_s"], compact[f"{label.lower()}_{case}_xy"]
            full = (time >= start-1e-9) & (time <= end+1e-9)
            core = full & (time >= low_core-1e-9) & (time <= high_core+1e-9)
            color = (".35", "#1f77b4", "#ff7f0e", "#2ca02c")[column]
            axis.plot(reference[:, 0], reference[:, 1], "--", color=".75", lw=.9)
            axis.plot(pos[full, 0], pos[full, 1], color=color, lw=.9, alpha=.5)
            axis.plot(pos[core, 0], pos[core, 1], color=color, lw=1.9)
            bounds.append(pos[full])
            if label == "NMPC":
                indices = np.flatnonzero(full)[::80]
                wheels = compact[f"{case}_wheel_pos"]
                for side in range(2):
                    for index in indices:
                        if not compact[f"{case}_root_course_to_wheel_valid"][index, side]: continue
                        heading = compact[f"{case}_wheel_rolling_plane_heading_rad"][index, side]
                        delta = .05*np.array([np.cos(heading), np.sin(heading)])
                        segment = wheels[index, side, :2]+np.array([-delta, delta])
                        contact = compact[f"{case}_contact_count"][index, side] > 0 and compact[f"{case}_contact_force"][index, side, 2] > 5.
                        axis.plot(segment[:, 0], segment[:, 1], color=color, lw=.9, alpha=.9 if contact else .15)
                arrows = indices[compact[f"{case}_root_beta_valid"][indices]]
                course = compact[f"{case}_course_rad"][arrows]
                axis.quiver(pos[arrows, 0], pos[arrows, 1], np.cos(course), np.sin(course), color=color,
                    angles="xy", scale_units="xy", scale=8, width=.004)
            title = f"{label} · μ={mu:g}"
            if label in ("PPO", "SAC"): title += " (XY only)"
            if label == "reference": title = f"Reference · μ={mu:g} row (contacts unmeasured)"
            axis.set(title=title, xlabel="World X [m]", ylabel="World Y [m]")
            axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
        xy = np.concatenate(bounds); low, high = xy.min(axis=0), xy.max(axis=0)
        margin = max(.15, .06*float(np.max(high-low)))
        for axis in axes[row]:
            axis.set_xlim(low[0]-margin, high[0]+margin); axis.set_ylim(low[1]-margin, high[1]+margin)
    fig.suptitle("MuJoCo friction DEVELOPMENT · CEM nominal NOT qualified · no controller ranking", fontsize=12)
    fig.legend(handles=[Line2D([], [], color=".3", lw=1.9, label="Thick: unchanged evaluation core"),
        Line2D([], [], color="#2ca02c", lw=.9, label="NMPC ticks: wheel planes (pale: no contact); arrows: COM XY velocity")],
        loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .975)); fig.savefig(output, dpi=150); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing output; choose a NEW output directory")
    directories = parse_runs(args.run)
    spec = verify_reference_pair(args.motion_file, args.reference_file, PRIMARY_REFERENCE_ID)
    require(spec["frames"] == 349 and spec["fps"] == 50., "Require locked primary reference")
    policy = {algorithm: load_policy_grid(getattr(args, algorithm.lower()+"_grid"), algorithm, spec)
              for algorithm in ("PPO", "SAC")}
    traces = {"reference": reference_trace(load_arrays(args.motion_file))}; audits = {}; first = None; first_header = None
    for mu, directory in directories.items():
        evidence, trace, public = audit_episode(directory, mu, spec, sha256(args.model))
        header = {key: public[key] for key in ("controller_contract", "native_identity", "reported_settings", "fk_source_helpers_sha256")}
        require(first_header is None or header == first_header, "Controller/source/actuator/settings changed across coefficients")
        first_header = header
        if evidence is not None:
            if first is None: first = evidence
            else: validate_case_against_nominal(evidence, first)
            traces[case_name(mu)] = trace
        audits[case_name(mu)] = public
    with source_model(args.source_root, args.model) as (model, loader):
        require(loader.get("source_helpers_sha256") == first_header["fk_source_helpers_sha256"],
                "FK loader helpers differ from recorded plant implementation")
        fk = WheelKinematics(model); recon = {label: fk.reconstruct(trace) for label, trace in traces.items()}
    interval = spec["evaluation_core_interval_s"]; reference = traces["reference"]
    metrics = {}
    for label, trace in traces.items():
        contacts = {} if label == "reference" else dict(wheel_ground_contact_count=trace["contact_count"],
            wheel_ground_force_w=trace["contact_force"], wheel_contact_lateral_slip_m_s=trace["contact_slip"])
        metrics[label] = summarize_turn(trace["time_s"], trace["position"], recon[label]["signals"],
            reference["time_s"], reference["position"][:, :2], recon["reference"]["signals"]["speed_m_s"], interval, **contacts)
    rows = []
    for mu in expected_grid():
        row = {"mu": mu, "NMPC": metrics.get(case_name(mu), {
            "coverage_complete": False, "censored": True, "reference_xy_rmse_m": None,
            "reference_xy_peak_m": None, "large_xy_exploratory_screen": None,
            "data_availability": "No complete hash-bound physical episode; not a physical XY verdict"})}
        for algorithm, (_, saved, _) in policy.items():
            source = saved[mu]["mujoco"]
            row[algorithm] = {"reference_xy_rmse_m": source["full_core_reference_xy_rmse_m"],
                "reference_xy_peak_m": source["full_core_reference_xy_peak_m"],
                "coverage_complete": source["core_complete"], "large_xy_exploratory_screen": source["large_reference_xy_screen"],
                "speed_retention": None, "wheel_heading": None, "contact_slip": None,
                "data_availability": "Published native-clock root XY only; unavailable fields are not zero"}
        rows.append(row)
    evaluated = [mu for mu in directories if not audits[case_name(mu)]["censored"]]
    missing = [mu for mu in expected_grid() if mu not in evaluated]
    censored = [mu for mu in directories if audits[case_name(mu)]["censored"]]
    summary = {"schema_version": 1, "kind": "unqualified_mujoco_cem_fixed_friction_development",
        "status": "development_analysis_only", "engine": "MuJoCo", "reference_id": PRIMARY_REFERENCE_ID,
        "reference_spec": spec, "provided_mu": list(directories), "evaluated_mu": evaluated,
        "censored_mu": censored, "expected_mu": expected_grid(), "missing_or_censored_mu": missing,
        "complete_predeclared_grid": not missing, "runs": audits, "rows": rows, "windows": {"core": metrics},
        "effective_cem_config": copy.deepcopy(PILOT_CEM), "effective_cem_config_sha256": PILOT_CONFIG_SHA256,
        "prediction_mu_scenarios": [.7], "policy_grids": {name: value[0] for name, value in policy.items()},
        **{name: False for name in ("two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
            "sim2sim_comparison", "eligible_for_controller_ranking", "pooled_success_statistics", "robustness_certificate",
            "controlled_execution_success_claimed", "height_or_flight_failure_gate", "heading_offset_is_mismatch",
            "dynamics_stepping", "learned_inference", "new_policy_training_performed", "actuation_delay_injected")},
        "settings_changed_across_plant_mu": False,
        "model_loader_audit": loader, "fk_audit": {name: value["audit"] for name, value in recon.items()},
        "posture_diagnostics": {name: posture_diagnostic(traces[name]) for name in audits if name in traces},
        "contact_diagnostics": {name: contact_diagnostic(traces[name], interval) for name in audits if name in traces},
        "input_sha256": {name: sha256(getattr(args, name)) for name in ("motion_file", "reference_file", "model")},
        "reference_catalog_sha256": sha256(DEFAULT_CATALOG), "evaluation_protocol_sha256": sha256(EVALUATION),
        "analysis_source_sha256": {name: sha256(ROOT/"scripts"/name) for name in (
            "analyze_nmpc_friction_development.py", "nmpc_validation_contract.py", "analyze_nmpc_nominal_development.py",
            "analyze_friction_grid.py", "analyze_saved_xy.py", "analyze_tvlqr_pilot.py", "analyze_controller_pair.py",
            "controller_metrics.py", "controller_trace.py", "controller_references.py")},
        "limitations": ["Descriptive MuJoCo-only fixed CEM development: nominal XY already fails, and no qualified freeze/driver or Isaac friction grid is claimed.",
            "Only validation plant friction changes. Fixed prediction mu=.7, H30/B10/C64/I3/E8, seed42, task/actuators/initialization remain unchanged; no post-grid retuning.",
            "The settings file still declares default H15/B5; explicit recorded H30/B10 CLI overrides are bound by the effective configuration hash, not silently treated as file defaults.",
            "PPO/SAC results are existing independently replayed MuJoCo XY evidence, not new evaluations. Their COM speed, heading and measured slip are unavailable in these compact archives.",
            "One episode per coefficient and seed is not a statistical success rate, universal friction threshold, algorithm ranking or calibrated contact law.",
            "Full simulator state/reference preview and uncontrolled optimizer wall time differ from learned actor interfaces. Simulator waits for solves; no real-time or injected delay claim.",
            "Height, flight, heading offsets and contact loss are descriptive, not extra failure gates; small XY error alone does not certify upright execution.",
            "Native clocks/world axes are unchanged; no alignment or time warp. Reference contacts are unmeasured; S-avoidance supplement is not pooled."]}
    compact = public_compact(traces, recon)
    for algorithm, (_, _, source) in policy.items():
        for mu in expected_grid():
            name = case_name(mu); prefix = algorithm.lower()+"_"+name
            compact[prefix+"_time_s"] = source[name+"_mujoco_t"].copy()
            compact[prefix+"_xy"] = source[name+"_mujoco_xy"].copy()
    output.mkdir(parents=True); np.savez_compressed(output/"development_trace.npz", **compact)
    render_topdown(compact, summary, output/"topdown.png")
    summary["artifacts"] = {name: {"file": filename, "sha256": sha256(output/filename)}
        for name, filename in (("compact_trace", "development_trace.npz"), ("topdown", "topdown.png"))}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--motion", "--motion-file", dest="motion_file", type=Path, required=True)
    for name in ("reference-file", "model", "source-root", "ppo-grid", "sac-grid", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--run", action="append", required=True, metavar="MU=DIR")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "cases": len(result["runs"]),
        "complete_predeclared_grid": result["complete_predeclared_grid"], "nominal_qualified": False}))

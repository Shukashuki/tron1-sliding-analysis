"""Offline SA29 MuJoCo-only fixed-NMPC friction DEVELOPMENT.

Consume original recordings at their actual plant coefficients. The 17-case
declaration is separate from the earlier one-episode pilot. No dynamics, IK,
policy inference, retargeting, alignment, tuning or qualified-driver execution.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from analyze_controller_pair import (load_arrays, plant_source_provenance,
    posture_diagnostic, read_json, reference_trace, sha256, source_model)
from analyze_nmpc_friction_development import (PILOT_CEM, audit_header as audit_fixed_header,
    case_name)
from analyze_nmpc_nominal_development import (configuration_sha256, contact_diagnostic,
    require, safe_configuration, source_digest)
from analyze_nmpc_sa29_pilot import (CORE, EVENTS, FRAMES, SAMPLES, REFERENCE_ID,
    audit_trace, solver_diagnostic, summarize_windows)
from analyze_tvlqr_pilot import public_compact
from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, ROOT, get_reference, verify_reference_pair
from controller_trace import WheelKinematics
from nmpc_tracking_objective import HeightSmoothTrackingObjective
from nmpc_validation_contract import episode_evidence, validate_case_against_nominal


PREDECLARATION = ROOT/"config/nmpc_sa29_friction_development.json"
PREDECLARATION_SHA256 = "c6b3397431ebcb22116c6dde60e78f0d15bd0a5750511b7dcd438a49fa59a62c"
GRID = (.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.)
SELECTED_MU = (.01, .1, .7, 1.6, 2.)
REUSED_NOMINAL_INPUT_SHA256 = {
    "report": "fe703e078fce33c79670d0bae14a3966b21c03b7a68d5ae90871f55374ec9356",
    "trace": "aecb80f386f43c831664e8e3f41b678c15551922f9c4147d92475573e827a388",
    "solver": "0b101dd0dac648566068575836e9cbea6333fdf6e942c6764dd94b9b4a37c725",
}
FROZEN_HELPER_SHA256 = {
    "analyze_nmpc_sa29_pilot.py": "caf088c29b4ad826d041ed0f400c3720e215835b82824314a0b7e3574f6617e8",
    "analyze_nmpc_friction_development.py": "112759eb9266f70618c02e436e6a236bfcb8e25786ae7bc53ca9eb456aa1f691",
    "analyze_nmpc_nominal_development.py": "7b7ba81cbdf2a86c49f88f9abcf23b7486e12127e9106ed7bfe15bf24bb228b9",
    "analyze_tvlqr_pilot.py": "4db3b27de2c6f18b50bc06bfff8d822cf1ba77fff5cdb6597110ba955b1c1703",
    "analyze_controller_pair.py": "bac9dc6c81b02e46143322462ca6e53de1a1057da72e19ccaeb8632725d6bd2d",
    "controller_metrics.py": "40921dad0c268d4bd871e0910086957555116fdd07188ba7cb4f8c76ca4343de",
    "controller_trace.py": "673d17bda4d9f6c93b76b0b963b199aef3e3abe8982cfc6cc6580184e5b0a3e6",
    "controller_references.py": "2f2746946c37a527298ef83d6a4f07063a9d95250c49815f4f1478d5141b4bf6",
    "nmpc_validation_contract.py": "f0018aeb1ca007650835ca344ae220a6cba289af0637bd2f137659d58ca641ff",
    "nmpc_tracking_objective.py": "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860",
}


def expected_grid():
    return list(GRID)


def audit_predeclaration(manifest=None):
    require(sha256(PREDECLARATION) == PREDECLARATION_SHA256,
            "SA29 17-case predeclaration byte identity changed")
    locked = read_json(PREDECLARATION); manifest = locked if manifest is None else manifest
    require(manifest == locked, "Only the locked SA29 17-case declaration may authorize this analysis scope")
    require(manifest["reference_id"] == REFERENCE_ID and manifest["role"] == "separate_supplement"
            and manifest["stratum"] == "high_sideslip_s_avoidance" and manifest["plant_mu"] == expected_grid()
            and manifest["intended_local_episodes"] == 17 and manifest["new_local_episodes"] == 16
            and manifest["reused_existing_plant_mu"] == [.7] and len(manifest["source_sha256"]) == 21,
            "Require the separately declared SA29 17-case/reused-nominal scope")
    evaluation = manifest["evaluation"]
    require(evaluation["frames"] == FRAMES and evaluation["physical_samples"] == SAMPLES
            and evaluation["duration_s"] == 5.16 and evaluation["physics_dt_s"] == .005
            and evaluation["control_dt_s"] == .02 and evaluation["core_window_s"] == CORE
            and evaluation["supplementary_event_windows_s"] == EVENTS
            and evaluation["events_are_descriptive_not_gates"] is True,
            "SA29 predeclared physical clocks/windows differ")
    for name, expected in FROZEN_HELPER_SHA256.items():
        require(sha256(ROOT/"scripts"/name) == expected, "Changed frozen offline analysis helper: "+name)
    return manifest


def parse_runs(values):
    require(len(values) >= 1, "Provide at least one --run MU=DIR; missing cases remain censored")
    result = {}
    for value in values:
        number, separator, directory = value.partition("=")
        require(separator and directory, "Use actual MU=DIR run arguments")
        try:
            mu = float(number)
        except ValueError as error:
            raise ValueError("Run coefficient must be numeric") from error
        require(np.isfinite(mu) and mu in GRID and mu not in result,
                "Duplicate or undeclared actual plant coefficient")
        path = Path(directory)
        require(path.is_dir(), "Missing original case directory: "+case_name(mu))
        result[mu] = path
    require(len({path.resolve() for path in result.values()}) == len(result), "Run directories must be distinct")
    return dict(sorted(result.items()))


def audit_header(report, mu, spec, *, model_sha256, declaration=None):
    declaration = audit_predeclaration(declaration)
    require(mu in GRID and spec == get_reference(REFERENCE_ID),
            "Require an actual declared coefficient and locked SA29 supplement; no primary relabeling")
    header = audit_fixed_header(report, mu, spec, model_sha256)
    require(report.get("role") == spec["role"] and report.get("stratum") == spec["stratum"]
            and report.get("pooled_success_statistics") is False and report.get("isaac_result") is False
            and report.get("action_replay", False) is False,
            "Unsupported supplementary, policy/replay or Isaac claim")
    require(model_sha256 == declaration["input_sha256"]["model"]
            and report["motion_sha256"] == declaration["input_sha256"]["motion"]
            and spec["robot_reference_npz_sha256"] == declaration["input_sha256"]["robot_reference"]
            and configuration_sha256(report["effective_cem_config"])
                == declaration["effective_cem_configuration_sha256"], "Changed SA29 input/controller identity")
    require(report.get("settings") == read_json(ROOT/declaration["settings"]),
            "Reported original settings changed; effective H30/B10 remain explicit fixed overrides")
    parameterization = report.get("action_parameterization", {})
    require(parameterization.get("mode") == "warm_start_residual_blocks"
            and parameterization.get("baseline_copied_and_fixed_per_solve") is True
            and parameterization.get("learned_baseline") is False
            and report.get("contact_threshold_n") == 5.
            and report.get("wheel_lateral_contact_slip_evidence_kind") == "actual_ground_contact_points",
            "Residual decoder or measured ground-contact evidence contract differs")
    sources = declaration["source_sha256"]
    for relative, expected in sources.items():
        require(sha256(ROOT/relative) == expected and source_digest(report, relative) == expected,
                "Changed predeclared original/on-disk runtime source: "+relative)
    actuator = declaration["input_sha256"]["actuator_contract"]
    require(source_digest(report, "policy_contract.json") == actuator
            and sorted(report["source_sha256"].values()) == sorted([*sources.values(), actuator]),
            "Original 22-source/actuator bindings differ")
    require(report.get("objective_contract")
            == HeightSmoothTrackingObjective(read_json(ROOT/declaration["task_profile"])).contract(),
            "Exact common height-free 15-term objective differs")
    packed = report.get("packed_state_contract", {})
    require(packed.get("dimension") == 111 and packed.get("qpos_dimension") == 15
            and packed.get("qvel_dimension") == 14 and packed.get("contact_force_history_shape") == [3, 9, 3],
            "Physical/contact-memory state contract differs")
    require(header["native_identity"]["native_library_sha256"]
            == declaration["input_sha256"]["native_library"], "Fixed native predictor binary differs")
    return {**header, "actual_plant_mu": float(mu), "prediction_mu_scenarios": [.7],
        "plant_material_readback": safe_configuration(report["plant_material_readback"]),
        "runtime_source_sha256": dict(sources), "actuator_contract_sha256": actuator,
        "predeclaration_sha256": PREDECLARATION_SHA256, "no_retuning": True}


def audit_episode(directory, mu, spec, model_sha256, declaration=None):
    directory = Path(directory); report = read_json(directory/"report.json")
    public = audit_header(report, mu, spec, model_sha256=model_sha256, declaration=declaration)
    public.update(recorded_report_status=report["status"], reported_source_provenance=plant_source_provenance(report))
    if report["status"] != "completed":
        completed = report.get("completed_policy_steps")
        require(completed is None or type(completed) is int and 0 <= completed <= FRAMES,
                "Invalid incomplete SA29 decision count")
        public.update(input_sha256={"report": sha256(directory/"report.json")},
            complete_physical_evidence=False, censored=True, physical_xy_failure_claimed=False,
            reported_completed_policy_steps=completed,
            scope="No complete hash-bound physical episode; failed/interrupted/running is censored, never a partial XY verdict or process-exit inference.")
        return None, None, public
    evidence = episode_evidence(directory, "mujoco", spec,
        {"effective_cem_config": copy.deepcopy(PILOT_CEM), "prediction_mu_scenarios": [.7]}, mu)
    if mu == .7:
        require(evidence["hashes"] == REUSED_NOMINAL_INPUT_SHA256,
                "The nominal case must reuse the original SA29 pilot, never a substituted rerun")
    raw = load_arrays(directory/"rollout.npz")
    solver = json.loads((directory/"solver_log.json").read_text())
    trace = audit_trace(raw, solver, spec)
    public.update(input_sha256=evidence["hashes"], complete_physical_evidence=True, censored=False,
        solver_diagnostics=solver_diagnostic(report, raw, solver), nominal_pilot_reused=mu == .7,
        simulation_budget={"actual_plant_physics_substeps": SAMPLES-1,
            "requested_prediction_substeps": sum(row["solver"]["scenario_evaluations"] for row in solver)*30*4,
            "failed_prediction_requests_prove_completed_steps": False})
    return evidence, trace, public


def summarize_cases(traces, recon, spec):
    require(spec == get_reference(REFERENCE_ID) and set(traces) == set(recon)
            and "reference" in traces and set(traces) <= {"reference", *[case_name(mu) for mu in GRID]},
            "SA29 metrics require only the reference and declared physical NMPC cases")
    reference = traces["reference"]
    def reference_metric(interval):
        return summarize_turn(reference["time_s"], reference["position"], recon["reference"]["signals"],
            reference["time_s"], reference["position"][:, :2], recon["reference"]["signals"]["speed_m_s"], interval)
    result = {"core": {"reference": reference_metric(CORE)}, "supplementary_events": [
        {"interval_s": interval, "descriptive_not_gate": True, "metrics": {"reference": reference_metric(interval)}} for interval in EVENTS]}
    for label, trace in traces.items():
        if label == "reference": continue
        windows = summarize_windows({"reference": reference, "NMPC": trace},
            {"reference": recon["reference"], "NMPC": recon[label]}, spec)
        result["core"][label] = windows["core"]["NMPC"]
        for destination, source in zip(result["supplementary_events"], windows["supplementary_events"]):
            destination["metrics"][label] = source["metrics"]["NMPC"]
    return result


def _censored_metric(interval):
    return {"window_s": list(interval), "coverage_complete": False, "censored": True,
        "reference_xy_rmse_m": None, "reference_xy_peak_m": None, "large_xy_exploratory_screen": None,
        "mean_speed_m_s": None, "mean_speed_retention_ratio": None, "heading_geometry": None,
        "ground_contact_heading_geometry": None, "ground_contact_lateral_slip_m_s": None,
        "data_availability": "No complete hash-bound physical case; unavailable is not zero, flight or a physical failure verdict"}


def make_rows(metrics, audits):
    rows = []
    for mu in GRID:
        label = case_name(mu); available = label in metrics["core"]
        require(not available or label in audits and audits[label]["censored"] is False,
                "Physical metrics cannot substitute a missing/censored case")
        rows.append({"mu": mu, "case": label, "censored": not available,
            "NMPC": metrics["core"].get(label, _censored_metric(CORE)),
            "supplementary_events": [{"interval_s": event["interval_s"], "descriptive_not_gate": True,
                "NMPC": event["metrics"].get(label, _censored_metric(event["interval_s"]))}
                for event in metrics["supplementary_events"]]})
    return rows


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(len(SELECTED_MU), 2, figsize=(11., 3.6*len(SELECTED_MU)), squeeze=False)
    reference = compact["reference_position"][:, :2]
    for row, mu in enumerate(SELECTED_MU):
        label = case_name(mu); bounds = [reference]
        for axis, prefix, color in zip(axes[row], ("reference", label), (".35", "#2ca02c")):
            if prefix != "reference" and prefix+"_time_s" not in compact:
                axis.plot(reference[:, 0], reference[:, 1], "--", color=".75", lw=1.)
                axis.text(.5, .5, "Censored: no complete\nphysical episode", ha="center", va="center", transform=axis.transAxes)
                axis.set_title(f"NMPC μ={mu:g} · unavailable")
            else:
                time, pos, wheels = [compact[prefix+"_"+name] for name in ("time_s", "position", "wheel_pos")]
                core = (time >= CORE[0]-1e-9) & (time <= CORE[1]+1e-9)
                axis.plot(reference[:, 0], reference[:, 1], "--", color=".75", lw=.9)
                axis.plot(pos[:, 0], pos[:, 1], color=color, lw=.8, alpha=.35)
                axis.plot(pos[core, 0], pos[core, 1], color=color, lw=2.)
                indices = np.arange(0, len(time), max(1, round(.4/float(compact[prefix+"_physics_dt_s"]))))
                for side in range(2):
                    axis.plot(wheels[:, side, 0], wheels[:, side, 1], color=color, lw=.5, alpha=.2)
                    for index in indices:
                        if not compact[prefix+"_root_course_to_wheel_valid"][index, side]: continue
                        heading = compact[prefix+"_wheel_rolling_plane_heading_rad"][index, side]
                        delta = .055*np.array([np.cos(heading), np.sin(heading)])
                        segment = wheels[index, side, :2]+np.array([-delta, delta])
                        eligible = prefix == "reference" or (compact[prefix+"_contact_count"][index, side] > 0
                            and compact[prefix+"_contact_force"][index, side, 2] > 5.)
                        axis.plot(segment[:, 0], segment[:, 1], color=color, lw=1.1, alpha=.9 if eligible else .15)
                arrows = indices[compact[prefix+"_root_beta_valid"][indices]]
                velocity = compact[prefix+"_com_velocity"][arrows, :2]
                axis.quiver(pos[arrows, 0], pos[arrows, 1], velocity[:, 0], velocity[:, 1], color=color,
                    angles="xy", scale_units="xy", scale=7, width=.003)
                bounds.extend((pos[:, :2], wheels[:, :, :2].reshape(-1, 2)))
                axis.set_title("Kinematic reference · contacts UNMEASURED" if prefix == "reference"
                    else f"NMPC μ={mu:g} · MuJoCo measured contacts")
            axis.set(xlabel="World X [m]", ylabel="World Y [m]")
            axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
        xy = np.concatenate(bounds); low, high = xy.min(axis=0), xy.max(axis=0)
        margin = max(.15, .06*float(np.max(high-low)))
        for axis in axes[row]:
            axis.set_xlim(low[0]-margin, high[0]+margin); axis.set_ylim(low[1]-margin, high[1]+margin)
    fig.suptitle("SA29 fixed-NMPC friction DEVELOPMENT · separate S-avoidance / MuJoCo only / not ranked", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=".3", lw=2., label="Thick: core; thin: full playback; COM velocity arrows"),
        Line2D([], [], color=".3", lw=1., label="Wheel-plane ticks; pale: no eligible contact (not proof of flight)")],
        loc="lower center", ncol=1, fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .975)); fig.savefig(output, dpi=150); plt.close(fig)


def run(args):
    output = Path(args.output_dir); require(not output.exists(), "Preserve existing output; choose a NEW directory")
    declaration = audit_predeclaration(); directories = parse_runs(args.run)
    spec = verify_reference_pair(args.motion_file, args.reference_file, REFERENCE_ID)
    inputs = {"motion": sha256(args.motion_file), "robot_reference": sha256(args.reference_file), "model": sha256(args.model)}
    require(inputs == {name: declaration["input_sha256"][name] for name in inputs}, "Locked SA29 input bytes differ")
    reference = reference_trace(load_arrays(args.motion_file))
    traces = {"reference": reference}; audits = {}; first_header = None; first_evidence = None
    for mu, directory in directories.items():
        evidence, trace, public = audit_episode(directory, mu, spec, inputs["model"], declaration)
        header = {key: public[key] for key in ("controller_contract", "native_identity", "reported_settings", "fk_source_helpers_sha256")}
        require(first_header is None or header == first_header, "Runtime/native/actuator/settings changed across actual plant coefficients")
        first_header = header
        if evidence is not None:
            if first_evidence is None: first_evidence = evidence
            else: validate_case_against_nominal(evidence, first_evidence)
            raw = load_arrays(directory/"rollout.npz")
            require(np.array_equal(raw["reference_root_xy_m"], reference["position"][:, :2]),
                    "Recorded per-case reference XY differs from locked SA29")
            traces[case_name(mu)] = trace
        audits[case_name(mu)] = public
    with source_model(args.source_root, args.model) as (model, loader):
        require(loader["source_helpers_sha256"] == first_header["fk_source_helpers_sha256"],
                "FK loader bytes differ from original runtime helpers")
        fk = WheelKinematics(model); recon = {label: fk.reconstruct(trace) for label, trace in traces.items()}
    metrics = summarize_cases(traces, recon, spec); rows = make_rows(metrics, audits)
    evaluated = [row["mu"] for row in rows if not row["censored"]]
    missing = [mu for mu in GRID if mu not in evaluated]
    summary = {"schema_version": 1, "kind": "separate_sa29_mujoco_fixed_nmpc_friction_development",
        "status": "development_analysis_only", "engine": "MuJoCo", "reference_id": REFERENCE_ID,
        "reference_spec": spec, "role": spec["role"], "stratum": spec["stratum"],
        "predeclaration_sha256": PREDECLARATION_SHA256, "predeclaration_scope": "Separate 17-case SA29 local MuJoCo development; prior one-case pilot declaration authorizes only the reused nominal recording",
        "intended_local_episodes": 17, "predeclared_new_local_episodes": 16,
        "provided_mu": list(directories), "evaluated_mu": evaluated, "expected_mu": expected_grid(),
        "censored_mu": [mu for mu in directories if audits[case_name(mu)]["censored"]],
        "missing_or_censored_mu": missing, "complete_predeclared_grid": not missing,
        "reused_nominal_pilot_verified": .7 in evaluated, "reused_nominal_input_sha256": REUSED_NOMINAL_INPUT_SHA256,
        "frames_per_complete_episode": FRAMES, "physical_samples_per_complete_episode": SAMPLES,
        "duration_s_per_complete_episode": 5.16, "physics_dt_s": .005, "control_dt_s": .02,
        "prediction_mu_scenarios": [.7], "effective_cem_config": copy.deepcopy(PILOT_CEM),
        "effective_cem_config_sha256": declaration["effective_cem_configuration_sha256"],
        "runs": audits, "rows": rows, "windows": metrics,
        "events_are_descriptive_not_success_gates": True,
        "posture_diagnostics": {label: posture_diagnostic(trace) for label, trace in traces.items() if label != "reference"},
        "posture_scope": "Full 5.16 s playback, not a core-only fall-time assertion",
        "contact_diagnostics": {label: {"core": contact_diagnostic(trace, CORE), "supplementary_events": [
            {"interval_s": interval, "descriptive_not_gate": True, "diagnostic": contact_diagnostic(trace, interval)} for interval in EVENTS]}
            for label, trace in traces.items() if label != "reference"},
        "objective_contract": HeightSmoothTrackingObjective(read_json(ROOT/declaration["task_profile"])).contract(),
        "model_loader_audit": loader, "fk_audit": {label: value["audit"] for label, value in recon.items()},
        "input_sha256": inputs, "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
        "settings_changed_across_plant_mu": False,
        **{name: False for name in ("pooled_success_statistics", "primary_task_replicate", "sim2sim_comparison",
            "isaac_evidence_available", "supplementary_policy_evidence_available", "two_engine_nominal_qualified",
            "qualified_freeze_created", "qualified_driver_executed", "frozen_benchmark", "eligible_for_controller_ranking",
            "robustness_certificate", "real_time_verified", "controlled_execution_success_claimed", "controller_retuned",
            "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "inverse_kinematics",
            "retargeting", "learned_inference", "new_policy_training_performed", "actuation_delay_injected")},
        "analysis_source_sha256": {**FROZEN_HELPER_SHA256,
            "analyze_nmpc_sa29_friction_development.py": sha256(Path(__file__))},
        "limitations": ["One fixed seed/episode per declared coefficient is a descriptive local grid, not a universal friction threshold, statistical success rate or controller ranking.",
            "SA29 is CARLA S-avoidance, not physical-car capture or a second primary large-angle-turn trial; no pooling with traj_4.",
            "Only actual validation plant mu varies; prediction remains .7, with identical H30/B10/C64/I3/E8, seed42, task, initialization, model and actuator bytes.",
            "Settings file defaults remain H15/B5; effective H30/B10 are explicit fixed overrides, not hidden file retuning.",
            "No supplementary-specific PPO/SAC checkpoint or Isaac recording is supplied or inferred; no sim2sim, qualified freeze, stability or real-time claim.",
            "Original clocks/world axes retained without alignment/time warp. Measured COM velocities and ground contacts are not differentiated or re-simulated.",
            "Reference contact is UNMEASURED. Missing eligible contact is not proof of flight; height, tilt, flight, heading offsets and slip are descriptive, not added failure gates.",
            "Only the declared core uses the existing XY screen; event windows are descriptive and full playback includes synthetic recovery/tail.",
            "Simulator waits for the solver; wall-latency actuation delay is not injected, and finite failed-controller fallbacks are retained."]}
    safe_configuration(summary)
    compact = public_compact(traces, recon); output.mkdir(parents=True)
    np.savez_compressed(output/"development_trace.npz", **compact)
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
    result.add_argument("--run", action="append", required=True, metavar="MU=DIR")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "complete_predeclared_grid": result["complete_predeclared_grid"],
        "evaluated_cases": len(result["evaluated_mu"]), "reference_id": result["reference_id"]}))

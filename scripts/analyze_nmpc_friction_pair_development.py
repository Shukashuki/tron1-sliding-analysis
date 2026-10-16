"""Offline two-engine fixed-CEM friction DEVELOPMENT, never auto-qualified.

Combine real Isaac episodes with the hash-bound MuJoCo development replay and
existing PPO/SAC paired XY grids. Preserve original clocks, material readbacks,
solver failures and cross-engine differences. No simulation or optimization.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import numpy as np

from analyze_controller_pair import (load_arrays, pair_xy_gap, plant_source_provenance,
    posture_diagnostic, read_json, reference_trace, sha256, source_model)
from analyze_nmpc_friction_development import (EVALUATION, FK_HELPERS, PILOT_CEM,
    PILOT_CONFIG_SHA256, PILOT_SOURCE_SHA256, SELECTED_MU, case_name, expected_grid,
    load_policy_grid)
from analyze_nmpc_nominal_development import (
    contact_diagnostic, require, safe_configuration, solver_diagnostic, source_digest,
)
from analyze_tvlqr_pilot import PUBLIC_SIGNALS, PUBLIC_TRACE_FIELDS, digest, public_compact
from analyze_policy_friction_dynamics import (
    audit_xy_replay as audit_policy_xy_numeric, reference_numeric_representation_audit,
)
from controller_metrics import summarize_turn
from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT, verify_reference_pair
from controller_trace import WheelKinematics, native_trace
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES
from nmpc_validation_contract import (
    _controller_contract, _material, _native_identity, episode_evidence, validate_case_against_nominal,
)


ISAAC_RUNNER_SHA256 = "494387154c15cba786f771356a38d1a51514821112e055cb50cad462d6eabb09"
ISAAC_NATIVE_LIBRARY_SHA256 = "ce676e8414d38beb0676e8529cd88ebfea980ef9d91494c052e21414c4f89ea5"
COMMON_SOURCE_NAMES = ("run_nmpc_tracking.py", "nmpc_shooting.py", "nmpc_tracking_objective.py",
    "classical_tracking.py", "controller_references.py", "height_smoothness.py", "nmpc_batch_objective.py",
    "nmpc_native_batch.py", "nmpc_batch.cpp", "eval_tracking_mujoco.py", "eval_tracking.py",
    "tron1_domain_randomization.py", "tron1_contact_slip.py", "nmpc_height_smooth_residual.json",
    "car_drift_height_smooth_task.json")
POLICY_ANALYSIS_SOURCES = ("analyze_policy_friction_dynamics.py", "analyze_controller_pair.py",
    "analyze_nmpc_friction_development.py", "analyze_tvlqr_pilot.py", "controller_trace.py",
    "controller_metrics.py", "controller_references.py")


def parse_runs(values):
    require(bool(values), "Provide at least one --run MU=ISAAC_DIR")
    result = {}
    for value in values:
        number, separator, directory = value.partition("=")
        require(separator and directory, "Use MU=ISAAC_DIR run arguments")
        try: mu = float(number)
        except ValueError as error: raise ValueError("Run coefficient must be numeric") from error
        require(np.isfinite(mu) and mu in expected_grid() and mu not in result,
                "Duplicate coefficient or coefficient outside predeclared grid")
        path = Path(directory)
        require(path.is_dir(), "Missing original Isaac run directory")
        result[mu] = path
    require(len({path.resolve() for path in result.values()}) == len(result), "Isaac directories must be distinct")
    return dict(sorted(result.items()))


def summarize_trace(trace, geometry, reference, reference_geometry, interval):
    contacts = {} if "contact_count" not in trace else dict(wheel_ground_contact_count=trace["contact_count"],
        wheel_ground_force_w=trace["contact_force"], wheel_contact_lateral_slip_m_s=trace["contact_slip"])
    return summarize_turn(trace["time_s"], trace["position"], geometry["signals"], reference["time_s"],
        reference["position"][:, :2], reference_geometry["signals"]["speed_m_s"], interval, **contacts)


def compact_case(data, prefix, *, require_nonwheel=True):
    trace = {name: data[f"{prefix}_{name}"].copy() for name in PUBLIC_TRACE_FIELDS if f"{prefix}_{name}" in data}
    geometry = {"signals": {name: data[f"{prefix}_{name}"].copy() for name in PUBLIC_SIGNALS},
        **{name: data[f"{prefix}_{name}"].copy() for name in ("wheel_quaternion_wxyz", "wheel_center_velocity_w")}}
    n = 349 if prefix == "reference" else 1397
    dt = .02 if prefix == "reference" else .005
    require(trace.get("time_s", np.empty(0)).shape == (n,)
            and np.allclose(trace["time_s"], np.arange(n)*dt, rtol=0, atol=1e-9), "Invalid compact native clock")
    for name, shape in (("position", (n, 3)), ("quaternion", (n, 4)), ("com_velocity", (n, 3)),
                        ("angular_velocity", (n, 3)), ("wheel_pos", (n, 2, 3))):
        require(trace.get(name, np.empty(0)).shape == shape and np.isfinite(trace[name]).all(),
                "Invalid compact physical geometry: "+name)
    for values in geometry["signals"].values():
        require(values.dtype.kind in "biuf" and np.isfinite(values).all(), "Nonfinite compact heading/speed signal")
    if prefix != "reference":
        fields = [("contact_count", (n, 2)), ("contact_force", (n, 2, 3)), ("contact_slip", (n, 2))]
        nonwheel = ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n")
        require(require_nonwheel or all(name in trace for name in nonwheel) or all(name not in trace for name in nonwheel),
                "Incomplete optional nonwheel-contact evidence")
        if require_nonwheel or all(name in trace for name in nonwheel): fields.extend((name, (n,)) for name in nonwheel)
        for name, shape in fields:
            require(trace.get(name, np.empty(0)).shape == shape and np.isfinite(trace[name]).all(),
                    "Missing measured compact contact field: "+name)
    return trace, geometry


def load_mujoco_bundle(directory, spec, inputs):
    directory = Path(directory); summary = read_json(directory/"summary.json")
    require(summary.get("kind") == "unqualified_mujoco_cem_fixed_friction_development"
            and summary.get("status") == "development_analysis_only" and summary.get("engine") == "MuJoCo"
            and summary.get("complete_predeclared_grid") is True
            and summary.get("evaluated_mu") == expected_grid() and summary.get("reference_spec") == spec,
            "Require complete real 17-case MuJoCo development bundle")
    require(summary.get("input_sha256") == inputs and summary.get("effective_cem_config") == PILOT_CEM
            and summary.get("effective_cem_config_sha256") == PILOT_CONFIG_SHA256
            and summary.get("prediction_mu_scenarios") == [.7], "MuJoCo input/controller identity mismatch")
    for flag in ("two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
                 "eligible_for_controller_ranking", "height_or_flight_failure_gate", "heading_offset_is_mismatch"):
        require(summary.get(flag) is False, "Unexpected MuJoCo development qualification/gate: "+flag)
    require(summary.get("analysis_source_sha256", {}).get("analyze_nmpc_friction_development.py")
            == sha256(ROOT/"scripts/analyze_nmpc_friction_development.py"), "MuJoCo development analyzer source differs")
    hashes = {"summary.json": sha256(directory/"summary.json")}
    for name, filename in (("compact_trace", "development_trace.npz"), ("topdown", "topdown.png")):
        require(summary.get("artifacts", {}).get(name, {}).get("file") == filename, "Unexpected MuJoCo artifact filename")
        hashes[filename] = sha256(directory/filename)
        require(summary["artifacts"][name].get("sha256") == hashes[filename], "MuJoCo bundle artifact SHA-256 mismatch")
    data = load_arrays(directory/"development_trace.npz")
    require(all(not array.dtype.hasobject for array in data.values()), "Object arrays forbidden in compact replay")
    reference, reference_geometry = compact_case(data, "reference")
    traces, geometries = {}, {}
    interval = spec["evaluation_core_interval_s"]
    require(summarize_trace(reference, reference_geometry, reference, reference_geometry, interval)
            == summary["windows"]["core"]["reference"], "MuJoCo bundle reference metrics fail replay")
    for mu in expected_grid():
        case = case_name(mu); trace, geometry = compact_case(data, case)
        require(summary.get("runs", {}).get(case, {}).get("complete_physical_evidence") is True,
                "MuJoCo case lacks complete physical evidence")
        require(summarize_trace(trace, geometry, reference, reference_geometry, interval)
                == summary["windows"]["core"][case], "MuJoCo case canonical metrics fail exact compact replay")
        require(posture_diagnostic(trace) == summary["posture_diagnostics"][case]
                and contact_diagnostic(trace, interval) == summary["contact_diagnostics"][case],
                "MuJoCo posture/contact diagnostics fail replay")
        traces[case], geometries[case] = trace, geometry
    return summary, hashes, data, reference, reference_geometry, traces, geometries


def load_policy_dynamics_bundle(directory, spec, inputs, policies, provenances, reference_proof, reference):
    """Independently replay derived original-policy dynamics, without weights.

    The full compact bundle is optional; without it the legacy XY-only scope
    remains explicit. Recorded nonwheel contact may be absent, never invented.
    """
    directory = Path(directory); summary = read_json(directory/"summary.json")
    require(summary.get("kind") == "existing_policy_friction_dynamics_replay"
            and summary.get("reference_spec") == spec and summary.get("evaluation_mu") == expected_grid(),
            "Wrong original-policy dynamics reference/grid")
    require(summary.get("input_sha256") == {"motion": inputs["motion_file"], "reference_file": inputs["reference_file"], "model": inputs["model"]},
            "Policy dynamics model/reference hashes differ")
    for name in ("new_policy_evaluation", "new_training", "dynamics_stepping", "learned_inference",
                 "height_or_flight_failure_gate", "heading_offset_is_mismatch", "eligible_for_controller_ranking"):
        require(summary.get(name) is False, "Unexpected policy dynamics declaration: "+name)
    require(summary.get("reference_numeric_representation_audit") == reference_proof,
            "Policy reference representation proof differs from actual locked inputs")
    require(set(summary.get("analysis_source_sha256", {})) == set(POLICY_ANALYSIS_SOURCES),
            "Unexpected policy dynamics analysis source roles")
    for name, value in summary["analysis_source_sha256"].items():
        require(sha256(ROOT/"scripts"/name) == value, "Policy dynamics analysis implementation differs: "+name)
    require(summary.get("analysis_source_sha256", {}).get("analyze_policy_friction_dynamics.py")
            == sha256(ROOT/"scripts/analyze_policy_friction_dynamics.py"), "Missing policy dynamics implementation identity")
    hashes = {"summary.json": sha256(directory/"summary.json")}
    for filename in ("policy_dynamics_trace.npz", "topdown.png"):
        hashes[filename] = sha256(directory/filename)
        require(summary.get("artifacts", {}).get(filename, {}).get("file") == filename
                and summary["artifacts"][filename].get("sha256") == hashes[filename], "Policy dynamics artifact SHA-256 mismatch")
    data = load_arrays(directory/"policy_dynamics_trace.npz")
    policy_reference, ref_geometry = compact_case(data, "reference")
    for name in ("time_s", "position", "quaternion", "com_velocity", "angular_velocity", "wheel_pos"):
        require(np.array_equal(policy_reference[name], reference[name]), "Policy dynamics reference changed or aligned")
    interval = spec["evaluation_core_interval_s"]
    require(set(summary.get("runs", {})) == {name+"_"+case_name(mu) for name in policies for mu in expected_grid()},
            "Require all 34 original policy pairs")
    for name, (public, _, original_xy) in policies.items():
        recorded = summary.get("policies", {}).get(name, {})
        require(recorded.get("source_xy_grid") == public and recorded.get("original_physical_archives_used") is True
                and recorded.get("measured_com_speed_and_ground_contact_available") is True,
                "Policy dynamics source differs from original published policy grid")
        for mu in expected_grid():
            case = case_name(mu); run = summary["runs"][name+"_"+case]
            require(run.get("algorithm") == name and run.get("mu") == mu, "Wrong policy pair label")
            expected_raw = {role: provenances[name]["cases"][case][role+"_sha256"] for role in (
                "isaac_trace", "isaac_report", "mujoco_trace", "mujoco_report")}
            require(run.get("input_sha256") == expected_raw, "Policy dynamics original raw archive identities differ")
            training = run.get("training", {})
            require(training.get("verified_formal_training") is True and training.get("checkpoint_bytes_verified") is True
                    and training.get("manifest_sha256") == public["training_manifest_sha256"]
                    and training.get("checkpoint_sha256") == public["checkpoint_sha256"], "Policy dynamics training/checkpoint evidence differs")
            pair = {}; pair_geometry = {}
            for engine in ("isaac", "mujoco"):
                trace, geometry = compact_case(data, name+"_"+case+"_"+engine, require_nonwheel=False)
                require(np.array_equal(trace["time_s"], original_xy[case+"_"+engine+"_t"])
                        and np.array_equal(trace["position"][:, :2], original_xy[case+"_"+engine+"_xy"]),
                        "Policy dynamics XY differs from original recorded grid; no registration allowed")
                metrics = summarize_trace(trace, geometry, policy_reference, ref_geometry, interval)
                require(metrics == run["windows"]["core"][engine], "Policy speed/contact/heading metrics fail exact replay")
                numeric_replay = audit_policy_xy_numeric(metrics, policies[name][1][mu][engine],
                    reference_audit=reference_proof, observed_xy=trace["position"][:, :2])
                require(numeric_replay == run.get("xy_numeric_representation_replay", {}).get(engine),
                        "Policy legacy/canonical numeric representation audit fails replay")
                require(posture_diagnostic(trace) == run["posture_diagnostics"][engine], "Policy posture diagnostic fails replay")
                pair[engine], pair_geometry[engine] = trace, geometry
            require(pair_xy_gap(pair["isaac"], pair["mujoco"], interval) == run["windows"]["core"]["sim2sim_xy_gap"],
                    "Policy paired XY gap fails exact replay")
    return {"input_sha256": hashes, "original_physical_archives_used": True,
        "reference_numeric_representation_audit": reference_proof,
        "evidence_scope": "Exactly replayed existing hash-bound physical policy recordings; no new inference/training or nonwheel-contact imputation"}, data, summary["runs"]


def basename_source(record, name):
    records = record.get("reported_source_provenance", {}).get("reported_source_hashes", [])
    values = {item.get("sha256") for item in records if item.get("source_name") == name}
    require(len(values) == 1, "Missing/ambiguous source identity: "+name)
    return digest(values.pop())


def audit_device(value):
    device = safe_configuration(value)
    require(isinstance(device, dict) and device.get("status") == "verified"
            and device.get("native_and_tensor_device_consistency_verified") is True
            and device.get("required_unmasked") is True and device.get("cuda_visible_devices") is None,
            "Require actual unmasked native/tensor Isaac device audit")
    requested = device.get("requested_device", "")
    require(isinstance(requested, str) and re.fullmatch(r"cuda:[0-9]+", requested), "Invalid actual Isaac CUDA ordinal")
    index = int(requested.split(":")[1])
    require(all(device.get(name) == requested for name in ("environment_device", "physics_sim_view_frontend_device",
            "robot_native_transform_tensor_device")) and all(device.get(name) == index for name in (
            "native_physics_backend_device_ordinal", "torch_current_device", "physics_cudaDevice_setting", "device_index")),
            "Isaac native/tensor device identity disagreement")
    return device


def material_projection(raw):
    """Keep numeric material readbacks, not filesystem-looking USD paths."""
    result = {name: safe_configuration(raw.get(name)) for name in
        ("expected_mu", "robot_static_dynamic_range", "robot_restitution_range")}
    ground = raw.get("ground_bound_materials", [])
    require(np.allclose(result["robot_restitution_range"], 0., rtol=0, atol=1e-8)
            and all(len(item.get("coefficients", [])) == 3 and abs(item["coefficients"][2]) <= 1e-8 for item in ground),
            "Actual Isaac restitution differs from fixed zero contract")
    result["ground_bound_materials"] = [{"coefficients": safe_configuration(item["coefficients"])} for item in ground]
    result["scope"] = "Numeric native robot/bound-ground material readbacks; USD scene addresses omitted; original report SHA-256 retained"
    return result


def actuator_projection(raw):
    result = safe_configuration(raw)
    require(isinstance(result, dict), "Missing realized Isaac actuator readback")
    for group, kp, kd, effort, speed in (("legs", 500., 10., 80., 15.), ("wheels", 0., 0., 12., 100.)):
        expected = {"stiffness": kp, "damping": kd, "effort_limit_nm": effort, "velocity_limit_rad_s": speed,
            "saturation_effort_nm": effort, "velocity_at_effort_limit_rad_s": 2*speed}
        for name, value in expected.items():
            measured = np.asarray(result.get(group, {}).get(name, []), dtype=float)
            require(measured.shape == (2,) and np.isfinite(measured).all()
                    and np.allclose(measured, value, atol=1e-6, rtol=1e-6), "Realized Isaac motor differs: "+group+"."+name)
    return result


def audit_isaac_header(report, mu, spec, model_sha256, mj_record):
    require(report.get("kind") == "receding_horizon_nmpc" and report.get("engine") == "Isaac/PhysX"
            and report.get("reference_spec") == spec and report.get("mujoco_model_sha256") == model_sha256,
            "Isaac controller/reference/model identity mismatch")
    require(type(report.get("plant_mu")) in (int, float) and report["plant_mu"] == mu
            and report.get("effective_cem_config") == PILOT_CEM and report.get("prediction_mu_scenarios") == [.7]
            and report.get("plant_seed") == 42, "Actual Isaac plant/controller/seed differs")
    _material(report, "isaac", mu)
    for name in ("learned_policy_used", "tvlqr_used", "action_replay", "root_state_prescribed_after_reset",
                 "external_root_forces_used", "prediction_receives_plant_mu", "pose_height_failure_gate",
                 "truncated_smoke", "robustness_certificate", "real_time_verified"):
        require(report.get(name) is False, "Unsupported Isaac declaration: "+name)
    require(report.get("model_generated_nominal_plan") == {"enabled": False}
            and report.get("adaptation_contract") is None, "No model plan/adaptation in fixed grid")
    for suffix, value in PILOT_SOURCE_SHA256.items():
        require(source_digest(report, suffix) == value, "Changed common controller source: "+suffix)
    require(source_digest(report, "scripts/run_nmpc_tracking_isaac.py") == ISAAC_RUNNER_SHA256,
            "Different approved Isaac runner source")
    provenance = {"reported_source_provenance": plant_source_provenance(report)}
    for name in COMMON_SOURCE_NAMES:
        require(basename_source(provenance, name) == basename_source(mj_record, name),
                "Different shared controller/predictor/actuator input: "+name)
    actuator_hashes = {value for name, value in report.get("source_sha256", {}).items()
        if Path(name).name in ("policy_contract.json", "nominal_actuator_contract.json")}
    require(actuator_hashes == {basename_source(mj_record, "policy_contract.json")},
            "Actual actuator-contract bytes differ (host filenames may differ)")
    ia_objective, mj_objective = report.get("objective_contract", {}), mj_record["controller_contract"]["objective_contract"]
    for name in ("body_names", "joint_names", "dt_s", "weights", "stds", "height_regularization",
                 "absolute_height_target", "reference_vertical_velocity_target", "reference_order"):
        require(ia_objective.get(name) == mj_objective.get(name), "Different common task objective: "+name)
    require(report.get("action_parameterization") == mj_record["controller_contract"]["action_parameterization"],
            "Different cross-engine action decoder")
    identity = _native_identity(report, "isaac")
    require(identity["backend"] == "native" and identity["native_library_sha256"] == ISAAC_NATIVE_LIBRARY_SHA256,
            "Different approved Isaac native binary")
    contract = _controller_contract(report, "isaac")
    ia_workers = contract["backend"]["workers"]; mj_workers = mj_record["controller_contract"]["backend"]["workers"]
    require(set(ia_workers) == set(mj_workers) == {"0.7"}, "Require fixed singleton prediction model")
    for name in ("source_sha256", "mujoco_header_sha256", "mujoco_library_sha256", "mujoco_version",
                 "header_version", "abi_version", "model_snapshot_sha256", "body_names", "joint_names", "physics_dt_s", "control_dt_s"):
        require(ia_workers["0.7"].get(name) == mj_workers["0.7"].get(name),
                "Cross-engine prediction build/model identity differs: "+name)
    for name in ("source_sha256", "mujoco_header_sha256", "mujoco_library_sha256", "model_snapshot_sha256"):
        digest(ia_workers["0.7"].get(name))
    require(contract["backend"].get("production_cem_wired") is True,
            "Native predictor must be attached to actual CEM controller")
    return {"actual_plant_mu": float(mu), "plant_seed": 42, "asset_sha256": digest(report.get("asset_sha256")),
        "device_audit": audit_device(report.get("device_audit")),
        "material_audit": material_projection(report["material_audit"]),
        "actuator_audit": actuator_projection(report.get("actuator_audit")),
        "controller_contract": safe_configuration(contract), "native_identity": safe_configuration(identity),
        "reported_source_provenance": provenance["reported_source_provenance"],
        "fk_source_helpers_sha256": {name: source_digest(report, name) for name in FK_HELPERS}}


def audit_isaac_trace(report, raw, log):
    trace = native_trace(raw, "nmpc")
    counter = np.asarray(raw["simulation_step_counter"])
    require(counter.shape == (1397,) and counter.dtype.kind in "iu"
            and np.array_equal(counter-counter[0], np.arange(1397)), "Isaac physical simulator counter skipped/reset")
    for name in ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"):
        trace[name] = np.asarray(raw[name], dtype=float).copy()
    packed = np.asarray(raw["packed_controller_states"], dtype=float)
    require(packed.shape == (349, 111) and np.isfinite(packed).all()
            and np.allclose(packed[:, :3], trace["position"][::4][:-1], rtol=0, atol=1e-9)
            and np.allclose(np.linalg.norm(packed[:, 3:7], axis=1), 1., rtol=0, atol=1e-5)
            and np.allclose(np.abs(np.sum(packed[:, 3:7]*trace["quaternion"][::4][:-1], axis=1)), 1., rtol=0, atol=1e-5),
            "Isaac pre-action converted states do not match measured root position")
    command = np.asarray([row.get("raw_cem_action", []) for row in log], dtype=float)
    require(command.shape == (349, 8) and np.isfinite(command).all() and np.max(np.abs(command)) <= 1.+1e-9
            and np.array_equal(command.astype(np.float32).astype(float), raw["actions"]),
            "Isaac raw CEM action differs from actual applied float32 command")
    conversions, rewards, stage = [], [], []
    for index, row in enumerate(log):
        info = row["solver"]
        require(info.get("fallback_used") == info.get("optimization_failed"), "Isaac CEM fallback/failure declarations differ")
        if info.get("fallback_used"):
            previous = np.zeros(8) if index == 0 else raw["actions"][index-1]
            require(np.array_equal(raw["actions"][index], previous), "Isaac finite solver fallback must hold previous applied command")
        objective = row.get("actual_isaac_objective", {})
        values = [objective.get("stage_cost"), objective.get("total_reward"), row.get("actual_dt_scaled_reward"),
                  row.get("same_state_dt_reward_abs_error"), row.get("coordinate_conversion_max_velocity_error")]
        require(all(type(value) in (int, float) and np.isfinite(value) for value in values),
                "Missing actual Isaac objective/reward/conversion measurements")
        require(values[3] >= 0 and 0 <= values[4] <= 1e-8
                and np.isclose(values[3], abs(values[2]-.02*values[1]), rtol=1e-12, atol=1e-12),
                "Isaac reward/conversion log inconsistency")
        stage.append(values[0]); rewards.append(values[3]); conversions.append(values[4])
    require(report.get("maximum_same_state_dt_reward_abs_error") == max(rewards)
            and report.get("maximum_coordinate_conversion_error") == max(conversions),
            "Isaac report-bound reward/conversion diagnostic differs")
    # This is an explicit timing-field adapter only. No kind, mu, material,
    # configuration, objective, action or outcome declaration is rewritten.
    timing = dict(report, wall_s=report.get("elapsed_s"))
    diagnostic = solver_diagnostic(timing, raw, log, {"mode": "warm_start_residual_blocks"})
    diagnostic["timing"]["reported_episode_elapsed_s"] = diagnostic["timing"].pop("recorded_rollout_wall_s")
    diagnostic["timing"]["scope"] = "Original Isaac replanning calls including native IPC; CUDA extraction, plant stepping and logging excluded from call times. Episode elapsed_s has its original broader runner scope."
    diagnostic["actual_isaac_objective"] = {"summed_stage_cost": float(np.sum(stage)),
        "maximum_dt_reward_reconstruction_abs_error": max(rewards),
        "maximum_coordinate_conversion_velocity_abs_error": max(conversions),
        "scope": "Saved same-state actual Isaac reward reconstruction, not prediction accuracy or exact cross-engine objective parity"}
    return trace, diagnostic


def audit_isaac_episode(directory, mu, spec, model_sha256, mj_record):
    directory = Path(directory); report = read_json(directory/"report.json")
    require(report.get("kind") == "receding_horizon_nmpc" and report.get("engine") == "Isaac/PhysX"
            and report.get("reference_spec") == spec, "Wrong Isaac controller/reference even for censored evidence")
    require(report.get("status") in ("completed", "failed", "running", "initializing"), "Unsupported Isaac report status")
    report_sha = sha256(directory/"report.json")
    if report["status"] != "completed":
        header = audit_isaac_header(report, mu, spec, model_sha256, mj_record) if "effective_cem_config" in report else {}
        return None, None, {**header, "input_sha256": {"report": report_sha}, "censored": True,
            "complete_physical_evidence": False, "recorded_report_status": report["status"],
            "actual_controller_identity_available": bool(header), "physical_xy_failure_claimed": False,
            "scope": "No complete hash-bound physical episode; report alone does not prove process exit, XY failure or success"}
    header = audit_isaac_header(report, mu, spec, model_sha256, mj_record)
    evidence = episode_evidence(directory, "isaac", spec,
        {"effective_cem_config": copy.deepcopy(PILOT_CEM), "prediction_mu_scenarios": [.7]}, mu)
    raw = load_arrays(directory/"rollout.npz"); log = json.loads((directory/"solver_log.json").read_text())
    trace, diagnostic = audit_isaac_trace(report, raw, log)
    return evidence, trace, {**header, "input_sha256": evidence["hashes"], "censored": False,
        "complete_physical_evidence": True, "actual_controller_identity_available": True,
        "solver_diagnostics": diagnostic,
        "simulation_budget": {"requested_prediction_substeps": diagnostic["counts"]["scenario_evaluations"]*30*4,
            "actual_plant_physics_substeps": 1396, "failed_requests_prove_completed_native_steps": False}}


def render_topdown(data, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(5, 4, figsize=(16., 17.5), squeeze=False)
    spec = summary["reference_spec"]; start, end = spec["source_demo_interval_s"]
    lower, upper = spec["evaluation_core_interval_s"]
    rt, rp = data["reference_time_s"], data["reference_position"][:, :2]
    full_policy = summary.get("policy_feature_scope") == "original_physical_recording_dynamics"
    ref = rp[(rt >= start-1e-9) & (rt <= end+1e-9)]
    for row, mu in enumerate(SELECTED_MU):
        case = case_name(mu); bounds = [ref]
        for column, algorithm in enumerate(("Reference", "PPO", "SAC", "NMPC")):
            axis = axes[row, column]; axis.plot(ref[:, 0], ref[:, 1], "--", color=".7", lw=.8)
            if algorithm == "Reference":
                axis.plot(ref[:, 0], ref[:, 1], color=".3", lw=1.7)
                full = (rt >= start-1e-9) & (rt <= end+1e-9)
                indices = np.flatnonzero(full)[::max(1, round(.4/float(data["reference_physics_dt_s"])))]
                wheels = data["reference_wheel_pos"]
                for side in range(2):
                    for index in indices:
                        if not data["reference_root_course_to_wheel_valid"][index, side]: continue
                        heading = data["reference_wheel_rolling_plane_heading_rad"][index, side]
                        delta = .045*np.array([np.cos(heading), np.sin(heading)])
                        segment = wheels[index, side, :2]+np.array([-delta, delta])
                        # Reference contacts are unmeasured: no contact gate or
                        # pale/grounded interpretation is invented for ticks.
                        axis.plot(segment[:, 0], segment[:, 1], color=".3", lw=.7, alpha=.85)
                arrows = indices[data["reference_root_beta_valid"][indices]]
                course = data["reference_course_rad"][arrows]
                axis.quiver(rp[arrows, 0], rp[arrows, 1], np.cos(course), np.sin(course), color=".3",
                    angles="xy", scale_units="xy", scale=8, width=.003)
                bounds.append(wheels[full, :, :2].reshape(-1, 2))
            else:
                for engine, color in (("isaac", "#d62728"), ("mujoco", "#1f77b4")):
                    prefix = (engine+"_"+case if algorithm == "NMPC" else algorithm.lower()+"_"+case+"_"+engine)
                    key = "position" if algorithm == "NMPC" or full_policy else "xy"
                    if prefix+"_time_s" not in data:
                        axis.text(.02, .03, "Isaac censored: no full trace", transform=axis.transAxes, fontsize=7)
                        continue
                    time, pos = data[prefix+"_time_s"], data[prefix+"_"+key][:, :2]
                    full = (time >= start-1e-9) & (time <= end+1e-9)
                    core = full & (time >= lower-1e-9) & (time <= upper+1e-9)
                    axis.plot(pos[full, 0], pos[full, 1], color=color, lw=.8, alpha=.45)
                    axis.plot(pos[core, 0], pos[core, 1], color=color, lw=1.7)
                    bounds.append(pos[full])
                    if algorithm == "NMPC" or full_policy:
                        indices = np.flatnonzero(full)[::max(1, round(.4/float(data[prefix+"_physics_dt_s"])))]
                        for side in range(2):
                            for index in indices:
                                if not data[prefix+"_root_course_to_wheel_valid"][index, side]: continue
                                heading = data[prefix+"_wheel_rolling_plane_heading_rad"][index, side]
                                delta = .045*np.array([np.cos(heading), np.sin(heading)])
                                segment = data[prefix+"_wheel_pos"][index, side, :2]+np.array([-delta, delta])
                                contact = data[prefix+"_contact_count"][index, side] > 0 and data[prefix+"_contact_force"][index, side, 2] > 5.
                                axis.plot(segment[:, 0], segment[:, 1], color=color, lw=.7, alpha=.85 if contact else .12)
                        arrows = indices[data[prefix+"_root_beta_valid"][indices]]
                        course = data[prefix+"_course_rad"][arrows]
                        axis.quiver(pos[arrows, 0], pos[arrows, 1], np.cos(course), np.sin(course), color=color,
                            angles="xy", scale_units="xy", scale=8, width=.003)
            title = f"{algorithm} · μ={mu:g}"+(" (XY only)" if algorithm in ("PPO", "SAC") and not full_policy else "")
            if algorithm == "Reference": title += "\nKinematic; contacts UNMEASURED"
            axis.set(title=title,
                xlabel="World X [m]", ylabel="World Y [m]")
            axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
        xy = np.concatenate(bounds); lo, hi = xy.min(axis=0), xy.max(axis=0); margin = max(.15, .06*float(np.max(hi-lo)))
        for axis in axes[row]: axis.set_xlim(lo[0]-margin, hi[0]+margin); axis.set_ylim(lo[1]-margin, hi[1]+margin)
    fig.suptitle("Two-engine friction DEVELOPMENT · not qualified/frozen/ranked · original clocks/world axes", fontsize=12)
    fig.legend(handles=[Line2D([], [], color="#d62728", lw=1.7, label="Isaac"),
        Line2D([], [], color="#1f77b4", lw=1.7, label="MuJoCo"),
        Line2D([], [], color=".3", lw=1.7, label="Thick: core; arrows: COM velocity; ticks: wheel plane (pale: no gated contact)")], loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, .035, 1, .975)); fig.savefig(output, dpi=150); plt.close(fig)


def policy_xy(row):
    return {"reference_xy_rmse_m": row["full_core_reference_xy_rmse_m"],
        "reference_xy_peak_m": row["full_core_reference_xy_peak_m"], "coverage_complete": row["core_complete"],
        "large_xy_exploratory_screen": row["large_reference_xy_screen"], "measured_com_speed": None,
        "wheel_heading": None, "measured_contact_slip": None, "data_availability": "Published native-clock XY only"}


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing output; choose a NEW output directory")
    directories = parse_runs(args.run)
    spec = verify_reference_pair(args.motion_file, args.reference_file, PRIMARY_REFERENCE_ID)
    inputs = {name: sha256(getattr(args, name)) for name in ("motion_file", "reference_file", "model")}
    mj, mj_hashes, mj_data, reference, reference_geometry, mj_traces, mj_geometry = load_mujoco_bundle(args.mujoco_bundle, spec, inputs)
    actual_reference = reference_trace(load_arrays(args.motion_file))
    for name in ("time_s", "position", "quaternion", "com_velocity", "angular_velocity", "wheel_pos"):
        require(np.array_equal(actual_reference[name], reference[name]), "Compact reference differs from locked actual motion")
    policies = {name: load_policy_grid(getattr(args, name.lower()+"_grid"), name, spec) for name in ("PPO", "SAC")}
    require({name: value[0] for name, value in policies.items()} == mj["policy_grids"],
            "Policy grid bytes/metadata differ from supplied MuJoCo bundle")
    policy_dynamics = policy_dynamics_data = policy_dynamics_runs = None
    if getattr(args, "policy_dynamics", None) is not None:
        reference_proof = reference_numeric_representation_audit(load_arrays(args.motion_file), load_arrays(args.reference_file))
        provenances = {name: read_json(Path(getattr(args, name.lower()+"_grid"))/"xy_trace.provenance.json") for name in policies}
        policy_dynamics, policy_dynamics_data, policy_dynamics_runs = load_policy_dynamics_bundle(
            args.policy_dynamics, spec, inputs, policies, provenances, reference_proof, reference)
    audits, traces = {}, {}; first = None; first_header = None
    for mu, directory in directories.items():
        case = case_name(mu)
        evidence, trace, audit = audit_isaac_episode(directory, mu, spec, inputs["model"], mj["runs"][case])
        if audit["actual_controller_identity_available"]:
            header = {name: audit[name] for name in ("controller_contract", "native_identity", "asset_sha256", "fk_source_helpers_sha256")}
            require(first_header is None or first_header == header, "Isaac controller/source/asset identity changed across mu")
            first_header = header
        if evidence is not None:
            if first is None: first = evidence
            else: validate_case_against_nominal(evidence, first)
            traces["isaac_"+case] = trace
        audits[case] = audit
    with source_model(args.source_root, args.model) as (model, loader):
        require(loader.get("source_helpers_sha256") == mj["model_loader_audit"]["source_helpers_sha256"],
                "Pair FK loader differs from hash-bound MuJoCo replay")
        fk = WheelKinematics(model); geometry = {name: fk.reconstruct(trace) for name, trace in traces.items()}
    interval = spec["evaluation_core_interval_s"]
    ia_metrics = {name: summarize_trace(trace, geometry[name], reference, reference_geometry, interval) for name, trace in traces.items()}
    rows = []
    for mu in expected_grid():
        case = case_name(mu); ia_name = "isaac_"+case
        nmpc = {"isaac": ia_metrics.get(ia_name, {"coverage_complete": False, "censored": True,
            "reference_xy_rmse_m": None, "reference_xy_peak_m": None, "large_xy_exploratory_screen": None}),
            "mujoco": mj["windows"]["core"][case],
            "sim2sim": pair_xy_gap(traces[ia_name], mj_traces[case], interval) if ia_name in traces else None}
        row = {"mu": mu, "NMPC": nmpc}
        for name, (_, saved, _) in policies.items():
            saved_row = saved[mu]
            if policy_dynamics_runs is None:
                row[name] = {engine: policy_xy(saved_row[engine]) for engine in ("isaac", "mujoco")}
                row[name]["sim2sim"] = {"xy_rmse_m": saved_row["sim2sim"]["full_core_xy_rmse_m"],
                    "coverage_complete": saved_row["sim2sim"]["full_core_overlap"], "source": "Exactly replayed existing paired XY grid"}
            else:
                run = policy_dynamics_runs[name+"_"+case]
                row[name] = {engine: run["windows"]["core"][engine] for engine in ("isaac", "mujoco")}
                row[name].update(sim2sim=run["windows"]["core"]["sim2sim_xy_gap"],
                    posture_diagnostics=run["posture_diagnostics"], data_availability="Original physical COM velocity, wheel geometry and measured ground contact; no new inference")
        rows.append(row)
    evaluated = [mu for mu in directories if not audits[case_name(mu)]["censored"]]
    summary = {"schema_version": 1, "kind": "unqualified_two_engine_cem_friction_development",
        "status": "development_analysis_only", "reference_id": PRIMARY_REFERENCE_ID, "reference_spec": spec,
        "provided_isaac_mu": list(directories), "evaluated_isaac_mu": evaluated, "expected_mu": expected_grid(),
        "missing_or_censored_isaac_mu": [mu for mu in expected_grid() if mu not in evaluated],
        "complete_two_engine_grid": evaluated == expected_grid(), "rows": rows, "isaac_runs": audits,
        "mujoco_bundle_sha256": mj_hashes, "policy_grids": {name: value[0] for name, value in policies.items()},
        "policy_feature_scope": "original_physical_recording_dynamics" if policy_dynamics is not None else "published_xy_only",
        "policy_dynamics_bundle": policy_dynamics,
        "effective_cem_config": copy.deepcopy(PILOT_CEM), "effective_cem_config_sha256": PILOT_CONFIG_SHA256,
        "prediction_mu_scenarios": [.7], "sim2sim_trajectory_comparison": True,
        **{name: False for name in ("two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
            "frozen_benchmark", "pooled_success_statistics", "eligible_for_controller_ranking", "robustness_certificate", "real_time_verified", "controlled_execution_success_claimed",
            "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "learned_inference", "actuation_delay_injected")},
        "model_loader_audit": loader, "isaac_fk_audit": {name: value["audit"] for name, value in geometry.items()},
        "isaac_posture_diagnostics": {name: posture_diagnostic(trace) for name, trace in traces.items()},
        "isaac_contact_diagnostics": {name: contact_diagnostic(trace, interval) for name, trace in traces.items()},
        "input_sha256": inputs, "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
        "evaluation_protocol_sha256": sha256(EVALUATION),
        "analysis_source_sha256": {name: sha256(ROOT/"scripts"/name) for name in (
            "analyze_nmpc_friction_pair_development.py", "analyze_nmpc_friction_development.py", "nmpc_validation_contract.py",
            "analyze_controller_pair.py", "analyze_tvlqr_pilot.py", "analyze_policy_friction_dynamics.py", "controller_metrics.py", "controller_trace.py")},
        "limitations": ["Two-engine descriptive development, not a nominal-qualified freeze, controller ranking, statistical success rate, hardware/real-time result or robustness certificate.",
            "Large XY error, finite solver fallbacks and fallen postures remain outcomes. Incomplete runtime evidence is censored, not a physical XY verdict.",
            "Only plant material varies; same CEM settings, seed42, privileged measured state/reference preview and fixed prediction mu=.7. No trajectory alignment or time warp.",
            "Isaac and MuJoCo native binaries differ, while recorded C++/headers/MuJoCo library/prediction-model identities match. Runtime soft limits and contact sensor contracts remain engine-specific.",
            "Policy dynamics are available only when the optional hash-bound original-physical-recording replay is supplied; the default original compact grids remain explicitly XY-only. Missing nonwheel contact is never imputed and existing policy data are not rerun.",
            "Equal numeric materials are not calibrated physical tire equivalence. Height, flight and heading offsets are descriptive, not extra failure gates.",
            "Original solve timings are uncontrolled; simulated time waits for optimization, without injected wall-clock actuation delay."]}
    compact = public_compact(traces, geometry)
    for prefix in ("reference", *[case_name(mu) for mu in expected_grid()]):
        destination = "reference" if prefix == "reference" else "mujoco_"+prefix
        for name in (*PUBLIC_TRACE_FIELDS, *PUBLIC_SIGNALS, "wheel_quaternion_wxyz", "wheel_center_velocity_w"):
            key = prefix+"_"+name
            if key in mj_data: compact[destination+"_"+name] = mj_data[key].copy()
    for name, (_, _, data) in policies.items():
        for mu in expected_grid():
            case = case_name(mu)
            for engine in ("isaac", "mujoco"):
                prefix = name.lower()+"_"+case+"_"+engine
                compact[prefix+"_time_s"] = data[case+"_"+engine+"_t"].copy()
                compact[prefix+"_xy"] = data[case+"_"+engine+"_xy"].copy()
                if policy_dynamics_data is not None:
                    original = name+"_"+case+"_"+engine
                    for field in (*PUBLIC_TRACE_FIELDS, *PUBLIC_SIGNALS, "wheel_quaternion_wxyz", "wheel_center_velocity_w"):
                        key = original+"_"+field
                        if key in policy_dynamics_data: compact[prefix+"_"+field] = policy_dynamics_data[key].copy()
    output.mkdir(parents=True); np.savez_compressed(output/"development_trace.npz", **compact)
    render_topdown(compact, summary, output/"topdown.png")
    summary["artifacts"] = {name: {"file": filename, "sha256": sha256(output/filename)}
        for name, filename in (("compact_trace", "development_trace.npz"), ("topdown", "topdown.png"))}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--motion", "--motion-file", dest="motion_file", type=Path, required=True)
    for name in ("reference-file", "model", "source-root", "mujoco-bundle", "ppo-grid", "sac-grid", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--run", action="append", required=True, metavar="MU=ISAAC_DIR")
    result.add_argument("--policy-dynamics", type=Path, help="Optional derived original-policy physical-recording replay; no inference")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "complete_two_engine_grid": result["complete_two_engine_grid"], "qualified": False}))

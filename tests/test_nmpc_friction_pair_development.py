"""Pure synthetic pair-development evidence; no assets, dynamics or network."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_nmpc_friction_pair_development as analysis
import analyze_nmpc_friction_development as mj_analysis
import analyze_nmpc_nominal_development as nominal
from analyze_controller_pair import plant_source_provenance
from controller_references import get_reference
from test_nmpc_friction_development import write_case
from test_nmpc_nominal_development import canonical, fixture_arguments, geometry


def fixture_reports(tmp_path, mu=.7, spec=None, model_sha=None):
    spec = spec or get_reference(); directory = tmp_path/"mj_source"; directory.mkdir(exist_ok=True)
    mj = write_case(directory, mu, spec=spec, model_sha256=model_sha)
    mj["action_parameterization"] = {"mode": "warm_start_residual_blocks", "learned_baseline": False}
    for index, name in enumerate(analysis.COMMON_SOURCE_NAMES):
        if not any(Path(key).name == name for key in mj["source_sha256"]):
            mj["source_sha256"]["/private/source/scripts/"+name] = f"{index+20:064x}"
    worker = mj["native_prediction_backend"]["workers"]["0.7"]
    worker.update(source_sha256=analysis.PILOT_SOURCE_SHA256["native/nmpc_batch.cpp"],
        mujoco_header_sha256="a"*64, mujoco_library_sha256="b"*64, model_snapshot_sha256="c"*64,
        mujoco_version="3.8.1", header_version=3008001, abi_version=1,
        body_names=list(analysis.BODY_NAMES), joint_names=list(analysis.JOINT_NAMES),
        physics_dt_s=.005, control_dt_s=.02)
    record = {"controller_contract": analysis._controller_contract(mj, "mujoco"),
        "native_identity": analysis._native_identity(mj, "mujoco"),
        "reported_source_provenance": plant_source_provenance(mj), "complete_physical_evidence": True}
    ia = copy.deepcopy(mj)
    ia.update(kind="receding_horizon_nmpc", engine="Isaac/PhysX", mujoco_model_sha256=mj["model_sha256"],
        plant_seed=42, asset_sha256="d"*64, fixed_evaluation={"all_termination_terms_removed": True},
        elapsed_s=12., physics_samples=1397, maximum_same_state_dt_reward_abs_error=0.,
        maximum_coordinate_conversion_error=0., action_replay=False)
    ia["source_sha256"]["/private/snapshot/scripts/run_nmpc_tracking_isaac.py"] = analysis.ISAAC_RUNNER_SHA256
    value = ia["source_sha256"].pop("/private/archived/policy_contract.json")
    ia["source_sha256"]["/private/bridge/config/nominal_actuator_contract.json"] = value
    ia["prediction_backend_metadata"] = copy.deepcopy(mj["native_prediction_backend"])
    ia["prediction_backend_metadata"]["workers"]["0.7"]["native_library_sha256"] = analysis.ISAAC_NATIVE_LIBRARY_SHA256
    ia["material_audit"] = {"expected_mu": mu, "robot_static_dynamic_range": [mu, mu],
        "robot_restitution_range": [0., 0.], "ground_bound_materials": [{"collider": "/World/ground/plane",
            "material": "/World/ground/material", "coefficients": [mu, mu, 0.]}]}
    ia["device_audit"] = {"status": "verified", "native_and_tensor_device_consistency_verified": True,
        "required_unmasked": True, "cuda_visible_devices": None, "requested_device": "cuda:2",
        **{name: "cuda:2" for name in ("environment_device", "physics_sim_view_frontend_device", "robot_native_transform_tensor_device")},
        **{name: 2 for name in ("native_physics_backend_device_ordinal", "torch_current_device", "physics_cudaDevice_setting", "device_index")},
        "physical_mapping_verified": False, "exclusivity_verified": False}
    ia["actuator_audit"] = {}
    for group, kp, kd, effort, speed in (("legs", 500., 10., 80., 15.), ("wheels", 0., 0., 12., 100.)):
        ia["actuator_audit"][group] = {name: [value, value] for name, value in {
            "stiffness": kp, "damping": kd, "effort_limit_nm": effort, "velocity_limit_rad_s": speed,
            "saturation_effort_nm": effort, "velocity_at_effort_limit_rad_s": 2*speed}.items()}
    return mj, record, ia, directory


def write_isaac(directory, report, source):
    directory.mkdir(exist_ok=True)
    raw = analysis.load_arrays(source/"rollout.npz")
    raw["simulation_step_counter"] = np.arange(1397)+100
    raw["packed_controller_states"] = np.zeros((349, 111))
    raw["packed_controller_states"][:, :3] = raw["root_pos"][::4][:-1]
    raw["packed_controller_states"][:, 3:7] = raw["root_quaternion_wxyz"][::4][:-1]
    log = json.loads((source/"solver_log.json").read_text())
    for row in log:
        row.update(raw_cem_action=row["action"].copy(), actual_isaac_objective={"stage_cost": .5, "total_reward": 1.},
            actual_dt_scaled_reward=.02, same_state_dt_reward_abs_error=0., coordinate_conversion_max_velocity_error=0.)
    np.savez_compressed(directory/"rollout.npz", **raw)
    (directory/"solver_log.json").write_text(json.dumps(log))
    report = copy.deepcopy(report)
    report.update(rollout_sha256=analysis.sha256(directory/"rollout.npz"), solver_log_sha256=analysis.sha256(directory/"solver_log.json"))
    (directory/"report.json").write_text(json.dumps(report))
    return report


def test_actual_isaac_metadata_retains_real_mu_binary_device_without_private_paths(tmp_path):
    _, record, report, source = fixture_reports(tmp_path, .1)
    directory = tmp_path/"ia"; report = write_isaac(directory, report, source)
    evidence, trace, public = analysis.audit_isaac_episode(directory, .1, get_reference(), report["mujoco_model_sha256"], record)
    assert evidence["plant_mu"] == .1 and public["actual_plant_mu"] == .1
    assert public["native_identity"]["native_library_sha256"] == analysis.ISAAC_NATIVE_LIBRARY_SHA256
    assert public["device_audit"]["exclusivity_verified"] is False
    assert trace["time_s"].shape == (1397,) and public["solver_diagnostics"]["counts"]["optimization_failed"] == 0
    payload = json.dumps(public)
    assert not any(token in payload for token in ("/private/", "compiler_command", "worker_pid", "/World/"))


@pytest.mark.parametrize("problem", ["plant_mu", "material", "prediction_mu", "source", "binary", "contract_alias",
    "device", "masked", "actuator", "restitution", "counter", "raw_action", "reward", "packed", "report_hash"])
def test_changed_real_material_controller_device_counter_and_raw_logs_fail_closed(tmp_path, problem):
    _, record, report, source = fixture_reports(tmp_path, .1)
    directory = tmp_path/"ia"; report = write_isaac(directory, report, source)
    if problem == "plant_mu": report["plant_mu"] = .7
    elif problem == "material": report["material_audit"]["robot_static_dynamic_range"] = [.7, .7]
    elif problem == "prediction_mu": report["prediction_mu_scenarios"] = [.1]
    elif problem == "source": report["source_sha256"]["/private/archived/scripts/nmpc_shooting.py"] = "0"*64
    elif problem == "binary": report["prediction_backend_metadata"]["workers"]["0.7"]["native_library_sha256"] = "0"*64
    elif problem == "contract_alias": report["source_sha256"]["/private/bridge/config/nominal_actuator_contract.json"] = "0"*64
    elif problem == "device": report["device_audit"]["torch_current_device"] = 1
    elif problem == "masked": report["device_audit"]["cuda_visible_devices"] = "2"
    elif problem == "actuator": report["actuator_audit"]["wheels"]["effort_limit_nm"] = [80., 80.]
    elif problem == "restitution": report["material_audit"]["robot_restitution_range"] = [.1, .1]
    elif problem == "report_hash": report["rollout_sha256"] = "0"*64
    elif problem in ("counter", "packed"):
        raw = analysis.load_arrays(directory/"rollout.npz")
        if problem == "counter": raw["simulation_step_counter"][5:] += 1
        else: raw["packed_controller_states"][5, 0] += .1
        np.savez_compressed(directory/"rollout.npz", **raw)
        report["rollout_sha256"] = analysis.sha256(directory/"rollout.npz")
    else:
        log = json.loads((directory/"solver_log.json").read_text())
        if problem == "raw_action": log[5]["raw_cem_action"][0] = .2
        else: log[5]["same_state_dt_reward_abs_error"] = .2
        (directory/"solver_log.json").write_text(json.dumps(log))
        report["solver_log_sha256"] = analysis.sha256(directory/"solver_log.json")
    (directory/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError): analysis.audit_isaac_episode(directory, .1, get_reference(), report["model_sha256"], record)


def test_early_runtime_failure_is_censored_without_claimed_identity_or_xy_verdict(tmp_path):
    (tmp_path/"report.json").write_text(json.dumps({"status": "failed", "error": "/private/unavailable",
        "kind": "receding_horizon_nmpc", "engine": "Isaac/PhysX", "reference_spec": get_reference()}))
    evidence, trace, public = analysis.audit_isaac_episode(tmp_path, .1, get_reference(), "a"*64, {})
    assert evidence is trace is None and public["censored"] is True
    assert public["actual_controller_identity_available"] is False and public["physical_xy_failure_claimed"] is False
    assert "/private/" not in json.dumps(public)


def test_finite_isaac_solver_failure_stays_a_complete_episode_with_charged_fallback(tmp_path):
    _, record, report, source = fixture_reports(tmp_path, .1)
    directory = tmp_path/"ia"; report = write_isaac(directory, report, source)
    log = json.loads((directory/"solver_log.json").read_text())
    info = log[-1]["solver"]
    info.update(optimization_failed=True, fallback_used=True, finite_candidates=0, failed_candidates=194,
        failed_scenarios=194, failed_batch_prediction_calls=5)
    report["solver_failures"] = 1
    (directory/"solver_log.json").write_text(json.dumps(log))
    report["solver_log_sha256"] = analysis.sha256(directory/"solver_log.json")
    (directory/"report.json").write_text(json.dumps(report))
    _, trace, public = analysis.audit_isaac_episode(directory, .1, get_reference(), report["model_sha256"], record)
    assert trace is not None and public["complete_physical_evidence"] is True
    assert public["solver_diagnostics"]["counts"]["fallback_used"] == 1


def test_partial_parser_accepts_one_genuine_isaac_episode_without_fabricated_replicates(tmp_path):
    assert analysis.parse_runs([".7="+str(tmp_path)]) == {.7: tmp_path}
    with pytest.raises(ValueError, match="Duplicate coefficient"):
        analysis.parse_runs([".7="+str(tmp_path), ".70="+str(tmp_path)])


def make_bundle(directory, reference, record, spec, inputs, policy_public):
    directory.mkdir(); traces = {"reference": reference}; geometries = {"reference": geometry(reference)}
    for mu in analysis.expected_grid():
        trace = canonical(1397, .005)
        trace.update(nonwheel_ground_contact_count=np.zeros(1397), nonwheel_ground_contact_force_norm_n=np.zeros(1397))
        case = analysis.case_name(mu); traces[case], geometries[case] = trace, geometry(trace)
    compact = analysis.public_compact(traces, geometries)
    np.savez_compressed(directory/"development_trace.npz", **compact)
    (directory/"topdown.png").write_bytes(b"synthetic artifact identity token; no source visualization consumed")
    summary = {"kind": "unqualified_mujoco_cem_fixed_friction_development", "status": "development_analysis_only",
        "engine": "MuJoCo", "complete_predeclared_grid": True, "evaluated_mu": analysis.expected_grid(),
        "reference_spec": spec, "input_sha256": inputs, "effective_cem_config": copy.deepcopy(analysis.PILOT_CEM),
        "effective_cem_config_sha256": analysis.PILOT_CONFIG_SHA256, "prediction_mu_scenarios": [.7],
        "analysis_source_sha256": {"analyze_nmpc_friction_development.py": analysis.sha256(ROOT/"scripts/analyze_nmpc_friction_development.py")},
        "runs": {analysis.case_name(mu): copy.deepcopy(record) for mu in analysis.expected_grid()},
        "windows": {"core": {name: analysis.summarize_trace(trace, geometries[name], reference, geometries["reference"], spec["evaluation_core_interval_s"])
            for name, trace in traces.items()}},
        "posture_diagnostics": {name: analysis.posture_diagnostic(trace) for name, trace in traces.items() if name != "reference"},
        "contact_diagnostics": {name: analysis.contact_diagnostic(trace, spec["evaluation_core_interval_s"]) for name, trace in traces.items() if name != "reference"},
        "policy_grids": policy_public,
        "model_loader_audit": {"source_helpers_sha256": {name: "f"*64 for name in analysis.FK_HELPERS}},
        "artifacts": {name: {"file": filename, "sha256": analysis.sha256(directory/filename)} for name, filename in (
            ("compact_trace", "development_trace.npz"), ("topdown", "topdown.png"))},
        **{name: False for name in ("two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
            "eligible_for_controller_ranking", "height_or_flight_failure_gate", "heading_offset_is_mismatch")}}
    (directory/"summary.json").write_text(json.dumps(summary))
    return summary


def arguments(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path, monkeypatch)
    spec = nominal.verify_reference_pair(None, None, None)
    motion = analysis.load_arrays(args.motion_file)
    motion["body_pos_w"] = motion["body_pos_w"].astype(np.float32)
    np.savez_compressed(args.motion_file, **motion)
    spec["motion_npz_sha256"] = analysis.sha256(args.motion_file)
    monkeypatch.setattr(analysis, "verify_reference_pair", lambda *unused: copy.deepcopy(spec))
    inputs = {name: analysis.sha256(getattr(args, name)) for name in ("motion_file", "reference_file", "model")}
    _, record, report, source = fixture_reports(tmp_path, spec=spec, model_sha=inputs["model"])
    args.run = []
    for mu in (.1, .3, .7):
        report = copy.deepcopy(report); report["plant_mu"] = mu
        report["material_audit"].update(expected_mu=mu, robot_static_dynamic_range=[mu, mu])
        report["material_audit"]["ground_bound_materials"][0]["coefficients"] = [mu, mu, 0.]
        directory = tmp_path/analysis.case_name(mu)
        write_isaac(directory, report, source); args.run.append(f"{mu}={directory}")
    reference = analysis.reference_trace(analysis.load_arrays(args.motion_file))
    public = {name: {"algorithm": name, "input_sha256": {"synthetic": "a"*64},
        "training_manifest_sha256": "a"*64, "checkpoint_sha256": "b"*64} for name in ("PPO", "SAC")}
    args.mujoco_bundle = tmp_path/"mj_bundle"
    make_bundle(args.mujoco_bundle, reference, record, spec, inputs, public)
    args.ppo_grid, args.sac_grid = tmp_path/"ppo", tmp_path/"sac"
    def policies(unused, name, unused_spec):
        rows, data = {}, {}
        robot = analysis.load_arrays(args.reference_file)
        for mu in analysis.expected_grid():
            trace = canonical(1397, .005)
            time = trace["time_s"]; core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
            target = np.column_stack([np.interp(time, robot["time_s"], robot["root_target_position_m"][:, j]) for j in (0, 1)])
            error = np.linalg.norm(trace["position"][:, :2]-target, axis=1)
            metric = {"full_core_reference_xy_rmse_m": float(np.sqrt(np.mean(error[core]**2))),
                "full_core_reference_xy_peak_m": float(error[core].max()), "observed_core_samples": int(core.sum()),
                "core_complete": True, "large_reference_xy_screen": False}
            rows[mu] = {"isaac": metric.copy(), "mujoco": metric.copy(), "sim2sim": {"full_core_xy_rmse_m": 0., "full_core_overlap": True}}
            case = analysis.case_name(mu)
            for engine in ("isaac", "mujoco"):
                data[case+"_"+engine+"_t"], data[case+"_"+engine+"_xy"] = trace["time_s"], trace["position"][:, :2]
        return public[name], rows, data
    monkeypatch.setattr(analysis, "load_policy_grid", policies)
    @contextmanager
    def loader(*unused): yield None, {"source_helpers_sha256": {name: "f"*64 for name in analysis.FK_HELPERS}}
    monkeypatch.setattr(analysis, "source_model", loader)
    monkeypatch.setattr(analysis, "WheelKinematics", nominal.WheelKinematics)
    return args


def test_partial_pair_pipeline_has17rows_original_clocks_no_raw_actions_or_invented_policy_features(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    args = arguments(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    value = analysis.run(args)
    assert len(value["rows"]) == 17 and value["evaluated_isaac_mu"] == [.1, .3, .7]
    assert value["complete_two_engine_grid"] is False and value["sim2sim_trajectory_comparison"] is True
    for name in ("two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
        "eligible_for_controller_ranking", "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping"):
        assert value[name] is False
    row = next(row for row in value["rows"] if row["mu"] == .1)
    assert row["NMPC"]["sim2sim"]["xy_rmse_m"] == pytest.approx(0.)
    assert row["PPO"]["isaac"]["measured_com_speed"] is None
    assert row["SAC"]["mujoco"]["measured_contact_slip"] is None
    arrays = analysis.load_arrays(args.output_dir/"development_trace.npz")
    assert np.array_equal(arrays["isaac_mu0p100_time_s"], np.arange(1397)*.005)
    assert "reference_contact_count" not in arrays and "isaac_mu0p100_contact_count" in arrays
    assert not any(name.endswith(("qpos", "qvel", "packed_controller_states", "actions", "held_action")) for name in arrays)
    payload = (args.output_dir/"summary.json").read_text()
    assert "/private/" not in payload and str(tmp_path) not in payload
    for artifact in value["artifacts"].values(): assert analysis.sha256(args.output_dir/artifact["file"]) == artifact["sha256"]
    for path, content in before.items(): assert path.read_bytes() == content


@pytest.mark.parametrize("problem", ["artifact_hash", "replayed_metric", "input", "qualified", "policy_binding"])
def test_pair_rejects_unbound_or_unreplayable_mujoco_and_policy_evidence(tmp_path, monkeypatch, problem):
    args = arguments(tmp_path, monkeypatch)
    path = args.mujoco_bundle/"summary.json"; value = analysis.read_json(path)
    if problem == "artifact_hash": value["artifacts"]["compact_trace"]["sha256"] = "0"*64
    elif problem == "replayed_metric": value["windows"]["core"]["mu0p100"]["reference_xy_rmse_m"] += .2
    elif problem == "input": value["input_sha256"]["model"] = "0"*64
    elif problem == "qualified": value["qualified_freeze_created"] = True
    else: value["policy_grids"]["PPO"]["input_sha256"]["synthetic"] = "0"*64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): analysis.run(args)
    assert not args.output_dir.exists()


def test_parser_and_existing_output_guard(tmp_path):
    args = analysis.parser().parse_args(["--motion", "m", "--reference-file", "r", "--model", "x", "--source-root", "s",
        "--mujoco-bundle", "b", "--ppo-grid", "p", "--sac-grid", "a", "--output-dir", "o", "--run", ".1=d"])
    assert args.motion_file == Path("m")
    with pytest.raises(ValueError, match="Preserve existing"): analysis.run(SimpleNamespace(output_dir=tmp_path))


def add_policy_dynamics(args):
    directory = args.output_dir.parent/"policy_dynamics"; directory.mkdir()
    spec = analysis.verify_reference_pair(None, None, None)
    reference = analysis.reference_trace(analysis.load_arrays(args.motion_file))
    reference_geometry = geometry(reference)
    proof = analysis.reference_numeric_representation_audit(analysis.load_arrays(args.motion_file), analysis.load_arrays(args.reference_file))
    traces, recon, runs, policies = {"reference": reference}, {"reference": reference_geometry}, {}, {}
    raw_roles = ("isaac_trace", "isaac_report", "mujoco_trace", "mujoco_report")
    for name, source in (("PPO", args.ppo_grid), ("SAC", args.sac_grid)):
        public, legacy, _ = analysis.load_policy_grid(source, name, spec)
        policies[name] = {"source_xy_grid": public, "original_physical_archives_used": True,
            "measured_com_speed_and_ground_contact_available": True}
        provenance = {"cases": {analysis.case_name(mu): {role+"_sha256": "e"*64 for role in raw_roles} for mu in analysis.expected_grid()}}
        source.mkdir(); (source/"xy_trace.provenance.json").write_text(json.dumps(provenance))
        for mu in analysis.expected_grid():
            case = analysis.case_name(mu); pair, pair_geometry, core, numeric, posture = {}, {}, {}, {}, {}
            for engine in ("isaac", "mujoco"):
                trace = canonical(1397, .005)  # Optional nonwheel arrays deliberately ABSENT.
                result = geometry(trace); prefix = name+"_"+case+"_"+engine
                traces[prefix], recon[prefix] = trace, result
                pair[engine], pair_geometry[engine] = trace, result
                core[engine] = analysis.summarize_trace(trace, result, reference, reference_geometry, spec["evaluation_core_interval_s"])
                numeric[engine] = analysis.audit_policy_xy_numeric(core[engine], legacy[mu][engine],
                    reference_audit=proof, observed_xy=trace["position"][:, :2])
                posture[engine] = analysis.posture_diagnostic(trace)
            core["sim2sim_xy_gap"] = analysis.pair_xy_gap(pair["isaac"], pair["mujoco"], spec["evaluation_core_interval_s"])
            runs[name+"_"+case] = {"algorithm": name, "mu": mu, "input_sha256": {role: "e"*64 for role in raw_roles},
                "training": {"verified_formal_training": True, "checkpoint_bytes_verified": True,
                    "manifest_sha256": public["training_manifest_sha256"], "checkpoint_sha256": public["checkpoint_sha256"]},
                "windows": {"core": core}, "xy_numeric_representation_replay": numeric, "posture_diagnostics": posture}
    np.savez_compressed(directory/"policy_dynamics_trace.npz", **analysis.public_compact(traces, recon))
    (directory/"topdown.png").write_bytes(b"synthetic identity token; no image decoding")
    summary = {"kind": "existing_policy_friction_dynamics_replay", "reference_spec": spec,
        "evaluation_mu": analysis.expected_grid(), "runs": runs, "policies": policies,
        "reference_numeric_representation_audit": proof,
        "input_sha256": {"motion": analysis.sha256(args.motion_file), "reference_file": analysis.sha256(args.reference_file), "model": analysis.sha256(args.model)},
        "analysis_source_sha256": {name: analysis.sha256(ROOT/"scripts"/name) for name in analysis.POLICY_ANALYSIS_SOURCES},
        "artifacts": {name: {"file": name, "sha256": analysis.sha256(directory/name)} for name in ("policy_dynamics_trace.npz", "topdown.png")},
        **{name: False for name in ("new_policy_evaluation", "new_training", "dynamics_stepping", "learned_inference",
            "height_or_flight_failure_gate", "heading_offset_is_mismatch", "eligible_for_controller_ranking")}}
    (directory/"summary.json").write_text(json.dumps(summary))
    args.policy_dynamics = directory
    return summary


def test_optional_original_policy_dynamics_yields_measured_speed_and_contact_for_all_controllers(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    args = arguments(tmp_path, monkeypatch); add_policy_dynamics(args)
    result = analysis.run(args)
    assert result["policy_feature_scope"] == "original_physical_recording_dynamics"
    assert result["policy_dynamics_bundle"]["original_physical_archives_used"] is True
    for row in result["rows"]:
        assert row["PPO"]["isaac"]["mean_speed_m_s"] == pytest.approx(1.)
        assert row["SAC"]["mujoco"]["ground_contact_lateral_slip_m_s"]["left"] == pytest.approx(.2)
        assert row["SAC"]["mujoco"]["ground_contact_lateral_slip_m_s"]["right"] is None
    compact = analysis.load_arrays(args.output_dir/"development_trace.npz")
    assert "ppo_mu0p100_isaac_wheel_rolling_plane_heading_rad" in compact
    assert "sac_mu0p100_mujoco_com_velocity" in compact
    assert "ppo_mu0p100_isaac_nonwheel_ground_contact_count" not in compact


@pytest.mark.parametrize("problem", ["metric_replay", "raw_identity", "numeric_proof"])
def test_optional_policy_dynamics_rejects_unreplayable_speed_source_or_reference_claims(tmp_path, monkeypatch, problem):
    args = arguments(tmp_path, monkeypatch); summary = add_policy_dynamics(args)
    if problem == "metric_replay": summary["runs"]["PPO_mu0p100"]["windows"]["core"]["isaac"]["mean_speed_m_s"] += .2
    elif problem == "raw_identity": summary["runs"]["PPO_mu0p100"]["input_sha256"]["isaac_trace"] = "0"*64
    else: summary["reference_numeric_representation_audit"]["analytic_metric_change_bound_m"] += .001
    (args.policy_dynamics/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError): analysis.run(args)
    assert not args.output_dir.exists()


def test_reference_panel_draws_kinematic_ticks_and_velocity_without_imputed_contact(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    reference = canonical(349, .02)
    for name in ("contact_count", "contact_force", "contact_slip"): reference.pop(name)
    compact = analysis.public_compact({"reference": reference}, {"reference": geometry(reference)})
    assert "reference_contact_count" not in compact
    trace = canonical(1397, .005)
    for mu in analysis.SELECTED_MU:
        for algorithm in ("ppo", "sac"):
            for engine in ("isaac", "mujoco"):
                prefix = algorithm+"_"+analysis.case_name(mu)+"_"+engine
                compact[prefix+"_time_s"], compact[prefix+"_xy"] = trace["time_s"], trace["position"][:, :2]
    original = Axes.quiver; references = []
    def spy(axis, *args, **kwargs):
        if axis.get_subplotspec().colspan.start == 0: references.append(axis)
        return original(axis, *args, **kwargs)
    monkeypatch.setattr(Axes, "quiver", spy)
    analysis.render_topdown(compact, {"reference_spec": get_reference(), "policy_feature_scope": "published_xy_only"}, tmp_path/"topdown.png")
    assert len(references) == 5 and all(len(axis.lines) > 10 for axis in references)
    assert all("contacts UNMEASURED" in axis.get_title() for axis in references)

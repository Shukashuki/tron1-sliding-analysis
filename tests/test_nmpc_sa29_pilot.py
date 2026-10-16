"""SA29 fixed-controller audit: synthetic evidence, no physics or inference."""
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_nmpc_sa29_pilot as analysis
from analyze_controller_pair import posture_diagnostic
from analyze_nmpc_friction_development import PILOT_CEM
from analyze_nmpc_nominal_development import contact_diagnostic
from analyze_tvlqr_pilot import PUBLIC_SIGNALS, PUBLIC_TRACE_FIELDS
from controller_metrics import heading_signals
from controller_references import get_reference
from nmpc_tracking_objective import HeightSmoothTrackingObjective
from test_nmpc_nominal_development import canonical, geometry, report_fixture


FRAMES, SAMPLES = 258, 1033
TRACE, SOLVER = "2" * 64, "3" * 64
PUBLIC_SHA256 = {
    "summary.json": "df511be87cf615a22c6546459c18a84df15a4e19aa59a8252fd65f944b103480",
    "pilot_trace.npz": "e983b5f308b786d5416dd08cad2b4f1789be57bddca2c9f32e5b17ad78ef1fd7",
    "topdown.png": "2f360b2a68289fcbd2fd8fcad0ffd74f8c43d650f6d26502d87928386c7a638e"}
RAW_IDENTITY_SHA256 = {
    "report": "fe703e078fce33c79670d0bae14a3966b21c03b7a68d5ae90871f55374ec9356",
    "trace": "aecb80f386f43c831664e8e3f41b678c15551922f9c4147d92475573e827a388",
    "solver": "0b101dd0dac648566068575836e9cbea6333fdf6e942c6764dd94b9b4a37c725"}


def declaration_fixture():
    return json.loads((ROOT / "config/nmpc_sa29_nominal_development.json").read_text())


def sa29_report():
    manifest = declaration_fixture()
    spec = get_reference("doa_sa29_scale2")
    report = report_fixture()
    profile = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    objective = HeightSmoothTrackingObjective(profile).contract()
    actions = np.linspace(-.8, .8, FRAMES * 8).reshape(FRAMES, 8).astype(np.float32).astype(float)
    rate = np.sum(np.diff(np.vstack((np.zeros((1, 8)), actions)), axis=0) ** 2, axis=1)
    upper = sum(max(0., w) for w in objective["weights"].values())
    report.update(reference_id=spec["reference_id"], role=spec["role"], stratum=spec["stratum"],
        reference_spec=spec, pooled_success_statistics=False, motion_sha256=spec["motion_npz_sha256"],
        model_sha256=manifest["input_sha256"]["model"], effective_cem_config=copy.deepcopy(PILOT_CEM),
        model_generated_nominal_plan={"enabled": False}, prediction_backend="native", robustness_certificate=False,
        settings=json.loads((ROOT / "config/nmpc_height_smooth_residual.json").read_text()),
        completed_policy_steps=FRAMES, evaluated_policy_steps=FRAMES, total_reference_policy_steps=FRAMES,
        physics_samples=SAMPLES, duration_s=5.16, objective_contract=objective, isaac_result=False,
        contact_threshold_n=5., wheel_lateral_contact_slip_evidence_kind="actual_ground_contact_points",
        action_parameterization={"mode": "warm_start_residual_blocks",
            "baseline_copied_and_fixed_per_solve": True, "learned_baseline": False},
        total_stage_cost=float(np.sum(.02 * (upper - objective["weights"]["action_rate_l2"] * rate))))
    report["source_sha256"] = {"/private/archived/" + name: digest
        for name, digest in manifest["source_sha256"].items()}
    report["source_sha256"]["/private/archived/policy_contract.json"] = manifest["input_sha256"]["actuator_contract"]
    report["native_prediction_backend"] = {
        "production_cem_wired": True, "plant_mu_used_for_prediction": False,
        "prediction_mu_scenarios": [.7], "learned_initialization": False, "terminal_cost": None,
        "workers": {"0.7": {"native_library_sha256": manifest["input_sha256"]["native_library"],
            "compiler_command": ["c++", "/private/source.cpp"], "worker_pid": 987}}}
    return report


def sa29_raw():
    trace = canonical(SAMPLES, .005)
    names = dict(position="root_pos", quaternion="root_quaternion_wxyz", com_velocity="base_com_velocity_w",
        angular_velocity="base_angular_velocity_w", joint_pos="joint_pos", joint_vel="joint_vel",
        wheel_pos="wheel_pos", contact_count="wheel_ground_contact_count", contact_force="wheel_ground_contact_force_w_n",
        contact_slip="wheel_lateral_contact_slip_m_s")
    raw = {target: trace[source].copy() for source, target in names.items()}
    raw_actions = np.linspace(-.8, .8, FRAMES * 8).reshape(FRAMES, 8)
    actions = raw_actions.astype(np.float32).astype(float)
    raw.update(time_s=trace["time_s"], physics_dt_s=np.array(.005), policy_dt_s=np.array(.02),
        action_joint_names=np.array(analysis.JOINT_NAMES), body_names=np.array(analysis.BODY_NAMES),
        actions=actions, held_action=np.vstack((np.zeros((1, 8)), np.repeat(actions, 4, axis=0))),
        compute_s=np.full(FRAMES, .03), reference_frame_for_action=np.r_[0, np.repeat(np.arange(FRAMES), 4)],
        action_was_applied=np.r_[False, np.ones(SAMPLES - 1, dtype=bool)],
        nonwheel_ground_contact_count=np.zeros(SAMPLES), nonwheel_ground_contact_force_norm_n=np.zeros(SAMPLES),
        qpos=np.column_stack((trace["position"], trace["quaternion"], trace["joint_pos"])),
        qvel=np.column_stack((np.ones(SAMPLES), np.zeros((SAMPLES, 5)), trace["joint_vel"])),
        qpos_joint_names=np.array(analysis.JOINT_NAMES), reference_id=np.array("doa_sa29_scale2"),
        reference_role=np.array("separate_supplement"), reference_stratum=np.array("high_sideslip_s_avoidance"),
        reference_evaluation_core_interval_s=np.array([0., 3.14]),
        reference_supplementary_event_windows_s=np.array([[.78, 1.54], [1.88, 3.04]]),
        reference_time_s=np.arange(FRAMES) * .02,
        reference_root_xy_m=np.column_stack((np.arange(FRAMES) * .02, np.zeros(FRAMES))))
    objective = sa29_report()["objective_contract"]
    weights = objective["weights"]
    upper = sum(max(0., weight) for weight in weights.values())
    stages = []
    log = []
    for k in range(FRAMES):
        count = 194 if k == 0 else 195
        counters = dict(candidate_evaluations=count, finite_candidates=count, failed_candidates=0,
            scenario_evaluations=count, failed_scenarios=0, batch_prediction_calls=5 if k == 0 else 6,
            failed_batch_prediction_calls=0, optimization_failed=False, fallback_used=False,
            warm_start_used=k > 0, initial_candidate_generation_failed=False, control_deadline_enforced=False,
            real_time_verified=False, timing_scope="controller_including_predictor_rollouts",
            action_parameterization="warm_start_residual_blocks", residual_baseline_fixed_per_solve=True,
            optimizer="bounded_cem_shooting", horizon_steps=30, control_blocks=10,
            prediction_mu_scenarios=[.7], reference_convention="held_pre_action",
            population_evaluation_backend="predictor_rollout_batch", robustness_certificate=False,
            learned_policy_fallback=False)
        terms = {name: 0. for name in weights}
        terms["action_rate_l2"] = float(np.sum((actions[k] - (actions[k - 1] if k else np.zeros(8))) ** 2))
        weighted = {name: weight * terms[name] for name, weight in weights.items()}
        reward = sum(weighted.values()); stage = .02 * (upper - reward); stages.append(stage)
        log.append({"frame": k, "reference_frame_for_reward": k, "time_s": (k + 1) * .02,
            "action": actions[k].tolist(), "raw_cem_action": raw_actions[k].tolist(), "compute_s": .03,
            "solver": counters, "objective": {"raw_terms": terms, "weighted_terms": weighted,
                "total_reward": reward, "reward_upper_bound": upper,
                "stage_cost": stage, "base_com_vz": 0., "diagnostics": {}, "surrogate_flags": []}})
    raw["stage_cost"] = np.array(stages)
    return raw, log


def header(report):
    return analysis.audit_header(report, get_reference("doa_sa29_scale2"),
        model_sha256=declaration_fixture()["input_sha256"]["model"], trace_sha256=TRACE, solver_sha256=SOLVER)


def mutate(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def test_predeclared_supplement_reuses_exact_primary_controller_but_not_primary_reference():
    manifest = declaration_fixture()
    primary = json.loads((ROOT / "config/nmpc_friction_development.json").read_text())
    assert analysis.audit_predeclaration() == manifest
    assert manifest["source_sha256"] == primary["source_sha256"] and len(manifest["source_sha256"]) == 21
    assert manifest["reference_id"] != primary["reference_id"]
    assert manifest["intended_local_episodes"] == 1 and manifest["pooled_success_statistics"] is False
    assert manifest["evaluation"]["frames"] == FRAMES and manifest["evaluation"]["physical_samples"] == SAMPLES
    assert manifest["evaluation"]["duration_s"] == 5.16
    assert manifest["evaluation"]["events_are_descriptive_not_gates"] is True


@pytest.mark.parametrize("path,value", [
    (("reference_id",), "drift_drl_traj4"), (("pooled_success_statistics",), True),
    (("intended_local_episodes",), 2), (("plant_mu",), .3), (("prediction_mu_scenarios",), [.3]),
    (("explicit_overrides", "iterations"), 4), (("evaluation", "frames"), 349),
    (("no_retuning",), False), (("no_remote_upload_or_execution",), False),
    (("source_sha256", "scripts/nmpc_shooting.py"), "0" * 64),
])
def test_predeclaration_mutations_cannot_silently_change_the_one_episode_experiment(path, value):
    manifest = declaration_fixture(); mutate(manifest, path, value)
    with pytest.raises(ValueError):
        analysis.audit_predeclaration(manifest)


def test_header_preserves_original_report_and_removes_private_runtime_metadata():
    report = sa29_report(); before = copy.deepcopy(report)
    result = header(report)
    assert report == before
    text = json.dumps(result)
    assert "/private/" not in text and "compiler_command" not in text and "worker_pid" not in text


@pytest.mark.parametrize("path,value", [
    (("reference_id",), "drift_drl_traj4"), (("status",), "running"), (("status",), "failed"),
    (("completed_policy_steps",), 257), (("total_reference_policy_steps",), 349),
    (("duration_s",), 6.98), (("rollout_sha256",), "0" * 64),
    (("motion_sha256",), "0" * 64), (("model_sha256",), "0" * 64),
    (("plant_mu",), .3), (("prediction_mu_scenarios",), [.3]),
    (("effective_cem_config", "iterations"), 4), (("learned_policy_used",), True),
    (("model_generated_nominal_plan", "enabled"), True), (("pose_height_failure_gate",), True),
    (("real_time_verified",), True), (("robustness_certificate",), True),
    (("action_parameterization", "learned_baseline"), True), (("contact_threshold_n",), 1.),
    (("settings", "nmpc", "horizon_steps"), 30),
    (("source_sha256", "/private/archived/scripts/nmpc_shooting.py"), "0" * 64),
    (("native_prediction_backend", "workers", "0.7", "native_library_sha256"), "0" * 64),
])
def test_changed_identity_partial_episode_or_unsupported_claim_fails_closed(path, value):
    report = sa29_report(); mutate(report, path, value)
    with pytest.raises(ValueError):
        header(report)


def test_native_trace_is_the_real_258_decision_1033_sample_supplement_clock():
    raw, log = sa29_raw()
    trace = analysis.audit_trace(raw, log, get_reference("doa_sa29_scale2"))
    assert np.array_equal(trace["time_s"], np.arange(SAMPLES) * .005)
    assert trace["time_s"][-1] == 5.16
    assert trace["position"].shape == (SAMPLES, 3)
    assert trace["nonwheel_ground_contact_count"].shape == (SAMPLES,)


@pytest.mark.parametrize("problem", ["clock_gap", "samples", "phase", "held", "action_precision", "int_applied",
    "solver_frame", "solver_action", "qpos", "qvel_joint", "bad_joint_order", "nonwheel"])
def test_native_clock_action_phase_and_physical_state_mutations_fail_closed(problem):
    raw, log = sa29_raw()
    if problem == "clock_gap": raw["time_s"][100:] += .005
    elif problem == "samples": raw["time_s"] = raw["time_s"][:-1]
    elif problem == "phase": raw["reference_frame_for_action"][100] += 1
    elif problem == "held": raw["held_action"][100, 0] += .1
    elif problem == "action_precision": raw["actions"][0, 0] += 1e-12
    elif problem == "int_applied": raw["action_was_applied"] = raw["action_was_applied"].astype(int)
    elif problem == "solver_frame": log[30]["reference_frame_for_reward"] += 1
    elif problem == "solver_action": log[30]["action"][0] += .01
    elif problem == "qpos": raw["qpos"][30, 0] += .1
    elif problem == "qvel_joint": raw["qvel"][30, 6] += .1
    elif problem == "bad_joint_order": raw["qpos_joint_names"][-1] = raw["qpos_joint_names"][0]
    else: raw["nonwheel_ground_contact_count"][3] = -.1
    with pytest.raises((ValueError, KeyError)):
        analysis.audit_trace(raw, log, get_reference("doa_sa29_scale2"))


def test_original_solver_budget_and_wall_clock_do_not_become_real_time_or_delay_claims():
    raw, log = sa29_raw()
    result = analysis.solver_diagnostic(sa29_report(), raw, log)
    assert result["decision_samples"] == FRAMES
    assert result["counts"]["candidate_evaluations"] == result["counts"]["scenario_evaluations"] == 50309
    assert result["counts"]["scenario_evaluations"] * 30 * 4 == 6037080
    assert result["counts"]["warm_start_used"] == 257
    assert result["timing"]["sample_count"] == FRAMES
    assert result["timing"]["fraction_above_20ms"] == 1.
    assert result["timing"]["real_time_verified"] is False
    assert result["timing"]["actuation_delay_injected"] is False


@pytest.mark.parametrize("problem", ["quantiles", "timing", "failure_total", "counter", "raw_command", "term", "stage", "upper_bound"])
def test_original_solver_reward_and_command_evidence_is_replayed_not_trusted(problem):
    report = sa29_report(); raw, log = sa29_raw()
    if problem == "quantiles": report["compute_ms_median_p95_max"][1] += 1.
    elif problem == "timing": log[10]["compute_s"] += .001
    elif problem == "failure_total": report["solver_failures"] = 1
    elif problem == "counter": log[10]["solver"]["finite_candidates"] = 1
    elif problem == "raw_command": log[10]["raw_cem_action"][0] += .1
    elif problem == "term": log[10]["objective"]["weighted_terms"]["drift_speed"] = 1.
    elif problem == "stage": raw["stage_cost"][10] += 1.
    else: log[10]["objective"]["reward_upper_bound"] += 1.
    with pytest.raises(ValueError):
        analysis.solver_diagnostic(report, raw, log)


def test_core_and_two_events_keep_geometric_offset_separate_from_measured_contact_slip():
    reference = canonical(FRAMES, .02)
    raw, log = sa29_raw()
    native = analysis.audit_trace(raw, log, get_reference("doa_sa29_scale2"))
    # A 60 degree root heading offset and arbitrarily large contact slip are
    # descriptive features; matching XY still remains matching XY.
    native["quaternion"][:] = [np.cos(np.pi / 6), 0., 0., np.sin(np.pi / 6)]
    native["contact_slip"][:, 0] = 3.
    traces = {"reference": reference, "NMPC": native}
    recon = {name: geometry(trace) for name, trace in traces.items()}
    result = analysis.summarize_windows(traces, recon, get_reference("doa_sa29_scale2"))
    windows = [(result["core"], 629)] + [(event["metrics"], count)
        for event, count in zip(result["supplementary_events"], (153, 233))]
    assert [item["interval_s"] for item in result["supplementary_events"]] == [[.78, 1.54], [1.88, 3.04]]
    assert all(item["descriptive_not_gate"] is True for item in result["supplementary_events"])
    for metrics, count in windows:
        value = metrics["NMPC"]
        assert value["samples_in_window"] == count and value["coverage_complete"] is True
        assert value["reference_xy_rmse_m"] == pytest.approx(0., abs=1e-14)
        assert value["large_xy_exploratory_screen"] is False
        assert value["heading_offset_is_mismatch"] is value["height_or_flight_failure_gate"] is False
        assert value["heading_geometry"]["root_directed_beta"]["mean_absolute_deg"] == pytest.approx(60.)
        assert value["ground_contact_lateral_slip_m_s"] == {"left": 3., "right": None}
        assert metrics["reference"]["ground_contact_lateral_slip_m_s"] is None


def test_force_threshold_and_no_contact_zeros_never_invent_rolling_or_reference_contact():
    raw, log = sa29_raw()
    native = analysis.audit_trace(raw, log, get_reference("doa_sa29_scale2"))
    native["contact_force"][:, 0, 2] = 5.  # Eligibility is strictly greater than 5 N.
    native["contact_slip"][:] = 0.
    traces = {"reference": canonical(FRAMES, .02), "NMPC": native}
    recon = {name: geometry(trace) for name, trace in traces.items()}
    metrics = analysis.summarize_windows(traces, recon, get_reference("doa_sa29_scale2"))["core"]
    assert metrics["reference"]["ground_contact_heading_geometry"] is None
    assert metrics["NMPC"]["ground_contact_heading_geometry"]["contact_fraction_left_right"] == [0., 0.]
    assert metrics["NMPC"]["ground_contact_lateral_slip_m_s"] == {"left": None, "right": None}
    assert metrics["NMPC"]["heading_geometry"]["per_wheel"]["left"]["root_course_to_wheel_gamma"]["valid_samples"] == 629
    assert metrics["NMPC"]["ground_contact_heading_geometry"]["per_wheel"]["left"]["root_course_to_wheel_gamma"]["valid_samples"] == 0


def test_partial_core_coverage_is_censored_without_a_fake_xy_pass_or_failure():
    traces = {"reference": canonical(FRAMES, .02), "NMPC": canonical(600, .005)}
    recon = {name: geometry(trace) for name, trace in traces.items()}
    value = analysis.summarize_windows(traces, recon, get_reference("doa_sa29_scale2"))["core"]["NMPC"]
    assert value["coverage_complete"] is False
    assert value["reference_xy_rmse_m"] is value["reference_xy_peak_m"] is value["large_xy_exploratory_screen"] is None


def refresh_action_scores(report, raw, log):
    weights = report["objective_contract"]["weights"]
    upper = sum(weight for weight in weights.values() if weight > 0)
    previous = np.zeros(8)
    for k, row in enumerate(log):
        action = raw["actions"][k]
        row["action"] = action.tolist()
        terms = row["objective"]["raw_terms"]
        terms["action_rate_l2"] = float(np.sum((action - previous) ** 2))
        weighted = {name: weight * terms[name] for name, weight in weights.items()}
        reward = sum(weighted.values())
        row["objective"].update(weighted_terms=weighted, total_reward=reward, stage_cost=.02 * (upper - reward))
        raw["stage_cost"][k] = row["objective"]["stage_cost"]
        previous = action
    raw["held_action"] = np.vstack((np.zeros((1, 8)), np.repeat(raw["actions"], 4, axis=0)))
    report["total_stage_cost"] = float(raw["stage_cost"].sum())


def test_finite_previous_action_fallback_is_retained_as_an_outcome_not_an_xy_gate():
    report = sa29_report(); raw, log = sa29_raw()
    raw["actions"][10] = raw["actions"][9]
    log[10]["raw_cem_action"] = raw["actions"][9].tolist()
    log[10]["solver"].update(optimization_failed=True, fallback_used=True)
    report["solver_failures"] = 1
    refresh_action_scores(report, raw, log)
    result = analysis.solver_diagnostic(report, raw, log)
    assert result["counts"]["optimization_failed"] == result["counts"]["fallback_used"] == 1
    assert result["finite_fallbacks_retained_as_descriptive_outcomes"] is True
    log[10]["solver"]["fallback_used"] = False
    with pytest.raises(ValueError, match="fallback"):
        analysis.solver_diagnostic(report, raw, log)


def test_absolute_root_height_is_not_a_tracking_or_failure_constraint():
    raw, log = sa29_raw()
    raw["root_pos"][:, 2] += 100.
    raw["qpos"][:, 2] += 100.
    analysis.audit_trace(raw, log, get_reference("doa_sa29_scale2"))
    result = analysis.solver_diagnostic(sa29_report(), raw, log)
    assert result["common_score_arithmetic_verified"] is True


def test_cli_requires_explicit_supplement_assets_and_preserves_existing_outputs(tmp_path):
    args = analysis.parser().parse_args(["--motion", "m", "--reference-file", "r", "--model", "x",
        "--source-root", "s", "--run-dir", "d", "--output-dir", "o"])
    assert args.motion_file == Path("m") and args.run_dir == Path("d")
    with pytest.raises(ValueError, match="Preserve existing"):
        analysis.run(SimpleNamespace(output_dir=tmp_path, run_dir=tmp_path))


def test_run_reads_a_real_json_list_solver_file_before_the_header_audit(tmp_path, monkeypatch):
    """Regression: the shared object-only read_json cannot decode solver lists.

    Actual parser, file hashes, NPZ reader, report JSON reader and list JSON
    decoder run here. Stop at the header boundary, before any asset/model/FK
    work; those require the original locked assets rather than fake identities.
    """
    run_dir = tmp_path / "recording"; run_dir.mkdir()
    (run_dir / "report.json").write_text(json.dumps({"kind": "synthetic-loader-sentinel"}))
    (run_dir / "solver_log.json").write_text(json.dumps([{"frame": 0, "solver": {}}]))
    np.savez(run_dir / "rollout.npz", time_s=np.array([0., .005]))
    model = tmp_path / "model.xml"; model.write_text("synthetic token; never loaded")
    output = tmp_path / "new-output"
    args = analysis.parser().parse_args(["--motion", str(tmp_path / "motion.npz"),
        "--reference-file", str(tmp_path / "reference.npz"), "--model", str(model),
        "--source-root", str(tmp_path / "source"), "--run-dir", str(run_dir), "--output-dir", str(output)])
    monkeypatch.setattr(analysis, "verify_reference_pair", lambda *unused: get_reference("doa_sa29_scale2"))
    class HeaderBoundaryReached(Exception):
        pass
    def boundary(report, *unused, **keywords):
        assert report == {"kind": "synthetic-loader-sentinel"}
        assert keywords["solver_sha256"] == analysis.sha256(run_dir / "solver_log.json")
        raise HeaderBoundaryReached
    monkeypatch.setattr(analysis, "audit_header", boundary)
    original = {p: p.read_bytes() for p in run_dir.iterdir()}
    with pytest.raises(HeaderBoundaryReached):
        analysis.run(args)
    assert not output.exists()
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.fixture(scope="module")
def published_sa29():
    """Only the portable three-file bundle; no raw state/model/weights."""
    directory = ROOT / "results/nmpc_sa29_nominal_development_20261016_v1"
    blobs = {path.name: path.read_bytes() for path in directory.iterdir()}
    with np.load(io.BytesIO(blobs["pilot_trace.npz"]), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return {"summary": json.loads(blobs["summary.json"]), "arrays": arrays,
        "hashes": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}}


def public_trace(arrays, label):
    trace = {name: arrays[label + "_" + name] for name in PUBLIC_TRACE_FIELDS if label + "_" + name in arrays}
    samples, interval = (FRAMES, .02) if label == "reference" else (SAMPLES, .005)
    # Preserve the two original producers' clocks exactly: reference arange/fps
    # and the plant's accumulated 5 ms simulation-time additions.
    expected_clock = np.arange(samples) / 50. if label == "reference" else np.r_[0., np.cumsum(np.full(samples - 1, .005))]
    np.testing.assert_array_equal(trace["time_s"], expected_clock)
    assert trace["time_s"].shape == (samples,) and float(trace["physics_dt_s"]) == interval
    signals = heading_signals(trace["quaternion"], trace["com_velocity"],
        arrays[label + "_wheel_quaternion_wxyz"], arrays[label + "_wheel_center_velocity_w"])
    for name in PUBLIC_SIGNALS:
        np.testing.assert_array_equal(signals[name], arrays[label + "_" + name])
    np.testing.assert_array_equal(signals["speed_m_s"],
        np.linalg.norm(np.asarray(trace["com_velocity"], dtype=float)[:, :2], axis=1))
    return trace, {"signals": signals}


def test_final_three_file_publication_locks_actual_reference_source_budget_and_original_raw_identities(published_sa29):
    summary = published_sa29["summary"]; manifest = declaration_fixture()
    assert published_sa29["hashes"] == PUBLIC_SHA256
    assert summary["kind"] == "separate_sa29_mujoco_fixed_nmpc_nominal_development"
    assert summary["status"] == "development_analysis_only" and summary["engine"] == "MuJoCo"
    assert summary["reference_spec"] == get_reference("doa_sa29_scale2")
    assert summary["predeclaration_sha256"] == analysis.PREDECLARATION_SHA256
    assert summary["intended_local_episodes"] == summary["completed_physical_episodes"] == 1
    assert summary["frames"] == FRAMES and summary["physical_samples"] == SAMPLES and summary["duration_s"] == 5.16
    assert summary["actual_plant_mu"] == .7 and summary["prediction_mu_scenarios"] == [.7]
    assert summary["effective_cem_config"] == PILOT_CEM
    assert summary["effective_cem_config_sha256"] == manifest["effective_cem_configuration_sha256"]
    assert summary["input_sha256"] == {name: manifest["input_sha256"][name] for name in ("motion", "robot_reference", "model")}
    assert summary["reference_catalog_sha256"] == manifest["source_sha256"]["config/controller_references.json"]
    run = summary["run"]
    assert run["input_sha256"] == RAW_IDENTITY_SHA256
    assert run["runtime_source_sha256"] == manifest["source_sha256"]
    provenance = {item["source_name"]: item["sha256"] for item in run["reported_source_provenance"]["reported_source_hashes"]}
    assert provenance == {**{Path(name).name: digest for name, digest in manifest["source_sha256"].items()},
        "policy_contract.json": manifest["input_sha256"]["actuator_contract"]}
    assert run["native_identity"]["native_library_sha256"] == manifest["input_sha256"]["native_library"]
    assert run["plant_material_readback"]["compiled_sliding_friction_range"] == [.7, .7]
    assert run["complete_physical_evidence"] is True and run["censored"] is False and run["no_retuning"] is True
    assert run["reported_settings"] == json.loads((ROOT / manifest["settings"]).read_text())
    counts = run["solver_diagnostics"]["counts"]
    assert counts["candidate_evaluations"] == counts["finite_candidates"] == counts["scenario_evaluations"] == 50309
    assert counts["batch_prediction_calls"] == 1547 and counts["warm_start_used"] == 257
    assert all(counts[name] == 0 for name in ("failed_candidates", "failed_scenarios", "failed_batch_prediction_calls",
        "optimization_failed", "fallback_used", "initial_candidate_generation_failed"))
    assert summary["simulation_budget"] == {"actual_plant_physics_substeps": 1032,
        "requested_prediction_substeps": 6037080, "failed_prediction_requests_prove_completed_steps": False}
    for artifact in summary["artifacts"].values():
        assert artifact["sha256"] == PUBLIC_SHA256[artifact["file"]]
    for name, digest in summary["analysis_source_sha256"].items():
        assert hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() == digest


def test_published_vectors_exactly_replay_core_events_heading_contact_posture_and_com_speed(published_sa29):
    summary, arrays = published_sa29["summary"], published_sa29["arrays"]
    traces, recon = {}, {}
    for label in ("reference", "NMPC"):
        traces[label], recon[label] = public_trace(arrays, label)
    assert analysis.summarize_windows(traces, recon, get_reference("doa_sa29_scale2")) == summary["windows"]
    assert posture_diagnostic(traces["NMPC"]) == summary["posture_diagnostics"]["NMPC"]
    assert contact_diagnostic(traces["NMPC"], [0., 3.14]) == summary["contact_diagnostics"]["core"]
    for item in summary["contact_diagnostics"]["supplementary_events"]:
        assert item["descriptive_not_gate"] is True
        assert contact_diagnostic(traces["NMPC"], item["interval_s"]) == item["diagnostic"]
    windows = [(summary["windows"]["core"], [0., 3.14], 629)] + [
        (item["metrics"], item["interval_s"], n) for item, n in
        zip(summary["windows"]["supplementary_events"], (153, 233))]
    trace, reference = traces["NMPC"], traces["reference"]
    for metrics, interval, count in windows:
        chosen = (trace["time_s"] >= interval[0] - 1e-9) & (trace["time_s"] <= interval[1] + 1e-9)
        assert int(chosen.sum()) == metrics["NMPC"]["samples_in_window"] == count
        target = np.column_stack([np.interp(trace["time_s"][chosen], reference["time_s"],
            reference["position"][:, axis]) for axis in (0, 1)])
        error = np.linalg.norm(trace["position"][chosen, :2] - target, axis=1)
        assert float(np.sqrt(np.mean(error ** 2))) == metrics["NMPC"]["reference_xy_rmse_m"]
        assert float(error.max()) == metrics["NMPC"]["reference_xy_peak_m"]
        gate = (trace["contact_count"][chosen] > 0) & (trace["contact_force"][chosen, :, 2] > 5.)
        assert gate.mean(axis=0).tolist() == metrics["NMPC"]["ground_contact_heading_geometry"]["contact_fraction_left_right"]
        for side, name in enumerate(("left", "right")):
            slip = trace["contact_slip"][chosen, side][gate[:, side]]
            expected = float(np.sqrt(np.mean(slip ** 2))) if len(slip) else None
            assert expected == metrics["NMPC"]["ground_contact_lateral_slip_m_s"][name]
        assert metrics["reference"]["ground_contact_heading_geometry"] is None
        assert metrics["reference"]["ground_contact_lateral_slip_m_s"] is None


def test_publication_projection_contains_no_raw_state_commands_private_paths_or_credentials(published_sa29):
    summary, arrays = published_sa29["summary"], published_sa29["arrays"]
    allowed = set(PUBLIC_TRACE_FIELDS) | set(PUBLIC_SIGNALS) | {"wheel_quaternion_wxyz", "wheel_center_velocity_w"}
    for name, value in arrays.items():
        if name not in {"schema_version", "wheel_order"}:
            prefix, separator, field = name.partition("_")
            assert separator and prefix in {"reference", "NMPC"} and field in allowed
        assert value.dtype.kind != "O"
    assert "reference_contact_count" not in arrays and "NMPC_contact_count" in arrays
    private = re.compile(r"-----BEGIN .*PRIVATE KEY|\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b")
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {"qpos", "qvel", "actions", "raw_cem_action", "held_action", "password",
                    "private_key", "policy_weights", "compiler_command", "worker_pid"}
                inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
        elif isinstance(value, str):
            assert not value.startswith(("/", "~", "file:", "C:\\", "D:\\"))
            assert not any(path in value for path in ("/home/", "/data/", "/root/", "/tmp/"))
            assert private.search(value) is None
    inspect(summary)
    for name in ("pooled_success_statistics", "primary_task_replicate", "sim2sim_comparison", "isaac_evidence_available",
        "supplementary_policy_evidence_available", "friction_grid_complete", "two_engine_nominal_qualified",
        "qualified_freeze_created", "qualified_driver_executed", "frozen_benchmark", "eligible_for_controller_ranking",
        "robustness_certificate", "real_time_verified", "controlled_execution_success_claimed",
        "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "inverse_kinematics",
        "retargeting", "learned_inference", "actuation_delay_injected", "controller_retuned"):
        assert summary[name] is False
    assert summary["events_are_descriptive_not_success_gates"] is True

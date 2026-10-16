"""Asset-free MPPI artifact audits; no dynamics, models, optimization or SSH."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_mppi_pilot as analysis
import analyze_nmpc_nominal_development as nominal
from controller_references import get_reference
from mppi_shooting import MPPIConfig, MPPIController
from test_nmpc_nominal_development import (
    MODEL, TRACE, SOLVER, fixture_arguments, raw_fixture, report_fixture, geometry,
)


def shared_sources(report):
    for index, name in enumerate(analysis.SHARED_SOURCES):
        existing = [key for key in report["source_sha256"] if key.endswith("/"+name)]
        if not existing:
            report["source_sha256"]["/private/archived/"+name] = f"{index+10:064x}"
    return report


def mppi_report():
    report = shared_sources(report_fixture())
    config = MPPIConfig(); controller = MPPIController(SimpleNamespace(), config,
        action_low=-np.ones(8), action_high=np.ones(8))
    parameters = json.loads(json.dumps(asdict(config)))
    report.pop("effective_cem_config")
    report.update(kind=analysis.MPPI_KIND, effective_mppi_config=parameters,
        physics_samples=1397, action_parameterization=controller.parameterization_contract(),
        algorithm_contract=controller.algorithm_contract(), model_generated_nominal_plan={"enabled": False},
        settings={"schema_version": 1, "task": "car_drift_height_smooth", "policy_dt_s": .02,
                  "physics_dt_s": .005, "prediction_mu_scenarios": [.7], "mppi": copy.deepcopy(parameters)},
        prediction_backend="native", common_15_term_physical_rollout_score_unchanged=True,
        total_stage_cost=174.5)
    report.update({name: False for name in ("height_or_flight_failure_gate", "heading_offset_is_mismatch",
        "adaptive_model_selection_used", "sim2sim_comparison", "isaac_result", "optimization_equals_pure_task_cost_cem",
        "robustness_certificate")})
    for name in ("run_mppi_tracking.py", "mppi_shooting.py"):
        report["source_sha256"]["/private/archived/scripts/"+name] = "a"*64
    report["native_prediction_backend"] = {
        "kind": "optional_native_batch_exact_objective_adapter", "production_cem_wired": False,
        "production_mppi_wired": True, "runtime_mppi_population_path": "callable predictor.rollout_batch",
        "objective": {"production_cem_wired": False, "production_mppi_wired": True},
        "prediction_mu_scenarios": [.7], "plant_mu_used_for_prediction": False,
        "learned_initialization": False, "terminal_cost": None,
        "workers": {"0.7": {"source_sha256": analysis.source_digest(report, "native/nmpc_batch.cpp"),
            "mujoco_header_sha256": "a"*64, "mujoco_library_sha256": "b"*64,
            "native_library_sha256": "c"*64, "model_snapshot_sha256": "d"*64,
            "physics_dt_s": .005, "control_dt_s": .02, "worker_pid": 999,
            "compiler_command": ["c++", "/private/compiler/source.cpp"]}}}
    return report


def audited(report):
    return analysis.audit_mppi_report(report, get_reference(), model_sha256=MODEL,
        trace_sha256=TRACE, solver_sha256=SOLVER)


def mppi_raw():
    raw, log = raw_fixture()
    raw.update(qpos=np.column_stack((raw["root_pos"], raw["root_quaternion_wxyz"], raw["joint_pos"])),
        qvel=np.column_stack((np.ones(1397), np.zeros((1397, 5)), raw["joint_vel"])),
        qpos_joint_names=np.array(analysis.JOINT_NAMES), stage_cost=np.full(349, .5))
    for k, row in enumerate(log):
        info = row["solver"]
        initial_count = 2 if k == 0 else 3
        count = initial_count+3*64
        info.update(optimizer="fixed_covariance_latent_residual_mppi", reference_convention="held_pre_action",
            prediction_mu_scenarios=[.7], population_evaluation_backend="predictor_rollout_batch",
            action_parameterization="unbounded_gaussian_latent_residual_blocks", residual_baseline_fixed_per_solve=True,
            implicit_gaussian_kl_prior=True, learned_policy_fallback=False, robustness_certificate=False,
            horizon_steps=30, control_blocks=10, samples=63, iterations_requested=3, iterations_attempted=3, iterations_completed=3,
            temperature=1., noise_std=[.04]*6+[.35]*2, compute_time_s=.02,
            initial_candidate_evaluations=[{"model_cost": float(index+1)} for index in range(initial_count)],
            initial_model_cost=1., selected_model_cost=.5, selected_scenario_costs=[.5], predicted_cost_nonincreasing=True,
            candidate_evaluations=count, finite_candidates=count, scenario_evaluations=count,
            batch_prediction_calls=initial_count+6, warm_start_used=k > 0,
            mppi_iterations=[{"iteration": index, "raw_cost_min": .3, "raw_cost_max": .8,
                "importance_correction_min": 0., "importance_correction_max": 0., "log_weight_max": -.3,
                "shifted_weight_normalizer": 63., "ess": 63., "max_weight": 1/63,
                "finite_population_candidates": 63, "update_l2": .1,
                "updated_mean_model_cost": .5, "population_decode_clip_fraction": .1,
                "updated_mean_decode_clip_fraction": .05} for index in range(3)])
        row.update(raw_mppi_action=[0.]*8, objective={"stage_cost": .5})
    return raw, log


def test_report_is_genuine_mppi_without_cem_fields_and_private_native_metadata():
    report = mppi_report(); before = copy.deepcopy(report); value = audited(report)
    assert value["controller_kind"] == analysis.MPPI_KIND
    assert "effective_cem_config" not in value and "elite_count" not in json.dumps(value)
    assert value["mppi_algorithm_contract"]["covariance_fitting"] is False
    assert value["native_prediction_backend"]["production_mppi_wired"] is True
    assert not any(token in json.dumps(value) for token in ("/private/", "compiler_command", "worker_pid"))
    assert report == before
    with pytest.raises(ValueError, match="completed full-body"):
        analysis.audit_cem_report(report, get_reference(), model_sha256=MODEL, trace_sha256=TRACE, solver_sha256=SOLVER)


@pytest.mark.parametrize("field,value", [("kind", analysis.CEM_KIND), ("engine", "Isaac/PhysX"),
    ("status", "running"), ("plant_mu", .3), ("prediction_mu_scenarios", [.3, .7]),
    ("rollout_sha256", "0"*64), ("solver_log_sha256", "0"*64), ("model_sha256", "0"*64),
    ("motion_sha256", "0"*64), ("physics_samples", 1396), ("completed_policy_steps", 348),
    ("prediction_receives_plant_mu", True), ("root_state_prescribed_after_reset", True),
    ("pose_height_failure_gate", True), ("heading_offset_is_mismatch", True),
    ("action_replay", True), ("real_time_verified", True), ("optimization_equals_pure_task_cost_cem", True),
    ("common_15_term_physical_rollout_score_unchanged", False)])
def test_report_rejects_identity_hash_incomplete_or_false_claims(field, value):
    report = mppi_report(); report[field] = value
    with pytest.raises(ValueError): audited(report)


@pytest.mark.parametrize("problem", ["temperature", "boolean_temperature", "samples", "seed", "noise", "fallback", "settings",
    "covariance", "elite", "correction", "clipped_latent", "CEM_metadata", "objective_metadata",
    "worker_clock", "worker_source", "private_config", "model_plan", "invented_CEM"])
def test_sampler_contract_and_native_wiring_are_explicit_and_fail_closed(problem):
    report = mppi_report()
    if problem == "temperature": report["effective_mppi_config"]["temperature"] = .1
    elif problem == "boolean_temperature": report["effective_mppi_config"]["temperature"] = True
    elif problem == "samples": report["effective_mppi_config"]["samples"] = 64
    elif problem == "seed": report["effective_mppi_config"]["seed"] = True
    elif problem == "noise": report["effective_mppi_config"]["noise_std"] = [.1]*8
    elif problem == "fallback": report["effective_mppi_config"]["fallback_action"] = "neutral"
    elif problem == "settings": report["settings"]["mppi"]["iterations"] = 8
    elif problem == "covariance": report["algorithm_contract"]["covariance_fitting"] = True
    elif problem == "elite": report["algorithm_contract"]["elite_selection"] = True
    elif problem == "correction": report["algorithm_contract"]["log_weights_up_to_constant"] = "-task_cost"
    elif problem == "clipped_latent": report["action_parameterization"]["latent_clip"] = True
    elif problem == "CEM_metadata": report["native_prediction_backend"]["production_cem_wired"] = True
    elif problem == "objective_metadata": report["native_prediction_backend"]["objective"]["production_mppi_wired"] = False
    elif problem == "worker_clock": report["native_prediction_backend"]["workers"]["0.7"]["physics_dt_s"] = .01
    elif problem == "worker_source": report["native_prediction_backend"]["workers"]["0.7"]["source_sha256"] = "f"*64
    elif problem == "private_config": report["algorithm_contract"]["private_state"] = "/private/state.bin"
    elif problem == "model_plan": report["model_generated_nominal_plan"] = {"enabled": True}
    else: report["effective_cem_config"] = {"elite_count": 8}
    with pytest.raises(ValueError): audited(report)


def test_weighted_mean_not_best_sample_is_selected_and_every_rollout_is_charged():
    report = mppi_report(); raw, log = mppi_raw()
    value = analysis.mppi_diagnostic(report, raw, log)
    assert value["counts"]["candidate_evaluations"] == 68054
    assert value["counts"]["batch_prediction_calls"] == 3140
    assert value["counts"]["warm_start_used"] == 348
    assert value["counts"]["optimization_failed"] == 0
    assert value["weighted_update_diagnostics"]["ess"]["samples"] == 1047
    assert value["weighted_update_diagnostics"]["updated_mean_model_cost"]["minimum"] == .5
    assert value["weighted_update_diagnostics"]["raw_cost_min"]["minimum"] == .3
    assert value["timing"]["actuation_delay_injected"] is False


@pytest.mark.parametrize("problem", ["timing", "raw_action", "counter", "missing_prior", "elite",
    "order", "normalizer", "ESS", "selected_cost", "model", "nonincreasing", "initial_cost",
    "missing_iteration", "finite_population", "inverted_cost", "clip_fraction", "NaN", "unrecorded_fallback", "warm_history"])
def test_log_weight_update_budget_and_command_consistency_fail_closed(problem):
    report = mppi_report(); raw, log = mppi_raw(); row = log[5]; info = row["solver"]
    if problem == "timing": row["compute_s"] += .01
    elif problem == "raw_action": row["raw_mppi_action"][0] = .1
    elif problem == "counter": info["candidate_evaluations"] -= 1
    elif problem == "missing_prior": info["implicit_gaussian_kl_prior"] = False
    elif problem == "elite": info["elite_count"] = 8
    elif problem == "order": info["mppi_iterations"][1]["iteration"] = 2
    elif problem == "normalizer": info["mppi_iterations"][1]["shifted_weight_normalizer"] = 10.
    elif problem == "ESS": info["mppi_iterations"][1]["ess"] = 64.
    elif problem == "selected_cost": info["selected_model_cost"] = .3
    elif problem == "model": info["prediction_mu_scenarios"] = [.3]
    elif problem == "nonincreasing": info["predicted_cost_nonincreasing"] = False
    elif problem == "initial_cost": info["initial_model_cost"] = .8
    elif problem == "missing_iteration": info["mppi_iterations"].pop()
    elif problem == "finite_population": info["mppi_iterations"][1]["finite_population_candidates"] = 62
    elif problem == "inverted_cost": info["mppi_iterations"][1]["raw_cost_max"] = .2
    elif problem == "clip_fraction": info["mppi_iterations"][1]["updated_mean_decode_clip_fraction"] = 1.2
    elif problem == "NaN": info["mppi_iterations"][1]["ess"] = float("nan")
    elif problem == "warm_history": info["warm_start_used"] = False
    else: info["fallback_used"] = True
    with pytest.raises(ValueError): analysis.mppi_diagnostic(report, raw, log)


def test_failed_population_retains_previous_command_and_null_not_invented_diagnostics():
    report = mppi_report(); raw, log = mppi_raw(); info = log[-1]["solver"]
    info.update(optimization_failed=True, fallback_used=True, optimization_failure_reason="all_population_predictions_failed",
        selected_model_cost=None, selected_scenario_costs=None, candidate_evaluations=66, finite_candidates=3,
        failed_candidates=63, scenario_evaluations=66, failed_scenarios=63,
        batch_prediction_calls=4, failed_batch_prediction_calls=1, iterations_attempted=1, iterations_completed=0,
        mppi_iterations=[{"iteration": 0, "finite_population_candidates": 0,
            "population_decode_clip_fraction": .1, "updated_mean_model_cost": None,
            "failure": "all_population_predictions_failed"}])
    report["solver_failures"] = 1
    result = analysis.mppi_diagnostic(report, raw, log)
    assert result["counts"]["fallback_used"] == 1 and result["all_population_failed_iterations"] == 1
    assert result["weighted_update_diagnostics"]["ess"]["null_or_unavailable_samples"] == 1
    assert result["weighted_update_diagnostics"]["ess"]["samples"] == 348*3


def test_updated_mean_may_be_worse_than_initial_without_best_sample_substitution():
    report = mppi_report(); raw, log = mppi_raw(); info = log[5]["solver"]
    info.update(selected_model_cost=1.5, selected_scenario_costs=[1.5], predicted_cost_nonincreasing=False)
    info["mppi_iterations"][-1]["updated_mean_model_cost"] = 1.5
    assert analysis.mppi_diagnostic(report, raw, log)["weighted_update_diagnostics"]["updated_mean_model_cost"]["median_p95_max"][-1] == 1.5


def test_partial_finite_population_is_not_automatically_a_failed_controller():
    report = mppi_report(); raw, log = mppi_raw(); info = log[5]["solver"]
    item = info["mppi_iterations"][1]
    item.update(finite_population_candidates=62, shifted_weight_normalizer=62., ess=62., max_weight=1/62)
    info.update(finite_candidates=194, failed_candidates=1, failed_scenarios=1)
    value = analysis.mppi_diagnostic(report, raw, log)
    assert value["counts"]["failed_candidates"] == 1
    assert value["counts"]["optimization_failed"] == 0


def test_failed_updated_mean_uses_previous_action_and_charges_failed_re_evaluation():
    report = mppi_report(); raw, log = mppi_raw(); info = log[-1]["solver"]
    info["mppi_iterations"] = info["mppi_iterations"][:1]
    info["mppi_iterations"][0]["updated_mean_model_cost"] = None
    info.update(optimization_failed=True, fallback_used=True, optimization_failure_reason="updated_mean_prediction_failed",
        selected_model_cost=None, selected_scenario_costs=None, candidate_evaluations=67, finite_candidates=66,
        failed_candidates=1, scenario_evaluations=67, failed_scenarios=1,
        batch_prediction_calls=5, failed_batch_prediction_calls=1, iterations_attempted=1, iterations_completed=0)
    report["solver_failures"] = 1
    value = analysis.mppi_diagnostic(report, raw, log)
    assert value["counts"]["candidate_evaluations"] == 68054-195+67
    assert value["weighted_update_diagnostics"]["updated_mean_model_cost"]["null_or_unavailable_samples"] == 1


def arguments(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path, monkeypatch)
    for name in ("verify_reference_pair", "source_model", "WheelKinematics"):
        monkeypatch.setattr(analysis, name, getattr(nominal, name))
    args.run = args.run[:2]
    for index, value in enumerate(args.run):
        directory = Path(value.partition("=")[2])
        raw, log = mppi_raw()
        report = mppi_report() if index else shared_sources(analysis.read_json(directory/"report.json"))
        if not index:
            _, cem_log = raw_fixture()
            for row in cem_log: row["objective"] = {"stage_cost": .5}
            log = cem_log
            report["total_stage_cost"] = 174.5
        np.savez(directory/"rollout.npz", **raw)
        (directory/"solver_log.json").write_text(json.dumps(log))
        report.update(model_sha256=analysis.sha256(args.model), motion_sha256=analysis.sha256(args.motion_file),
            rollout_sha256=analysis.sha256(directory/"rollout.npz"), solver_log_sha256=analysis.sha256(directory/"solver_log.json"))
        (directory/"report.json").write_text(json.dumps(report))
    return args


def test_end_to_end_offline_native_topdown_portable_geometry_and_bound_inputs(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    args = arguments(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    value = analysis.run(args)
    assert value["engine"] == "MuJoCo" and value["status"] == "development_analysis_only"
    assert value["common_15_term_evaluation_objective"] is True
    for name in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking",
                 "matched_optimization_objective", "controlled_execution_success_claimed", "robustness_certificate",
                 "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "learned_inference"):
        assert value[name] is False
    assert value["runs"]["H30"]["simulation_budget_diagnostic"]["planning_predicted_physics_substeps"] == 8166480
    assert value["runs"]["H30"]["controller_kind"] == analysis.MPPI_KIND
    payload = (args.output_dir/"summary.json").read_text()
    assert "/private/" not in payload and str(tmp_path) not in payload and "worker_pid" not in payload
    arrays = analysis.load_arrays(args.output_dir/"development_trace.npz")
    assert np.array_equal(arrays["H30_time_s"], np.arange(1397)*.005)
    assert "reference_contact_count" not in arrays and "H30_contact_count" in arrays
    assert not any(key.endswith(("actions", "held_action", "qpos", "qvel", "raw_mppi_action")) for key in arrays)
    assert not any(value.dtype.hasobject for value in arrays.values())
    for artifact in value["artifacts"].values():
        assert analysis.sha256(args.output_dir/artifact["file"]) == artifact["sha256"]
    assert (args.output_dir/"topdown.png").stat().st_size > 1000
    for path, content in before.items(): assert path.read_bytes() == content


@pytest.mark.parametrize("problem", ["hash", "source", "objective", "actuator", "stage_cost", "physical_state", "phase", "clock", "no_mppi", "no_cem"])
def test_pipeline_fails_before_publishing_changed_or_incomplete_evidence(tmp_path, monkeypatch, problem):
    args = arguments(tmp_path, monkeypatch)
    directory = tmp_path/"H30"; report = analysis.read_json(directory/"report.json")
    if problem == "hash": report["rollout_sha256"] = "0"*64
    elif problem == "source": report["source_sha256"]["/private/archived/scripts/nmpc_tracking_objective.py"] = "0"*64
    elif problem == "objective": report["objective_contract"]["weights"]["drift_speed"] += .1
    elif problem == "actuator": report["source_sha256"]["/private/archived/policy_contract.json"] = "0"*64
    elif problem == "stage_cost": report["total_stage_cost"] += .1
    elif problem == "no_mppi": args.run = args.run[:1]
    elif problem == "no_cem": args.run = args.run[1:]
    else:
        raw = analysis.load_arrays(directory/"rollout.npz")
        if problem == "physical_state": raw["qpos"][4, 0] += .2
        elif problem == "phase": raw["reference_frame_for_action"][5] += 1
        else: raw["time_s"][5:] += .005
        np.savez(directory/"rollout.npz", **raw)
        report["rollout_sha256"] = analysis.sha256(directory/"rollout.npz")
    (directory/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError): analysis.run(args)
    assert not args.output_dir.exists()


def test_parser_preserves_existing_outputs_and_rejects_duplicate_reserved_labels(tmp_path):
    parsed = analysis.parser().parse_args(["--motion", "m", "--reference-file", "r", "--model", "x",
        "--source-root", "s", "--output-dir", "o", "--run", "cem=a", "--run", "mppi=b"])
    assert parsed.motion_file == Path("m")
    with pytest.raises(ValueError, match="Preserve existing"):
        analysis.run(SimpleNamespace(output_dir=tmp_path))
    with pytest.raises(ValueError, match="Duplicate/reserved"):
        analysis.parse_runs(["reference="+str(tmp_path), "mppi="+str(tmp_path)])
    with pytest.raises(ValueError, match="distinct"):
        analysis.parse_runs(["cem="+str(tmp_path), "mppi="+str(tmp_path)])

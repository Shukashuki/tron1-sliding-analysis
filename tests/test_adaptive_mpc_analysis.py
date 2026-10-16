"""Asset-free offline adaptive-MPC audit tests; no models, physics or remote IO."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_adaptive_mpc as analysis
from adaptive_contact_model import AdaptiveContactConfig
from adaptive_mpc_runtime import adaptation_summary
from controller_references import get_reference
from test_nmpc_nominal_development import (
    MODEL, TRACE, SOLVER, report_fixture, raw_fixture,
    fixture_arguments as nominal_fixture_arguments,
)
import analyze_nmpc_nominal_development as nominal


def adaptive_fixture():
    report = report_fixture()
    report.update(kind=analysis.ADAPTIVE_KIND, prediction_mu_scenarios=None,
                  prediction_mu_model_bank=analysis.MODEL_BANK.copy())
    report["adaptation_contract"] = {
        "kind": "causal_multiple_model_adaptive_prediction_selection",
        "selector": AdaptiveContactConfig().contract(), "model_bank": analysis.MODEL_BANK.copy(),
        "prediction_models_per_solve": 1, "identification_model_substeps": 4,
        "identification_interval_s": .02,
        **{key: False for key in ("selected_mu_is_physical_friction_estimate", "plant_mu_used_for_prediction",
                                  "future_plant_state_used", "reference_used_for_identification", "robustness_certificate")}}
    raw, solver = raw_fixture()
    for k, row in enumerate(solver):
        row["solver"].update(prediction_mu_scenarios=[.7], compute_time_s=.02,
            adaptive_contact={"status": "initial_prior" if k == 0 else "low_spread", "updated": False,
                "selected_mu": .7, "previous_mu": .7, "scores": None,
                "prediction_spread_rms": None if k == 0 else 0.,
                "window_steps": min(k, 10), "informative_steps": 0,
                "model_bank": [.3, .7, 1.6], "identification_prediction_calls": 0 if k == 0 else 3,
                "identification_physics_substeps": 0 if k == 0 else 12,
                "estimation_compute_s": .001, "source_interval_reference_index": k-1 if k else None,
                "transition_duration_s": .02 if k else None, "prediction_models_per_solve": 1,
                "selected_mu_is_physical_friction_estimate": False})
    report["adaptation_summary"] = adaptation_summary(solver)
    return report, raw, solver


def audited(report):
    return analysis.audit_report(report, get_reference(), model_sha256=MODEL,
                                 trace_sha256=TRACE, solver_sha256=SOLVER)


def test_custom_adaptive_audit_preserves_actual_plant_and_bank_without_relaxing_nominal_audit():
    report, _, _ = adaptive_fixture()
    report["plant_mu"] = .3
    report["plant_material_readback"]["compiled_sliding_friction_range"] = [.3, .3]
    before = copy.deepcopy(report)
    value = audited(report)
    assert value["actual_plant_mu"] == .3
    assert value["prediction_mu_model_bank"] == [.3, .7, 1.6]
    assert value["prediction_mu_scenarios"] is None
    assert report == before
    with pytest.raises(ValueError, match="completed full-body"):
        analysis.audit_nominal_common_schema(report, get_reference(), model_sha256=MODEL,
                                             trace_sha256=TRACE, solver_sha256=SOLVER)


def test_fixed_singleton_baseline_can_have_validation_friction_distinct_from_model():
    report = report_fixture()
    report["plant_mu"] = .3
    report["plant_material_readback"]["compiled_sliding_friction_range"] = [.3, .3]
    value = audited(report)
    assert value["actual_plant_mu"] == .3 and value["prediction_mu_scenarios"] == [.7]
    assert value["adaptation_contract"] is None
    _, solver = raw_fixture()
    for row in solver: row["solver"]["prediction_mu_scenarios"] = [.7]
    assert analysis.adaptation_diagnostic(report, solver, {}) == {"enabled": False, "prediction_mu_scenarios": [.7]}
    solver[2]["solver"]["prediction_mu_scenarios"] = [.3, .7]
    with pytest.raises(ValueError, match="Fixed baseline"):
        analysis.adaptation_diagnostic(report, solver, {})


@pytest.mark.parametrize("field,value", [
    ("kind", "robust_certified_mpc"), ("engine", "Isaac/PhysX"), ("status", "running"),
    ("plant_mu", -.1), ("plant_mu", True), ("prediction_mu_scenarios", [.3, .7, 1.6]),
    ("prediction_mu_model_bank", [.3, .7]), ("rollout_sha256", "0"*64),
    ("solver_log_sha256", "0"*64), ("prediction_receives_plant_mu", True),
    ("root_state_prescribed_after_reset", True), ("external_root_forces_used", True),
    ("completed_policy_steps", 348), ("physics_dt_s", .01),
])
def test_report_rejects_variant_oracle_incomplete_and_hash_changes(field, value):
    report, _, _ = adaptive_fixture()
    report[field] = value
    with pytest.raises(ValueError): audited(report)


@pytest.mark.parametrize("field", ["selected_mu_is_physical_friction_estimate", "plant_mu_used_for_prediction",
                                 "future_plant_state_used", "reference_used_for_identification", "robustness_certificate"])
def test_causal_contract_requires_explicit_nonoracle_noncertificate_flags(field):
    report, _, _ = adaptive_fixture()
    report["adaptation_contract"][field] = True
    with pytest.raises(ValueError, match="unsafe adaptation"):
        audited(report)


def test_adaptation_totals_and_per_decision_model_selection_include_identification_cost():
    report, raw, solver = adaptive_fixture()
    value = analysis.adaptation_diagnostic(report, solver, raw)
    assert value["status_counts"] == {"initial_prior": 1, "low_spread": 348}
    assert value["model_switches"] == 0
    assert value["identification_prediction_calls"] == 1044
    assert value["identification_predicted_physics_substeps"] == 4176
    assert value["identification_compute_total_s"] == pytest.approx(.349)
    assert value["selected_model_per_decision"] == [.7]*349
    assert value["selection_time_s"][1] == .02
    assert value["physical_friction_estimate_claimed"] is False


@pytest.mark.parametrize("problem", ["future_index", "wrong_duration", "multi_solve", "different_selection", "missing",
    "bad_switch", "wrong_previous", "wrong_substeps", "wrong_calls", "negative_compute", "excluded_latency",
    "bad_window", "invented_scores", "bad_low_spread", "bad_summary", "first_transition"])
def test_adaptation_fails_closed_on_noncausal_inconsistent_and_hidden_overhead(problem):
    report, raw, solver = adaptive_fixture()
    row = solver[2]; observation = row["solver"]["adaptive_contact"]
    if problem == "future_index": observation["source_interval_reference_index"] = 2
    elif problem == "wrong_duration": observation["transition_duration_s"] = .04
    elif problem == "multi_solve": row["solver"]["prediction_mu_scenarios"] = [.3, .7, 1.6]
    elif problem == "different_selection": row["solver"]["prediction_mu_scenarios"] = [.3]
    elif problem == "missing": del row["solver"]["adaptive_contact"]
    elif problem == "bad_switch": observation["updated"] = True
    elif problem == "wrong_previous": observation["previous_mu"] = 1.6
    elif problem == "wrong_substeps": observation["identification_physics_substeps"] = 4
    elif problem == "wrong_calls": observation["identification_prediction_calls"] = 0; observation["identification_physics_substeps"] = 0
    elif problem == "negative_compute": observation["estimation_compute_s"] = -.1
    elif problem == "excluded_latency": observation["estimation_compute_s"] = .02
    elif problem == "bad_window": observation["window_steps"] = 10
    elif problem == "invented_scores": observation["scores"] = [0., 0., 0.]
    elif problem == "bad_low_spread": observation["prediction_spread_rms"] = 2.
    elif problem == "bad_summary": report["adaptation_summary"]["identification_prediction_calls"] += 1
    else: solver[0]["solver"]["adaptive_contact"]["source_interval_reference_index"] = 0
    with pytest.raises(ValueError): analysis.adaptation_diagnostic(report, solver, raw)


@pytest.mark.parametrize("contact_status", ["no_contact", "nonwheel_contact"])
def test_identification_skip_is_bound_to_actual_preceding_contact_samples(contact_status):
    report, raw, solver = adaptive_fixture()
    if contact_status == "no_contact": raw["wheel_ground_contact_count"][5:9] = 0
    else: raw["nonwheel_ground_contact_count"][5:9] = 1
    observation = solver[2]["solver"]["adaptive_contact"]
    observation.update(status=contact_status, identification_prediction_calls=0, identification_physics_substeps=0)
    report["adaptation_summary"] = adaptation_summary(solver)
    result = analysis.adaptation_diagnostic(report, solver, raw)
    assert result["status_counts"][contact_status] == 1
    raw["wheel_ground_contact_count"][5:9] = 1
    raw["nonwheel_ground_contact_count"][5:9] = 0
    with pytest.raises(ValueError, match="contact samples"):
        analysis.adaptation_diagnostic(report, solver, raw)


def test_report_summary_is_allowlisted_and_never_copies_private_paths_or_raw_plans():
    report, raw, solver = adaptive_fixture()
    solver[2]["solver"]["adaptive_contact"]["private_path"] = "/private/candidate.bin"
    solver[2]["solver"]["raw_plan"] = [[1.]]
    value = analysis.adaptation_diagnostic(report, solver, raw)
    payload = json.dumps(value)
    assert "private_path" not in payload and "/private/" not in payload and "raw_plan" not in payload
    report["adaptation_contract"]["private_path"] = "/private/state.npz"
    with pytest.raises(ValueError, match="private/absolute"):
        audited(report)


def test_parser_accepts_single_episode_and_preserves_existing_outputs(tmp_path):
    run = tmp_path/"case"; run.mkdir()
    assert analysis.parse_runs(["adaptive="+str(run)]) == {"adaptive": run}
    with pytest.raises(ValueError, match="distinct"):
        analysis.parse_runs(["a="+str(run), "b="+str(run)])
    with pytest.raises(ValueError, match="reserved"):
        analysis.parse_runs(["reference="+str(run)])
    with pytest.raises(ValueError, match="Preserve existing"):
        analysis.run(SimpleNamespace(output_dir=run))
    parsed = analysis.parser().parse_args(["--motion", "m", "--reference-file", "r", "--model", "x",
                                          "--source-root", "s", "--output-dir", "o", "--run", "a=d"])
    assert parsed.motion_file == Path("m")


def analysis_arguments(tmp_path, monkeypatch):
    args = nominal_fixture_arguments(tmp_path, monkeypatch)
    for name in ("verify_reference_pair", "source_model", "WheelKinematics"):
        monkeypatch.setattr(analysis, name, getattr(nominal, name))
    for index, value in enumerate(args.run):
        directory = Path(value.partition("=")[2])
        report = analysis.read_json(directory/"report.json")
        solver = json.loads((directory/"solver_log.json").read_text())
        if index == 0:
            adaptive, _, adapted_solver = adaptive_fixture()
            for key in ("kind", "prediction_mu_scenarios", "prediction_mu_model_bank", "adaptation_contract", "adaptation_summary"):
                report[key] = adaptive[key]
            solver = adapted_solver
        else:
            for row in solver: row["solver"]["prediction_mu_scenarios"] = [.7]
        if index == 1:
            report["plant_mu"] = .3
            report["plant_material_readback"]["compiled_sliding_friction_range"] = [.3, .3]
        (directory/"solver_log.json").write_text(json.dumps(solver))
        report["solver_log_sha256"] = analysis.sha256(directory/"solver_log.json")
        (directory/"report.json").write_text(json.dumps(report))
    return args


def test_end_to_end_asset_free_topdown_native_clock_and_portable_aggregate(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    args = analysis_arguments(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = analysis.run(args)
    assert result["engine"] == "MuJoCo" and result["status"] == "development_analysis_only"
    for key in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking",
                "continuous_uncertainty_robustness_certified", "controlled_execution_success_claimed",
                "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping",
                "identification_bank_is_worst_case_scenario_set"):
        assert result[key] is False
    assert result["runs"]["baseline"]["adaptation_diagnostics"]["identification_prediction_calls"] == 1044
    assert result["runs"]["H30"]["actual_plant_mu"] == .3
    assert result["runs"]["H30"]["adaptation_diagnostics"]["enabled"] is False
    assert result["windows"]["core"]["baseline"]["large_xy_exploratory_screen"] is False
    text = (args.output_dir/"summary.json").read_text()
    assert str(tmp_path) not in text and "/private/" not in text
    arrays = analysis.load_arrays(args.output_dir/"development_trace.npz")
    assert "reference_contact_count" not in arrays
    assert np.array_equal(arrays["baseline_time_s"], np.arange(1397)*.005)
    assert not any(value.dtype.hasobject for value in arrays.values())
    assert result["artifacts"]["compact_trace_sha256"] == analysis.sha256(args.output_dir/"development_trace.npz")
    assert result["artifacts"]["topdown_sha256"] == analysis.sha256(args.output_dir/"topdown.png")
    assert (args.output_dir/"topdown.png").stat().st_size > 1000
    for path, value in before.items(): assert path.read_bytes() == value


@pytest.mark.parametrize("problem", ["hash", "objective", "actuator", "clock", "phase", "no_adaptive"])
def test_pipeline_rejects_inconsistent_inputs_before_publishing(tmp_path, monkeypatch, problem):
    args = analysis_arguments(tmp_path, monkeypatch)
    directory = tmp_path/"H30"
    report = analysis.read_json(directory/"report.json")
    if problem == "hash": report["rollout_sha256"] = "0"*64
    elif problem == "objective": report["objective_contract"]["weights"]["drift_speed"] += .1
    elif problem == "actuator": report["source_sha256"]["/private/archived/policy_contract.json"] = "b"*64
    elif problem == "no_adaptive": args.run = args.run[1:]
    else:
        raw = analysis.load_arrays(directory/"rollout.npz")
        if problem == "clock": raw["time_s"][20:] += .005
        else: raw["reference_frame_for_action"][5] = 2
        np.savez_compressed(directory/"rollout.npz", **raw)
        report["rollout_sha256"] = analysis.sha256(directory/"rollout.npz")
    (directory/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError): analysis.run(args)
    assert not args.output_dir.exists()

"""Read-only public-record and optional archived-artifact consistency checks.

No MuJoCo import, native build, rollout, training, or simulator launch occurs.
Optional local raw archives are not redistributed and are skipped if absent.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/2026-10-15-native-backend-parity.json"
RAW = ROOT / "outputs/controller_backend_parity_20261015_v1"


def record():
    return json.loads(PUBLIC.read_text())


def raw_json(relative):
    path = RAW / relative
    if not path.is_file():
        pytest.skip(f"Optional local archived evidence unavailable: {relative}")
    return json.loads(path.read_text())


def test_public_record_is_portable_and_does_not_claim_turn_success_or_real_time():
    data = record(); text = PUBLIC.read_text()
    for private in ("/home/", "/data/", "zengxiong", "14.103.233.29", "117.50.226.213"):
        assert private not in text
    assert data["status"] == "passed"
    assert data["coverage"]["policy_steps_each_run"] == 10
    assert data["coverage"]["physics_samples_each_run"] == 41
    assert data["coverage"]["duration_s"] == pytest.approx(.2)
    assert not data["coverage"]["full_turn_evaluated"]
    assert not data["coverage"]["core_coverage_complete"]
    assert data["coverage"]["core_xy_rmse_m"] is None
    assert not data["coverage"]["isaac_app_launched"] and not data["coverage"]["gpu_used"]
    assert data["cem"]["horizon_steps"] == 3 and data["cem"]["candidates"] == 4
    assert data["plant_mu"] == .18 and data["prediction_mu_scenarios"] == [.7]
    assert data["timing"]["timing_sample_count_each_run"] == 10
    assert data["timing"]["native_fraction_above_20ms"] == .2
    assert not data["timing"]["real_time_verified"]
    assert not data["timing"]["deadline_enforced"]
    assert not data["timing"]["injected_actuation_delay"]
    assert "startup excluded" in data["timing"]["scope"]
    assert "IPC" in data["timing"]["scope"]
    assert "WSL2" in data["runtime"]["platform"]
    repro = data["reproduction"]
    assert repro["working_directory"] == "<repository-root>"
    assert "python -c" in repro["explicit_build"] and "build_native_library" in repro["explicit_build"]
    assert "python scripts/run_nmpc_tracking.py" in repro["common_command"]
    assert "/path/to/" in repro["common_command"]
    assert "--prediction-backend native" in repro["native_suffix"]
    assert "--native-library" in repro["native_suffix"]
    assert repro["preserve_previous_outputs"]
    assert all(len(sha) == 64 and int(sha, 16) >= 0 for sha in data["source_sha256"].values())


def test_public_record_changes_only_private_reproduction_fields_from_raw_manifest():
    archived = copy.deepcopy(raw_json("comparison.json"))
    archived["reproduction"] = record()["reproduction"]
    assert record() == archived  # All counts, fields, metrics, timings and hashes retained.


@pytest.mark.parametrize("relative", list(record()["artifacts_sha256"]))
def test_available_raw_artifacts_have_published_sha256(relative):
    path = RAW / relative
    if not path.is_file():
        pytest.skip(f"Optional local archived artifact unavailable: {relative}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record()["artifacts_sha256"][relative]


def test_archived_native_report_supports_all_published_source_hashes():
    native = raw_json("native/report.json")
    python = raw_json("python/report.json")
    for relative, sha in record()["source_sha256"].items():
        native_matches = [value for name, value in native["source_sha256"].items() if name.endswith("/" + relative)]
        assert native_matches == [sha], relative
        common = [value for name, value in python["source_sha256"].items() if name.endswith("/" + relative)]
        if common:
            assert common == [sha], relative
    # Archived source hashes are intentionally not asserted against later code revisions.


def test_independent_saved_trace_parity_and_observed_metrics_without_rollouts():
    paths = [RAW / name / "rollout.npz" for name in ("python", "native")]
    if not all(path.is_file() for path in paths):
        pytest.skip("Optional local archived physical traces unavailable")
    data = record()
    with np.load(paths[0], allow_pickle=False) as python, np.load(paths[1], allow_pickle=False) as native:
        assert set(python.files) == set(native.files)
        fields = set(python.files) - {"compute_s"}
        assert len(fields) == data["parity"]["compared_nontiming_trace_fields"] == 37
        for field in fields:
            np.testing.assert_array_equal(python[field], native[field], err_msg=field)
        np.testing.assert_allclose(python["time_s"], np.arange(41)*.005, atol=1e-15, rtol=0)
        observed = data["observed_window_only"]
        assert np.sqrt(np.mean(python["xy_error_m"]**2)) == pytest.approx(observed["xy_rmse_m_each_run"])
        assert python["speed_m_s"].mean() == pytest.approx(observed["base_com_xy_speed_mean_m_s_each_run"])
        assert python["stage_cost"].sum() == pytest.approx(observed["total_stage_cost_each_run"])
        assert np.abs(python["applied_joint_torque_nm"]).max() == pytest.approx(observed["maximum_abs_torque_nm"])
        assert np.linalg.norm(python["wheel_ground_contact_force_w_n"], axis=-1).max() == pytest.approx(observed["maximum_wheel_ground_force_n"])
        assert int((python["wheel_ground_contact_count"] > 0).sum()) == observed["positive_wheel_ground_contact_samples"]
        for name, archive in (("python", python), ("native", native)):
            quantiles = 1000*np.quantile(archive["compute_s"], [.5, .95, 1.])
            np.testing.assert_allclose(quantiles, data["timing"][name + "_compute_ms_median_p95_max"], rtol=1e-12, atol=1e-10)
            assert np.mean(archive["compute_s"] > .02) == data["timing"][name + "_fraction_above_20ms"]


def test_independent_solver_objective_cost_and_count_parity_without_rollouts():
    python, native = raw_json("python/solver_log.json"), raw_json("native/solver_log.json")
    data = record()
    assert len(python) == len(native) == data["coverage"]["policy_steps_each_run"]
    max_cost_error = 0.
    counts = ("candidate_evaluations", "finite_candidates", "failed_candidates", "scenario_evaluations",
              "failed_scenarios", "failed_scenarios_by_mu", "optimization_failed", "fallback_used",
              "warm_start_used", "initial_candidate_generation_failed")
    costs = ("initial_model_cost", "selected_model_cost", "selected_scenario_costs", "best_cost_by_iteration")
    for before, after in zip(python, native):
        assert before["raw_cem_action"] == after["raw_cem_action"]
        assert before["action"] == after["action"]
        assert before["objective"] == after["objective"]
        assert len(before["objective"]["raw_terms"]) == len(before["objective"]["weighted_terms"]) == 15
        for name in counts:
            assert before["solver"][name] == after["solver"][name], name
        for name in costs:
            error = np.max(np.abs(np.asarray(before["solver"][name])-np.asarray(after["solver"][name])))
            max_cost_error = max(max_cost_error, float(error))
        assert after["solver"]["population_evaluation_backend"] == "predictor_rollout_batch"
    assert max_cost_error == pytest.approx(data["parity"]["solver_selected_initial_scenario_and_iteration_costs_max_abs_error"], rel=1e-12, abs=1e-28)
    for solver_log in (python, native):
        assert sum(row["solver"]["candidate_evaluations"] for row in solver_log) == data["parity"]["candidate_evaluations_each_run"]
        assert sum(row["solver"]["optimization_failed"] for row in solver_log) == data["parity"]["solver_failures_each_run"]
    assert sum(row["solver"]["batch_prediction_calls"] for row in native) == data["parity"]["native_batch_calls"]
    assert sum(row["solver"]["failed_batch_prediction_calls"] for row in native) == data["parity"]["native_failed_batch_calls"]

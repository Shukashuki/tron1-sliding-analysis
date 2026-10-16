"""Verify the completed PPO-only publication without simulators or remote access."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/controller_primary_ppo_height_smooth_seed42_v1"
REPORT = ROOT / "results/2026-10-15-primary-ppo-height-smooth-grid.md"
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_friction_grid import summarize_new_grid
from controller_references import get_reference


def data():
    return tuple(json.loads((PUBLIC / name).read_text()) for name in
                 ("summary.json", "xy_trace.provenance.json", "training_metadata.json"))


def test_completed_ppo_training_budget_and_height_objective_are_not_historical_smoke():
    summary, provenance, training = data()
    assert training["algorithm"] == "PPO" and training["training_seed"] == 42
    assert training["training_status"] == "training_completed"
    assert training["trained_policy_claimed"] and not training["smoke_only"]
    budget = training["training_budget"]
    assert budget["num_envs"]*budget["iterations_requested"]*budget["num_steps_per_env"] == 12288000
    assert budget["environment_transitions_completed"] == budget["environment_transitions_requested"] == 12288000
    assert budget["policy_steps_completed"] == budget["policy_steps_requested"] == 24000
    assert budget["learning_iteration_index_completed"] == 999
    assert budget["setup_smoke_control_steps"] == 0
    height = training["height_regularization"]
    assert height["mode"] == "vertical_acceleration_only"
    assert height["position_and_linear_velocity_tracking_axes"] == ["x", "y"]
    assert height["weight"] == -.1 and height["acceleration_scale_m_s2"] == 5.
    assert not height["absolute_height_reward"] and not height["height_termination"]
    assert not height["reference_vertical_velocity_reward"]
    assert training["reference_id"] == summary["reference_id"] == "drift_drl_traj4"
    assert training["reference_role"] == "primary"
    assert training["task_stratum"] == summary["task_stratum"] == "large_angle_turn"
    assert training["checkpoint_sha256"] == provenance["checkpoint_sha256"] == "6ec411efeaea37fc61389285978b377aa7c5be80e326ce7f18581efc9acdf27b"
    assert training["training_manifest_sha256"] == provenance["training_manifest_sha256"]
    assert not training["pooled_success_statistics"] and not summary["pooled_success_statistics"]


def test_training_dr_is_distinct_from_fixed_evaluation_materials():
    _, _, training = data()
    dr = training["training_domain_randomization"]
    assert dr["friction_static"] == [.3, 1.6] and dr["friction_dynamic"] == [.3, 1.2]
    assert dr["restitution"] == [0., .5] and dr["material_buckets"] == 64
    assert dr["joint_default_position_offset_rad"] == [-.01, .01]
    assert dr["push_interval_s"] == [1., 3.] and dr["ground_friction"] == 1.
    assert dr["base_com_offset_m"] == {"x": [-.025, .025], "y": [-.05, .05], "z": [-.05, .05]}
    evaluation = training["evaluation"]
    assert not evaluation["domain_randomization_enabled"] and not evaluation["observation_noise_enabled"]
    assert evaluation["episodes_per_coefficient_per_engine"] == 1
    assert evaluation["engine_runs"] == evaluation["completed_full_reference_engine_runs"] == 34
    assert evaluation["full_horizon_diagnostic"] and evaluation["pose_predicates_recorded_not_terminating"]
    assert not evaluation["height_or_flight_failure_gate"] and not evaluation["heading_offset_failure_gate"]
    assert not evaluation["statistical_success_rate_claimed"] and not evaluation["controller_ranking_claimed"]


def test_all_17_unique_cases_are_complete_with_verified_material_and_reference_identity():
    summary, provenance, training = data()
    expected = [.01, .03, .05, .10, .15, .20, .30, .45, .60, .70, .80, 1., 1.2, 1.4, 1.6, 1.8, 2.]
    assert training["evaluation"]["coefficients_static_dynamic_equal"] == expected
    assert len(summary["rows"]) == len(provenance["cases"]) == 17
    assert sorted(row["evaluation_mu_static_dynamic_equal"] for row in summary["rows"]) == expected
    assert len({row["case"] for row in summary["rows"]}) == 17
    assert summary["provenance"] == provenance
    assert provenance["reference_spec"] == get_reference("drift_drl_traj4")
    assert provenance["motion_sha256"] == provenance["reference_spec"]["motion_npz_sha256"] == training["motion_sha256"]
    assert provenance["reference_sha256"] == provenance["reference_spec"]["robot_reference_npz_sha256"] == training["robot_reference_sha256"]
    assert summary["core_interval_s"] == training["evaluation"]["core_interval_s"] == [1.44, 4.]
    for row in summary["rows"]:
        mu = row["evaluation_mu_static_dynamic_equal"]
        case = provenance["cases"][row["case"]]
        assert case["mu"] == mu and row["sim2sim"]["full_core_overlap"]
        for engine in ("isaac", "mujoco"):
            assert row[engine]["core_complete"]
            assert row[engine]["last_observed_time_s"] == pytest.approx(6.98, abs=1e-10)
            assert case[engine + "_episode_end"]["completed_full_reference"] is True
        np.testing.assert_allclose(case["isaac_material_readback"]["robot_runtime_friction_range"], [mu, mu], rtol=1e-6)
        np.testing.assert_allclose(case["mujoco_compiled_friction_range"], [mu, mu], rtol=0, atol=1e-14)
        for material in case["isaac_material_readback"]["ground_colliders"]:
            np.testing.assert_allclose([material["static_friction"], material["dynamic_friction"]], [mu, mu], rtol=1e-6)


def test_exact_xy_metric_replay_from_compact_numeric_data_without_simulators():
    summary, provenance, _ = data()
    with np.load(PUBLIC / "xy_trace.npz", allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
        assert len(trace) == 70
        assert all(value.dtype.kind == "f" and np.isfinite(value).all() for value in trace.values())
        assert trace["reference_t"].shape == (349,) and trace["reference_xy"].shape == (349, 2)
        for case in provenance["cases"]:
            for engine in ("isaac", "mujoco"):
                time, xy = trace[f"{case}_{engine}_t"], trace[f"{case}_{engine}_xy"]
                assert xy.shape == (len(time), 2)  # No Z, policy weights, full poses or source mocap.
                assert len(time) == 1397 and time[0] == 0.
                np.testing.assert_allclose(np.diff(time), .005, rtol=0, atol=1e-12)
        cases = {name: (float(item["mu"]), "") for name, item in provenance["cases"].items()}
        assert summarize_new_grid(trace, provenance, cases) == summary


def test_xy_screens_and_documented_table_are_exact_rounded_summary_values():
    summary, _, _ = data(); document = REPORT.read_text()
    screened = {engine: {row["evaluation_mu_static_dynamic_equal"] for row in summary["rows"]
                         if row[engine]["large_reference_xy_screen"]} for engine in ("isaac", "mujoco")}
    assert screened == {"isaac": {.01, .03, .05, .10}, "mujoco": {.01, .03, .05, .10, .15, .20}}
    for row in summary["rows"]:
        mu = row["evaluation_mu_static_dynamic_equal"]
        values = [row[engine][key] for engine in ("isaac", "mujoco") for key in
                  ("full_core_reference_xy_rmse_m", "full_core_reference_xy_peak_m")]
        for engine in ("isaac", "mujoco"):
            metrics = row[engine]
            assert metrics["large_reference_xy_screen"] == (
                metrics["full_core_reference_xy_rmse_m"] > .15 or metrics["full_core_reference_xy_peak_m"] > .30)
        label = "Both" if mu in screened["isaac"] else "MuJoCo" if mu in screened["mujoco"] else "Neither"
        expected = (f"| {mu:.2f} | {values[0]:.4f} / {values[1]:.4f} | {values[2]:.4f} / {values[3]:.4f} | "
                    f"{row['sim2sim']['full_core_xy_rmse_m']:.4f} | {label} |")
        assert expected in document
    assert "not universal" in document and "not a PPO/SAC/NMPC ranking" in document


def test_publication_has_exact_artifact_hashes_and_no_private_paths_or_raw_assets():
    _, _, training = data()
    for name, expected in training["published_artifacts_sha256"].items():
        assert hashlib.sha256((PUBLIC / name).read_bytes()).hexdigest() == expected
    for path in [REPORT, *PUBLIC.glob("*.json")]:
        text = path.read_text()
        for private in ("/home/", "/data/", "zengxiong", "14.103.233.29", "117.50.226.213"):
            assert private not in text, (path.name, private)
    assert {path.name for path in PUBLIC.iterdir()} == {
        "summary.json", "xy_trace.npz", "xy_trace.provenance.json", "topdown.png", "training_metadata.json"}


def test_optional_original_training_artifacts_support_sanitized_metadata():
    raw = ROOT / "outputs/controller_primary_ppo_mu070_20261015_v1/training"
    if not (raw / "manifest.json").is_file():
        pytest.skip("Optional private original training metadata unavailable")
    _, _, training = data()
    for filename, key in (("manifest.json", "training_manifest_sha256"), ("run_status.json", "training_run_status_sha256"),
                          ("model_final.pt", "checkpoint_sha256")):
        if not (raw / filename).is_file():
            pytest.skip(f"Optional private training artifact unavailable: {filename}")
        assert hashlib.sha256((raw / filename).read_bytes()).hexdigest() == training[key]
    manifest = json.loads((raw / "manifest.json").read_text())
    status = json.loads((raw / "run_status.json").read_text())
    assert status["status"] == training["training_status"]
    assert status["environment_transitions"] == training["training_budget"]["environment_transitions_completed"]
    assert status["policy_steps_completed"] == training["training_budget"]["policy_steps_completed"]
    assert manifest["height_regularization"] == training["height_regularization"]
    assert manifest["source_training_modules_sha256"] == training["source_training_modules_sha256"]
    assert manifest["task_profile"]["reward_weights"] == training["drift_reward_weights"]
    assert status["final_material_draw_audit"] == training["final_training_material_draw_audit"]
    for key, value in training["training_domain_randomization"].items():
        if key != "robot_mapping":  # Public wording omits the old inaccurate wheel velocity-control label.
            assert manifest["recipe"][key] == value

"""Public milestone must not silently turn smoke checks into rankings."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_integration_record_is_scoped_to_smokes_and_separate_references():
    record = json.loads((ROOT / "results/2026-10-15-controller-integration.json").read_text())
    catalog = json.loads((ROOT / record["selected_references"]["catalog"]).read_text())
    assert record["status"] == "integration_only_not_controller_ranking"
    assert not record["full_training_results_included"]
    assert not record["friction_boundary_results_included"]
    assert not record["selected_references"]["pooled_success_statistics"]
    assert catalog["default_reference_id"] == record["selected_references"]["primary"]
    assert record["selected_references"]["separate_supplement"] == "doa_sa29_scale2"
    for name in ("ppo_learner_smoke", "sac_learner_smoke"):
        assert not record[name]["trained_policy_claimed"]
    pipeline = record["sac_untrained_actor_pipeline"]
    assert not pipeline["tracking_success_claimed"] and not pipeline["training_claimed"]
    assert pipeline["policy_steps_per_engine"] == catalog["references"]["drift_drl_traj4"]["frames"]
    assert pipeline["hidden_mujoco_resets"] == 0
    assert pipeline["mujoco_root_state_writes"] == 1
    nmpc = record["mujoco_nmpc_runtime_smoke"]
    assert not nmpc["turn_core_reached"] and nmpc["core_xy_rmse_m"] is None
    assert not nmpc["prediction_receives_plant_mu"]
    assert nmpc["compute_ms_median_p95_max"][0] > nmpc["control_deadline_ms"]


def test_recorded_height_objective_does_not_reintroduce_absolute_z():
    record = json.loads((ROOT / "results/2026-10-15-controller-integration.json").read_text())
    task = record["task_revision"]
    profile = json.loads((ROOT / task["profile"]).read_text())
    assert profile["height_regularization"]["mode"] == "vertical_acceleration_only"
    assert not task["absolute_height_reward"] and not task["height_termination"]
    assert not task["reference_vertical_velocity_reward"]
    assert (task["actor_dim"], task["critic_dim"], task["action_dim"]) == (51, 120, 8)
    parity = record["same_state_isaac_numpy_objective_parity"]
    assert parity["reward_terms"] == 15
    assert parity["maximum_dt_scaled_total_reward_abs_error"] < 5e-5

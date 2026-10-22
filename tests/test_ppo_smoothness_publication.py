"""Asset-free numeric replay of the published smoothing comparisons.

Repacking public arrays below is a test-only reconstruction, not a claim to
recreate original NPZ bytes, simulator contacts, policy inference or training.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ppo_smoothness import analyze_trajectory, compare_candidate, sha256


PUBLICATIONS = sorted(path for path in (ROOT / "results").glob("ppo_smoothness_action_rate_*_20261021_v1")
                      if (path / "metrics.csv").is_file())
REFERENCE = ROOT / "results/policy_primary_friction_dynamics_20261016_v1/policy_dynamics_trace.npz"


@pytest.fixture(params=PUBLICATIONS, ids=lambda path: path.name)
def publication(request):
    path = request.param
    with np.load(path / "derived_traces.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return path, json.loads((path / "summary.json").read_text()), arrays


def reconstructed_recording(arrays, prefix):
    def field(name):
        return arrays[prefix + "_" + name]
    time = field("time_s")
    decision_time, actions = field("decision_time_s"), field("decision_actions_clipped")
    assert time.shape == (1397,)
    assert actions.shape == (349, 8)
    assert np.allclose(decision_time, np.arange(349) * .02, atol=1e-12, rtol=0)
    held = np.zeros((1397, 8))
    held[1:] = np.repeat(actions, 4, axis=0)
    applied = np.ones(1397, dtype=bool)
    applied[0] = False
    frames = np.r_[0, np.repeat(np.arange(349), 4)]
    mapping = {"root_position_m": "position_m", "root_quaternion_wxyz": "quaternion_wxyz",
               "root_linear_velocity_w_m_s": "com_velocity_w_m_s",
               "root_angular_velocity_w_rad_s": "angular_velocity_w_rad_s",
               "joint_position_rad": "joint_position_rad", "joint_velocity_rad_s": "joint_velocity_rad_s",
               "applied_joint_torque_nm": "applied_joint_torque_nm"}
    return {**{target: field(source) for target, source in mapping.items()},
            "time_s": time, "physics_dt_s": np.array(.005), "policy_dt_s": np.array(.02),
            "joint_names": field("joint_names"), "valid_mask": np.ones(1397, dtype=bool),
            "actions_clipped": held, "action_was_applied": applied,
            "reference_frame_for_action": frames, "simulation_step_counter": np.arange(1397)}


def test_publication_and_sources_are_hash_bound(publication):
    path, summary, _ = publication
    for item in summary["artifacts"].values():
        assert sha256(path / item["file"]) == item["sha256"]
    for name, digest in summary["analysis_source_sha256"].items():
        assert sha256(ROOT / "scripts" / name) == digest
    assert summary["jitter_resolved_claimed"] is False
    assert summary["requires_fullbody_visual_review"] is True
    assert summary["statistical_success_claimed"] is False
    assert summary["causal_improvement_claimed"] is False


def test_core_metrics_and_screen_replay_without_assets(publication, tmp_path):
    _, summary, arrays = publication
    for case in ("mu0p700", "mu0p150"):
        replayed = {}
        for cohort in ("baseline", "candidate"):
            prefix = case + "_" + cohort
            target = tmp_path / (prefix + ".npz")
            np.savez_compressed(target, **reconstructed_recording(arrays, prefix))
            measured = analyze_trajectory(target, reference=REFERENCE)
            expected = summary["cases"][case][cohort]
            for window in ("full", "core"):
                current, previous = measured["windows"][window], expected["windows"][window]
                assert current["physical_samples"] == previous["physical_samples"]
                assert current["control_decisions"] == previous["control_decisions"]
                assert current["action"]["leg_delta_rms"] == pytest.approx(previous["action"]["leg_delta_rms"], abs=1e-12)
                assert current["torque"]["leg_near_80nm_fraction"] == previous["torque"]["leg_near_80nm_fraction"]
                assert current["body"]["angular_xy_spectrum"]["high_frequency_rms"] == pytest.approx(
                    previous["body"]["angular_xy_spectrum"]["high_frequency_rms"], abs=1e-10)
            for metric in ("xy_rmse_m", "xy_peak_m", "speed_retention", "beta_rmse_deg"):
                assert measured["windows"]["core"]["tracking"][metric] == pytest.approx(
                    expected["windows"]["core"]["tracking"][metric], abs=1e-9)
            replayed[cohort] = measured
        screen = compare_candidate(replayed["candidate"], replayed["baseline"], beta_extra_rmse_deg=5.)
        assert screen["passes"] == summary["cases"][case]["comparison"]["passes"]


def test_fullbody_replay_preserves_public_native_poses(publication):
    path, summary, arrays = publication
    render_path = path.with_name(path.name.replace("_20261021_v1", "_fullbody_20261021_v1"))
    render = json.loads((render_path / "summary.json").read_text())
    assert render["candidate_checkpoint_sha256"] == summary["candidate_training"]["checkpoint_sha256"]
    assert render["dynamics_stepping"] is False
    assert render["camera"]["follows_actual_robot"] is False
    assert render["camera"]["follows_actual_orientation"] is False
    assert render["robot_mesh_alpha"] == 1.
    for item in render["artifacts"].values():
        assert sha256(render_path / item["file"]) == item["sha256"]
    with np.load(render_path / "replay_states.npz", allow_pickle=False) as replay:
        for key in render["case_order"]:
            qpos = replay[key + "_qpos"]
            assert np.array_equal(qpos[:, :3], arrays[key + "_position_m"])
            assert np.array_equal(qpos[:, 3:7], arrays[key + "_quaternion_wxyz"])
            addresses = render["cases"][key]["named_joint_qpos_addresses"]
            assert np.array_equal(qpos[:, addresses], arrays[key + "_joint_position_rad"])
            assert render["cases"][key]["fk_validation"]["samples_checked"] == 1397
            assert render["cases"][key]["fk_validation"]["max_wheel_origin_error_m"] < 2e-6

"""Replay the historical PhysX-only PPO comparison without simulation or GPU.

The full source archive also contains SAC and MuJoCo recordings.  Deliberately
load neither here: this comparison must stand on its PhysX PPO evidence alone.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_physx_ppo_low_friction as comparison


GRID = [.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.]
CHECKPOINT = "6ec411efeaea37fc61389285978b377aa7c5be80e326ce7f18581efc9acdf27b"


@pytest.fixture(scope="module")
def evidence():
    source = comparison.DEFAULT_SOURCE
    summary = json.loads((source / "summary.json").read_text())
    training = json.loads(comparison.DEFAULT_TRAINING.read_text())
    with np.load(source / "policy_dynamics_trace.npz", allow_pickle=False) as archive:
        arrays = {
            name: archive[name] for name in archive.files
            if name in ("schema_version", "wheel_order")
            or name.startswith("reference_")
            or (name.startswith("PPO_") and "_isaac_" in name)
        }
    hashes = {"summary.json": comparison.sha256(source / "summary.json"),
              "policy_dynamics_trace.npz": comparison.sha256(source / "policy_dynamics_trace.npz"),
              "training_metadata.json": comparison.sha256(comparison.DEFAULT_TRAINING)}
    return arrays, summary, training, hashes


@pytest.fixture(scope="module")
def replay(evidence):
    arrays, source, training, hashes = evidence
    return comparison.build_summary(arrays, source, training, input_hashes=hashes)


def test_all_17_cases_use_one_completed_physx_ppo_checkpoint(evidence, replay):
    _, source, training, hashes = evidence
    assert replay["algorithm"] == "PPO"
    assert replay["training_engine"] == replay["evaluation_engine"] == "PhysX"
    assert replay["evaluation_mu"] == GRID
    assert [row["mu"] for row in replay["runs"]] == GRID
    assert len({row["label"] for row in replay["runs"]}) == 17
    assert replay["checkpoint_sha256"] == training["checkpoint_sha256"] == CHECKPOINT
    assert training["training_status"] == "training_completed"
    assert replay["training_seed"] == 42
    assert replay["training_budget"]["environment_transitions_completed"] == 12_288_000
    assert "not_new_matched_backend_pair" in replay["historical_cohort"]
    assert replay["input_sha256"] == hashes
    for row in replay["runs"]:
        original = source["runs"]["PPO_" + comparison.case_name(row["mu"])]
        assert row["engine"] == row["training_engine"] == "PhysX"
        assert original["training"]["checkpoint_sha256"] == CHECKPOINT
        assert row["source_physical_trace_sha256"] == original["input_sha256"]["isaac_trace"]
        assert row["source_physical_report_sha256"] == original["input_sha256"]["isaac_report"]
        assert row["canonical_metrics_recomputed_exactly"] is True


def test_recomputed_native_core_xy_matches_independent_formula_and_canonical(evidence, replay):
    arrays, source, _, _ = evidence
    reference_time = arrays["reference_time_s"]
    reference_xy = arrays["reference_position"][:, :2]
    for row in replay["runs"]:
        label = row["label"]
        time = arrays[label + "_time_s"]
        assert time.shape == (1397,)
        np.testing.assert_allclose(np.diff(time), .005, rtol=0, atol=1e-12)
        core = (time >= 1.44 - 1e-9) & (time <= 4. + 1e-9)
        assert int(core.sum()) == row["samples_in_window"] == 513
        target = np.column_stack([np.interp(time[core], reference_time, reference_xy[:, j])
                                  for j in range(2)])
        error = np.linalg.norm(arrays[label + "_position"][core, :2] - target, axis=1)
        expected_rmse, expected_peak = float(np.sqrt(np.mean(error ** 2))), float(error.max())
        canonical = source["runs"]["PPO_" + comparison.case_name(row["mu"])]["windows"]["core"]["isaac"]
        assert row["window_s"] == [1.44, 4.]
        assert row["coverage_complete"] is True
        assert row["reference_xy_rmse_m"] == expected_rmse == canonical["reference_xy_rmse_m"]
        assert row["reference_xy_peak_m"] == expected_peak == canonical["reference_xy_peak_m"]


def test_xy_screen_is_exact_and_selected_cases_span_both_outcomes(replay):
    assert replay["large_xy_mu"] == [.01, .03, .05, .1]
    assert replay["selected_mu"] == [.7, .15, .1, .05]
    rows = {row["mu"]: row for row in replay["runs"]}
    assert [rows[mu]["large_xy_exploratory_screen"] for mu in replay["selected_mu"]] == [False, False, True, True]
    for row in replay["runs"]:
        assert row["large_xy_exploratory_screen"] == (
            row["reference_xy_rmse_m"] > .15 or row["reference_xy_peak_m"] > .30)
        assert row["xy_screen_limits_m"] == {"rmse": .15, "peak": .30}
    assert not replay["causal_friction_threshold_claimed"]
    assert not replay["statistical_success_rate_claimed"]


def test_speed_comes_from_recorded_com_velocity_not_position_differences(evidence, replay):
    arrays, _, _, _ = evidence
    assert "COM" in replay["com_speed_source"]
    assert "no position differentiation" in replay["com_speed_source"]
    for row in replay["runs"]:
        label = row["label"]
        time = arrays[label + "_time_s"]
        core = (time >= 1.44 - 1e-9) & (time <= 4. + 1e-9)
        velocity = np.asarray(arrays[label + "_com_velocity"], dtype=float)
        speed = np.linalg.norm(velocity[:, :2], axis=1)
        np.testing.assert_allclose(arrays[label + "_speed_m_s"], speed, rtol=1e-12, atol=1e-12)
        assert row["mean_speed_m_s"] == float(speed[core].mean())
        ref_speed = np.interp(time[core], arrays["reference_time_s"], arrays["reference_speed_m_s"])
        assert row["reference_mean_speed_m_s"] == float(ref_speed.mean())
        assert row["mean_speed_retention_ratio"] == float(speed[core].mean() / ref_speed.mean())


def test_no_sac_or_mujoco_evidence_is_required_or_consumed(evidence, replay):
    arrays, source, training, hashes = evidence
    assert not any("mujoco" in name.lower() or name.startswith("SAC_") for name in arrays)
    physx_only = copy.deepcopy(source)
    physx_only["policies"] = {"PPO": physx_only["policies"]["PPO"]}
    physx_only["runs"] = {key: run for key, run in physx_only["runs"].items() if key.startswith("PPO_")}
    for run in physx_only["runs"].values():
        for window in run["windows"].values():
            window.pop("mujoco", None)
        for key in ("reported_source_provenance", "xy_numeric_representation_replay"):
            run[key].pop("mujoco", None)
        for key in ("mujoco_trace", "mujoco_report"):
            run["input_sha256"].pop(key, None)
    actual = comparison.build_summary(arrays, physx_only, training, input_hashes=hashes)
    assert actual == replay


def test_height_contact_and_heading_diagnostics_never_gate_xy_failure(evidence, replay):
    arrays, source, training, _ = evidence
    changed = dict(arrays)
    for row in replay["runs"]:
        label = row["label"]
        changed[label + "_position"] = arrays[label + "_position"].copy()
        changed[label + "_position"][:, 2] += 100.
        changed[label + "_contact_count"] = np.zeros_like(arrays[label + "_contact_count"])
        changed[label + "_contact_force"] = np.zeros_like(arrays[label + "_contact_force"])
        changed[label + "_root_directed_beta_rad"] = np.full_like(arrays[label + "_root_directed_beta_rad"], 1.5)
    # Skip byte-identity enforcement only for this synthetic diagnostic test.
    actual = comparison.build_summary(changed, source, training)
    assert actual["large_xy_mu"] == replay["large_xy_mu"]
    assert not actual["height_or_flight_failure_gate"]
    assert not actual["heading_offset_is_mismatch"]
    for before, after in zip(replay["runs"], actual["runs"]):
        for key in ("reference_xy_rmse_m", "reference_xy_peak_m", "large_xy_exploratory_screen"):
            assert before[key] == after[key]
        assert after["ground_contact_heading_geometry"]["contact_fraction_left_right"] == [0., 0.]
        assert not after["height_or_flight_failure_gate"] and not after["heading_offset_is_mismatch"]


@pytest.mark.parametrize("name", ["training_metadata.json", "policy_dynamics_trace.npz"])
def test_input_hash_mutation_is_rejected(evidence, name):
    arrays, source, training, hashes = evidence
    changed = {**hashes, name: "0" * 64}
    with pytest.raises(ValueError, match="bytes differ|hash mismatch"):
        comparison.build_summary(arrays, source, training, input_hashes=changed)


def test_per_case_checkpoint_mutation_is_rejected(evidence):
    arrays, source, training, _ = evidence
    changed = copy.deepcopy(source)
    changed["runs"]["PPO_mu0p100"]["training"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="training identity mismatch"):
        comparison.build_summary(arrays, changed, training)


def test_xy_recording_mutation_cannot_silently_reuse_canonical_metrics(evidence):
    arrays, source, training, _ = evidence
    changed = dict(arrays)
    name = comparison.prefix(.7) + "_position"
    changed[name] = arrays[name].copy()
    changed[name][400, 0] += .2
    with pytest.raises(ValueError, match="Exact canonical replay differs"):
        comparison.build_summary(changed, source, training)


def test_velocity_mutation_cannot_fall_back_to_saved_or_differentiated_speed(evidence):
    arrays, source, training, _ = evidence
    changed = dict(arrays)
    name = comparison.prefix(.7) + "_com_velocity"
    changed[name] = arrays[name].copy()
    changed[name][400, 0] += 1.
    with pytest.raises(ValueError, match="Speed must use recorded"):
        comparison.build_summary(changed, source, training)


def test_missing_coefficient_or_changed_physics_clock_is_rejected(evidence):
    arrays, source, training, _ = evidence
    changed = copy.deepcopy(source)
    changed["evaluation_mu"] = GRID[:-1]
    with pytest.raises(ValueError, match="complete historical 17-point grid"):
        comparison.build_summary(arrays, changed, training)
    changed_arrays = dict(arrays)
    clock = comparison.prefix(.7) + "_time_s"
    changed_arrays[clock] = arrays[clock].copy()
    changed_arrays[clock][400] += .001
    with pytest.raises(ValueError, match="native 200 Hz physical clock"):
        comparison.build_summary(changed_arrays, source, training)


def test_fresh_output_has_physx_only_replay_hashes_and_never_overwrites(evidence, tmp_path, monkeypatch):
    arrays, _, _, hashes = evidence
    output = tmp_path / "offline-comparison"
    # Renderer execution is tested separately; these tiny files test packaging
    # and provenance without FFmpeg, graphics drivers, plotting, or a simulator.
    def render_stub(_arrays, _summary, destination):
        Path(destination).write_bytes(b"offline-render-test")
    monkeypatch.setattr(comparison, "render_topdown", render_stub)
    monkeypatch.setattr(comparison, "render_friction", render_stub)
    args = comparison.parser().parse_args(["--output-dir", str(output), "--skip-video"])
    result = comparison.run(args)
    assert result["input_sha256"] == hashes
    assert len(result["runs"]) == 17
    assert not result["new_training"] and not result["new_policy_evaluation"]
    assert not result["dynamics_stepping"] and not result["learned_inference"]
    assert not result["pose_alignment"] and not result["time_warp"]
    assert "video" not in result
    assert {path.name for path in output.iterdir()} == {
        "summary.json", "selected_physx_trace.npz", "metrics.csv", "topdown.png", "friction_sensitivity.png"}
    with np.load(output / "selected_physx_trace.npz", allow_pickle=False) as archive:
        selected_prefixes = [comparison.prefix(mu) + "_" for mu in comparison.SELECTED_MU]
        assert not any("mujoco" in name.lower() or name.startswith("SAC_") for name in archive.files)
        for name in archive.files:
            assert (name in ("schema_version", "wheel_order") or name.startswith("reference_")
                    or any(name.startswith(label) for label in selected_prefixes))
            np.testing.assert_array_equal(archive[name], arrays[name])
    with (output / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["mu"]) for row in rows] == GRID
    for row in rows:
        assert row["engine"] == row["training_engine"] == "PhysX"
    for name, item in result["artifacts"].items():
        assert comparison.sha256(output / name) == item["sha256"]
    assert result["analysis_source_sha256"]["compare_physx_ppo_low_friction.py"] == comparison.sha256(comparison.__file__)
    before = {path.name: comparison.sha256(path) for path in output.iterdir()}
    with pytest.raises(ValueError, match="Preserve existing results"):
        comparison.run(args)
    assert {path.name: comparison.sha256(path) for path in output.iterdir()} == before


def test_plot_headings_remain_visible_at_zero_speed_unless_axis_projection_is_degenerate():
    label = comparison.prefix(.1)
    half = np.sqrt(.5)
    arrays = {
        label + "_quaternion": np.array([[1., 0., 0., 0.], [half, 0., half, 0.]]),
        label + "_wheel_quaternion_wxyz": np.array([
            [[1., 0., 0., 0.]]*2, [[half, half, 0., 0.]]*2]),
        label + "_speed_m_s": np.zeros(2),
        label + "_speed_threshold_m_s": np.array(.25),
    }
    masks = comparison.plot_masks(arrays, label)
    np.testing.assert_array_equal(masks["body"], [True, False])
    np.testing.assert_array_equal(masks["wheel"], [[True, True], [False, False]])
    np.testing.assert_array_equal(masks["velocity"], [False, False])


def test_published_comparison_matches_replay_source_and_all_artifact_hashes(replay):
    directory = ROOT/"results/physx_ppo_low_friction_20261021_v1"
    published = json.loads((directory/"summary.json").read_text())
    for key, value in replay.items():
        assert published[key] == value
    for name, expected in published["analysis_source_sha256"].items():
        assert comparison.sha256(ROOT/"scripts"/name) == expected
    for name, entry in published["artifacts"].items():
        assert comparison.sha256(directory/name) == entry["sha256"]
    assert published["video"] == {
        "fps": 50, "frames": 350, "native_sample_stride": 4,
        "first_native_time_s": 0., "last_native_time_s": 6.98,
        "encoded_duration_s": 7., "final_sample_display_duration_s": .02,
        "physics_or_reference_time_warp": False,
    }

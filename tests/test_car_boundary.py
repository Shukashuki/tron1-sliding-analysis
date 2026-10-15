from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import analyze_car_boundary as boundary
import run_car_validation as driver
import analyze_friction_grid as grid
import analyze_saved_xy as xy

ROOT = Path(__file__).parents[1]


def test_short_xy_spikes_do_not_become_sustained_failure():
    t = np.arange(11) * .1
    err = np.array([0., .4, 0., .5, .5, .5, 0., 0., 0., 0., .6])
    result = boundary.sustained_onset(t, err)
    assert result["first_above_threshold_s"] == pytest.approx(.1)
    assert result["first_sustained_window"]["start_s"] == pytest.approx(.3)
    assert result["first_sustained_window"]["observed_duration_s"] == pytest.approx(.2)
    assert result["right_censored_last_high_run"] is True
    assert boundary.onset_phase(result, [1.44, 4.]) == "before_turn_core"
    assert boundary.onset_phase({"first_sustained_window": None}, [1.44, 4.]) is None


def test_onset_uses_elapsed_time_not_sample_count_and_strict_threshold():
    result = boundary.sustained_onset([0., .02, .04, .06], [.31, .31, .31, .31])
    assert result["first_sustained_window"] is None
    result = boundary.sustained_onset([0., .1, .2], [.3, .3, .3])
    assert result["first_above_threshold_s"] is None
    for t, e in [([0., 0.], [0., 1.]), ([0., 1.], [0., float("nan")])]:
        with pytest.raises(ValueError):
            boundary.sustained_onset(t, e)


def test_sampled_changes_do_not_impose_monotonicity_or_bridge_censoring():
    def row(mu, large):
        return {"evaluation_mu_static_dynamic_equal": mu, "isaac": {"large_reference_xy_screen": large}}
    changes = boundary.transitions([row(.1, True), row(.15, False), row(.2, True), row(.3, None), row(.4, False)], "isaac")
    assert [(r["mu_left"], r["mu_right"]) for r in changes] == [(.1, .15), (.15, .2)]


def test_runtime_preflight_and_zero_exit_without_artifacts_are_not_success(tmp_path, monkeypatch):
    monkeypatch.delenv("TRON1_BEYONDMIMIC_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="TRON1_BEYONDMIMIC_ROOT"):
        driver.require_beyondmimic_source(tmp_path)
    upstream = tmp_path / "bm"
    target = upstream / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    target.parent.mkdir(parents=True)
    target.write_text("# test dependency")
    monkeypatch.setenv("TRON1_BEYONDMIMIC_ROOT", str(upstream))
    assert driver.require_beyondmimic_source(tmp_path) == upstream
    with pytest.raises(RuntimeError, match="NOT a physical pass/fail"):
        driver.require_isaac_artifacts(tmp_path)
    for name in ("actor_normalized.pt", "policy_contract.json", "trajectory.npz", "physics_trajectory.npz", "report.json"):
        (tmp_path / name).write_bytes(b"artifact")
    driver.require_isaac_artifacts(tmp_path)
    (tmp_path / "report.json").write_bytes(b"")
    with pytest.raises(RuntimeError):
        driver.require_isaac_artifacts(tmp_path)


def test_published_refinement_replays_and_keeps_precore_onsets_distinct(tmp_path):
    source = ROOT / "results/car_bm_dr_refine_seed42"
    result = grid.replay(source / "xy_trace.npz", tmp_path / "replay", source / "summary.json")
    assert boundary.transitions(result["rows"], "isaac") == [
        {"mu_left": .16, "mu_right": .18, "left_large_xy": True, "right_large_xy": False}]
    assert boundary.transitions(result["rows"], "mujoco") == [
        {"mu_left": .14, "mu_right": .15, "left_large_xy": True, "right_large_xy": False}]
    trace = xy.arrays(source / "xy_trace.npz")
    saved = json.loads((source / "boundary_summary.json").read_text())
    for case, engines in saved["native_clock_xy_onsets"].items():
        for engine, expected in engines.items():
            onset = boundary.sustained_onset(*boundary.curves(trace, case, engine))
            onset["first_sustained_onset_phase"] = boundary.onset_phase(onset, [1.44, 4.])
            assert onset == expected
    assert saved["native_clock_xy_onsets"]["mu0p150"]["isaac"]["first_sustained_onset_phase"] == "before_turn_core"
    assert saved["native_clock_xy_onsets"]["mu0p160"]["isaac"]["first_sustained_onset_phase"] == "within_turn_core"
    assert saved["analysis_source_sha256"] == xy.sha(ROOT / "scripts/analyze_car_boundary.py")
    assert saved["root_xy_replay"]["gif_sha256"] == xy.sha(source / "root_xy_replay.gif")
    assert all(row[e]["repeat_xy_peak_m"] == 0 for row in saved["control_repeats"]["cases"] for e in ("isaac", "mujoco"))

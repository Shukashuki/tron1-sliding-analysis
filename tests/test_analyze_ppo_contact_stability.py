"""Synthetic contact chronology and real-recording offline regression tests."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_contact_stability as contact


def synthetic(n=401):
    time = np.arange(n)*.005
    counts = np.zeros((n, 2), dtype=int)
    counts[:, 0] = (np.arange(n) % 4 < 2)
    counts[:, 1] = (np.arange(n) % 5 < 3)
    force = np.zeros((n, 2, 3)); force[:, :, 2] = counts*100.
    velocity = np.zeros((n, 3)); velocity[:, 2] = .1*np.sin(2*np.pi*25*time)
    angular = np.zeros((n, 3)); angular[:, 0] = np.sin(2*np.pi*25*time)
    return time, counts, force, velocity, angular


def test_interior_run_length_and_boundary_censoring():
    time = np.arange(8)*.005
    runs = contact.binary_runs(time, np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=bool))
    assert [run["observed_duration_s"] for run in runs] == pytest.approx([.01, .01, .005])
    assert runs[0]["left_window_censored"] is True
    assert runs[1]["left_window_censored"] is runs[1]["right_window_censored"] is False
    assert runs[2]["right_window_censored"] is True
    assert all(run["end_s"] <= time[-1] for run in runs)
    stats = contact.run_statistics(time, np.ones(8, dtype=bool))
    assert stats["max_observed_duration_s"] == pytest.approx(.035)
    assert stats["complete_runs"] == 0
    assert stats["left_window_censored_runs"] == stats["right_window_censored_runs"] == 1


def test_last_sample_has_no_invented_future_duration():
    runs = contact.binary_runs(np.arange(4)*.005, np.array([0, 0, 0, 1], dtype=bool))
    assert runs[0]["observed_duration_s"] == 0.
    assert runs[0]["right_window_censored"] is True


def test_empty_true_runs_are_explicit_not_nan():
    stats = contact.run_statistics(np.arange(5)*.005, np.zeros(5, dtype=bool))
    assert stats["runs"] == 0 and stats["max_observed_duration_s"] == 0
    assert stats["mean_observed_duration_s"] is None


def test_sample_duty_and_zoh_time_duty_are_distinct():
    value = contact.contact_statistics(np.arange(4)*.005, np.array([1, 0, 0, 0], dtype=bool))
    assert value["sample_duty_fraction"] == .25
    assert value["time_duty_fraction_zoh"] == pytest.approx(1/3)
    assert value["state_switch_count"] == 1
    assert value["state_switches_per_s"] == pytest.approx(1/.015)


def test_point_count_and_upward_load_are_separate_gates():
    counts = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    force_z = np.array([[3., 10.], [5., 6.], [10., 10.], [0., 4.]])
    value = contact.contact_masks(counts, force_z)
    assert value["measured_force_bearing_contact"]["left"].tolist() == [True, True, False, False]
    assert value["load_eligibility"]["left"].tolist() == [False]*4
    assert value["load_eligibility"]["right"].tolist() == [False, True, False, False]
    assert value["measured_force_bearing_contact"]["none"].tolist() == [False, False, True, False]


def test_no_contact_runs_are_named_without_inverting_neither_condition():
    args = list(synthetic(101)); args[1][:] = 0; args[2][:] = 0
    row = contact.analyze_signals(*args)["full"]
    missing = row["missing_contact_runs"]["measured_force_bearing_contact"]
    assert missing["neither_wheel_has_contact"]["max_observed_duration_s"] == pytest.approx(.5)
    assert missing["at_least_one_wheel_missing"]["max_observed_duration_s"] == pytest.approx(.5)
    assert row["wheel_load_sum"]["mg_normalized"] is None
    assert row["wheel_vertical_velocity"]["available"] is False


def test_chopped_contacts_are_missed_by_50hz_subsampling():
    row = contact.analyze_signals(*synthetic())["full"]
    original = row["contact"]["measured_force_bearing_contact"]["left"]
    reduced = row["policy_rate_subsample_diagnostic"]["contact"]["measured_force_bearing_contact"]["left"]
    assert original["state_switch_count"] == 200
    assert reduced["state_switch_count"] == 0
    assert reduced["sample_duty_fraction"] == 1.
    assert row["policy_rate_subsample_diagnostic"]["aliasing_possible"] is True


def test_native_force_and_vertical_velocity_spectra_are_recorded_units():
    row = contact.analyze_signals(*synthetic())["full"]
    spectrum = row["base_com_vertical_velocity_spectrum"]
    assert spectrum["dominant_frequency_hz"] == pytest.approx(25., abs=1.)
    assert spectrum["sample_hz"] == pytest.approx(200.)
    assert row["force_z"]["left"]["maximum_n"] == 100.
    assert row["force_z"]["left"]["step_rms_n"] > 0
    assert row["wheel_load_sum"]["mass_normalization_available"] is False


@pytest.mark.parametrize("change", ["irregular", "reverse", "nan", "50hz"])
def test_bad_or_downsampled_clock_is_rejected(change):
    values = list(synthetic())
    if change == "irregular": values[0][5] += .001
    if change == "reverse": values[0][5] = values[0][4]
    if change == "nan": values[0][5] = np.nan
    if change == "50hz": values = [array[::4] for array in values]
    with pytest.raises(ValueError):
        contact.analyze_signals(*values)


@pytest.mark.parametrize("kind", ["negative", "fractional", "shape", "nan_force"])
def test_invalid_contact_arrays_rejected(kind):
    values = list(synthetic())
    if kind == "negative": values[1][0, 0] = -1
    if kind == "fractional": values[1] = values[1].astype(float); values[1][0, 0] = .5
    if kind == "shape": values[1] = values[1][:, :1]
    if kind == "nan_force": values[2][0, 0, 2] = np.nan
    with pytest.raises(ValueError):
        contact.analyze_signals(*values)


@pytest.fixture(scope="module")
def archived():
    return contact.load_comparison()


def test_original_nominal_cohorts_core_contact_regression(archived):
    arrays, summary = archived
    assert summary["evaluation_mu"] == .7
    before = summary["cases"]["baseline"]["windows"]["core"]
    after = summary["cases"]["action_rate_0p5"]["windows"]["core"]
    first = before["contact"]["measured_force_bearing_contact"]
    second = after["contact"]["measured_force_bearing_contact"]
    assert first["both"]["sample_duty_fraction"] == pytest.approx(.0935672514619883)
    assert second["both"]["sample_duty_fraction"] == pytest.approx(.341130604288499)
    assert second["none"]["sample_duty_fraction"] == pytest.approx(.27485380116959063)
    assert second["left"]["state_switches_per_s"] == pytest.approx(24.21875)
    assert after["missing_contact_runs"]["measured_force_bearing_contact"]["neither_wheel_has_contact"]["max_observed_duration_s"] == pytest.approx(.06)
    assert after["samples"] == 513 and after["coverage_complete"] is True
    assert arrays["action_rate_0p5_count"].shape == (1397, 2)
    assert not summary["height_or_flight_failure_gate"] and not summary["contact_failure_threshold_added"]
    assert summary["wheel_vertical_velocity_available"] is False


def test_full_tail_contact_improvement_is_not_core_continuity(archived):
    _, summary = archived
    windows = summary["cases"]["action_rate_0p5"]["windows"]
    assert windows["full"]["contact"]["measured_force_bearing_contact"]["both"]["sample_duty_fraction"] > .69
    assert windows["core"]["contact"]["measured_force_bearing_contact"]["both"]["sample_duty_fraction"] < .35
    assert set(windows) == {"full", "core", "pre_core", "post_core", "core_early", "core_late"}


def test_no_overwrite_before_loading_or_rendering(tmp_path):
    args = contact.parser().parse_args(["--baseline", "missing", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Preserve existing"):
        contact.run(args)


def test_archive_hash_and_window_metrics_are_packaged(tmp_path, monkeypatch):
    monkeypatch.setattr(contact, "render_time_figure", lambda arrays, summary, path: Path(path).write_bytes(b"figure fixture"))
    monkeypatch.setattr(contact, "render_spectrum_figure", lambda arrays, summary, path: Path(path).write_bytes(b"spectrum fixture"))
    output = tmp_path/"fresh"
    summary = contact.run(contact.parser().parse_args(["--output-dir", str(output)]))
    for name, artifact in summary["artifacts"].items():
        assert artifact["sha256"] == contact.sha256(output/name)
    saved = json.loads((output/"summary.json").read_text())
    assert saved["mass_normalization_available"] is False
    assert saved["cases"]["action_rate_0p3"]["action_rate_weight"] == -.3

"""Replay published nominal diagnostics without assets, physics or inference."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_controller_pair import posture_diagnostic, summarize_pair
from controller_metrics import heading_signals
from controller_references import get_reference

CASES = [
    ("PPO", "controller_primary_ppo_nominal_pair_v1", "2026-10-15-primary-ppo-nominal-pair.md"),
    ("SAC", "controller_primary_sac_nominal_pair_v2", "2026-10-16-primary-sac-nominal-pair.md"),
    ("NMPC", "controller_primary_nmpc_nominal_pair_v1", "2026-10-16-primary-nmpc-nominal-pair.md"),
]


def publication(directory):
    folder = ROOT / "results" / directory
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "pair_trace.npz", allow_pickle=False) as archive:
        compact = {name: archive[name] for name in archive.files}
    return folder, summary, compact


@pytest.mark.parametrize("algorithm,directory,note", CASES)
def test_public_bundle_identity_full_coverage_and_asset_allowlist(algorithm, directory, note):
    folder, summary, compact = publication(directory)
    assert {path.name for path in folder.iterdir()} == {"summary.json", "pair_trace.npz", "topdown.png"}
    assert summary["algorithm"] == algorithm
    assert summary["reference_spec"] == get_reference("drift_drl_traj4")
    assert summary["fixed_mu"] == .7 and summary["core_coverage_complete"]
    assert not summary["integration_only"] and not summary["pooled_success_statistics"]
    assert not summary["height_or_flight_failure_gate"] and not summary["heading_offset_is_mismatch"]
    assert not summary["hardware_ready"] and not summary["controlled_execution_success_claimed"]
    for name, key in (("pair_trace.npz", "compact_trace_sha256"), ("topdown.png", "topdown_sha256")):
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == summary["artifacts"][key]
    for path in (folder / "summary.json", ROOT / "results" / note):
        content = path.read_text()
        for private in ("/home/", "/data/", "zengxiong", "14.103.233.29", "117.50.226.213"):
            assert private not in content
    assert compact["wheel_order"].tolist() == ["left", "right"]
    assert all(value.dtype.kind != "O" for value in compact.values())
    for label, samples, dt in (("reference", 349, .02), ("isaac", 1397, .005), ("mujoco", 1397, .005)):
        time = compact[f"{label}_time_s"]
        assert len(time) == samples and time[0] == 0.
        np.testing.assert_allclose(np.diff(time), dt, atol=1e-9, rtol=0)
        assert time[-1] == pytest.approx(6.96 if label == "reference" else 6.98, abs=1e-9)
        assert compact[f"{label}_position"].shape == (samples, 3)
        assert compact[f"{label}_joint_pos"].shape == (samples, 8)
    # Reference contact labels must not be fabricated from FK or height.
    assert not any(name.startswith("reference_contact_") for name in compact)


@pytest.mark.parametrize("algorithm,directory,note", CASES)
def test_exact_heading_and_all_window_metric_replay_from_published_vectors(algorithm, directory, note):
    _, summary, compact = publication(directory)
    traces, reconstructions = {}, {}
    for label in ("reference", "isaac", "mujoco"):
        keys = ("time_s", "position", "quaternion")
        if label != "reference":
            keys += ("contact_count", "contact_force", "contact_slip")
        traces[label] = {name: compact[f"{label}_{name}"] for name in keys}
        signals = heading_signals(
            compact[f"{label}_quaternion"], compact[f"{label}_com_velocity"],
            compact[f"{label}_wheel_quaternion_wxyz"], compact[f"{label}_wheel_center_velocity_w"],
            speed_threshold_m_s=float(compact[f"{label}_speed_threshold_m_s"]))
        for name, value in signals.items():
            np.testing.assert_array_equal(value, compact[f"{label}_{name}"])
        reconstructions[label] = {"signals": signals}
    assert summarize_pair(summary["reference_spec"], traces, reconstructions) == summary["windows"]
    assert {label: posture_diagnostic(traces[label]) for label in ("isaac", "mujoco")} == summary["posture_diagnostics"]
    core = summary["windows"]["core"]
    for label in ("isaac", "mujoco"):
        item = core[label]
        assert item["large_xy_exploratory_screen"] == (
            item["reference_xy_rmse_m"] > .15 or item["reference_xy_peak_m"] > .30)
        assert item["samples_in_window"] == 513
        assert item["ground_contact_heading_geometry"]["measured_contact_gate_applied"]
    assert core["reference"]["ground_contact_heading_geometry"] is None
    assert core["reference"]["ground_contact_lateral_slip_m_s"] is None


def test_completed_policy_training_and_nmpc_development_remain_distinct():
    summaries = {algorithm: publication(directory)[1] for algorithm, directory, _ in CASES}
    for algorithm in ("PPO", "SAC"):
        training = summaries[algorithm]["training_provenance"]
        assert training["verified_formal_training"] and training["checkpoint_bytes_verified"]
        assert training["status"] == "training_completed" and training["algorithm"] == algorithm
    note = (ROOT / "results" / CASES[2][2]).read_text().lower()
    assert "untuned" in note
    assert "not frozen" in note or "not a frozen controller" in note
    assert "not a" in note and "ranking" in note
    # Passing file/coverage audits is not project approval to freeze NMPC.
    assert not summaries["NMPC"]["controlled_execution_success_claimed"]

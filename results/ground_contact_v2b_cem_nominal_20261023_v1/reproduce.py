"""Read-only exact replay of public state/contact metrics; no robot assets.

Run from a checkout: python results/ground_contact_v2b_cem_nominal_20261023_v1/reproduce.py
No dynamics, FK, policy inference, training, extraction or file writes occur.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_cem_ground_contact_v2 as metrics
import diagnose_ground_contact_windows_v2 as contact
from controller_metrics import heading_signals


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_metrics(trace, reference):
    signals = heading_signals(trace["quaternion"], trace["com_velocity"],
        trace["wheel_quaternion_wxyz"], np.zeros((2793, 2, 3)))
    windows = {name: metrics.metrics.window_metrics(trace, reference, signals, interval) for name, interval in
        (("core", (1.44, 4.)), ("whole_reference_overlap", (0., 6.96)), ("full_native", (0., 6.98)))}
    support = {"core": metrics.support_summary(trace["time_s"], trace["ground_normal_Fz_n"], (1.44, 4.)),
        "full_contact": metrics.support_summary(trace["time_s"], trace["ground_normal_Fz_n"], (.02, 6.96))}
    return {"windows": windows, "any_wheel_ground_support": support,
        "nonzero_beta_is_intended": True, "height_contact_tilt_or_beta_used_as_XY_failure_gate": False,
        "whole_tracking_reference_end_s": 6.96, "physical_recording_end_s": 6.98,
        "trajectory_aligned_smoothed_or_extrapolated": False}


def replay(bundle=Path(__file__).resolve().parent):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/"artifact_manifest.json").read_text())
    for item in manifest["files"]:
        path = bundle/item["path"]
        assert path.is_file() and not path.is_symlink(), item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], item["path"]
    for name, expected in manifest["replay_helper_sha256"].items():
        assert digest(ROOT/name) == expected, "Replay helper changed: "+name
    analysis = json.loads((bundle/"cem_analysis.json").read_text())
    with np.load(bundle/"native_comparison_signals.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    reference = {key.removeprefix("reference__"): value for key, value in arrays.items() if key.startswith("reference__")}
    expected = {"original_grounded_baseline": analysis["original_grounded_baseline"],
        "prior_actual_V2a": analysis["optional_prior_actual_V2a"]["metrics"], "actual_V2b": analysis["actual_V2b"]}
    reproduced = {}
    for case, value in expected.items():
        prefix = case+"__"
        trace = {key.removeprefix(prefix): array for key, array in arrays.items() if key.startswith(prefix)}
        reproduced[case] = recorded_metrics(trace, reference)
        assert reproduced[case] == value, "State/contact metric mismatch: "+case
    protocol = json.loads((bundle/"protocol.json").read_text())
    screen = metrics.nominal_screen(reproduced["actual_V2b"], reproduced["original_grounded_baseline"], protocol)
    assert screen == analysis["nominal_screen"], "Fixed B nominal screen mismatch"
    with np.load(bundle/"native_contact_context.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    diagnosis = json.loads((bundle/"contact_diagnostics.json").read_text())
    reproduced_contact = contact.recompute_compact(arrays, include_sensitivity=False)
    for key, value in reproduced_contact.items():
        assert value == diagnosis[key], "Contact-window diagnostic mismatch: "+key
    return {"status": "ALL_THREE_CASE_METRICS_AND_ACTUAL_B_CONTACT_DIAGNOSTICS_EXACT",
        "conditions": list(reproduced), "actual_B_failed_checks": screen["failed_checks"],
        "robot_assets_or_weights_required": False, "new_simulation_FK_inference_training_or_file_writes": False,
        "PPO_B_result_claimed": False, "overall_goal_complete_claimed": False}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))

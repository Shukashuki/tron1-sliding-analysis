"""Exact, read-only five-case native metric replay; no robot assets or weights.

Run: python results/ground_contact_v2d_cem_nominal_20261023_v1/reproduce.py
This replays state/contact metrics, NOT omitted full stage-reward histories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_cem_ground_contact_v2 as metrics
import diagnose_ground_contact_windows_v2 as contact
from controller_metrics import heading_signals


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_metrics(trace, reference):
    signals = heading_signals(trace["quaternion"], trace["com_velocity"],
        trace["wheel_quaternion_wxyz"], np.zeros((2793, 2, 3)))
    windows = {name: metrics.metrics.window_metrics(trace, reference, signals, interval)
        for name, interval in (("core", (1.44, 4.)), ("whole_reference_overlap", (0., 6.96)),
                               ("full_native", (0., 6.98)))}
    support = {"core": metrics.support_summary(trace["time_s"], trace["ground_normal_Fz_n"], (1.44, 4.)),
        "full_contact": metrics.support_summary(trace["time_s"], trace["ground_normal_Fz_n"], (.02, 6.96))}
    return {"windows": windows, "any_wheel_ground_support": support,
        "nonzero_beta_is_intended": True, "height_contact_tilt_or_beta_used_as_XY_failure_gate": False,
        "whole_tracking_reference_end_s": 6.96, "physical_recording_end_s": 6.98,
        "trajectory_aligned_smoothed_or_extrapolated": False}


def replay(bundle=Path(__file__).resolve().parent):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "artifact_manifest.json").read_text())
    assert manifest["kind"] == "ACTUAL_CEM_D_COMPLETED_FAILURE_PARTIAL_PPO_D_RUNNING_FIVE_CASE_PROOF"
    expected_files = {item["path"] for item in manifest["files"]}
    assert len(expected_files) == len(manifest["files"])
    assert {p.name for p in bundle.iterdir() if p.is_file()} == expected_files | {"artifact_manifest.json"}
    for item in manifest["files"]:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = bundle / relative
        assert path.is_file() and not path.is_symlink(), item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], item["path"]
    for name, expected in manifest["replay_helper_sha256"].items():
        assert digest(ROOT / name) == expected, "Replay helper changed: " + name
    source = json.loads((bundle / "source_snapshot.manifest.json").read_text())
    assert digest(bundle / "executed_source.tar.gz") == source["archive"]["sha256"]
    assert source["executed_controller_source_sha256"] == manifest["executed_D_source_sha256"]
    entries = {entry["path"]: entry for entry in source["files"]}
    assert len(entries) == len(source["files"])
    with tarfile.open(bundle / "executed_source.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(entries) and {m.name for m in members} == set(entries)
        for member in members:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert path.suffix in (".py", ".json", ".cpp")
            entry = entries[member.name]
            assert member.size == entry["bytes"]
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == entry["sha256"]
    analysis = json.loads((bundle / "cem_analysis.json").read_text())
    report = json.loads((bundle / "cem_report.json").read_text())
    experiment = json.loads((bundle / "cem_experiment.json").read_text())
    assert report["source_sha256"] == experiment["source_sha256"] == manifest["executed_D_source_sha256"]
    assert len(report["source_sha256"]) == 20
    assert report["status"] == experiment["status"] == "completed"
    assert (report["completed_policy_steps"], report["native_samples"]) == (349, 2793)
    assert analysis["actual_D_reward_audit"] == report["actual_reward_audit"]
    assert analysis["actual_D_reward_audit"]["status"] == "verified_exact"
    assert analysis["actual_D_reward_audit"]["stages"] == 349
    assert analysis["actual_D_reward_audit"]["reward_terms"] == 15
    with np.load(bundle / "native_comparison_signals.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    reference = {key.removeprefix("reference__"): value for key, value in arrays.items()
                 if key.startswith("reference__")}
    expected = {"original_grounded_baseline": analysis["original_grounded_baseline"],
        "prior_actual_V2a": analysis["optional_prior_actual_V2a"]["metrics"],
        "prior_actual_V2b": analysis["optional_prior_actual_V2b"]["metrics"],
        "prior_actual_V2c": analysis["optional_prior_actual_V2c"]["metrics"],
        "actual_V2d": analysis["actual_V2d"]}
    reproduced = {}
    for case, value in expected.items():
        prefix = case + "__"
        trace = {key.removeprefix(prefix): array for key, array in arrays.items() if key.startswith(prefix)}
        reproduced[case] = recorded_metrics(trace, reference)
        assert reproduced[case] == value, "State/contact metric mismatch: " + case
    protocol = json.loads((bundle / "protocol.json").read_text())
    screen = metrics.nominal_screen(reproduced["actual_V2d"], reproduced["original_grounded_baseline"], protocol)
    assert screen == analysis["nominal_screen"], "Fixed D nominal screen mismatch"
    with np.load(bundle / "native_contact_context.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    diagnosis = json.loads((bundle / "contact_diagnostics.json").read_text())
    sidecar = json.loads((bundle / "completed_CEM_D_provenance.json").read_text())
    assert sidecar["diagnostic_sha256"] == digest(bundle / "contact_diagnostics.json")
    assert sidecar["input_sha256"] == analysis["input_sha256"]
    assert sidecar["standalone_all_contact_metric_keys_exact"] is True
    assert sidecar["PPO_D_result_claimed"] is False
    reproduced_contact = contact.recompute_compact(arrays, include_sensitivity=False)
    for key, value in reproduced_contact.items():
        assert value == diagnosis[key], "Contact diagnostic mismatch: " + key
    status = json.loads((bundle / "status.json").read_text())
    assert status["CEM_D"]["failed_checks"] == screen["failed_checks"]
    assert status["CEM_D"]["numeric_nominal_episode_pass"] == screen["numeric_nominal_episode_pass"]
    assert status["PPO_D"]["new_formal_policy_results_claimed"] is False
    assert status["goal_complete"] is False and manifest["goal_completion_claimed"] is False
    return {"status": "ALL_FIVE_CASE_METRICS_AND_ACTUAL_D_CONTACT_DIAGNOSTICS_EXACT",
        "conditions": list(reproduced), "actual_D_failed_checks": screen["failed_checks"],
        "full349x15_stage_reward_audit_preserved_not_recomputed_from_compact": True,
        "robot_assets_or_weights_required": False,
        "new_simulation_FK_inference_training_or_file_writes": False,
        "PPO_D_result_claimed": False, "overall_goal_complete_claimed": False}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))

"""Offline provenance/camera tests; no GL context, training or physics jobs."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import render_ppo_smoothness_comparison as render


@pytest.fixture
def candidate(tmp_path):
    grid, training = tmp_path/"validation", tmp_path/"training"
    training.mkdir()
    checkpoint = "b"*64
    for mu in render.MU:
        name = "mu"+f"{mu:.3f}".replace(".", "p")
        source = render.DEFAULT_BASELINE/name/"isaac"
        output = grid/name/"isaac"
        output.mkdir(parents=True)
        shutil.copyfile(source/"physics_trajectory.npz", output/"physics_trajectory.npz")
        report = render.read_json(source/"report.json")
        report["checkpoint_sha256"] = checkpoint
        (output/"report.json").write_text(json.dumps(report))
    (training/"manifest.json").write_text(json.dumps({"training_reward_overrides": {"action_rate_l2": -.3}}))
    (training/"run_status.json").write_text(json.dumps({"status": "training_completed", "trained_policy_claimed": True,
                                                      "environment_transitions": 12288000}))
    ablation = {"status": "training_completed", "smoke_only": False,
        "initialization": "from_scratch_no_checkpoint_resume", "actual_environment_transitions": 12288000,
        "final_training_manifest_sha256": render.sha256(training/"manifest.json"),
        "checkpoint_sha256": checkpoint, "reward_override": {"term": "action_rate_l2", "baseline_weight": -.1, "weight": -.3}}
    (training/"ablation_manifest.json").write_text(json.dumps(ablation))
    return grid, training


def test_native_case_uses_original_pose_and_all_samples():
    trace, audit = render.load_case(render.DEFAULT_BASELINE/"mu0p150/isaac", mu=.15,
                                    checkpoint_sha256=render.BASELINE_CHECKPOINT)
    assert trace["time_s"].shape == (1397,)
    assert trace["time_s"][-1] == pytest.approx(6.98)
    assert audit["checkpoint_sha256"] == render.BASELINE_CHECKPOINT
    assert np.array_equal(trace["time_s"], np.arange(1397)*.005)
    assert tuple(trace["joint_names"]) == ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
                                           "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint")


def test_case_checkpoint_or_friction_mismatch_rejected():
    with pytest.raises(ValueError, match="checkpoint"):
        render.load_case(render.DEFAULT_BASELINE/"mu0p150/isaac", mu=.15, checkpoint_sha256="x"*64)
    with pytest.raises(ValueError, match="friction"):
        render.load_case(render.DEFAULT_BASELINE/"mu0p150/isaac", mu=.7, checkpoint_sha256=render.BASELINE_CHECKPOINT)


def test_native_four_case_comparison_is_bound(candidate):
    grid, training = candidate
    cases, reference, summary = render.load_data(render.DEFAULT_BASELINE, grid, training)
    assert set(cases) == set(render.ORDER)
    assert summary["candidate_action_rate_weight"] == -.3
    assert summary["baseline_action_rate_weight"] == -.1
    assert summary["jitter_reduction_claimed_by_renderer"] is False
    assert summary["root_height_corrected"] is False
    assert summary["video_frames"] == len(range(0, 1397, 4)) == 350
    assert reference["sha256"] == render.MOTION_SHA256
    for trace in cases.values():
        assert np.array_equal(trace["time_s"], cases[render.ORDER[0]]["time_s"])


@pytest.mark.parametrize("file,key,value", [
    ("run_status.json", "status", "training"),
    ("run_status.json", "trained_policy_claimed", False),
    ("run_status.json", "environment_transitions", 12),
    ("ablation_manifest.json", "smoke_only", True),
    ("ablation_manifest.json", "initialization", "resumed"),
    ("ablation_manifest.json", "final_training_manifest_sha256", "x"*64)])
def test_incomplete_or_relabelled_candidate_rejected(candidate, file, key, value):
    grid, training = candidate
    path = training/file
    report = render.read_json(path); report[key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        render.load_data(render.DEFAULT_BASELINE, grid, training)


def test_case_raw_hash_tamper_rejected(candidate):
    grid, _ = candidate
    path = grid/"mu0p150/isaac/physics_trajectory.npz"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA"):
        render.load_case(path.parent, mu=.15, checkpoint_sha256="b"*64)


def test_candidate_checkpoint_bytes_if_present_are_verified(candidate):
    grid, training = candidate
    (training/"model_final.pt").write_bytes(b"not the declared checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        render.load_data(render.DEFAULT_BASELINE, grid, training)


def test_common_camera_uses_reference_only_and_holds_last_knot():
    reference = {"time_s": np.array([0., 1.]), "position": np.array([[1., 2., .75], [3., 4., .75]])}
    assert np.array_equal(render.common_camera_target(reference, .5), [2., 3., .51])
    assert np.array_equal(render.common_camera_target(reference, 1.2), [3., 4., .51])
    # This function intentionally accepts no actual robot pose or orientation.
    assert list(__import__("inspect").signature(render.common_camera_target).parameters) == ["reference", "time_s"]


def test_preserve_existing_output_before_loading_candidate(tmp_path):
    args = render.parser().parse_args(["--candidate-root", "not-present", "--candidate-training", "not-present",
                                       "--output-dir", str(tmp_path), "--skip-video"])
    with pytest.raises(ValueError, match="Preserve existing"):
        render.run(args)


def test_ambiguous_case_layout_rejected(candidate):
    grid, _ = candidate
    shutil.copyfile(grid/"mu0p150/isaac/physics_trajectory.npz", grid/"mu0p150/physics_trajectory.npz")
    with pytest.raises(ValueError, match="unambiguous"):
        render.case_folder(grid, .15)


def test_no_new_simulation_or_inference_calls():
    tree = ast.parse(Path(render.__file__).read_text())
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not calls.intersection({"mj_step", "mj_forward", "mj_fwdPosition", "act_inference", "learn", "adapt_source", "solve_reference"})

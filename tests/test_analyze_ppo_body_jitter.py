"""Additional base-body diagnostics without changing any declared screen."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_ppo_body_jitter as jitter


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def refresh_hashes(folder):
    path = folder/"summary.json"
    summary = json.loads(path.read_text())
    for name, artifact in summary["artifacts"].items():
        artifact["sha256"] = jitter.sha256(folder/name)
    write_json(path, summary)


@pytest.fixture
def publication(tmp_path):
    folder = tmp_path/"publication"
    folder.mkdir()
    time = np.arange(1397)*.005
    arrays = {}
    for mu in (.7, .15):
        for cohort in ("baseline", "candidate"):
            key = jitter.case_name(mu)+"_"+cohort
            candidate = cohort == "candidate"
            omega = np.column_stack(((1. if candidate else 2.)*np.sin(2*np.pi*30*time),
                                     np.zeros(len(time)), .5+.1*np.sin(2*np.pi*18*time)))
            velocity = np.column_stack((1.+.2*np.sin(2*np.pi*20*time), np.zeros(len(time)),
                                        (.3 if candidate else .2)*np.sin(2*np.pi*30*time)))
            position = np.column_stack((time+.001*np.sin(2*np.pi*30*time), np.zeros(len(time)),
                                        .75+(.002 if candidate else .001)*np.sin(2*np.pi*30*time)))
            arrays.update({key+"_time_s": time.copy(), key+"_position_m": position,
                key+"_com_velocity_w_m_s": velocity, key+"_angular_velocity_w_rad_s": omega,
                key+"_quaternion_wxyz": np.tile([1., 0., 0., 0.], (len(time), 1))})
    np.savez_compressed(folder/"derived_traces.npz", **arrays)
    (folder/"metrics.csv").write_text("mu,old_screen\n0.7,fail\n")
    (folder/"diagnostics.png").write_bytes(b"synthetic artifact binding only")
    summary = {"kind": "recorded_physx_ppo_smoothness_publication", "training_engine": "PhysX", "evaluation_engine": "PhysX",
        "coefficients": [.7, .15], "status": "screen_not_passed_requires_review",
        "engineering_screen_passes_both_conditions": False,
        "cases": {"mu0p700": {"comparison": {"passes": False, "status": "fails_unvalidated_screen", "sentinel": "do not change"}},
                  "mu0p150": {"comparison": {"passes": False, "status": "fails_unvalidated_screen", "sentinel": "do not change"}}},
        "analysis_source_sha256": {"analyze_ppo_smoothness.py": jitter.sha256(jitter.HELPER)},
        "artifacts": {name: {"file": name, "sha256": jitter.sha256(folder/name)} for name in
                      ("derived_traces.npz", "metrics.csv", "diagnostics.png")}}
    write_json(folder/"summary.json", summary)
    return folder


def rows_by_key(result):
    return {(row["mu"], row["cohort"], row["window"], row["signal"]): row for row in result["rows"]}


def test_six_dof_signals_keep_units_points_native_clock_and_absolute_energy(publication):
    result = jitter.analyze_publication(publication)
    rows = rows_by_key(result)
    assert len(result["rows"]) == 56 and len(result["relative_changes"]) == 28
    angular = rows[(.7, "baseline", "core", "angular_world_xy")]
    assert angular["units"] == "rad/s" and angular["spectral_metrics"]["samples"] == 513
    assert angular["spectral_metrics"]["sample_hz"] == pytest.approx(200.)
    assert angular["spectral_metrics"]["high_frequency_rms"] == pytest.approx(1., rel=1e-3)
    vz = rows[(.7, "baseline", "core", "com_linear_velocity_world_z")]
    assert vz["units"] == "m/s" and "center of mass" in vz["point"]
    assert vz["spectral_metrics"]["high_frequency_rms"] == pytest.approx(.2/np.sqrt(2.), rel=1e-3)
    height = rows[(.7, "baseline", "core", "root_link_origin_position_world_z")]
    assert height["units"] == "m" and "link-origin" in height["point"]
    assert height["spectral_metrics"]["high_frequency_rms"] == pytest.approx(.001/np.sqrt(2.), rel=1e-3)
    assert rows[(.7, "baseline", "full", "angular_world_xy")]["spectral_metrics"]["samples"] == 1397
    assert rows[(.7, "baseline", "core", "angular_body_xy")]["spectral_metrics"] == angular["spectral_metrics"]


def test_reduction_in_angular_signal_does_not_hide_translation_increase(publication):
    result = jitter.analyze_publication(publication)
    changes = {(row["mu"], row["window"], row["signal"]): row for row in result["relative_changes"]}
    assert changes[(.7, "core", "angular_world_xy")]["reduction_percent"] == pytest.approx(50., abs=1e-12)
    vertical = changes[(.7, "core", "com_linear_velocity_world_z")]
    assert vertical["absolute_hf_rms_increased"] is True
    assert vertical["change_percent"] == pytest.approx(50., abs=1e-12)
    height = changes[(.7, "core", "root_link_origin_position_world_z")]
    assert height["change_percent"] == pytest.approx(100., abs=1e-9)
    assert result["whole_body_jitter_reduction_claimed"] is False
    assert result["jitter_resolved_claimed"] is False


def test_predeclared_screen_and_free_height_intended_yaw_are_unchanged(publication):
    parent = json.loads((publication/"summary.json").read_text())
    result = jitter.analyze_publication(publication)
    original = result["predeclared_engineering_screen"]
    assert original["unchanged"] is True
    assert original["original_status"] == parent["status"]
    assert original["original_passes_both_conditions"] is False
    assert original["original_case_comparisons"] == {name: case["comparison"] for name, case in parent["cases"].items()}
    assert not result["new_reward_or_constraint"] and not result["absolute_height_target_added"]
    assert not result["intended_yaw_or_sideslip_targeted_to_zero"]
    assert not result["reference_or_video_resampled"]
    assert not result["new_training"] and not result["new_simulation"] and not result["new_policy_inference"]
    assert result["requires_fullbody_visual_review"]


def test_optional_body_frame_can_be_omitted_without_other_changes(publication):
    full = jitter.analyze_publication(publication)
    limited = jitter.analyze_publication(publication, include_body_frame=False)
    assert len(limited["rows"]) == 48 and len(limited["relative_changes"]) == 24
    assert not any(row["signal"] == "angular_body_xy" for row in limited["rows"])
    expected = [row for row in full["rows"] if row["signal"] != "angular_body_xy"]
    assert limited["rows"] == expected


def test_verified_artifact_bytes_are_required(publication):
    (publication/"metrics.csv").write_text("tampered prior result")
    with pytest.raises(ValueError, match="Published artifact SHA mismatch: metrics.csv"):
        jitter.analyze_publication(publication)


def test_changed_generic_source_cannot_silently_recompute_old_publication(publication):
    path = publication/"summary.json"
    summary = json.loads(path.read_text())
    summary["analysis_source_sha256"]["analyze_ppo_smoothness.py"] = "0"*64
    write_json(path, summary)
    with pytest.raises(ValueError, match="spectral helper differs"):
        jitter.analyze_publication(publication)


@pytest.mark.parametrize("mutation,match", [("missing", "Missing published signal"),
    ("nonfinite", "finite native-state"), ("50hz", "200-Hz"), ("irregular", "uniform"),
    ("quaternion", "unit wxyz")])
def test_missing_or_nonphysical_sampling_is_not_fabricated(publication, mutation, match):
    path = publication/"derived_traces.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    prefix = "mu0p700_baseline"
    if mutation == "missing":
        arrays.pop(prefix+"_com_velocity_w_m_s")
    elif mutation == "nonfinite":
        arrays[prefix+"_position_m"][2, 2] = np.nan
    elif mutation == "50hz":
        arrays[prefix+"_time_s"] = np.arange(1397)*.02
    elif mutation == "irregular":
        arrays[prefix+"_time_s"][5] += .001
    else:
        arrays[prefix+"_quaternion_wxyz"][0] *= 2.
    np.savez_compressed(path, **arrays)
    refresh_hashes(publication)
    with pytest.raises(ValueError, match=match):
        jitter.analyze_publication(publication)


def test_no_percentage_is_invented_from_zero_baseline(publication):
    path = publication/"derived_traces.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["mu0p700_baseline_com_velocity_w_m_s"][:, 2] = 0.
    np.savez_compressed(path, **arrays)
    refresh_hashes(publication)
    result = jitter.analyze_publication(publication)
    change = next(row for row in result["relative_changes"] if row["mu"] == .7 and row["window"] == "core"
                  and row["signal"] == "com_linear_velocity_world_z")
    assert change["baseline_hf_rms"] == 0.
    assert change["ratio"] is None and change["change_percent"] is None
    assert change["candidate_hf_rms"] > 0.


def test_cli_is_fresh_output_only_and_never_mutates_parent(publication, tmp_path):
    before = {path.name: jitter.sha256(path) for path in publication.iterdir()}
    output = tmp_path/"body_jitter.json"
    args = jitter.parser().parse_args(["--publication-dir", str(publication), "--output", str(output)])
    result = jitter.run(args)
    assert json.loads(output.read_text()) == result
    assert result["source_publication_sha256"] == before["summary.json"]
    assert result["analysis_source_sha256"]["analyze_ppo_body_jitter.py"] == jitter.sha256(jitter.__file__)
    assert result["verified_source_artifact_sha256"]["derived_traces.npz"] == before["derived_traces.npz"]
    old = output.read_bytes()
    with pytest.raises(ValueError, match="Preserve existing diagnostics"):
        jitter.run(args)
    assert output.read_bytes() == old
    assert {path.name: jitter.sha256(path) for path in publication.iterdir()} == before


def test_actual_published_action_rate_trial_retains_angular_and_translation_tradeoffs():
    folder = ROOT/"results/ppo_smoothness_action_rate_0p3_20261021_v1"
    if not (folder/"summary.json").is_file():
        pytest.skip("Optional published first action-rate trial is unavailable")
    result = jitter.analyze_publication(folder)
    rows = rows_by_key(result)
    expected = {(.7, "baseline", "com_linear_velocity_world_z"): .1597178,
                (.7, "candidate", "com_linear_velocity_world_z"): .1608935,
                (.7, "baseline", "root_link_origin_position_world_z"): .0015755,
                (.7, "candidate", "root_link_origin_position_world_z"): .0018692,
                (.15, "baseline", "com_linear_velocity_world_z"): .1941043,
                (.15, "candidate", "com_linear_velocity_world_z"): .1472201,
                (.15, "baseline", "root_link_origin_position_world_z"): .0013431,
                (.15, "candidate", "root_link_origin_position_world_z"): .0017754}
    for (mu, cohort, signal), value in expected.items():
        assert rows[(mu, cohort, "core", signal)]["spectral_metrics"]["high_frequency_rms"] == pytest.approx(value, abs=5e-8)
    assert not result["predeclared_engineering_screen"]["original_passes_both_conditions"]
    assert not result["whole_body_jitter_reduction_claimed"]

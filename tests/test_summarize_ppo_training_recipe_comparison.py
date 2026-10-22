from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_ppo_training_recipe_comparison as mod
from test_analyze_ppo_backend_400hz_matrix import signal_fixture


def fixture():
    traces, rows = {}, []
    for i, name in enumerate(mod.STAGES):
        trace, reference, _ = signal_fixture(.02 * (i + 1))
        train_hz, eval_hz, weight = mod.EXPECTED[name]
        if eval_hz == 200:
            trace = {key: value[::2].copy() if isinstance(value, np.ndarray) and len(value) == 2793 else value
                     for key, value in trace.items()}
            trace["physics_dt_s"] = .005
        trace["signals"] = mod.root_signals(trace)
        traces[name] = trace
        rows.append({"id": name, "title": mod.LABELS[name], "training_physics_hz": train_hz,
            "evaluation_physics_hz": eval_hz, "action_rate_weight": weight,
            "checkpoint_sha256": ("b" if name in ("smooth200", "deploy400") else str(i)) * 64})
    provenance = {"stage_order": list(mod.STAGES), "rows": rows, "grounded_reward_weight_all": .4,
        "dense_ground_support_reward_used": False, "all_main_stages_sideslip_reward_weight": 2.}
    return traces, reference, provenance


def test_actual_native_clocks_and_consistent_core_metrics():
    traces, reference, provenance = fixture()
    result = mod.analyze(traces, reference, provenance)
    for i, row in enumerate(result["rows"]):
        core = row["windows"]["core"]
        assert core["native_samples"] == (513 if i < 2 else 1025)
        assert core["tracking"]["xy_rmse_m"] == pytest.approx(.02 * (i + 1))
        assert core["tracking"]["speed_retention"] == pytest.approx(1.)
        assert core["tracking"]["beta_rmse_deg"] == pytest.approx(0.)
        assert core["world_angular_xy_common10_100"]["common_band_rms"] == pytest.approx(.5, abs=.002)
        assert core["world_angular_xy_common10_100"]["sample_hz"] == (200 if i < 2 else 400)
        assert core["contact"]["load_eligibility"]["both"]["time_duty_fraction_zoh"] == 1.
        assert row["windows"]["full"]["tracking"]["available"] is False
        assert row["windows"]["reference_overlap"]["tracking"]["available"] is True
    assert result["dense400Hz_contact_training_exists"] is False
    assert result["causal_frequency_effect_claimed"] is False
    assert result["beta_on_off_ablation"] is False


def test_load_duty_is_time_weighted_not_sample_fraction():
    traces, reference, _ = fixture()
    trace = traces["original200"]
    trace["contact_count"][:] = 0.
    trace["contact_count"][800] = 1.  # At the final core timestamp only.
    value = mod.window_metrics(trace, reference, (1.44, 4.))
    both = value["contact"]["load_eligibility"]["both"]
    assert both["sample_duty_fraction"] == pytest.approx(1. / 513)
    assert both["time_duty_fraction_zoh"] == 0.
    assert value["contact"]["load_eligibility"]["none"]["time_duty_fraction_zoh"] == 1.


def test_root_only_beta_preserves_nonzero_reference_and_invalid_axes():
    traces, reference, _ = fixture()
    trace = traces["deploy400"]
    trace["com_velocity"][:] = [.5, .5, 0.]
    reference["velocity"][:] = [.5, .5, 0.]
    trace["signals"] = mod.root_signals(trace)
    assert set(trace["signals"]) == {"root_directed_beta_rad", "root_beta_valid"}
    assert np.degrees(trace["signals"]["root_directed_beta_rad"][0]) == pytest.approx(45.)
    assert mod.window_metrics(trace, reference, (1.44, 4.))["tracking"]["beta_rmse_deg"] == pytest.approx(0.)
    trace["quaternion"][:] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
    assert not mod.root_signals(trace)["root_beta_valid"].any()


@pytest.mark.parametrize("change", [
    lambda p:p.update(dense_ground_support_reward_used=True),
    lambda p:p.update(all_main_stages_sideslip_reward_weight=0.),
    lambda p:p["rows"][1].update(training_physics_hz=400),
    lambda p:p["rows"][2].update(checkpoint_sha256="f" * 64),
])
def test_mixed_treatments_or_wrong_actor_identity_rejected(change):
    traces, reference, provenance = fixture(); change(provenance)
    with pytest.raises(ValueError): mod.analyze(traces, reference, provenance)


def test_native_bundle_exact_arrays_and_metric_recomputation(tmp_path):
    traces, reference, provenance = fixture()
    summary = mod.analyze(traces, reference, provenance)
    path = tmp_path / "signals.npz"
    audit = mod.write_bundle(path, traces, reference)
    assert audit["sample_counts"] == {"original200": 1397, "smooth200": 1397, "deploy400": 2793, "retrain400": 2793}
    assert audit["checkpoint_actor_or_mesh_included"] is False
    with np.load(path, allow_pickle=False) as archive:
        for name in mod.STAGES:
            for key in mod.BUNDLE_FIELDS:
                assert np.array_equal(archive[name + "__" + key], traces[name][key])
        assert not any("wheel_quat" in key or "actor" in key for key in archive.files)
    assert mod.verify_bundle_metrics(path, summary)["status"] == "verified"
    summary["rows"][0]["windows"]["core"]["tracking"]["xy_rmse_m"] += .1
    with pytest.raises(ValueError, match="recompute published"):
        mod.verify_bundle_metrics(path, summary)


def test_run_writes_four_row_lf_csv_plot_and_separate_negative(tmp_path, monkeypatch):
    traces, reference, provenance = fixture()
    monkeypatch.setattr(mod, "load_stages", lambda *a: (traces, reference, provenance))
    monkeypatch.setattr(mod, "load_historical_negative", lambda *a: {"not_a400Hz_contact_training_result": True})
    files = [tmp_path / name for name in ("reference.npz", "model.xml", "legacy.py")]
    for path in files:path.write_bytes(path.name.encode())
    args = NS(reference=files[0], model=files[1], legacy_mujoco_evaluator=files[2], output_dir=tmp_path / "result")
    summary = mod.run(args)
    content = (args.output_dir / "metric_table.csv").read_bytes()
    assert content.count(b"\n") == 5 and b"\r" not in content
    assert (args.output_dir / "recipe_comparison.png").is_file()
    assert (args.output_dir / "comparison_signals.npz").is_file()
    assert json.loads((args.output_dir / "historical_ground_reward_negative.json").read_text())["not_a400Hz_contact_training_result"] is True
    assert len(summary["rows"]) == 4
    assert summary["compact_signal_bundle"]["recomputed_metrics"]["status"] == "verified"
    assert all(mod.sha256(args.output_dir / name) == digest for name, digest in summary["artifacts"].items())


def test_existing_output_rejected_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_stages", lambda *a: pytest.fail("Fresh output check must come first"))
    with pytest.raises(ValueError, match="Preserve"):
        mod.run(NS(output_dir=tmp_path))


def test_real_existing_four_native_recordings_match_audited_metrics_without_fk_or_inference():
    reference = Path('/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz')
    folders = [ROOT / value for value in (
        "outputs/controller_primary_ppo_mu070_20261015_v1/mu0p700/isaac",
        "outputs/ppo_contact_timestep_20261022_v1/dt0p005",
        "outputs/ppo_contact_timestep_20261022_v1/dt0p0025",
        "outputs/ppo_backend_400hz_nominal_20261022_v1/cases/physx/physx")]
    if not reference.is_file() or not all((p / "physics_trajectory.npz").is_file() for p in folders):
        pytest.skip("Private original records are optional regression fixtures")
    expected = [(0.056502488325627764, 2.172409620288345, .09375),
        (.056221388688688805, 1.168982703108129, .33984375),
        (.05522103300021738, .9476500756207856, .3271484375),
        (.1279416909811804, 1.2429810920447077, .3564453125)]
    ref = mod.smooth._load_reference(reference)
    for folder, (xy, omega, duty) in zip(folders, expected):
        with np.load(folder / "physics_trajectory.npz", allow_pickle=False) as archive:
            trace = mod.native_trace({key: archive[key] for key in archive.files}, "policy_isaac")
        trace["signals"] = mod.root_signals(trace)
        metric = mod.window_metrics(trace, ref, (1.44, 4.))
        assert metric["tracking"]["xy_rmse_m"] == pytest.approx(xy, abs=1e-12)
        assert metric["world_angular_xy_common10_100"]["common_band_rms"] == pytest.approx(omega, abs=1e-12)
        assert metric["contact"]["load_eligibility"]["both"]["time_duty_fraction_zoh"] == pytest.approx(duty, abs=1e-12)


def test_real_strict_historical_dense_negative_is_separate_and_beta_validity_matches():
    reference = mod.ground.DEFAULT_REFERENCE
    if not Path(reference).is_file() or not all((ROOT / value[0]).is_file() for value in mod.ground.TRAINING_FILES.values()):
        pytest.skip("Private historical metadata are optional regression fixtures")
    result = mod.load_historical_negative(reference)
    assert result["not_a400Hz_contact_training_result"] is True
    assert result["reward_weight_increased"] is False
    baseline, negative = result["cases"]
    assert baseline["windows"]["core"]["tracking"]["xy_rmse_m"] == pytest.approx(.056221388688688805)
    assert negative["windows"]["core"]["tracking"]["xy_rmse_m"] == pytest.approx(.9443547141010037)
    assert negative["windows"]["core"]["tracking"]["beta_valid_samples"] == 335
    assert negative["events_full"]["tilt_ge_30_deg"]["first_persistent_event"]["onset_s"] == 2.335


def test_source_no_dynamics_or_policy_execution_and_no_400only_window_helper():
    source = Path(mod.__file__).read_text()
    for forbidden in ("mj_step(", "torch.jit.load(", "analysis.window_metrics("):
        assert forbidden not in source

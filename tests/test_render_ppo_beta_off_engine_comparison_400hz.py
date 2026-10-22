from pathlib import Path
import copy
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import render_ppo_beta_off_engine_comparison_400hz as mod


def test_partial_two_engine_scope_and_one_actual_trained_actor():
    assert mod.ORDER == ("physx", "mujoco") and mod.CANVAS_SIZE == (1280, 750)
    assert mod.previous.OFF["checkpoint"].startswith("fd9bb855")
    assert mod.previous.OFF["actor"].startswith("3e327483")
    source = Path(mod.__file__).read_text()
    assert '"training_backend_comparison_complete": False' in source
    assert '"MJWarp_trained_actor_included": False' in source
    assert '"beta_reward_weight": 0.' in source
    assert '"lower_HF_alone_establishes_stability": False' in source
    assert "mj_step(" not in source and "mj_forward(" not in source and "torch.jit.load(" not in source


def test_fresh_output_rejected_before_inputs_graphics(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_comparison", lambda *a, **k: pytest.fail("No work before fresh-output guard"))
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(NS(output_dir=tmp_path))


def test_missing_or_wrong_physx_case_rejected_before_reading_raw(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        mod.validate_physx(tmp_path, {"cases": []}, {}, {})
    with pytest.raises(ValueError, match="SAME completed"):
        mod.validate_physx(tmp_path, {"cases": [{"directory": "physx", "case_id": "mjlab_to_physx", "training_backend": "mjlab"}]}, {}, {})


def test_wrong_record_pin_rejected_before_loading_actor_or_raw(tmp_path):
    path = tmp_path/"physx"; path.mkdir()
    (path/"physics_trajectory.npz").write_bytes(b"bad")
    (path/"report.json").write_text("{}")
    case = {"directory": "physx", "case_id": "physx_to_physx_beta_off", "training_backend": "physx", "engine_tag": "physx",
        "validation_backend": "PhysX", "mu": .7, "checkpoint_sha256": mod.previous.OFF["checkpoint"],
        "actor_sha256": mod.previous.OFF["actor"], "raw_file": "physics_trajectory.npz", "report_file": "report.json",
        "recording_sha256": mod.sha256(path/"physics_trajectory.npz"), "report_sha256": mod.sha256(path/"report.json")}
    ledger = {str(file.relative_to(tmp_path)): mod.sha256(file) for file in path.iterdir()}
    with pytest.raises(ValueError, match="Pinned PhysX"):
        mod.validate_physx(tmp_path, {"cases": [case]}, ledger, {})


def test_fixed_follow_two_panels_original_native2000_not_resampled(monkeypatch):
    from PIL import ImageFont
    n = 2793; traces, qpos, targets = {}, {}, {}
    for column, engine in enumerate(mod.ORDER):
        traces[engine] = {"time_s": np.arange(n)*.0025, "position": np.tile([column, 0., .75], (n, 1)),
            "quaternion": np.tile([1., 0., 0., 0.], (n, 1)), "com_velocity": np.tile([1., 0., 0.], (n, 1)),
            "contact_count": np.ones((n, 2)), "contact_force": np.tile([[0., 0., 10.], [0., 0., 10.]], (n, 1, 1)),
            "signals": {"body_heading_rad": np.zeros(n)}}
        qpos[engine] = np.column_stack((np.arange(n), np.full((n, 14), column)))
        targets[engine] = traces[engine]["position"][mod.previous.native_indices()] + [0., 0., -.28]
    camera = NS(lookat=np.zeros(3)); seen, maps = [], []
    runtime = NS(model=object(), data=object(), option=object(), cameras=[camera], font=ImageFont.load_default(), small=ImageFont.load_default(),
        renderer=NS(update_scene=lambda *a, **k: None, render=lambda: np.zeros((360, 640, 3), np.uint8)))
    monkeypatch.setattr(mod, "_pose", lambda model, data, pose: seen.append((pose[0], pose[1], camera.lookat.copy(), camera.distance, camera.azimuth, camera.elevation)))
    monkeypatch.setattr(mod, "_map_panel", lambda draw, trace, ref, frame, *a: maps.append((len(trace["time_s"]), frame)))
    plan = {"lookat_by_case": targets, "distance_m": 1.9, "azimuth_deg": 135., "elevation_deg": -30.}
    reference = {"time_s": np.arange(349)*.02, "position": np.tile([0., 0., .75], (349, 1))}
    image = mod.render_frame(runtime, traces, qpos, reference, {"world_bounds": [[-1., -1.], [2., 1.]]}, plan, 250)
    assert image.size == (1280, 750) and maps == [(350, 250)]*2
    assert [row[:2] for row in seen] == [(2000, 0), (2000, 1)]
    for row, engine in zip(seen, mod.ORDER):
        assert np.array_equal(row[2], targets[engine][250])
        assert row[3:] == (1.9, 135., -30.)


@pytest.fixture(scope="module")
def actual():
    pytest.importorskip("mujoco")
    required = [mod.DEFAULT_PHYSX/"manifest.json", mod.DEFAULT_PHYSX/"physx/physics_trajectory.npz",
        mod.previous.DEFAULT_OFF/"manifest.json", mod.previous.DEFAULT_ON/"manifest.json", mod.previous.DEFAULT_REFERENCE]
    if not all(path.is_file() for path in required):
        pytest.skip("Private/local strict completed retrieval bundles are not included in a clean checkout")
    before = {str(path): mod.sha256(path) for path in required}
    result = mod.load_comparison()
    assert before == {str(path): mod.sha256(path) for path in required}
    return result


def test_actual_same_actor_and_native400_proofs_without_complete_training_matrix(actual):
    traces, reference, summary = actual
    assert list(traces) == ["physx", "mujoco"]
    assert summary["training_backend_comparison_complete"] is False
    assert summary["MJWarp_trained_actor_included"] is False
    assert summary["same_actual_source_actor"] is True and summary["beta_reward_weight"] == 0.
    assert summary["checkpoint_sha256"] == mod.previous.OFF["checkpoint"]
    assert summary["original_grounded_reward_weight"] == .4
    assert all(row["checkpoint_sha256"] == summary["checkpoint_sha256"] for row in summary["rows"])
    assert all(len(trace["time_s"]) == 2793 and trace["physics_dt_s"] == .0025 for trace in traces.values())
    assert summary["rows"][0]["recording_sha256"] == mod.PHYSX_PINS["recording_sha256"]
    assert reference["time_s"][-1] == pytest.approx(6.96)
    assert traces["mujoco"]["time_s"][-1] != 6.98  # original accumulated native float64, not rewritten


def test_actual_full_overlap_late_loss_kept_not_lowerhf_success(actual):
    _, _, summary = actual
    physx, mujoco = [row["windows"] for row in summary["rows"]]
    assert physx["core"]["tracking"]["xy_rmse_m"] == pytest.approx(.373991369444119)
    assert physx["reference_overlap"]["tracking"]["xy_rmse_m"] == pytest.approx(1.3232400891205038)
    assert mujoco["reference_overlap"]["tracking"]["xy_rmse_m"] == pytest.approx(.7383207303946876)
    assert physx["full"]["maximum_tilt_deg"] > 90. and mujoco["full"]["maximum_tilt_deg"] > 90.
    assert all(window["full"]["tracking"]["available"] is False for window in (physx, mujoco))
    assert all(window["core"]["native_samples"] == 1025 for window in (physx, mujoco))
    assert summary["height_or_airborne_failure_gate"] is False
    assert summary["lower_HF_alone_establishes_stability"] is False


def test_actual_compact_native_bundle_reproduces_all_four_windows_exactly(actual, tmp_path):
    traces, reference, summary = actual
    arrays = mod.compact_arrays(traces, reference)
    np.savez_compressed(tmp_path/"native.npz", **arrays)
    with np.load(tmp_path/"native.npz", allow_pickle=False) as data:
        copied = {name: data[name] for name in data.files}
    recomputed = mod.recompute_compact(copied)
    assert all(recomputed[row["id"]] == row["windows"] for row in summary["rows"])
    assert np.array_equal(copied["physx_position"], traces["physx"]["position"])
    assert np.array_equal(copied["mujoco_time_s"], traces["mujoco"]["time_s"])
    rows = mod.previous.overview_rows(summary)
    assert len(rows) == 8 and {row["id"] for row in rows} == {"physx", "mujoco"}
    assert all(row["beta_reward_weight"] == 0. for row in rows)

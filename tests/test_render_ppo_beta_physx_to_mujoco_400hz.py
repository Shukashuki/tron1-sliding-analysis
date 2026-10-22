from pathlib import Path
import copy
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import render_ppo_beta_physx_to_mujoco_400hz as mod


def test_exact400_native_display_selection_preserves_full_horizon():
    indices = mod.native_indices()
    assert indices.shape == (350,) and indices.dtype == np.int64
    assert indices[0] == 0 and indices[150] == 1200 and indices[250] == 2000 and indices[-1] == 2792
    assert np.all(np.diff(indices) == 8)
    assert np.allclose(indices*.0025, np.arange(350)*.02, rtol=0, atol=1e-12)
    assert mod.CANVAS_SIZE == (1280, 750)


def test_only_the_beta_profile_value_may_change():
    on = {"reward_weights": {"sideslip": 2., "grounded": .4}, "action_rate_weight": -.5,
        "height_regularization": {"mode": "vertical_acceleration_only"}}
    off = copy.deepcopy(on); off["reward_weights"]["sideslip"] = 0.
    mod.check_profile_pair(on, off)
    for value in (-.1, -.3):
        wrong = copy.deepcopy(off); wrong["action_rate_weight"] = value
        with pytest.raises(ValueError, match="beyond beta"):
            mod.check_profile_pair(on, wrong)
    wrong = copy.deepcopy(off); wrong["reward_weights"]["sideslip"] = 1e-9
    with pytest.raises(ValueError, match="beyond beta"):
        mod.check_profile_pair(on, wrong)


def test_fresh_output_checked_before_private_data_graphics_or_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_comparison", lambda *a, **k: pytest.fail("Must reject overwrite first"))
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(NS(output_dir=tmp_path))


def test_empty_off_inventory_and_path_escape_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        mod.check_off_case(tmp_path, {"cases": []}, {}, {})
    with pytest.raises(ValueError, match="relative"):
        mod.local_file(tmp_path, "../other")


def test_input_ledger_rejects_changed_file_and_symlink(tmp_path):
    item = tmp_path/"data.json"; item.write_text("{}")
    manifest = {"files": [{"path": "data.json", "size": 2, "sha256": mod.sha256(item)}]}
    assert mod.verify_files(tmp_path, manifest)["data.json"] == mod.sha256(item)
    item.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="SHA/size"):
        mod.verify_files(tmp_path, manifest)
    indirect = tmp_path/"indirect.json"; indirect.symlink_to(item)
    with pytest.raises(ValueError, match="indirect"):
        mod.local_file(tmp_path, "indirect.json")


def test_no_dynamics_actor_load_or_200hz_substitution_and_explicit_caveats():
    source = Path(mod.__file__).read_text()
    assert "torch.jit.load(" not in source and "mj_step(" not in source and "mj_forward(" not in source
    assert '"old_200hz_actor_used": False' in source
    assert '"lower_HF_alone_establishes_stability": False' in source
    assert '"height_or_airborne_failure_gate": False' in source
    assert "GPU4 versus GPU6" in source
    assert "preview_original_time_s=5." in source
    assert 'lineterminator="\\n"' in source


def test_two_panel_original_pose_and_fixed_camera_no_zoom(monkeypatch):
    from PIL import ImageFont
    n = 2793; traces, poses, targets = {}, {}, {}
    for column, label in enumerate(mod.ORDER):
        traces[label] = {"time_s": np.arange(n)*.0025, "position": np.tile([column, 0., .75], (n, 1)),
            "quaternion": np.tile([1., 0., 0., 0.], (n, 1)), "com_velocity": np.tile([1., 0., 0.], (n, 1)),
            "contact_count": np.ones((n, 2)), "contact_force": np.tile([[0., 0., 10.], [0., 0., 10.]], (n, 1, 1)),
            "signals": {"body_heading_rad": np.zeros(n)}}
        poses[label] = np.column_stack((np.arange(n), np.full((n, 14), column)))
        targets[label] = traces[label]["position"][mod.native_indices()] + [0., 0., -.28]
    camera = NS(lookat=np.zeros(3), distance=0.)
    runtime = NS(model=object(), data=object(), option=object(), cameras=[camera],
        font=ImageFont.load_default(), small=ImageFont.load_default(),
        renderer=NS(update_scene=lambda *a, **k: None, render=lambda: np.zeros((360, 640, 3), np.uint8)))
    plan = {"lookat_by_case": targets, "distance_m": 1.8, "azimuth_deg": 135., "elevation_deg": -30.}
    seen, maps = [], []
    monkeypatch.setattr(mod, "_pose", lambda model, data, q: seen.append((q[0], q[1], camera.lookat.copy(), camera.distance, camera.azimuth, camera.elevation)))
    monkeypatch.setattr(mod, "_map_panel", lambda draw, trace, reference, frame, *a: maps.append((len(trace["time_s"]), frame)))
    reference = {"time_s": np.arange(349)*.02, "position": np.tile([0., 0., .75], (349, 1))}
    image = mod.render_frame(runtime, traces, poses, reference, {"world_bounds": [[-1., -1.], [2., 1.]]}, plan, 250)
    assert image.size == (1280, 750) and maps == [(350, 250)]*2
    assert [row[:2] for row in seen] == [(2000, 0), (2000, 1)]
    for row, label in zip(seen, mod.ORDER):
        assert np.array_equal(row[2], targets[label][250])
        assert row[3:] == (1.8, 135., -30.)


@pytest.fixture(scope="module")
def actual():
    pytest.importorskip("mujoco")
    required = [mod.DEFAULT_OFF/"manifest.json", mod.DEFAULT_OFF/"off/rollout.npz",
        mod.DEFAULT_ON/"manifest.json", mod.DEFAULT_REFERENCE, mod.DEFAULT_MODEL, mod.DEFAULT_LEGACY]
    if not all(path.is_file() for path in required):
        pytest.skip("Locked local retrieval payloads are unavailable in a clean checkout")
    hashes = {str(path): mod.sha256(path) for path in required}
    value = mod.load_comparison()
    assert hashes == {str(path): mod.sha256(path) for path in required}
    return value


def test_actual_strict_pair_uses_completed400_physx_actors_and_common_plant(actual):
    traces, reference, summary = actual
    assert list(traces) == ["beta_on", "beta_off"]
    assert all(len(trace["time_s"]) == 2793 for trace in traces.values())
    assert summary["training_backend"] == "PhysX" and summary["validation_backend"] == "native CPU MuJoCo"
    assert summary["training_pair_audit"]["same_initial_network_sha256"] == mod.INITIAL_SHA
    assert summary["training_pair_audit"]["same_native_plant_and_material_readback"] is True
    assert summary["original_grounded_reward_weight"] == .4
    assert summary["dense_ground_support_reward_used"] is False
    assert summary["rows"][1]["checkpoint_sha256"] == mod.OFF["checkpoint"]
    assert summary["rows"][1]["recording_sha256"] == mod.OFF["recording"]
    assert reference["time_s"][-1] == pytest.approx(6.96)
    # Preserve native accumulated float64 times, not rewritten ideal timestamps.
    assert traces["beta_off"]["time_s"][-1] != 6.98
    assert traces["beta_off"]["time_s"][-1] == pytest.approx(6.98, abs=1e-10)


def test_actual_core_does_not_hide_postcore_loss_or_extrapolate_reference(actual):
    _, _, summary = actual
    on, off = [row["windows"] for row in summary["rows"]]
    assert on["core"]["native_samples"] == off["core"]["native_samples"] == 1025
    assert off["core"]["tracking"]["xy_rmse_m"] == pytest.approx(.1480565273872139)
    assert on["reference_overlap"]["tracking"]["xy_rmse_m"] == pytest.approx(.10798092488376734)
    assert off["reference_overlap"]["tracking"]["xy_rmse_m"] == pytest.approx(.7383207303946876)
    assert off["full"]["maximum_tilt_deg"] == pytest.approx(94.85760812379654)
    event = off["full"]["persistent_tilt_diagnostics"]["30"]["first_persistent_event"]
    assert event["onset_s"] == pytest.approx(4.01) and event["confirmation_s"] == pytest.approx(4.11)
    assert on["full"]["persistent_tilt_diagnostics"]["30"]["first_persistent_event"] is None
    assert on["full"]["tracking"]["available"] is False
    assert off["full"]["tracking"]["available"] is False
    assert on["reference_overlap"]["tracking"]["beta_valid_samples"] > off["reference_overlap"]["tracking"]["beta_valid_samples"]
    assert off["core"]["world_angular_xy_common10_100"]["common_band_rms"] < on["core"]["world_angular_xy_common10_100"]["common_band_rms"]
    assert summary["lower_HF_alone_establishes_stability"] is False


def test_actual_native_compact_bundle_recomputes_every_window_exactly(actual, tmp_path):
    traces, reference, summary = actual
    arrays = mod.compact_arrays(traces, reference)
    np.savez_compressed(tmp_path/"compact.npz", **arrays)
    with np.load(tmp_path/"compact.npz", allow_pickle=False) as data:
        restored = {name: data[name] for name in data.files}
    recomputed = mod.recompute_compact(restored)
    assert all(recomputed[row["id"]] == row["windows"] for row in summary["rows"])
    assert np.array_equal(restored["beta_off_position"], traces["beta_off"]["position"])
    assert len(restored["beta_off_time_s"]) == 2793
    rows = mod.overview_rows(summary)
    assert len(rows) == 8
    assert {row["window"] for row in rows} == {"core", "full", "reference_overlap", "post_core"}

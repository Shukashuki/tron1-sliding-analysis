from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_ppo_training_recipe_comparison as mod


def synthetic():
    traces, poses = {}, {}
    for column, stage in enumerate(mod.ORDER):
        n = 349 * int(round(.02 / mod.DT[stage])) + 1
        time = np.arange(n) * mod.DT[stage]
        position = np.column_stack((time+column, time*.2, np.full(n, .75)))
        traces[stage] = {"time_s": time, "position": position, "physics_dt_s": mod.DT[stage],
            "quaternion": np.tile([1., 0., 0., 0.], (n, 1)), "com_velocity": np.tile([1., 0., 0.], (n, 1)),
            "contact_count": np.ones((n, 2)), "contact_force": np.tile([[0., 0., 10.], [0., 0., 10.]], (n, 1, 1)),
            "signals": {"body_heading_rad": time.copy()}, "joint_names": tuple("abcdefgh")}
        poses[stage] = np.column_stack((np.arange(n), np.full((n, 14), column)))
    return traces, poses


def test_mixed_native_indices_are_exact_common50hz_not_shared_native_rows():
    expected_last = [1396, 1396, 2792, 2792]
    for stage, last in zip(mod.ORDER, expected_last):
        index = mod.native_indices(stage)
        assert index.dtype == np.int64 and len(index) == 350 and index[-1] == last
        assert np.allclose(index * mod.DT[stage], np.arange(350)*.02, rtol=0, atol=1e-12)
    assert mod.native_indices("original200")[150] == 600
    assert mod.native_indices("deploy400")[150] == 1200
    with pytest.raises(ValueError, match="Unknown"):
        mod.native_indices("other")


def test_exact_display_views_leave_full_native_poses_and_signals_unchanged():
    traces, poses = synthetic()
    old = {stage: poses[stage].copy() for stage in mod.ORDER}
    display, selected = mod.selected_views(traces, poses)
    for stage in mod.ORDER:
        indices = mod.native_indices(stage)
        assert np.array_equal(selected[stage], poses[stage][indices])
        assert np.array_equal(display[stage]["position"], traces[stage]["position"][indices])
        assert np.array_equal(display[stage]["signals"]["body_heading_rad"], traces[stage]["signals"]["body_heading_rad"][indices])
        assert display[stage]["joint_names"] == traces[stage]["joint_names"]
        assert np.array_equal(old[stage], poses[stage])
        assert len(display[stage]["time_s"]) == 350


def test_actor_identity_and_grounded_are_not_dense400_claims():
    assert mod.CHECKPOINTS["smooth200"] == mod.CHECKPOINTS["deploy400"]
    assert len(set(mod.CHECKPOINTS.values())) == 3
    source = Path(mod.__file__).read_text()
    assert '"dense_ground_support_reward_used": False' in source
    assert '"causal_hz_only_ablation_claimed": False' in source
    assert '"grounded_reward_weight_all": .4' in source
    assert '"actor_weights_loaded": False' in source
    assert "torch.jit.load(" not in source
    assert "mj_step(" not in source and "mj_forward(" not in source


def test_existing_output_rejected_before_any_loader_or_graphics(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_stages", lambda *a, **k: pytest.fail("No loading before fresh-output guard"))
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(NS(output_dir=tmp_path))


def test_tampered_smooth_manifest_rejected_before_reading_metadata(tmp_path):
    for name in mod.SMOOTH_TRAINING_PINS:
        (tmp_path/name).write_text("{}")
    with pytest.raises(ValueError, match="actual training manifest"):
        mod.verify_smooth_training(tmp_path)


def test_stage_guard_rejects_native_clock_interpolation_or_wrong_actor():
    trace = {"time_s": np.arange(1397)*.0025, "physics_dt_s": .0025}
    with pytest.raises(ValueError, match="native clock"):
        mod.validate_stage("original200", trace, {})
    trace = {"time_s": np.arange(1397)*.005, "physics_dt_s": .005}
    with pytest.raises(ValueError, match="actor/reference"):
        mod.validate_stage("original200", trace, {"status": "evaluated", "simulator": "IsaacLab/PhysX", "checkpoint_sha256": "wrong"})


def test_horizontal_frame_uses_600_600_1200_1200_and_constant_follow_camera(monkeypatch):
    from PIL import ImageFont
    traces, poses = synthetic()
    display, _ = mod.selected_views(traces, poses)
    camera = NS(lookat=np.zeros(3), distance=0.)
    runtime = NS(model=object(), data=object(), option=object(), cameras=[camera], font=ImageFont.load_default(), small=ImageFont.load_default(),
        renderer=NS(update_scene=lambda *a, **k: None, render=lambda: np.zeros((360, 640, 3), np.uint8)))
    targets = {stage: display[stage]["position"] + [0., 0., -.28] for stage in mod.ORDER}
    plan = {"lookat_by_case": targets, "distance_m": 1.8, "azimuth_deg": 135., "elevation_deg": -30.}
    seen, maps = [], []
    monkeypatch.setattr(mod, "_pose", lambda model, data, pose: seen.append((pose[0], camera.distance, camera.lookat.copy(), camera.azimuth, camera.elevation)))
    monkeypatch.setattr(mod, "_map_panel", lambda draw, trace, reference, frame, *a: maps.append((len(trace["time_s"]), frame)))
    reference = {"time_s": np.arange(349)*.02, "position": np.tile([0., 0., .75], (349, 1))}
    image = mod.render_frame(runtime, traces, poses, display, reference, {"world_bounds": [[0., 0.], [8., 2.]]}, plan, 150)
    assert image.size == (2560, 720)
    assert [row[0] for row in seen] == [600, 600, 1200, 1200]
    assert maps == [(350, 150)]*4
    assert all(row[1] == 1.8 and row[3:] == (135., -30.) for row in seen)
    assert all(np.array_equal(row[2], targets[stage][150]) for row, stage in zip(seen, mod.ORDER))


def test_hud_reference_end_not_extrapolated():
    traces, _ = synthetic()
    reference = {"time_s": np.arange(349)*.02, "position": np.tile([0., 0., .75], (349, 1))}
    for stage in mod.ORDER:
        result = mod.diagnostics(traces[stage], reference, int(mod.native_indices(stage)[-1]))
        assert result["xy_error_m"] is None
        assert result["height_m"] == .75


def test_minimap_retains_all_exact50hz_path_points_and_axial_glyphs():
    lines, polygons = [], []
    draw = NS(rectangle=lambda *a, **k: None, text=lambda *a, **k: None, ellipse=lambda *a, **k: None,
        line=lambda points, **kw: lines.append((points, kw)), polygon=lambda points, **kw: polygons.append(points))
    trace = {"position": np.column_stack((np.arange(350), np.zeros(350), np.ones(350))),
        "wheel_pos": np.zeros((350, 2, 3)), "body_heading_geometry_valid": np.ones(350, dtype=bool),
        "wheel_heading_geometry_valid": np.ones((350, 2), dtype=bool),
        "signals": {"body_heading_rad": np.zeros(350), "wheel_rolling_plane_heading_rad": np.zeros((350, 2))}}
    reference = {"position": np.zeros((349, 3))}
    mod._map_panel(draw, trace, reference, 150, (0, 0, 630, 172), [[-1., -1.], [351., 1.]], None)
    assert len(lines[1][0]) == 151
    assert len(polygons) == 5  # one directed body tip and two tips per wheel plane


def test_actual_four_stages_strict_loader_provenance_and_source_immutability():
    pytest.importorskip("mujoco")
    paths = [mod.DEFAULT_ORIGINAL/"physics_trajectory.npz", mod.DEFAULT_ORIGINAL/"report.json",
        mod.DEFAULT_SMOOTH/"physics_trajectory.npz", mod.DEFAULT_SMOOTH/"report.json",
        mod.DEFAULT_TIMESTEP/"dt0p0025/physics_trajectory.npz", mod.DEFAULT_TIMESTEP/"dt0p0025/report.json",
        mod.DEFAULT_NOMINAL/"manifest.json", mod.DEFAULT_NOMINAL/"cases/physx/physx/physics_trajectory.npz",
        mod.DEFAULT_NOMINAL/"cases/physx/physx/report.json", mod.DEFAULT_SMOOTH_TRAINING/"manifest.json",
        mod.DEFAULT_SMOOTH_TRAINING/"ablation_manifest.json", mod.beta.DEFAULT_REFERENCE, mod.DEFAULT_MODEL, mod.DEFAULT_LEGACY]
    if not all(path.is_file() for path in paths):
        pytest.skip("Historical/local recordings and locked metadata are not in a clean checkout")
    before = {str(path): mod.sha256(path) for path in paths}
    traces, reference, summary = mod.load_stages()
    assert set(traces) == set(mod.ORDER)
    assert [len(traces[stage]["time_s"]) for stage in mod.ORDER] == [1397, 1397, 2793, 2793]
    assert [row["training_physics_hz"] for row in summary["rows"]] == [200, 200, 200, 400]
    assert [row["evaluation_physics_hz"] for row in summary["rows"]] == [200, 200, 400, 400]
    assert [row["action_rate_weight"] for row in summary["rows"]] == [-.1, -.5, -.5, -.5]
    assert summary["rows"][1]["recording_sha256"] == mod.SMOOTH_RAW_SHA
    assert summary["rows"][1]["report_sha256"] == mod.SMOOTH_REPORT_SHA
    assert summary["grounded_reward_weight_all"] == .4 and summary["dense_ground_support_reward_used"] is False
    assert summary["B_C_same_checkpoint"] is True and summary["causal_hz_only_ablation_claimed"] is False
    override = summary["B_C_actual_training_override_provenance"]
    assert override["reward_override"]["weight"] == -.5
    assert override["training_reward_overrides"] == {"action_rate_l2": -.5}
    assert reference["time_s"][-1] == pytest.approx(6.96)
    assert all(trace["wheel_quaternion_wxyz"].shape == (len(trace["time_s"]), 2, 4) for trace in traces.values())
    assert before == {str(path): mod.sha256(path) for path in paths}

from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_ppo_backend_400hz_nominal as mod
from controller_metrics import heading_signals


def traces_fixture():
    n = 2793
    result = {}
    for column, key in enumerate(mod.ORDER):
        t = np.arange(n) * .0025
        position = np.column_stack((.5*t + column*.1, t*0., t*0. + .7))
        quaternion = np.tile([1., 0., 0., 0.], (n, 1))
        velocity = np.tile([.5, 0., 0.], (n, 1))
        wheelq = np.tile([1., 0., 0., 0.], (n, 2, 1))
        signals = heading_signals(quaternion, velocity, wheelq, np.repeat(velocity[:, None], 2, axis=1))
        result[key] = {"time_s": t, "physics_dt_s": .0025, "position": position, "quaternion": quaternion,
            "com_velocity": velocity, "wheel_pos": position[:, None] + np.tile([[0., .2, -.6], [0., -.2, -.6]], (n, 1, 1)),
            "wheel_quaternion_wxyz": wheelq, "signals": signals, "body_heading_geometry_valid": np.ones(n, bool),
            "wheel_heading_geometry_valid": np.ones((n, 2), bool), "contact_count": np.ones((n, 2)),
            "contact_force": np.tile([[0., 0., 10.], [0., 0., 20.]], (n, 1, 1))}
    return result


def reference_fixture():
    t = np.arange(349) * .02
    return {"time_s": t, "position": np.column_stack((t*.5, t*0., t*0.+.7))}


def runtime_fixture():
    from PIL import ImageFont
    renderer = NS(update_scene=lambda *a, **k: None, render=lambda: np.zeros((360, 640, 3), np.uint8), close=lambda: None)
    model = NS(vis=NS(global_=NS(fovy=45.), map=NS(zfar=50.)), stat=NS(extent=1.))
    return NS(width=640, height=360, model=model, data=object(), renderer=renderer, option=object(),
        cameras=[NS(lookat=np.zeros(3), distance=0., azimuth=0., elevation=0.)],
        font=ImageFont.load_default(), small=ImageFont.load_default())


def test_native50hz_indices350_exact_original_rows():
    indices = mod.native_indices()
    assert len(indices) == 350 and indices[0] == 0 and indices[-1] == 2792
    assert np.array_equal(np.diff(indices), np.full(349, 8))
    assert indices[150] == 1200
    assert np.allclose(indices * .0025, np.arange(350) / 50., rtol=0, atol=1e-15)


def test_common_camera_responds_to_any_original_case_without_moving_poses():
    traces, reference = traces_fixture(), reference_fixture()
    before = {key: trace["position"].copy() for key, trace in traces.items()}
    center, distance = mod.camera_schedule(traces, reference)
    assert center.shape == (350, 3) and distance.shape == (350,)
    assert np.all(distance >= mod.CAMERA["minimum_distance_m"])
    traces[mod.ORDER[-1]]["position"][1200, 0] += 12.
    newcenter, newdistance = mod.camera_schedule(traces, reference)
    assert newdistance[150] > distance[150]
    assert newcenter[150, 0] > center[150, 0]
    assert np.array_equal(newcenter[:150], center[:150])
    for key in mod.ORDER[:-1]:
        np.testing.assert_array_equal(traces[key]["position"], before[key])


def test_global_map_bounds_include_full_drift_and_original_wheels():
    traces, reference = traces_fixture(), reference_fixture()
    traces[mod.ORDER[-1]]["position"][-1, 0] = 100.
    low, high = mod.world_bounds(traces, reference)
    assert high[0] == pytest.approx(100.25)
    assert low[1] <= -.45 and high[1] >= .45


def test_diagnostics_no_reference_endpoint_extrapolation_and_no_height_correction():
    trace = traces_fixture()[mod.ORDER[0]]
    trace["position"][-1, 2] = -.125
    result = mod.diagnostics(trace, reference_fixture(), 2792)
    assert result["xy_error_m"] is None
    assert result["height_m"] == -.125
    assert result["fz"].tolist() == [10., 20.]
    assert mod.diagnostics(trace, reference_fixture(), 1200)["xy_error_m"] == pytest.approx(0.)


def test_load_data_uses_shared_nominal_loader_and_accepts_onlynative400(monkeypatch):
    traces, reference = traces_fixture(), reference_fixture()
    summary = {"nominal_only": True, "full_matrix_complete": False, "rows": [],
        "matrix_status": {"status": "stopped_without_retry", "counts": {"formal_completed": 46}}}
    calls = []
    def loader(*args):
        calls.append(args)
        return traces, reference, summary
    monkeypatch.setitem(sys.modules, "summarize_ppo_backend_400hz_nominal", NS(load_nominal=loader))
    monkeypatch.setattr(mod, "sha256", lambda path: mod.INPUT_PINS["model"])
    loaded, _, audit = mod.load_data("inputs", "ref", "model", "legacy", "protocol")
    assert calls == [("inputs", "ref", "model", "legacy", "protocol")]
    assert audit["video_frames"] == 350 and audit["native_sample_stride"] == 8
    assert audit["visual_reviewed"] is False and audit["state_interpolation"] is False
    assert audit["new_policy_inference"] is False
    assert "46/68" in audit["rendering_limitations"][0]
    assert "may still be running" not in str(audit)
    assert "MJWarp-trained" in audit["panel_titles"]["mjlab_to_mujoco"]
    traces[mod.ORDER[-1]]["physics_dt_s"] = .005
    with pytest.raises(ValueError, match="Native400Hz"):
        mod.load_data("inputs", "ref", "model", "legacy", "protocol")


def test_render_uses_same_camera_allfour_native1200_originalheight(monkeypatch):
    runtime, traces, reference = runtime_fixture(), traces_fixture(), reference_fixture()
    centers, distances = mod.camera_schedule(traces, reference)
    summary = {"camera_lookat_m": centers.tolist(), "camera_distance_m": distances.tolist(),
               "world_bounds": [v.tolist() for v in mod.world_bounds(traces, reference)]}
    poses = {key: np.tile(np.arange(15.), (2793, 1)) for key in mod.ORDER}
    for column, key in enumerate(mod.ORDER):
        poses[key][1200, 0] = column + 10.
    calls = []
    monkeypatch.setattr(mod, "_pose", lambda model, data, pose: calls.append((pose.copy(), runtime.cameras[0].lookat.copy(), runtime.cameras[0].distance)))
    image = mod.render_frame(runtime, traces, poses, reference, summary, 150)
    assert image.size == (2560, 720)
    assert [call[0][0] for call in calls] == [10., 11., 12., 13.]
    assert all(np.array_equal(call[1], centers[150]) for call in calls)
    assert all(call[2] == distances[150] for call in calls)
    with pytest.raises(ValueError, match="Invalid"):
        mod.render_frame(runtime, traces, poses, reference, summary, 350)


def test_fresh_output_guard_before_loader_or_graphics(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_data", lambda *a: pytest.fail("Existing output guard must run first"))
    with pytest.raises(ValueError, match="Preserve existing"):
        mod.run(NS(output_dir=tmp_path))


def test_source_no_dynamics_solver_actorload_or_nativeqposcopy():
    tree = ast.parse(Path(mod.__file__).read_text())
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not {"mj_step", "mj_forward", "mj_fwdPosition", "load_torchscript"}.intersection(calls)
    assert "torch.jit.load(" not in Path(mod.__file__).read_text()


def test_mocked_still_run_saves_nativeposes_camera_and_closesgraphics(tmp_path, monkeypatch):
    traces, reference, runtime = traces_fixture(), reference_fixture(), runtime_fixture()
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    originals = {key: trace["position"].copy() for key, trace in traces.items()}
    monkeypatch.setattr(mod, "load_data", lambda *a: (traces, reference, {"nominal_only": True, "full_matrix_complete": False}))
    monkeypatch.setattr(mod, "create_runtime", lambda *a, **k: runtime)
    def poses(model, trace):
        value = np.zeros((2793, 15))
        value[:, :3], value[:, 3:7] = trace["position"], trace["quaternion"]
        return value, np.arange(7, 15)
    monkeypatch.setattr(mod, "build_qpos", poses)
    monkeypatch.setattr(mod, "replay_fk_audit", lambda *a: {"root_pose_copied_exactly": True})
    geometry = [np.array([[0., 0., 0.], [1., 1., 1.]])] * 350
    monkeypatch.setattr(mod, "geometry_schedule", lambda *a: (geometry, 9))
    monkeypatch.setattr(mod, "actual_camera_basis", lambda *a: (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])))
    monkeypatch.setattr(mod, "mesh_provenance", lambda *a: {"opaque": True})
    monkeypatch.setattr(mod, "sha256", lambda *a: "f" * 64)
    from PIL import Image
    seen, closed = [], []
    def frame(*a):
        seen.append(a[-1])
        return Image.new("RGB", mod.CANVAS_SIZE)
    monkeypatch.setattr(mod, "render_frame", frame)
    runtime.renderer.close = lambda: closed.append(True)
    output = tmp_path / "fresh"
    summary = mod.run(NS(input_root=Path("inputs"), reference=Path("ref"), model=Path("model"),
        legacy_mujoco_evaluator=Path("legacy"), cohort_protocol=Path("protocol"), output_dir=output, skip_video=True))
    assert seen == [150] and closed == [True]
    assert summary["video_generated"] is False
    assert summary["camera_same_across_panels"] is True
    assert summary["camera_individual_recenter"] is False
    assert (output / "comparison.png").is_file()
    with np.load(output / "replay_states.npz", allow_pickle=False) as data:
        for key in mod.ORDER:
            name = key[0] + "_to_" + key[1]
            assert data[name+"_qpos"].shape == (2793, 15)
            np.testing.assert_array_equal(data[name+"_original_root_position_m"], originals[key])
            np.testing.assert_array_equal(data[name+"_selected_native_indices"], mod.native_indices())
        assert data["common_camera_lookat_m"].shape == (350, 3)


def test_selects_cpu_osmesa_before_loader_imports_mujoco(tmp_path, monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    observed = []
    def loader(*a):
        import os
        observed.append(os.environ.get("MUJOCO_GL"))
        raise RuntimeError("stop after backend selection")
    monkeypatch.setattr(mod, "load_data", loader)
    with pytest.raises(RuntimeError, match="backend selection"):
        mod.run(NS(output_dir=tmp_path / "fresh", input_root="inputs", reference="reference", model="model",
            legacy_mujoco_evaluator="legacy", cohort_protocol="protocol"))
    assert observed == ["osmesa"]


def test_camera_fit_all_native_mesh_bounds_and_strict_original_fk(monkeypatch):
    traces, reference = traces_fixture(), reference_fixture()
    geometry = [np.array([[0., 0., 0.], [1., 1., 1.]])] * 350
    basis = (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]))
    centers, distances = mod.camera_schedule(traces, reference, geometry=geometry, basis=basis)
    audit = mod.camera_fit_audit(geometry, centers, distances, basis)
    assert audit["all_visible_mesh_AABB_corners_inside"] is True
    assert audit["maximum_normalized_frustum_extent"] < 1.
    trace = traces[mod.ORDER[0]]
    qpos = np.zeros((2793, 15)); qpos[:, :3], qpos[:, 3:7] = trace["position"], trace["quaternion"]
    monkeypatch.setattr(mod, "validate_fk", lambda *a: {"tolerance_m": 2e-6})
    assert mod.replay_fk_audit(object(), trace, qpos)["tolerance_m"] == 2e-6

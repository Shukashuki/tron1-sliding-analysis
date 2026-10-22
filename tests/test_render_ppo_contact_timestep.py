"""CPU fixtures: native-rate sampling only, without graphics or inference."""
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import render_ppo_contact_timestep as render


def fake_trace(label):
    n, dt = render.NATIVE_SAMPLES[label], render.PHYSICS_DT[label]
    return {"time_s": np.arange(n)*dt, "physics_dt_s": dt,
        "position": np.tile([0., 0., .75], (n, 1)),
        "quaternion": np.tile([1., 0., 0., 0.], (n, 1)),
        "com_velocity": np.tile([1., 0., 0.], (n, 1)),
        "contact_count": np.ones((n, 2), dtype=int),
        "contact_force": np.tile([[0., 0., 0.], [0., 0., 100.]], (n, 1, 1))}


def fake_summary():
    return {"kind": "audited_timestep_fixture", "case_order": list(render.ORDER), "cases": {
        label: {"checkpoint_sha256": render.CHECKPOINT_SHA256,
            "physics_dt_s": render.PHYSICS_DT[label], "decimation": render.DECIMATION[label],
            "native_sample_hz": 1/render.PHYSICS_DT[label],
            "physical_samples": render.NATIVE_SAMPLES[label], "control_decisions": 349}
        for label in render.ORDER}}


def fake_reference():
    return {"time_s": np.array([0., 6.98]), "position": np.tile([0., 0., .75], (2, 1))}


def fake_runtime():
    from PIL import ImageFont
    renderer = SimpleNamespace(update_scene=lambda *a, **kw: None,
        render=lambda: np.zeros((360, 640, 3), dtype=np.uint8), close=lambda: None)
    return SimpleNamespace(width=640, height=360, model=object(), data=object(), option=object(), renderer=renderer,
        cameras=[SimpleNamespace(lookat=np.zeros(3))], font=ImageFont.load_default(), small=ImageFont.load_default())


@pytest.fixture
def loader(monkeypatch):
    raw = {label: {"valid_mask": np.ones((render.NATIVE_SAMPLES[label], 1), dtype=bool), "label": label}
           for label in render.ORDER}
    summary = fake_summary(); calls = []
    def load(*args):
        calls.append(args)
        return copy.deepcopy(raw), copy.deepcopy(summary)
    monkeypatch.setitem(sys.modules, "analyze_ppo_contact_timestep", SimpleNamespace(load_comparison=load))
    monkeypatch.setattr(render, "native_trace", lambda data, kind: fake_trace(data["label"]))
    monkeypatch.setattr(render, "sha256", lambda path: render.MODEL_SHA256)
    monkeypatch.setattr(render, "load_reference", lambda path: fake_reference())
    return raw, summary, calls


def test_native_indices_select_same_original350_clocks_without_interpolation():
    baseline = render.video_native_indices("baseline")
    candidate = render.video_native_indices("candidate")
    assert len(baseline) == len(candidate) == 350
    assert baseline[-1] == 1396 and candidate[-1] == 2792
    np.testing.assert_array_equal(baseline*.005, candidate*.0025)
    assert baseline[120] == 480 and candidate[120] == 960
    with pytest.raises(ValueError, match="Unknown"):
        render.video_native_indices("other")


def test_loads_audited_native_rates_same_actor_poses_and_no_lpf(loader):
    _, _, calls = loader
    traces, _, summary = render.load_data("inputs", "locked_motion")
    assert calls == [("inputs", "locked_motion")]
    assert summary["audited_analysis_kind"] == "audited_timestep_fixture"
    assert summary["native_physical_hz_by_panel"] == {"baseline": 200, "candidate": 400}
    assert summary["native_sample_stride_by_panel"] == {"baseline": 4, "candidate": 8}
    assert set(summary["checkpoint_sha256_by_panel"].values()) == {render.CHECKPOINT_SHA256}
    assert summary["same_actor_comparison"] and summary["actor_last_action_is_requested"]
    for key in ("state_interpolation", "state_filtering", "pose_alignment", "time_alignment",
                "leg_targets_filtered", "wheel_outputs_filtered", "dynamics_stepping"):
        assert summary[key] is False
    assert len(traces["baseline"]["time_s"]) == 1397 and len(traces["candidate"]["time_s"]) == 2793


@pytest.mark.parametrize("field,bad", [("physics_dt_s", .005), ("decimation", 4),
    ("native_sample_hz", 200), ("physical_samples", 1397), ("control_decisions", 348)])
def test_wrong_audited_candidate_rate_and_counts_rejected(loader, field, bad):
    _, summary, _ = loader
    summary["cases"]["candidate"][field] = bad
    with pytest.raises(ValueError, match="native dt/PD/policy/count"):
        render.load_data("inputs")


def test_checkpoint_invalid_mask_shifted_clock_and_model_rejected(loader, monkeypatch):
    raw, summary, _ = loader
    summary["cases"]["candidate"]["checkpoint_sha256"] = "a"*64
    with pytest.raises(ValueError, match="same original8ba"):
        render.load_data("inputs")
    summary["cases"]["candidate"]["checkpoint_sha256"] = render.CHECKPOINT_SHA256
    raw["candidate"]["valid_mask"][1400] = False
    with pytest.raises(ValueError, match="native first episode"):
        render.load_data("inputs")
    raw["candidate"]["valid_mask"][:] = True
    def shifted(data, kind):
        trace = fake_trace(data["label"])
        if data["label"] == "candidate": trace["time_s"] += .0025
        return trace
    monkeypatch.setattr(render, "native_trace", shifted)
    with pytest.raises(ValueError, match="native first episode"):
        render.load_data("inputs")
    monkeypatch.setattr(render, "sha256", lambda path: "b"*64)
    with pytest.raises(ValueError, match="display-model"):
        render.load_data("inputs")


def test_frame_uses480_and960_at2p4s_same_camera_original_height(monkeypatch):
    runtime = fake_runtime(); traces = {label: fake_trace(label) for label in render.ORDER}
    traces["candidate"]["position"][:, [0, 2]] = [2., 1.1]
    original = copy.deepcopy(traces)
    qposes = {label: np.full((render.NATIVE_SAMPLES[label], 15), value)
              for label, value in (("baseline", 1.), ("candidate", 2.))}
    qposes["baseline"][480, 0] = 480.; qposes["candidate"][960, 0] = 960.
    seen = []
    monkeypatch.setattr(render, "_pose", lambda model, data, pose: seen.append(pose.copy()))
    image = render.render_frame(runtime, traces, qposes, fake_reference(), fake_summary(), 120)
    assert image.size == (1280, 506) and all(value % 2 == 0 for value in image.size)
    assert runtime.cameras[0].lookat == pytest.approx([0., 0., .51])
    assert seen[0][0] == 480. and seen[1][0] == 960.
    for label in render.ORDER:
        np.testing.assert_array_equal(traces[label]["position"], original[label]["position"])
    assert "PD200Hz" in render.TITLES["baseline"] and "PD400Hz" in render.TITLES["candidate"]


def test_zero_force_count_is_not_load_eligible_or_flight_label():
    values = render.frame_diagnostics(fake_trace("candidate"), fake_reference(), 960)
    assert values["count"].tolist() == [1, 1]
    assert values["eligible"].tolist() == [False, True]


def test_existing_output_rejected_before_loading_or_graphics(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "load_data", lambda *a, **kw: pytest.fail("Must not load"))
    monkeypatch.setattr(render, "create_runtime", lambda *a, **kw: pytest.fail("Must not create graphics"))
    args = render.parser().parse_args(["--input-root", "missing", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Preserve existing"):
        render.run(args)


def test_mocked_run_exports_every_native_state_and_exact_stride_pair(tmp_path, monkeypatch, loader):
    output = tmp_path/"fresh"; runtime = fake_runtime(); closed = []
    runtime.renderer.close = lambda: closed.append(True)
    monkeypatch.setattr(render, "create_runtime", lambda *a, **kw: runtime)
    monkeypatch.setattr(render, "build_qpos", lambda model, trace:
        (np.tile(np.arange(15.), (len(trace["time_s"]), 1)), np.arange(8)))
    monkeypatch.setattr(render, "validate_fk", lambda model, trace, qpos:
        {"samples_checked": len(qpos), "max_wheel_origin_error_m": 1e-6})
    monkeypatch.setattr(render, "mesh_provenance", lambda *a: {"alpha": 1.})
    from PIL import Image
    seen = []
    def frame(*args):
        seen.append(args[-1]); return Image.new("RGB", render.CANVAS_SIZE)
    monkeypatch.setattr(render, "render_frame", frame)
    class Stream:
        def __init__(self): self.frames = 0; self.closed = False
        def send(self, value):
            if value is not None: self.frames += 1
        def close(self): self.closed = True
    stream = Stream()
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(write_frames=lambda *a, **kw: stream))
    args = render.parser().parse_args(["--input-root", "inputs", "--output-dir", str(output)])
    summary = render.run(args)
    assert stream.frames == 350 and stream.closed and closed == [True]
    assert seen[:2] == [120, 0] and seen[2:] == list(range(350))
    with np.load(output/"replay_states.npz", allow_pickle=False) as data:
        for label in render.ORDER:
            assert data[label+"_qpos"].shape == (render.NATIVE_SAMPLES[label], 15)
            np.testing.assert_array_equal(data[label+"_video_native_indices"], render.video_native_indices(label))
            np.testing.assert_array_equal(data[label+"_native_time_s"], fake_trace(label)["time_s"])
        np.testing.assert_array_equal(data["baseline_native_time_s"][data["baseline_video_native_indices"]],
            data["candidate_native_time_s"][data["candidate_video_native_indices"]])
    written = json.loads((output/"summary.json").read_text())
    assert written["same_actor_comparison"] and written["video_generated"]
    assert summary["cases"]["candidate"]["fk_validation"]["samples_checked"] == 2793
    assert set(written["artifacts"]) == {"comparison.png", "replay_states.npz", "fullbodycomparison.mp4"}


def test_source_has_no_physics_inference_filtering_resize_or_state_interpolation_calls():
    tree = ast.parse(Path(render.__file__).read_text())
    calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
             for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not set(calls).intersection({"mj_step", "mj_forward", "step", "infer", "predict", "learn", "filter", "resize", "interp"})
    assert {"_pose", "validate_fk", "video_native_indices"}.issubset(calls)

"""No-GL, injected-loader tests; no training, simulator or inference executes."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import sys

import numpy as np
from PIL import ImageFont
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import render_ppo_ground_support_comparison as render


def fake_trace():
    return {"time_s": np.arange(1397)*.005, "position": np.tile([0., 0., .75], (1397, 1)),
        "com_velocity": np.tile([1., 0., 0.], (1397, 1)), "quaternion": np.tile([1., 0., 0., 0.], (1397, 1)),
        "contact_count": np.tile([1, 1], (1397, 1)), "contact_force": np.tile([[0., 0., 4.], [0., 0., 100.]], (1397, 1, 1))}


def fake_runtime():
    class Renderer:
        def update_scene(self, *args, **kwargs): pass
        def render(self): return np.zeros((360, 640, 3), dtype=np.uint8)
    font = ImageFont.load_default()
    return SimpleNamespace(width=640, height=360, font=font, small=font, renderer=Renderer(), model=None, data=None,
        option=None, cameras=[SimpleNamespace(lookat=np.zeros(3))])


def test_instantaneous_eligibility_is_distinct_from_count_and_not_flight():
    trace = fake_trace(); reference = {"time_s": np.array([0., 6.98]), "position": np.tile([0., 0., .75], (2, 1))}
    value = render.frame_diagnostics(trace, reference, 0)
    assert value["count"].tolist() == [1, 1] and value["eligible"].tolist() == [False, True]
    assert value["com_speed_m_s"] == 1 and value["tilt_deg"] == 0
    assert value["height_m"] == .75 and value["xy_error_m"] == 0
    trace["contact_count"][0] = [0, 0]
    assert not render.frame_diagnostics(trace, reference, 0)["eligible"].any()


def test_canvas_even_shared_camera_original_poses_and_distinct_titles(monkeypatch):
    runtime = fake_runtime(); trace = fake_trace()
    traces = {key: copy.deepcopy(trace) for key in render.ORDER}
    traces["candidate"]["position"][:, 0] += 1.
    before = traces["candidate"]["position"].copy()
    qposes = {key: np.zeros((1397, 15)) for key in render.ORDER}
    reference = {"time_s": np.array([0., 6.98]), "position": np.tile([0., 0., .75], (2, 1))}
    summary = {"checkpoint_sha256_by_panel": {"baseline": "1"*64, "candidate": "2"*64}}
    monkeypatch.setattr(render, "_pose", lambda *args: None)
    image = render.render_frame(runtime, traces, qposes, reference, summary, 480)
    assert image.size == (1280, 506) and all(x % 2 == 0 for x in image.size)
    assert np.array_equal(traces["candidate"]["position"], before)
    assert runtime.cameras[0].lookat == pytest.approx([0., 0., .51])
    assert "replacement +0.4" in render.TITLES["candidate"]
    assert "original contact reward" in render.TITLES["baseline"]


@pytest.fixture
def loader(monkeypatch):
    raw = {label: {"valid_mask": np.ones((1397, 1), dtype=bool)} for label in render.ORDER}
    summary = {"kind": "audited_ground_support_fixture", "case_order": list(render.ORDER),
        "cases": {"baseline": {"checkpoint_sha256": render.BASELINE_CHECKPOINT_SHA256}, "candidate": {"checkpoint_sha256": "a"*64}}}
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs)); return copy.deepcopy(raw), copy.deepcopy(summary)
    monkeypatch.setitem(sys.modules, "analyze_ppo_ground_support_comparison", SimpleNamespace(load_comparison=load))
    monkeypatch.setattr(render, "native_trace", lambda *a: fake_trace())
    monkeypatch.setattr(render, "sha256", lambda path: render.MODEL_SHA256)
    return raw, summary, calls


def test_loader_binds_distinct_checkpoints_and_propagates_audited_metadata(loader):
    _, _, calls = loader
    traces, reference, summary = render.load_data("case", "training", candidate_status="status")
    assert calls[0][0][:2] == ("case", "training") and calls[0][1]["candidate_status"] == "status"
    assert summary["same_actor_comparison"] is False
    assert summary["audited_analysis_kind"] == "audited_ground_support_fixture"
    assert summary["checkpoint_sha256_by_panel"]["baseline"] != summary["checkpoint_sha256_by_panel"]["candidate"]
    assert summary["state_filtering"] is summary["dynamics_stepping"] is summary["pose_alignment"] is False
    assert len(traces["candidate"]["time_s"]) == 1397 and summary["video_frames"] == 350


def test_same_checkpoint_or_invalid_clock_not_misrepresented(loader, monkeypatch):
    _, summary, _ = loader
    summary["cases"]["candidate"]["checkpoint_sha256"] = render.BASELINE_CHECKPOINT_SHA256
    with pytest.raises(ValueError, match="Independent"):
        render.load_data("case", "training")
    summary["cases"]["candidate"]["checkpoint_sha256"] = "a"*64
    trace = fake_trace(); trace["time_s"][5] += .001
    monkeypatch.setattr(render, "native_trace", lambda *a: trace)
    with pytest.raises(ValueError, match="first episode"):
        render.load_data("case", "training")


def test_source_has_no_physics_inference_or_image_filter_calls():
    tree = ast.parse(Path(render.__file__).read_text())
    calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "" for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not set(calls).intersection({"mj_step", "mj_forward", "step", "infer", "predict", "learn", "filter", "resize"})
    assert {"_pose", "validate_fk"}.issubset(calls)
    assert list(range(0, 1397, 4))[-1] == 1396


def test_fresh_output_precedes_inputs_and_gl(tmp_path):
    args = render.parser().parse_args(["--candidate-dir", "missing", "--candidate-training-dir", "missing", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Preserve existing"):
        render.run(args)

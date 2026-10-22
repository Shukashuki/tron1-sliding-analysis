"""CPU/no-GL renderer tests using only injected state/metadata fixtures."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import render_ppo_torque_control_v2_comparison as render
from test_render_ppo_ground_support_comparison import fake_trace, fake_runtime


def test_distinct_reward_labels_original_poses_even_canvas_common_camera(monkeypatch):
    runtime = fake_runtime(); traces = {label: fake_trace() for label in render.ORDER}
    traces["candidate"]["position"][:, 0] = 1.
    before = traces["candidate"]["position"].copy()
    qposes = {label: np.zeros((1397, 15)) for label in render.ORDER}
    reference = {"time_s": np.array([0., 6.98]), "position": np.tile([0., 0., .75], (2, 1))}
    summary = {"checkpoint_sha256_by_panel": {"baseline": "1"*64, "candidate": "2"*64}}
    monkeypatch.setattr(render, "_pose", lambda *args: None)
    image = render.render_frame(runtime, traces, qposes, reference, summary, 480)
    assert image.size == (1280, 506) and all(x % 2 == 0 for x in image.size)
    assert np.array_equal(traces["candidate"]["position"], before)
    assert runtime.cameras[0].lookat == pytest.approx([0., 0., .51])
    assert "-0.05" in render.TITLES["candidate"] and "original contact +0.4" in render.TITLES["candidate"]
    assert "replacement" not in render.TITLES["candidate"]


@pytest.fixture
def loader(monkeypatch):
    raw = {label: {"valid_mask": np.ones((1397, 1), dtype=bool)} for label in render.ORDER}
    summary = {"kind": "audited_torque_control_v2_fixture", "case_order": list(render.ORDER),
        "cases": {"baseline": {"checkpoint_sha256": render.BASELINE_CHECKPOINT_SHA256}, "candidate": {"checkpoint_sha256": "a"*64}}}
    calls = []
    def load(*args, **kwargs): calls.append((args, kwargs)); return copy.deepcopy(raw), copy.deepcopy(summary)
    monkeypatch.setitem(sys.modules, "analyze_ppo_torque_control_v2", SimpleNamespace(load_comparison=load))
    monkeypatch.setattr(render, "native_trace", lambda *a: fake_trace())
    monkeypatch.setattr(render, "sha256", lambda path: render.MODEL_SHA256)
    return raw, summary, calls


def test_audited_loader_metadata_and_independent_checkpoints_retained(loader):
    _, _, calls = loader
    traces, reference, summary = render.load_data("case", "training", candidate_status="torque_control_v2_status.json")
    assert calls[0][0][:2] == ("case", "training") and calls[0][1]["candidate_status"] == "torque_control_v2_status.json"
    assert summary["audited_analysis_kind"] == "audited_torque_control_v2_fixture"
    assert summary["same_actor_comparison"] is False
    assert summary["checkpoint_sha256_by_panel"]["baseline"] != summary["checkpoint_sha256_by_panel"]["candidate"]
    assert "not a contact reward replacement" in summary["renderer_reward_labels"]["candidate_addition"]
    assert summary["state_filtering"] is summary["pose_alignment"] is summary["dynamics_stepping"] is False
    assert len(traces["candidate"]["time_s"]) == 1397 and summary["video_frames"] == 350


def test_same_checkpoint_and_shifted_clock_rejected(loader, monkeypatch):
    _, summary, _ = loader
    summary["cases"]["candidate"]["checkpoint_sha256"] = render.BASELINE_CHECKPOINT_SHA256
    with pytest.raises(ValueError, match="Independent"):
        render.load_data("case", "training")
    summary["cases"]["candidate"]["checkpoint_sha256"] = "a"*64
    trace = fake_trace(); trace["time_s"] += .005
    monkeypatch.setattr(render, "native_trace", lambda *a: trace)
    with pytest.raises(ValueError, match="first episode"):
        render.load_data("case", "training")


def test_contact_eligibility_is_not_count_or_flight():
    trace = fake_trace(); reference = {"time_s": np.array([0., 6.98]), "position": np.tile([0., 0., .75], (2, 1))}
    value = render.frame_diagnostics(trace, reference, 0)
    assert value["count"].tolist() == [1, 1] and value["eligible"].tolist() == [False, True]
    assert value["com_speed_m_s"] == 1. and value["height_m"] == .75


def test_no_physics_inference_or_filter_calls_and_original_playback_clock():
    tree = ast.parse(Path(render.__file__).read_text())
    calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "" for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not set(calls).intersection({"mj_step", "mj_forward", "step", "infer", "predict", "learn", "filter", "resize"})
    assert {"_pose", "validate_fk"}.issubset(calls)
    ids = np.arange(0, 1397, 4); assert len(ids) == 350 and ids[-1]*.005 == pytest.approx(6.98)


def test_fresh_output_before_loader_or_gl(tmp_path):
    args = render.parser().parse_args(["--candidate-dir", "missing", "--candidate-training-dir", "missing", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Preserve existing"):
        render.run(args)

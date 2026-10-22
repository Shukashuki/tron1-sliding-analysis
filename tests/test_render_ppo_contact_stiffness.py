"""CPU-only replay invariants and even, unscaled movie-canvas regression."""
from pathlib import Path
from types import SimpleNamespace
import ast
import sys

import numpy as np
from PIL import ImageFont
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import render_ppo_contact_stiffness as render


def test_no_new_dynamics_or_inference_calls():
    tree = ast.parse(Path(render.__file__).read_text())
    names = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "" for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not set(names).intersection({"mj_step", "mj_forward", "step", "infer", "predict", "learn"})
    assert {"_pose", "validate_fk"}.issubset(names)


def test_original_clock_and_common_reference_target():
    reference = render.load_reference(render.DEFAULT_REFERENCE)
    assert render.common_camera_target(reference, 2.4) == pytest.approx(reference["position"][120]+[0., 0., -.24])
    assert render.common_camera_target(reference, 6.98) == pytest.approx(reference["position"][-1]+[0., 0., -.24])
    assert len(range(0, 1397, 4)) == 350
    assert np.arange(0, 1397, 4)[-1]*.005 == pytest.approx(6.98)


def test_canvas_is_exactly_even_without_resizing(monkeypatch):
    class Renderer:
        def update_scene(self, *args, **kwargs):
            pass
        def render(self):
            return np.zeros((360, 640, 3), dtype=np.uint8)
    font = ImageFont.load_default()
    runtime = SimpleNamespace(width=640, height=360, font=font, small=font, renderer=Renderer(),
        model=None, data=None, option=None, cameras=[SimpleNamespace(lookat=np.zeros(3))])
    trace = {"time_s": np.array([0.]), "position": np.array([[0., 0., .75]]),
        "com_velocity": np.array([[1., 0., 0.]]), "quaternion": np.array([[1., 0., 0., 0.]]),
        "contact_count": np.ones((1, 2), dtype=int), "contact_force": np.zeros((1, 2, 3))}
    traces = {key: trace for key in render.ORDER}
    qpos = {key: np.zeros((1, 15)) for key in render.ORDER}
    reference = {"time_s": np.array([0.]), "position": np.array([[0., 0., .75]])}
    summary = {"cases": {"kp500": {"deployed_leg_kp": 500.}, "kp250": {"deployed_leg_kp": 250.}}}
    monkeypatch.setattr(render, "_pose", lambda *args: None)
    image = render.render_frame(runtime, traces, qpos, reference, summary, 0)
    assert image.size == (1280, render.CANVAS_HEIGHT) == (1280, 506)
    assert all(size % 2 == 0 for size in image.size)


def test_output_freshness_precedes_loading_and_gl(tmp_path):
    args = render.parser().parse_args(["--candidate", "missing", "--output-dir", str(tmp_path), "--skip-video"])
    with pytest.raises(ValueError, match="Preserve existing"):
        render.run(args)

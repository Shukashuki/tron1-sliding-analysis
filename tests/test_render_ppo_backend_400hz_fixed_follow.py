from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import render_ppo_backend_400hz_fixed_follow as mod


def test_preserves_existing_output_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.original, "load_data", lambda *a: pytest.fail("Must not load"))
    with pytest.raises(ValueError, match="Preserve"):
        mod.run(NS(output_dir=tmp_path))


def test_frozen_original_graphics_and_loader_identity():
    from ppo_backend_400hz_contract import sha256
    assert sha256(mod.original.__file__) == mod.ORIGINAL_SHA256
    assert len(mod.original.native_indices()) == 350
    assert mod.original.native_indices()[-1] == 2792


def test_each_panel_uses_fixed_follow_without_changing_original_pose(monkeypatch):
    from PIL import ImageFont
    draw_calls = []
    camera = NS(distance=1.5, lookat=np.zeros(3))
    renderer = NS(update_scene=lambda data, **kw: draw_calls.append(kw["camera"].lookat.copy()),
        render=lambda: np.zeros((360, 640, 3), dtype=np.uint8))
    runtime = NS(model=None, data=None, option=None, renderer=renderer,
        font=ImageFont.load_default(), small=ImageFont.load_default())
    poses = {key:np.tile(np.arange(15.), (2793, 1)) for key in mod.ORDER}
    traces = {key:{} for key in mod.ORDER}
    applied, observed = [], []
    monkeypatch.setattr(mod, "_pose", lambda model,data,pose: observed.append(pose.copy()))
    def follow(runtime, plan, key, frame):
        applied.append((key, frame))
        camera.lookat[:] = [len(applied), 0., 0.]
        return camera
    monkeypatch.setattr(mod, "apply_camera", follow)
    monkeypatch.setattr(mod.original, "diagnostics", lambda *a: dict(xy_error_m=.1, speed_m_s=1.,
        tilt_deg=2., height_m=.75, count=[1,1], fz=[100.,100.]))
    monkeypatch.setattr(mod.original, "_map_panel", lambda *a:None)
    result = mod.render_frame(runtime, traces, poses, {}, [], {"distance_m":1.5}, 150)
    assert result.size == (2560, 720)
    assert applied == [(key,150) for key in mod.ORDER]
    assert [v[0] for v in draw_calls] == [1.,2.,3.,4.]
    assert all(np.array_equal(v,np.arange(15.)) for v in observed)
    assert camera.distance == 1.5


def test_invalid_video_frame_is_rejected():
    with pytest.raises(ValueError, match="Invalid"):
        mod.render_frame(None, None, None, None, None, None, 350)

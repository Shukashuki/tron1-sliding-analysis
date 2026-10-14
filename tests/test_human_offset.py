import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("attach_human_wheel_offset", ROOT / "scripts/attach_human_wheel_offset.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_bilateral_offsets_remain_distinct():
    time = np.array([0., .1, .2, .3])
    root = np.column_stack((time * 10., np.zeros_like(time)))
    wheel = np.zeros((4, 2))
    heel = np.zeros((4, 2, 3))
    toe = np.zeros_like(heel)
    toe[:, 0, :2] = [1., 0.]
    toe[:, 1, :2] = [0., 1.]
    out = mod.attach(time, wheel, root, heel, toe, 0.)
    np.testing.assert_allclose(out["human_to_gmr_wheel_heading_offset_deg"], [[0., 90.]] * 4)
    assert out["analysis_valid_mask"].all()
    np.testing.assert_allclose(out["human_course_foot_offset_deg"], [[0., -90.]] * 4)


def test_short_foot_axis_rejected():
    time = np.array([0., .1, .2])
    marker = np.zeros((3, 2, 3))
    with pytest.raises(ValueError, match="too short"):
        mod.attach(time, np.zeros((3, 2)), np.zeros((3, 2)), marker, marker, 0.)

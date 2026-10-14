import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_saved_xy", ROOT / "scripts/analyze_saved_xy.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_published_trace_recomputes_summary():
    trace = mod.arrays(ROOT / "results/car_baseline/xy_trace.npz")
    summary = mod.summarize(trace, {}, __import__("json").loads(
        (ROOT / "config/evaluation_friction.json").read_text()))
    rows = {row["case"]: row for row in summary["rows"]}
    assert rows["mu0p05"]["isaac"]["large_reference_xy_screen"] is None
    assert rows["mu0p05"]["mujoco"]["large_reference_xy_screen"] is None
    assert rows["mu0p70"]["isaac"]["large_reference_xy_screen"] is False
    assert rows["mu0p70"]["mujoco"]["large_reference_xy_screen"] is False
    assert rows["mu1p00"]["isaac"]["large_reference_xy_screen"] is True
    assert rows["mu1p00"]["mujoco"]["large_reference_xy_screen"] is True
    assert rows["mu0p70"]["sim2sim"]["full_core_xy_rmse_m"] == pytest.approx(.0152982, abs=1e-6)


def test_interpolation_refuses_extrapolation():
    with pytest.raises(ValueError, match="extrapolate"):
        mod.interpolate(np.array([0., 1.]), np.zeros((2, 2)), np.array([1.1]))


def test_early_stop_is_censored_not_pass():
    time = np.array([0., .5, 1.0])
    ref_time = np.array([0., .5, 1., 1.5, 2.])
    ref_xy = np.zeros((5, 2))
    result = mod.path_error(time, np.zeros((3, 2)), ref_time, ref_xy, (.5, 1.5))
    assert result["core_complete"] is False
    assert result["full_core_reference_xy_rmse_m"] is None

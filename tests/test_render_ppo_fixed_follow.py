from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import render_ppo_fixed_follow as mod


def test_single_distance_all_cases_frames_no_zoom():
    targets = {"one": np.array([[0., 0., 0.], [20., 30., 0.]]), "two": np.array([[3., 5., 0.], [-5., 2., 0.]])}
    box = np.array([[-.5, -.5, -.5], [.5, .5, .5]])
    geometry = {key: [target + box for target in values] for key, values in targets.items()}
    basis = (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]))
    distance, audit = mod.fit_fixed_distance(geometry, targets, basis)
    assert isinstance(distance, float) and distance >= 1.3
    assert audit["all_mesh_bounds_inside"] is True
    assert audit["distance_changed_with_time"] is False
    assert audit["distance_differs_between_cases"] is False
    shifted = {key: [value + 100. for value in values] for key, values in geometry.items()}
    shifted_targets = {key: value + 100. for key, value in targets.items()}
    assert mod.fit_fixed_distance(shifted, shifted_targets, basis)[0] == pytest.approx(distance)


def test_apply_camera_only_translation_follows_different_roots():
    camera = NS(lookat=np.zeros(3), distance=0., azimuth=0., elevation=0.)
    runtime = NS(cameras=[camera])
    targets = {"a": np.array([[1., 2., .4], [5., 7., .6]]), "b": np.array([[9., 1., .2], [3., 2., .1]])}
    plan = {"lookat_by_case": targets, "distance_m": 1.8, "azimuth_deg": 135., "elevation_deg": -30.}
    for key in targets:
        for frame in range(2):
            returned = mod.apply_camera(runtime, plan, key, frame)
            assert np.array_equal(returned.lookat, targets[key][frame])
            assert returned.distance == 1.8
            assert (returned.azimuth, returned.elevation) == (135., -30.)


def test_worst_case_geometry_selects_one_larger_fixed_distance():
    basis = tuple(np.eye(3))
    targets = {"a": np.zeros((2, 3)), "b": np.zeros((2, 3))}
    geometry = {key: [np.array([[-.1, -.1, 0.], [.1, .1, 0.]])] * 2 for key in targets}
    original = mod.fit_fixed_distance(geometry, targets, basis)[0]
    geometry["b"] = [geometry["b"][0], np.array([[-2., -2., 0.], [2., 2., 0.]])]
    assert mod.fit_fixed_distance(geometry, targets, basis)[0] > original

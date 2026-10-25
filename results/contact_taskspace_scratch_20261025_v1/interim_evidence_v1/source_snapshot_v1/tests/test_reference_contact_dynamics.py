"""Force-allocation assumptions: single-wheel, cancellation and frame invariance."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from diagnose_reference_contact_dynamics import closest_load_allocation


def test_degenerate_equal_directions_cannot_hide_magnitude_error():
    force, split, error = closest_load_allocation([[0., 5.]], [[[0., .7], [0., .7]]], [100.])
    np.testing.assert_allclose(force, [[0., 70.]])
    np.testing.assert_allclose(split, [.5])
    np.testing.assert_allclose(error, [65.])


def test_single_wheel_and_unequal_support_allowed():
    directions = np.array([[[1., 0.], [0., 1.]]]*3)
    force, split, error = closest_load_allocation([[100., 0.], [25., 75.], [0., 100.]], directions, [100.]*3)
    np.testing.assert_allclose(force, [[100., 0.], [25., 75.], [0., 100.]])
    np.testing.assert_allclose(split, [1., .25, 0.]); np.testing.assert_allclose(error, 0.)


def test_opposite_slips_can_cancel_without_false_infeasibility():
    force, split, error = closest_load_allocation([[0., 0.]], [[[0., 1.], [0., -1.]]], [200.])
    np.testing.assert_allclose(force, 0.); np.testing.assert_allclose(split, .5); np.testing.assert_allclose(error, 0.)


def test_projection_clamps_and_is_frame_and_wheel_order_invariant():
    request = np.array([[10., -3.]])
    direction = np.array([[[1., 0.], [0., 1.]]])
    force, split, error = closest_load_allocation(request, direction, [2.])
    np.testing.assert_allclose(force, [[2., 0.]]); np.testing.assert_allclose(split, [1.])
    np.testing.assert_allclose(error, [np.sqrt(73.)])
    angle = .83; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    f2, s2, e2 = closest_load_allocation(request@rot.T, (direction@rot.T)[:, ::-1], [2.])
    np.testing.assert_allclose(f2, force@rot.T); np.testing.assert_allclose(s2, 1-split); np.testing.assert_allclose(e2, error)


@pytest.mark.parametrize("normal", [[0.], [-1.], [np.nan], [np.inf]])
def test_reject_invalid_total_normal_load(normal):
    with pytest.raises(ValueError):
        closest_load_allocation([[0., 0.]], [[[1., 0.], [0., 1.]]], normal)


def test_reject_nonfinite_vectors_and_misaligned_samples():
    with pytest.raises(ValueError):
        closest_load_allocation([[np.nan, 0.]], [[[1., 0.], [0., 1.]]], [1.])
    with pytest.raises(ValueError):
        closest_load_allocation([[0., 0.]], [[[1., 0.], [0., 1.]]]*2, [1.])

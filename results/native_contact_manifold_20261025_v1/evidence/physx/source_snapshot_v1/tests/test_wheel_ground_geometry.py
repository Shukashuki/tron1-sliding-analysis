"""Independent finite-cylinder geometry/frame checks; no dynamics or training."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from wheel_ground_geometry import cylinder_support, point_velocity, rotate


def test_horizontal_and_vertical_axle_use_correct_radius_and_width():
    a = cylinder_support([0, 0, .2], [0, 1, 0], .127, .025)
    assert a['authored_signed_gap_m'] == pytest.approx(.073)
    b = cylinder_support([0, 0, .2], [0, 0, 1], .127, .025)
    assert b['authored_signed_gap_m'] == pytest.approx(.175)
    np.testing.assert_allclose(b['authored_lowest_point_w_m'], [0, 0, .175])


def test_tilted_support_agrees_with_dense_independent_surface_search():
    axis = np.array([0., .8, .6]); center = np.array([.4, -.3, .2])
    radial1 = np.array([1., 0., 0.]); radial2 = np.cross(axis, radial1)
    angle = np.linspace(0, 2*np.pi, 10001)
    ring = .127*(np.cos(angle)[:, None]*radial1+np.sin(angle)[:, None]*radial2)
    vertices = np.concatenate([center+sign*.025*axis+ring for sign in (-1, 1)])
    actual = cylinder_support(center, axis, .127, .025)
    assert abs(actual['authored_signed_gap_m']-vertices[:, 2].min()) < 1e-8


def test_com_frame_choice_does_not_change_physical_point_velocity():
    point = np.array([.3, -.1, .2]); com = np.array([.2, 0., .3])
    velocity = np.array([1., 2., 3.]); omega = np.array([2., -1., .5]); shift = np.array([.04, -.02, .07])
    np.testing.assert_allclose(point_velocity(point, com, velocity, omega),
        point_velocity(point, com+shift, velocity+np.cross(omega, shift), omega), atol=1e-14)


def test_wxyz_rotation_preserves_actual_collider_translation():
    q = np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)])
    np.testing.assert_allclose(rotate(q, [0., .00707, 0.]), [-.00707, 0., 0.], atol=1e-14)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError): cylinder_support([0, 0, .2], [0, 2, 0], .127, .025)
    with pytest.raises(ValueError): rotate([2., 0., 0., 0.], [1, 0, 0])

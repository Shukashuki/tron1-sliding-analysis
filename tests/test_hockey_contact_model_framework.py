"""Analytic counterexamples for the offline friction diagnostics."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_hockey_contact_model_framework import (
    anchor_summary, cone_metrics, direction_consistency, dynamic_capacity_screen,
    material_velocities, opposition)


def test_inside_cone_does_not_imply_dissipative_direction():
    rho = cone_metrics(np.array([100.]), np.array([50.]), np.array([50.]), .7)
    assert rho['resultant'][0] < 1
    assert opposition([[50., 0., 0.]], [[1., 0., 0.]])[0] == -1


def test_resultant_hides_opposed_anchor_forces():
    # Opposite 80 N forces cancel, but their 160 N sum exceeds a 70 N budget.
    rho = cone_metrics(np.array([100.]), np.array([0.]), np.array([160.]), .7)
    assert rho['resultant'][0] == 0
    assert rho['anchor_norm_sum'][0] > 2


def test_zero_force_or_speed_is_undefined_not_a_direction_success():
    assert np.isnan(opposition([[0., 0., 0.]], [[1., 0., 0.]])[0])
    assert np.isnan(opposition([[1., 0., 0.]], [[0., 0., 0.]])[0])
    assert np.isnan(cone_metrics(np.array([0.]), np.array([0.]), np.array([0.]), .7)['resultant'][0])


def test_rigid_translation_is_exact_for_all_three_velocity_readouts():
    points = np.array([[0., 0., -.1], [0., .05, -.1]])
    v = np.array([1., .3, 0.]); dt = .0025
    velocities = material_velocities(points, np.zeros(3), v*dt, np.eye(3), np.eye(3),
                                     v, v, np.zeros(3), np.zeros(3), dt)
    for value in velocities: np.testing.assert_allclose(value, np.tile(v, (2, 1)))


def test_rolling_material_point_is_not_geometric_minimum_migration():
    dt = .0025; omega = np.array([0., 10., 0.]); v = np.array([1., 0., 0.])
    velocities = material_velocities(np.array([[0., 0., -.1]]), np.zeros(3), v*dt,
        np.eye(3), Rotation.from_rotvec(omega*dt).as_matrix(), v, v, omega, omega, dt)
    assert abs(velocities[0][0, 0]) < 1e-10
    assert abs(velocities[2][0, 0]) < .001
    assert velocities[1][0, 2] > 0  # leaving material point != contact loss


def test_reversal_is_separate_from_persistent_direction_sliding():
    pre = np.array([[1., 0., 0.]]); post = -pre; step = pre*.2
    assert direction_consistency(pre, post, step)[0] == -1
    a = np.array([[0., 0., 0., 0., 10., 1., 1., .2, 1., -1., 1., -10., 10., -2., -1., 1.]])
    summary = anchor_summary(a, .0025)
    assert summary['sliding_in_all_three_readouts_records'] == 1
    assert summary['sliding_direction_change_records'] == 1
    assert summary['coherent_sliding_records'] == 0


def test_orthogonal_friction_dissipates_no_power():
    np.testing.assert_allclose(opposition([[0., 2., 0.]], [[1., 0., 0.]]), [0.])


def test_dynamic_screen_requires_every_force_active_anchor_to_slide():
    a = np.array([[0., 0., 0., 0., 10., 1., 1., 1., 1., 1., 1., -10., -10., -10., 1., 1.],
                  [0., 0., 0., 1., 10., 0., 0., 0., np.nan, np.nan, np.nan, 0., 0., 0., np.nan, 1.]])
    r = dynamic_capacity_screen(a, np.array([True]), np.array([1.]), .0025)
    assert r['all_active_anchors_sliding_intervals'] == 0
    # Eligibility cannot be biased by selecting only dissipative forces.
    a = a[:1].copy(); a[:, 8:11] = -1.; a[:, 11:14] = 10.
    r = dynamic_capacity_screen(a, np.array([True]), np.array([1.2]), .0025)
    assert r['over_dynamic_1p05_intervals'] == 1

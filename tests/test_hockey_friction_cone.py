"""Meaningful geometric checks for unloaded and coupled force limits."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_hockey_friction_cone import cone_coordinates


def test_coupled_limit_and_directional_slice():
    # 3-4-5 triangle lies on the cone slice, although each component alone
    # is less than the limit. Reversing lateral force remains geometrically
    # possible, even with zero headroom in its current sign.
    r=cone_coordinates(np.array([3.]),np.array([4.]),np.array([10.]),mu=.5)
    np.testing.assert_allclose(r['rho'],[1.])
    np.testing.assert_allclose(r['lateral_same_sign_headroom_N'],[0.])
    np.testing.assert_allclose(r['delta_lateral_min_N'],[-8.])
    np.testing.assert_allclose(r['delta_lateral_max_N'],[0.])
    q=cone_coordinates(np.array([4.]),np.array([4.]),np.array([10.]),mu=.5)
    assert q['outside_tolerance'][0]


def test_unloaded_and_force_cancellation_are_not_certificates():
    r=cone_coordinates(np.array([0.,0.]),np.array([0.,0.]),np.array([0.,10.]),mu=.5)
    assert np.isnan(r['rho'][0]) and np.isnan(r['delta_lateral_max_N'][0])
    # Two opposite 3 N anchor forces cancel; a zero aggregate ratio cannot
    # prove both anchors fit hypothetical per-point 2.5 N limits.
    assert r['rho'][1]==0
    assert 3+3>.5*10


def test_same_mu_load_loss_shrinks_slice_not_angle():
    r=cone_coordinates(np.array([0.,0.]),np.array([4.,4.]),np.array([20.,10.]),mu=.5)
    np.testing.assert_allclose(r['rho'],[.4,.8])
    np.testing.assert_allclose(r['capacity_N'],[10.,5.])
    np.testing.assert_allclose(r['lateral_same_sign_headroom_N'],[6.,1.])

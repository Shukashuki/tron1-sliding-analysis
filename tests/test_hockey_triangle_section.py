"""A radial friction section must account for both tangential components."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plot_hockey_friction_from_start import section_coordinates


def test_combined_tangential_force_can_exceed_cone_when_each_component_does_not():
    # Both 60 N and 80 N fit below mu_s Fn = 90 N, but their 100 N
    # resultant does not. Projecting only lateral force would hide violation.
    x,y=section_coordinates(np.array([100.]),np.array([100.]),np.array([60.]),
                            np.array([80.]),np.array([True]),.9,'triangle')
    assert x[0]>.9*y[0]
    u,v=section_coordinates(np.array([100.]),np.array([100.]),np.array([60.]),
                            np.array([80.]),np.array([True]),.9,'circle')
    np.testing.assert_allclose(x/(.9*y),np.hypot(u,v))


def test_noncontact_or_zero_load_does_not_become_valid_cone_point():
    x,y=section_coordinates(np.array([0.,100.]),np.array([0.,70.]),
        np.array([0.,70.]),np.zeros(2),np.array([True,False]),.9,'triangle')
    assert np.isnan(x).all() and np.isnan(y).all()

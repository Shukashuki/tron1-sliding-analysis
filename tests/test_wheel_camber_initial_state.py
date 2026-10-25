"""Actual lowest-point kinematics for a finite cylinder at initial camber."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from probe_physx_wheel_geometry_controls import initial_state


@pytest.mark.parametrize("phi_deg",[-20.,-15.,0.,15.,20.])
def test_initial_spin_follows_axle_and_lowest_point_has_only_intended_lateral_slip(phi_deg):
    pose,velocity=initial_state(1.,20.,phi_deg)
    phi=np.deg2rad(phi_deg);axle=np.array([0.,np.cos(phi),np.sin(phi)])
    np.testing.assert_allclose(velocity[3:],axle*np.cos(np.deg2rad(20))/.127,atol=1e-14)
    normal=np.array([0.,0.,1.]);radial=normal-np.dot(normal,axle)*axle
    radial=-.127*radial/np.linalg.norm(radial)
    point=radial-.025*np.sign(axle[2])*axle
    assert pose[2]+point[2]==pytest.approx(1e-5)
    contact_velocity=velocity[:3]+np.cross(velocity[3:],point)
    np.testing.assert_allclose(contact_velocity,[0.,np.sin(np.deg2rad(20)),0.],atol=1e-14)
    assert np.linalg.norm(pose[3:])==pytest.approx(1.)


@pytest.mark.parametrize("speed,camber",[(0.,0.),(-1.,0.),(1.,21.),(1.,float("nan"))])
def test_invalid_initial_state_rejected(speed,camber):
    with pytest.raises(ValueError):initial_state(speed,20.,camber)

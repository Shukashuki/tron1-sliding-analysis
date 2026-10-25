"""Geometry and velocity tests for the prospective kinematic reference model."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import generate_camber_reference as ref
import independent_camber_reference as independent


@pytest.mark.parametrize("angle",[-35.,-10.,0.,10.,35.])
def test_exact_cylinder_support_and_axle_sign(angle):
    gamma=np.deg2rad(angle);axis=np.array([0.,np.cos(gamma),np.sin(gamma)])
    point,_=ref.cylinder_lowest_point([.2,.1,.7],axis,.127,.025)
    expected=.7-.127*abs(np.cos(gamma))-.025*abs(np.sin(gamma))
    assert point[2]==pytest.approx(expected,abs=1e-14)
    opposite,_=ref.cylinder_lowest_point([.2,.1,.7],-axis,.127,.025)
    np.testing.assert_allclose(opposite,point,atol=1e-14)


def test_line_to_rim_branch_is_explicit():
    flat,branch=ref.cylinder_lowest_point([0,0,.127],[0,1,0],.127,.025)
    assert branch==0 and flat[1]==0
    g=1e-6;tilted,branch=ref.cylinder_lowest_point([0,0,.127],[0,np.cos(g),np.sin(g)],.127,.025)
    assert branch==1 and tilted[1]==pytest.approx(-.025,abs=1e-6)


def test_bernstein_profile_bounded_and_closed():
    t=np.linspace(0,7,7001);c=np.array([[-15,-10,10,15],[15,10,-10,-15]])
    a=ref.camber_profile(t,c,[.4,1.44,4,5.2])
    assert abs(a).max()<=np.deg2rad(15)+1e-15
    np.testing.assert_array_equal(a[(t<=.4)|(t>=5.2)],0.)
    np.testing.assert_allclose(a[:,0],-a[:,1],atol=1e-15)


@pytest.mark.parametrize("coeff,timing",[(np.ones((2,3)),[.4,1.44,4,5.2]),(np.full((2,4),16),[.4,1.44,4,5.2]),(np.zeros((2,4)),[.4,4,1.44,5.2])])
def test_invalid_profiles_rejected(coeff,timing):
    with pytest.raises(ValueError):ref.camber_profile(np.arange(10),coeff,timing)


def test_differentiate_rotating_free_body():
    import mujoco
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    t=np.arange(10)*.02;q=np.zeros((10,7));q[:,0]=2*t;q[:,3]=np.cos(.3*t/2);q[:,6]=np.sin(.3*t/2)
    v=ref.differentiate(m,q,.02)
    np.testing.assert_allclose(v[:,0],2,atol=1e-14);np.testing.assert_allclose(v[:,5],.3,atol=1e-14)


def test_jacobian_is_material_velocity_not_motion_of_geometric_minimum():
    import mujoco
    # Tilting about X moves the lowest geometric candidate along a rim. Its
    # time derivative is not the material contact velocity J(q,p)*qdot.
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><joint axis="1 0 0"/><geom type="cylinder" size=".127 .025" quat=".7071067811865476 .7071067811865476 0 0"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m);d.qpos[0]=.2
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    axis=d.xmat[1].reshape(3,3)[:,1]
    p,_=ref.cylinder_lowest_point(d.geom_xpos[0],axis,.127,.025)
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,p,1)
    np.testing.assert_allclose(jp[:,0],np.cross([1,0,0],p-d.xpos[1]),atol=1e-14)


@pytest.mark.parametrize("hip,knee",[(.34,-.79),(.25,-.6),(.5,-1.)])
def test_planar_reconstruction_recovers_analytic_chain(hip,knee):
    thigh,shin=np.array([-.15,-.25981]),np.array([.15,-.25981])
    def rotate(v,t):return np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])@v
    point=rotate(thigh,hip)+rotate(shin,hip+knee)
    h,k=independent.planar_angles(point,thigh,shin,1)
    np.testing.assert_allclose([h,k],[hip,knee],atol=1e-14)


def test_unreachable_planar_target_is_not_clipped_to_success():
    with pytest.raises(ValueError,match="unreachable"):
        independent.planar_angles([2.,0.],[-.15,-.25981],[.15,-.25981],1)


def test_independent_profiles_do_not_force_same_magnitude_or_sign():
    t=np.linspace(0,7,351)
    coeff=np.array([[10,10,10,10],[-5,-5,-5,-5]])
    angles=ref.camber_profile(t,coeff,[.4,1.44,4,5.2])
    selected=(t>=1.44)&(t<=4)
    np.testing.assert_allclose(angles[selected,0],np.deg2rad(10),atol=1e-14)
    np.testing.assert_allclose(angles[selected,1],np.deg2rad(-5),atol=1e-14)

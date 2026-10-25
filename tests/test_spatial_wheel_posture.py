"""Check geometric reconstruction and frame conventions against authored FK."""
from pathlib import Path
import sys

import numpy as np
import pytest
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from spatial_wheel_posture import Geometry,axle_geometry


@pytest.fixture
def geometry():
    with np.load(ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz',allow_pickle=False) as a:
        source={k:a[k].copy() for k in a.files}
    model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml'))
    return Geometry(model,source)


@pytest.mark.parametrize('roll,pitch,abduction,height',[
    (0,0,[0,0],.75),(10,15,[5,-10],.65),(30,30,[0,20],.55),(-15,-15,[20,-20],.65)])
def test_pose_preserves_root_route_heading_and_both_geometric_supports(geometry,roll,pitch,abduction,height):
    g=geometry;frame=120;r,p=np.deg2rad([roll,pitch]);ab=np.deg2rad(abduction)
    q=g.posture(frame,r,p,ab,height);points,axes,_=g.wheel_geometry()
    np.testing.assert_array_equal(q[:2],g.source['qpos_mujoco'][frame,:2])
    np.testing.assert_allclose(points[:,2],0.,atol=1e-12,rtol=0)
    root=g.data.xmat[g.model.body('base_Link').id].reshape(3,3)
    body_centers=(g.data.geom_xpos[g.geoms]-q[:3])@root
    np.testing.assert_allclose(body_centers[:,0],g.original_centers[:,0],atol=1e-12,rtol=0)
    yaw=2*np.arctan2(g.source['qpos_mujoco'][frame,6],g.source['qpos_mujoco'][frame,3])
    assert np.arctan2(root[1,0],root[0,0])==pytest.approx(yaw,abs=1e-12)
    Rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    expected,camber,heading=axle_geometry(r,p,ab)
    np.testing.assert_allclose(axes,expected@Rz.T,atol=1e-12,rtol=0)
    np.testing.assert_allclose(camber,np.arcsin(axes[:,2]),atol=1e-12,rtol=0)
    assert g.nonwheel_clearance()>0 and g.opposite_leg_clearance()>0
    assert g.data.time==0.


def test_yaw_only_posture_cannot_steer_by_changing_abduction():
    _,_,heading=axle_geometry(0.,0.,np.deg2rad([-20.,60.]))
    np.testing.assert_allclose(heading,0.,atol=0,rtol=0)
    _,_,coupled=axle_geometry(np.deg2rad(30),np.deg2rad(30),np.deg2rad([0,20]))
    assert coupled[0]<0 and coupled[1]<coupled[0] and abs(coupled[1])>np.deg2rad(25.)


def test_instantaneous_motion_spin_cancels_longitudinal_velocity_and_respects_world_angular_frame(geometry):
    g=geometry;q=g.posture(120,*np.deg2rad([30,30]),np.deg2rad([0,20]),.55)
    linear=np.array([.9,.6,0.]);omega=np.array([0.,0.,.6])
    v,slip,points,axes=g.instantaneous_surface_motion(q,linear,omega)
    for i,(point,body,axis) in enumerate(zip(points,g.wheels,axes)):
        forward=np.cross(axis,[0.,0.,1.]);forward/=np.linalg.norm(forward)
        assert abs(forward@slip[i])<1e-12
        assert abs(slip[i,2])<1e-12
        mujoco.mj_jac(g.model,g.data,g.jp,g.jr,g.data.xipos[body],body)
        com=g.jp@v;angular=g.jr@v
        np.testing.assert_allclose(slip[i],com+np.cross(angular,point-g.data.xipos[body]),atol=1e-12,rtol=0)
    assert g.data.time==0.


def test_original_posture_reconstruction_and_infeasible_inputs(geometry):
    g=geometry;q=g.posture(0,0.,0.,np.zeros(2),.75)
    np.testing.assert_allclose(q,g.source['qpos_mujoco'][0],atol=1e-12,rtol=0)
    with pytest.raises(ValueError):g.posture(0,0.,0.,np.deg2rad([0,60]),.75)
    with pytest.raises(ValueError):g.posture(0,0.,0.,np.zeros(2),1.5)
    with pytest.raises(ValueError):g.posture(0,float('nan'),0.,np.zeros(2),.75)


def test_recursive_inverse_force_matches_independent_mass_matrix_plus_bias(geometry):
    g=geometry;s=g.frozen_shape_inverse_state(120,*np.deg2rad([-30,-30]),np.zeros(2),.55)
    d=mujoco.MjData(g.model);d.qpos[:]=s['qpos'];d.qvel[:]=s['qvel'];mujoco.mj_forward(g.model,d)
    inertial=np.empty(g.model.nv);mujoco.mj_mulM(g.model,d,inertial,s['qacc'])
    np.testing.assert_allclose(s['required_generalized_force'],inertial+d.qfrc_bias,atol=1e-10,rtol=0)
    assert g.data.time==0 and d.time==0


def test_static_inverse_force_has_actual_weight_and_no_fabricated_horizontal_force(geometry):
    g=geometry;g.source['qvel_mujoco'][:]=0.
    s=g.frozen_shape_inverse_state(120,*np.deg2rad([30,30]),np.deg2rad([0,20]),.55)
    np.testing.assert_allclose(s['required_generalized_force'][:3],[0.,0.,float(g.model.body_mass.sum())*9.81],atol=1e-10,rtol=0)
    assert not s['qvel'].any() and not s['qacc'].any()


def test_independent_fore_aft_placement_preserves_both_floor_supports(geometry):
    from fit_spatial_wheel_postures import Placement
    g=Placement(geometry.model,geometry.source);shift=np.array([.03,-.04]);g.set_longitudinal_shift(shift)
    q=g.posture(120,*np.deg2rad([10,15]),np.deg2rad([5,-10]),.65)
    root=g.data.xmat[g.model.body('base_Link').id].reshape(3,3)
    center=(g.data.geom_xpos[g.geoms]-q[:3])@root
    np.testing.assert_allclose(center[:,0],g.original_centers[:,0]+shift,atol=1e-12,rtol=0)
    np.testing.assert_allclose(g.wheel_geometry()[0][:,2],0.,atol=1e-12,rtol=0)


def test_failed_placement_measure_restores_unshifted_geometry_inputs(geometry):
    from fit_spatial_wheel_postures import Placement,measure
    g=Placement(geometry.model,geometry.source)
    with pytest.raises(ValueError):measure(g,120,[0,0,0,0,1.5,.1,-.1],.7,220.,.01)
    np.testing.assert_array_equal([r['x'] for r in g.links],g.original_longitudinal)

"""Time-coupled dynamics,unilateral rim geometry and sparse-derivative coverage."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import mujoco

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from hybrid_contact_reference import Problem,RESIDUAL_DIM,DIM,dc_bounds,rim_points
from generate_camber_reference import cylinder_lowest_point


@pytest.fixture
def problem():
    with np.load(ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz',allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml'))
    config=json.loads((ROOT/'config/hybrid_contact_reference_search_v2.json').read_text())
    return Problem(model,source,config)


def test_basis_partition_and_nonnegative_normal_force_coefficients(problem):
    p=problem;np.testing.assert_allclose(p.B.sum(-1),1.,atol=1e-14,rtol=0)
    assert p.B.shape==(349,12) and p.C.shape==(12,DIM)
    values,q,v,a=p.poses(p.C)
    np.testing.assert_array_equal(q[:,:2],p.source['qpos_mujoco'][:,:2])
    np.testing.assert_allclose(q[:,3:7],p.source['qpos_mujoco'][:,3:7],atol=1e-15,rtol=0)
    assert (values[:,[13,16,19,22]]>0.).all()


def test_two_rims_recover_set_valued_support_without_a_sign_switch():
    center=np.array([0.,0.,.2]);radius,halfwidth=.127,.025
    for angle in [0.,-1e-8,1e-8,-.4,.4]:
        axle=np.array([0.,np.cos(angle),np.sin(angle)])
        points=rim_points(center,axle,radius,halfwidth)
        lowest,_=cylinder_lowest_point(center,axle,radius,halfwidth)
        assert points[:,2].min()==pytest.approx(lowest[2],abs=1e-12)
    zero=rim_points(center,np.array([0.,1.,0.]),radius,halfwidth)
    near=rim_points(center,np.array([0.,np.cos(1e-8),np.sin(1e-8)]),radius,halfwidth)
    assert abs(zero-near).max()<2e-9
    assert zero[0,2]==zero[1,2]


def test_dc_braking_envelope_matches_nominal_semantics_without_clipping_state():
    v=np.array([0.,15.,30.,-15.,-30.]);before=v.copy();lo,hi=dc_bounds(v,80.,15.)
    np.testing.assert_array_equal(v,before)
    np.testing.assert_allclose(lo,[-80,-80,-80,0,80],atol=0,rtol=0)
    np.testing.assert_allclose(hi,[80,0,-80,80,80],atol=0,rtol=0)


def test_complete_changing_shape_inverse_force_and_point_wrench(problem):
    p=problem;c=p.C.copy();c[5,0]=.08;c[6,2]=.72;c[5,3]=.03
    residual,state=p.evaluate(c,save=True)
    assert residual.shape==(349,RESIDUAL_DIM) and np.isfinite(residual).all()
    assert np.max(abs(state['qacc'][:,3:6]))>.05
    frame=160;d=mujoco.MjData(p.model);d.qpos[:]=state['qpos'][frame];d.qvel[:]=state['qvel'][frame]
    mujoco.mj_forward(p.model,d);force=np.empty(p.model.nv);mujoco.mj_mulM(p.model,d,force,state['qacc'][frame]);force+=d.qfrc_bias
    np.testing.assert_allclose(force,state['required'][frame],atol=1e-10,rtol=0)
    forces=state['force_w_n'][frame];points=state['points'][frame];root=state['qpos'][frame,:3]
    rotation=d.xmat[p.model.body('base_Link').id].reshape(3,3)
    direct=np.r_[forces.sum(0),rotation.T@np.cross(points-root,forces).sum(0)]
    np.testing.assert_allclose(direct,state['contact'][frame,:6],atol=1e-10,rtol=0)
    assert p.data.time==0 and d.time==0


@pytest.mark.parametrize('column',[0,3,9,11,13,22])
def test_declared_jacobian_sparsity_covers_actual_trajectory_dependencies(problem,column):
    p=problem;base=p.C.copy();changed=base.copy();changed[6,column]+=1e-5
    initial=p.evaluate(base).ravel();actual=p.evaluate(changed).ravel()
    full_index=6*DIM+column;free_indices=np.flatnonzero(p.free.ravel());index=np.flatnonzero(free_indices==full_index).item()
    allowed=p.sparsity()[:,index].toarray().ravel().astype(bool)
    assert np.max(abs(actual[~allowed]-initial[~allowed]))<1e-12
    assert np.max(abs(actual[allowed]-initial[allowed]))>0

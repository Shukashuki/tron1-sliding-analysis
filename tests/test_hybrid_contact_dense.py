"""Analytic body-frame angular derivatives and original car-node preservation."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import mujoco

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_hybrid_contact_dense import AnalyticProblem


@pytest.fixture
def problem():
    with np.load(ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz',allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml'))
    cfg=json.loads((ROOT/'config/hybrid_contact_reference_search_v3.json').read_text())
    return AnalyticProblem(model,source,cfg)


def test_dense_root_interpolation_preserves_original_nodes_and_derivatives(problem):
    p=problem;t=p.source['time_s'];q=p.source['qpos_mujoco'];v=p.source['qvel_mujoco']
    assert len(p.time)==2785
    np.testing.assert_allclose(p.root_curve(t)[:,:2],q[:,:2],atol=1e-12,rtol=0)
    np.testing.assert_allclose(p.root_curve.derivative(1)(t),v[:,[0,1,5]],atol=1e-11,rtol=0)
    assert np.max(abs(p.root_curve(t)[1:,2]-np.unwrap(2*np.arctan2(q[:,6],q[:,3]))[1:]))<1e-12


def test_exact_changing_shape_velocities_and_accelerations_match_pose_finite_differences(problem):
    p=problem;c=p.C.copy();c[5,0]=.14;c[6,1]=-.18;c[5,2]=.71;c[6,3]=.1
    for time in [1.73,2.37,3.21,4.91]:
        eps=1e-5
        _,q,v,a=p.kinematics(c,[time-eps,time,time+eps]);finite=np.empty(p.model.nv)
        mujoco.mj_differentiatePos(p.model,finite,2*eps,q[0],q[2])
        np.testing.assert_allclose(v[1],finite,atol=2e-7,rtol=0)
        np.testing.assert_allclose(a[1],(v[2]-v[0])/(2*eps),atol=2e-6,rtol=0)

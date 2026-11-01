"""Support-height derivatives and geometry at off-grid times."""
from pathlib import Path
import json
import sys

import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_hybrid_contact_support_dense import DenseSupportProblem


def test_derived_support_height_and_normal_velocity_are_consistent_with_pose_derivative():
    with np.load(ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz',allow_pickle=False) as f:s={k:f[k].copy() for k in f.files}
    m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml'))
    cfg=json.loads((ROOT/'config/hybrid_contact_reference_search_v4.json').read_text())
    p=DenseSupportProblem(m,s,cfg,np.tile([0,0,.25]*4,(349,1)))
    c=p.C.copy();c[5,0]=.1;c[6,4]=-.12;c[5,7]=.9
    for time in [1.73,2.37,3.21,4.91]:
        eps=1e-5;_,q,v,a=p.kinematics(c,[time-eps,time,time+eps]);finite=np.empty(m.nv)
        mujoco.mj_differentiatePos(m,finite,2*eps,q[0],q[2])
        np.testing.assert_allclose(v[1],finite,atol=2e-7,rtol=0)
        d=mujoco.MjData(m);d.qpos[:]=q[1];mujoco.mj_kinematics(m,d)
        geom=p.geoms[p.rolling];axle=d.xmat[p.wheel_bodies[p.rolling]].reshape(3,3)[:,1]
        radius,width=m.geom_size[geom,:2]
        gap=d.geom_xpos[geom,2]-radius*np.sqrt(1-axle[2]**2)-width*abs(axle[2])
        assert abs(gap)<1e-12

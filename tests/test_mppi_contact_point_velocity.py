"""Check material point velocity against an independent MuJoCo Jacobian."""
from pathlib import Path
import sys,json
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import generate_mppi_contact_reference as g


def test_twist_cross_product_contact_velocity_matches_point_jacobian():
    a=g.parser().parse_args(['--output-dir','unused'])
    m=g.load_model(a.model,json.loads(a.material_report.read_text()));d=mujoco.MjData(m)
    with np.load(g.ROOT/'outputs/mppi_contact_reference_20261025_v1/independent_v1/rollout.npz') as f:
        i=int(np.flatnonzero(f['ground_normal_force_w'][...,2].max(-1)>5)[0])
        d.qpos[:]=f['qpos'][i];d.qvel[:]=f['qvel'][i]
    mujoco.mj_forward(m,d);wheels={m.body('wheel_'+s+'_Link').id for s in ('L','R')};tested=0
    for contact in d.contact:
        for geom in contact.geom:
            body=int(m.geom_bodyid[geom])
            if body not in wheels:continue
            twist=np.empty(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,twist,0)
            expected=twist[3:]+np.cross(twist[:3],contact.pos-d.xipos[body])
            jp,jr=np.empty((3,m.nv)),np.empty((3,m.nv));mujoco.mj_jac(m,d,jp,jr,contact.pos,body)
            np.testing.assert_allclose(expected,jp@d.qvel,atol=1e-12,rtol=0);tested+=1
    assert tested>0


def test_actual_replay_forces_obey_native_pyramid_and_its_inscribed_circle_bound():
    for name in ('independent_audit_v2','smooth_audit_v1'):
        with np.load(g.ROOT/'results/mppi_contact_reference_20261025_v1'/name/'contact_audit.npz') as f:
            valid=f['loaded_point_count']>0
            assert valid.any()
            l1=f['pyramidal_friction_utilization_max'][valid]
            l2=f['euclidean_friction_utilization_max'][valid]
            assert np.max(l1)<=1+1e-10
            assert np.all(l2<=l1+1e-12)

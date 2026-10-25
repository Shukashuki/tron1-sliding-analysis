"""Native parity and reference-planning contracts."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import generate_mppi_contact_reference as g


@pytest.fixture
def setup():
    a=g.parser().parse_args(['--output-dir','unused-test-path'])
    report=json.loads(a.material_report.read_text());cfg=json.loads(a.config.read_text())
    m=g.load_model(a.model,report)
    with np.load(a.reference,allow_pickle=False) as f:s={k:f[k].copy() for k in f.files}
    p=g.Planner(None,m,s,cfg);d=mujoco.MjData(m)
    d.qpos[:]=report['initial_state']['qpos'];d.qvel[:]=report['initial_state']['qvel'];mujoco.mj_forward(m,d)
    return a,m,d,p


def test_native_first_interval_matches_continuous_torque_loop(setup):
    a,m,d,p=setup;action=np.array([.03,-.01,.02,-.03,.01,-.02,.2,-.15]);state=np.r_[d.qpos,d.qvel]
    contract=dict(g.ACTUATOR_SETTINGS,actuator_model='dc_motor',physics_dt=.0025,decimation=8)
    with g.NativeBatchTransition(m,p.q0,contract,a.native_library) as native:
        raw=native.rollout(state[:15][None],state[15:][None],action[None,None],initial_history=np.zeros((1,3,9,3)))
    aids=[m.actuator(n).id for n in g.JOINT_NAMES]
    for i in range(8):
        tau=g.torque(m,d,p.q0,action);d.ctrl[aids]=tau;mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.qpos,raw['qpos'][0,0,i],atol=1e-12,rtol=0)
        np.testing.assert_allclose(d.qvel,raw['qvel'][0,0,i],atol=1e-12,rtol=0)
        np.testing.assert_allclose(tau,raw['torques'][0,0,i],atol=1e-12,rtol=0)
        np.testing.assert_allclose(g.ground_contact_readout(m,d)['ground_normal_force_w'],raw['ground_normal_force_w'][0,0,i],atol=1e-9,rtol=0)
    assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)


def synthetic(p):
    r=dict(qpos=np.tile(p.source['qpos_mujoco'][0],(1,1,8,1)),
        body_lin_vel_w=np.zeros((1,1,8,9,3)),body_ang_vel_w=np.zeros((1,1,8,9,3)),
        body_quat_w=np.tile([1.,0,0,0],(1,1,8,9,1)),body_net_forces_w=np.zeros((1,1,8,9,3)),
        ground_normal_force_w=np.zeros((1,1,8,2,3)))
    r['body_lin_vel_w'][...,0,:]=p.source['body_lin_vel_w'][0,0]
    r['ground_normal_force_w'][...,0,2]=.5*p.weight
    return r


def test_support_allows_one_wheel_but_detects_single_substep_gap(setup):
    _,_,_,p=setup;r=synthetic(p);a=np.zeros((1,1,8));z=np.zeros(8)
    _,terms=p.cost(r,a,0,z,0,return_terms=True)
    assert terms['support_mean'].item()==0 and terms['support_worst'].item()==0
    r['ground_normal_force_w'][0,0,3]=0
    _,terms=p.cost(r,a,0,z,0,return_terms=True)
    assert terms['support_mean'].item()==pytest.approx(1/8) and terms['support_worst'].item()==1


def test_cost_has_no_absolute_height_or_wheel_position_target(setup):
    _,_,_,p=setup;r=synthetic(p);a=np.zeros((1,1,8));z=np.zeros(8)
    before=p.cost(r,a,0,z,0);r['qpos'][...,2]+=1.5;after=p.cost(r,a,0,z,0)
    np.testing.assert_array_equal(before,after)
    assert 'body_pos_w' not in r


def test_axle_camber_sign_and_wheel_heading_are_distinct():
    from spatial_wheel_posture import root_quaternion,axle_geometry
    roll,pitch,yaw=np.deg2rad([10,15,40]);ab=np.deg2rad([5,-10])
    q=np.array([root_quaternion(yaw,roll+alpha,pitch) for alpha in ab])
    camber,heading=g.wheel_angles(q);_,ec,eh=axle_geometry(roll,pitch,ab)
    np.testing.assert_allclose(camber,ec,atol=1e-12);np.testing.assert_allclose(g.angle(heading-yaw),eh,atol=1e-12)
    assert abs(camber[0]-camber[1])>np.deg2rad(10)


def test_export_recomputes_imu_and_all_six_reference_streams(setup,tmp_path):
    _,m,d,p=setup;rows=[]
    for i in range(17):
        d.qpos[2]=.7+.002*i;d.qvel[2]=.1;mujoco.mj_forward(m,d);d.time=i*.0025
        rows.append(g.record(m,d,np.zeros(8)))
    raw={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    path=tmp_path/'candidate.npz';g.export_motion(m,raw,p.source,path)
    with np.load(path) as f:
        assert f['body_pos_w'].shape==(2,10,3)
        np.testing.assert_allclose(f['qpos_mujoco'][:,2],[.7,.716]);np.testing.assert_allclose(f['body_pos_w'][:,0,2],[.7,.716])
        np.testing.assert_allclose(f['body_lin_vel_w'][:,0,2],.1)
        assert not any('force' in k for k in f.files)
        imu=list(f['body_names']).index('limx_imu')
        np.testing.assert_allclose(f['body_pos_w'][:,imu,2]-f['body_pos_w'][:,0,2],0,atol=1e-10)

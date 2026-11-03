"""Physical-clock, paired-state and safe handoff regressions for the speed study."""
import copy
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from paired_stop_speed_reference import reference_pair, scaled_reset
from score_paired_stop_speed import boundary
from dispatch_paired_stop_speed import check_identity, process_identity, inspect
from analyze_direct_brake_comparison import analyze_case
from prepare_paired_stop_speed import build


@pytest.fixture(scope='module')
def original():
    source=ROOT/'results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2'
    model=mujoco.MjModel.from_xml_path('/home/roy/tron1-learning-from-human/assets/robots/WF_TRON1A/mujoco/robot.xml')
    return model,dict(np.load(source/'motion.npz')),dict(np.load(source/'command_diagnostics.npz'))


@pytest.mark.parametrize('scale',[1.5,2.,2.5,3.])
def test_fixed_distance_fk_and_com_velocity_consistency(original,scale):
    model,old,diag=original
    pair,geometry,meta=reference_pair(model,old,diag,scale)
    assert meta['endpoint_s']-meta['brake_end_s']==pytest.approx(.96)
    assert 0<=meta['brake_start_s']-meta['kinematic_brake_start_s']<.02
    np.testing.assert_array_equal(pair['direct']['body_pos_w'][:,0,:2],pair['hockey']['body_pos_w'][:,0,:2])
    for behavior,motion in pair.items():
        np.testing.assert_allclose(motion['body_pos_w'][-1,0,:2],old['body_pos_w'][-1,0,:2],atol=1e-10)
        assert np.max(abs(motion['joint_vel'][:,:6]))<15
        assert np.isfinite(motion['body_lin_vel_w']).all()
        np.testing.assert_allclose(motion['body_lin_vel_w'][-1],0,atol=1e-8)
        # Independently check the full rigid-body Jacobian relation J(q) qdot.
        # A 50 Hz secant is not an instantaneous velocity after time compression.
        data=mujoco.MjData(model)
        ids=[model.body('base_Link' if str(n)=='limx_imu' else str(n)).id for n in old['body_names']]
        jids=[model.jnt_dofadr[model.joint(str(n)).id] for n in old['joint_names']]
        for i,q in enumerate(geometry[behavior]['qpos']):
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            omega=motion['body_ang_vel_w'][i,0];v=np.zeros(model.nv)
            v[:3]=motion['body_lin_vel_w'][i,0]-np.cross(omega,data.xipos[ids[0]]-data.xpos[ids[0]])
            v[3:6]=data.xmat[ids[0]].reshape(3,3).T@omega
            v[jids]=motion['joint_vel'][i]
            for k,bid in enumerate(ids):
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacBodyCom(model,data,jp,jr,bid)
                np.testing.assert_allclose(jp@v,motion['body_lin_vel_w'][i,k],atol=2e-5,rtol=1e-5)
                np.testing.assert_allclose(jr@v,motion['body_ang_vel_w'][i,k],atol=2e-5,rtol=1e-5)
    np.testing.assert_allclose(pair['direct']['joint_vel'],0,atol=1e-10)


def test_common_state_scales_all_velocities_only():
    state=dict(np.load(ROOT/'results/hockey_equal_distance_20261102_v1/common_evaluation_initial_state.npz'))
    new=scaled_reset(state,2.5)
    keys={'root_com_vel','root_omega','joint_vel','body_com_vel','body_omega'}
    for k in state:np.testing.assert_array_equal(new[k],state[k]*2.5 if k in keys else state[k])
    assert np.linalg.norm(new['root_com_vel'][:2])==pytest.approx(3.25822,abs=1e-5)


def test_boundary_symmetric_and_incomplete_pair_rejected():
    assert boundary([33,29]) and boundary([29,33]) and boundary([0,0])
    assert not boundary([30,33])
    for counts in ([33],[33,34],[33,None],[33,True]):
        with pytest.raises(ValueError):boundary(counts)


def test_process_identity_guards_against_pid_reuse_and_live_child():
    actual=process_identity(os.getpid());check_identity(actual,actual)
    for field in ('uid','start_ticks','pid','argv'):
        changed=copy.deepcopy(actual)
        changed[field]=['changed'] if field=='argv' else actual[field]+1
        with pytest.raises(ValueError):check_identity(actual,changed,zombie=True)
    zombie=dict(actual,state='Z',argv=[])
    check_identity(zombie,actual,zombie=True)
    with pytest.raises(ValueError):check_identity(dict(zombie,state='R'),actual,zombie=True)


def test_old_nominal_metrics_unchanged_by_phase_parameterization():
    base=ROOT/'results/direct_brake_comparison_20261103_v1'
    if not (base/'evidence/evaluation/nominal/final_rollout.npz').exists():
        pytest.skip('Integration regression requires the archived raw evaluation')
    ref=ROOT/'results/hockey_equal_distance_20261102_v1/direct/motion.npz'
    budget=json.loads((ROOT/'results/hockey_equal_distance_20261102_v1/report.json').read_text())['budget']
    # Frozen old low-speed data remains a regression target, not a new trial.
    if not ref.exists():
        ref=ROOT/'results/direct_brake_training_20261102_v1/source_snapshot_v1/motion.npz'
    row,_=analyze_case(base/'evidence/evaluation/nominal',ref,budget)
    assert row['strict_direct_task_pass']
    assert row['angular_xy_10_100hz_rms_rad_s']==pytest.approx(.006079966,rel=1e-5)
    with pytest.raises(ValueError,match='phase clock'):
        analyze_case(base/'evidence/evaluation/nominal',ref,budget,brake_start=1.8,brake_end=.4)


def test_frozen_bundle_budget_and_reward_pairing(tmp_path):
    published=ROOT/'results/paired_stop_speed_20261103_v1/source_snapshot_v1'
    archived=json.loads((published/'plan.json').read_text())
    build(tmp_path/'study',archived['predecessor']['handoff_sha256'])
    bundle=tmp_path/'study/source_snapshot_v1'
    plan,_=inspect(bundle,verify_runtime=False)
    for level in plan['levels']:
        for stage in ('parent','contact'):
            direct=json.loads((bundle/f'protocols/{level["id"]}_direct_{stage}.json').read_text())
            hockey=json.loads((bundle/f'protocols/{level["id"]}_hockey_{stage}.json').read_text())
            assert direct['common_evaluation_initial_state']==hockey['common_evaluation_initial_state']
            assert direct['formal']==hockey['formal']
            for key in ('reference_sha256','speed_comparison'):direct[key]=hockey[key]
            direct['reward_weights']['beta']=hockey['reward_weights']['beta']
            if 'bilateral_sliding' in direct:
                direct['bilateral_sliding']['weights']['sliding']=hockey['bilateral_sliding']['weights']['sliding']
            assert direct==hockey

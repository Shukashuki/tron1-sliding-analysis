from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from eval_ppo_k_leg_ramp import NumpyRamp,materialize_mujoco,materialize_physx,save_targets


def test_numpy_native_targets_exact_endpoints_and_measured_reset_start():
    r=NumpyRamp();q=np.arange(8)*.03;q0=np.zeros(8)
    contract={'leg_position_scale':1.,'raw_action_clip':[-1.,1.],'wheel_torque_scale_nm':12.}
    for frame,action in enumerate([np.ones(8)*.5,np.ones(8)*-.3]):
        for substep in range(8):target=r.target(frame,substep,action,q,q0,contract)
        np.testing.assert_array_equal(target,action[:6])
    np.testing.assert_array_equal(r.rows[0]['start'],q[:6])
    np.testing.assert_array_equal(r.rows[8]['start'],np.full(6,.5))
    np.testing.assert_array_equal(r.rows[0]['wheel_effort'],[6.,6.])
    with pytest.raises(AssertionError):r.target(2,1,np.zeros(8),q,q0,contract)


def test_native_integration_changes_only_explicit_leg_target_before_motor_clipping():
    path=ROOT/'outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_mujoco.py'
    source=path.read_text();new=materialize_mujoco(source)
    assert new.count('mujoco.mj_step(model, data)')==source.count('mujoco.mj_step(model, data)')
    assert 'leg_target_override=actual_target' in new
    for line in ['    torque[6:] = contract["wheel_torque_scale_nm"] * action[6:]',
                 '        torque = np.clip(torque, lower, upper)']:
        assert new.count(line)==source.count(line)==1
    with pytest.raises(AssertionError):materialize_mujoco('pass\n')


def test_physx_installs_K_owner_after_native_recorder():
    p=ROOT/'outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_friction_isaac.py'
    result=materialize_physx(p.read_text())
    assert result.index('env.scene.update = physics_scene_update')<result.index('_leg_ramp_install(env, _new_plan)')
    assert result.count('_leg_ramp_install(env, _new_plan)')==1


def test_complete_record_verifies_window_fractions_and_rejects_corruption(tmp_path):
    r=NumpyRamp();contract={'leg_position_scale':1.,'raw_action_clip':[-1.,1.],'wheel_torque_scale_nm':12.}
    for frame in range(349):
        for substep in range(8):r.target(frame,substep,np.full(8,np.sin(frame*.02)),np.zeros(8),np.zeros(8),contract)
    proof=save_targets(r.rows,tmp_path/'good.npz',names=['a','b','c','d','e','f'])
    assert proof['native_target_applications']==2792
    r.rows[200]['actual'][1]+=.01
    with pytest.raises(AssertionError):save_targets(r.rows,tmp_path/'bad.npz',names=['a','b','c','d','e','f'])

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from classical_tracking import affine_riccati, Predictor, ShootingNMPC, load_pinned_evaluator


def test_affine_lqr_accounts_for_unreachable_reference_defect():
    A=B=np.ones((1,1,1))
    K,feed,P=affine_riccati(A,B,np.ones((1,1)),np.eye(1),np.eye(1),np.eye(1))
    assert K[0,0,0]==pytest.approx(.5)
    assert feed[0,0]==pytest.approx(-.5)
    assert P[0,0,0]==pytest.approx(1.5)


def test_fast_torques_match_original_dc_motor_helper():
    api=load_pinned_evaluator()
    contract=api.read_contract()
    qids=np.arange(7,15);vids=np.arange(6,14)
    p=SimpleNamespace(qids=qids,vids=vids,q0=np.zeros(8),effort=np.r_[[80.]*6,[12.]*2],no_load=np.r_[[15.]*6,[100.]*2])
    rng=np.random.default_rng(42)
    for _ in range(50):
        q=np.r_[np.zeros(7),rng.normal(0,.3,8)];v=np.r_[np.zeros(6),rng.normal(0,150,8)];a=rng.uniform(-2,2,8)
        expected=api.action_torques(a,q[7:],v[6:],p.q0,contract)
        np.testing.assert_allclose(Predictor.torque(p,a,q,v),expected,atol=1e-12)


def test_sampling_nmpc_is_bounded_reproducible_and_keeps_best_candidate(tmp_path):
    settings={'seed':42,'nmpc':{'horizon_steps':3,'population':4,'iterations':2,'elite':2,
                              'control_blocks':1,'leg_noise_std_rad':.1,'wheel_noise_std_normalized':.2}}
    p=SimpleNamespace(n=10,settings=settings,targets=np.zeros((10,2)),nominal_action=np.zeros((10,8)),Q=np.eye(2),R=np.eye(8))
    p.error=lambda a,b:b-a
    p.transition=lambda x,u:x+np.array([u[0],0])
    path=tmp_path/'design.npz'
    np.savez(path,targets=p.targets,nominal_action=p.nominal_action,K=np.zeros((10,8,2)),
             feed=np.tile(np.r_[-.5,np.zeros(7)],(10,1)),P=np.tile(np.eye(2),(11,1,1)),settings_json=np.array(json.dumps(settings)))
    a,info=ShootingNMPC(p,path).action(np.array([1.,0.]),0)
    b,_=ShootingNMPC(p,path).action(np.array([1.,0.]),0)
    np.testing.assert_array_equal(a,b)
    assert np.max(np.abs(a))<=1
    assert info['predicted_cost_nonincreasing'] is True

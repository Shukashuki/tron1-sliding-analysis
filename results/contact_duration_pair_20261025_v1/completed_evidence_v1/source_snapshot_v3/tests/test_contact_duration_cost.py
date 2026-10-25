"""Temporal cost discriminability,matched scale,and real eight-sample lifecycle."""
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import copy
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"tests"))
import ground_contact_reward_v2k as k
import ground_contact_reward_v2s as s
import ground_contact_reward_v2t as t
import ppo_ground_contact_v2s as owner_s
import ppo_ground_contact_v2t as owner_t
import test_ppo_ground_contact_v2k as fixture


def forces(gaps=0,load=100.):
    a=np.zeros((8,2));a[:,0]=load;a[:gaps]=0.
    return a


@pytest.mark.parametrize("load",[0.,5.,5.01,50.,100.,200.])
def test_uniform_force_windows_match_at_all_load_scales(load):
    args=(forces(load=load),200.,.7,.8,.9,1.)
    assert s.reward_numpy(*args)["raw_reward"]==pytest.approx(t.reward_numpy(*args)["raw_reward"],abs=1e-14)


def test_repair_credit_when_another_gap_remains_and_beta_gate_zero():
    for gaps in range(2,9):
        args=(forces(gaps),200.,0.,0.,0.,1.)
        repaired=(forces(gaps-1),200.,0.,0.,0.,1.)
        assert s.reward_numpy(*repaired)["raw_reward"]-s.reward_numpy(*args)["raw_reward"]==pytest.approx(.125)
        for module in [k,t]:assert module.reward_numpy(*repaired)["raw_reward"]==module.reward_numpy(*args)["raw_reward"]


def test_matched_bounds_no_bilateral_requirement_and_positive_bonus_unchanged():
    for gaps in range(9):
        args=(forces(gaps),200.,.7,.8,.9,1.)
        original=k.reward_numpy(*args)
        for module in [s,t]:
            r=module.reward_numpy(*args)
            assert 0<=r["support_gap_penalty"]<=2
            assert -2<=r["raw_reward"]<=1
            assert r["G_gated_support_bonus"]==original["G_gated_support_bonus"]
            flipped=module.reward_numpy(forces(gaps)[:,::-1].copy(),*args[1:])
            assert r["raw_reward"]==flipped["raw_reward"]


@pytest.mark.parametrize("module",[s,t])
def test_real_torch_parity_and_nonfinite_poison(module):
    rng=np.random.default_rng(42);f=rng.uniform(0,200,(5,8,2));weight=np.array([150.,170.,200.,220.,250.])
    a=module.reward_numpy(f,weight,.4,.7,.9,1.)
    b=module.reward_torch(torch.tensor(f),torch.tensor(weight),.4,.7,.9,1.)
    for field in a:np.testing.assert_allclose(b[field].numpy(),a[field],atol=1e-12,rtol=1e-12)
    bad=torch.tensor(f,dtype=torch.float32);bad[1,3,0]=float('nan')
    c=module.reward_torch(bad,torch.tensor(weight),.4,.7,.9,1.,validate=False)
    assert torch.isnan(c["raw_reward"][1]) and torch.isfinite(c["raw_reward"][[0,2,3,4]]).all()


@pytest.mark.parametrize("module",[s,t])
def test_no_reward_host_scalar_or_file_reads_and_settings_are_frozen(module):
    f=torch.tensor(forces(2),dtype=torch.float32)
    with patch.object(Path,"read_bytes",side_effect=AssertionError("Hot reward file read")),patch.object(torch.Tensor,"item",side_effect=AssertionError("Host scalar read")):
        result=module.reward_torch(f,200.,1.,1.,1.,1.,validate=False)
    assert torch.isfinite(result["raw_reward"])
    bad=copy.deepcopy(module.DEFAULT_SETTINGS);bad["support_force_threshold_n"]=1.
    with pytest.raises(ValueError):module.validate_settings(bad)


@pytest.mark.parametrize("module,klass",[(owner_s,"GroundContactV2S"),(owner_t,"GroundContactV2T")])
def test_collector_eight_real_samples_repeated_reads_and_old_owner_rejection(module,klass):
    height=ModuleType("height_smoothness");height.anchor_xy_tracking=lambda *a:torch.ones(2)
    training=ModuleType("training");training.__path__=[]
    drift=ModuleType("training.tron1_drift");drift.speed_tracking=lambda *a,**k:torch.ones(2);drift.sideslip_tracking=lambda *a:torch.zeros(2)
    with patch.dict(sys.modules,{"height_smoothness":height,"training":training,"training.tron1_drift":drift}),patch.object(fixture,"contact",module):
        env=fixture.Env();owner=getattr(module,klass)(env,allow_cpu_fixture=True)
        try:
            env.read_twice=True;env.force_sequence[0].zero_();env.force_sequence[1].zero_()
            value=env.step();audit=owner.audit(health=True)
            assert audit["physics_samples"]==8 and audit["unique_reward_windows"]==1
            assert audit["sampled_environment_substeps"]==16 and audit["raw_contact_reward_bounds"]==[-2.,1.]
            assert not audit["bilateral_constraint"] and audit["device_finite_reward_health"]
            assert torch.equal(value,torch.full((2,),-1.25 if module is owner_s else -2.))
            opposite=owner_t if module is owner_s else owner_s;opclass="GroundContactV2T" if module is owner_s else "GroundContactV2S"
            with pytest.raises(ValueError,match="stack"):getattr(opposite,opclass)(env,allow_cpu_fixture=True)
        finally:owner.close()
        assert module.MARKER not in vars(env)

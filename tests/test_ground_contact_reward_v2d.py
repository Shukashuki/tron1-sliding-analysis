"""Pure algebra fixtures, not D policy results or evidence of feasible motion."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ground_contact_reward_v2d as reward


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_D_minus_C_identity_and_all_other_components_exact(dtype):
    rng=np.random.default_rng(42);force=rng.uniform(0.,80.,(2,3,8,2)).astype(dtype);force[0,0,3]=0.
    original=force.copy();mg=np.array([[100.],[200.]],dtype=dtype)
    c=reward.base.reward_numpy(force,mg,.8,.3);d=reward.reward_numpy(force,mg,.8,.3)
    for name in c.keys()-{"raw_reward","support","gated_support_bonus"}:np.testing.assert_array_equal(d[name],c[name])
    np.testing.assert_array_equal(d["support"],c["support_mean"])
    np.testing.assert_allclose(d["raw_reward"]-c["raw_reward"],c["tracking_gate"]*c["impact_gate"]*(c["support_mean"]-c["support_min"]),rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(force,original);assert d["raw_reward"].dtype==dtype


def test_packed_split_fixture_removes_min_cliff_not_discount_timing():
    grouped=np.ones((2,8));grouped[0,3:5]=0.
    split=np.ones((2,8));split[:,3]=0.
    values=[]
    for q in (grouped,split):
        f=np.stack((50*q,np.zeros_like(q)),axis=-1);values.append(reward.reward_numpy(f,100.,1.,1.)["raw_reward"])
    np.testing.assert_array_equal(values[0],[.5,1.]);np.testing.assert_array_equal(values[1],[.75,.75])
    assert values[0].sum()==values[1].sum()==1.5
    assert values[0]@np.array([1.,.99])==pytest.approx(1.49)
    assert values[1]@np.array([1.,.99])==pytest.approx(1.4925)


def test_unsupported_single_and_alternating_load_and_ungated_deficit():
    single=np.tile([100.,0.],(8,1));alternate=single.copy();alternate[1::2]=[0.,100.]
    for f in (single,alternate,np.full((8,2),50.)):
        assert reward.reward_numpy(f,100.,1.,1.)["raw_reward"]==1.
    for gate in (0.,1.):assert reward.reward_numpy(np.zeros((8,2)),100.,gate,gate)["raw_reward"]==-1.
    gap=single.copy();gap[3]=0.
    assert reward.reward_numpy(gap,100.,1.,1.)["raw_reward"]==.75
    assert reward.reward_numpy(gap,100.,0.,1.)["raw_reward"]==-.125


@pytest.mark.parametrize("shape", ((4,2),(7,2),(9,2),(8,3),(8,),(0,8,2)))
def test_empty_or_wrong_window_rejected(shape):
    with pytest.raises(ValueError):reward.reward_numpy(np.zeros(shape),100.,1.,1.)


@pytest.mark.parametrize("bad", (np.nan,np.inf,-.001))
def test_invalid_forces_rejected(bad):
    f=np.ones((8,2));f[0,0]=bad
    with pytest.raises(ValueError):reward.reward_numpy(f,100.,1.,1.)


@pytest.mark.parametrize("field,bad", (("mg",0.),("mg",np.nan),("anchor",-1.),("anchor",np.nan),("speed",2.),("speed",np.inf)))
def test_invalid_mass_and_tracking_rejected(field,bad):
    a={"mg":100.,"anchor":1.,"speed":1.};a[field]=bad
    with pytest.raises(ValueError):reward.reward_numpy(np.ones((8,2)),a['mg'],a['anchor'],a['speed'])


def test_bounds_tolerated_negative_settings_ancestors_and_no_hotpath_IO():
    r=reward.reward_numpy(np.full((8,2),-1e-4),100.,1.,1.);assert r['raw_reward']==-1.
    rng=np.random.default_rng(1);r=reward.reward_numpy(rng.uniform(0,1e4,(1000,8,2)),100.,rng.random(1000),rng.random(1000))
    assert ((r['raw_reward']>=-1)&(r['raw_reward']<=1)).all()
    for field,value in (("mean_weight",0.),("minimum_weight",1.),("support_deficit_weight",0.),("support_deficit_weight",True),("candidate","ground_contact_v2c")):
        cfg=reward.validate_settings();cfg[field]=value
        with pytest.raises(ValueError):reward.validate_settings(cfg)
    with patch.object(Path,"read_bytes",side_effect=AssertionError('No reward filesystem I/O')):
        assert reward.reward_numpy(np.zeros((8,2)),100.,1.,1.)['raw_reward']==-1.
    with patch.object(reward,"BASE_C_DEFINITION_SHA256","0"*64):
        with pytest.raises(ValueError,match='V2c arithmetic'):reward.verify_base()
    meta=reward.formula_metadata();assert meta['raw_reward_bounds']==[-1.,1.] and not meta['cross_window_streak_memory']
    assert 'support=mean_substeps(q)' in meta['formula']


@pytest.mark.parametrize('dtype_name',('float32','float64'))
def test_optional_torch_numpy_parity_all_components(dtype_name):
    torch=pytest.importorskip('torch');rng=np.random.default_rng(42);f=rng.uniform(0,90,(2,3,8,2)).astype(dtype_name);f[0,0,3]=0.
    expected=reward.reward_numpy(f,100.,.8,.3);actual=reward.reward_torch(torch.as_tensor(f),100.,.8,.3)
    assert actual.keys()==expected.keys()
    for k in expected:np.testing.assert_allclose(actual[k].numpy(),expected[k],rtol=4e-6 if dtype_name=='float32' else 2e-14,atol=2e-7 if dtype_name=='float32' else 2e-15)


def test_optional_torch_invalid_poison_and_gradient():
    torch=pytest.importorskip('torch');f=torch.full((4,8,2),5.,dtype=torch.float64);f[1,0,0]=-.001;f[2,0,0]=float('nan')
    r=reward.reward_torch(f,100.,torch.tensor([1.,1.,1.,2.]),1.,validate=False)
    for v in r.values():assert torch.isfinite(v[0]).all() and torch.isnan(v[1:]).all()
    f=torch.full((8,2),5.,dtype=torch.float64,requires_grad=True);a=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
    reward.reward_torch(f,100.,a,.5)['raw_reward'].backward();assert torch.isfinite(f.grad).all() and f.grad.abs().sum()>0 and a.grad==pytest.approx(.1)

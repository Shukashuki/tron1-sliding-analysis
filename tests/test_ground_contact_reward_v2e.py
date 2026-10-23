"""E reward-level factorial arithmetic only; no empirical E policy results."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ground_contact_reward_v2e as reward


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_E_only_disables_D_deficit_all_other_components_identical(dtype):
    rng=np.random.default_rng(42);f=rng.uniform(0.,90.,(2,3,8,2)).astype(dtype);f[0,0,3]=0.
    original=f.copy();mg=np.array([[100.],[200.]],dtype=dtype)
    d=reward.base.reward_numpy(f,mg,.8,.3);e=reward.reward_numpy(f,mg,.8,.3)
    for name in d.keys()-{"raw_reward","support_deficit_penalty"}: np.testing.assert_array_equal(d[name],e[name])
    np.testing.assert_array_equal(e["support_deficit_penalty"],np.zeros_like(e["support_deficit"]))
    np.testing.assert_array_equal(e["raw_reward"],d["gated_support_bonus"])
    np.testing.assert_array_equal(f,original);assert e["raw_reward"].dtype==dtype


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_reward_level_2x2_factorial_zero_DiD_not_performance_evidence(dtype):
    rng=np.random.default_rng(7);f=rng.uniform(0.,100.,(64,8,2)).astype(dtype);f[:,2]=0.
    anchor=rng.random(64).astype(dtype);speed=rng.random(64).astype(dtype)
    b=reward.base.base.base.reward_numpy(f,100.,anchor,speed)
    c=reward.base.base.reward_numpy(f,100.,anchor,speed)
    d=reward.base.reward_numpy(f,100.,anchor,speed);e=reward.reward_numpy(f,100.,anchor,speed)
    tol=2e-7 if dtype==np.float32 else 2e-15
    delta=d["tracking_gate"]*d["impact_gate"]*(d["support_mean"]-d["support_min"])
    np.testing.assert_allclose(d["raw_reward"]-e["raw_reward"],-d["support_deficit"],atol=tol,rtol=2e-6)
    np.testing.assert_allclose(c["raw_reward"]-b["raw_reward"],-d["support_deficit"],atol=tol,rtol=2e-6)
    np.testing.assert_allclose(d["raw_reward"]-c["raw_reward"],delta,atol=tol,rtol=2e-6)
    np.testing.assert_allclose(e["raw_reward"]-b["raw_reward"],delta,atol=tol,rtol=2e-6)
    np.testing.assert_allclose((d["raw_reward"]-c["raw_reward"])-(e["raw_reward"]-b["raw_reward"]),0.,atol=tol)


def test_single_balanced_alternating_unsupported_and_partial_support():
    single=np.tile([100.,0.],(8,1));alternate=single.copy();alternate[1::2]=[0.,100.]
    for f in (single,alternate,np.full((8,2),50.)):
        assert reward.reward_numpy(f,100.,1.,1.)["raw_reward"]==1.
    for gate in (0.,1.):assert reward.reward_numpy(np.zeros((8,2)),100.,gate,gate)["raw_reward"]==0.
    gap=single.copy();gap[3]=0.
    assert reward.reward_numpy(gap,100.,1.,1.)["raw_reward"]==.875
    assert reward.reward_numpy(gap,100.,0.,1.)["raw_reward"]==0.
    assert reward.reward_numpy(gap,100.,1.,1.)["support_deficit"]==.125


def test_packed_split_fixture_equal_undiscounted_mean_support():
    packed=np.ones((2,8));packed[0,3:5]=0.;split=np.ones((2,8));split[:,3]=0.
    values=[reward.reward_numpy(np.stack((50*q,np.zeros_like(q)),axis=-1),100.,1.,1.)["raw_reward"] for q in (packed,split)]
    np.testing.assert_array_equal(values[0],[.75,1.]);np.testing.assert_array_equal(values[1],[.875,.875])
    assert values[0].sum()==values[1].sum()==1.75


@pytest.mark.parametrize("shape", ((4,2),(7,2),(9,2),(8,3),(8,),(0,8,2)))
def test_empty_or_wrong_window_rejected(shape):
    with pytest.raises(ValueError):reward.reward_numpy(np.zeros(shape),100.,1.,1.)


@pytest.mark.parametrize("bad", (np.nan,np.inf,-.001))
def test_zero_deficit_does_not_hide_invalid_numpy_forces(bad):
    f=np.ones((8,2));f[0,0]=bad
    with pytest.raises(ValueError):reward.reward_numpy(f,100.,1.,1.)


@pytest.mark.parametrize("field,bad", (("mg",0.),("mg",np.nan),("anchor",-1.),("anchor",np.nan),("speed",2.),("speed",np.inf)))
def test_invalid_mass_and_gates_rejected(field,bad):
    a={"mg":100.,"anchor":1.,"speed":1.};a[field]=bad
    with pytest.raises(ValueError):reward.reward_numpy(np.ones((8,2)),a["mg"],a["anchor"],a["speed"])


def test_positive_bounds_settings_ancestor_pin_metadata_and_no_hotpath_IO():
    assert reward.reward_numpy(np.full((8,2),-1e-4),100.,1.,1.)["raw_reward"]==0.
    rng=np.random.default_rng(1);r=reward.reward_numpy(rng.uniform(0,1e4,(1000,8,2)),100.,rng.random(1000),rng.random(1000))
    assert ((r["raw_reward"]>=0)&(r["raw_reward"]<=1)).all()
    for field,value in (("mean_weight",0.),("minimum_weight",1.),("support_deficit_weight",1),("support_deficit_weight",True),("candidate","ground_contact_v2d")):
        cfg=reward.validate_settings();cfg[field]=value
        with pytest.raises(ValueError):reward.validate_settings(cfg)
    with patch.object(Path,"read_bytes",side_effect=AssertionError("No reward filesystem I/O")):
        assert reward.reward_numpy(np.zeros((8,2)),100.,1.,1.)["raw_reward"]==0.
    with patch.object(reward,"BASE_D_DEFINITION_SHA256","0"*64):
        with pytest.raises(ValueError,match="V2d arithmetic"):reward.verify_base()
    meta=reward.formula_metadata();assert meta["raw_reward_bounds"]==[0.,1.] and not meta["cross_window_streak_memory"]
    assert "support=mean_substeps(q)" in meta["formula"] and "raw=gated_support_bonus-0*support_deficit" in meta["formula"]
    assert meta["immutable_v2d_definition_sha256"]==reward.BASE_D_DEFINITION_SHA256
    assert meta["zero_weight_preserves_invalid_force_poison"] and not meta["empirical_performance_interaction_claimed"]


@pytest.mark.parametrize("dtype_name",("float32","float64"))
def test_optional_torch_numpy_parity_all_components_and_factorial(dtype_name):
    torch=pytest.importorskip("torch");rng=np.random.default_rng(42);f=rng.uniform(0,90,(2,3,8,2)).astype(dtype_name);f[0,0,3]=0.
    expected=reward.reward_numpy(f,100.,.8,.3);actual=reward.reward_torch(torch.as_tensor(f),100.,.8,.3)
    assert actual.keys()==expected.keys()
    for k in expected:np.testing.assert_allclose(actual[k].numpy(),expected[k],rtol=4e-6 if dtype_name=="float32" else 2e-14,atol=2e-7 if dtype_name=="float32" else 2e-15)
    b=reward.base.base.base.reward_torch(torch.as_tensor(f),100.,.8,.3)
    c=reward.base.base.reward_torch(torch.as_tensor(f),100.,.8,.3);d=reward.base.reward_torch(torch.as_tensor(f),100.,.8,.3)
    torch.testing.assert_close(d["raw_reward"]-actual["raw_reward"],-d["support_deficit"])
    torch.testing.assert_close(c["raw_reward"]-b["raw_reward"],-d["support_deficit"])
    torch.testing.assert_close((d["raw_reward"]-c["raw_reward"])-(actual["raw_reward"]-b["raw_reward"]),torch.zeros_like(d["raw_reward"]),atol=2e-7,rtol=2e-6)


def test_optional_torch_zero_weight_keeps_invalid_nan_poison_and_gradients():
    torch=pytest.importorskip("torch");f=torch.full((4,8,2),5.,dtype=torch.float64);f[1,0,0]=-.001;f[2,0,0]=float("nan")
    r=reward.reward_torch(f,100.,torch.tensor([1.,1.,1.,2.]),1.,validate=False)
    for v in r.values():assert torch.isfinite(v[0]).all() and torch.isnan(v[1:]).all()
    f=torch.full((8,2),5.,dtype=torch.float64,requires_grad=True);a=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
    reward.reward_torch(f,100.,a,.5)["raw_reward"].backward()
    assert torch.isfinite(f.grad).all() and f.grad.abs().sum()>0 and a.grad==pytest.approx(.1)

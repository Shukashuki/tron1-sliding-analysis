"""Causal reward and geometry checks, not evidence of learned behavior."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ground_contact_reward_v2q as q


def case(force, gap):
    return q.reward_numpy(force, np.full(force.shape[:-2],200.),1.,1.,1.,1.,geometry_gap_m=gap)


def test_gap_signal_survives_zero_force_and_has_declared_scale():
    forces=np.zeros((5,8,2));gaps=np.broadcast_to(np.array([0.,.002,.02,.04,1.])[:,None,None],forces.shape)
    result=case(forces,gaps)
    np.testing.assert_allclose(result['unsupported_geometry_gap_cost'],1-np.exp(-gaps[:,0,0]/.02))
    assert np.all(np.diff(result['raw_reward'])<0)
    assert np.all(result['raw_reward']>=-2.)


def test_one_loaded_wheel_disables_cost_even_if_other_high():
    force=np.zeros((2,8,2));force[0,:,0]=100.;force[1,:,1]=100.
    gaps=np.ones_like(force)
    actual=case(force,gaps)
    expected=q.previous.reward_numpy(force,np.full(2,200.),1.,1.,1.,1.)
    np.testing.assert_array_equal(actual['raw_reward'],expected['raw_reward'])


def test_minwheel_not_average_and_worst_substep_not_endpoint():
    force=np.zeros((2,8,2));gaps=np.full_like(force,-.01)
    gaps[0,:,1]=1.;gaps[1,3,:]=.02
    result=case(force,gaps)
    np.testing.assert_allclose(result['unsupported_geometry_gap_cost'],[0.,1-np.exp(-1.)])


def test_threshold_stays_strict_and_scale_is_not_tolerance():
    force=np.zeros((3,8,2));force[:,:,0]=np.array([5.,5.0001,4.999])[:,None]
    result=case(force,np.full_like(force,.0001))
    cost=result['unsupported_geometry_gap_cost']
    assert cost[0]>0 and cost[1]==0 and cost[2]>0
    assert q.formula_metadata()['geometry_gap_scale_is_acceptance_tolerance'] is False


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_torch_numpy_parity_and_invalid_gap_poison(dtype):
    torch=pytest.importorskip('torch');rng=np.random.default_rng(42)
    force=rng.uniform(0.,200.,(13,8,2)).astype(dtype);force[::2]=0.
    gap=rng.uniform(-.01,.10,force.shape).astype(dtype)
    np_result=case(force,gap)
    tdtype=getattr(torch,dtype);tforce=torch.tensor(force)
    result=q.reward_torch(tforce,torch.full((13,),200.,dtype=tdtype),1.,1.,1.,1.,geometry_gap_m=torch.tensor(gap),validate=False)
    for name in np_result:
        np.testing.assert_allclose(result[name].numpy(),np_result[name],rtol=2e-6,atol=2e-6,err_msg=name)
    gap[2,3,0]=np.nan
    result=q.reward_torch(tforce,torch.full((13,),200.,dtype=tdtype),1.,1.,1.,1.,geometry_gap_m=torch.tensor(gap),validate=False)
    assert all(torch.isnan(v[2]).all() for v in result.values())
    assert all(torch.isfinite(v[1]).all() for v in result.values())
    with pytest.raises(ValueError):case(force,gap)


def test_batched_rotated_geometry_matches_independent_cpu_support():
    torch=pytest.importorskip('torch')
    import wheel_ground_geometry as reference
    from wheel_ground_geometry_torch import authored_gaps
    rng=np.random.default_rng(5);p=rng.normal(size=(37,2,3));quat=rng.normal(size=(37,2,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True)
    centers=np.array([[0.,.00707,0.],[0.,-.00707,0.]]);axes=np.array([[0.,-1.,0.]]*2)
    actual=authored_gaps(*map(torch.tensor,(p,quat,centers,axes,np.full(2,.127),np.full(2,.025)))).numpy()
    for i in range(2):
        expected=reference.cylinder_support(p[:,i]+reference.rotate(quat[:,i],centers[i]),reference.rotate(quat[:,i],axes[i]),.127,.025)['authored_signed_gap_m']
        np.testing.assert_allclose(actual[:,i],expected,atol=1e-12,rtol=1e-12)

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_hockey_contact_momentum import authored_masses, momentum_balance


def test_internal_impulses_cancel_and_external_impulse_matches_interval():
    dt=.0025;m=np.array([2.,1.]);v=np.zeros((4,2,3))
    impulses=np.array([1.,3.,-2.])
    for k,impulse in enumerate(impulses):
        v[k+1]=v[k]
        v[k+1,0,0]+=.6/m[0]  # Internal pair, equal and opposite.
        v[k+1,1,0]-=.6/m[1]
        v[k+1,1,0]+=impulse/m[1]
    ft=np.zeros((3,2,3));ft[:,1,0]=impulses/dt
    fn=np.full((3,2),m.sum()*9.81/2)
    result=momentum_balance(v,m,fn,ft,dt)
    np.testing.assert_allclose(result['residual_N'],0,atol=1e-10)
    shifted=momentum_balance(v,m,fn,np.roll(ft,1,axis=0),dt)
    assert np.linalg.norm(shifted['residual_N'])>100


def test_missing_end_state_rejected():
    with pytest.raises(ValueError,match='Invalid complete'):
        momentum_balance(np.zeros((3,2,3)),np.ones(2),np.zeros((3,2)),np.zeros((3,2,3)),.0025)


def test_authored_mass_reader_requires_explicit_positive_unique_values():
    text='    def Xform "base"\n        float physics:mass = 9.585\n        def Xform "collisions"\n'
    assert authored_masses(text)=={'base':9.585}
    with pytest.raises(ValueError,match='duplicate'):
        authored_masses(text+text)
    with pytest.raises(ValueError,match='Invalid'):
        authored_masses(text.replace('9.585','0'))

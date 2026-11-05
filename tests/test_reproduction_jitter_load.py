"""Numerical controls for jitter bands, force windows and interval alignment."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_reproduction_jitter_load import spectrum,vertical_balance,force_window_stats,contact_geometry


@pytest.mark.parametrize('dt',[.005,.0025])
def test_native_band_separates_reference_resolvable_and_faster_motion(dt):
    t=np.arange(round(4/dt))*dt
    x=3+2*t+np.sin(15*2*np.pi*t)+2*np.sin(40*2*np.pi*t)
    assert spectrum(x,dt,10,20)[0]['rms']==pytest.approx(1/np.sqrt(2),rel=.005)
    assert spectrum(x,dt,25,100)[0]['rms']==pytest.approx(np.sqrt(2),rel=.005)
    assert spectrum(3+2*t,dt)[0]['rms']<1e-10


def test_equal_duration_force_windows_preserve_impulse_not_native_peak():
    coarse=np.array([200.,400.,200.,0.])
    fine=np.array([0.,400.,0.,800.,200.,200.,0.,0.])
    assert fine.max()==2*coarse.max()
    assert force_window_stats(coarse,.005,.02)==force_window_stats(fine,.0025,.02)
    with pytest.raises(ValueError):force_window_stats(fine[:-1],.0025,.02)


def test_whole_body_momentum_cancels_internal_motion_and_rejects_force_shift():
    dt=.0025;m=np.array([2.,3.]);force=np.array([[10.,20.],[30.,50.],[15.,20.],[20.,10.]])
    v=np.zeros((5,2,3))
    az=(force.sum(-1)-m.sum()*9.81)/m.sum()
    v[1:,:,2]=np.cumsum(az)[:,None]*dt
    internal=np.arange(5)**2*.01
    v[:,0,2]+=internal;v[:,1,2]-=internal*m[0]/m[1]
    np.testing.assert_allclose(vertical_balance(v,m,force,dt),0,atol=1e-12)
    assert np.std(vertical_balance(v,m,np.roll(force,1,axis=0),dt))>10
    with pytest.raises(ValueError):vertical_balance(v[:-1],m,force,dt)


def test_geometry_does_not_confuse_zero_load_or_invalid_normal_with_airborne():
    raw={}
    for side in (0,1):
        raw[f'w{side}_normal_buffer_4']=np.ones((3,1,1),int)
        raw[f'w{side}_normal_buffer_5']=np.zeros((3,1,1),int)
        raw[f'w{side}_normal_buffer_3']=np.array([0.,.002,0.]).reshape(3,1,1)
        raw[f'w{side}_normal_buffer_2']=np.array([[0,0,1],[0,0,1],[0,0,0.]]).reshape(3,1,3)
    touch,unknown,_=contact_geometry(raw,3)
    np.testing.assert_array_equal(touch[:,0],[True,False,False])
    np.testing.assert_array_equal(unknown[:,0],[False,False,True])

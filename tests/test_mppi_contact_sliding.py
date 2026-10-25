"""Reject false persistent-slip claims across load gaps or wheel switches."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from screen_mppi_sliding import runs,evaluate


def inputs():
    t=1.44+np.arange(41)*.0025
    raw=dict(time_s=t,base_com_velocity_w=np.tile([1.,0,0],(len(t),1)),ground_normal_force_w=np.zeros((len(t),2,3)))
    raw['ground_normal_force_w'][:,:,2]=100
    a=dict(loaded_point_count=np.ones((len(t),2)),lateral_rms_m_s=np.ones((len(t),2))*.1,tangent_rms_m_s=np.ones((len(t),2))*.1)
    return raw,a


def test_any_contact_gap_breaks_sustained_slip_and_unloaded_velocity_is_not_counted():
    r,a=inputs();r['ground_normal_force_w'][::10,:,2]=0
    result=evaluate(r,a)
    assert not result['sustained_lateral_sliding_observed']
    assert all(s['lateral']['longest_contiguous_sliding_s']<=.0200001 for s in result['rows'])


def test_alternating_wheels_does_not_fabricate_same_wheel_continuity():
    r,a=inputs();r['ground_normal_force_w'][::2,0,2]=0;r['ground_normal_force_w'][1::2,1,2]=0
    assert np.all(r['ground_normal_force_w'][...,2].max(-1)>5)
    result=evaluate(r,a);assert not result['sustained_lateral_sliding_observed']
    assert all(s['lateral']['longest_contiguous_sliding_s']==0 for s in result['rows'])


def test_continuous_moving_loaded_lateral_slip_is_detected():
    r,a=inputs();result=evaluate(r,a)
    assert result['sustained_lateral_sliding_observed']
    assert all(s['lateral']['persistent_event_count']==1 for s in result['rows'])


def test_support_without_valid_contact_slip_measurement_is_unknown():
    r,a=inputs();a['loaded_point_count'][:]=0;a['lateral_rms_m_s'][:]=0
    result=evaluate(r,a);assert not result['sustained_lateral_sliding_observed']
    assert all(s['core_supported_but_slip_unobserved_samples']==41 for s in result['rows'])
    assert all(s['lateral']['loaded_sample_sliding_fraction'] is None for s in result['rows'])


def test_single_sample_has_zero_observed_duration_and_no_gap_interpolation():
    e=runs([0.,.0025,.005],[False,True,False]);assert e[0]['duration_s']==0 and e[0]['samples']==1

"""Adversarial qualification cases, not simulator-success claims."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_hockey_physx import heading_offsets,segments,qualify_stop,reconstruct_contacts,derive_signals
from prepare_hockey_physx_reference import stopping_speed


def test_actor_input_reconstruction_preserves_real_state_and_intervention_scope():
    from evaluate_hockey_physx import audit_actor_observations,load
    root=Path(__file__).resolve().parents[1]
    trace=load(root/'results/hockey_stop_physx_probe_20261026_v2/evidence/probes_v2/baseline/final_rollout.npz')
    reference=load(root/'results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2/motion.npz')
    assert audit_actor_observations(trace,reference)['maximum_error']<2e-6
    conditioning=load(root/'results/hockey_stop_conditioning_audit_20261027_v1/nominal_conditioning.npz')['command']
    trace['observations'][:,6:12]=conditioning[:,6:12]
    assert audit_actor_observations(trace,reference,conditioning,'reference_dq')['maximum_error']<2e-6
    with pytest.raises(ValueError,match='physical states/reference intervention'):
        audit_actor_observations(trace,reference,conditioning,'nominal_command')
    trace['observations'][20,23]+=.01
    with pytest.raises(ValueError,match='physical states/reference intervention'):
        audit_actor_observations(trace,reference,conditioning,'reference_dq')


def test_collected_execution_evidence_recovers_missing_annotation_without_editing_report():
    from evaluate_hockey_physx import verify_execution_evidence,sha
    root=Path(__file__).resolve().parents[1]/'results/hockey_stop_physx_probe_20261026_v2/evidence'
    run=root/'probes_v2/baseline';before=sha(run/'report.json')
    audit=verify_execution_evidence(root,run)
    assert not audit['wrapper_post_run_annotations_present']
    assert sha(run/'report.json')==before


@pytest.mark.parametrize('target',['probes_v2/status.json','probes_v2/baseline/report.json'])
def test_collected_execution_evidence_rejects_changed_launch_or_report(tmp_path,target):
    import shutil
    from evaluate_hockey_physx import verify_execution_evidence
    root=Path(__file__).resolve().parents[1]/'results/hockey_stop_physx_probe_20261026_v2/evidence'
    copied=tmp_path/'evidence';shutil.copytree(root,copied)
    with (copied/target).open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='Execution collection changed'):
        verify_execution_evidence(copied,copied/'probes_v2/baseline')


def ideal_signals():
    t=np.arange(1105)*.0025;rt=np.arange(139)*.02
    speed=stopping_speed(t,1.3,.4,1.8);ref_speed=stopping_speed(rt,1.3,.4,1.8)
    # Quadrature error below a millimetre is irrelevant to these threshold tests.
    position=np.column_stack((np.cumsum(speed)*.0025,np.zeros((len(t),2))))
    refpos=np.column_stack((np.interp(rt,t,position[:,0]),np.zeros((len(rt),2))))
    trace=dict(root_pos=position,root_com_vel=np.column_stack((speed,np.zeros((len(t),2)))))
    s=dict(time_s=t,body_speed_m_s=speed,normal_fz=np.tile([100.,0],(len(t),1)),
        native_sample_valid=t>0,root_tilt_deg=np.zeros(len(t)),
        normal_point_lateral_rms=np.tile([.2,0],(len(t),1)),aggregate_utilization=np.tile([1.,0],(len(t),1)),
        anchor_power=np.tile([-5.,0],(len(t),1)),dissipative_slipping_anchor_count=np.tile([1,0],(len(t),1)),
        wheel_beta_valid=np.tile((speed>=.25)[:,None],(1,2)),wheel_axial_beta_deg=np.tile([60.,0],(len(t),1)))
    yaw=np.radians(60*np.clip((rt-.4)/.4,0,1));q=np.zeros((len(rt),3,4))
    q[:,:,0]=np.cos(yaw[:,None]/2);q[:,:,3]=np.sin(yaw[:,None]/2)
    s['body_heading_deg']=np.degrees(np.interp(t,rt,yaw))
    s['wheel_heading_deg']=np.repeat(s['body_heading_deg'][:,None],2,axis=1)
    reference=dict(fps=50.,body_pos_w=refpos[:,None],body_lin_vel_w=np.column_stack((ref_speed,np.zeros((len(rt),2))))[:,None],
        body_quat_w=q,body_names=np.array(['base_Link','wheel_L_Link','wheel_R_Link']))
    command=dict(brake_start_s=.4,brake_end_s=1.8)
    return trace,s,reference,command


def test_prospective_success_gate_accepts_complete_idealized_evidence():
    assert qualify_stop(*ideal_signals())['task_success_claimed']


def test_no_entry_motion_is_not_a_successful_hockey_stop():
    trace,s,r,c=ideal_signals();s['body_speed_m_s'][:]=0.;trace['root_com_vel'][:]=0.
    out=qualify_stop(trace,s,r,c)
    assert not out['gates']['entry_speed_at_least_0p8']
    assert not out['gates']['approach_speed_retained']
    assert not out['task_success_claimed']


def test_partial_trace_cannot_shorten_required_reference_or_hold():
    trace,s,r,c=ideal_signals();n=100
    trace={k:v[:n] for k,v in trace.items()};s={k:v[:n] for k,v in s.items()}
    out=qualify_stop(trace,s,r,c)
    assert not out['gates']['full_reference_observed']
    assert not out['gates']['upright_supported_terminal_hold_ge_0p5']
    assert not out['task_success_claimed']


def test_499_millisecond_stop_and_earlier_stop_do_not_substitute_for_terminal_hold():
    trace,s,r,c=ideal_signals();t=s['time_s']
    s['body_speed_m_s'][np.isclose(t,2.26)]=.2
    out=qualify_stop(trace,s,r,c)
    assert out['terminal_stop_hold_s']==pytest.approx(.4975)
    assert not out['task_success_claimed']
    s['body_speed_m_s'][-1]=.2
    assert not qualify_stop(trace,s,r,c)['gates']['upright_supported_terminal_hold_ge_0p5']


@pytest.mark.parametrize('field,value',[
    ('normal_fz',[0,100]),('normal_point_lateral_rms',[0,.2]),
    ('aggregate_utilization',[.4,1]),('aggregate_utilization',[1.3,1]),('anchor_power',[5,-5]),
    ('dissipative_slipping_anchor_count',[0,1]),('wheel_axial_beta_deg',[0,60])])
def test_corroboration_must_occur_on_same_wheel(field,value):
    trace,s,r,c=ideal_signals();s[field][:]=value
    assert not qualify_stop(trace,s,r,c)['gates']['loaded_lateral_sliding_during_braking_ge_0p05']


def test_late_or_fallen_sliding_does_not_certify_braking():
    trace,s,r,c=ideal_signals();s['normal_point_lateral_rms'][s['time_s']<=1.8]=0
    assert not qualify_stop(trace,s,r,c)['gates']['loaded_lateral_sliding_during_braking_ge_0p05']
    trace,s,r,c=ideal_signals();s['root_tilt_deg'][:]=80
    out=qualify_stop(trace,s,r,c)
    assert not out['gates']['loaded_lateral_sliding_during_braking_ge_0p05']
    assert not out['gates']['upright_supported_terminal_hold_ge_0p5']


def test_wrong_heading_cannot_pass_as_motion_imitation():
    trace,s,r,c=ideal_signals();s['body_heading_deg']+=40;s['wheel_heading_deg']+=40
    out=qualify_stop(trace,s,r,c)
    assert not out['gates']['braking_body_heading_RMSE_le_15deg']
    assert not out['gates']['loaded_braking_wheel_heading_RMSE_le_20deg']
    assert not out['task_success_claimed']


def test_any_wheel_support_allows_a_lift_but_not_simultaneous_gap():
    trace,s,r,c=ideal_signals()
    assert qualify_stop(trace,s,r,c)['gates']['anywheel_continuously_supported']
    s['normal_fz'][300]=0
    out=qualify_stop(trace,s,r,c)
    assert out['longest_support_gap_s']==pytest.approx(.0025)
    assert not out['gates']['anywheel_continuously_supported']


def test_reverse_rolling_is_zero_axial_beta_and_rest_beta_is_undefined():
    d,a,m=heading_offsets(np.array([[-1.,0,0],[0,0,0],[0,1.,0]]),np.zeros(3))
    np.testing.assert_allclose(a[[0,2]],[0,-90]);assert np.isnan(a[1]) and not m[1]
    assert abs(d[0])==180


def test_no_interpolation_over_missing_state_samples():
    with pytest.raises(ValueError,match='Missing'):
        segments([0,.0025,.0075],[True]*3)


def native_fixture():
    n=3;b=3;z=np.zeros
    q=np.tile([1.,0,0,0],(n,b,1))
    trace=dict(root_pos=z((n,3)),root_quat=q[:,0],root_com_vel=z((n,3)),
        body_names=np.array(['base_Link','wheel_L_Link','wheel_R_Link']),
        body_com_pos=z((n,b,3)),body_com_vel=z((n,b,3)),body_omega=z((n,b,3)),body_quat=q,
        normal_fz=z((n,2)),lateral_slip_rms=z((n,2)),utilization_proxy=z((n,2)),
        friction_anchor_power=z((n,2)),friction_force_w=z((n,2,3)),environment_origin_w=np.array([10.,-2.,0.]),
        physics_dt_s=.0025,policy_dt_s=.02)
    trace['body_com_vel'][1:,1,1]=.2
    trace['normal_fz'][1:,0]=100;trace['lateral_slip_rms'][1:,0]=.2
    trace['utilization_proxy'][1:,0]=1.;trace['friction_anchor_power'][1:,0]=-14
    trace['friction_force_w'][1:,0,1]=-70
    raw=dict(physical_sample_indices=np.array([1,2]))
    for side in (0,1):
        active=1-side
        normal=[np.full((2,1,1),100.*active),np.tile(trace['environment_origin_w'],(2,1,1)),
            np.tile([0.,0,1.],(2,1,1)),z((2,1,1)),np.full((2,1,1),active,dtype=int),np.zeros((2,1,1),dtype=int)]
        friction=[np.tile([0.,-70.*active,0],(2,1,1)),np.tile(trace['environment_origin_w'],(2,1,1)),
            np.full((2,1,1),active,dtype=int),np.zeros((2,1,1),dtype=int)]
        raw.update({f'w{side}_normal_buffer_{i}':v for i,v in enumerate(normal)})
        raw.update({f'w{side}_friction_buffer_{i}':v for i,v in enumerate(friction)})
    return trace,raw


def test_native_anchor_reconstruction_and_environment_origin():
    trace,raw=native_fixture();contact,audit=reconstruct_contacts(trace,raw,.7)
    np.testing.assert_allclose(contact['anchor_power'][1:,0],-14)
    np.testing.assert_allclose(contact['dissipative_slipping_anchor_count'][1:,0],1)
    np.testing.assert_allclose(contact['aggregate_utilization'][1:,0],1)
    assert audit['physical_samples_checked']==2
    signals=derive_signals(trace,contact)
    assert not signals['wheel_beta_valid'].any()  # 0.2 m/s is below beta threshold.


def test_native_missing_sample_or_wrong_force_fails_closed():
    trace,raw=native_fixture();raw['physical_sample_indices'][1]=3
    with pytest.raises(ValueError,match='Missing'):reconstruct_contacts(trace,raw,.7)
    trace,raw=native_fixture();trace['normal_fz'][1,0]=90
    with pytest.raises(ValueError,match='readback mismatch'):reconstruct_contacts(trace,raw,.7)

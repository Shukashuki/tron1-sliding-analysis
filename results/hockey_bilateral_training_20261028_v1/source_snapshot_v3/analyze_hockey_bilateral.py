"""Independent bilateral-contact qualification for full-horizon PPO recordings."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, sha, q_matrix, reconstruct_contacts, derive_signals, qualify_stop, segments
from analyze_hockey_ppo import audit_observations
from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace, vector_signals
from evaluate_hockey_contact_priority import bilateral_gate
from hockey_speed_study import wilson_interval


def qualify(trace,raw,reference,command,mu=.7):
    force,native_audit=reconstruct_contacts(trace,raw,mu)
    old=derive_signals(trace,force)
    historical=qualify_stop(trace,old,reference,command)
    c=contact_kinematics(previous_state_trace(trace),raw,.0001)
    v=vector_signals(trace,raw,.0001)
    n=len(trace['root_pos']);pre=np.maximum(np.arange(n)-1,0)
    end=(len(reference['joint_pos'])-1)/float(reference['fps'])
    te=np.arange(n)*.0025;ts=te-.0025
    gate=bilateral_gate(ts,te,c['touching'],c['contact_unknown'],(0.,end))
    inside=(ts>=-1e-9)&(ts<=end+1e-9)
    both=c['touching'].all(-1)
    speed=old['body_speed_m_s'][pre];tilt=old['root_tilt_deg'][pre]
    terminal=inside&(ts>=end-.5-1e-9)
    stop=terminal&both&(speed<=.1)&(tilt<=30)
    hold=max((e['duration_s'] for e in segments(ts,stop)),default=0.)
    brake=(ts>=command['brake_start_s']-1e-9)&(te<=command['brake_end_s']+1e-9)
    rho=np.zeros_like(force['normal_fz'])
    np.divide(np.linalg.norm(force['friction_force'][...,:2],axis=-1),mu*force['normal_fz'],
              out=rho,where=force['normal_fz']>1e-6)
    ids=[trace['body_names'].astype(str).tolist().index('wheel_'+s+'_Link') for s in ('L','R')]
    axes=q_matrix(trace['body_quat'][pre][:,ids])[...,:,1].copy();axes[...,2]=0
    axes/=np.maximum(np.linalg.norm(axes,axis=-1,keepdims=True),1e-8)
    interval_lateral=np.abs((v['slip_velocity_interval']*axes).sum(-1))
    slip=(both[:,None]&(c['lateral_rms_m_s']>.05)&(interval_lateral>.05)&(rho>=.9)&(rho<=1.05)
          &(v['anchor_power_pre']<-.05)&(v['opposing_lateral_anchors_pre']>=1)
          &old['wheel_beta_valid'][pre]&(np.abs(old['wheel_axial_beta_deg'][pre])>=20)
          &brake[:,None]&(tilt[:,None]<45))
    first_fall=np.flatnonzero(inside&(tilt>=45))
    if len(first_fall):slip[np.arange(n)>=first_fall[0]]=False
    events=[segments(te,slip[:,side],interval_samples=True) for side in (0,1)]
    longest=[max((e['duration_s'] for e in ev),default=0.) for ev in events]
    rt=np.arange(len(reference['joint_pos']))/float(reference['fps'])
    ref_ids=[reference['body_names'].astype(str).tolist().index('wheel_'+s+'_Link') for s in ('L','R')]
    a=q_matrix(reference['body_quat_w'][:,ref_ids])[...,:,1]
    ref_heading=np.arctan2(-a[...,0],a[...,1])
    target=np.column_stack([np.degrees(np.interp(ts,rt,np.unwrap(2*ref_heading[:,side])/2)) for side in (0,1)])
    error=(old['wheel_heading_deg'][pre]-target+90)%180-90
    mask=brake[:,None]&c['touching']&old['wheel_beta_valid'][pre]
    wheel_rmse=float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None
    # Preserve the prior non-contact tracking/entry criteria. Replace every
    # load-conditioned contact, stop, wheel-heading and slide gate explicitly.
    gates={k:v for k,v in historical['gates'].items() if k not in (
        'anywheel_continuously_supported','upright_supported_terminal_hold_ge_0p5',
        'loaded_lateral_sliding_during_braking_ge_0p05','loaded_braking_wheel_heading_RMSE_le_20deg',
        'tilt_below_45deg_during_task')}
    gates.update(bilateral_continuous_contact=gate['eligible_for_task_comparison'],
        bilateral_terminal_stop_hold_ge_0p5=hold>=.5-1e-9,
        bilateral_contact_corroborated_lateral_slide_ge_0p05=max(longest)>=.05-1e-9,
        touching_braking_wheel_heading_RMSE_le_20deg=wheel_rmse is not None and wheel_rmse<=20,
        tilt_below_45deg_during_task=bool((tilt[inside]<45).all()))
    return dict(primary_contact_gate=gate,gates=gates,task_success_claimed=all(gates.values()),
        failed_gates=[k for k,v in gates.items() if not v],
        secondary_metrics_scope=gate['secondary_metrics_scope'],
        xy_RMSE_m=historical['xy_RMSE_m'],xy_peak_m=historical['xy_peak_m'],
        entry_mean_speed_m_s=historical['entry_mean_speed_last_0p1s_m_s'],
        approach_speed_ratio=historical['approach_COM_speed_ratio'],
        terminal_stop_hold_s=hold,corroborated_slide_longest_s=longest,
        corroborated_slide_total_s=(slip.sum(0)*.0025).tolist(),corroborated_slide_events=events,
        touching_braking_wheel_heading_RMSE_deg=wheel_rmse,
        first_45deg_tilt_s=float(ts[first_fall[0]]) if len(first_fall) else None,
        body_angular_acceleration_RMS_rad_s2=float(np.sqrt(np.mean(np.sum((np.diff(trace['root_omega'],axis=0)/.0025)**2,axis=-1)))),
        action_change_RMS=float(np.sqrt(np.mean(np.diff(trace['actions'],axis=0)**2))),
        no_absolute_height_gate=True,no_equal_load_gate=True,pointwise_friction_cone_certified=False,
        native_force_audit=native_audit)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bundle','protocol','command-dir','training-dir','suite','actor','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda p:json.loads(p.read_text())
    cfg=read(a.protocol);training=read(a.training_dir/'report.json');suite=read(a.suite/'report.json')
    for item in read(a.bundle/'manifest.json')['files']:
        path=(a.bundle/item['path']).resolve()
        if not path.is_relative_to(a.bundle.resolve()) or sha(path)!=item['sha256']:raise ValueError('Snapshot changed')
    assert training['transitions']==12288000 and training['status']=='training_completed_not_task_success'
    assert suite['status']=='validation_suite_completed_requires_independent_qualification'
    assert training['actor_sha256']==suite['actor_sha256']==sha(a.actor)
    for r in (training,suite):
        for key,path in [('source_sha256',a.bundle/'train_hockey_bilateral.py'),('bilateral_source_sha256',a.bundle/'hockey_bilateral_task.py'),
                         ('config_sha256',a.protocol),('motion_sha256',a.command_dir/'motion.npz')]:
            assert r[key]==sha(path)
    assert [c['id'] for c in suite['cases']]==[c['id'] for c in cfg['evaluation_cases']]
    import torch
    torch.set_num_threads(2);actor=torch.jit.load(str(a.actor),map_location='cpu').eval()
    reference=load(a.command_dir/'motion.npz');command=read(a.command_dir/'report.json')
    a.output.mkdir(parents=True);rows=[]
    for case,entry in zip(cfg['evaluation_cases'],suite['cases']):
        d=a.suite/case['id'];run=read(d/'report.json')
        assert sha(d/'report.json')==entry['report_sha256']
        assert run['actor_sha256']==sha(a.actor) and run['worlds']==1 and run['steps']==139 and run['external_forces']==0
        for key in ('source_sha256','bilateral_source_sha256','config_sha256','motion_sha256'):assert run[key]==suite[key]
        for f,h in run['artifacts'].items():assert sha(d/f)==h
        reset=read(d/'case.json');assert reset['case']==case and reset['maximum_initial_state_error']<=5e-5
        m=run['material_audit'];assert m['live_physx_materials_verified'] and m['ground_material_usd_verified']
        assert np.allclose(m['wheel_mu_range'],[.7,.7],atol=1e-6,rtol=0)
        assert all(g['friction']==1 and g['combine']=='multiply' for g in m['ground_colliders'])
        tr=load(d/'final_rollout.npz');raw=load(d/'native_contact_buffers.npz')
        assert tr['root_pos'].shape==(1113,3) and tr['actions'].shape==(139,8)
        obs=audit_observations(tr,reference,cfg)
        with torch.inference_mode():predicted=actor(torch.as_tensor(tr['observations'],dtype=torch.float32)).clamp(-1,1).numpy()
        error=float(np.max(np.abs(predicted-tr['actions'])))
        if not np.isfinite(error) or error>5e-5:raise ValueError('Actor replay differs')
        result=dict(case=case,input_sha256=run['artifacts'],observation_audit=obs,action_replay_error=error,
                    qualification=qualify(tr,raw,reference,command))
        (a.output/(case['id']+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        q=result['qualification'];rows.append(dict(id=case['id'],kind=case['kind'],success=q['task_success_claimed'],
            contact_gate=q['primary_contact_gate']['gate_status'],failed_gates=q['failed_gates'],
            xy_RMSE_m=q['xy_RMSE_m'],report_sha256=sha(a.output/(case['id']+'.json'))))
    pert=[r for r in rows if r['kind']=='perturbed'];k=sum(r['success'] for r in pert)
    report=dict(kind='hockey_bilateral_sliding_qualification_v1',nominal_success=rows[0]['success'],
        perturbation_trials=len(pert),perturbation_successes=k,conditional_Wilson_95_interval=wilson_interval(k,len(pert)),
        definition='Whole-episode bilateral geometric contact first, then independent stop, tracking, posture and corroborated sliding gates',
        force_threshold_for_contact=None,equal_load_target=False,early_termination_during_validation=False,
        training_report_sha256=sha(a.training_dir/'report.json'),suite_report_sha256=sha(a.suite/'report.json'),
        source_manifest_sha256=sha(a.bundle/'manifest.json'),cases=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(nominal_success=report['nominal_success'],perturbed_successes=k,trials=len(pert))))


if __name__=='__main__':main()

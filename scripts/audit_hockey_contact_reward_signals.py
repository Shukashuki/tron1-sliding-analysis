"""Diagnose reward resolution on completed nominal contact-screen recordings.

This is offline scoring, not logged training reward. In particular, the stop
proxy uses archived native normal forces; the training filtered force matrix
was not archived, so equality of the two reward values is not certified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from audit_hockey_reference_contact_gap import wheel_geometry
from evaluate_hockey_physx import load,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','evidence','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-name',default='run_v2')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh audit output required')
    read=lambda path:json.loads(path.read_text())
    manifest=read(a.bundle/'manifest.json');plan=read(a.bundle/'plan.json')
    for row in manifest['files']:
        if sha(a.bundle/row['path'])!=row['sha256']:raise ValueError('Snapshot changed')
    for name in ('hockey_ppo_task.py','hockey_contact_search.py','hockey_bilateral_task.py'):
        if sha(Path(__file__).with_name(name))!=sha(a.bundle/name):raise ValueError('Audited reward source changed')
    rows=[];saved={}
    for cell in plan['cells']:
        directory=a.evidence/a.run_name/cell['id'];qpath=directory/'nominal_qualification.json'
        if not qpath.exists():continue
        protocol=a.bundle/cell['protocol'];cfg=read(protocol);checked=read(qpath)
        run=read(directory/'nominal/report.json')
        if checked['protocol_sha256']!=sha(protocol) or checked['run_report_sha256']!=sha(directory/'nominal/report.json'):
            raise ValueError('Case identity mismatch')
        for file,digest in run['artifacts'].items():
            if sha(directory/'nominal'/file)!=digest:raise ValueError('Recording changed')
        trace=load(directory/'nominal/final_rollout.npz');raw=load(directory/'nominal/native_contact_buffers.npz')
        dt=float(trace['physics_dt_s']);n=len(trace['root_pos'])
        if not np.isclose(dt,.0025,atol=1e-12,rtol=0) or n!=1113:raise ValueError('Incomplete 400 Hz recording')
        c=contact_kinematics(previous_state_trace(trace),raw,cfg['bilateral_sliding']['contact_tolerance_m'])
        _,_,gap=wheel_geometry(trace,'body_pos','body_quat',cfg['bilateral_sliding']['geometry'])
        control_end=np.arange(8,n,8);control_start=control_end-8
        endpoint_s=control_end*dt;start_s=control_start*dt
        horizon=checked['qualification']['primary_contact_gate']['window_s'][1]
        within=(start_s>=-1e-9)&(endpoint_s<=horizon+1e-9)
        terminal=(start_s>=horizon-.5-1e-9)&(endpoint_s<=horizon+1e-9)
        pre_touch=c['touching'][1:].reshape(-1,8,2)
        post_gap=gap[1:].reshape(-1,8,2)
        tolerance=cfg['bilateral_sliding']['contact_tolerance_m']
        both=pre_touch.all(-1)&(post_gap<=tolerance).all(-1)
        fraction=both.mean(-1);continuous=both.all(-1)
        gap_cost=np.clip((post_gap-tolerance)/cfg['bilateral_sliding']['gap_scale_m'],0,1).max(-1)
        clearance=.5*(gap_cost.mean(-1)+gap_cost.max(-1))
        weights=cfg['bilateral_sliding']['weights']
        contact_terms=weights['contact_fraction']*fraction+weights['contact_loss']*(1-fraction)+weights['clearance']*clearance
        speed=np.linalg.norm(trace['root_com_vel'][:,:2],axis=-1)
        q=trace['root_quat'];tilt=np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))
        good_kinematics=(speed<=.1)&(tilt<=np.deg2rad(30))
        endpoint_good=good_kinematics[control_end]
        complete_kinematics=good_kinematics[:-1].reshape(-1,8).all(-1)&good_kinematics[1:].reshape(-1,8).all(-1)
        supported=(trace['normal_fz'][1:].reshape(-1,8,2).max(-1)>5).mean(-1)
        phase=(np.arange(len(control_end))+1)>=round(cfg['brake_end_s']/.02)
        scales=cfg['reward_scales']
        proxy=phase*np.exp(-(speed[control_end]/scales['stop_speed_m_s'])**2)*np.exp(-(tilt[control_end]/scales['stop_tilt_rad'])**2)*supported
        proxy*=cfg['reward_weights']['stop_hold']*.02
        alpha=np.linalg.norm(np.diff(trace['root_omega'],axis=0)/dt,axis=-1).reshape(-1,8)
        scale=cfg['bilateral_sliding']['angular_acceleration_scale_rad_s2']
        saturation=alpha>=scale
        missed=terminal&endpoint_good&~complete_kinematics
        incompatible=terminal&~continuous&(proxy>1e-6)
        row=dict(cell=cell['id'],qualifier_sha256=sha(qpath),raw_sha256=run['artifacts'],
            terminal_control_windows=int(terminal.sum()),
            terminal_endpoint_good_but_substep_speed_or_tilt_violation_windows=int(missed.sum()),
            missed_endpoint_windows_s=[[float(start_s[i]),float(endpoint_s[i])] for i in np.flatnonzero(missed)],
            terminal_contact_loss_windows=int((terminal&~continuous).sum()),
            terminal_positive_native_force_stop_proxy_with_contact_loss_windows=int(incompatible.sum()),
            positive_stop_proxy_contact_loss_windows_s=[[float(start_s[i]),float(endpoint_s[i])] for i in np.flatnonzero(incompatible)],
            terminal_stop_proxy_sum=float(proxy[terminal].sum()),
            terminal_stop_proxy_sum_during_contact_loss=float(proxy[incompatible].sum()),
            angular_acceleration_scale_rad_s2=scale,
            angular_penalty_saturated_substep_fraction=float(saturation[within].mean()),
            angular_acceleration_peak_rad_s2=float(alpha[within].max()),
            contact_terms_weighted_integral=float((contact_terms[within]*.02).sum()),
            task_success=checked['qualification']['task_success_claimed'])
        rows.append(row)
        signals=dict(control_start_s=start_s,control_end_s=endpoint_s,within_reference=within,terminal=terminal,
            bilateral_window=continuous,contact_fraction=fraction,clearance=clearance,
            endpoint_speed_m_s=speed[control_end],endpoint_tilt_rad=tilt[control_end],
            endpoint_kinematics_pass=endpoint_good,all_substep_kinematics_pass=complete_kinematics,
            native_force_supported_fraction=supported,stop_reward_proxy=proxy,
            angular_penalty_saturation_fraction=saturation.mean(-1),contact_terms=contact_terms)
        saved.update({cell['id']+'__'+key:value for key,value in signals.items()})
    if not rows:raise ValueError('No completed nominal records')
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'signals.npz',**saved)
    report=dict(kind='hockey_contact_reward_signal_audit_v1',script_sha256=sha(__file__),
        source_manifest_sha256=sha(a.bundle/'manifest.json'),signals_sha256=sha(a.output/'signals.npz'),
        physics_steps=0,training_changed=False,criterion_changed=False,
        logged_training_reward=False,causal_failure_mechanism_proven=False,
        stop_proxy_force_source='Recorded native summed normal force; training filtered force matrix not archived',
        stop_reward_definition='Control-end speed/tilt multiplied by mean any-wheel force>5N fraction; no hold-duration state',
        contact_loss_definition='Native pre-step bilateral touching AND post-step finite-cylinder gap <=0.1mm',
        smoothness_definition='Per-substep squared angular acceleration, capped at scale^2 before averaging',cases=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()

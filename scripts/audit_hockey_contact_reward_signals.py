"""Diagnose reward resolution on completed nominal contact-screen recordings.

This is offline scoring, not logged training reward. In particular, the stop
proxy uses archived native normal forces; the training filtered force matrix
was not archived, so equality of the two reward values is not certified.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from audit_hockey_reference_contact_gap import wheel_geometry
from evaluate_hockey_physx import load,sha


def frozen_refinement_code(bundle):
    """Load the selected validator/math without replacing live local modules.

    The adapter captures its math imports during execution, so bind the frozen
    dependency temporarily and restore the previous module even on failure.
    No simulator, policy or adapter instance is created by this operation.
    """
    hashes={row['path']:row['sha256'] for row in json.loads((bundle/'manifest.json').read_text())['files']}
    names=('hockey_contact_refinement_math','hockey_contact_refinement')
    for name in names:
        if hashes.get(name+'.py')!=sha(bundle/(name+'.py')):
            raise ValueError('Frozen refinement source changed')
    def load_module(name):
        spec=importlib.util.spec_from_file_location('_hockey_audit_frozen_'+name,bundle/(name+'.py'))
        module=importlib.util.module_from_spec(spec)
        previous_bytecode=sys.dont_write_bytecode
        sys.dont_write_bytecode=True
        # Execute the verified source itself, never a possibly stale bytecode cache.
        try:exec(compile((bundle/(name+'.py')).read_bytes(),str(bundle/(name+'.py')),'exec'),module.__dict__)
        finally:sys.dont_write_bytecode=previous_bytecode
        return module
    math_module=load_module(names[0]);previous=sys.modules.get(names[0])
    sys.modules[names[0]]=math_module
    try:
        adapter=load_module(names[1])
    finally:
        if previous is None:sys.modules.pop(names[0],None)
        else:sys.modules[names[0]]=previous
    return adapter.validate_settings,math_module


def declared_scores(cfg,pre_touch,post_gap,velocity,up,phase,alpha_sq,legacy_stop_proxy,*,reward_code=None):
    """Replay the selected formulas, retaining the legacy force-proxy caveat."""
    settings=cfg.get('reward_refinement')
    if settings is not None:
        if reward_code is None:
            from hockey_contact_refinement import validate_settings
            import hockey_contact_refinement_math as math_module
        else:
            validate_settings,math_module=reward_code
        settings=validate_settings(settings)
    else:
        settings=dict(stop_term='unchanged',angular_term='bounded_quadratic')
    scale=cfg['bilateral_sliding']['angular_acceleration_scale_rad_s2']
    exceeds=alpha_sq>=scale**2
    angular=np.clip(alpha_sq/scale**2,0,1).mean(-1)
    stop=legacy_stop_proxy.copy();force_proxy=True
    if settings['angular_term']=='pseudo_huber' or settings['stop_term'] in ('bilateral_substeps','bilateral_substeps_xyz','bilateral_substeps_xyz_huber'):
        import torch
        tensor=lambda value:torch.as_tensor(value,dtype=torch.float32)
        if settings['angular_term']=='pseudo_huber':angular=math_module.angular_pseudo_huber(tensor(alpha_sq),scale).numpy()
        if settings['stop_term'] in ('bilateral_substeps','bilateral_substeps_xyz','bilateral_substeps_xyz_huber'):
            n=len(alpha_sq);xyz=settings['stop_term'] in ('bilateral_substeps_xyz','bilateral_substeps_xyz_huber');dimensions=3 if xyz else 2
            if velocity.ndim!=2 or velocity.shape[1]<dimensions:
                raise ValueError('Missing declared stop-velocity dimensions')
            selected=velocity[:,:dimensions]
            options={'velocity_dimensions':3} if xyz else {}
            if settings['stop_term']=='bilateral_substeps_xyz_huber':options['speed_form']='separate_pseudo_huber'
            stop=math_module.bilateral_substep_stop(torch.as_tensor(pre_touch),tensor(post_gap),
                tensor(selected[:-1].reshape(n,8,dimensions)),tensor(selected[1:].reshape(n,8,dimensions)),
                tensor(up[:-1].reshape(n,8)),tensor(up[1:].reshape(n,8)),torch.as_tensor(phase),
                contact_tolerance_m=cfg['bilateral_sliding']['contact_tolerance_m'],
                speed_scale_m_s=cfg['reward_scales']['stop_speed_m_s'],tilt_scale_rad=cfg['reward_scales']['stop_tilt_rad'],**options).numpy()
            stop*=cfg['reward_weights']['stop_hold']*.02;force_proxy=False
    saturation=exceeds if settings['angular_term']=='bounded_quadratic' else np.zeros_like(exceeds)
    return dict(angular=angular,stop=stop,saturation=saturation,legacy_scale_exceedance=exceeds,
        angular_form=settings['angular_term'],stop_form=settings['stop_term'],stop_uses_force_proxy=force_proxy)


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
        reward_code=None
        if cfg.get('reward_refinement') is not None:
            if run.get('reward_refinement')!=cfg['reward_refinement']:raise ValueError('Refinement declaration changed')
            reward_code=frozen_refinement_code(a.bundle)
            for key,name in (('reward_refinement_source_sha256','hockey_contact_refinement.py'),
                             ('reward_refinement_math_sha256','hockey_contact_refinement_math.py')):
                if run.get(key)!=sha(a.bundle/name):raise ValueError('Recorded refinement source changed')
        if (checked['protocol_sha256']!=sha(protocol)
            or checked['source_manifest_sha256']!=sha(a.bundle/'manifest.json')
            or checked['run_report_sha256']!=sha(directory/'nominal/report.json')):
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
        alpha_sq=np.sum((np.diff(trace['root_omega'],axis=0)/dt)**2,axis=-1).reshape(-1,8)
        alpha=np.sqrt(alpha_sq)
        scale=cfg['bilateral_sliding']['angular_acceleration_scale_rad_s2']
        up=1-2*(q[:,1]**2+q[:,2]**2)
        declared=declared_scores(cfg,pre_touch,post_gap,trace['root_com_vel'],up,phase,alpha_sq,proxy,reward_code=reward_code)
        saturation=declared['saturation']
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
            declared_stop_form=declared['stop_form'],declared_stop_uses_native_force_proxy=declared['stop_uses_force_proxy'],
            declared_terminal_stop_score_sum=float(declared['stop'][terminal].sum()),
            declared_angular_form=declared['angular_form'],
            declared_angular_penalty_integral=float((declared['angular'][within]*weights['angular_acceleration']*.02).sum()),
            angular_acceleration_scale_rad_s2=scale,
            angular_penalty_saturated_substep_fraction=float(saturation[within].mean()),
            angular_legacy_scale_exceedance_substep_fraction=float(declared['legacy_scale_exceedance'][within].mean()),
            angular_acceleration_peak_rad_s2=float(alpha[within].max()),
            contact_terms_weighted_integral=float((contact_terms[within]*.02).sum()),
            task_success=checked['qualification']['task_success_claimed'])
        rows.append(row)
        signals=dict(control_start_s=start_s,control_end_s=endpoint_s,within_reference=within,terminal=terminal,
            bilateral_window=continuous,contact_fraction=fraction,clearance=clearance,
            endpoint_speed_m_s=speed[control_end],endpoint_tilt_rad=tilt[control_end],
            endpoint_kinematics_pass=endpoint_good,all_substep_kinematics_pass=complete_kinematics,
            native_force_supported_fraction=supported,stop_reward_proxy=proxy,
            angular_penalty_saturation_fraction=saturation.mean(-1),contact_terms=contact_terms,
            declared_angular_component=declared['angular'],declared_stop_score=declared['stop'],
            angular_legacy_scale_exceedance_fraction=declared['legacy_scale_exceedance'].mean(-1))
        saved.update({cell['id']+'__'+key:value for key,value in signals.items()})
    if not rows:raise ValueError('No completed nominal records')
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'signals.npz',**saved)
    report=dict(kind='hockey_contact_reward_signal_audit_v2',script_sha256=sha(__file__),
        source_manifest_sha256=sha(a.bundle/'manifest.json'),signals_sha256=sha(a.output/'signals.npz'),
        physics_steps=0,training_changed=False,criterion_changed=False,
        scoring_scope='Complete control windows within the frozen reference horizon; validation recordings, not training returns',
        logged_training_reward=False,causal_failure_mechanism_proven=False,
        stop_proxy_force_source='Recorded native summed normal force; training filtered force matrix not archived',
        stop_reward_definition='Selected per case; legacy endpoint score uses a native-force proxy, refined substep score uses bilateral geometry without force',
        contact_loss_definition='Native pre-step bilateral touching AND post-step finite-cylinder gap <=0.1mm',
        smoothness_definition='Selected per case: bounded quadratic or pseudo-Huber; legacy-scale exceedance is not saturation of pseudo-Huber',cases=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()

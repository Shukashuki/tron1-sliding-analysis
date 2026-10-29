"""Score proposed refinements on unchanged, completed PhysX recordings.

No policy is trained or simulated. The legacy stop proxy uses archived native
normal forces, whose equality to the training filtered matrix is unverified.
Score changes are diagnostic and cannot predict a learned success rate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from audit_hockey_reference_contact_gap import wheel_geometry
from evaluate_hockey_physx import load,sha
from hockey_contact_refinement_math import bilateral_substep_stop,angular_pseudo_huber


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','evidence','reward-audit','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--run-name',default='run_v2')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh score directory required')
    read=lambda path:json.loads(path.read_text())
    previous=read(a.reward_audit/'report.json');plan=read(a.bundle/'plan.json')
    if previous['source_manifest_sha256']!=sha(a.bundle/'manifest.json'):raise ValueError('Different source snapshot')
    if sha(a.reward_audit/'signals.npz')!=previous['signals_sha256']:raise ValueError('Audited signals changed')
    legacy=load(a.reward_audit/'signals.npz');rows=[];archive={}
    for case in previous['cases']:
        cell=next(row for row in plan['cells'] if row['id']==case['cell']);prefix=cell['id']+'__'
        cfg=read(a.bundle/cell['protocol']);d=a.evidence/a.run_name/cell['id']/'nominal'
        if cfg.get('reward_refinement') is not None:
            raise ValueError('This candidate comparison requires an unrefined baseline; use the reward-signal auditor for trained refinements')
        for name,digest in case['raw_sha256'].items():
            if sha(d/name)!=digest:raise ValueError('Physical record changed')
        trace=load(d/'final_rollout.npz');raw=load(d/'native_contact_buffers.npz')
        geo=cfg['bilateral_sliding']['geometry'];c=contact_kinematics(previous_state_trace(trace),raw,.0001)
        _,_,gap=wheel_geometry(trace,'body_pos','body_quat',geo)
        n=len(trace['actions'])
        if n!=139 or trace['root_pos'].shape!=(1113,3) or float(trace['physics_dt_s'])!=.0025:
            raise ValueError('Require the locked 139-frame, 400 Hz screening recording')
        shape=(n,8);tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
        velocity=trace['root_com_vel'][:,:2];q=trace['root_quat'];up=1-2*(q[:,1]**2+q[:,2]**2)
        end_frame=np.minimum(np.arange(n)+1,138)
        phase=end_frame>=round(cfg['brake_end_s']/.02)
        stop=bilateral_substep_stop(torch.as_tensor(c['touching'][1:].reshape(n,8,2)),
            tensor(gap[1:].reshape(n,8,2)),tensor(velocity[:-1].reshape(n,8,2)),tensor(velocity[1:].reshape(n,8,2)),
            tensor(up[:-1].reshape(shape)),tensor(up[1:].reshape(shape)),torch.as_tensor(phase),
            contact_tolerance_m=cfg['bilateral_sliding']['contact_tolerance_m'],
            speed_scale_m_s=cfg['reward_scales']['stop_speed_m_s'],tilt_scale_rad=cfg['reward_scales']['stop_tilt_rad']).numpy()
        alpha_sq=np.sum((np.diff(trace['root_omega'],axis=0)/.0025)**2,axis=-1).reshape(shape)
        scale=cfg['bilateral_sliding']['angular_acceleration_scale_rad_s2']
        angular=angular_pseudo_huber(tensor(alpha_sq),scale).numpy()
        old_angular=np.clip(alpha_sq/scale**2,0,1).mean(-1)
        stop*=cfg['reward_weights']['stop_hold']*.02
        angular_weight=cfg['bilateral_sliding']['weights']['angular_acceleration']
        within=legacy[prefix+'within_reference'];terminal=legacy[prefix+'terminal'];bilateral=legacy[prefix+'bilateral_window']
        old_stop=legacy[prefix+'stop_reward_proxy']
        rows.append(dict(cell=cell['id'],input_sha256=case['raw_sha256'],
            legacy_terminal_stop_proxy_sum=float(old_stop[terminal].sum()),
            proposed_terminal_stop_score_sum=float(stop[terminal].sum()),
            legacy_stop_proxy_on_contact_loss_windows=float(old_stop[terminal&~bilateral].sum()),
            proposed_stop_score_on_contact_loss_windows=float(stop[terminal&~bilateral].sum()),
            legacy_angular_penalty_integral=float((old_angular[within]*angular_weight*.02).sum()),
            proposed_angular_penalty_integral=float((angular[within]*angular_weight*.02).sum()),
            proposed_angular_component_max=float(angular[within].max()),
            observed_task_success=case['task_success'],trained_with_refinement=False))
        for name,value in dict(proposed_stop_score=stop,proposed_angular_component=angular,
                               legacy_angular_component=old_angular).items():archive[prefix+name]=value
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'signals.npz',**archive)
    report=dict(kind='offline_hockey_refinement_scores_v1',script_sha256=sha(__file__),
        candidate_math_sha256=sha(Path(__file__).with_name('hockey_contact_refinement_math.py')),
        source_reward_audit_sha256=sha(a.reward_audit/'report.json'),signals_sha256=sha(a.output/'signals.npz'),
        physics_steps=0,training_steps=0,success_criterion_changed=False,predicted_policy_success=False,
        legacy_stop_uses_native_force_proxy=True,legacy_filtered_force_equivalence_verified=False,cases=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k not in ('input_sha256',)} for row in rows]))


if __name__=='__main__':main()

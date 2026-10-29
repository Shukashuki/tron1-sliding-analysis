"""Check whether the frozen reference requests the observed late leg extension.

Position targets are reconstructed from the verified tracking configuration and
recorded actions. They are not measured motor torques or a causal reward audit.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cases-json','reference','tracking-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda path:json.loads(path.read_text())
    sources=read(a.cases_json)
    if len(sources)!=4:raise ValueError('Four comparison columns required')
    reference=load(a.reference/'motion.npz');rt=np.arange(len(reference['joint_pos']))/float(reference['fps'])
    horizon=float(rt[-1]);start=horizon-.5;mask=rt>=start-1e-9
    names=reference['joint_names'].astype(str).tolist();body=reference['body_names'].astype(str).tolist().index('base_Link')
    legs=[names.index(n) for n in ('abad_L_Joint','abad_R_Joint','hip_L_Joint','hip_R_Joint','knee_L_Joint','knee_R_Joint')]
    scale=None
    for node in ast.parse(a.tracking_source.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LEG_ACTION_SCALE' for t in node.targets):
            scale=ast.literal_eval(node.value)
    if scale!=1.:raise ValueError('Unrecognized action scale; audit the target mapping before proceeding')
    report=dict(kind='hockey_terminal_command_audit_v1',script_sha256=sha(__file__),
        reference_sha256=sha(a.reference/'motion.npz'),tracking_source_sha256=sha(a.tracking_source),
        cases_selection_sha256=sha(a.cases_json),window_s=[start,horizon],physics_steps=0,
        reference_or_training_changed=False,criterion_changed=False,causal_reward_mechanism_proven=False,
        target_mapping='Initial reference leg positions + recorded clipped leg actions * verified LEG_ACTION_SCALE; use_default_offset=True in the pinned tracking source',
        targets_are_reconstructed=True,torques_or_actuator_saturation_inferred=False,
        reference=dict(leg_joint_names=[names[i] for i in legs],
            leg_position_range_rad=np.ptp(reference['joint_pos'][mask][:,legs],axis=0).tolist(),
            leg_velocity_peak_abs_rad_s=np.max(abs(reference['joint_vel'][mask][:,legs]),axis=0).tolist(),
            root_height_range_m=float(np.ptp(reference['body_pos_w'][mask,body,2])),
            root_linear_velocity_peak_abs_m_s=np.max(abs(reference['body_lin_vel_w'][mask,body]),axis=0).tolist()),cases=[])
    arrays={};panels=[]
    for number,source in enumerate(sources):
        bundle=(a.cases_json.parent/source['bundle']).resolve();evidence=(a.cases_json.parent/source['evidence']).resolve()
        cell,stage=source['case'].split(':')
        if stage not in ('parent_transfer','nominal'):raise ValueError('Qualified validation required')
        plan=read(bundle/'plan.json');entry=next(c for c in plan['cells'] if c['id']==cell)
        protocol=bundle/entry['protocol'];cfg=read(protocol)
        directory=evidence/source['run_name']/cell/stage
        qualifier=directory.parent/(stage+'_qualification.json');checked=read(qualifier);run=read(directory/'report.json')
        if (checked['protocol_sha256']!=sha(protocol) or checked['run_report_sha256']!=sha(directory/'report.json')
            or checked['source_manifest_sha256']!=sha(bundle/'manifest.json')
            or cfg['tracking_sha256']!=report['tracking_source_sha256'] or run['motion_sha256']!=report['reference_sha256']):
            raise ValueError('Source/reference/qualification identity mismatch')
        for name,digest in run['artifacts'].items():
            if sha(directory/name)!=digest:raise ValueError('Recording changed')
        trace=load(directory/'final_rollout.npz');joint_names=trace['joint_names'].astype(str).tolist()
        if joint_names!=names or trace['actions'].shape!=(139,8) or float(trace['physics_dt_s'])!=.0025:
            raise ValueError('Joint order or clock changed')
        if not np.isclose(checked['qualification']['primary_contact_gate']['window_s'][1],horizon):
            raise ValueError('Task horizon changed')
        indices=np.arange(round(start/.0025),round(horizon/.0025)+1)
        controls=np.arange(round(start/.02),round(horizon/.02))
        if len(indices)!=201 or len(controls)!=25:raise ValueError('Incomplete terminal interval')
        action=trace['actions'][controls]
        if not np.isfinite(action).all() or abs(action).max()>1.000001:raise ValueError('Invalid recorded control')
        target=reference['joint_pos'][0,legs]+action[:,:6]*scale
        actual=trace['joint_pos'][indices][:,legs]
        vz=trace['root_com_vel'][indices,2]
        row=dict(case=source['case'],qualification_sha256=sha(qualifier),raw_sha256=run['artifacts'],
            reference_frames=int(mask.sum()),physical_state_samples=len(indices),executed_control_intervals=len(controls),
            final_action_at_horizon_excluded=True,actual_joint_change_rad=(actual[-1]-actual[0]).tolist(),
            reconstructed_target_change_rad=(target[-1]-target[0]).tolist(),
            reconstructed_target_range_rad=np.ptp(target,axis=0).tolist(),
            clipped_leg_action_boundary_fraction=float((abs(action[:,:6])>=.999).mean()),
            terminal_vz_m_s=float(vz[-1]),last_half_second_peak_abs_vz_m_s=float(abs(vz).max()),
            task_success=checked['qualification']['task_success_claimed'])
        report['cases'].append(row)
        time=indices*.0025;ct=controls*.02
        for key,value in dict(time_s=time,control_time_s=ct,actual_leg_joint_pos_rad=actual,
                              reconstructed_leg_targets_rad=target,root_vz_m_s=vz).items():
            arrays[f'case{number}_{key}']=value
        title=entry['model']+'\n'+('Frozen parent' if stage=='parent_transfer' else entry['reward'].replace('_',' '))
        panels.append((title,time,ct,actual,target,vz))
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'signals.npz',**arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(21,8),sharey='row',constrained_layout=True)
    for col,(label,time,ct,actual,target,vz) in enumerate(panels):
        for row,j in enumerate((4,5)):
            ax=axes[row,col]
            ax.plot(time,actual[:,j],label='Recorded joint',color='#2677ad')
            ax.step(np.r_[ct,horizon],np.r_[target[:,j],target[-1,j]],where='post',label='Reconstructed motor target',color='#db772a')
            ax.plot(rt[mask],reference['joint_pos'][mask,legs[j]],'k--',label='Reference',lw=1)
            ax.set_ylabel(('Left','Right')[row]+' knee [rad]')
        axes[0,col].set_title(label,fontsize=10)
        axes[2,col].plot(time,vz,color='#595959');axes[2,col].axhline(0,color='black',ls='--',lw=.7)
        axes[2,col].set(xlabel='Time [s]',ylabel='Base COM vertical velocity [m/s]')
        for ax in axes[:,col]:ax.set_xlim(start,horizon);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Final 0.5 s | fixed reference posture | recorded PhysX | reconstructed position targets are not torques')
    fig.savefig(a.output/'terminal_commands_4x1.png',dpi=140);plt.close(fig)
    report['artifacts']={name:sha(a.output/name) for name in ('signals.npz','terminal_commands_4x1.png')}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(reference=report['reference'],cases=report['cases'])))


if __name__=='__main__':main()

"""Resolve post-braking native contact points, material slip and wheel capacity.

Normal contacts and friction anchors are kept separate: the archived API does
not supply their patch/load correspondence. No per-point cone is fabricated.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_hockey_contact_cone_relation import read_case, finite_stats
from audit_hockey_contact_timing import material_step_velocity
from audit_hockey_contact_momentum import momentum_balance
from evaluate_hockey_physx import load, q_matrix, sha


def persistent_onsets(low, sliding, contact, dt, minimum_s=.02, gap_s=.05):
    """Find a sustained low-slip period before a sustained slip onset.

    This detects kinematics, not a solver static/dynamic friction mode.
    Both contact endpoints must remain present throughout the lookback.
    """
    low=np.asarray(low,bool);sliding=np.asarray(sliding,bool);contact=np.asarray(contact,bool)
    if low.ndim!=1 or low.shape!=sliding.shape or low.shape!=contact.shape or dt<=0:
        raise ValueError('Require aligned one-dimensional interval arrays')
    if (low&sliding).any():raise ValueError('Low-slip and sliding states must be disjoint')
    width=int(np.ceil(minimum_s/dt));gap=int(np.floor(gap_s/dt));events=[]
    for i in np.flatnonzero(sliding & ~np.r_[False,sliding[:-1]]):
        if i<width or i+width>len(low) or not sliding[i:i+width].all():continue
        for end in range(i,max(width,i-gap)-1,-1):
            if (low[end-width:end].all() and not sliding[end:i].any()
                    and contact[end-width:i+width].all()):
                events.append(dict(onset_index=int(i),low_slip_start_index=int(end-width),low_slip_end_index=int(end)))
                break
    return events


def contact_points(trace,raw,n):
    """Ragged normal-point and friction-anchor samples, without identity tracking."""
    columns=('time_s','side','row','buffer_index','x_m','y_m','z_m','gap_m','Fn_N',
             'tangent_pre_m_s','tangent_interval_m_s','lateral_pre_m_s','lateral_interval_m_s')
    points=[];anchors=[];dt=float(trace['physics_dt_s']);origin=trace['environment_origin_w']
    tangent_pre=np.full((n,2),np.nan);tangent_interval=tangent_pre.copy()
    for side,name in enumerate(('wheel_L_Link','wheel_R_Link')):
        bid=trace['body_names'].astype(str).tolist().index(name)
        rotations=q_matrix(trace['body_quat'][:,bid])
        for row in range(n):
            i=int(raw['physical_sample_indices'][row]);before=i-1
            if before!=row:raise ValueError('Contact geometry timestamp changed')
            com=trace['body_com_pos'][before,bid].astype(float)
            vel=trace['body_com_vel'][before,bid].astype(float);omega=trace['body_omega'][before,bid].astype(float)
            count=int(raw[f'w{side}_normal_buffer_4'][row].item());start=int(raw[f'w{side}_normal_buffer_5'][row].item())
            indices=np.arange(start,start+count)
            normal=raw[f'w{side}_normal_buffer_2'][row,indices].astype(float)
            gap=raw[f'w{side}_normal_buffer_3'][row,indices].reshape(-1).astype(float)
            valid=(np.linalg.norm(normal,axis=-1)>1e-8)&(gap<=.0001)
            indices=indices[valid];normal=normal[valid];gap=gap[valid]
            axis=rotations[before,:,1].copy();axis[2]=0;axis/=max(np.linalg.norm(axis),1e-8)
            if len(indices):
                normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
                pts=raw[f'w{side}_normal_buffer_1'][row,indices].astype(float)-origin
                fn=raw[f'w{side}_normal_buffer_0'][row,indices].reshape(-1).astype(float)
                pre=vel+np.cross(omega,pts-com)
                step=material_step_velocity(pts,com,trace['body_com_pos'][i,bid],rotations[before],rotations[i],dt)
                pre-=(pre*normal).sum(-1,keepdims=True)*normal
                step-=(step*normal).sum(-1,keepdims=True)*normal
                pre_norm=np.linalg.norm(pre,axis=-1);step_norm=np.linalg.norm(step,axis=-1)
                tangent_pre[row,side]=np.sqrt(np.mean(pre_norm**2))
                tangent_interval[row,side]=np.sqrt(np.mean(step_norm**2))
                for k,idx in enumerate(indices):
                    points.append([row*dt,side,row,int(idx),*pts[k],gap[k],fn[k],pre_norm[k],step_norm[k],pre[k]@axis,step[k]@axis])
            count=int(raw[f'w{side}_friction_buffer_2'][row].item());start=int(raw[f'w{side}_friction_buffer_3'][row].item())
            for idx in range(start,start+count):
                pos=raw[f'w{side}_friction_buffer_1'][row,idx].astype(float)-origin
                force=raw[f'w{side}_friction_buffer_0'][row,idx].astype(float)
                velocity=vel+np.cross(omega,pos-com)
                anchors.append([row*dt,side,row,idx,*pos,*force,float(force@velocity)])
    return dict(point_columns=list(columns),points=np.asarray(points,dtype=float),
                anchor_columns=['time_s','side','row','buffer_index','x_m','y_m','z_m','Fx_N','Fy_N','Fz_N','pre_power_W'],
                anchors=np.asarray(anchors,dtype=float),point_tangent_pre_rms=tangent_pre,point_tangent_interval_rms=tangent_interval)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cases-json','reference','mass-report','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda p:json.loads(p.read_text())
    selection=read(a.cases_json);mass_report=read(a.mass_report)
    import torch
    torch.set_num_threads(2)
    rows=[];saved={};plots=[]
    for k,source in enumerate(selection):
        row,s=read_case(source,a.cases_json,a.reference)
        bundle=(a.cases_json.parent/source['bundle']).resolve()
        cell,stage=source['case'].split(':');plan=read(bundle/'plan.json')
        cfg=read(bundle/next(x['protocol'] for x in plan['cells'] if x['id']==cell))
        if mass_report['asset_sha256']!=cfg['asset_sha256']:raise ValueError('Mass asset mismatch')
        directory=(a.cases_json.parent/source['evidence']/source['run_name']/cell/stage).resolve()
        tr=load(directory/'final_rollout.npz');raw=load(directory/'native_contact_buffers.npz')
        n=len(s['time_start_s']);dt=float(tr['physics_dt_s']);t=s['time_start_s']
        native=contact_points(tr,raw,n)
        names=tr['body_names'].astype(str).tolist();massmap=mass_report['authored_body_masses_kg']
        if set(names)!=set(massmap):raise ValueError('Mass/body identities differ')
        masses=np.array([massmap[name] for name in names]);mass=masses.sum()
        velocity=(tr['body_com_vel'][:n+1]*masses[None,:,None]).sum(1)/mass
        position=(tr['body_com_pos'][:n+1]*masses[None,:,None]).sum(1)/mass
        balance=momentum_balance(tr['body_com_vel'][:n+1],masses,tr['normal_fz'][1:n+1],tr['friction_force_w'][1:n+1],dt)
        speed=np.linalg.norm(velocity[:,:2],axis=-1)
        heading=velocity[:-1,:2]/np.maximum(speed[:-1,None],1e-8)
        acceleration=np.diff(velocity,axis=0)/dt
        deceleration=-(acceleration[:,:2]*heading).sum(-1)
        force_deceleration=-(balance['wheel_force_N'][:,:2]*heading).sum(-1)/mass
        low=(native['point_tangent_pre_rms']<.02)&(native['point_tangent_interval_rms']<.02)
        sliding=(native['point_tangent_pre_rms']>.05)&(native['point_tangent_interval_rms']>.05)
        persistent=s['contact_at_both_endpoints']
        wheels=[]
        for side in (0,1):
            events=persistent_onsets(low[:,side],sliding[:,side],persistent[:,side],dt)
            selected=[]
            for event in events:
                i=event['onset_index']
                if t[i]<cfg['brake_start_s']-1e-9:continue
                windows={label:slice(max(0,i-20),i) if label=='before_50ms' else slice(i,min(n,i+20)) for label in ('before_50ms','after_50ms')}
                summary={}
                for label,w in windows.items():
                    valid=s['ratio_valid'][w,side]
                    summary[label]={key:finite_stats(s[key][w,side][valid]) for key in ('normal_force_N','tangent_force_N','rho_static','rho_dynamic','interval_lateral_speed_m_s')}
                    summary[label]['COM_deceleration_m_s2']=finite_stats(deceleration[w])
                selected.append(dict(**event,onset_s=float(t[i]),windows=summary,solver_mode_observed=False))
            post=t>=cfg['brake_start_s']-1e-9
            pts=native['points'];pointmask=(pts[:,1]==side)&(pts[:,0]>=cfg['brake_start_s']-1e-9)
            count=np.bincount(pts[pts[:,1]==side,2].astype(int),minlength=n)
            wheels.append(dict(side=('L','R')[side],post_braking_contact_point_samples=int(pointmask.sum()),
                point_count_per_interval=finite_stats(count[post]),
                low_slip_contact_s=float((post&low[:,side]&persistent[:,side]).sum()*dt),
                slipping_contact_s=float((post&sliding[:,side]&persistent[:,side]).sum()*dt),
                slip_present_at_brake=bool(sliding[round(cfg['brake_start_s']/dt),side]),
                new_sustained_low_to_high_slip_events=selected,
                post_braking_point_speed_m_s=finite_stats(pts[pointmask,9]),
                post_braking_point_normal_N=finite_stats(pts[pointmask,8])))
        phases=[]
        for label,start,end in [('approach',0.,.4),('brake_onset',.4,.6),('braking',.6,1.8),('stop',1.8,2.76)]:
            mask=(t>=start-1e-9)&(t+dt<=end+1e-9)
            phases.append(dict(phase=label,start_s=start,end_s=end,
                COM_speed_start_m_s=float(speed[round(start/dt)]),COM_speed_end_m_s=float(speed[round(end/dt)]),
                COM_xy_path_length_m=float(np.linalg.norm(np.diff(position[:,:2],axis=0)[mask],axis=-1).sum()),
                COM_deceleration_m_s2=finite_stats(deceleration[mask]),
                contact_impulse_xy_Ns=(balance['wheel_force_N'][mask,:2].sum(0)*dt).tolist(),
                bilateral_interruption_ms=float((mask&~persistent.all(-1)).sum()*dt*1000),
                near_dynamic_lateral_slide_s=[float((mask&s['corroborated_sliding'][:,side]).sum()*dt) for side in (0,1)],
                normal_force_N=[finite_stats(s['normal_force_N'][mask,side]) for side in (0,1)]))
        axis=q_matrix(tr['body_quat'][:n,[names.index('wheel_L_Link'),names.index('wheel_R_Link')]])[..., :,1]
        camber=np.degrees(np.arcsin(np.clip(axis[...,2],-1,1)))
        joint_names=tr['joint_names'].astype(str).tolist();wheel_ids=[joint_names.index('wheel_'+x+'_Joint') for x in ('L','R')]
        for name,value in s.items():saved[f'case{k}_{name}']=value
        for name in ('points','anchors','point_tangent_pre_rms','point_tangent_interval_rms'):
            saved[f'case{k}_{name}']=native[name]
        for name,value in dict(COM_velocity=velocity,COM_position=position,COM_speed=speed,COM_deceleration=deceleration,
                               force_deceleration=force_deceleration,wheel_camber_deg=camber,wheel_spin_rad_s=tr['joint_vel'][:n,wheel_ids]).items():
            saved[f'case{k}_{name}']=value
        row.update(point_analysis=wheels,phases=phases,point_columns=native['point_columns'],anchor_columns=native['anchor_columns'],
            authored_total_mass_kg=float(mass),whole_task_linear_momentum_RMS_residual_N=float(np.sqrt(np.mean(np.sum(balance['residual_N']**2,axis=-1)))),
            analytical_capacity_drop_at_fixed_Fn_fraction=1-row['mu_dynamic']/row['mu_static'])
        rows.append(row);plots.append((row,s,native,speed,deceleration,camber,tr['joint_vel'][:n,wheel_ids]))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'signals.npz',**saved)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('#2574a9','#d5782c')
    for index,(row,s,native,speed,decel,camber,spin) in enumerate(plots):
        pts=native['points'];mask=pts[:,0]>=.4-1e-9;post=pts[mask];t=s['time_start_s'];angle=np.linspace(0,2*np.pi,361)
        fig,axs=plt.subplots(1,4,figsize=(24,5.1),constrained_layout=True)
        for side in (0,1):
            p=post[post[:,1]==side]
            sc=axs[0].scatter(p[:,4],p[:,5],c=p[:,0],s=8+np.clip(p[:,8],0,200)/10,cmap='viridis',vmin=.4,vmax=2.76,marker=('o','^')[side],label=('Left','Right')[side])
            axs[1].scatter(p[:,0],p[:,9],s=5,alpha=.4,color=colors[side],label=('Left','Right')[side])
            ax=axs[side+2];valid=(t>=.4)&s['ratio_valid'][:,side];cap=row['mu_static']*s['normal_force_N'][valid,side]
            ax.plot(np.cos(angle),np.sin(angle),'k-',label='Static boundary')
            ratio=row['mu_dynamic']/row['mu_static'];ax.plot(ratio*np.cos(angle),ratio*np.sin(angle),'k--',label='Dynamic boundary')
            ax.scatter(s['force_longitudinal_N'][valid,side]/cap,s['force_lateral_N'][valid,side]/cap,c=t[valid],s=7,cmap='viridis',vmin=.4,vmax=2.76)
            radius=max(1.1,float(np.nanmax(s['rho_static'][valid,side]))*1.05)
            ax.set(title=('Left','Right')[side]+' wheel resultant',xlabel='F_long / (mu_s Fn)',ylabel='F_lat / (mu_s Fn)',aspect='equal',xlim=(-radius,radius),ylim=(-radius,radius))
            ax.legend(fontsize=7);ax.grid(alpha=.2)
        axs[0].set(title='Actual touching normal points',xlabel='World X [m]',ylabel='World Y [m]',aspect='equal');axs[0].legend(fontsize=7)
        axs[1].set(title='Tangential material speed at those points',xlabel='Time [s]',ylabel='Pre-step tangential speed [m/s]');axs[1].axhline(.05,color='gray',ls=':');axs[1].legend(fontsize=7)
        fig.colorbar(sc,ax=axs,shrink=.7,label='Time since task start [s]')
        fig.suptitle(row['label']+' | Post-braking points, t = 0.4–2.76 s\nPoint samples are not persistent IDs; normal points and friction anchors are separate; no per-point cone certificate')
        fig.savefig(a.output/f'case{index}_contact_points_4x1.png',dpi=140);plt.close(fig)
        fig,axs=plt.subplots(1,4,figsize=(24,5),constrained_layout=True)
        for side,color in enumerate(colors):
            label=('L','R')[side]
            axs[0].plot(t,s['normal_force_N'][:,side],color=color,label=label+' Fn')
            axs[0].plot(t,s['tangent_force_N'][:,side],color=color,ls='--',label=label+' |Ft|')
            axs[1].plot(t,s['rho_static'][:,side],color=color,label=label+' rho_s')
            if row['mu_static']!=row['mu_dynamic']:axs[1].plot(t,s['rho_dynamic'][:,side],color=color,ls='--',label=label+' rho_d')
            axs[2].plot(t,camber[:,side],color=color,label=label+' camber')
        axs[3].plot(np.r_[t,t[-1]+.0025],speed,color='black',label='System COM speed')
        twin=axs[3].twinx();twin.plot(t,decel,color='#8b60a4',alpha=.65,lw=.7,label='COM deceleration');twin.set_ylabel('Signed deceleration [m/s²]')
        for ax in axs:
            ax.set_xlim(.3,2.76);ax.axvline(.4,color='crimson',ls=':');ax.axvline(1.8,color='gray',ls=':');ax.set_xlabel('Time [s]');ax.grid(alpha=.2);ax.legend(fontsize=7)
        axs[0].set(title='Normal and tangential forces',ylabel='Force [N]')
        axs[1].axhline(1,color='black',ls='--');axs[1].set(title='Wheel capacity utilization',ylabel='Utilization')
        axs[2].set(title='Wheel inclination',ylabel='Signed axle elevation [deg]')
        axs[3].set(title='Motion response',ylabel='System COM XY speed [m/s]')
        fig.suptitle(row['label']+' | Geometry at interval start; force averages following 2.5 ms | No solver stick/slip flag')
        fig.savefig(a.output/f'case{index}_braking_response_4x1.png',dpi=140);plt.close(fig)
    report=dict(kind='hockey_post_braking_contact_points_v1',source_sha256=sha(__file__),selection_sha256=sha(a.cases_json),
        reference_sha256=sha(a.reference/'motion.npz'),mass_report_sha256=sha(a.mass_report),
        dependencies={name:sha(Path(__file__).with_name(name)) for name in ('audit_hockey_contact_cone_relation.py','audit_hockey_contact_timing.py','audit_hockey_contact_momentum.py','evaluate_hockey_physx.py')},
        physics_steps=0,training_changed=False,gates_changed=False,per_point_cone_certified=False,solver_static_dynamic_flag_observed=False,
        contact_tolerance_m=.0001,kinematic_thresholds=dict(low_m_s=.02,sliding_m_s=.05,min_duration_s=.02,max_transition_gap_s=.05),
        limitations=['Native normal points and friction anchors have no recorded patch/load correspondence.',
            'Contact-point buffer indices are frame-local, not persistent point IDs.',
            'Low-to-high-slip events are observable kinematics, not certified solver mode switches.',
            'Friction capacity can change because Fn changes even with fixed material coefficients.',
            'Same-policy probes are closed-loop responses, not identical-action open-loop physics tests.',
            'COM momentum uses authored masses; no new runtime mass readback.'],cases=rows,
        artifact_sha256={p.name:sha(p) for p in a.output.iterdir()})
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(label=r['label'],interruption_ms=r['bilateral_interruption_ms'],
        post_brake_points=[w['post_braking_contact_point_samples'] for w in r['point_analysis']],
        sustained_slip_onsets=[len(w['new_sustained_low_to_high_slip_events']) for w in r['point_analysis']]) for r in rows]))


if __name__=='__main__':main()

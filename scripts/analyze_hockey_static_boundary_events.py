"""Inspect the largest braking-phase static-bound utilization in a frozen suite.

Selection is explicitly post hoc. A large force ratio or a subsequent force
drop is not itself evidence of a solver switch from static to dynamic friction.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from analyze_hockey_braking_contacts import contact_points
from evaluate_hockey_physx import load,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audit','evidence','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    audit=json.loads((a.audit/'report.json').read_text());mu_s=audit['mu_static'];mu_d=audit['mu_dynamic']
    if not audit['all_33_qualifications_reproduced']:raise ValueError('Frozen suite replay required')
    peaks=[]
    for case in audit['cases']:
        name=case['id'];directory=a.evidence/'eval_suite'/name
        if sha(directory/'final_rollout.npz')!=audit['case_hashes'][name]['final_rollout.npz']:raise ValueError('State recording changed')
        tr=load(directory/'final_rollout.npz');dt=float(tr['physics_dt_s']);n=round(2.76/dt)
        t=np.arange(n)*dt;mask=(t>=.4)&(t<1.8);fn=tr['normal_fz'][1:n+1].astype(float)
        ft=np.linalg.norm(tr['friction_force_w'][1:n+1,:,:2].astype(float),axis=-1)
        for side in (0,1):
            indices=np.flatnonzero(mask&(fn[:,side]>0))
            if len(indices)==0:raise ValueError('No valid braking contacts')
            ratios=ft[indices,side]/(mu_s*fn[indices,side]);j=int(indices[np.argmax(ratios)])
            peaks.append(dict(case=name,side=side,index=j,time_s=float(t[j]),rho_static=float(ft[j,side]/(mu_s*fn[j,side]))))
    peaks.sort(key=lambda r:r['rho_static'],reverse=True);peak=peaks[0];directory=a.evidence/'eval_suite'/peak['case']
    if sha(directory/'native_contact_buffers.npz')!=audit['case_hashes'][peak['case']]['native_contact_buffers.npz']:raise ValueError('Contact recording changed')
    tr=load(directory/'final_rollout.npz');raw=load(directory/'native_contact_buffers.npz');point=contact_points(tr,raw,n)
    side=peak['side'];center=peak['index'];indices=np.arange(max(0,center-24),min(n,center+25))
    forces=tr['friction_force_w'][1:n+1,side,:2].astype(float);fn=tr['normal_fz'][1:n+1,side].astype(float)
    ft=np.linalg.norm(forces,axis=-1);rows=[]
    for j in indices:
        pts=point['points'][(point['points'][:,1]==side)&(point['points'][:,2]==j)]
        rows.append(dict(index=int(j),time_s=float(t[j]),Fn_N=float(fn[j]),Ft_N=float(ft[j]),
            rho_static=float(ft[j]/(mu_s*fn[j])),rho_dynamic=float(ft[j]/(mu_d*fn[j])),
            tangent_pre_m_s=float(point['point_tangent_pre_rms'][j,side]),
            tangent_interval_m_s=float(point['point_tangent_interval_rms'][j,side]),touching_point_count=len(pts)))
    k=center-indices[0];selected=rows[k];following=rows[k+1]
    local_arrest=(selected['tangent_pre_m_s']>.02 and selected['tangent_interval_m_s']<.02
                  and following['tangent_pre_m_s']<.02 and following['tangent_interval_m_s']<.02)
    a.output.mkdir(parents=True)
    with (a.output/'selected_event.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,4,figsize=(22,4.7),constrained_layout=True);tt=t[indices]
    for values,label,color in [(fn[indices],'Fn','#777777'),(ft[indices],'|Ft|','#d37527'),(mu_s*fn[indices],'mu_s Fn','black'),(mu_d*fn[indices],'mu_d Fn','#2779a5')]:
        axs[0].step(tt,values,where='post',label=label,color=color)
    axs[0].set(title='Measured interval forces and capacities',ylabel='Force [N]')
    axs[1].step(tt,ft[indices]/(mu_s*fn[indices]),where='post',label='rho_s',color='#d37527')
    axs[1].step(tt,ft[indices]/(mu_d*fn[indices]),where='post',label='rho_d',color='#2779a5');axs[1].axhline(1,color='black',ls='--')
    axs[1].set(title='Wheel-resultant capacity use',ylabel='Utilization')
    for key,label,color in [('point_tangent_pre_rms','Pre-step','#2779a5'),('point_tangent_interval_rms','Same point over interval','#d37527')]:
        axs[2].plot(tt,point[key][indices,side],label=label,color=color)
    axs[2].axhline(.02,color='gray',ls=':');axs[2].set(title='Material-point tangential velocity',ylabel='RMS speed [m/s]')
    angle=np.linspace(0,2*np.pi,361);ratio=mu_d/mu_s
    axs[3].plot(np.cos(angle),np.sin(angle),'k-',label='Static boundary')
    axs[3].plot(ratio*np.cos(angle),ratio*np.sin(angle),'k--',label='Dynamic boundary')
    scaled=forces[indices]/(mu_s*fn[indices,None]);dots=axs[3].scatter(scaled[:,0],scaled[:,1],c=tt,cmap='viridis',s=24)
    axs[3].plot(scaled[k,0],scaled[k,1],'rx',ms=9,label='Selected peak')
    axs[3].set(title='Aggregate force section',xlabel='World Fx / (mu_s Fn)',ylabel='World Fy / (mu_s Fn)',aspect='equal',xlim=(-1.1,1.1),ylim=(-1.1,1.1))
    fig.colorbar(dots,ax=axs[3],label='Time [s]',shrink=.7)
    for ax in axs[:3]:ax.axvline(peak['time_s'],color='red',ls=':');ax.set(xlim=(tt[0],tt[-1]),xlabel='Time [s]')
    for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
    event_text='interval velocity falls near zero at the peak' if local_arrest else 'local arrest criterion not met'
    fig.suptitle(f"{peak['case']} | {'Left' if side==0 else 'Right'} wheel | Largest braking rho_s among 33 cases\nPost hoc diagnostic example; {event_text}; no observed solver mode flag")
    fig.savefig(a.output/'static_boundary_event_4x1.png',dpi=140);plt.close(fig)
    result=dict(kind='hockey_static_boundary_posthoc_diagnostic_v1',source_sha256=sha(__file__),
        audit_sha256=sha(a.audit/'report.json'),contact_point_math_sha256=sha(Path(__file__).with_name('analyze_hockey_braking_contacts.py')),
        case_hashes=audit['case_hashes'][peak['case']],mu_static=mu_s,mu_dynamic=mu_d,
        selection_rule='Largest wheel-resultant rho_s during [0.4,1.8) s among all 33 audited cases; no trial excluded',
        all_peaks=peaks,selected=selected,following_interval=following,
        local_arrest_kinematics_observed=local_arrest,local_arrest_speed_threshold_m_s=.02,
        following_tangent_force_drop_fraction=1-following['Ft_N']/selected['Ft_N'],
        all_local_intervals_have_touching_points=all(r['touching_point_count']>0 for r in rows),
        physics_steps=0,solver_static_dynamic_mode_observed=False,
        interpretation=[('This selected peak arrests local tangential motion rather than initiating sustained sliding.' if local_arrest
                         else 'The selected peak does not meet the stated local arrest criterion; its mode is unclassified.'),
            'The following force drop alone therefore does not demonstrate a static-to-dynamic transition.',
            'Normal force and demanded acceleration change too; force ratios use wheel sums, not per-point patch loads.'],
        artifact_sha256={p.name:sha(p) for p in a.output.iterdir()})
    (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(peak=peak,selected=selected,following_interval=following)))


if __name__=='__main__':main()

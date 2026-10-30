"""Compare identical-policy material probes without assuming a friction-mode switch."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','selection','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda p:json.loads(p.read_text())
    report=read(a.analysis/'report.json');sources=read(a.selection)
    if report['selection_sha256']!=sha(a.selection) or len(report['cases'])!=3:
        raise ValueError('Require the three declared fixed-policy probes')
    for name,digest in report['artifact_sha256'].items():
        if sha(a.analysis/name)!=digest:raise ValueError('Prior analysis changed')
    signals=np.load(a.analysis/'signals.npz',allow_pickle=False)
    rows=[];actors=set();exceedances=[]
    for k,(source,case) in enumerate(zip(sources,report['cases'])):
        directory=(a.selection.parent/source['evidence']/source['run_name']/source['case'].split(':')[0]).resolve()
        qualified=read(directory/'nominal_qualification.json');q=qualified['qualification'];actors.add(qualified['actor_sha256'])
        if sha(directory/'nominal_qualification.json')!=case['qualification_sha256']:raise ValueError('Qualification changed')
        row=dict(label=case['label'],mu_static=case['mu_static'],mu_dynamic=case['mu_dynamic'],
            success=q['task_success_claimed'],failed_gates=q['failed_gates'],xy_rmse_m=q['xy_RMSE_m'],xy_peak_m=q['xy_peak_m'],
            contact_interruption_ms=case['bilateral_interruption_ms'],stop_hold_s=q['terminal_stop_hold_s'],
            angular_acceleration_rms_rad_s2=q['body_angular_acceleration_RMS_rad_s2'],
            qualified_slide_total_s=q['corroborated_slide_total_s'],
            diagnostic_braking_slide_s=[w['bilateral_endpoint_corroborated_slide_s'] for w in case['wheels']],
            braking_COM_xy_path_m=sum(p['COM_xy_path_length_m'] for p in case['phases'] if p['phase'] in ('brake_onset','braking')),
            post_braking_normal_point_samples=[w['post_braking_contact_point_samples'] for w in case['point_analysis']],
            static_bound_exceedance_count=sum(w['static_ratio_over_1p05_intervals'] for w in case['wheels']))
        rows.append(row)
        if k==1:
            for side in (0,1):
                mask=signals[f'case{k}_braking']&signals[f'case{k}_ratio_valid'][:,side]&(signals[f'case{k}_rho_dynamic'][:,side]>1.05)
                for j in np.flatnonzero(mask):
                    exceedances.append(dict(side=('L','R')[side],time_s=float(signals[f'case{k}_time_start_s'][j]),
                        normal_N=float(signals[f'case{k}_normal_force_N'][j,side]),
                        tangent_N=float(signals[f'case{k}_tangent_force_N'][j,side]),
                        rho_dynamic=float(signals[f'case{k}_rho_dynamic'][j,side]),
                        rho_static=float(signals[f'case{k}_rho_static'][j,side]),
                        point_tangent_pre_rms_m_s=float(signals[f'case{k}_point_tangent_pre_rms'][j,side]),
                        point_tangent_interval_rms_m_s=float(signals[f'case{k}_point_tangent_interval_rms'][j,side]),
                        both_contact_endpoints=bool(signals[f'case{k}_contact_at_both_endpoints'][j].all())))
    if len(actors)!=1:raise ValueError('Material probes do not use the same actor')
    events=report['cases'][1]['point_analysis'][0]['new_sustained_low_to_high_slip_events']
    selected=next(e for e in events if .4<=e['onset_s']<1.8)
    onset=selected['onset_s'];start=onset-.0625;end=onset+.0675
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('#2779a5','#d37527','#755098')
    a.output.mkdir(parents=True)
    fig,axs=plt.subplots(1,4,figsize=(24,4.8),constrained_layout=True)
    for k,color in enumerate(colors):
        t=signals[f'case{k}_time_start_s'];m=(t>=start-1e-9)&(t<=end+1e-9)
        label=f"{rows[k]['mu_static']:g} / {rows[k]['mu_dynamic']:g}"
        axs[0].plot(t[m],signals[f'case{k}_normal_force_N'][m,0],color=color,label=label)
        axs[1].plot(t[m],signals[f'case{k}_tangent_force_N'][m,0],color=color,label=label)
        axs[2].plot(t[m],signals[f'case{k}_rho_static'][m,0]*rows[k]['mu_static'],color=color,label=label)
        axs[3].plot(t[m],signals[f'case{k}_point_tangent_interval_rms'][m,0],color=color,label=label)
    for ax,title,y in zip(axs,['Left-wheel normal force','Left-wheel tangential force','Resultant force / normal force','Material-point tangential slip'],['Fn [N]','|Ft| [N]','|Ft| / Fn','Interval RMS [m/s]']):
        ax.set(title=title,ylabel=y,xlabel='Time [s]');ax.axvline(onset,color='gray',ls=':');ax.grid(alpha=.2);ax.legend(title='Static / dynamic',fontsize=8)
    axs[2].axhline(.7,color='black',ls='--',lw=.8);axs[2].axhline(.9,color='black',ls=':',lw=.8)
    fig.suptitle('First sustained left low-slip → slip event after braking in the 0.9/0.7 probe\nSame frozen actor, same reset | Force averages following 2.5 ms | No observed solver mode flag')
    fig.savefig(a.output/'slip_onset_4x1.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(24,4.8),constrained_layout=True)
    t=signals['case0_time_start_s'];angle=np.linspace(0,2*np.pi,361)
    for k,color in enumerate(colors):
        label=f"{rows[k]['mu_static']:g} / {rows[k]['mu_dynamic']:g}"
        axs[0].plot(np.r_[t,t[-1]+.0025],signals[f'case{k}_COM_speed'],color=color,label=label)
        for side in (0,1):
            mask=(t>=.4)&(t<1.8)
            axs[side+1].plot(t[mask],signals[f'case{k}_point_tangent_interval_rms'][mask,side],color=color,label=label)
        axs[3].bar(k-.16,rows[k]['diagnostic_braking_slide_s'][0],width=.3,color=color)
        axs[3].bar(k+.16,rows[k]['diagnostic_braking_slide_s'][1],width=.3,color=color,alpha=.45)
    for ax in axs[:3]:ax.grid(alpha=.2);ax.set_xlabel('Time [s]');ax.legend(title='Static / dynamic',fontsize=8)
    axs[0].set(title='Whole-system COM speed',ylabel='XY speed [m/s]',xlim=(.4,2.76))
    axs[1].set(title='Left contact-point slip',ylabel='Interval tangential RMS [m/s]')
    axs[2].set(title='Right contact-point slip',ylabel='Interval tangential RMS [m/s]')
    axs[3].set(title='Near-capacity lateral sliding',ylabel='Braking duration [s]',xticks=range(3),xticklabels=['0.7/0.7','0.9/0.7','0.9/0.9'])
    axs[3].text(.03,.97,'Solid: left; pale: right',transform=axs[3].transAxes,va='top');axs[3].grid(axis='y',alpha=.2)
    fig.suptitle('Same-policy material comparison | All three nominal cases pass with uninterrupted bilateral contact\nDiagnostic slide durations differ from the frozen beta/tilt-conditioned task qualifier')
    fig.savefig(a.output/'material_comparison_4x1.png',dpi=140);plt.close(fig)
    with (a.output/'above_dynamic_below_static.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exceedances[0]),lineterminator='\n');writer.writeheader();writer.writerows(exceedances)
    summary=dict(kind='hockey_same_policy_static_dynamic_effects_v1',source_sha256=sha(__file__),
        analysis_sha256=sha(a.analysis/'report.json'),selection_sha256=sha(a.selection),actor_sha256=actors.pop(),
        same_actor=True,same_reference=True,materials_constant_during_each_rollout=True,
        physics_steps=0,solver_mode_switch_observed=False,instantaneous_22_percent_force_drop_established=False,
        first_left_post_brake_slip_event=selected,
        above_dynamic_intervals=exceedances,above_dynamic_count=len(exceedances),
        above_dynamic_all_below_static=all(e['rho_static']<=1.05 for e in exceedances),
        above_dynamic_all_interval_slip_below_0p02=all(e['point_tangent_interval_rms_m_s']<.02 for e in exceedances),
        interpretation='At fixed normal force the analytical capacity decreases 22.2% from mu_s=0.9 to mu_d=0.7. Actual resultant force depends on demanded force, normal load, patch cancellation and closed-loop response; a matching instantaneous jump is not inferred.',
        cases=rows,artifact_sha256={p.name:sha(p) for p in a.output.iterdir()})
    (a.output/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=rows,above_dynamic_count=len(exceedances),above_dynamic_all_interval_slip_below_0p02=summary['above_dynamic_all_interval_slip_below_0p02'])))


if __name__=='__main__':main()

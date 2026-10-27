#!/usr/bin/env python3
"""Separate pre-instability contact evidence from later sliding in the v2 probe."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load,segments,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    a=p.parse_args();r=a.result
    trace=load(r/'evidence/probes_v2/baseline/final_rollout.npz')
    smoke=load(r/'evidence/probes_v2/smoke/final_rollout.npz')
    s=load(r/'qualification/signals.npz')
    report=json.loads((r/'qualification/report.json').read_text())
    t=s['time_s'];tilt=report['first_45deg_tilt_diagnostic_s']
    before=(t<tilt) if tilt is not None else np.ones(len(t),bool)
    support=s['normal_fz']>5
    corroborated=support&(s['normal_point_lateral_rms']>.05)&(s['aggregate_utilization']>=.9)&(s['aggregate_utilization']<=1.05)
    corroborated&=(s['anchor_power']<-.05)&(s['dissipative_slipping_anchor_count']>=1)
    high_beta=s['wheel_beta_valid']&(abs(s['wheel_axial_beta_deg'])>=20)
    out=dict(kind='hockey_stop_v2_phase_separated_evidence',analyzer_sha256=sha(__file__),
        qualification_sha256=sha(r/'qualification/report.json'),
        first_45deg_tilt_s=tilt,planned_braking_start_s=.4,
        instability_precedes_planned_braking=bool(tilt is not None and tilt<.4),
        final_speed_m_s=float(s['body_speed_m_s'][np.searchsorted(t,2.76)]),
        maximum_tilt_deg=float(s['root_tilt_deg'].max()),
        smoke_vs_full_initial_root_position_max_difference_m=float(abs(smoke['root_pos']-trace['root_pos'][:len(smoke['root_pos'])]).max()),
        support_gaps=segments(t,(t-.0025>=.02-1e-9)&~support.any(-1),interval_samples=True),
        wheels=[])
    for k,side in enumerate(('left','right')):
        mask=corroborated[:,k]&before&s['native_sample_valid']
        ev=segments(t,mask,interval_samples=True)
        row=dict(side=side,pre_first_tilt_slip_saturation_dissipation_events=ev,
            pre_first_tilt_longest_corroborated_s=max((e['duration_s'] for e in ev),default=0),
            pre_first_tilt_high_beta_corroborated_events=segments(t,mask&high_beta[:,k],interval_samples=True))
        if ev:
            best=max(ev,key=lambda e:e['duration_s']);sl=slice(best['first_index'],best['last_index']+1)
            row['longest_pre_first_tilt_event_ranges']={key:[float(s[key][sl,k].min()),float(s[key][sl,k].max())]
                for key in ('normal_fz','normal_point_lateral_rms','aggregate_utilization','anchor_power')}
        row['instantaneously_upright_braking_events_after_first_tilt']=[e for e in report['qualification']['bilateral_loaded_slip_events'][k] if tilt is not None and e['start_s']>=tilt]
        out['wheels'].append(row)
    (r/'phase_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4),layout='constrained')
    colors=('#2469ba','#cb6a22')
    panels=[('normal_fz','Wheel load [N]',5),('normal_point_lateral_rms','Lateral material speed [m/s]',.05),
        ('aggregate_utilization','Aggregate friction utilization',.9),('wheel_axial_beta_deg','Wheel axial beta [deg]',20)]
    for ax,(key,title,threshold) in zip(axes,panels):
        for k,color in enumerate(colors):
            y=s[key][:,k].copy()
            if key in ('aggregate_utilization','wheel_axial_beta_deg'):y[~support[:,k]]=np.nan
            ax.plot(t,y,color=color,lw=1,label=('Left','Right')[k])
        ax.axhline(threshold,color='gray',ls='--',lw=.8)
        if key=='aggregate_utilization':ax.axhline(1.05,color='gray',ls=':',lw=.8)
        if tilt is not None:ax.axvspan(tilt,.4,color='red',alpha=.06);ax.axvline(tilt,color='red',ls=':',lw=1)
        ax.set(title=title,xlabel='Time [s]',xlim=(0,.4));ax.grid(alpha=.2)
        subset=s[key][(t<=.4)&s['native_sample_valid']]
        if key=='aggregate_utilization':ax.set_ylim(0,max(1.1,float(subset.max())*1.05))
        elif key=='wheel_axial_beta_deg':ax.set_ylim(-95,95)
        elif key!='normal_fz':ax.set_ylim(0,float(subset.max())*1.05+.01)
        ax.legend(fontsize=8)
    fig.suptitle('Before planned braking at 0.4 s: short loaded sliding, then first 45-degree tilt at 0.275 s')
    fig.savefig(r/'pre_braking_contact_4x1.png',dpi=150);plt.close(fig)
    print(json.dumps({k:out[k] for k in ('first_45deg_tilt_s','final_speed_m_s','maximum_tilt_deg','smoke_vs_full_initial_root_position_max_difference_m')},indent=2))


if __name__=='__main__':main()

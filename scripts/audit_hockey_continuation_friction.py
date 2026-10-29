"""Compare force-capacity evidence before/after the fixed-reward continuation."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_hockey_contact_cone_relation import read_case
from evaluate_hockey_physx import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cases-json','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sources=json.loads(a.cases_json.read_text())
    if len(sources)!=2:raise ValueError('Require before/after recordings')
    import torch
    torch.set_num_threads(2)
    cases=[];arrays=[]
    for source in sources:
        row,signals=read_case(source,a.cases_json,a.reference)
        if row['mu_static']!=.7 or row['mu_dynamic']!=.7:
            raise ValueError('Expected equal-friction cylinder comparison at 0.7')
        cases.append(row);arrays.append(signals)
    report=dict(kind='fixed_reward_continuation_friction_audit_v1',source_sha256=sha(Path(__file__)),
        input_selection_sha256=sha(a.cases_json),reference_sha256=sha(a.reference/'motion.npz'),
        dependencies={name:sha(Path(__file__).with_name(name)) for name in
            ('audit_hockey_contact_cone_relation.py','analyze_hockey_ground_contact.py',
             'audit_hockey_contact_timing.py','audit_hockey_contact_liftoff.py','evaluate_hockey_physx.py',
             'audit_physx_contact_float32.py','physx_reference_math.py','wheel_friction_capacity.py')},
        physics_steps=0,training_changed=False,success_gates_changed=False,
        pointwise_cone_certified=False,solver_stick_slip_flag_observed=False,
        contact_tolerance_m=.0001,ratio_division_safeguard_N=1e-6,
        definitions='Same aligned contact/force/velocity diagnostic as the six-screen friction audit. Both endpoints and both wheels required for sliding corroboration; no new task success gate.',
        limitations=['Wheel-resultant and same-anchor dissipation evidence, not a pointwise cone certificate.',
            'Near-capacity ratios require positive normal force; zero force is not redefined as missing geometric contact.',
            'Per-wheel durations can overlap; the diagnostic differs from the frozen beta/tilt-conditioned qualifier.',
            'This pair compares one selected seed before/after additional training, not an isolated reward change or robustness estimate.'],
        cases=cases)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'signals.npz',**{f'case{i}_{k}':v for i,s in enumerate(arrays) for k,v in s.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(23,5.3),constrained_layout=True)
    for row,s,color in zip(cases,arrays,('#b87b47','#286fa0')):
        t=s['time_start_s'];mask=s['braking']
        axes[0].plot(t[mask],s['normal_force_N'][mask,1],color=color,lw=.85,label=row['label'])
        axes[1].plot(t[mask],s['rho_dynamic'][mask,1],color=color,lw=.85,label=row['label'])
    axes[0].set(title='Right-wheel normal force',xlabel='Time [s]',ylabel='Interval-average force [N]')
    axes[1].axhspan(.9,1.05,color='gray',alpha=.15)
    axes[1].set(title='Right-wheel capacity utilization',xlabel='Time [s]',ylabel='rho_d = |sum Ft| / (mu_d sum Fn)')
    for ax in axes[:2]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    row,s=cases[1],arrays[1];phi=np.linspace(0,2*np.pi,361)
    radius=max(1.1,float(np.nanmax(s['rho_dynamic'][s['braking']]))*1.05)
    for side,ax in enumerate(axes[2:]):
        mask=s['braking']&s['ratio_valid'][:,side];capacity=.7*s['normal_force_N'][mask,side]
        ax.plot(np.cos(phi),np.sin(phi),color='black',lw=1)
        sc=ax.scatter(s['force_longitudinal_N'][mask,side]/capacity,s['force_lateral_N'][mask,side]/capacity,
            c=s['time_start_s'][mask],vmin=.4,vmax=1.8,cmap='viridis',s=10)
        ax.set(title='After continuation / '+('left','right')[side],xlabel='F_long / (mu_d Fn)',ylabel='F_lat / (mu_d Fn)',
            xlim=(-radius,radius),ylim=(-radius,radius),aspect='equal');ax.grid(alpha=.2)
    fig.colorbar(sc,ax=axes[2:],shrink=.7,label='Braking time [s]')
    fig.suptitle('Fixed-reward continuation | Contact and force evidence at mu_s = mu_d = 0.7\n'
        'Ratios are undefined at zero normal force; previous outliers remain shown | Wheel-resultant evidence only')
    fig.savefig(a.output/'continuation_friction_4x1.png',dpi=140);plt.close(fig)
    report['artifact_sha256']={x.name:sha(x) for x in sorted(a.output.iterdir())}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(label=c['label'],interruption_ms=c['bilateral_interruption_ms'],
        anomalies=len(c['force_consistency_anomalies']),slide_s=[w['bilateral_endpoint_corroborated_slide_s'] for w in c['wheels']],
        zero_force_intervals=[w['contact_zero_normal_intervals'] for w in c['wheels']]) for c in cases]))


if __name__=='__main__':main()

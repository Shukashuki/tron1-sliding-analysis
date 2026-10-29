"""Independent whole-robot linear-momentum check of archived contact forces.

Masses come from the hash-locked authored USD, not a new simulator readback.
Internal joint forces cancel in the system sum. Whole-body agreement does not
recover individual wheel/patch forces or prove their Coulomb admissibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from evaluate_hockey_physx import load, sha


def authored_masses(text):
    """Read explicit masses in this experiment's flat, pinned USDA hierarchy."""
    parts=re.split(r'^    def Xform "([^\"]+)"',text,flags=re.M)
    result={}
    for name,block in zip(parts[1::2],parts[2::2]):
        values=re.findall(r'^        float physics:mass = ([0-9.eE+-]+)\s*$',block,re.M)
        if len(values)>1:
            raise ValueError('Ambiguous authored mass')
        if values:
            value=float(values[0])
            if name in result or not np.isfinite(value) or value<=0:
                raise ValueError('Invalid or duplicate body mass')
            result[name]=value
    if not result:
        raise ValueError('No explicit body masses')
    return result


def momentum_balance(velocities, masses, normal_forces, tangent_forces, dt, gravity=9.81):
    """Interval j connects velocities j and j+1; forces already use that interval."""
    v=np.asarray(velocities,dtype=float);m=np.asarray(masses,dtype=float)
    fn=np.asarray(normal_forces,dtype=float);ft=np.asarray(tangent_forces,dtype=float)
    if (v.ndim!=3 or v.shape[-1]!=3 or m.shape!=(v.shape[1],)
            or fn.shape!=(v.shape[0]-1,2) or ft.shape!=(*fn.shape,3)
            or not all(np.isfinite(a).all() for a in (v,m,fn,ft))
            or not np.isfinite([dt,gravity]).all() or dt<=0 or gravity<0 or (m<=0).any()):
        raise ValueError('Invalid complete state/force/mass arrays')
    momentum=(v*m[None,:,None]).sum(1)
    measured=np.diff(momentum,axis=0)/dt
    wheel_force=ft.sum(1);wheel_force[:,2]+=fn.sum(1)
    expected=wheel_force.copy();expected[:,2]-=m.sum()*gravity
    return dict(momentum_kg_m_s=momentum, measured_dP_dt_N=measured,
                wheel_force_N=wheel_force, wheel_force_plus_gravity_N=expected,
                residual_N=measured-expected)


def summarize(residual, mask):
    x=np.asarray(residual)[mask]
    if not len(x):
        return None
    norm=np.linalg.norm(x,axis=-1)
    return dict(intervals=len(x),component_RMS_N=np.sqrt(np.mean(x*x,axis=0)).tolist(),
        vector_RMS_N=float(np.sqrt(np.mean(norm*norm))),vector_p95_N=float(np.quantile(norm,.95)),
        vector_max_N=float(norm.max()),mean_residual_N=x.mean(0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cases-json','cone-report','asset','tracking-source','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    read=lambda path:json.loads(path.read_text())
    selection=read(a.cases_json);cone=read(a.cone_report)
    if cone['selection_sha256']!=sha(a.cases_json) or len(selection)!=len(cone['cases']):
        raise ValueError('Mismatched case selection')
    for name,digest in cone['artifact_sha256'].items():
        if sha(a.cone_report.parent/name)!=digest:
            raise ValueError('Prior force audit changed')
    mass_map=authored_masses(a.asset.read_text());source=a.tracking_source.read_text()
    # No general USD composition or Python-config interpretation is claimed.
    if not all(s in source for s in ('linear_damping=0.0','angular_damping=0.0','disable_gravity=False')):
        raise ValueError('Expected explicit no-damping, gravity-enabled source')
    saved={};rows=[];plotted=[]
    aligned=load(a.cone_report.parent/'signals.npz')
    for i,(entry,prior) in enumerate(zip(selection,cone['cases'])):
        bundle=(a.cases_json.parent/entry['bundle']).resolve()
        plan=read(bundle/'plan.json');cell,stage=entry['case'].split(':')
        selected=next(c for c in plan['cells'] if c['id']==cell)
        cfg=read(bundle/selected['protocol'])
        if cfg['asset_sha256']!=sha(a.asset) or cfg['tracking_sha256']!=sha(a.tracking_source):
            raise ValueError('Authored asset or tracking source differs from frozen protocol')
        directory=(a.cases_json.parent/entry['evidence']/entry['run_name']/cell/stage).resolve()
        run=read(directory/'report.json');q=read(directory.parent/'nominal_qualification.json')
        if (prior['cell']!=cell or prior['raw_sha256']!=run['artifacts']
                or prior['qualification_sha256']!=sha(directory.parent/'nominal_qualification.json')
                or q['run_report_sha256']!=sha(directory/'report.json')
                or q['protocol_sha256']!=sha(bundle/selected['protocol'])
                or prior['source_manifest_sha256']!=sha(bundle/'manifest.json')):
            raise ValueError('Source/recording provenance mismatch')
        for name,digest in run['artifacts'].items():
            if sha(directory/name)!=digest:
                raise ValueError('Recorded trajectory/contact channels changed')
        if run['external_forces']!=0 or run['worlds']!=1:
            raise ValueError('Expected single-world, no imposed external forces')
        tr=load(directory/'final_rollout.npz');names=tr['body_names'].astype(str).tolist()
        if set(names)!=set(mass_map) or len(names)!=len(set(names)):
            raise ValueError('Body/mass identity mismatch; no silent fixed-link merging')
        mass=np.array([mass_map[n] for n in names]);dt=float(tr['physics_dt_s'])
        n=len(aligned[f'case{i}_time_start_s'])
        if n!=1104 or not np.isclose(dt,.0025,atol=1e-12,rtol=0):
            raise ValueError('Require frozen 2.76 s/400 Hz task')
        balance=momentum_balance(tr['body_com_vel'][:n+1],mass,
            tr['normal_fz'][1:n+1],tr['friction_force_w'][1:n+1],dt)
        brake=aligned[f'case{i}_braking'];rho=aligned[f'case{i}_rho_static']
        anomalies=brake&(rho>1.05).any(-1)
        # Temporal negative controls retain exactly the same braking intervals.
        if brake[0] or brake[-1]:
            raise ValueError('Braking interval must allow both one-step shifts')
        observed=balance['measured_dP_dt_N'];expected=balance['wheel_force_plus_gravity_N']
        shifted={str(lag):summarize(observed-np.roll(expected,-lag,axis=0),brake) for lag in (-1,0,1)}
        k=int(np.argmax(np.where(brake,np.max(np.where(np.isfinite(rho),rho,0),axis=-1),-1)))
        row=dict(cell=cell,label=entry['label'],body_names=names,authored_body_masses_kg=mass.tolist(),
            total_authored_mass_kg=float(mass.sum()),qualification_sha256=prior['qualification_sha256'],
            raw_sha256=run['artifacts'],source_manifest_sha256=prior['source_manifest_sha256'],
            whole_task=summarize(balance['residual_N'],np.ones(n,dtype=bool)),
            braking=summarize(balance['residual_N'],brake),
            static_bound_exceedance_intervals=summarize(balance['residual_N'],anomalies),
            braking_force_time_shift_steps=shifted,
            cumulative_residual_impulse_kg_m_s=(balance['residual_N'].sum(0)*dt).tolist(),
            largest_static_ratio_sample=dict(time_start_s=k*dt,rho_static_per_wheel=[float(v) if np.isfinite(v) else None for v in rho[k]],
                observed_dP_dt_N=observed[k].tolist(),summed_contact_plus_gravity_N=expected[k].tolist(),
                residual_N=balance['residual_N'][k].tolist()))
        rows.append(row);plotted.append((entry['label'],balance,brake))
        for key,value in balance.items():saved[f'case{i}_{key}']=value
    report=dict(kind='hockey_whole_robot_momentum_audit_v1',source_sha256=sha(Path(__file__)),
        asset_sha256=sha(a.asset),tracking_source_sha256=sha(a.tracking_source),
        cone_report_sha256=sha(a.cone_report),selection_sha256=sha(a.cases_json),
        dependency_sha256={'evaluate_hockey_physx.py':sha(Path(__file__).with_name('evaluate_hockey_physx.py'))},
        sample_dt_s=.0025,gravity_m_s2=9.81,authored_body_masses_kg=mass_map,
        runtime_mass_readback_performed=False,additional_simulator_steps=0,training_changed=False,
        success_gate_changed=False,pointwise_cone_certified=False,
        equation='sum_b m_b (v_b[k+1]-v_b[k])/dt = sum_w (F_t,w[k] + F_n,w[k] e_z) + M g + residual',
        scope='Whole-system linear force balance from independently recorded body COM velocities and authored masses',
        limitations=['Authored masses are hash-locked; this is not a new live mass/gravity/solver-flag readback.',
            'Internal joint forces and motor reactions cancel in a whole-system linear momentum sum.',
            'Small residual validates the net external-force accounting, not each wheel/patch force allocation.',
            'Equal-and-opposite force channel errors could cancel. No per-wheel Newton-Euler or angular-momentum audit is performed.',
            'Unrecorded external forces are not individually identified; their net contribution is constrained only by the residual.',
            'Momentum consistency does not prove that the simulator obeys an ideal static/dynamic Coulomb bound.'],cases=rows)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'signals.npz',**saved)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(23,5.4),constrained_layout=True)
    colors=plt.cm.tab10(np.arange(6));t=np.arange(1104)*.0025
    for color,(label,balance,brake) in zip(colors,plotted):
        for component in range(3):
            axes[component].plot(t[brake],balance['residual_N'][brake,component],color=color,lw=.65,label=label)
            axes[component].set(title=('X','Y','Z')[component]+' momentum residual',xlabel='Time [s]',ylabel='dP/dt - contact - gravity [N]')
    x=np.arange(6)
    for offset,lag,color in ((-.25,-1,'#b5a07b'),(0,0,'#338299'),(.25,1,'#c68164')):
        axes[3].bar(x+offset,[r['braking_force_time_shift_steps'][str(lag)]['vector_RMS_N'] for r in rows],
            width=.25,color=color,label=f'Force shift {lag:+d} step')
    axes[3].set_xticks(x,['Eq v6','Eq v7','Un v6','Un v7','Sl v6','Sl v7'],rotation=30)
    axes[3].set(title='Alignment negative controls',ylabel='Braking vector RMS residual [N]',yscale='log')
    axes[0].legend(fontsize=7);axes[3].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Whole-robot momentum vs recorded contact forces | Authored mass 22.273 kg | No new simulation\n'
                 'Agreement checks the net applied force; it does not certify individual wheel/patch Coulomb limits')
    fig.savefig(a.output/'momentum_comparison_4x1.png',dpi=140);plt.close(fig)
    report['artifact_sha256']={p.name:sha(p) for p in sorted(a.output.iterdir())}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(label=r['label'],braking_RMS_N=r['braking']['vector_RMS_N'],
        shifted_RMS_N={k:v['vector_RMS_N'] for k,v in r['braking_force_time_shift_steps'].items()}) for r in rows]))


if __name__=='__main__':main()

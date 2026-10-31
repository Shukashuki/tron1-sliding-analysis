"""Compare contact geometry/law/time-step factors without policy confounding."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_wheel_model_factors import reduce_native


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(folder):
    meta=json.loads((folder/'status.json').read_text())
    assert sha(folder/'native_trace.npz')==meta['native_trace_sha256']
    with np.load(folder/'native_trace.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    q=a['root_state_w'].astype(float);dt=meta['physics_dt_s'];count=len(q)-1
    assert count==round(meta['duration_s']/dt)
    np.testing.assert_allclose(a['time_s'],np.arange(count+1)*dt,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(np.diff(a['native_step_counters']),np.ones(count-1))
    native=a['normal_force_w_n']+a['friction_force_w_n'];external=a['applied_tangent_force_w_n']
    measured=meta['mass_kg']*np.diff(q[:,7:10],axis=0)/dt;measured[:,2]+=meta['mass_kg']*9.81
    residual=measured-native-external
    normal_load=np.zeros(count);anchor_norm=np.zeros(count);max_tilt=0.;sliding=np.zeros(count,dtype=bool)
    gaps=q[:,2]-.127
    geometric=(gaps[:-1]<=.0001)&(gaps[1:]<=.0001)
    force_points=np.zeros(count,dtype=int)
    for k in range(count):
        reduced=reduce_native(tuple(a[f'normal_{i}'][k] for i in range(6)),tuple(a[f'friction_{i}'][k] for i in range(4)))
        np.testing.assert_allclose(reduced['normal_force_w_n'],a['normal_force_w_n'][k],atol=1e-5,rtol=1e-6)
        np.testing.assert_allclose(reduced['friction_force_w_n'],a['friction_force_w_n'][k],atol=1e-5,rtol=1e-6)
        load=reduced['normal_scalars'];normal_load[k]=load.sum()
        force_points[k]=len(load)
        anchor_norm[k]=np.linalg.norm(reduced['friction_vectors'],axis=1).sum()
        active=load>.001
        if not active.any():continue
        normals=reduced['normal_normals'][active].astype(float)
        max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(normals[:,2],-1,1))).max()))
        points=reduced['normal_points'][active]
        v0=q[k,7:10]+np.cross(q[k,10:13],points-q[k,:3])
        v1=q[k+1,7:10]+np.cross(q[k+1,10:13],points-q[k+1,:3])
        v0-=np.sum(v0*normals,axis=1)[:,None]*normals
        v1-=np.sum(v1*normals,axis=1)[:,None]*normals
        sliding[k]=geometric[k] and bool(((np.linalg.norm(v0,axis=1)>.05)&(np.linalg.norm(v1,axis=1)>.05)).all())
    valid=normal_load>.001
    native_ratio=np.divide(anchor_norm,meta['mu']*normal_load,out=np.full(count,np.nan),where=valid)
    prior=a['preceding_scalar_normal_load_n']
    external_ratio=np.divide(np.linalg.norm(external,axis=1),meta['mu']*prior,
                            out=np.full(count,np.nan),where=prior>.001)
    no_current_contact=(normal_load<=.001)&(np.linalg.norm(external,axis=1)>.001)
    stop=np.flatnonzero(np.abs(q[:,8])<=.05)
    def finite_max(x):return float(np.nanmax(x)) if np.isfinite(x).any() else None
    metrics=dict(geometry_slices=meta['geometry_slices'],law=meta['law'],physics_dt_s=dt,
        native_trace_sha256=sha(folder/'native_trace.npz'),final_xy_m=q[-1,:2].tolist(),
        final_velocity_xy_m_s=q[-1,7:9].tolist(),first_lateral_speed_below_0p05_s=float(stop[0]*dt) if len(stop) else None,
        geometric_contact_fraction=float(geometric.mean()),force_active_fraction=float(valid.mean()),
        contact_conditioned_slip_duration_s=float(sliding.sum()*dt),maximum_loaded_normal_tilt_deg=max_tilt,
        native_anchor_norm_budget_max=finite_max(native_ratio),
        external_previous_load_capacity_ratio_max=finite_max(external_ratio),
        external_force_without_current_normal_sample_s=float(no_current_contact.sum()*dt),
        external_pre_step_power_max_w=float(a['external_contact_power_w'].max()),
        external_pre_step_dissipated_work_j=float(-a['external_contact_power_w'].sum()*dt),
        momentum_residual_rms_by_axis_n=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        peak_native_contact_points=int(force_points.max()),gap_min_max_m=[float(gaps.min()),float(gaps.max())],
        mass_kg=meta['mass_kg'],inertia=meta['inertia'],initial_state=meta['initial_state'])
    assert max(metrics['momentum_residual_rms_by_axis_n'])<.002
    assert metrics['external_pre_step_power_max_w']<1e-8
    return metrics,a


def run(root,output):
    suite=json.loads((root/'status.json').read_text())
    assert suite['status']=='completed_requires_analysis' and len(suite['cases'])==24
    if output.exists():raise FileExistsError(output)
    rows=[];data={}
    for case in suite['cases']:
        folder=root/case['name']
        assert case['status']=='completed_verified' and sha(folder/'status.json')==case['status_sha256']
        metrics,trace=analyze(folder)
        assert metrics['native_trace_sha256']==case['trace_sha256']
        if case['name'].startswith('smoke_'):continue
        rows.append(dict(name=case['name'],**metrics));data[case['name']]=trace
    assert len(rows)==18
    for row in rows:
        for name in ('mass_kg','inertia','initial_state'):
            np.testing.assert_allclose(row[name],rows[0][name],atol=1e-7,rtol=0)
    output.mkdir(parents=True)
    report=dict(status='completed_verified',kind='policy_free_wheel_factorial',formal_cases=18,smoke_cases=6,
        source_suite_sha256=sha(root/'status.json'),rows=rows,
        matched_mass_inertia_initial_state=True,ground_seam_offset_y_m=-5,contact_offset_m=.001,
        native_vs_external_capacity_denominators_are_different=True,
        limitations=['Native friction uses current native scalar normal load; external law uses the preceding physical interval.',
            'Aggregate anchor norm sums do not identify individual friction-patch saturation.',
            'A loaded, geometrically eligible interval with point velocity is required for the slip screen.',
            'External law has no static adhesion or tire identification; pre-step dissipativity does not prove discrete-passive coupling.',
            'Axle-constrained 1.08 kg fixture is not whole-body TRON1 or real-tire validation.'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for hz in (200,400,800):
        fig,axs=plt.subplots(1,4,figsize=(20,4.2))
        for row in rows:
            if round(1/row['physics_dt_s'])!=hz:continue
            a=data[row['name']];q=a['root_state_w'];t=a['time_s']
            label=f"{row['geometry_slices']} slice(s) / "+row['law'].replace('combined_slip_magic_formula','nonlinear').replace('external_coulomb_regularized','external Coulomb')
            axs[0].plot(q[:,0],q[:,1],label=label)
            axs[1].plot(t,q[:,8]);axs[2].plot(t[1:],a['friction_force_w_n'][:,1]+a['applied_tangent_force_w_n'][:,1])
            axs[3].plot(t[1:],a['normal_force_w_n'][:,2])
        for ax,title,xlabel,ylabel in zip(axs,['XY path','Lateral velocity','Tangential force Y','Normal force Z'],
                ['X (m)','Time (s)','Time (s)','Time (s)'],['Y (m)','m/s','N','N']):
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2)
        axs[0].set_aspect('equal',adjustable='datalim')
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
        fig.suptitle(f'Policy-free wheel | {hz} Hz | mu=0.7 | matched mass/inertia | seam shifted away')
        fig.tight_layout(rect=(0,.15,1,.94));fig.savefig(output/f'wheel_factors_{hz}hz.png',dpi=140);plt.close(fig)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.suite,a.output_dir)

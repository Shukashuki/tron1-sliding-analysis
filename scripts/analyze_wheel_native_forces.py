"""Audit native wheel forces, initialization history, and matched CPU MuJoCo.

No time-shifting, transient deletion, or fitted parameters are used to improve
the nominal result. Shifted forces are deliberately incorrect timing controls.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_wheel_native_forces import reduce_native
from benchmark_wheel_backend_foundation import simulate, trajectory_gap, load_archived


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(a): return np.sqrt(np.mean(np.asarray(a)**2,axis=0)).tolist()


def analyze(trace, meta):
    q=trace['root_state_w'].astype(float);dt=meta['physics_dt_s'];mass=meta['mass_kg']
    normal=trace['normal_force_w_n'].astype(float);friction=trace['friction_force_w_n'].astype(float)
    force=normal+friction
    measured=mass*np.diff(q[:,7:10],axis=0)/dt;measured[:,2]+=mass*9.81
    residual=measured-force
    loaded=normal[:,2]>.001
    gaps=q[:,2]-.127
    contact=(gaps[:-1]<=.0001)&(gaps[1:]<=.0001)
    eligible=np.zeros(len(force),dtype=bool);sum_norm=np.zeros(len(force));counts=[]
    normal_scalar_sum=np.zeros(len(force));tilted_load=np.zeros(len(force));max_tilt=0.
    tilt_per_step=np.full(len(force),np.nan)
    for k in range(len(force)):
        n=tuple(trace[f'normal_{i}'][k] for i in range(6))
        f=tuple(trace[f'friction_{i}'][k] for i in range(4))
        read=reduce_native(n,f)
        np.testing.assert_allclose(read['normal_force_w_n'],normal[k],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(read['friction_force_w_n'],friction[k],atol=1e-6,rtol=1e-6)
        active=np.linalg.norm(read['friction_vectors'],axis=1)>.001
        points=read['friction_points'][active].astype(float)
        sum_norm[k]=np.linalg.norm(read['friction_vectors'],axis=1).sum()
        normal_scalar_sum[k]=read['normal_scalars'].sum()
        tilts=np.rad2deg(np.arccos(np.clip(read['normal_normals'][:,2],-1.,1.)))
        carrying=read['normal_scalars']>.001
        if carrying.any():
            tilt_per_step[k]=float(tilts[carrying].max());max_tilt=max(max_tilt,tilt_per_step[k])
        tilted_load[k]=read['normal_scalars'][tilts>5.].sum()
        counts.append([read['normal_count'],read['friction_count']])
        if not (loaded[k] and contact[k] and len(points)): continue
        # Native point geometry is supplied by the completed solve. Evaluate
        # both endpoint velocity estimates; do not invent patch-normal pairing.
        pre=q[k,7:10]+np.cross(q[k,10:13],points-q[k,:3])
        post=q[k+1,7:10]+np.cross(q[k+1,10:13],points-q[k+1,:3])
        s0=np.linalg.norm(pre[:,:2],axis=1);s1=np.linalg.norm(post[:,:2],axis=1)
        coherent=(pre[:,:2]*post[:,:2]).sum(axis=1)>=np.cos(np.deg2rad(30))*s0*s1
        eligible[k]=bool(((s0>.05)&(s1>.05)&coherent).all())
    ratios=np.full(len(force),np.nan);valid=normal_scalar_sum>.001
    ratios[valid]=np.linalg.norm(friction[valid],axis=1)/(.7*normal_scalar_sum[valid])
    anchor_budget=np.full(len(force),np.nan)
    anchor_budget[valid]=sum_norm[valid]/(.7*normal_scalar_sum[valid])
    def ratio_summary(mask):
        r=ratios[mask&np.isfinite(ratios)]
        return dict(intervals=int(mask.sum()),duration_s=float(mask.sum()*dt),
                    max=float(r.max()) if len(r) else None,
                    p50=float(np.median(r)) if len(r) else None,
                    above_1p05=int((r>1.05).sum()))
    timing={}
    for shift in (-1,0,1):
        if shift==-1: err=measured[1:]-force[:-1]
        elif shift==1: err=measured[:-1]-force[1:]
        else:err=residual
        timing[str(shift)]=rms(err)
    # Predeclared diagnostic screens, not a real-world accuracy certificate.
    threshold=.01*mass*9.81
    stop=np.flatnonzero(np.abs(q[:,8])<=.05)
    out=dict(momentum_residual_rms_by_axis_n=rms(residual),
             momentum_residual_peak_by_axis_n=np.abs(residual).max(axis=0).tolist(),
             peak_residual_after_20ms_by_axis_n=np.abs(residual[np.arange(len(force))*dt>=.02]).max(axis=0).tolist(),
             force_timing_shift_rms_by_axis_n=timing,
             consistency_RMS_screen_threshold_n=threshold,
             consistency_RMS_screen_passed=bool(np.max(np.sqrt(np.mean(residual**2,axis=0)))<=threshold),
             normal_sum_matrix_max_difference_n=float(np.abs(normal-trace['matrix_force_w_n']).max()),
             maximum_loaded_normal_tilt_deg=max_tilt,
             scalar_normal_impulse_fraction_above_5deg=float(tilted_load.sum()/normal_scalar_sum.sum()) if normal_scalar_sum.sum()>0 else None,
             maximum_abs_friction_world_z_n=float(np.abs(friction[:,2]).max()),
             maximum_abs_normal_world_xy_n=float(np.linalg.norm(normal[:,:2],axis=1).max()),
             anchor_norm_budget_max=float(np.nanmax(anchor_budget)) if valid.any() else None,
             sliding_native_anchor_screen=ratio_summary(eligible),
             loaded_resultant_ratio=ratio_summary(loaded),
             geometric_contact_fraction=float(contact.mean()),normal_force_active_fraction=float(loaded.mean()),
             gap_min_max_m=[float(gaps.min()),float(gaps.max())],
             max_normal_friction_counts=np.max(counts,axis=0).tolist(),
             final_xy_m=q[-1,:2].tolist(),final_velocity_xy_m_s=q[-1,7:9].tolist(),
             first_abs_vy_below_0p05_s=float(stop[0]*dt) if len(stop) else None)
    return out,dict(force_w_n=force,momentum_force_w_n=measured,residual_w_n=residual,
                    utilization_resultant=ratios,sliding_anchor_screen=eligible,
                    sum_anchor_tangent_norm_n=sum_norm,normal_scalar_sum_n=normal_scalar_sum,
                    utilization_anchor_norm_sum=anchor_budget,max_loaded_normal_tilt_deg=tilt_per_step)


def run(a):
    if a.output_dir.exists(): raise FileExistsError('Fresh output required')
    suite=json.loads((a.suite/'status.json').read_text())
    assert suite['status']=='completed_requires_analysis' and len(suite['cases'])==10
    a.output_dir.mkdir(parents=True)
    traces={};metadata={};metrics={};signals={}
    for row in suite['cases']:
        name=row['name'];path=a.suite/name
        meta=json.loads((path/'status.json').read_text())
        assert row['status']=='completed' and meta['status']=='completed_requires_analysis'
        assert meta['source_sha256']==suite['probe_sha256']
        assert sha(path/'native_trace.npz')==row['trace_sha256']==meta['native_trace_sha256']
        with np.load(path/'native_trace.npz',allow_pickle=False) as z: tr={k:z[k].copy() for k in z.files}
        n=round(meta['duration_s']/meta['physics_dt_s'])
        assert len(tr['time_s'])==len(tr['root_state_w'])==n+1
        np.testing.assert_array_equal(np.diff(tr['native_step_counters']),np.ones(n-1))
        metrics[name],signals[name]=analyze(tr,meta)
        traces[name]=tr;metadata[name]=meta
        np.savez_compressed(a.output_dir/f'{name}_analysis.npz',**signals[name])
    archive_comparison=[]
    base=Path(__file__).resolve().parents[1]/'results/physx_tire_comparison_20261024_v1/raw'
    for hz in (400,800):
        old,om=load_archived(base/'single_wheel_bench_collection_v1.tar.gz',base/'single_wheel_bench_collection_v1.manifest.json',hz)
        new=traces[f'legacy_{hz}hz']
        archive_comparison.append(dict(hz=hz,
            max_state_difference=float(np.abs(old['root_state_w']-new['root_state_w']).max()),
            old_normal_z_vs_native_normal_z_max_n=float(np.abs(old['interval_force_w_n'][:,2]-new['normal_force_w_n'][:,2]).max()),
            old_xy_total_minus_native_normal_xy_vs_native_friction_xy_RMS_n=rms(
                old['interval_force_w_n'][:,:2]-new['normal_force_w_n'][:,:2]-new['friction_force_w_n'][:,:2]),
            source_hashes=om['source_hashes']))
    history=[];numerics=[];engine_pairs=[]
    for hz in (200,400,800):
        old=f'legacy_{hz}hz';matched=f'cold_replay_{hz}hz';cold=f'cold_{hz}hz'
        np.testing.assert_array_equal(traces[old]['root_state_w'][0],traces[matched]['root_state_w'][0])
        assert metadata[matched]['pre_initialization_contact_count']==metadata[cold]['pre_initialization_contact_count']==0
        history.append(dict(hz=hz,initial_state_max_difference=0.,legacy=old,cold_matched=matched,
                            **trajectory_gap(traces[old],traces[matched])))
        for source in (old,cold):
            mj,mm,xml=simulate(traces[source]['root_state_w'][0],1/hz)
            key=f'mujoco_{source}'
            np.savez_compressed(a.output_dir/f'{key}.npz',**mj)
            (a.output_dir/f'{key}.xml').write_text(xml)
            traces[key]=mj;metadata[key]=mm
            engine_pairs.append(dict(physx=source,mujoco=key,**trajectory_gap(traces[source],mj)))
        if hz!=800:
            np.testing.assert_array_equal(traces[cold]['root_state_w'][0],traces['cold_800hz']['root_state_w'][0])
            numerics.append(dict(coarse=cold,fine='cold_800hz',**trajectory_gap(traces[cold],traces['cold_800hz'])))
    seam=[]
    if a.shifted_suite:
        extra=json.loads((a.shifted_suite/'status.json').read_text())
        assert extra['status']=='completed_requires_analysis' and len(extra['cases'])==4
        for row in extra['cases']:
            if 'smoke' in row['name']:continue
            base_name=row['name'];name='shifted_'+base_name;path=a.shifted_suite/base_name
            meta=json.loads((path/'status.json').read_text())
            assert meta['status']=='completed_requires_analysis' and meta['floor_y_offset_m']==-5.
            assert meta['source_sha256']==extra['probe_sha256']
            assert sha(path/'native_trace.npz')==row['trace_sha256']==meta['native_trace_sha256']
            with np.load(path/'native_trace.npz',allow_pickle=False) as z:tr={k:z[k].copy() for k in z.files}
            np.testing.assert_array_equal(tr['root_state_w'][0],traces[base_name]['root_state_w'][0])
            for field in ('physics_dt_s','duration_s','mass_kg','inertia','com_frame','materials','contact_offsets_m',
                          'rest_offsets_m','rigid_solver_attributes','scene_physx_cfg'):
                assert meta[field]==metadata[base_name][field],field
            metrics[name],signals[name]=analyze(tr,meta);metadata[name]=meta;traces[name]=tr
            np.savez_compressed(a.output_dir/f'{name}_analysis.npz',**signals[name])
            mj_name='mujoco_'+base_name
            pair=dict(physx=name,mujoco=mj_name,**trajectory_gap(tr,traces[mj_name]));engine_pairs.append(pair)
            seam.append(dict(original=base_name,shifted=name,initial_state_max_difference=0.,
                             **trajectory_gap(traces[base_name],tr)))
    report=dict(status='native_force_initialization_audit_completed',new_full_PhysX_episodes=9,
        short_PhysX_smokes_in_this_suite=1,new_MuJoCo_CPU_episodes=6,metrics=metrics,
        additional_seam_control_episodes=3 if a.shifted_suite else 0,
        additional_seam_control_smokes=1 if a.shifted_suite else 0,
        archived_state_and_force_comparison=archive_comparison,seam_controls=seam,
        matched_initial_state_history_comparison=history,cold_PhysX_timestep_comparison=numerics,
        engine_pairs=engine_pairs,metadata=metadata,suite_status_sha256=sha(a.suite/'status.json'),
        source_sha256={p:sha(Path(__file__).with_name(p)) for p in
            ('analyze_wheel_native_forces.py','probe_wheel_native_forces.py','benchmark_wheel_backend_foundation.py')},
        limits=['Neither engine is real-world ground truth.',
                'Matching root state does not match solver contact cache; this contrast deliberately changes contact history.',
                'Native friction anchors lack individual normal-patch assignment; resultant ratios are not per-patch certificates.',
                'Endpoint anchor velocity estimates do not expose solver-internal velocities.',
                'USD solver defaults and scene limits are recorded; effective native per-body iteration counts are not read back.',
                'The low-load wheel fixture does not establish whole-robot stability or policy transfer.'])
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4.4),dpi=150)
    plot_names=('legacy_400hz','cold_400hz','shifted_cold_400hz','mujoco_cold_400hz') if a.shifted_suite else ('legacy_400hz','cold_replay_400hz','cold_400hz','mujoco_cold_400hz')
    for name in plot_names:
        tr=traces[name];q=tr['root_state_w'];t=tr['time_s']
        kw=dict(label=name.replace('_',' '))
        f=signals[name]['force_w_n'] if name in signals else tr['interval_force_w_n']
        axes[0].plot(q[:,0],q[:,1],**kw);axes[1].plot(t,q[:,8],**kw)
        axes[2].plot(t[:-1],f[:,1],**kw);axes[3].plot(t[:-1],f[:,2],**kw)
    axes[0].set(title='XY (unequal axis scales)',xlabel='X (m)',ylabel='Y (m)')
    axes[1].set(title='Lateral velocity',xlabel='Time (s)',ylabel='Vy (m/s)',xlim=(0,.15))
    axes[2].set(title='Total contact Fy',xlabel='Time (s)',ylabel='Fy (N)',xlim=(0,.15))
    axes[3].set(title='Total contact Fz',xlabel='Time (s)',ylabel='Fz (N)',xlim=(0,.15))
    axes[1].legend(fontsize=6)
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('400 Hz wheel diagnostic | initialization history and native force readback | mu = 0.7')
    fig.tight_layout();fig.savefig(a.output_dir/'native_wheel_4x1.png');plt.close(fig)
    if a.shifted_suite:
        fig,axes=plt.subplots(1,4,figsize=(20,4.4),dpi=150)
        for name in ('cold_400hz','shifted_cold_400hz'):
            tr=traces[name];t=tr['time_s'][:-1];sig=signals[name]
            kw=dict(label=name.replace('_',' '))
            axes[0].plot(t,sig['max_loaded_normal_tilt_deg'],**kw)
            axes[1].plot(t,np.linalg.norm(tr['normal_force_w_n'][:,:2],axis=1),**kw)
            axes[2].plot(t,tr['friction_force_w_n'][:,2],**kw)
            axes[3].plot(t,sig['utilization_anchor_norm_sum'],**kw)
        for ax in axes:ax.set_xlim(0,.15);ax.set_xlabel('Time (s)');ax.grid(alpha=.25)
        axes[0].set(title='Loaded normal tilt from world Z',ylabel='Degrees');axes[0].legend(fontsize=7)
        axes[1].set(title='Normal force in world XY',ylabel='N')
        axes[2].set(title='Friction force in world Z',ylabel='N')
        axes[3].set(title='Sum |friction| / (mu sum Fn)',ylabel='Aggregate capacity ratio')
        axes[3].axhline(1,color='black',ls=':',lw=1)
        fig.suptitle('Coplanar triangle-mesh seam control | 400 Hz | same wheel, initial state, material and solver')
        fig.tight_layout();fig.savefig(a.output_dir/'contact_geometry_4x1.png');plt.close(fig)
    print(json.dumps(dict(history=history,engine_pairs=engine_pairs,
        native_RMS={k:v['momentum_residual_rms_by_axis_n'] for k,v in metrics.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--shifted-suite',type=Path)
    run(p.parse_args())

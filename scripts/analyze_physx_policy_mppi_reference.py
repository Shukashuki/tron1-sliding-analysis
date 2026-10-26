"""Qualify PhysX candidate tracking and wheel-level sliding corroboration.

Native friction anchors cannot be assigned individual normal loads from these
APIs. These are aggregate indicators, explicitly NOT pointwise cone proofs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from physx_reference_math import trajectory_metrics


def longest_run(mask,dt=.0025):
    edges=np.flatnonzero(np.diff(np.r_[False,np.asarray(mask,dtype=bool),False]))
    return float(np.max(edges[1::2]-edges[::2],initial=0)*dt)


def sliding(trace):
    time=np.arange(len(trace['root_pos']))*.0025
    core=(time>=1.44-1e-9)&(time<=4.+1e-9)
    speed=np.linalg.norm(trace['root_com_vel'][:,:2],axis=-1)
    rows=[]
    for side in range(2):
        loaded=trace['normal_fz'][:,side]>5.
        lat=trace['lateral_slip_rms'][:,side]>.05
        proxy=trace['utilization_proxy'][:,side]>=.9
        dissipative=trace['friction_anchor_power'][:,side]<-1e-4
        mask=loaded&lat&proxy&dissipative&(speed>=.25)&core
        edge=np.flatnonzero(np.diff(np.r_[False,mask,False]))
        events=[dict(start_s=float(time[first]),end_exclusive_s=float(last*.0025),duration_s=float((last-first)*.0025))
                for first,last in zip(edge[::2],edge[1::2])]
        rows.append(dict(wheel=['left','right'][side],core_support_fraction=float(np.mean(loaded[core])),
            core_loaded_slip_fraction=float(np.mean(lat[loaded&core])) if np.any(loaded&core) else None,
            core_loaded_utilization_proxy_p95=float(np.quantile(trace['utilization_proxy'][loaded&core,side],.95)) if np.any(loaded&core) else None,
            longest_continuous_core_aggregate_corroboration_s=longest_run(mask),
            aggregate_corroboration_events_ge50ms=[e for e in events if e['duration_s']>=.05-1e-12],
            core_corroborated_samples=int(mask.sum()),
            core_poststep_friction_power_integral_proxy_j=float(np.sum(trace['friction_anchor_power'][core,side])*.0025)))
    return rows


def verify_native_buffers(trace,path):
    """Independent NumPy reconstruction from all raw SDK contact buffers."""
    with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    indices=raw['physical_sample_indices']
    assert np.array_equal(indices,np.arange(1,len(trace['root_pos'])))
    bodies=trace['body_names'].astype(str).tolist()
    errors=dict(normal_fz_N=0.,slip_RMS_m_s=0.,anchor_power_W=0.,aggregate_friction_N=0.)
    for side,name in enumerate(('wheel_L_Link','wheel_R_Link')):
        bid=bodies.index(name)
        for row,i in enumerate(indices):
            normal=[raw[f'w{side}_normal_buffer_{j}'][row] for j in range(6)]
            friction=[raw[f'w{side}_friction_buffer_{j}'][row] for j in range(4)]
            count=int(normal[4].reshape(-1)[0]);start=int(normal[5].reshape(-1)[0])
            assert count>=0 and start>=0 and start+count<=len(normal[0])
            fn=normal[0].reshape(-1)[start:start+count].astype(float)
            point=normal[1][start:start+count].astype(float)-trace['environment_origin_w']
            n=normal[2][start:start+count].astype(float)
            assert np.isfinite(fn).all() and np.all(fn>=-1e-4) and np.isfinite(point).all() and np.isfinite(n).all()
            if count:
                lengths=np.linalg.norm(n,axis=-1,keepdims=True)
                assert np.all(lengths[fn>5.] > 1e-8)
                n/=np.maximum(lengths,1e-12)
            fz=float((fn[:,None]*n)[:,2].sum())
            q=trace['body_quat'][i,bid].astype(float);w,x,y,z=q
            axle=np.array([2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)])
            center=trace['body_com_pos'][i,bid].astype(float)
            v=trace['body_com_vel'][i,bid].astype(float);omega=trace['body_omega'][i,bid].astype(float)
            loaded=fn>5.
            lateral=axle[None]-(n@axle)[:,None]*n
            if np.any(loaded):
                direction=lateral[loaded]/np.linalg.norm(lateral[loaded],axis=-1,keepdims=True)
                velocity=v+np.cross(omega,point[loaded]-center)
                lat=(velocity*direction).sum(-1)
                slip=float(np.sqrt(np.sum(fn[loaded]*lat*lat)/fn[loaded].sum()))
            else:slip=0.
            ac=int(friction[2].reshape(-1)[0]);st=int(friction[3].reshape(-1)[0])
            assert ac>=0 and st>=0 and st+ac<=len(friction[0])
            ft=friction[0][st:st+ac].astype(float)
            anchor=friction[1][st:st+ac].astype(float)-trace['environment_origin_w']
            assert np.isfinite(ft).all() and np.isfinite(anchor).all()
            av=v+np.cross(omega,anchor-center)
            power=float(np.sum(ft*av))
            errors['normal_fz_N']=max(errors['normal_fz_N'],abs(fz-trace['normal_fz'][i,side]))
            errors['slip_RMS_m_s']=max(errors['slip_RMS_m_s'],abs(slip-trace['lateral_slip_rms'][i,side]))
            errors['anchor_power_W']=max(errors['anchor_power_W'],abs(power-trace['friction_anchor_power'][i,side]))
            errors['aggregate_friction_N']=max(errors['aggregate_friction_N'],float(np.max(abs(ft.sum(0)-trace['friction_force_w'][i,side]))))
    assert errors['normal_fz_N']<1e-3 and errors['slip_RMS_m_s']<4e-5 and errors['anchor_power_W']<5e-3 and errors['aggregate_friction_N']<1e-3
    return dict(physical_samples_checked=len(indices),wheels=2,maximum_absolute_errors={k:float(v) for k,v in errors.items()},
                all_recorded_buffers_reduced_independently_in_NumPy=True,
                raw_buffers_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def run(a):
    assert not a.output.exists()
    load=lambda p:{k:v for k,v in np.load(p,allow_pickle=False).items()}
    baseline=load(a.baseline_validation/'final_rollout.npz');final=load(a.validation/'final_rollout.npz');reference=load(a.reference)
    search=json.loads((a.search/'report.json').read_text());validation=json.loads((a.validation/'report.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert search['artifacts']['baseline.npz']==sha(a.search/'baseline.npz')
    br=json.loads((a.baseline_validation/'report.json').read_text())
    assert br['single_world_validation'] and br['artifacts']['final_rollout.npz']==sha(a.baseline_validation/'final_rollout.npz')
    np.testing.assert_array_equal(load(a.baseline_validation/'mean.npz')['latent_mean'],np.zeros((12,8)))
    assert validation['artifacts']['final_rollout.npz']==sha(a.validation/'final_rollout.npz')
    assert validation['single_world_validation'] and validation['steps']==349
    assert search['artifacts']['mean.npz']==sha(a.search/'mean.npz')
    np.testing.assert_array_equal(load(a.search/'mean.npz')['latent_mean'],load(a.validation/'mean.npz')['latent_mean'])
    native_verification=verify_native_buffers(final,a.validation/'native_contact_buffers.npz')
    baseline_native_verification=verify_native_buffers(baseline,a.baseline_validation/'native_contact_buffers.npz')
    cases={}
    from analyze_ppo_contact_timestep import band_spectrum
    for key,trace in [('PPO_zero_residual',baseline),('PPO_MPPI_fresh_single_world',final)]:
        m=trajectory_metrics(trace,reference);c=m['core'];w=m['whole']
        m['screen']=dict(core_xy=c['xy_rmse_m']<=.15,whole_xy=w['xy_rmse_m']<=.15,
            peak_xy=w['xy_peak_m']<=.30,speed_retention=.95<=c['speed_retention']<=1.10,
            beta=c['beta_rmse_deg']<=10.,continuous_any_wheel_support=m['longest_support_gap_s']==0.)
        m['listed_numerical_screens_all_pass']=all(m['screen'].values())
        m['angular_jitter_relative_to_original_PPO_gate_evaluated']=False
        m['task_success_claimed']=False
        time=np.arange(len(trace['root_pos']))*.0025
        mask=(time>=1.44-1e-9)&(time<=4.+1e-9)
        m['core_angular_xy_10_to_100Hz']=band_spectrum(trace['root_omega'][mask,:2],.0025)
        m['sliding_aggregate_evidence']=sliding(trace)
        cases[key]=m
    hf_ratio=cases['PPO_MPPI_fresh_single_world']['core_angular_xy_10_to_100Hz']['common_band_rms']/cases['PPO_zero_residual']['core_angular_xy_10_to_100Hz']['common_band_rms']
    report=dict(kind='PhysX_policy_assisted_MPPI_reference_qualification',cases=cases,native_verification=native_verification,
        baseline_native_verification=baseline_native_verification,
        MPPI_to_PPO_baseline_angular_HF_RMS_ratio=hf_ratio,
        tracking_analysis_convention='linear interpolation of locked50Hz car reference onto native400Hz times; unwrap yaw; whole window0..6.96s; legacy generator reports held-reference approximations separately',
        compared_trajectories_both_fresh_single_world=True,
        zero_residual_batch_clone_maximum_position_difference_m=search['search_history'][0]['maximum_clone_root_position_difference_m'],
        search_cost_identical_across_clones=False,
        search_limitation='Zero-residual clone divergence confounds finite-sample MPPI costs. Single-world baseline/final comparisons are exploratory N=1, not a causal improvement claim.',
        interpretation='Actual native dynamics candidates; not accepted stable-contact/sliding reference yet.',
        support_interval_s=[.02,6.96],core_interval_s=[1.44,4.],
        sliding_screen=dict(upward_normal_force_strict_N=5.,actual_contact_lateral_RMS_strict_m_s=.05,
            wheel_aggregate_utilization_proxy_min=.9,friction_anchor_power_strict_W=-1e-4,minimum_COM_speed_m_s=.25,
            continuity='same wheel, successive400Hz samples; no gap joining or wheel switching'),
        pointwise_normal_to_friction_anchor_mapping_verified=False,
        friction_cone_saturation_certified=False,
        poststep_power_is_exact_discrete_contact_work=False,
        power_limitation='SDK dt-averaged friction force is paired with post-step material velocity at its reported anchor. The rectangular time integral is a proxy, not exact impulse work or an energy/passivity certificate; positive values need force-time/anchor-history diagnosis.',
        limitation='Wheel aggregate saturation proxy and slip can originate from different contact regions. Native anchor power pairs force and velocity at the SAME anchor; per-anchor normal load is unavailable. This corroborates at wheel level only.',
        learned_policy_reused=True,policy_training=False,independent_pure_model_baseline=False,
        final_mean_replaced_by_best_sample=False,source_sha256=sha(__file__),
        search_report_sha256=sha(a.search/'report.json'),validation_report_sha256=sha(a.validation/'report.json'),
        reference_sha256=sha(a.reference),search_vs_single_validation_max_root_difference_m=float(np.max(abs(load(a.search/'final_rollout.npz')['root_pos']-final['root_pos']))))
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('search','validation','baseline-validation','reference','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())

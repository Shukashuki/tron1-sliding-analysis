"""Speed, tracking and contact evidence for physically retimed PhysX episodes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from physx_reference_math import trajectory_metrics
from analyze_physx_policy_mppi_reference import sliding,verify_native_buffers
from analyze_ppo_contact_timestep import band_spectrum
from prepare_physx_speed_references import load


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def case_metrics(trace,reference,scale):
    interval=(1.44/scale,4./scale)
    m=trajectory_metrics(trace,reference,core_interval_s=interval)
    t=np.arange(len(trace['root_pos']))*.0025
    core=(t>=interval[0]-1e-9)&(t<=interval[1]+1e-9)
    vel=trace['root_com_vel'];speed=np.linalg.norm(vel[:,:2],axis=-1)
    q=trace['root_quat'];tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
    rp=reference['body_pos_w'][:,0,:2];rt=np.arange(len(rp))*.02
    target=np.column_stack([np.interp(t,rt,rp[:,i]) for i in range(2)])
    rv=np.column_stack([np.interp(t,rt,reference['body_lin_vel_w'][:,0,i]) for i in range(2)])
    rq=reference['body_quat_w'][:,0]
    refyaw=np.interp(t,rt,np.unwrap(np.arctan2(2*(rq[:,0]*rq[:,3]+rq[:,1]*rq[:,2]),1-2*(rq[:,2]**2+rq[:,3]**2))))
    actualyaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    beta_error=np.arctan2(vel[:,1],vel[:,0])-actualyaw-np.arctan2(rv[:,1],rv[:,0])+refyaw
    beta_error=np.arctan2(np.sin(beta_error),np.cos(beta_error))
    valid=core&(speed>=.25)&(np.linalg.norm(rv,axis=-1)>=.3)
    loaded=valid&(np.max(trace['normal_fz'],axis=-1)>5.)
    def beta_row(mask):
        return dict(samples=int(mask.sum()),core_coverage_fraction=float(np.mean(mask[core])),
            beta_rmse_deg=None if not np.any(mask) else float(np.degrees(np.sqrt(np.mean(beta_error[mask]**2)))))
    error=np.linalg.norm(trace['root_pos'][:,:2]-target,axis=-1)
    mask=(error>.30)&(t<=rt[-1]+1e-9)
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    departures=[dict(start_s=float(t[first]),duration_s=float((last-first)*.0025))
                for first,last in zip(edges[::2],edges[1::2]) if (last-first)*.0025>=.1-1e-9]
    m.update(core_interval_s=list(interval),actual_core_mean_speed_m_s=float(speed[core].mean()),
        actual_core_peak_speed_m_s=float(speed[core].max()),initial_speed_m_s=float(speed[0]),
        initial_joint_velocity_rad_s=trace['joint_vel'][0].tolist(),
        whole_body_inverted_observed=bool(np.any(tilt>90.)),
        first_inversion_s=None if not np.any(tilt>90.) else float(t[np.flatnonzero(tilt>90.)[0]]),
        persistent_xy_departures_gt0p30m_ge0p10s=departures,
        core_angular_xy_10_to_100Hz=band_spectrum(trace['root_omega'][core,:2],.0025),
        beta_core_valid_moving=beta_row(valid),beta_core_valid_moving_anywheel_loaded=beta_row(loaded),
        sliding_aggregate_evidence=sliding(trace,interval),
        listed_screens=dict(core_xy_rmse=m['core']['xy_rmse_m']<=.15,whole_xy_rmse=m['whole']['xy_rmse_m']<=.15,
            xy_peak=m['whole']['xy_peak_m']<=.30,speed_retention=.95<=m['core']['speed_retention']<=1.10,
            beta=m['core']['beta_rmse_deg']<=10.,continuous_anywheel_support=m['longest_support_gap_s']==0.),
        task_success_claimed=False,nonwheel_ground_contact_measured=False)
    m['listed_screens_all_pass']=all(m['listed_screens'].values())
    return m


def run(a):
    assert not a.output.exists()
    manifest=json.loads((a.references/'manifest.json').read_text())
    driver=json.loads((a.probes/'status.json').read_text());assert driver['status']=='completed_all_serial_single_world_probes'
    source=load(a.original_reference);v=source['body_lin_vel_w'][:,0,:2].astype(float)
    st=np.arange(len(v))*.02;core=(st>=1.44-1e-9)&(st<=4.+1e-9)
    s=np.linalg.norm(v,axis=-1);c=np.unwrap(np.arctan2(v[:,1],v[:,0]));alat=s*abs(np.gradient(c,.02))
    cases=[]
    for spec in manifest['cases']:
        folder=a.probes/spec['id'];report=json.loads((folder/'report.json').read_text())
        assert report['worlds']==1 and report['single_world_validation'] and report['steps']==spec['frames']
        assert all(sha(folder/k)==value for k,value in report['artifacts'].items())
        assert report['motion_sha256']==spec['motion_sha256'] and report['original_motion_sha256']==spec['original_car_motion_sha256']
        assert sha(a.references/spec['id']/'motion.npz')==spec['motion_sha256']
        reference=load(a.references/spec['id']/'car_reference.npz');trace=load(folder/'final_rollout.npz');scale=spec['effective_scale']
        m=case_metrics(trace,reference,scale)
        verification=verify_native_buffers(trace,folder/'native_contact_buffers.npz')
        case=dict(id=spec['id'],requested_scale=spec['requested_scale'],effective_scale=scale,
            reference_demand=spec['reference_demand'],metrics=m,native_contact_verification=verification,
            ideal_source_peak_lateral_demand_over_mu_g=float(np.max(alat[core])*scale**2/(.7*9.81)),
            ideal_source_core_mean_speed_m_s=float(np.mean(s[core])*scale),
            reference_duration_s=float(reference['time_s'][-1]),physical_duration_s=float((len(trace['root_pos'])-1)*.0025),
            source_recording_sha256=sha(folder/'final_rollout.npz'))
        reconstructed=np.column_stack([np.interp(st,reference['source_time_s'],reference['body_pos_w'][:,0,j]) for j in range(2)])
        case['maximum_source_XY_resampling_error_m']=float(np.max(np.linalg.norm(reconstructed-source['body_pos_w'][:,0,:2],axis=-1)))
        cases.append(case)
    result=dict(kind='single_world_PhysX_speed_ladder_diagnostic',cases=cases,mu=.7,
        actor_unchanged=True,policy_training=False,independent_model_controller=False,MPPI_search_executed=False,
        entry_state='same pose; initial COM and joint velocities scaled; initial kinetic energy grows as scale squared; not acceleration from rest',
        reference_demand_note='Retiming a discrete50Hz reference can undersample sharp curvature peaks. Report both derivative of resampled50Hz velocity and ideal source-resolution peak times effective_scale^2. Neither is a full-body friction feasibility proof.',
        friction_saturation_certified=False,contact_evidence_scope='wheel aggregate; no pointwise normal-to-friction-anchor mapping',
        slide_power_note='dt-averaged native force with post-step anchor velocity; integral is a proxy, not exact contact work',
        source_sha256=sha(__file__),source_manifest_sha256=sha(a.references/'manifest.json'),driver_status_sha256=sha(a.probes/'status.json'))
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(id=c['id'],mean_speed=c['metrics']['actual_core_mean_speed_m_s'],speed_ratio=c['metrics']['core']['speed_retention'],xy=c['metrics']['core']['xy_rmse_m'],support=c['metrics']['support_fraction'],tilt=c['metrics']['maximum_root_tilt_deg'],gap_ms=1000*c['metrics']['longest_support_gap_s']) for c in cases],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('probes','references','original-reference','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())

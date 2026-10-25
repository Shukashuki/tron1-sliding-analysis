"""Replay generated actions unchanged and audit native contact-point motion.

The original continuous plant must reproduce before any contact diagnostics
are accepted. No root correction, controller re-solve or state reset occurs.
Loads, slip and friction utilization below come from actual replay contacts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import generate_mppi_contact_reference as g
import analyze_ppo_backend_400hz_matrix as shared
from analyze_cem_ground_contact_v2 import support_summary


def contacts(model,data):
    wheels=[model.body('wheel_'+s+'_Link').id for s in ('L','R')]
    floor=model.geom('floor').id
    out={k:np.zeros(2) for k in ('loaded_point_count','normal_weight_n','signed_lateral_m_s',
        'lateral_rms_m_s','tangent_rms_m_s','euclidean_friction_utilization_max','pyramidal_friction_utilization_max')}
    for c,contact in enumerate(data.contact):
        geoms=list(contact.geom)
        if floor not in geoms:continue
        other=geoms[1] if geoms[0]==floor else geoms[0];body=int(model.geom_bodyid[other])
        if body not in wheels:continue
        i=wheels.index(body);wrench=np.empty(6);mujoco.mj_contactForce(model,data,c,wrench)
        if wrench[0]<=5:continue
        twist=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,twist,0)
        velocity=twist[3:]+np.cross(twist[:3],contact.pos-data.xipos[body])
        axis=data.xmat[body].reshape(3,3)[:,1].copy();axis[2]=0;axis/=np.linalg.norm(axis)
        lateral=float(velocity@axis);fz=float(wrench[0])
        out['loaded_point_count'][i]+=1;out['normal_weight_n'][i]+=fz
        out['signed_lateral_m_s'][i]+=fz*lateral
        out['lateral_rms_m_s'][i]+=fz*lateral*lateral
        out['tangent_rms_m_s'][i]+=fz*float(velocity[:2]@velocity[:2])
        mu=np.asarray(contact.friction[:2])
        utilization=float(np.linalg.norm(wrench[1:3]/np.maximum(mu*fz,1e-12)))
        out['euclidean_friction_utilization_max'][i]=max(out['euclidean_friction_utilization_max'][i],utilization)
        # condim3 pyramid has edges [1,+/-mu1,0], [1,0,+/-mu2].
        # Thus its physical tangent section is an INSCRIBED L1 diamond,
        # not a square circumscribed about the circular cone.
        if model.opt.cone!=mujoco.mjtCone.mjCONE_PYRAMIDAL or contact.dim!=3:
            raise ValueError('This pyramid utilization formula requires the declared condim3 pyramidal model')
        pyramid=float(np.sum(np.abs(wrench[1:3])/np.maximum(mu*fz,1e-12)))
        out['pyramidal_friction_utilization_max'][i]=max(out['pyramidal_friction_utilization_max'][i],pyramid)
    denominator=np.maximum(out['normal_weight_n'],1e-12)
    out['signed_lateral_m_s']/=denominator
    for key in ('lateral_rms_m_s','tangent_rms_m_s'):out[key]=np.sqrt(out[key]/denominator)
    return out


def tracking_metrics(raw,source):
    ref_base=list(source['body_names']).index('base_Link')
    reference=dict(time_s=source['time_s'],position=source['body_pos_w'][:,ref_base],
        velocity=source['body_lin_vel_w'][:,ref_base],quaternion=source['body_quat_w'][:,ref_base])
    trace=dict(time_s=raw['time_s'],position=raw['root_pos'],quaternion=raw['root_quaternion_wxyz'],
        com_velocity=raw['base_com_velocity_w'],angular_velocity=raw['base_angular_velocity_w'],
        contact_count=raw['ground_contact_count'],contact_force=raw['ground_normal_force_w'])
    v=trace['com_velocity'];speed=np.linalg.norm(v[:,:2],axis=-1);q=trace['quaternion']
    forward_xy_norm=np.sqrt((1-2*(q[:,2]**2+q[:,3]**2))**2+(2*(q[:,1]*q[:,2]+q[:,0]*q[:,3]))**2)
    signals=dict(root_beta_valid=(speed>=.25)&(forward_xy_norm>1e-6),
        root_directed_beta_rad=g.angle(np.arctan2(v[:,1],v[:,0])-g.quaternion_heading(q)))
    windows={name:shared.window_metrics(trace,reference,signals,interval)
        for name,interval in (('core',(1.44,4.)),('whole',(0.,6.96)))}
    support=support_summary(raw['time_s'],raw['ground_normal_force_w'][...,2],(.02,6.96))
    camber,wheel_heading=g.wheel_angles(raw['body_quat_w'][:,-2:])
    wheel_v=raw['body_lin_vel_w'][:,-2:,:2]
    wheel_speed=np.linalg.norm(wheel_v,axis=-1)
    wheel_beta=g.angle(np.arctan2(wheel_v[...,1],wheel_v[...,0])-wheel_heading)
    selected=(raw['time_s']>=1.44-1e-9)&(raw['time_s']<=4+1e-9)
    wheels=[]
    for i,name in enumerate(('L','R')):
        loaded=selected&(raw['ground_normal_force_w'][:,i,2]>5)
        moving=loaded&(wheel_speed[:,i]>=.25)
        def quantile(value,mask):
            return np.quantile(value[mask],[.05,.5,.95]).tolist() if mask.any() else None
        wheels.append(dict(side=name,core_loaded_samples=int(loaded.sum()),
            core_loaded_camber_deg_p05_p50_p95=quantile(np.rad2deg(camber[:,i]),loaded),
            core_loaded_heading_offset_from_body_deg_p05_p50_p95=quantile(np.rad2deg(g.angle(wheel_heading[:,i]-g.quaternion_heading(q))),loaded),
            core_loaded_moving_wheel_COM_beta_deg_p05_p50_p95=quantile(np.rad2deg(wheel_beta[:,i]),moving),
            wheel_COM_beta_is_contact_point_slip=False))
    core,whole=windows['core']['tracking'],windows['whole']['tracking']
    checks=dict(core_XY_RMSE=core['xy_rmse_m']<=.15,core_XY_peak=core['xy_peak_m']<=.3,
        whole_XY_RMSE=whole['xy_rmse_m']<=.15,whole_XY_peak=whole['xy_peak_m']<=.3,
        core_speed=.95<=core['speed_retention']<=1.1,
        core_beta=core['beta_rmse_deg'] is not None and core['beta_rmse_deg']<=10,
        every_sample_support=support['every_physical_sample_supported'],zero_support_gap=support['longest_no_support_observed_s']==0)
    baseline_path=g.ROOT/'results/ppo_backend_400hz_nominal_20261022_v1/summary.json'
    if g.sha(baseline_path)!='6dcc65fb2b863d856190e452bcea744dfeb2bdd253d1d31389cb51146fcac5ac':raise ValueError('Original HF baseline summary changed')
    baseline=next(r for r in json.loads(baseline_path.read_text())['rows'] if r['id']=='physx_to_mujoco')
    reference_hf=baseline['windows']['core']['world_angular_xy_common10_100']['common_band_rms']
    ratio=windows['core']['world_angular_xy_common10_100']['common_band_rms']/reference_hf
    checks['core_HF_ratio']=ratio<=.7
    return dict(windows=windows,support=support,wheels=wheels,checks=checks,
        nine_numeric_checks_pass=all(checks.values()),
        core_HF_ratio_to_original_same_engine_PPO=ratio,
        HF_baseline_summary_sha256=g.sha(baseline_path),HF_relative_to_original_PPO_evaluated=True,
        absolute_height_gate=False,nominal_task_success_claimed=False)


def run(a):
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    report=json.loads((a.run/'report.json').read_text())
    if report['status']!='completed' or report['completed_control_intervals']!=349:raise ValueError('Require full completed trajectory')
    if g.sha(a.run/'rollout.npz')!=report['rollout_sha256']:raise ValueError('Rollout hash differs')
    smooth=report['kind']=='native_mppi_smooth_contact_reference_v1'
    generator=g.ROOT/'scripts/generate_mppi_smooth_contact_reference.py' if smooth else Path(g.__file__)
    if g.sha(generator)!=report['source_sha256']:raise ValueError('Replay generator source differs')
    if smooth and g.sha(g.__file__)!=report['base_generator_source_sha256']:raise ValueError('Base generator source differs')
    paths=g.parser().parse_args(['--output-dir','unused'])
    original=json.loads(paths.material_report.read_text())
    model=g.load_model(paths.model,original)
    with np.load(paths.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    with np.load(a.run/'rollout.npz',allow_pickle=False) as f:raw={k:f[k].copy() for k in f.files}
    d=mujoco.MjData(model);d.qpos[:]=original['initial_state']['qpos'];d.qvel[:]=original['initial_state']['qvel'];mujoco.mj_forward(model,d)
    q0=source['joint_pos'][0,[list(source['joint_names']).index(n) for n in g.JOINT_NAMES]]
    aids=[model.actuator(n).id for n in g.JOINT_NAMES];rows=[contacts(model,d)];errors=[];frame=0
    for action in raw['actions']:
        for sub in range(8):
            applied=g.torque(model,d,q0,action);d.ctrl[aids]=applied;mujoco.mj_step(model,d);mujoco.mj_forward(model,d);frame+=1
            errors.append([np.max(abs(d.qpos-raw['qpos'][frame])),np.max(abs(d.qvel-raw['qvel'][frame])),
                np.max(abs(g.ground_contact_readout(model,d)['ground_normal_force_w']-raw['ground_normal_force_w'][frame]))])
            rows.append(contacts(model,d))
    errors=np.asarray(errors)
    if np.max(errors[:,0])>1e-10 or np.max(errors[:,1])>1e-9 or np.max(errors[:,2])>1e-7:
        raise ValueError('Continuous action replay failed to reproduce; do not accept contact audit')
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};arrays['time_s']=raw['time_s']
    result=tracking_metrics(raw,source)
    core=(raw['time_s']>=1.44-1e-9)&(raw['time_s']<=4+1e-9)
    for i,row in enumerate(result['wheels']):
        mask=core&(arrays['loaded_point_count'][:,i]>0)&(raw['ground_normal_force_w'][:,i,2]>5)
        row['actual_loaded_contact_point_samples']=int(mask.sum())
        for key in ('signed_lateral_m_s','lateral_rms_m_s','tangent_rms_m_s','euclidean_friction_utilization_max','pyramidal_friction_utilization_max'):
            row['core_actual_contact_'+key+'_p05_p50_p95']=np.quantile(arrays[key][mask,i],[.05,.5,.95]).tolist() if mask.any() else None
    result.update(kind='MPPI_generated_reference_continuous_replay_audit',run_report_sha256=g.sha(a.run/'report.json'),
        rollout_sha256=g.sha(a.run/'rollout.npz'),source_sha256=g.sha(__file__),
        shared_metrics_sha256=g.sha(shared.__file__),support_metrics_sha256=g.sha(g.ROOT/'scripts/analyze_cem_ground_contact_v2.py'),
        physics_samples=len(raw['time_s']),replay_max_abs_error_qpos_qvel_Fz=errors.max(0).tolist(),
        actual_contact_point_loading_threshold_n=5.,unloaded_slip_zero_placeholder_is_rolling=False,
        friction_utilization_caveat='Euclidean L2 circle and native condim3 pyramidal L1 diamond are different boundaries; force utilization alone is not a static/sliding classifier',
        friction_cone_source='https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones',
        initializations=1,controller_resolves_during_replay=0,native_physics_engine='MuJoCo3.8.1 CPU')
    a.output_dir.mkdir(parents=True);np.savez_compressed(a.output_dir/'contact_audit.npz',**arrays)
    result['contact_audit_sha256']=g.sha(a.output_dir/'contact_audit.npz')
    g.write_json(a.output_dir/'analysis.json',result)
    print(json.dumps(dict(checks=result['checks'],core=result['windows']['core']['tracking'],support=result['support']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);run(p.parse_args())

"""Physical time compression at unchanged50Hz; preserve path and nonzero beta."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def interpolate_quaternions(q,query):
    q=np.asarray(q,dtype=float)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    lo=np.minimum(np.floor(query).astype(int),len(q)-2);u=(query-lo)[:,None,None]
    a=q[lo];b=q[lo+1]
    dot=np.sum(a*b,axis=-1,keepdims=True);b=np.where(dot<0,-b,b)
    dot=np.abs(dot).clip(0,1);angle=np.arccos(dot);denom=np.sin(angle)
    wa=np.divide(np.sin((1-u)*angle),denom,out=np.broadcast_to(1-u,angle.shape).copy(),where=denom>1e-8)
    wb=np.divide(np.sin(u*angle),denom,out=np.broadcast_to(u,angle.shape).copy(),where=denom>1e-8)
    out=wa*a+wb*b
    return out/np.linalg.norm(out,axis=-1,keepdims=True)


def retime(source,requested_scale):
    if not np.isfinite(requested_scale) or requested_scale<1:raise ValueError('Scale must be finite and >=1')
    old_frames=len(source['joint_pos'])
    if old_frames<2 or float(np.asarray(source['fps']).reshape(-1)[0])!=50.:raise ValueError('Need >=2 frames at50Hz')
    intervals=max(1,round((old_frames-1)/requested_scale));effective=(old_frames-1)/intervals
    query=np.linspace(0,old_frames-1,intervals+1)
    def interp(values):
        values=np.asarray(values);flat=values.reshape(old_frames,-1)
        return np.stack([np.interp(query,np.arange(old_frames),flat[:,j]) for j in range(flat.shape[1])],-1).reshape(len(query),*values.shape[1:])
    out={k:interp(source[k]) for k in ('joint_pos','body_pos_w')}
    out.update({k:interp(source[k])*effective for k in ('joint_vel','body_lin_vel_w','body_ang_vel_w')})
    out['body_quat_w']=interpolate_quaternions(source['body_quat_w'],query)
    out.update(fps=np.array([50.]),time_s=np.arange(len(query))*.02,source_time_s=query*.02,
        source_phase=query/(old_frames-1),speed_scale_requested=np.array(requested_scale),speed_scale_effective=np.array(effective))
    for k in ('joint_names','body_names','joint_tracking_mask','body_orientation_tracking_mask','body_angular_velocity_tracking_mask'):
        if k in source:out[k]=np.asarray(source[k]).copy()
    return out,effective


def demand(reference,scale,mu=.7):
    v=reference['body_lin_vel_w'][:,0,:2];t=reference['time_s']
    mask=(t>=1.44/scale-1e-9)&(t<=4./scale+1e-9)
    speed=np.linalg.norm(v,axis=-1);course=np.unwrap(np.arctan2(v[:,1],v[:,0]));rate=np.gradient(course,.02)
    lateral=speed*abs(rate)
    return dict(core_interval_s=[1.44/scale,4./scale],core_mean_speed_m_s=float(speed[mask].mean()),
        core_peak_lateral_acceleration_m_s2=float(lateral[mask].max()),
        peak_lateral_demand_over_mu_g=float(lateral[mask].max()/(mu*9.81)),
        force_capacity_assumption='planar total normal load=mg, no simultaneous longitudinal demand; feasibility NOT certified')


def run(a):
    assert not a.output_dir.exists()
    cfg=json.loads(a.base_config.read_text())
    assert sha(a.motion)==cfg['motion_sha256'] and sha(a.original_motion)==cfg['original_car_motion_sha256']
    a.output_dir.mkdir(parents=True)
    source=load(a.motion);car=load(a.original_motion);cases=[]
    for scale in a.scales:
        name='speed'+str(scale).replace('.','p');folder=a.output_dir/name;folder.mkdir()
        motion,effective=retime(source,scale);original,other=retime(car,scale);assert effective==other
        np.savez_compressed(folder/'motion.npz',**motion);np.savez_compressed(folder/'car_reference.npz',**original)
        spec=dict(requested_scale=scale,effective_scale=effective,frames=len(motion['joint_pos']),
            policy_dt_s=.02,source_motion_sha256=sha(a.motion),source_car_reference_sha256=sha(a.original_motion),
            source_config_sha256=sha(a.base_config),source_sha256=sha(__file__),
            original_path_shape_preserved=True,physical_time_retimed=True,video_playback_speed_changed=False,
            initial_velocity_scaling='all root COM linear/angular and joint velocities scaled; same initial pose; preaccelerated entry',
            initial_kinetic_energy_multiplier=effective**2,reference_demand=demand(original,effective))
        derived={**cfg,'motion_sha256':sha(folder/'motion.npz'),'original_car_motion_sha256':sha(folder/'car_reference.npz'),'speed_reference':spec}
        (folder/'config.json').write_text(json.dumps(derived,indent=2)+'\n')
        cases.append(dict(id=name,**spec,motion_sha256=sha(folder/'motion.npz'),original_car_motion_sha256=sha(folder/'car_reference.npz'),config_sha256=sha(folder/'config.json')))
    (a.output_dir/'manifest.json').write_text(json.dumps(dict(kind='physx_speed_reference_ladder',mu=.7,cases=cases),indent=2)+'\n')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('motion','original-motion','base-config','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--scales',type=float,nargs='+',default=[1.5,2.,2.5,3.])
    run(p.parse_args())

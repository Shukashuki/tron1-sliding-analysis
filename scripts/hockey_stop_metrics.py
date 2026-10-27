"""Stopping kinematics for calibrated world trajectories, not image landmarks."""
from __future__ import annotations
import numpy as np


def stopping_metrics(time_s,base_position_w,base_com_velocity_w,brake_onset_s,*,
                     metric_world_trajectory_verified=False,stop_speed_m_s=.10,hold_s=.50):
    if not metric_world_trajectory_verified:
        return dict(available=False,reason='Calibrated global3D trajectory required; hip-relative3D/image pixels are insufficient',
                    task_success_claimed=False)
    t=np.asarray(time_s,dtype=float);p=np.asarray(base_position_w,dtype=float);v=np.asarray(base_com_velocity_w,dtype=float)
    if (t.ndim!=1 or len(t)<3 or p.shape!=(len(t),3) or v.shape!=p.shape
            or not all(np.isfinite(x).all() for x in (t,p,v)) or np.any(np.diff(t)<=0)
            or not t[0]<=brake_onset_s<t[-1] or stop_speed_m_s<=0 or hold_s<=0):
        raise ValueError('Invalid calibrated trajectory or phase definition')
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,atol=1e-7,rtol=1e-4):
        raise ValueError('Missing/irregular samples must not be joined into a stop hold')
    speed=np.linalg.norm(v[:,:2],axis=-1);start=int(np.searchsorted(t,brake_onset_s))
    entry=v[max(0,start-1),:2];entry_speed=float(np.linalg.norm(entry))
    low=(t>=brake_onset_s-1e-9)&(speed<=stop_speed_m_s)
    edges=np.flatnonzero(np.diff(np.r_[False,low,False]));selected=None
    for first,last in zip(edges[::2],edges[1::2]):
        if t[last-1]-t[first]>=hold_s-1e-9:selected=(first,last);break
    result=dict(available=True,kinematic_stop_detected=selected is not None,entry_speed_m_s=entry_speed,
        brake_onset_s=float(brake_onset_s),stop_speed_m_s=stop_speed_m_s,required_hold_s=hold_s,
        entry_has_nontrivial_motion=entry_speed>=.8,
        beta_valid_mask=speed>=.25,trajectory_marker='base link origin; velocity=base COM, matching existing task ABI',
        task_success_claimed=False,contact_and_tracking_still_require_separate_evaluation=True)
    if selected is None:return result
    first,last=selected;p0=np.array([np.interp(brake_onset_s,t,p[:,j]) for j in range(2)])
    delta=p[first,:2]-p0
    result.update(stop_time_s=float(t[first]),braking_duration_s=float(t[first]-brake_onset_s),
        observed_stop_hold_s=float(t[last-1]-t[first]),stop_position_xy_m=p[first,:2].tolist(),
        stopping_displacement_xy_m=float(np.linalg.norm(delta)),
        stopping_path_length_m=float(np.linalg.norm(np.diff(np.vstack((p0,p[start:first+1,:2])),axis=0),axis=-1).sum()))
    if entry_speed>1e-9:
        direction=entry/entry_speed;normal=np.array([-direction[1],direction[0]])
        result.update(forward_stopping_distance_m=float(delta@direction),
            maximum_lateral_excursion_m=float(np.max(np.abs((p[start:first+1,:2]-p0)@normal))))
    return result

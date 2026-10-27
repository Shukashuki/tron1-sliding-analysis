#!/usr/bin/env python3
"""Reuse a frozen, previously deployed PhysX runner for a single hockey probe.

Only the input motion and post-run hockey diagnostics change. The PPO is the
existing car stabilizer; zero residual makes this a baseline, not MPPI success.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def longest(mask, dt):
    edges=np.flatnonzero(np.diff(np.r_[False,np.asarray(mask,dtype=bool),False]))
    return float(np.max(edges[1::2]-edges[::2],initial=0)*dt)


def hockey_metrics(trace, reference):
    dt=.0025;t=np.arange(len(trace['root_pos']))*dt
    rt=np.arange(len(reference['body_pos_w']))*.02
    end=min(t[-1],rt[-1]);valid=(t>.02)&(t<=end+1e-9)
    q=trace['root_quat'];v=trace['root_com_vel'];speed=np.linalg.norm(v[:,:2],axis=-1)
    yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
    wrap=lambda a:np.arctan2(np.sin(a),np.cos(a))
    beta=np.degrees(wrap(np.arctan2(v[:,1],v[:,0])-yaw));beta_valid=valid&(speed>=.25)
    target=np.stack([np.interp(t,rt,reference['body_pos_w'][:,0,k]) for k in (0,1)],axis=-1)
    xy=np.linalg.norm(trace['root_pos'][:,:2]-target,axis=-1)
    support=trace['normal_fz']>5.;anysupport=support.any(-1)
    slip=support&(trace['lateral_slip_rms']>.05)&(trace['utilization_proxy']>=.9)&(trace['friction_anchor_power']<-.05)
    stop=valid&(speed<=.1)&(tilt<=30)&anysupport
    stop_hold=max(0.,longest(stop,dt)-dt)
    stable_end=(t>=end-.5)&valid
    fell=np.flatnonzero(valid&(tilt>=45))
    peak_beta=float(abs(beta[beta_valid]).max()) if beta_valid.any() else None
    event_duration=[longest(slip[:,k]&valid,dt) for k in (0,1)]
    success=bool(np.sqrt(np.mean(xy[valid]**2))<=.15 and xy[valid].max()<=.3
        and np.all(stop[stable_end]) and np.all(anysupport[valid]) and max(event_duration)>=.05 and not len(fell))
    return dict(task='hockey_stop_probe',source_motion_is_designed_not_measured=True,
        xy_rmse_m=float(np.sqrt(np.mean(xy[valid]**2))),xy_peak_m=float(xy[valid].max()),
        initial_actual_speed_m_s=float(speed[0]),final_actual_speed_m_s=float(speed[valid][-1]),
        maximum_tilt_deg=float(tilt[valid].max()),first_45deg_tilt_s=float(t[fell[0]]) if len(fell) else None,
        anywheel_support_fraction=float(np.mean(anysupport[valid])),longest_support_gap_s=longest(~anysupport[valid],dt),
        upright_supported_stop_hold_s=stop_hold,body_beta_absolute_peak_deg=peak_beta,beta_min_speed_m_s=.25,
        per_wheel_loaded_slip_saturation_dissipation_longest_s=event_duration,
        slip_event_definition='same wheel: Fz>5N, normal-point lateral slip RMS>.05m/s, aggregate utilization>=.9, anchor power<-.05W',
        friction_utilization_is_aggregate_proxy=True,pointwise_cone_saturation_certified=False,
        task_success_claimed=success)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('engine-scripts','prior-launch','bundle','output-dir'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--steps',type=int,default=0)
    args=p.parse_args()
    engine=args.engine_scripts/'generate_physx_policy_mppi_reference.py'
    if sha(engine)!='3aacb6b4f3b62d32ce6fa2326f2cb6bfe372b693b53c9830a27a85f26f3f7763':raise ValueError('PhysX engine changed')
    old=json.loads(args.prior_launch.read_text())['argv']
    get=lambda flag:Path(old[old.index('--'+flag)+1])
    sys.path.insert(0,str(args.engine_scripts))
    spec=importlib.util.spec_from_file_location('frozen_physx_engine',engine)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.trajectory_metrics=hockey_metrics
    a=argparse.Namespace(config=args.bundle/'config.json',actor=get('actor'),
        motion=args.bundle/'motion.npz',asset=get('asset'),original_motion=args.bundle/'motion.npz',
        seed_report=get('seed-report'),seed_trace=get('seed-trace'),tracking_root=get('tracking-root'),
        beyondmimic_root=get('beyondmimic-root'),output_dir=args.output_dir,device=args.device,
        steps=args.steps,samples=None,iterations=0,validate_mean=args.bundle/'zero_mean.npz')
    module.run(a)
    report=json.loads((args.output_dir/'report.json').read_text())
    report.update(hockey_wrapper_sha256=sha(__file__),hockey_command_report_sha256=sha(args.bundle/'command_report.json'),
        car_core_interval_not_used=True,old_PPO_51D_command_has_no_explicit_braking_velocity=True,
        legacy_turn_cost_not_used_for_hockey_success_or_ranking=True,
        baseline_zero_residual=True,MPPI_search_run=False)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

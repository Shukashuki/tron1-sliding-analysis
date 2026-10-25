"""Separate instantaneous loaded slip, persistent wheel slip and unsupported time."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import generate_mppi_contact_reference as g


def runs(time,mask):
    time=np.asarray(time);mask=np.asarray(mask,dtype=bool)
    if time.ndim!=1 or mask.shape!=time.shape or not np.all(np.diff(time)>0):raise ValueError('Invalid event clock/mask')
    starts=np.flatnonzero(mask&np.r_[True,~mask[:-1]]);ends=np.flatnonzero(mask&np.r_[~mask[1:],True])+1
    return [dict(start_s=float(time[s]),last_valid_sample_s=float(time[e-1]),
        duration_s=float(time[e-1]-time[s]),samples=int(e-s),
        left_window_censored=bool(s==0),right_window_censored=bool(e==len(time))) for s,e in zip(starts,ends)]


def evaluate(raw,audit,*,threshold=.05,duration=.05):
    t=raw['time_s'];core=(t>=1.44-1e-9)&(t<=4+1e-9);moving=np.linalg.norm(raw['base_com_velocity_w'][:,:2],axis=1)>=.25
    force_support=raw['ground_normal_force_w'][...,2]>5
    observed=audit['loaded_point_count']>0
    rows=[]
    for side,name in enumerate(('L','R')):
        valid=core&moving&force_support[:,side]&observed[:,side]
        row=dict(side=name,core_force_supported_samples=int(np.sum(core&force_support[:,side])),
            core_supported_but_slip_unobserved_samples=int(np.sum(core&force_support[:,side]&~observed[:,side])),
            core_loaded_moving_valid_samples=int(valid.sum()))
        for kind,key in (('lateral','lateral_rms_m_s'),('all_tangent','tangent_rms_m_s')):
            active=valid&(audit[key][:,side]>threshold)
            events=runs(t[core],active[core]);persistent=[r for r in events if r['duration_s']>=duration-1e-9]
            row[kind]=dict(sliding_samples=int(active.sum()),loaded_sample_sliding_fraction=float(active.sum()/valid.sum()) if valid.any() else None,
                loaded_speed_m_s_p05_p50_p95=np.quantile(audit[key][valid,side],[.05,.5,.95]).tolist() if valid.any() else None,
                longest_contiguous_sliding_s=max((r['duration_s'] for r in events),default=0.),
                persistent_events=persistent,persistent_event_count=len(persistent))
        support_runs=runs(t[core],force_support[core,side])
        row['longest_contiguous_core_support_s']=max((r['duration_s'] for r in support_runs),default=0.)
        rows.append(row)
    return dict(kind='loaded_contact_sliding_and_continuity_screen',interval_s=[1.44,4.],
        wheel_support_Fz_threshold_n_strict=5.,contact_point_Fn_threshold_n_strict=5.,
        core_base_speed_threshold_m_s_min=.25,slip_rms_threshold_m_s_strict=threshold,
        persistent_duration_s_min=duration,thresholds_are_diagnostic=True,
        require_continuous_support_and_slip_on_same_wheel=True,
        same_geometrical_contact_point_identity_required=False,contact_gap_bridging=False,
        event_duration='last valid sample time minus first valid sample time; no unsupported interval is filled',
        unloaded_or_unobserved_zero_is_rolling=False,rows=rows,
        sustained_lateral_sliding_observed=any(r['lateral']['persistent_event_count'] for r in rows))


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.run/'rollout.npz') as f:raw={k:f[k].copy() for k in f.files}
    with np.load(a.audit/'contact_audit.npz') as f:data={k:f[k].copy() for k in f.files}
    proof=json.loads((a.audit/'analysis.json').read_text())
    if g.sha(a.run/'rollout.npz')!=proof['rollout_sha256'] or g.sha(a.audit/'contact_audit.npz')!=proof['contact_audit_sha256']:
        raise ValueError('Contact replay proof differs')
    result=evaluate(raw,data)
    result.update(source_sha256=g.sha(__file__),raw_sha256=g.sha(a.run/'rollout.npz'),contact_audit_sha256=g.sha(a.audit/'contact_audit.npz'))
    g.write_json(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('run','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

"""Bounded constant-shape six-axis contact-wrench posture fitting.

Adds independent fore/aft wheel placement to the admitted geometry model.
All results are snapshots,not a time-varying reference or native validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import generate_camber_reference as base
from spatial_wheel_posture import Geometry,axle_geometry
from audit_spatial_camber_wrenches import evaluate


class Placement(Geometry):
    def __init__(self,model,source):
        super().__init__(model,source)
        self.original_longitudinal=np.array([link['x'] for link in self.links])

    def set_longitudinal_shift(self,shift):
        value=np.asarray(shift,dtype=float)
        if value.shape!=(2,) or not np.isfinite(value).all():raise ValueError('Require finite L/R wheel placement')
        for i,link in enumerate(self.links):link['x']=self.original_longitudinal[i]+value[i]


def measure(g,frame,parameters,mu,weight,min_slide):
    parameters=np.asarray(parameters,dtype=float)
    if parameters.shape!=(7,) or not np.isfinite(parameters).all():raise ValueError('Require seven posture variables')
    g.set_longitudinal_shift(parameters[5:])
    try:
        value=evaluate(g,frame,parameters[:5],mu,weight)
        if g.nonwheel_clearance() < -1e-9 or g.opposite_leg_clearance() < -1e-9:raise ValueError('Geometric intersection')
        speeds=np.linalg.norm(np.asarray(value['wheel_point_velocity_w_m_s'])[:,:2],axis=-1)
        if np.any(speeds<min_slide):raise ValueError('Outside prospective fully sliding speed margin')
        _,camber,heading=axle_geometry(*np.deg2rad(parameters[:2]),np.deg2rad(parameters[2:4]))
        value.update(posture_parameters=parameters.tolist(),wheel_longitudinal_shift_L_R_m=parameters[5:].tolist(),
            world_camber_deg_L_R=np.rad2deg(camber).tolist(),wheel_heading_offset_deg_L_R=np.rad2deg(heading).tolist(),
            material_sliding_speed_L_R_m_s=speeds.tolist(),
            nonwheel_floor_clearance_m=float(g.nonwheel_clearance()),opposite_leg_clearance_m=float(g.opposite_leg_clearance()))
        return value
    finally:g.set_longitudinal_shift([0.,0.])


def run(a):
    import mujoco
    from scipy.optimize import minimize
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if base.sha(a.model)!=base.MODEL_SHA or base.sha(a.reference)!=base.SOURCE_SHA:raise ValueError('Wrong motion/model')
    cfg=json.loads(a.config.read_text());assert cfg['kind']=='constant_shape_spatial_placement_full_wrench_fit'
    prior=json.loads((a.wrench/'analysis.json').read_text())
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    g=Placement(mujoco.MjModel.from_xml_path(str(a.model)),source);weight=float(g.model.body_mass.sum())*9.81
    bounds=np.asarray(cfg['bounds'],dtype=float);assert bounds.shape==(7,2)
    rows=[]
    for item in prior['rows']:
        frame=item['frame'];seeds=prior['details'][str(frame)]['best_wrench_postures'][:cfg['seeds_per_frame']]
        best=None;attempts=[];counts=dict(total=0,admitted=0,rejected=0)
        for seed in seeds:
            x0=np.r_[seed['posture_deg_and_height_m'],0.,0.]
            def objective(x):
                nonlocal best
                counts['total']+=1
                try:value=measure(g,frame,x,cfg['mu'],weight,cfg['min_material_sliding_speed_m_s'])
                except ValueError:
                    counts['rejected']+=1
                    return 10.+1e-3*float(np.sum(((x-x0)/(bounds[:,1]-bounds[:,0]))**2))
                counts['admitted']+=1
                if best is None or value['weighted_base_wrench_residual']<best['weighted_base_wrench_residual']:best=value
                return value['weighted_base_wrench_residual']**2
            result=minimize(objective,x0,method='Powell',bounds=bounds,
                options=dict(maxfev=cfg['maxfev_per_seed'],xtol=cfg['xtol'],ftol=cfg['ftol']))
            attempts.append(dict(initial=x0.tolist(),success=bool(result.success),message=str(result.message),nfev=int(result.nfev),objective=float(result.fun)))
        assert best is not None
        rows.append(dict(frame=frame,time_s=item['time_s'],root_beta_deg=item['root_beta_deg'],
            prior_grid_best_wrench_residual=item['best_wrench_residual'],best=best,optimizer_attempts=attempts,objective_evaluations=counts))
        print(json.dumps(dict(frame=frame,residual=best['weighted_base_wrench_residual'],parameters=best['posture_parameters'],counts=counts)),flush=True)
    a.output_dir.mkdir(parents=True)
    result=dict(kind=cfg['kind'],rows=rows,config=cfg,source_sha256=base.sha(__file__),
        geometry_source_sha256=base.sha(Path(__file__).with_name('spatial_wheel_posture.py')),
        wrench_source_sha256=base.sha(Path(__file__).with_name('audit_spatial_camber_wrenches.py')),
        config_sha256=base.sha(a.config),prior_wrench_sha256=base.sha(a.wrench/'analysis.json'),
        model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),
        optimizer='scipy.optimize.minimize Powell,bounded,finite evaluation budget,retains best admitted evaluated snapshot',
        full_trajectory_constructed=False,native_contact_or_motor_feasibility_proven=False,
        physics_steps=0,policy_updates=0,task_success_claimed=False,
        limitations=['Shape/placement velocities and accelerations between snapshots are not modeled.',
            'A small conditional wrench residual cannot establish stability or native force agreement.',
            'The0.01m/s sliding-speed margin and parameter bounds are prospective snapshot search choices,not relaxed task gates.',
            'Passive/native actuator and multi-point contact equivalence is not certified.'])
    (a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    time=[r['time_s'] for r in rows]
    ax.plot(time,[r['prior_grid_best_wrench_residual'] for r in rows],'o-',label='Admitted grid,fixed wheel fore/aft')
    ax.plot(time,[r['best']['weighted_base_wrench_residual'] for r in rows],'o-',label='Continuous posture + independent wheel fore/aft')
    ax.set(xlabel='Reference time [s]',ylabel='Normalized conditional base wrench residual',title='Snapshot fitting only | changing-shape dynamics and native support remain unverified')
    ax.legend();ax.grid(alpha=.2);fig.savefig(a.output_dir/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','wrench','config','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

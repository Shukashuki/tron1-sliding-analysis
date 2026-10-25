"""Use explicit sliding-speed inequalities instead of a rejected-step penalty.

The previous TRF fit can stop at the0.01m/s speed boundary. This SLSQP stage
keeps the same physical objective and bounds,but lets feasible steps move
along that boundary. It is still a constant-shape snapshot calculation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import generate_camber_reference as base
from fit_spatial_wheel_postures import Placement,measure


def run(a):
    import mujoco
    from scipy.optimize import minimize
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    prior=json.loads((a.initial_fit/'analysis.json').read_text());cfg=prior['config']
    if base.sha(a.model)!=base.MODEL_SHA or base.sha(a.reference)!=base.SOURCE_SHA:raise ValueError('Wrong motion/model')
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    g=Placement(mujoco.MjModel.from_xml_path(str(a.model)),source);weight=float(g.model.body_mass.sum())*9.81
    bounds=np.asarray(cfg['bounds'],dtype=float);lower=bounds[:,0];span=bounds[:,1]-lower;rows=[]
    for old in prior['rows']:
        frame=old['frame'];initial=np.asarray(old['best']['posture_parameters']);best=measure(g,frame,initial,cfg['mu'],weight,cfg['min_material_sliding_speed_m_s'])
        counts=dict(total=0,admitted_above_speed_margin=0,invalid_geometry_or_near_static=0)
        cached={}
        def value(z):
            nonlocal best
            key=np.asarray(z).tobytes()
            if key in cached:return cached[key]
            counts['total']+=1;x=lower+z*span
            try:v=measure(g,frame,x,cfg['mu'],weight,1e-4)
            except ValueError:
                counts['invalid_geometry_or_near_static']+=1;cached[key]=None;return None
            if min(v['material_sliding_speed_L_R_m_s'])>=cfg['min_material_sliding_speed_m_s']:
                counts['admitted_above_speed_margin']+=1
                if v['weighted_base_wrench_residual']<best['weighted_base_wrench_residual']:best=v
            cached[key]=v;return v
        def objective(z):
            v=value(z)
            return 10.+1e-3*float(z@z) if v is None else v['weighted_base_wrench_residual']**2
        def constraints(z):
            v=value(z)
            return np.full(2,-1.) if v is None else np.asarray(v['material_sliding_speed_L_R_m_s'])-cfg['min_material_sliding_speed_m_s']
        z0=(initial-lower)/span
        result=minimize(objective,z0,method='SLSQP',bounds=[(0.,1.)]*7,
            constraints=[dict(type='ineq',fun=constraints)],options=dict(maxiter=150,ftol=1e-12,eps=1e-6))
        rows.append(dict(frame=frame,time_s=old['time_s'],root_beta_deg=old['root_beta_deg'],initial_wrench_residual=old['best']['weighted_base_wrench_residual'],
            best=best,optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),
            final_optimizer_objective=float(result.fun),retained_best_may_differ_from_final_iterate=True,objective_evaluations=counts))
        print(json.dumps(dict(frame=frame,residual=best['weighted_base_wrench_residual'],parameters=best['posture_parameters'],counts=counts,message=str(result.message))),flush=True)
    a.output_dir.mkdir(parents=True)
    proof=dict(kind='constant_shape_spatial_placement_SLSQP_explicit_sliding_constraints',rows=rows,config=cfg,
        source_sha256=base.sha(__file__),placement_source_sha256=base.sha(Path(__file__).with_name('fit_spatial_wheel_postures.py')),
        geometry_source_sha256=base.sha(Path(__file__).with_name('spatial_wheel_posture.py')),
        initial_fit_sha256=base.sha(a.initial_fit/'analysis.json'),model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),
        optimizer=dict(method='SLSQP',Jacobian='finite difference',normalized_variables=True,eps=1e-6,maxiter=150,
            explicit_inequalities='each material sliding speed >=0.01m/s'),
        physics_steps=0,policy_updates=0,full_time_varying_reference_produced=False,native_contact_stability_proven=False,task_success_claimed=False,
        limitations=prior['limitations'])
    (a.output_dir/'analysis.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');t=[r['time_s'] for r in rows]
    ax.plot(t,[r['initial_wrench_residual'] for r in rows],'o-',label='Admitted TRF snapshot')
    ax.plot(t,[r['best']['weighted_base_wrench_residual'] for r in rows],'o-',label='Explicit speed-constraint SLSQP')
    ax.set(xlabel='Reference time [s]',ylabel='Normalized conditional base wrench residual',title='Constant-shape snapshot fit | no full-trajectory or native-stability claim')
    ax.legend();ax.grid(alpha=.2);fig.savefig(a.output_dir/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','initial-fit','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

"""Local bounded TRF refinement of admitted seven-variable wrench snapshots.

Starts from the retained admitted Powell samples,not its possibly worse final
iterate. Changes no physical model,target or task-acceptance requirement.
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
    from scipy.optimize import least_squares
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    prior=json.loads((a.initial_fit/'analysis.json').read_text());cfg=prior['config']
    if base.sha(a.model)!=base.MODEL_SHA or base.sha(a.reference)!=base.SOURCE_SHA:raise ValueError('Wrong motion/model')
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    g=Placement(mujoco.MjModel.from_xml_path(str(a.model)),source);weight=float(g.model.body_mass.sum())*9.81
    bounds=np.asarray(cfg['bounds'],dtype=float);scale=np.array([weight]*3+[weight*.5]*3)
    rows=[]
    for old in prior['rows']:
        frame=old['frame'];initial=np.asarray(old['best']['posture_parameters']);best=measure(g,frame,initial,cfg['mu'],weight,cfg['min_material_sliding_speed_m_s'])
        np.testing.assert_allclose(best['weighted_base_wrench_residual'],old['best']['weighted_base_wrench_residual'],atol=1e-12,rtol=0)
        counts=dict(total=0,admitted=0,rejected=0)
        def residual(x):
            nonlocal best
            counts['total']+=1
            try:value=measure(g,frame,x,cfg['mu'],weight,cfg['min_material_sliding_speed_m_s'])
            except ValueError:
                counts['rejected']+=1
                return np.full(6,np.sqrt(10./6.)+1e-3*np.linalg.norm((x-initial)/(bounds[:,1]-bounds[:,0])))
            counts['admitted']+=1
            if value['weighted_base_wrench_residual']<best['weighted_base_wrench_residual']:best=value
            return np.asarray(value['unactuated_base_residual'])/scale
        result=least_squares(residual,initial,bounds=(bounds[:,0],bounds[:,1]),method='trf',
            x_scale=np.array([20.,20.,20.,20.,.1,.1,.1]),max_nfev=150,ftol=1e-10,xtol=1e-8,gtol=1e-8)
        rows.append(dict(frame=frame,time_s=old['time_s'],root_beta_deg=old['root_beta_deg'],
            initial_wrench_residual=old['best']['weighted_base_wrench_residual'],best=best,
            success=bool(result.success),message=str(result.message),nfev=int(result.nfev),
            retained_best_may_differ_from_final_iterate=True,objective_evaluations=counts))
        print(json.dumps(dict(frame=frame,residual=best['weighted_base_wrench_residual'],parameters=best['posture_parameters'],counts=counts)),flush=True)
    a.output_dir.mkdir(parents=True)
    proof=dict(kind='constant_shape_spatial_placement_TRF_refinement',rows=rows,config=cfg,
        source_sha256=base.sha(__file__),placement_source_sha256=base.sha(Path(__file__).with_name('fit_spatial_wheel_postures.py')),
        geometry_source_sha256=base.sha(Path(__file__).with_name('spatial_wheel_posture.py')),
        initial_fit_sha256=base.sha(a.initial_fit/'analysis.json'),model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),
        optimizer=dict(method='bounded TRF',Jacobian='finite difference2-point',max_nfev=150,x_scale=[20,20,20,20,.1,.1,.1]),
        physics_steps=0,policy_updates=0,full_time_varying_reference_produced=False,native_contact_stability_proven=False,task_success_claimed=False,
        limitations=prior['limitations'])
    (a.output_dir/'analysis.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');t=[r['time_s'] for r in rows]
    ax.plot(t,[r['initial_wrench_residual'] for r in rows],'o-',label='Retained admitted Powell snapshot')
    ax.plot(t,[r['best']['weighted_base_wrench_residual'] for r in rows],'o-',label='TRF seven-variable refinement')
    ax.set(xlabel='Reference time [s]',ylabel='Normalized conditional base wrench residual',title='Local constant-shape snapshots | no native force,stability or full trajectory claim')
    ax.grid(alpha=.2);ax.legend();fig.savefig(a.output_dir/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','initial-fit','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

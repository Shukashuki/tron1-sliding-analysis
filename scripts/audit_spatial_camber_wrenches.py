"""Reject XY-only snapshot solutions using full floating-base force/moment balance.

Constant-shape local kinematics,unmodified authored inertia,and saturated point
Coulomb forces are explicit assumptions. No optimizer result proves a full
time-varying reference or native contact stability.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import generate_camber_reference as base
from spatial_wheel_posture import Geometry


def projection(required,per_load,total_normal,weight,moment_length=.5):
    """Project six generalized base forces onto a nonnegative two-load segment."""
    scale=np.array([weight]*3+[weight*moment_length]*3)
    target=required[:6]/scale
    first=total_normal*per_load[1,:6]/scale
    delta=total_normal*(per_load[0,:6]-per_load[1,:6])/scale
    denom=float(delta@delta)
    left=.5 if denom<1e-24 else float(np.clip((target-first)@delta/denom,0.,1.))
    force=total_normal*(left*per_load[0]+(1-left)*per_load[1])
    return left,force,required-force,float(np.linalg.norm((required[:6]-force[:6])/scale))


def evaluate(g,frame,parameters,mu,weight):
    roll,pitch,left,right=np.deg2rad(parameters[:4]);height=parameters[4]
    state=g.frozen_shape_inverse_state(frame,roll,pitch,np.array([left,right]),height)
    required=state['required_generalized_force'];slip=state['material_surface_velocity_w'][:,:2]
    speed=np.linalg.norm(slip,axis=-1)
    if np.any(speed<=1e-4) or required[2]<=0:raise ValueError('Fully sliding positive normal hypothesis unavailable')
    directions=np.c_[-mu*slip/speed[:,None],np.ones(2)]
    generalized=[]
    for point,body,direction in zip(state['points_w'],g.wheels,directions):
        g.mj.mj_jac(g.model,g.data,g.jp,g.jr,point,body)
        generalized.append(g.jp.T@direction)
    generalized=np.asarray(generalized)
    np.testing.assert_allclose(generalized[:,2],1.,atol=1e-12,rtol=0)
    fraction,contact,residual,error=projection(required,generalized,required[2],weight)
    return dict(posture_deg_and_height_m=list(map(float,parameters)),
        conditional_normal_load_L_R_n=[float(required[2]*fraction),float(required[2]*(1-fraction))],
        required_total_normal_over_mg=float(required[2]/weight),
        weighted_base_wrench_residual=error,
        force_residual_over_mg=np.linalg.norm(residual[:3]).item()/weight,
        root_local_moment_residual_over_mg_halfmeter=np.linalg.norm(residual[3:6]).item()/(weight*.5),
        required_generalized_force=required.tolist(),conditional_contact_generalized_force=contact.tolist(),
        unactuated_base_residual=residual[:6].tolist(),joint_generalized_effort_required=(required-contact)[6:].tolist(),
        qpos=state['qpos'].tolist(),qvel=state['qvel'].tolist(),qacc=state['qacc'].tolist(),
        wheel_point_velocity_w_m_s=state['material_surface_velocity_w'].tolist())


def run(a):
    import mujoco
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    atlas=json.loads((a.postures/'analysis.json').read_text())
    assert base.sha(a.model)==atlas['model_sha256']==base.MODEL_SHA
    assert base.sha(a.reference)==atlas['reference_sha256']==base.SOURCE_SHA
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    model=mujoco.MjModel.from_xml_path(str(a.model));g=Geometry(model,source);weight=float(model.body_mass.sum())*9.81
    rows=[];records={};details={}
    with np.load(a.postures/'posture_screen.npz',allow_pickle=False) as grid:
        for row in atlas['rows']:
            frame=row['frame'];parameters=grid[str(frame)+'__posture_deg_and_height_m'];evaluated=[]
            for values in parameters:evaluated.append(evaluate(g,frame,values,atlas['mu'],weight))
            selected=sorted(evaluated,key=lambda r:r['weighted_base_wrench_residual'])[:5]
            xy_choice=evaluate(g,frame,row['best_posture_deg_and_height_m'],atlas['mu'],weight)
            details[str(frame)]=dict(best_wrench_postures=selected,previous_XY_only_choice=xy_choice)
            rows.append(dict(frame=frame,time_s=row['time_s'],root_beta_deg=row['root_beta_deg'],postures_recomputed=len(evaluated),
                previous_XY_only_choice_wrench_residual=xy_choice['weighted_base_wrench_residual'],
                best_wrench_residual=selected[0]['weighted_base_wrench_residual'],
                best_force_residual_over_mg=selected[0]['force_residual_over_mg'],
                best_moment_residual_over_mg_halfmeter=selected[0]['root_local_moment_residual_over_mg_halfmeter'],
                best_posture_deg_and_height_m=selected[0]['posture_deg_and_height_m']))
            records[str(frame)+'__weighted_wrench_residual']=np.asarray([r['weighted_base_wrench_residual'] for r in evaluated])
            print(json.dumps(rows[-1]),flush=True)
    a.output_dir.mkdir(parents=True)
    result=dict(kind='conditional_constant_shape_full_floating_base_wrench_audit',rows=rows,details=details,mu=atlas['mu'],
        moment_normalization_length_m=.5,weight_n=weight,source_sha256=base.sha(__file__),geometry_source_sha256=base.sha(Path(__file__).with_name('spatial_wheel_posture.py')),
        posture_atlas_sha256=base.sha(a.postures/'analysis.json'),model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),
        generalized_force_method='mj_kinematics+mj_comPos+mj_comVel+mj_rne;M*qacc+C;no contact solver or stepping',
        physics_steps=0,policy_updates=0,normal_force_or_full_trajectory_feasibility_proven=False,task_success_claimed=False,
        limitations=['Shape is locally constant;future changing roll/pitch/height/abduction adds acceleration and angular momentum.',
            'Only two ideal lowest cylinder material points and saturated kinetic Coulomb are modeled;not native multi-point contact.',
            'Motor/passive/actuator equivalence is not certified by required joint generalized efforts.',
            'Finite posture sampling cannot prove task impossibility or global feasibility.',
            'Normal force is constrained to required inertial/gravity Fz and may be allocated to one wheel only.'])
    (a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');np.savez_compressed(a.output_dir/'wrench_screen.npz',**records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    t=[r['time_s'] for r in rows]
    ax.plot(t,[r['previous_XY_only_choice_wrench_residual'] for r in rows],'o-',label='XY-only selected pose,now full wrench')
    ax.plot(t,[r['best_wrench_residual'] for r in rows],'o-',label='Best full-wrench pose on admitted grid')
    ax.set(xlabel='Reference time [s]',ylabel='Normalized base wrench residual',title='Conditional local constant shape | force/moment equilibrium not achieved by XY cancellation alone')
    ax.legend();ax.grid(alpha=.2);fig.savefig(a.output_dir/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','postures','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

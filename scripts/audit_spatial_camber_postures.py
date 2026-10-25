"""Screen reachable root/leg postures before constructing a dynamic reference.

The force calculation is a conditional fully-sliding XY-direction projection
with a root-acceleration proxy. It omits angular momentum and posture rates;
therefore no selected snapshot is claimed to be a feasible trajectory.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import generate_camber_reference as base
from spatial_wheel_posture import Geometry,axle_geometry
from diagnose_reference_contact_dynamics import closest_load_allocation


def run(a):
    import mujoco
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if base.sha(a.reference)!=base.SOURCE_SHA or base.sha(a.model)!=base.MODEL_SHA:raise ValueError('Require declared original motion/model')
    cfg=json.loads(a.config.read_text());assert cfg['kind']=='spatial_camber_instantaneous_geometry_and_force_direction_audit'
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    model=mujoco.MjModel.from_xml_path(str(a.model));g=Geometry(model,source)
    mass=float(model.body_mass.sum());weight=mass*9.81
    acceleration=np.gradient(source['qvel_mujoco'][:,:3],source['time_s'],axis=0,edge_order=2)
    grid=list(itertools.product(cfg['root_roll_deg'],cfg['root_pitch_deg'],cfg['left_abduction_deg'],cfg['right_abduction_deg'],cfg['root_height_samples_m']))
    summaries=[];arrays={};top={};max_axis_error=0.
    for frame in cfg['reference_frames']:
        candidates=[];rejected=dict(unreachable_or_joint_limit=0,nonwheel_floor_intersection=0,opposite_leg_intersection=0,not_fully_sliding=0)
        q0=source['qpos_mujoco'][frame];v0=source['qvel_mujoco'][frame];yaw=2*np.arctan2(q0[6],q0[3])
        Rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
        target=mass*acceleration[frame,:2]
        for row in grid:
            roll,pitch,left,right=np.deg2rad(row[:4]);height=row[4]
            try:q=g.posture(frame,roll,pitch,np.array([left,right]),height)
            except ValueError:rejected['unreachable_or_joint_limit']+=1;continue
            if g.nonwheel_clearance() < -1e-9:rejected['nonwheel_floor_intersection']+=1;continue
            if g.opposite_leg_clearance() < -1e-9:rejected['opposite_leg_intersection']+=1;continue
            v,slip,points,axes=g.instantaneous_surface_motion(q,v0[:3],np.array([0.,0.,v0[5]]))
            expected,camber,heading=axle_geometry(roll,pitch,[left,right])
            error=float(abs(axes-expected@Rz.T).max());max_axis_error=max(max_axis_error,error)
            assert error<1e-12
            speed=np.linalg.norm(slip[:,:2],axis=-1)
            if np.any(speed<=cfg['sliding_speed_cutoff_m_s']):rejected['not_fully_sliding']+=1;continue
            force,fraction,residual=closest_load_allocation(target,cfg['mu']*(-slip[:,:2]/speed[:,None]),np.asarray(weight))
            candidates.append(dict(posture_deg_and_height_m=list(row),root_tilt_deg=float(np.degrees(np.arccos(np.cos(roll)*np.cos(pitch)))),
                conditional_XY_force_residual_over_mg=float(residual/weight),conditional_left_load_fraction=float(fraction),
                conditional_XY_force_n=force.tolist(),rolling_heading_offset_deg_L_R=np.rad2deg(heading).tolist(),
                world_camber_deg_L_R=np.rad2deg(camber).tolist(),material_surface_velocity_w_m_s=slip.tolist(),
                minimum_nonwheel_floor_clearance_m=float(g.nonwheel_clearance()),minimum_opposite_leg_clearance_m=float(g.opposite_leg_clearance()),qpos=q.tolist(),qvel_frozen_shape_twist=v.tolist()))
        candidates.sort(key=lambda r:r['conditional_XY_force_residual_over_mg'])
        assert candidates
        baseline=[r for r in candidates if r['posture_deg_and_height_m'][:2]==[0,0]]
        assert baseline
        chosen=candidates[:cfg['top_postures_per_frame']];top[str(frame)]=chosen
        beta=float(np.degrees(np.arctan2(v0[1],v0[0])-yaw))
        summaries.append(dict(frame=frame,time_s=float(source['time_s'][frame]),root_beta_deg=beta,
            attempted=len(grid),admitted_instantaneous_postures=len(candidates),rejected=rejected,
            yaw_only_best_conditional_residual_over_mg=min(r['conditional_XY_force_residual_over_mg'] for r in baseline),
            spatial_best_conditional_residual_over_mg=chosen[0]['conditional_XY_force_residual_over_mg'],
            best_posture_deg_and_height_m=chosen[0]['posture_deg_and_height_m'],
            best_rolling_heading_offset_deg_L_R=chosen[0]['rolling_heading_offset_deg_L_R']))
        for name in ['posture_deg_and_height_m','root_tilt_deg','conditional_XY_force_residual_over_mg','rolling_heading_offset_deg_L_R','world_camber_deg_L_R']:
            arrays[str(frame)+'__'+name]=np.asarray([r[name] for r in candidates])
        print(json.dumps(summaries[-1]),flush=True)
    a.output_dir.mkdir(parents=True)
    result=dict(kind=cfg['kind'],status='instantaneous_posture_screen_completed',rows=summaries,top_postures=top,
        max_analytic_vs_native_FK_axle_error=max_axis_error,model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),
        source_sha256=base.sha(__file__),geometry_source_sha256=base.sha(Path(__file__).with_name('spatial_wheel_posture.py')),config_sha256=base.sha(a.config),
        mass_kg=mass,mu=cfg['mu'],physics_integration_steps=0,policy_updates=0,task_success_claimed=False,
        selected_snapshots_are_full_reference_trajectories=False,normal_loads_or_contact_feasibility_proven=False,
        limitations=['Frozen posture twist omits leg/root-shape rates and is not a trajectory derivative.',
            'Proxy force target is mass times original root acceleration,not recomputed whole-body COM acceleration.',
            'Assumed total normal force is mg;no moment balance,motor torque,angular momentum or native contact is solved.',
            'Fully sliding isotropic Coulomb is a conditional law,not a calibrated PhysX force predictor.',
            'A finite sampled posture grid does not establish global reachability or task impossibility.',
            'Root XY,projected body heading and reference speed stay fixed;sampled heights/angles add no task acceptance gates.'])
    (a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(a.output_dir/'posture_screen.npz',**arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(summaries),figsize=(20,4.5),layout='constrained',sharey=True)
    for ax,row in zip(axes,summaries):
        key=str(row['frame']);p=arrays[key+'__posture_deg_and_height_m'];error=arrays[key+'__conditional_XY_force_residual_over_mg']
        plot=ax.scatter(arrays[key+'__root_tilt_deg'],error,c=p[:,0],s=8,alpha=.45,cmap='coolwarm',vmin=-30,vmax=30)
        ax.set(title=f"t={row['time_s']:.2f}s,beta={row['root_beta_deg']:.1f}deg",xlabel='Root tilt [deg]');ax.grid(alpha=.2)
    axes[0].set_ylabel('Conditional snapshot XY residual /mg');fig.colorbar(plot,ax=axes,label='Root roll [deg]')
    fig.suptitle('Reachable geometry snapshots | frozen shape,proxy force target | not trajectory/dynamic validation')
    fig.savefig(a.output_dir/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','config','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

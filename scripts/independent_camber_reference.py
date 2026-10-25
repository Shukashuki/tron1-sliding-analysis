"""Independent left/right camber with analytic two-link contact reconstruction.

Abduction sets camber. Hip/knee reconstruction keeps both geometric wheel
supports on the floor and preserves their body-longitudinal coordinates.
Base path,orientation,speed and nominal reference height are unchanged. This
adds no absolute-height gate or reward;dynamic loading remains unverified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import generate_camber_reference as base


def planar_angles(target_xz, thigh_xz, shin_xz, branch):
    target=np.asarray(target_xz,dtype=float)
    thigh,shin=np.asarray(thigh_xz,dtype=float),np.asarray(shin_xz,dtype=float)
    a,b=np.linalg.norm(thigh),np.linalg.norm(shin)
    cosine=(np.sum(target*target,axis=-1)-a*a-b*b)/(2*a*b)
    if np.any(abs(cosine)>1+1e-10) or branch not in (-1,1):
        raise ValueError("Planar wheel target is unreachable")
    phi=branch*np.arccos(cosine.clip(-1,1))
    theta=np.arctan2(target[...,1],target[...,0])-np.arctan2(b*np.sin(phi),a+b*np.cos(phi))
    alpha,beta=np.arctan2(thigh[1],thigh[0]),np.arctan2(shin[1],shin[0])
    return alpha-theta,beta-alpha-phi


class Generator(base.Generator):
    def prepare_qpos(self,coefficients_deg,timing):
        original=self.source
        q=original["qpos_mujoco"].copy()
        gamma=base.camber_profile(self.time,coefficients_deg,timing)
        self.pose(q[0])
        # Work in the base frame;the accepted source is yaw-only and starts yaw0.
        if not np.allclose(q[0,3:7],[1,0,0,0],atol=1e-10):
            raise ValueError("Require the declared yaw-normalized car source")
        centers=self.data.geom_xpos[self.geoms].copy()-q[0,:3]
        for i,side in enumerate(("L","R")):
            abad=self.model.body("abad_"+side+"_Link").pos
            hip=self.model.body("hip_"+side+"_Link").pos
            thigh=self.model.body("knee_"+side+"_Link").pos
            shin=self.model.body("wheel_"+side+"_Link").pos
            g=self.geoms[i]
            lateral=hip[1]+thigh[1]+shin[1]+self.model.geom_pos[g,1]
            radius,halfwidth=self.model.geom_size[g,:2]
            angle=gamma[:,i]
            height=radius*np.cos(angle)+halfwidth*abs(np.sin(angle))
            z=(height-q[:,2]-abad[2]-np.sin(angle)*lateral)/np.cos(angle)
            x=np.full(len(q),centers[i,0]-abad[0]-hip[0])
            h=self.model.joint("hip_"+side+"_Joint")
            k=self.model.joint("knee_"+side+"_Joint")
            hs,ks=self.model.jnt_axis[h.id,1],self.model.jnt_axis[k.id,1]
            phi=np.arctan2(shin[2],shin[0])-np.arctan2(thigh[2],thigh[0])-ks*q[0,k.qposadr.item()]
            theta,knee=planar_angles(np.stack((x,z),-1),thigh[[0,2]],shin[[0,2]],int(np.sign(phi)))
            q[:,h.qposadr.item()]=theta/hs
            q[:,k.qposadr.item()]=knee/ks
        q[:,self.abads]=gamma
        return q

    def build(self,coefficients_deg,timing):
        original=self.source
        q=self.prepare_qpos(coefficients_deg,timing)
        prepared=dict(original,qpos_mujoco=q)
        try:
            self.source=prepared
            output=super().build(coefficients_deg,timing)
        finally:
            self.source=original
        if np.max(abs(output["wheel_geometric_gap_m"]))>1e-7:
            raise ValueError("Independent-camber reconstruction lost geometric contact")
        np.testing.assert_allclose(output["qpos_mujoco"][:,2],original["qpos_mujoco"][:,2],atol=1e-10,rtol=0)
        output["independent_wheel_camber_commands"]=np.asarray(True)
        return output


def predict(generator,coefficients_deg,timing):
    q=generator.prepare_qpos(coefficients_deg,timing)
    velocity=base.differentiate(generator.model,q,generator.dt)
    velocity[:,generator.spin_v]=0.
    values,gaps=[],[]
    for frame in range(len(q)):
        generator.pose(q[frame]);points,axes,_=generator.wheel_geometry()
        v=[]
        for p,a,b in zip(points,axes,generator.wheels):
            lateral=a-[0,0,a[2]];lateral/=np.linalg.norm(lateral)
            generator.mj.mj_jac(generator.model,generator.data,generator.jp,generator.jr,p,b)
            v.append(lateral@generator.jp@velocity[frame])
        values.append(v);gaps.append(points[:,2])
    gaps=np.asarray(gaps)
    if np.max(abs(gaps))>1e-7:
        raise ValueError("Scored geometric wheel contacts must both touch")
    return np.asarray(values)


def run(a):
    import mujoco
    import scipy
    from scipy.optimize import least_squares
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if base.sha(a.reference)!=base.SOURCE_SHA or base.sha(a.model)!=base.MODEL_SHA:
        raise ValueError("Require the hash-declared car source and robot model")
    cfg=json.loads(a.config.read_text())
    if cfg["kind"]!="independent_camber_kinematic_reference_v1":raise ValueError("Wrong configuration")
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    generator=Generator(mujoco.MjModel.from_xml_path(str(a.model)),source)
    timing=cfg["ramp_times_s"]
    core=(generator.time>=1.44-1e-9)&(generator.time<=4.+1e-9)
    zero=predict(generator,np.zeros((2,4)),timing)
    spec=list(cfg["fixed_cases"]);fits=[]
    for goal in cfg["fit_cases"]:
        ratios=np.asarray(goal["signed_velocity_ratios_L_R"],dtype=float)
        if ratios.shape!=(2,) or not np.isfinite(ratios).all():raise ValueError("Invalid independent slip targets")
        target=zero*ratios
        calls=0
        def residual(c):
            nonlocal calls
            calls+=1
            actual=predict(generator,c.reshape(2,4),timing)
            return np.r_[(actual[core]-target[core]).ravel(),cfg["coefficient_regularization_m_s"]*c/15.]
        initial=np.asarray(goal["initial_coefficients_deg"],dtype=float)
        result=least_squares(residual,initial.ravel(),bounds=(-15.,15.),diff_step=1e-3,
                             max_nfev=cfg["max_nfev"],ftol=1e-7,xtol=1e-7,gtol=1e-7)
        coeff=result.x.reshape(2,4)
        spec.append(dict(id=goal["id"],title=goal["title"],coefficients_deg=coeff.tolist()))
        actual=predict(generator,coeff,timing)
        fits.append(dict(id=goal["id"],target_signed_velocity_ratios_L_R=ratios.tolist(),
            fitted_coefficients_deg=coeff.tolist(),optimizer_success=bool(result.success),message=str(result.message),
            nfev=int(result.nfev),objective_evaluations=calls,
            core_target_RMSE_m_s_L_R=np.sqrt(np.mean((actual[core]-target[core])**2,axis=0)).tolist(),
            zero_core_signed_velocity_m_s_median_L_R=np.median(zero[core],axis=0).tolist(),
            target_core_signed_velocity_m_s_median_L_R=np.median(target[core],axis=0).tolist(),
            actual_core_signed_velocity_m_s_median_L_R=np.median(actual[core],axis=0).tolist(),
            optimization_convergence_is_target_or_dynamic_success=False))
    a.output_dir.mkdir(parents=True)
    rows=[]
    for row in spec:
        output=generator.build(row["coefficients_deg"],timing)
        fast=predict(generator,row["coefficients_deg"],timing)
        np.testing.assert_allclose(fast,output["wheel_point_lateral_velocity_m_s"],atol=2e-10,rtol=0)
        path=a.output_dir/(row["id"]+".npz");np.savez_compressed(path,**output)
        rows.append(dict(**row,reference_sha256=base.sha(path),metrics=base.describe(output,source)))
    proof=dict(kind=cfg["kind"],status="independent_kinematic_references_generated",rows=rows,fits=fits,
        generator_sha256=base.sha(__file__),forward_generator_sha256=base.sha(base.__file__),
        config_sha256=base.sha(a.config),reference_sha256=base.sha(a.reference),model_sha256=base.sha(a.model),
        scipy_version=scipy.__version__,mujoco_version=mujoco.__version__,frames=len(generator.time),fps=50,
        solver="bounded TRF numerical Jacobian over8 independent Bernstein coefficients;analytic hip/knee reconstruction",
        both_wheels_geometrically_touch=True,positive_normal_force_proven=False,absolute_height_gate_added=False,
        base_XY_speed_yaw_and_nominal_reference_Z_preserved=True,
        integration_steps=0,policy_updates=0,reward_changed=False,task_success_claimed=False,
        limits=["Contact is geometrical;normal loads and traction feasibility are not solved.",
            "This is direct kinematic reference synthesis,not human-motion GMR retargeting.",
            "The rigid cylinder changes from a bottom support line to a rim at nonzero camber;no tread elasticity or camber thrust is introduced.",
            "Wheel lateral XY is deliberately allowed to move;future tracking must use these generated wheel references.",
            "Motor torque,centroidal equilibrium and closed-loop continuous contact require native validation."])
    (a.output_dir/"generation.json").write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(fits=fits,rows=[dict(id=r["id"],metrics=r["metrics"]) for r in rows])))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("reference","model","config","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())

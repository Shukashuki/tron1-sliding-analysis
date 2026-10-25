"""Conditional force audit of generated camber references,no simulator rollout.

Fully sliding isotropic kinetic Coulomb is a stated hypothesis,not a native
PhysX prediction or task impossibility claim. Free nonnegative wheel load split
is allowed,including single-wheel support;angular momentum is not constrained.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import generate_camber_reference as reference
from diagnose_reference_contact_dynamics import closest_load_allocation


def run(a):
    import mujoco
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    proof=json.loads((a.references/"generation.json").read_text())
    assert reference.sha(a.model)==proof["model_sha256"]
    model=mujoco.MjModel.from_xml_path(str(a.model));mass=float(model.body_mass.sum())
    weight=mass*9.81
    rows=[];records={}
    for row in proof["rows"]:
        path=a.references/(row["id"]+".npz");assert reference.sha(path)==row["reference_sha256"]
        with np.load(path,allow_pickle=False) as f:raw={k:f[k].copy() for k in f.files}
        time=raw["time_s"];core=(time>=1.44-1e-9)&(time<=4.+1e-9)
        masses=np.array([model.body(str(n)).mass.item() for n in raw["body_names"]])
        assert abs(float(masses.sum())-mass)<1e-10
        com_velocity=np.average(raw["body_lin_vel_w"],axis=1,weights=masses)
        acceleration=np.gradient(com_velocity,time,axis=0,edge_order=2)
        required=mass*acceleration[:,:2];normal=mass*(9.81+acceleration[:,2])
        slip=raw["wheel_point_velocity_w_m_s"][...,:2]
        speed=np.linalg.norm(slip,axis=-1)
        valid=(speed>1e-4).all(-1)&(normal>0)
        assert valid[core].all() and raw["wheel_geometrically_touching"][core].all()
        direction=-slip[valid]/speed[valid,:,None]
        closest,fraction,residual=closest_load_allocation(required[valid],.7*direction,normal[valid])
        error=np.full(len(time),np.nan);error[valid]=residual/weight
        normal_q=np.quantile(normal[core]/weight,[.5,.95]).tolist()
        values=dict(id=row["id"],reference_sha256=row["reference_sha256"],
            core_best_force_residual_over_mg_median_p95=np.quantile(error[core],[.5,.95]).tolist(),
            core_required_COM_XY_acceleration_over_g_median_p95=np.quantile(np.linalg.norm(acceleration[core,:2],axis=-1)/9.81,[.5,.95]).tolist(),
            core_required_total_normal_over_mg_median_p95=normal_q,
            core_material_lateral_velocity_m_s_median_L_R=np.median(raw["wheel_point_lateral_velocity_m_s"][core],axis=0).tolist())
        rows.append(values)
        records[row["id"]]=dict(time_s=time,force_residual_over_mg=error,
            required_XY_force_n=required,required_total_normal_n=normal,
            core=core,material_surface_velocity_w_m_s=raw["wheel_point_velocity_w_m_s"],
            closest_XY_force_n_on_valid=closest,left_load_fraction_on_valid=fraction,valid=valid)
    a.output_dir.mkdir(parents=True)
    result=dict(kind="conditional_camber_reference_kinetic_Coulomb_force_audit",rows=rows,mu=.7,mass_kg=mass,
        ground_normal=[0,0,1],core_interval_s=[1.44,4.],new_physics_steps=0,new_policy_inference_or_updates=0,
        source_sha256=reference.sha(__file__),load_projection_source_sha256=reference.sha(Path(__file__).with_name("diagnose_reference_contact_dynamics.py")),
        reference_generation_sha256=reference.sha(a.references/"generation.json"),model_sha256=reference.sha(a.model),
        law="Ft_i=-0.7*Fn_i*v_surface_XY_i/|v_surface_XY_i|;FnL+FnR=m*(g+aCOMz);Fn_i>=0",
        force_equilibrium_proven=False,native_PhysX_forces_predicted=False,task_impossibility_claimed=False,task_success_claimed=False,
        inference="Small kinematic slip-target error does not establish force consistency.When both surface velocities slide nearly together,saturated Coulomb forces can require much more horizontal acceleration than the fixed body path.",
        limitations=["Conditional fully-sliding law excludes static adhesion and tire deformation/camber thrust.",
            "Load split is ideal and unconstrained by angular momentum,joint torques or actual contact solvers.",
            "Reference derivatives are sampled at50Hz and not measured contact forces.",
            "A policy can deviate from reference wheel motions while retaining body task objectives;this is not a task infeasibility proof.",
            "No friction change,reference scaling or relaxed task gate is introduced."])
    (a.output_dir/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    arrays={key+"__"+name:value for key,record in records.items() for name,value in record.items()}
    np.savez_compressed(a.output_dir/"force_diagnostic.npz",**arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout="constrained")
    for row in rows:
        record=records[row["id"]];t=record["time_s"]
        axes[0].plot(t,np.linalg.norm(record["required_XY_force_n"],axis=-1)/weight,label=row["id"])
        axes[1].plot(t,record["force_residual_over_mg"],label=row["id"])
    for ax,title in zip(axes,["Required horizontal COM force /mg","Best conditional saturated-friction force residual /mg"]):
        ax.axvspan(1.44,4,color="gray",alpha=.12);ax.set(title=title,xlabel="Reference time [s]",ylim=(0,None));ax.legend();ax.grid(alpha=.2)
    fig.suptitle("Kinematic camber references | conditional mu=.7 Coulomb hypothesis,not native force predictions")
    fig.savefig(a.output_dir/"comparison.png",dpi=150);plt.close(fig)
    print(json.dumps(rows))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("references","model","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())

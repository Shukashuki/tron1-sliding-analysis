"""Make an explicit spline seed from retained geometry snapshots;not a solution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

import generate_camber_reference as base
from hybrid_contact_reference import Problem,rim_points


def run(a):
    import mujoco
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if base.sha(a.model)!=base.MODEL_SHA or base.sha(a.reference)!=base.SOURCE_SHA:raise ValueError('Wrong model/motion')
    prior=json.loads(a.postures.read_text());cfg=json.loads(a.config.read_text())
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    p=Problem(mujoco.MjModel.from_xml_path(str(a.model)),source,cfg)
    original=np.r_[0.,0.,.75,source['qpos_mujoco'][0,p.legs]]
    times=[0.,.4];targets=[original,original]
    for row in prior['rows']:
        q=np.asarray(row['best']['qpos']);params=row['best']['posture_parameters']
        times.append(row['time_s']);targets.append(np.r_[np.deg2rad(params[:2]),q[2],q[p.legs]])
    times.extend([5.5,6.96]);targets.extend([original,original])
    target=PchipInterpolator(times,np.asarray(targets),axis=0)(p.time)
    c=p.C.copy();c[:,0:9]=np.linalg.lstsq(p.B,target,rcond=None)[0]
    c[:3,:9]=p.C[:3,:9];c=np.clip(c,p.lower+1e-10,p.upper-1e-10)
    _,q,v,_=p.poses(c);v[:,p.spin_dof]=0.;spin=np.zeros((len(q),2));lower_rim=[]
    for i in range(len(q)):
        p.data.qpos[:]=q[i];mujoco.mj_kinematics(p.model,p.data);mujoco.mj_comPos(p.model,p.data)
        chosen=[]
        for j,(body,geom) in enumerate(zip(p.wheel_bodies,p.geoms)):
            axle=p.data.xmat[body].reshape(3,3)[:,1];points=rim_points(p.data.geom_xpos[geom],axle,*p.model.geom_size[geom,:2])
            index=int(np.argmin(points[:,2]));chosen.append(index)
            forward=np.cross(axle,[0.,0.,1.]);forward/=max(np.linalg.norm(forward),1e-10)
            mujoco.mj_jac(p.model,p.data,p.jp,p.jr,points[index],body)
            spin[i,j]=-(forward@p.jp@v[i])/(forward@p.jp[:,p.spin_dof[j]])
        lower_rim.append(chosen)
    phase=np.zeros_like(spin);phase[0]=source['qpos_mujoco'][0,p.spin]
    phase[1:]=phase[0]+np.cumsum((spin[:-1]+spin[1:])*.01,axis=0)
    c[:,9:11]=np.linalg.lstsq(p.B,phase,rcond=None)[0];c[0,9:11]=phase[0]
    # Start force variables without pretending to solve time-coupled equilibrium.
    c[:,11:]=0.
    c[:,[13,16,19,22]]=.25
    c=np.clip(c,p.lower+1e-10,p.upper-1e-10)
    residual,records=p.evaluate(c,save=True)
    a.output_dir.mkdir(parents=True);np.savez_compressed(a.output_dir/'coefficients.npz',coefficients=c)
    proof=dict(kind='snapshot_informed_time_coupled_initialization_only',source_sha256=base.sha(__file__),
        planner_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference.py')),
        posture_fit_sha256=base.sha(a.postures),config_sha256=base.sha(a.config),
        model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),
        coefficients_sha256=base.sha(a.output_dir/'coefficients.npz'),initial_objective_sum_squares=float(np.sum(residual**2)),
        initial_qpos_max_delta_vs_source=float(abs(records['qpos'][0]-source['qpos_mujoco'][0]).max()),
        first_three_shape_coefficients_original=True,spin_seed='longitudinal rolling at lower geometric rim then integrated and B-spline fitted',
        dynamics_or_contact_feasibility_proven=False,task_success_claimed=False)
    (a.output_dir/'analysis.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','postures','config','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

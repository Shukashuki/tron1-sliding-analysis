"""400Hz resampling of exact-support states,retaining rigid-rim branch changes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import generate_camber_reference as base
from audit_hybrid_contact_dense import AnalyticProblem
from hybrid_contact_reference import rim_points,summary


class DenseSupportProblem(AnalyticProblem):
    def __init__(self,model,source,config,force_samples,dt=.0025):
        super().__init__(model,source,config,dt=dt)
        self.force_source=np.asarray(force_samples,dtype=float)
        if self.force_source.shape!=(len(self.original_time),12):raise ValueError('Wrong point-force samples')

    def kinematics(self,c,query):
        value,q,v,acc=super().kinematics(c,query);t=np.atleast_1d(query)
        value[:,11:]=np.stack([np.interp(t,self.original_time,self.force_source[:,j]) for j in range(12)],axis=-1)
        body=self.wheel_bodies[self.rolling];geom=self.geoms[self.rolling]
        radius,halfwidth=self.model.geom_size[geom,:2]
        for i in range(len(q)):
            self.data.qpos[:]=q[i];self.mj.mj_kinematics(self.model,self.data)
            axle=self.data.xmat[body].reshape(3,3)[:,1]
            lowest=self.data.geom_xpos[geom,2]-radius*np.sqrt(max(0.,1-axle[2]**2))-halfwidth*abs(axle[2])
            q[i,2]-=lowest;value[i,2]=q[i,2]
            self.data.qpos[:]=q[i];self.mj.mj_kinematics(self.model,self.data);self.mj.mj_comPos(self.model,self.data)
            points=rim_points(self.data.geom_xpos[geom],axle,radius,halfwidth);point=points[int(np.argmin(points[:,2]))]
            self.mj.mj_jac(self.model,self.data,self.jp,self.jr,point,body)
            v[i,2]=0.;v[i,2]=-(self.jp@v[i])[2]
        acc[:,2]=np.gradient(v[:,2],t,edge_order=2)
        return value,q,v,acc


def export_candidate(p,c,source,path):
    mj,m,d=p.mj,p.model,p.data
    values,q,v,acc=p.kinematics(c,source['time_s'])
    body_ids=[m.body(str(n)).id for n in source['body_names']]
    joints=[m.joint(str(n)).id for n in source['joint_names']]
    data={k:a.copy() for k,a in source.items()}
    output={k:[] for k in ['body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
    for pose,velocity in zip(q,v):
        d.qpos[:]=pose;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
        output['body_pos_w'].append(d.xpos[body_ids].copy());output['body_quat_w'].append(d.xquat[body_ids].copy())
        linear,angular=[],[]
        for body in body_ids:
            mj.mj_jacBodyCom(m,d,p.jp,p.jr,body);linear.append(p.jp@velocity);angular.append(p.jr@velocity)
        output['body_lin_vel_w'].append(linear);output['body_ang_vel_w'].append(angular)
        assert d.time==0.
    data.update({k:np.asarray(a) for k,a in output.items()})
    data.update(qpos_mujoco=q,qvel_mujoco=v,qacc_mujoco=acc,joint_pos=q[:,m.jnt_qposadr[joints]],joint_vel=v[:,m.jnt_dofadr[joints]],
        planned_rim_forces_w_n=values[:,11:].reshape(-1,4,3)*p.weight,
        planned_forces_are_native_sensor_measurements=np.asarray(False),
        source_original_car_reference_sha256=np.asarray(base.SOURCE_SHA),
        reference_type=np.asarray('time_coupled_hybrid_reference_candidate_not_native_validated'))
    np.savez_compressed(path,**data)


def run(a):
    import mujoco
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    fit=json.loads((a.fit/'analysis.json').read_text());cfg=fit['config']
    assert fit['kind']=='time_coupled_hybrid_reference_exact_support_local_forces'
    if base.sha(a.model)!=fit['model_sha256'] or base.sha(a.reference)!=fit['reference_sha256']:raise ValueError('Wrong model/motion')
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    with np.load(a.fit/'planned_state.npz',allow_pickle=False) as f:c=f['coefficients'].copy();forces=f['force_samples_over_mg'].copy()
    p=DenseSupportProblem(mujoco.MjModel.from_xml_path(str(a.model)),source,cfg,forces)
    residual,records=p.evaluate(c,save=True)
    a.output_dir.mkdir(parents=True);np.savez_compressed(a.output_dir/'dense_state.npz',**records)
    export_candidate(p,c,source,a.output_dir/'candidate_motion.npz')
    diagnostics=summary(p,records,residual)
    rolling=records['axes'][:,p.rolling,2];switches=np.flatnonzero(rolling[:-1]*rolling[1:]<0)
    proof=dict(kind='analytic400Hz_exact_support_hybrid_reference_audit',source_sha256=base.sha(__file__),
        support_planner_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference_support.py')),
        parent_planner_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference.py')),
        fit_sha256=base.sha(a.fit/'analysis.json'),state_sha256=base.sha(a.fit/'planned_state.npz'),
        model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),samples=len(p.time),dt_s=.0025,
        diagnostics=diagnostics,candidate_motion_sha256=base.sha(a.output_dir/'candidate_motion.npz'),
        model_support_rim_switch_intervals_start_s=p.time[switches].tolist(),
        interpolation='C2 Hermite root XY/yaw,cubic B-spline joint/RP;support-derived height;piecewise-linear model force samples',
        support_height_acceleration='sampled derivative of analytic normal-velocity constraint;rigid-rim cusp retained',
        full_reference_streams_exported=True,certified_for_policy_training=False,
        physics_steps=0,policy_updates=0,native_contact_stability_proven=False,task_success_claimed=False)
    (a.output_dir/'analysis.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n');print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','fit','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

"""Resample the full hybrid reference at400Hz with analytic shape derivatives.

Car root XY/yaw use local C2 quintic Hermite interpolation matching original
50Hz positions,velocities and sampled accelerations. This interpolation and
the model forces remain explicit assumptions,not a native rollout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BPoly,BSpline

import generate_camber_reference as base
from hybrid_contact_reference import Problem,root_quaternion,summary


class AnalyticProblem(Problem):
    def __init__(self,model,source,config,dt=.0025):
        super().__init__(model,source,config)
        original_time=self.time.copy();self.original_time=original_time
        yaw=np.unwrap(2*np.arctan2(source['qpos_mujoco'][:,6],source['qpos_mujoco'][:,3]))
        position=np.c_[source['qpos_mujoco'][:,:2],yaw]
        velocity=source['qvel_mujoco'][:,[0,1,5]]
        acceleration=np.gradient(velocity,original_time,axis=0,edge_order=2)
        self.root_curve=BPoly.from_derivatives(original_time,[[position[i],velocity[i],acceleration[i]] for i in range(len(original_time))])
        count=config['coefficients'];knots=np.r_[np.repeat(original_time[0],4),np.linspace(original_time[0],original_time[-1],count-2)[1:-1],np.repeat(original_time[-1],4)]
        self.spline=BSpline(knots,np.eye(count),3)
        self.time=np.linspace(original_time[0],original_time[-1],int(round((original_time[-1]-original_time[0])/dt))+1)
        assert np.allclose(np.diff(self.time),dt,atol=1e-12,rtol=0)
        self.B=self.spline(self.time)
        self.slide=(self.time>=config['slide_interval_s'][0]-1e-9)&(self.time<=config['slide_interval_s'][1]+1e-9)

    def kinematics(self,c,query):
        t=np.atleast_1d(query);value=self.spline(t)@c;dv=self.spline.derivative(1)(t)@c;da=self.spline.derivative(2)(t)@c
        root=self.root_curve(t);root_v=self.root_curve.derivative(1)(t);root_a=self.root_curve.derivative(2)(t)
        q=np.repeat(self.source['qpos_mujoco'][0,None],len(t),axis=0);q[:,:2]=root[:,:2];q[:,2]=value[:,2]
        for i in range(len(t)):q[i,3:7]=root_quaternion(root[i,2],value[i,0],value[i,1])
        q[:,self.joint_q]=value[:,3:11]
        v=np.zeros((len(t),self.model.nv));a=np.zeros_like(v)
        v[:,:2]=root_v[:,:2];v[:,2]=dv[:,2];a[:,:2]=root_a[:,:2];a[:,2]=da[:,2]
        roll,pitch=value[:,0],value[:,1];rd,pd=dv[:,0],dv[:,1];rdd,pdd=da[:,0],da[:,1];yd,ydd=root_v[:,2],root_a[:,2]
        sr,cr,sp,cp=np.sin(roll),np.cos(roll),np.sin(pitch),np.cos(pitch)
        v[:,3]=rd-yd*sp
        v[:,4]=pd*cr+yd*sr*cp
        v[:,5]=-pd*sr+yd*cr*cp
        a[:,3]=rdd-ydd*sp-yd*cp*pd
        a[:,4]=pdd*cr-pd*sr*rd+ydd*sr*cp+yd*cr*rd*cp-yd*sr*sp*pd
        a[:,5]=-pdd*sr-pd*cr*rd+ydd*cr*cp-yd*sr*rd*cp-yd*cr*sp*pd
        v[:,self.joint_dof]=dv[:,3:11];a[:,self.joint_dof]=da[:,3:11]
        return value,q,v,a

    def poses(self,c):return self.kinematics(c,self.time)


def run(a):
    import mujoco
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    fit=json.loads((a.fit/'analysis.json').read_text());cfg=fit['config']
    if base.sha(a.model)!=fit['model_sha256'] or base.sha(a.reference)!=fit['reference_sha256']:raise ValueError('Wrong source/model')
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    with np.load(a.fit/'planned_state.npz',allow_pickle=False) as f:c=f['coefficients'].copy()
    p=AnalyticProblem(mujoco.MjModel.from_xml_path(str(a.model)),source,cfg)
    residual,records=p.evaluate(c,save=True);result=summary(p,records,residual)
    node,_,node_v,_=p.kinematics(c,source['time_s'])
    np.testing.assert_allclose(p.root_curve(source['time_s'])[:,:2],source['qpos_mujoco'][:,:2],atol=1e-12,rtol=0)
    np.testing.assert_allclose(p.root_curve.derivative(1)(source['time_s'])[:,[0,1,2]],source['qvel_mujoco'][:,[0,1,5]],atol=1e-11,rtol=0)
    a.output_dir.mkdir(parents=True);np.savez_compressed(a.output_dir/'dense_state.npz',**records)
    proof=dict(kind='analytic400Hz_hybrid_reference_model_audit',source_sha256=base.sha(__file__),
        planning_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference.py')),
        fit_sha256=base.sha(a.fit/'analysis.json'),state_sha256=base.sha(a.fit/'planned_state.npz'),
        model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),samples=len(p.time),dt_s=.0025,
        interpolation='C2 quintic Hermite root XY/yaw;original node values,velocities and sampled accelerations;cubic B-spline shape/phase/force',
        diagnostics=result,physics_steps=0,policy_updates=0,native_contact_stability_proven=False,task_success_claimed=False)
    (a.output_dir/'analysis.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n');print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','fit','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

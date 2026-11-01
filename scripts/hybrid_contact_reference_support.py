"""Exact geometric support height and independent per-state rim force variables.

Keeps the same reference-planning objective as the parent. The rolling-wheel
lowest cylinder point fixes reference root height geometrically;no absolute
height reward is introduced. Per-state force variables allow rim load transfer
without imposing the state spline's time smoothness on contact forces.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack,kron,eye,csr_matrix

import generate_camber_reference as base
from hybrid_contact_reference import Problem,RESIDUAL_DIM,summary


class SupportProblem(Problem):
    def __init__(self,model,source,config):
        super().__init__(model,source,config)
        self.free[:,2]=False;self.free[:,11:]=False
        self.state_size=int(self.free.sum());self.force_samples=(self.B@self.C)[:,11:].copy()

    def poses(self,c):
        value,q,_,_=super().poses(c)
        body=self.wheel_bodies[self.rolling];geom=self.geoms[self.rolling]
        radius,halfwidth=self.model.geom_size[geom,:2]
        for i in range(len(q)):
            self.data.qpos[:]=q[i];self.mj.mj_kinematics(self.model,self.data)
            axle=self.data.xmat[body].reshape(3,3)[:,1]
            lowest=self.data.geom_xpos[geom,2]-radius*np.sqrt(max(0.,1-axle[2]**2))-halfwidth*abs(axle[2])
            q[i,2]-=lowest
        value[:,2]=q[:,2];value[:,11:]=self.force_samples
        v=base.differentiate(self.model,q,.02);acc=np.gradient(v,self.time,axis=0,edge_order=2)
        return value,q,v,acc

    def vector(self,c=None,forces=None):
        return np.r_[(self.C if c is None else c)[self.free],(self.force_samples if forces is None else forces).ravel()]

    def decode(self,x):
        c=self.C.copy();c[self.free]=x[:self.state_size]
        self.force_samples=x[self.state_size:].reshape(len(self.time),12).copy()
        return c

    def bounds(self):
        lo=np.broadcast_to(np.tile([-1.4,-1.4,0.],4),self.force_samples.shape)
        hi=np.broadcast_to(np.tile([1.4,1.4,2.],4),self.force_samples.shape)
        return np.r_[self.lower[self.free],lo.ravel()],np.r_[self.upper[self.free],hi.ravel()]

    def residual(self,x):
        result=self.evaluate(self.decode(x)).ravel();self.calls+=1;cost=float(result@result)
        if cost<self.best_cost:self.best_cost=cost;self.best_x=x.copy()
        now=time.monotonic()
        if now-self.last_log>30.:
            print(json.dumps(dict(residual_calls=self.calls,best_cost=self.best_cost,current_cost=cost)),flush=True);self.last_log=now
        return result

    def sparsity(self):
        state=super().sparsity()
        forces=kron(eye(len(self.time),format='csr'),csr_matrix(np.ones((RESIDUAL_DIM,12))),format='csr')
        return hstack([state,forces],format='csr')


def run(a):
    import mujoco
    from scipy.optimize import least_squares
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    cfg=json.loads(a.config.read_text());prior=json.loads((a.initial_fit/'analysis.json').read_text())
    assert base.sha(a.model)==prior['model_sha256']==base.MODEL_SHA and base.sha(a.reference)==prior['reference_sha256']==base.SOURCE_SHA
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    p=SupportProblem(mujoco.MjModel.from_xml_path(str(a.model)),source,cfg)
    with np.load(a.initial_fit/'planned_state.npz',allow_pickle=False) as f:
        c=f['coefficients'].copy();force=f['force_w_n']/p.weight
    p.C[p.free]=c[p.free];p.force_samples=force.reshape(len(p.time),12).copy()
    lo,hi=p.bounds();x=np.clip(p.vector(),lo+1e-10,hi-1e-10)
    a.output_dir.mkdir(parents=True)
    initial_r,initial=p.evaluate(p.decode(x),save=True);initial['force_samples_over_mg']=p.force_samples.copy()
    np.savez_compressed(a.output_dir/'initial_state.npz',**initial)
    initial_summary=summary(p,initial,initial_r)
    result=least_squares(p.residual,x,bounds=(lo,hi),jac_sparsity=p.sparsity(),method='trf',tr_solver='lsmr',
        x_scale='jac',max_nfev=cfg['max_nfev'],ftol=1e-6,xtol=1e-7,gtol=1e-6,verbose=1)
    c=p.decode(p.best_x);r,records=p.evaluate(c,save=True);records['force_samples_over_mg']=p.force_samples.copy()
    np.savez_compressed(a.output_dir/'planned_state.npz',**records)
    report=dict(kind='time_coupled_hybrid_reference_exact_support_local_forces',config=cfg,
        initial=initial_summary,final=summary(p,records,r),
        solver=dict(success=bool(result.success),message=str(result.message),nfev=int(result.nfev),residual_calls=p.calls,
            first_order_optimality=float(result.optimality),retained_best_may_be_FD_probe=True),
        source_sha256=base.sha(__file__),parent_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference.py')),
        config_sha256=base.sha(a.config),initial_fit_sha256=base.sha(a.initial_fit/'analysis.json'),
        model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),
        force_representation='force_samples_over_mg:349x12;stored coefficients force columns are unused',
        root_height_representation='derived from rolling-wheel lowest cylinder surface;coefficient height column unused',
        physics_steps=0,policy_updates=0,certified_reference_for_training=False,task_success_claimed=False,
        limitations=prior['limitations']+['Cylinder support height can have a derivative cusp at a camber sign change;dense validation must retain it.'])
    (a.output_dir/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['final']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','initial-fit','config','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

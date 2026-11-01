"""Time-coupled rolling/sliding reference search with unactuated RNE dynamics.

The original car root XY/yaw samples stay fixed. Cubic B-splines parameterize
root roll/pitch/height,six leg joints,two wheel phases,and four rim-point forces.
Velocities/accelerations come from the complete pose trajectory,not frozen
postures. This is a conditional planning model,not a native support rollout.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix

import generate_camber_reference as base
from spatial_wheel_posture import root_quaternion

LEGS=('abad_L_Joint','abad_R_Joint','hip_L_Joint','hip_R_Joint','knee_L_Joint','knee_R_Joint')
DIM=23
RESIDUAL_DIM=57


def spline_basis(t,count):
    knots=np.r_[np.repeat(t[0],4),np.linspace(t[0],t[-1],count-2)[1:-1],np.repeat(t[-1],4)]
    result=BSpline(knots,np.eye(count),3)(t)
    if result.shape!=(len(t),count):raise ValueError('Wrong spline basis')
    return result


def dc_bounds(velocity,effort,speed):
    # Same saturation=effort DCMotor envelope as the pinned native evaluator.
    curve=np.clip(velocity,-2*speed,2*speed)
    return np.maximum(effort*(-1-curve/speed),-effort),np.minimum(effort*(1-curve/speed),effort)


def cylinder_support(center,axis,radius,halfwidth):
    radial=np.array([0.,0.,1.])-axis[2]*axis
    point=center-radius*radial/max(np.linalg.norm(radial),1e-10)
    return point-halfwidth*np.sign(axis[2])*axis


def rim_points(center,axis,radius,halfwidth):
    radial=np.array([0.,0.,1.])-axis[2]*axis
    line_center=center-radius*radial/max(np.linalg.norm(radial),1e-10)
    return np.asarray([line_center-halfwidth*axis,line_center+halfwidth*axis])


class Problem:
    def __init__(self,model,source,config):
        import mujoco
        self.mj=mujoco;self.model=model;self.data=mujoco.MjData(model);self.source=source;self.config=config
        self.time=source['time_s'];self.B=spline_basis(self.time,config['coefficients']);self.count=self.B.shape[1]
        self.weight=float(model.body_mass.sum())*9.81
        self.legs=[model.joint(n).qposadr.item() for n in LEGS]
        self.leg_dof=[model.joint(n).dofadr.item() for n in LEGS]
        self.wheel_bodies=[model.body(n).id for n in base.WHEELS]
        self.spin=[model.joint(n).qposadr.item() for n in base.SPINS]
        self.spin_dof=[model.joint(n).dofadr.item() for n in base.SPINS]
        self.joint_q=self.legs+self.spin;self.joint_dof=self.leg_dof+self.spin_dof
        self.geoms=[model.geom('wheel_'+s+'_collision').id for s in ('L','R')]
        self.other_geoms=[i for i in range(model.ngeom) if model.geom_bodyid[i] and model.geom_contype[i] and model.geom_conaffinity[i] and i not in self.geoms]
        self.opposite=[[i for i in self.other_geoms+self.geoms if '_'+s+'_' in model.body(int(model.geom_bodyid[i])).name] for s in ('L','R')]
        self.yaw=2*np.arctan2(source['qpos_mujoco'][:,6],source['qpos_mujoco'][:,3])
        self.jp=np.empty((3,model.nv));self.jr=np.empty_like(self.jp)
        self.mu=config['mu'];self.rolling=config['rolling_wheel'];self.sliding=1-self.rolling
        self.slide=(self.time>=config['slide_interval_s'][0]-1e-9)&(self.time<=config['slide_interval_s'][1]+1e-9)
        self.effort=np.r_[np.full(6,80.),np.full(2,12.)];self.speed=np.r_[np.full(6,15.),np.full(2,100.)]
        self.C=np.zeros((self.count,DIM));self.C[:,2]=.75
        self.C[:,3:9]=source['qpos_mujoco'][0,self.legs]
        self.C[:,9:11]=np.linalg.lstsq(self.B,source['qpos_mujoco'][:,self.spin],rcond=None)[0]
        self.C[:,[13,16,19,22]]=.25
        self.lower=np.broadcast_to(np.r_[np.deg2rad([-30.,-30.]),.25,
            [model.joint(n).range[0] for n in LEGS],[-1000.,-1000.],np.tile([-1.4,-1.4,0.],4)],self.C.shape).copy()
        self.upper=np.broadcast_to(np.r_[np.deg2rad([30.,30.]),.85,
            [model.joint(n).range[1] for n in LEGS],[1000.,1000.],np.tile([1.4,1.4,2.],4)],self.C.shape).copy()
        self.free=np.ones(self.C.shape,dtype=bool);self.free[:3,:9]=False;self.free[0,9:11]=False
        self.C[0,9:11]=source['qpos_mujoco'][0,self.spin]
        self.calls=0;self.best_cost=float('inf');self.best_x=None;self.last_log=0.

    def unpack(self,x):
        c=self.C.copy();c[self.free]=x;return c

    def poses(self,c):
        values=self.B@c;q=self.source['qpos_mujoco'].copy();q[:,2]=values[:,2]
        for i in range(len(q)):q[i,3:7]=root_quaternion(self.yaw[i],values[i,0],values[i,1])
        q[:,self.joint_q]=values[:,3:11]
        v=base.differentiate(self.model,q,.02);acc=np.gradient(v,self.time,axis=0,edge_order=2)
        return values,q,v,acc

    def evaluate(self,c,save=False):
        mj,m,d=self.mj,self.model,self.data;values,q,v,acc=self.poses(c);forces=values[:,11:].reshape(-1,4,3)*self.weight
        rows=[];records={k:[] for k in ('points','axes','slip','required','contact','torque','nonwheel_clearance','opposite_clearance')}
        for i in range(len(q)):
            d.qpos[:]=q[i];d.qvel[:]=v[i];d.qacc[:]=acc[i]
            mj.mj_kinematics(m,d);mj.mj_comPos(m,d);mj.mj_comVel(m,d)
            required=np.empty(m.nv);mj.mj_rne(m,d,1,required)
            points=[];axes=[];slip=[];contact=np.zeros(m.nv)
            for wheel,(body,geom) in enumerate(zip(self.wheel_bodies,self.geoms)):
                axis=d.xmat[body].reshape(3,3)[:,1].copy();axes.append(axis)
                for side,point in enumerate(rim_points(d.geom_xpos[geom],axis,*m.geom_size[geom,:2])):
                    mj.mj_jac(m,d,self.jp,self.jr,point,body)
                    contact+=self.jp.T@forces[i,2*wheel+side];slip.append(self.jp@v[i]);points.append(point)
            points,axes,slip=np.asarray(points),np.asarray(axes),np.asarray(slip)
            torque=(required-contact)[self.joint_dof];lo,hi=dc_bounds(v[i,self.joint_dof],self.effort,self.speed)
            nonwheel=[]
            for geom in self.other_geoms:
                mat,size=d.geom_xmat[geom].reshape(3,3),m.geom_size[geom]
                if m.geom_type[geom]==mj.mjtGeom.mjGEOM_BOX:z=d.geom_xpos[geom,2]-abs(mat[2])@size
                else:z=cylinder_support(d.geom_xpos[geom],mat[:,2],*size[:2])[2]
                nonwheel.append(z)
            clearance=min(nonwheel)
            opposite=min(mj.mj_geomDistance(m,d,left,right,1.,None) for left in self.opposite[0] for right in self.opposite[1])
            base_error=(required-contact)[:6]/np.r_[np.full(3,self.weight),np.full(3,self.weight*.5)]
            ri=slice(2*self.rolling,2*self.rolling+2);si=slice(2*self.sliding,2*self.sliding+2)
            normal_fraction=forces[i,:,2]/self.weight
            loaded_weight=normal_fraction/(normal_fraction+.05)
            lateral_speed=slip[si,:2];speed=np.linalg.norm(lateral_speed,axis=-1)
            right_mode=(forces[i,si,:2]+self.mu*forces[i,si,2,None]*lateral_speed/np.maximum(speed[:,None],1e-6))/self.weight if self.slide[i] else loaded_weight[si,None]*lateral_speed/.1
            cone=np.maximum(0.,np.linalg.norm(forces[i,:,:2],axis=-1)-self.mu*forces[i,:,2])/self.weight
            motor=(np.maximum(0.,lo-torque)+np.maximum(0.,torque-hi))/self.effort
            target=q[i,self.legs]+(torque[:6]+10.*v[i,self.leg_dof])/500.
            reference_zero=self.source['qpos_mujoco'][0,self.legs]
            target_violation=np.maximum(0.,abs(target-reference_zero)-1.)
            smooth=np.r_[acc[i,2]/9.81,acc[i,3:6]/10.,acc[i,self.leg_dof]/10.]*.01
            row=np.r_[base_error,np.maximum(0.,-points[:,2])/.01,normal_fraction*points[:,2]/.01,
                (loaded_weight[ri,None]*slip[ri,:2]/.1).ravel(),right_mode.ravel(),cone,motor,target_violation,
                max(0.,.005-clearance)/.01,max(0.,.005-opposite)/.01,
                np.maximum(0.,.5-np.linalg.norm(axes[:,:2],axis=-1))/.1,
                loaded_weight[si]*np.maximum(0.,self.config['sliding_speed_min_m_s']-speed)/.1 if self.slide[i] else np.zeros(2),
                max(0.,self.config['rolling_normal_min_n']/self.weight-normal_fraction[ri].sum())/.05,smooth]
            assert row.shape==(RESIDUAL_DIM,),row.shape
            rows.append(row)
            if save:
                for key,value in [('points',points),('axes',axes),('slip',slip),('required',required),('contact',contact),('torque',torque),
                        ('nonwheel_clearance',clearance),('opposite_clearance',opposite)]:records[key].append(np.asarray(value).copy())
        result=np.asarray(rows)
        if not save:return result
        return result,dict(time_s=self.time.copy(),coefficients=c,values=values,qpos=q,qvel=v,qacc=acc,force_w_n=forces,
            **{k:np.asarray(a) for k,a in records.items()})

    def residual(self,x):
        result=self.evaluate(self.unpack(x)).ravel();self.calls+=1;cost=float(result@result)
        if cost<self.best_cost:self.best_cost=cost;self.best_x=x.copy()
        now=time.monotonic()
        if now-self.last_log>30.:
            print(json.dumps(dict(residual_calls=self.calls,best_cost=self.best_cost,current_cost=cost)),flush=True);self.last_log=now
        return result

    def sparsity(self):
        # Q velocities use neighboring poses;edge qacc stencils reach3 frames.
        pattern=lil_matrix((len(self.time)*RESIDUAL_DIM,self.C.size),dtype=int)
        for i in range(len(self.time)):
            support=np.flatnonzero(np.any(abs(self.B[max(0,i-3):min(len(self.time),i+4)])>0,axis=0))
            cols=np.ravel(support[:,None]*DIM+np.arange(DIM))
            pattern[i*RESIDUAL_DIM:(i+1)*RESIDUAL_DIM,cols]=1
        return pattern.tocsr()[:,self.free.ravel()]


def summary(problem,records,residual):
    forces=records['force_w_n'];slip=records['slip'];required=records['required'];contact=records['contact']
    base_error=(required-contact)[:,:6]/np.r_[np.full(3,problem.weight),np.full(3,problem.weight*.5)]
    lo,hi=dc_bounds(records['qvel'][:,problem.joint_dof],problem.effort,problem.speed)
    motor=np.maximum(0.,lo-records['torque'])+np.maximum(0.,records['torque']-hi)
    core=(problem.time>=1.44-1e-9)&(problem.time<=4.+1e-9)
    ri=slice(2*problem.rolling,2*problem.rolling+2)
    normal=forces[:,ri,2];weighted_slip=np.sqrt(np.sum(normal*np.sum(slip[:,ri,:2]**2,axis=-1),axis=-1)/np.maximum(normal.sum(-1),1e-9))
    loaded=forces[:,:,2]>5.
    target=records['qpos'][:,problem.legs]+(records['torque'][:,:6]+10.*records['qvel'][:,problem.leg_dof])/500.
    target_violation=np.maximum(0.,abs(target-problem.source['qpos_mujoco'][0,problem.legs])-1.)
    return dict(objective_sum_squares=float(np.sum(residual**2)),
        core_base_wrench_norm_median_p95_max=np.quantile(np.linalg.norm(base_error[core],axis=-1),[.5,.95,1.]).tolist(),
        full_max_floor_penetration_m=float(max(0.,-records['points'][:,:,2].min())),
        loaded_rim_point_gap_m_max=None if not loaded.any() else float(abs(records['points'][:,:,2][loaded]).max()),
        core_model_force_weighted_rolling_tangent_RMS_m_s_median_p95_max=np.quantile(weighted_slip[core],[.5,.95,1.]).tolist(),
        full_min_model_rolling_normal_n=float(normal.sum(-1).min()),
        full_max_DC_envelope_effort_violation_nm=float(motor.max()),
        full_max_raw_PD_position_target_violation_rad=float(target_violation.max()),
        minimum_nonwheel_clearance_m=float(records['nonwheel_clearance'].min()),minimum_opposite_leg_clearance_m=float(records['opposite_clearance'].min()),
        original_root_XY_yaw_and_speed_fixed=True,native_support_or_task_success_proven=False)


def run(a):
    import mujoco
    from scipy.optimize import least_squares
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if base.sha(a.reference)!=base.SOURCE_SHA or base.sha(a.model)!=base.MODEL_SHA:raise ValueError('Wrong motion/model')
    cfg=json.loads(a.config.read_text());assert cfg['kind']=='time_coupled_hybrid_point_contact_reference_search'
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    p=Problem(mujoco.MjModel.from_xml_path(str(a.model)),source,cfg)
    if a.initial_coefficients is not None:
        with np.load(a.initial_coefficients,allow_pickle=False) as f:initial=f['coefficients']
        assert initial.shape==p.C.shape
        p.C[p.free]=initial[p.free]
    p.C=np.clip(p.C,p.lower+1e-10,p.upper-1e-10)
    a.output_dir.mkdir(parents=True)
    initial_residual,initial_records=p.evaluate(p.C,save=True)
    initial_summary=summary(p,initial_records,initial_residual)
    np.savez_compressed(a.output_dir/'initial_state.npz',**initial_records)
    if a.evaluate_only:
        chosen=p.C;solver=None
    else:
        result=least_squares(p.residual,p.C[p.free],bounds=(p.lower[p.free],p.upper[p.free]),
            jac_sparsity=p.sparsity(),method='trf',tr_solver='lsmr',x_scale='jac',
            max_nfev=cfg['max_nfev'],ftol=1e-6,xtol=1e-7,gtol=1e-6,verbose=1)
        chosen=p.unpack(p.best_x)
        solver=dict(success=bool(result.success),message=str(result.message),nfev=int(result.nfev),
            residual_calls=p.calls,cost_final=float(result.cost),retained_best_may_be_FD_probe=True)
    final_residual,records=p.evaluate(chosen,save=True)
    np.savez_compressed(a.output_dir/'planned_state.npz',**records)
    report=dict(kind=cfg['kind'],config=cfg,initial=initial_summary,final=summary(p,records,final_residual),solver=solver,
        source_sha256=base.sha(__file__),config_sha256=base.sha(a.config),model_sha256=base.sha(a.model),reference_sha256=base.sha(a.reference),
        initial_coefficients_sha256=None if a.initial_coefficients is None else base.sha(a.initial_coefficients),
        physics_steps=0,policy_updates=0,full_time_coupled_state_produced=True,certified_reference_for_training=False,task_success_claimed=False,
        limitations=['Two rim candidates per cylinder and unilateral force/gap complementarity approximate the set-valued support line;not native contact equivalence.',
            'A prescribed rolling/sliding mode schedule is a planning assumption;forces are model variables,not sensor measurements.',
            'Finite-difference derivatives and optimization are at50Hz;400Hz contact/motor validation and native rollout remain required.',
            'All objective penalties are reference-planning costs;no PPO reward or original nominal acceptance gate is changed.',
            'Native measured upward force>5N at every2.5ms sample is not proven by model force bounds.'])
    (a.output_dir/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['final']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','config','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--initial-coefficients',type=Path);p.add_argument('--evaluate-only',action='store_true');run(p.parse_args())

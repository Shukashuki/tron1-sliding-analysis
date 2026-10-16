"""Full TRON1 MuJoCo predictor, affine TVLQR and sampling-based NMPC.

Predictive rollouts are distinct from the simulated plant. No direct root force,
root pose correction, tire no-slip constraint, or learned-policy warm start.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_pinned_evaluator():
    source = ROOT / 'upstream/tron1-learning-from-human'
    sys.path.insert(0, str(source / 'scripts'))
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_classical_pinned_evaluator', source / 'scripts/eval_tracking_mujoco.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def affine_riccati(A, B, defect, Q, R, terminal):
    """Finite-horizon e+=A e+B du+d; u=ubar+k-Ke, including affine d."""
    count, nx, nu = len(A), A.shape[1], B.shape[2]
    K, feed = np.zeros((count, nu, nx)), np.zeros((count, nu))
    values = np.zeros((count+1,nx,nx)); values[-1] = terminal
    gradient = np.zeros(nx)
    for k in range(count-1,-1,-1):
        P = values[k+1]
        H = R+B[k].T@P@B[k]
        G = B[k].T@P@A[k]
        h = B[k].T@(P@defect[k]+gradient)
        K[k] = np.linalg.solve(H,G)
        feed[k] = -np.linalg.solve(H,h)
        values[k] = Q+A[k].T@P@A[k]-G.T@K[k]
        values[k] = .5*(values[k]+values[k].T)
        gradient = A[k].T@(P@defect[k]+gradient)+G.T@feed[k]
    return K, feed, values


class Predictor:
    def __init__(self, model_path, motion_path, contract, settings):
        self.api = load_pinned_evaluator()
        self.contract = self.api.validate_contract(contract)
        if contract.get('domain_parameters') is not None:
            raise ValueError('Pilot requires nominal actuator parameters')
        self.model,_ = self.api.load_model(model_path,no_visual_mesh=True)
        self.api.configure_model(self.model,contract)
        self.model.geom_friction[:,0] = settings['nominal_prediction_mu']
        self.reference = self.api.MotionReference(motion_path,contract)
        self.data = mujoco.MjData(self.model)
        self.measure = self.api.MujocoState(self.model,self.data,self.reference)
        self.qids,self.vids,self.aids = self.measure.qids,self.measure.vids,self.measure.aids
        self.nv = self.model.nv
        self.n = self.reference.frames
        self.dt = float(contract['physics_dt'])*int(contract['decimation'])
        if not np.isclose(self.dt,settings['policy_dt_s']) or not np.isclose(self.model.opt.timestep,settings['physics_dt_s']):
            raise ValueError('Controller/physics clock mismatch')
        self.q0 = self.reference.q0
        self.effort = np.r_[np.full(6,80.),np.full(2,12.)]
        self.no_load = np.r_[np.full(6,15.),np.full(2,100.)]
        required = {'leg_kp':500.,'leg_kd':10.,'wheel_torque_scale_nm':12.,'leg_position_scale':1.,
                    'leg_saturation_effort_nm':80.,'wheel_saturation_effort_nm':12.,
                    'leg_torque_limit_nm':80.,'wheel_torque_limit_nm':12.,
                    'leg_motor_velocity_limit_rad_s':15.,'wheel_motor_velocity_limit_rad_s':100.}
        if any(not np.isclose(contract[k],v) for k,v in required.items()):
            raise ValueError('Unexpected actuator contract; refusing approximate mapping')
        self.targets,self.nominal_action = self.make_targets()
        p,v = settings['state_position_weights'],settings['state_velocity_weights']
        q = np.r_[[p['xy']]*2,p['z'],[p['roll_pitch']]*2,p['yaw'],np.zeros(8),[v['linear']]*3,[v['angular']]*3,np.zeros(8)]
        for name,qid,vid in zip(self.reference.action_names,self.qids,self.vids):
            q[vid] = p['wheel_angle'] if name.startswith('wheel_') else p['leg']
            q[self.nv+vid] = v['wheel'] if name.startswith('wheel_') else v['leg']
        self.Q = np.diag(q)
        self.R = np.diag([settings['action_weights']['leg']]*6+[settings['action_weights']['wheel']]*2)
        self.settings = settings

    def torque(self, action, qpos, qvel):
        a = np.clip(action,-1.,1.)
        q,dq = qpos[self.qids],qvel[self.vids]
        tau = np.r_[500.*(self.q0[:6]+a[:6]-q[:6])-10.*dq[:6],12.*a[6:]]
        clipped = np.clip(dq,-2*self.no_load,2*self.no_load)
        lower = np.maximum(self.effort*(-1-clipped/self.no_load),-self.effort)
        upper = np.minimum(self.effort*(1-clipped/self.no_load),self.effort)
        return np.clip(tau,lower,upper)

    def make_targets(self):
        states,controls = [],[]
        d,m = self.data,self.model
        for k in range(self.n):
            t = self.reference.sample(k)
            d.qpos[:7] = np.r_[t['root_pos'],t['root_quat_wxyz']]
            d.qpos[self.qids] = t['joint_pos']
            d.qvel[:] = 0
            d.qvel[self.vids] = t['joint_vel']
            mujoco.mj_forward(m,d)
            jp,jr = self.measure.root_jacobians()
            jac = np.vstack([jp,jr])
            d.qvel[:6] = np.linalg.solve(jac[:,:6],np.r_[t['root_com_lin_vel_w'],t['root_ang_vel_w']]-jac@d.qvel)
            mujoco.mj_forward(m,d)
            # Bias-force joint feedforward only. Any unactuated reference
            # infeasibility is retained in the affine transition defect.
            bias = d.qfrc_bias[self.vids]
            a = np.r_[t['joint_pos'][:6]-self.q0[:6]+(10*t['joint_vel'][:6]+bias[:6])/500.,bias[6:]/12.]
            states.append(np.r_[d.qpos.copy(),d.qvel.copy()]);controls.append(np.clip(a,-1.,1.))
        return np.array(states),np.array(controls)

    def error(self,target,state):
        diff = np.empty(self.nv)
        mujoco.mj_differentiatePos(self.model,diff,1.,target[:self.model.nq],state[:self.model.nq])
        return np.r_[diff,state[self.model.nq:]-target[self.model.nq:]]

    def perturb(self,state,index,amount):
        x = state.copy()
        if index<self.nv:
            v = np.zeros(self.nv);v[index]=amount
            mujoco.mj_integratePos(self.model,x[:self.model.nq],v,1.)
        else:x[self.model.nq+index-self.nv]+=amount
        return x

    def transition(self,state,action):
        d,m=self.data,self.model
        mujoco.mj_resetData(m,d)
        d.qpos[:],d.qvel[:] = state[:m.nq],state[m.nq:]
        for _ in range(self.contract['decimation']):
            d.ctrl[self.aids] = self.torque(action,d.qpos,d.qvel)
            mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
        x=np.r_[d.qpos.copy(),d.qvel.copy()]
        if not np.isfinite(x).all() or not np.isclose(d.time,self.dt,rtol=0,atol=1e-10):
            raise FloatingPointError('Predictor nonfinite state or hidden simulator time reset')
        return x

    def design(self,output):
        count,nx,nu=self.n,2*self.nv,8
        A,B,defect=np.empty((count,nx,nx)),np.empty((count,nx,nu)),np.empty((count,nx))
        eps=self.settings['finite_difference_epsilon'];start=time.perf_counter()
        for k in range(count):
            x,u=self.targets[k],self.nominal_action[k];target=self.targets[min(k+1,count-1)]
            defect[k]=self.error(target,self.transition(x,u))
            for j in range(nx):
                plus=self.error(target,self.transition(self.perturb(x,j,eps),u))
                minus=self.error(target,self.transition(self.perturb(x,j,-eps),u))
                A[k,:,j]=(plus-minus)/(2*eps)
            for j in range(nu):
                delta=np.zeros(nu);delta[j]=eps
                plus=self.error(target,self.transition(x,u+delta))
                minus=self.error(target,self.transition(x,u-delta))
                B[k,:,j]=(plus-minus)/(2*eps)
            if k%50==0:print('LINEARIZE',k,'/',count,flush=True)
        K,feed,P=affine_riccati(A,B,defect,self.Q,self.R,self.Q*self.settings['terminal_multiplier'])
        np.savez_compressed(output,K=K,feed=feed,P=P,defect=defect,Q=self.Q,R=self.R,
                            targets=self.targets,nominal_action=self.nominal_action,
                            settings_json=np.array(json.dumps(self.settings,sort_keys=True)))
        return {'elapsed_s':time.perf_counter()-start,'max_reference_defect_norm':float(np.linalg.norm(defect,axis=1).max()),
                'max_unclipped_affine_feedforward_action_abs':float(np.abs(self.nominal_action+feed).max())}


class TVLQR:
    def __init__(self,predictor,design):
        self.predictor=predictor
        with np.load(design,allow_pickle=False) as d:self.design={k:d[k].copy() for k in d.files}
        if json.loads(str(self.design['settings_json'])) != predictor.settings:
            raise ValueError('Controller design configuration mismatch')
        np.testing.assert_allclose(self.design['targets'],predictor.targets,atol=1e-10)

    def action(self,state,frame):
        k=min(frame,self.predictor.n-1)
        e=self.predictor.error(self.design['targets'][k],state)
        raw=self.design['nominal_action'][k]+self.design['feed'][k]-self.design['K'][k]@e
        return np.clip(raw,-1.,1.),{'raw_action_max_abs':float(np.abs(raw).max())}


class ShootingNMPC(TVLQR):
    def __init__(self,predictor,design):
        super().__init__(predictor,design)
        self.config=predictor.settings['nmpc'];self.rng=np.random.default_rng(predictor.settings['seed'])
        self.last_plan=None

    def score(self,state,plan,frame):
        p=self.predictor;cost=0.;x=state.copy()
        try:
            for j,a in enumerate(plan):
                k=min(frame+j,p.n-1)
                e=p.error(p.targets[k],x);du=a-p.nominal_action[k]
                cost+=e@p.Q@e+du@p.R@du
                x=p.transition(x,a)
            k=min(frame+len(plan),p.n-1)
            e=p.error(p.targets[k],x);cost+=e@self.design['P'][k]@e
        except FloatingPointError:return np.inf
        return float(cost)

    def action(self,state,frame):
        p=self.predictor;c=self.config;h=c['horizon_steps']
        baseline=[];x=state.copy()
        for j in range(h):
            a,_=super().action(x,frame+j);baseline.append(a)
            try:x=p.transition(x,a)
            except FloatingPointError:break
        baseline=np.array(baseline+[baseline[-1]]*(h-len(baseline)))
        best=baseline.copy();best_cost=self.score(state,best,frame)
        if self.last_plan is not None:
            shifted=np.r_[self.last_plan[1:],self.last_plan[-1:]]
            score=self.score(state,shifted,frame)
            if score<best_cost:best,best_cost=shifted,score
        initial_cost=best_cost
        blocks=c['control_blocks'];assignment=np.minimum(np.arange(h)*blocks//h,blocks-1)
        mean=np.zeros((blocks,8));std=np.tile(np.r_[np.full(6,c['leg_noise_std_rad']),np.full(2,c['wheel_noise_std_normalized'])],(blocks,1))
        for _ in range(c['iterations']):
            noise=self.rng.normal(size=(c['population'],blocks,8))*std+mean
            candidates=np.clip(best[None]+noise[:,assignment],-1.,1.)
            candidates[0]=best
            scores=np.array([self.score(state,plan,frame) for plan in candidates])
            elite=np.argsort(scores)[:c['elite']]
            winner=int(elite[0])
            mean=noise[elite].mean(axis=0);std=np.maximum(noise[elite].std(axis=0),.1*std)
            if scores[winner]<best_cost:best,best_cost=candidates[winner].copy(),float(scores[winner])
        if not np.isfinite(best_cost):
            a,_=super().action(state,frame)
            return a,{'optimization_failed':True,'baseline_fallback':True}
        self.last_plan=best.copy()
        return best[0],{'optimization_failed':False,'initial_model_cost':initial_cost,'selected_model_cost':best_cost,
                       'predicted_cost_nonincreasing':bool(best_cost<=initial_cost+1e-9)}

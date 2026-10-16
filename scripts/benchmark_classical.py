"""Design and run full-body classical tracking pilots in MuJoCo.

No Isaac result is implied by a MuJoCo-only run. Same original actuator mapping,
reset and reference as the actor evaluation; pose/height do not stop episodes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from classical_tracking import Predictor, TVLQR, ShootingNMPC, ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if args.output_dir.exists():
        raise FileExistsError('Preserving existing experiment output: '+str(args.output_dir))
    settings=json.loads(args.settings.read_text())
    contract=json.loads(args.contract.read_text())
    p=Predictor(args.model,args.motion,contract,settings)
    args.output_dir.mkdir(parents=True)
    report={'kind':'full_body_classical_controller_pilot','engine':'MuJoCo','mujoco_version':mujoco.__version__,
            'settings':settings,'model_sha256':sha(args.model),'motion_sha256':sha(args.motion),
            'actuator_contract_sha256':sha(args.contract),'controller_source_sha256':sha(ROOT/'scripts/classical_tracking.py'),
            'runner_source_sha256':sha(Path(__file__)),'state_access':'full simulator qpos/qvel with future reference preview',
            'learned_policy_used':False,'root_state_prescribed_after_reset':False,'pose_height_failure_gate':False}
    if args.controller=='design':
        start=time.perf_counter();report['design']=p.design(args.output_dir/'tvlqr_design.npz')
        report.update(status='designed',wall_s=time.perf_counter()-start)
    else:
        if not args.design or args.mu is None:
            raise ValueError('Evaluation requires a frozen design and plant friction')
        controller=TVLQR(p,args.design) if args.controller=='tvlqr' else ShootingNMPC(p,args.design)
        # A separate predictor instance/model is never the simulated plant.
        model,_=p.api.load_model(args.model,no_visual_mesh=True)
        p.api.configure_model(model,contract);model.geom_friction[:,0]=args.mu
        data=mujoco.MjData(model)
        measured=p.api.MujocoState(model,data,p.reference)
        measured.initialize()
        np.testing.assert_allclose(np.r_[data.qpos,data.qvel],p.targets[0],atol=1e-10)
        records=[];decisions=[];start=time.perf_counter()
        def record():
            state=measured.measure();state.update(measured.ground_contact_slip(5.))
            nw,nf=measured.nonwheel_ground_contacts()
            state.update(time_s=float(data.time),qpos=data.qpos.copy(),qvel=data.qvel.copy(),
                         nonwheel_ground_contact_count=nw,nonwheel_ground_contact_force_norm_n=nf)
            records.append(state)
        record()
        for k in range(p.n):
            now=time.perf_counter()
            action,info=controller.action(np.r_[data.qpos.copy(),data.qvel.copy()],k)
            elapsed=time.perf_counter()-now
            if action.shape!=(8,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.+1e-9:
                raise FloatingPointError('Invalid classical action')
            decisions.append({'frame':k,'time_s':k*p.dt,'action':action.copy(),'compute_s':elapsed,'solver':info})
            for _ in range(contract['decimation']):
                data.ctrl[measured.aids]=p.torque(action,data.qpos,data.qvel)
                mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
                record()
                expected=(len(records)-1)*model.opt.timestep
                if not np.isfinite(data.qpos).all() or not np.isclose(data.time,expected,atol=1e-8,rtol=0):
                    raise FloatingPointError('Plant nonfinite state or hidden simulator reset')
            if k%50==0:print(args.controller,'mu',args.mu,'frame',k,'compute_ms',round(1000*elapsed,2),flush=True)
        trace={name:np.asarray([r[name] for r in records]) for name in records[0]}
        trace.update(actions=np.asarray([r['action'] for r in decisions]),
                     compute_s=np.asarray([r['compute_s'] for r in decisions]),
                     qpos_joint_names=np.array([model.joint(i).name for i in range(1,model.njnt)]))
        np.savez_compressed(args.output_dir/'rollout.npz',**trace)
        t=trace['time_s'];r=trace['root_pos']
        rt=np.arange(p.n)*p.dt
        desired=np.column_stack([np.interp(t,rt,p.reference.archive['body_pos_w'][:,0,i]) for i in range(2)])
        error=np.linalg.norm(r[:,:2]-desired,axis=1);core=(t>=1.44-1e-9)&(t<=4.+1e-9)
        velocity=trace['root_com_lin_vel_w'];speed=np.linalg.norm(velocity[:,:2],axis=1)
        timing=trace['compute_s'];pose=[]
        for quat in trace['root_quat_wxyz']:
            mat=p.api.rotation(quat);pose.append(np.degrees(np.arccos(np.clip(mat[2,2],-1,1))))
        report.update(status='completed',controller=args.controller,validation_mu=args.mu,
                      prediction_mu=settings['nominal_prediction_mu'],duration_s=float(t[-1]),wall_s=time.perf_counter()-start,
                      design_sha256=sha(args.design),rollout_sha256=sha(args.output_dir/'rollout.npz'),
                      core_xy_rmse_m=float(np.sqrt(np.mean(error[core]**2))),core_xy_peak_m=float(error[core].max()),
                      core_speed_mean_m_s=float(speed[core].mean()),core_tilt_p95_deg=float(np.quantile(np.asarray(pose)[core],.95)),
                      nonwheel_core_contact_fraction=float(np.mean(trace['nonwheel_ground_contact_count'][core]>0)),
                      compute_ms_median_p95_max=(1000*np.quantile(timing,[.5,.95,1])).tolist(),
                      compute_over_20ms_fraction=float(np.mean(timing>.02)),
                      action_bound_fraction=float(np.mean(np.abs(trace['actions'])>=1.-1e-8)),
                      solver_failures=sum(bool(d['solver'].get('optimization_failed',False)) for d in decisions))
        (args.output_dir/'solver_log.json').write_text(json.dumps([{**r,'action':r['action'].tolist()} for r in decisions],indent=2,allow_nan=False)+'\n')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['settings']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--settings',type=Path,default=ROOT/'config/classical_tracking.json')
    parser.add_argument('--controller',choices=['design','tvlqr','nmpc'],required=True)
    parser.add_argument('--design',type=Path)
    parser.add_argument('--mu',type=float)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())

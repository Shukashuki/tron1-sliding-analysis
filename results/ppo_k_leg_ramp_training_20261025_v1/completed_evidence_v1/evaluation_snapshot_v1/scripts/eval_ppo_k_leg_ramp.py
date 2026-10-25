"""Native nominal validation with the same leg ramp used during K learning.

PhysX reuses the GPU runtime; native CPU MuJoCo receives identical actual leg
target endpoints and fractions before its unchanged DC motor computation.
Raw actor actions and observations, wheel effort and native states stay intact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import train_ppo_k_leg_ramp as training
import verify_ppo_k_leg_ramp as verification


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parent(k_snapshot):
    path=Path(k_snapshot).resolve()/'scripts/eval_ppo_ground_contact_v2k.py'
    assert sha(path)=='cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c'
    source=path.read_text();old='COHORT_ID = "'+training.OLD_ID+'"';assert source.count(old)==1
    source=source.replace(old,'COHORT_ID = "'+training.COHORT_ID+'"')
    namespace=dict(__name__='_K_leg_ramp_parent_evaluator',__file__=str(path))
    sys.path.append(str(path.parent));exec(compile(source,'<K_leg_ramp_parent_evaluator>','exec'),namespace)
    namespace['__file__']=str(Path(__file__).resolve())
    namespace['verify_training']=lambda d,p,allow_smoke=False:verification.verify_training(d,p,k_snapshot,allow_smoke=allow_smoke)
    return SimpleNamespace(**namespace)


class NumpyRamp:
    def __init__(self):self.previous=None;self.rows=[]
    def target(self,frame,substep,action,joint_pos,q0,contract):
        assert substep==len(self.rows)%8 and frame==len(self.rows)//8
        desired=np.asarray(q0[:6])+contract['leg_position_scale']*np.clip(action,*contract['raw_action_clip'])[:6]
        if substep==0:
            self.start=np.asarray(joint_pos[:6]).copy() if self.previous is None else self.previous.copy()
            self.desired=desired.copy()
        else:np.testing.assert_array_equal(desired,self.desired)
        command=self.desired.copy() if substep==7 else self.start+(self.desired-self.start)*((substep+1)/8.)
        self.rows.append(dict(start=self.start.copy(),desired=self.desired.copy(),actual=command.copy(),
            wheel_effort=contract['wheel_torque_scale_nm']*np.clip(action,*contract['raw_action_clip'])[6:]))
        self.previous=command.copy()
        return command


def save_targets(rows,output,*,names):
    assert len(rows)==2792
    arrays={k:np.asarray([r[k] for r in rows]) for k in ('start','desired','actual','wheel_effort')}
    arrays['time_before_step_s']=np.arange(2792)*.0025
    arrays['leg_joint_names']=np.asarray(names)
    assert len(names)==6
    assert all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in 'fi')
    start,desired,actual=[arrays[k].reshape(349,8,6) for k in ('start','desired','actual')]
    np.testing.assert_array_equal(start,np.broadcast_to(start[:,:1],start.shape))
    np.testing.assert_array_equal(desired,np.broadcast_to(desired[:,:1],desired.shape))
    np.testing.assert_array_equal(actual[:,-1],desired[:,-1])
    np.testing.assert_allclose(actual,start+(desired-start)*(np.arange(1,9)[None,:,None]/8.),rtol=2e-6,atol=2e-7)
    np.testing.assert_array_equal(start[1:,0],actual[:-1,-1])
    np.savez_compressed(output,**arrays)
    return dict(command_recording_sha256=sha(output),native_target_applications=2792,policy_windows=349,
        exact_endpoints_and_window_fractions_verified=True,reset_start='measured native leg joint positions',
        wheel_effort_interpolated=False,source_pose_smoothed_or_corrected=False)


def materialize_mujoco(source):
    pairs=[('def action_torques(action, joint_pos, joint_vel, q0, contract):',
            'def action_torques(action, joint_pos, joint_vel, q0, contract, *, leg_target_override=None):'),
        ('    target = q0[:6] + contract["leg_position_scale"] * action[:6]',
         '    target = q0[:6] + contract["leg_position_scale"] * action[:6] if leg_target_override is None else leg_target_override'),
        ('            torque = action_torques(action, state["joint_pos"], state["joint_vel"], reference.q0, contract)',
         '            actual_target = _leg_ramp.target(frame, substep, action, state["joint_pos"], reference.q0, contract)\n'
         '            torque = action_torques(action, state["joint_pos"], state["joint_vel"], reference.q0, contract, leg_target_override=actual_target)')]
    for old,new in pairs:
        assert source.count(old)==1,'Missing/ambiguous native target integration: '+old[:90]
        source=source.replace(old,new,1)
    compile(source,'<K_leg_ramp_native_mujoco>','exec');return source


def materialize_physx(source):
    old='            reward_accumulator = install_drift_reward_sampling(env, task_profile)'
    assert source.count(old)==1
    source=source.replace(old,'            reward_accumulator = _leg_ramp_install(env, _new_plan)')
    compile(source,'<K_leg_ramp_native_physx>','exec');return source


def main(argv=None):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--k-snapshot',type=Path,required=True);a,rest=p.parse_known_args(argv)
    parent=load_parent(a.k_snapshot);plan=parent.prepare_plan(rest);assert plan['mu']==.7
    plan['wrapper_sha256']=sha(__file__)
    sources={str(Path(p).resolve()):sha(p) for p in (__file__,training.__file__,verification.__file__)}
    def metadata(report,current,contract,rows,names,extra):
        command=save_targets(rows,current['output']/'leg_command_trajectory.npz',names=names)
        backend=current['args'].backend;name='rollout.npz' if backend=='mujoco' else 'physics_trajectory.npz'
        info=dict(kind='K_leg_ramp_trained_native_nominal_evaluation',status='completed_requires_analysis',
            training_cohort_id=training.COHORT_ID,leg_command_execution=training.EXECUTION,
            smoke_policy_only=current['cohort']['smoke_only'],
            normalized_actor_sha256=current['cohort']['input_sha256']['policy.pt'],
            validation_backend=backend,mu=.7,native_states=2793,policy_decisions=349,physics_dt_s=.0025,policy_dt_s=.02,
            recording_sha256=sha(current['output']/name),source_sha256=sources,command_audit=command,
            runtime=extra,task_success_claimed=False)
        assert all(sha(p)==h for p,h in sources.items())
        report['leg_ramp_evaluation']=info;contract['validation_leg_command_execution']=copy.deepcopy(info)
        with (current['output']/'leg_ramp_evaluation_status.json').open('x') as f:json.dump(info,f,indent=2,allow_nan=False)
    if plan['args'].backend=='mujoco':
        native=parent.native;control=NumpyRamp()
        generated=materialize_mujoco(native.materialize_source(plan['evaluator'].read_text(),plan['legacy_root']))
        plan['generated_sha256']=hashlib.sha256(generated.encode()).hexdigest()
        def finish(report,arrays,current):
            parent._finish(report,arrays,current)
            metadata(report,current,report['contract'],control.rows,np.asarray(arrays['joint_names']).tolist()[:6],
                dict(mode=training.EXECUTION['mode'],original_native_DC_motor_clipping_preserved=True))
        namespace=dict(__name__='_K_leg_ramp_native_mujoco',__file__=str(Path(__file__).resolve()),
            _new_plan=plan,_new_audit_materials=native.audit_materials,_new_finish_report=finish,_leg_ramp=control)
        previous=list(sys.path)
        try:
            sys.path[:0]=[str(plan['evaluator'].parent),str(plan['legacy_root'])]
            exec(compile(generated,'<K_leg_ramp_native_mujoco>','exec'),namespace)
            policy=namespace['load_torchscript'](plan['cohort']['actor'])
            return namespace['run_evaluation'](policy,plan['motion'],plan['args'].model,plan['output'],
                policy_path=plan['cohort']['actor'],no_visual_mesh=True,contract_path=plan['contract_path'])
        finally:sys.path[:]=previous
    import eval_ppo_backend_400hz_isaac as physx
    generated=materialize_physx(physx.materialize_source(plan['evaluator'].read_text(),plan['legacy_root']))
    plan['generated_sha256']=hashlib.sha256(generated.encode()).hexdigest()
    runtime=[];target_rows=[]
    def configure(cfg,profile,current):
        physx.configure(cfg,profile,current)
        import ppo_ground_contact_v2k as contact
        contact.configure_sensors(cfg.scene);cfg.rewards.drift_grounded=contact.reward_term(.4)
    def install(env,current):
        import ppo_ground_contact_v2k as contact
        import ppo_batched_leg_ramp as ramp_module
        owner=contact.install(env,current['cohort']['protocol']['ground_contact_reward_v2'])
        r=ramp_module.BatchedLegRamp(env,owner);runtime.append(r)
        sources[str(Path(ramp_module.__file__).resolve())]=sha(ramp_module.__file__)
        original=r.apply
        def observe():
            result=original()
            target_rows.append({k:v[0].detach().clone() for k,v in dict(start=r.start,desired=r.desired,
                actual=r.robot.data.joint_pos_target[:,r.ids],wheel_effort=r.wheels.processed_actions).items()})
            return result
        r.apply=observe
        return owner
    def runner(cohort,device):
        result=parent.common.StrictPPOExportRunner(cohort,device)
        result.inference_provenance=parent.provenance(cohort);return result
    def finish(report,contract,current,raw):
        parent._finish(report,raw,current,contract=contract)
        r=runtime[0];rows=[{k:v.cpu().numpy() for k,v in row.items()} for row in target_rows]
        metadata(report,current,contract,rows,[r.robot.joint_names[i] for i in r.ids],r.audit())
    namespace=dict(__name__='_K_leg_ramp_native_physx',__file__=str(Path(__file__).resolve()),_new_plan=plan,
        _new_configure=configure,_new_audit_materials=physx.audit_materials,_new_runner=runner,
        _new_finish_report=finish,_leg_ramp_install=install)
    previous=sys.argv
    try:
        os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
        sys.argv=[str(Path(__file__).resolve()),*plan['inherited_argv']]
        exec(compile(generated,'<K_leg_ramp_native_physx>','exec'),namespace);namespace['main']()
    finally:sys.argv=previous


if __name__=='__main__':main(sys.argv[1:])

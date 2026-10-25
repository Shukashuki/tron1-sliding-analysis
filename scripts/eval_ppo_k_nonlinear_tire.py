"""Full-horizon nominal evaluation of the newly nonlinear-tire-trained PPO.

PhysX nonlinear uses the exact training runtime, including reset-first zero
force. Native-Coulomb PhysX and native CPU MuJoCo are explicit cross-law
transfers. No reward, root target, reset splicing or policy adaptation occurs.
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

import train_ppo_k_nonlinear_tire as training
import verify_ppo_k_nonlinear_tire as verification


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parent(k_snapshot):
    path=Path(k_snapshot).resolve()/"scripts/eval_ppo_ground_contact_v2k.py"
    assert sha(path)=="cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c"
    source=path.read_text();anchor='COHORT_ID = "'+training.OLD_ID+'"'
    assert source.count(anchor)==1
    source=source.replace(anchor,'COHORT_ID = "'+training.COHORT_ID+'"')
    namespace=dict(__name__='_nonlinear_tire_training_parent_evaluator',__file__=str(path))
    sys.path.append(str(path.parent))
    exec(compile(source,'<nonlinear_tire_training_parent_evaluator>','exec'),namespace)
    namespace['__file__']=str(Path(__file__).resolve())
    namespace['verify_training']=lambda directory,protocol,allow_smoke=False:verification.verify_training(
        directory,protocol,k_snapshot,allow_smoke=allow_smoke)
    return SimpleNamespace(**namespace)


class TireTrace:
    """Read-only GPU snapshots around each original runtime write; no extra steps."""
    def __init__(self,env,runtime):
        import torch
        self.env,self.runtime,self.torch=env,runtime,torch
        self.original=env.scene.write_data_to_sim
        self.rows=[];self.last=runtime.calls
        env.scene.write_data_to_sim=self.write

    def write(self,*args,**kwargs):
        t=self.torch;r=self.runtime;data=r.robot.data
        counter=int(self.env._sim_step_counter)
        capture=r.owner.inside_step and counter==r.last_counter+1
        if capture:
            load=t.stack([s.data.force_matrix_w[:,0,0,2] for s in r.sensors],dim=1)
            row={
                'COM_position_w_m':data.body_com_pos_w[:,r.ids].clone(),
                'COM_velocity_w_m_s':data.body_com_lin_vel_w[:,r.ids].clone(),
                'angular_velocity_w_rad_s':data.body_com_ang_vel_w[:,r.ids].clone(),
                'wheel_quaternion_wxyz':data.body_link_quat_w[:,r.ids].clone(),
                'preceding_contact_position_w_m':t.stack([s.data.contact_pos_w[:,0,0] for s in r.sensors],dim=1).clone(),
                'preceding_normal_load_n':t.where(r.ready[:,None],load.clamp_min(0),0.).clone(),
                'history_valid':r.ready.clone(),
                'mu':r.mu.clone()}
            fragments,counts=[],[]
            for sensor in r.sensors:
                force,_,count,start=sensor.contact_physx_view.get_friction_data(self.env.physics_dt)
                assert tuple(count.shape)==tuple(start.shape)==(1,1)
                indices=start[:,0,None].to(t.long)+t.arange(128,device=force.device)[None]
                mask=t.arange(128,device=force.device)[None]<count[:,0,None]
                selected=force[indices.clamp(0,len(force)-1)]
                fragments.append(t.where(mask[...,None],selected,0.).sum(1))
                counts.append(count[:,0])
            row['native_friction_force_w_n']=t.stack(fragments,dim=1).clone()
            row['native_patch_count']=t.stack(counts,dim=1).clone()
        result=self.original(*args,**kwargs)
        if capture:
            assert r.calls==self.last+1
            row['applied_tangent_force_w_n']=r.robot._external_force_b[:,r.ids].clone()
            row['applied_tangent_moment_COM_w_nm']=r.robot._external_torque_b[:,r.ids].clone()
            self.rows.append(row);self.last=r.calls
        return result

    def save(self,path):
        assert len(self.rows)==2792 and self.runtime.calls==2792
        arrays={key:self.torch.stack([r[key] for r in self.rows]).cpu().numpy()[:,0] for key in self.rows[0]}
        arrays['time_before_step_s']=np.arange(2792)*.0025
        assert np.max(arrays['native_patch_count'])<128
        assert np.isfinite(arrays['native_friction_force_w_n']).all()
        assert np.max(np.linalg.norm(arrays['native_friction_force_w_n'][1:],axis=-1))<1e-3
        assert not arrays['history_valid'][0] and arrays['history_valid'][1:].all()
        np.testing.assert_array_equal(arrays['applied_tangent_force_w_n'][0],np.zeros((2,3)))
        np.savez_compressed(path,**arrays)


def main(argv=None):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--k-snapshot',type=Path,required=True)
    p.add_argument('--law',choices=('nonlinear','native'),required=True)
    a,remaining=p.parse_known_args(argv)
    parent=load_parent(a.k_snapshot);plan=parent.prepare_plan(remaining)
    assert plan['mu']==.7
    plan['wrapper_sha256']=sha(__file__)
    sources={str(Path(p).resolve()):sha(p) for p in (__file__,training.__file__,verification.__file__)}
    runtimes,traces=[],[]

    def metadata(report,current,contract):
        backend=current['args'].backend
        info=dict(kind='K_nonlinear_tire_trained_full_horizon_evaluation',status='completed_requires_analysis',
            training_model_id=training.MODEL_ID,training_cohort_id=training.COHORT_ID,
            training_model=current['cohort']['protocol']['wheel_contact_model'],
            normalized_actor_sha256=current['cohort']['input_sha256']['policy.pt'],
            model_training_audit_sha256=current['cohort']['model_training_audit_sha256'],
            validation_backend=backend,validation_force_law=a.law,mu=.7,
            cross_law_transfer=a.law=='native',native_states=2793,policy_decisions=349,
            physics_dt_s=.0025,policy_dt_s=.02,source_sha256=sources,task_success_claimed=False)
        if runtimes:
            traces[0].save(current['output']/'tire_trajectory.npz')
            info['runtime']=runtimes[0].audit()
            info['tire_trajectory_sha256']=sha(current['output']/'tire_trajectory.npz')
        recording='physics_trajectory.npz' if backend=='physx' else 'rollout.npz'
        info['recording_sha256']=sha(current['output']/recording)
        assert all(sha(p)==digest for p,digest in sources.items())
        report['nonlinear_tire_model_evaluation']=info
        contract['validation_wheel_contact_model']=copy.deepcopy(info)
        with (current['output']/'nonlinear_tire_evaluation_status.json').open('x') as f:
            json.dump(info,f,indent=2,allow_nan=False)

    if plan['args'].backend=='mujoco':
        assert a.law=='native','No nonlinear tire replacement is implemented in native MuJoCo'
        native=parent.native
        generated=native.materialize_source(plan['evaluator'].read_text(),plan['legacy_root'])
        plan['generated_sha256']=hashlib.sha256(generated.encode()).hexdigest()
        # Same native hooks as the immutable K evaluator, with an extra record
        # of this policy's nonlinear training and native-force transfer.
        original_finish=parent._finish
        def finish(report,arrays,current):
            original_finish(report,arrays,current)
            metadata(report,current,report['contract'])
        namespace=dict(__name__='_nonlinear_tire_trained_native_transfer',__file__=str(Path(__file__).resolve()),
            _new_plan=plan,_new_audit_materials=native.audit_materials,_new_finish_report=finish)
        previous=list(sys.path)
        try:
            sys.path[:0]=[str(plan['evaluator'].parent),str(plan['legacy_root'])]
            exec(compile(generated,'<nonlinear_tire_trained_native_transfer>','exec'),namespace)
            actor=namespace['load_torchscript'](plan['cohort']['actor'])
            return namespace['run_evaluation'](actor,plan['motion'],plan['args'].model,plan['output'],
                policy_path=plan['cohort']['actor'],no_visual_mesh=True,contract_path=plan['contract_path'])
        finally:sys.path[:]=previous

    import eval_ppo_backend_400hz_isaac as physx
    generated=physx.materialize_source(plan['evaluator'].read_text(),plan['legacy_root'])
    plan['generated_sha256']=hashlib.sha256(generated.encode()).hexdigest()

    def configure(cfg,profile,current):
        physx.configure(cfg,profile,current)
        if a.law=='nonlinear':
            from physx_batched_tire import configure_sensors
            configure_sensors(cfg.scene)

    def materials(env,current):
        import torch
        torch.cuda.set_device(current['args'].physical_gpu_index)
        observed=physx.audit_materials(env,current)
        if a.law=='nonlinear':
            import physx_batched_tire as model
            sources.update({str(Path(p).resolve()):sha(p) for p in
                (model.__file__,Path(model.__file__).with_name('nonlinear_tire_torch.py'))})
            owner=SimpleNamespace(env=env,scene=env.scene,inside_step=True)
            runtime=model.BatchedTireRuntime(env,owner=owner)
            runtimes.append(runtime);traces.append(TireTrace(env,runtime))
            observed['before_force_substitution']=copy.deepcopy(observed)
            observed.update(runtime.audit_materials())
        return observed

    def runner(cohort,device):
        result=parent.common.StrictPPOExportRunner(cohort,device)
        result.inference_provenance=parent.provenance(cohort)
        return result

    def finish(report,contract,current,raw):
        parent._finish(report,raw,current,contract=contract)
        metadata(report,current,contract)

    namespace=dict(__name__='_nonlinear_tire_trained_physx_evaluation',__file__=str(Path(__file__).resolve()),
        _new_plan=plan,_new_configure=configure,_new_audit_materials=materials,
        _new_runner=runner,_new_finish_report=finish)
    previous=sys.argv
    try:
        os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
        sys.argv=[str(Path(__file__).resolve()),*plan['inherited_argv']]
        exec(compile(generated,'<nonlinear_tire_trained_physx_evaluation>','exec'),namespace)
        namespace['main']()
    finally:sys.argv=previous


if __name__=='__main__':main(sys.argv[1:])

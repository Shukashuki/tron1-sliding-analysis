"""Reset-safe GPU leg-target ramp for policies trained with this execution.

K retains sole environment-step/scene-update ownership. Only the six processed
position commands are interpolated over its eight real physical applications.
No state, observation, raw action, wheel effort or contact law is modified.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import MethodType

import torch

MODE='leg_position_linear_8substeps_reset_at_measured_q_v1'
SDK_ACTION_SHA='1b3dcb55d969d886cee500e660f12331e49f8638fec98cde30d2b818a1ca1692'


class BatchedLegRamp:
    def __init__(self,env,owner,*,allow_cpu_fixture=False):
        self.env,self.owner,self.robot=env,owner,env.scene['robot']
        self.device=torch.device(env.device)
        assert (self.device.type=='cuda' and self.device.index is not None) or allow_cpu_fixture
        assert env.physics_dt==.0025 and env.step_dt==.02 and env.cfg.decimation==8
        assert owner.env is env and owner.robot is self.robot
        assert not hasattr(env,'_tron1_batched_leg_ramp')
        self.term=env.action_manager.get_term('joint_pos')
        self.wheels=env.action_manager.get_term('wheel_effort')
        assert self.term._asset is self.robot and self.wheels._asset is self.robot
        self.ids=[int(i) for i in self.term._joint_ids]
        self.wheel_ids=[int(i) for i in self.wheels._joint_ids]
        assert len(self.ids)==6 and len(set(self.ids))==6 and len(self.wheel_ids)==2
        assert not set(self.ids)&set(self.wheel_ids)
        self.n=env.num_envs
        assert self.term.processed_actions.shape==(self.n,6)
        self.original_apply,self.original_reset=self.term.apply_actions,self.term.reset
        self.original_wheel_apply=self.wheels.apply_actions
        self.sources={}
        if not allow_cpu_fixture:
            assert type(self.term).__name__=='JointPositionAction'
            for fun in (self.original_apply,self.original_reset,self.robot.set_joint_position_target):
                p=Path(inspect.getsourcefile(fun)).resolve()
                self.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            assert any(Path(p).name=='joint_actions.py' and digest==SDK_ACTION_SHA for p,digest in self.sources.items())
        self.start=torch.zeros((self.n,6),device=self.device)
        self.desired=torch.zeros_like(self.start)
        self.last_applied=torch.zeros_like(self.start)
        self.window_targets=torch.zeros((self.n,8,6),device=self.device)
        self.valid=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.fault=torch.zeros_like(self.valid)
        self.max_increment=torch.zeros((),device=self.device)
        self.reset_world_count=torch.zeros((),device=self.device,dtype=torch.long)
        self.calls=self.resets=0
        self.initial_counter=int(env._sim_step_counter)
        self.apply_wrapper=MethodType(lambda term:self.apply(),self.term)
        self.reset_wrapper=MethodType(lambda term,env_ids=None:self.reset(env_ids),self.term)
        self.term.apply_actions=self.apply_wrapper
        self.term.reset=self.reset_wrapper
        env._tron1_batched_leg_ramp=self

    def attached(self):
        assert self.term.apply_actions is self.apply_wrapper and self.term.reset is self.reset_wrapper
        assert self.wheels.apply_actions==self.original_wheel_apply
        assert self.env._tron1_batched_leg_ramp is self

    def reset(self,env_ids=None):
        self.attached()
        # Reset may occur only between windows, including K's post-reward
        # autoreset. Never carry a previous episode's command into a new one.
        assert not self.owner.inside_step or self.owner.samples==8
        result=self.original_reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        self.reset_world_count.add_(self.valid[ids].numel())
        self.valid[ids]=False
        self.resets+=1
        return result

    def apply(self):
        self.attached()
        phase=self.calls%8
        assert self.owner.inside_step and self.owner.samples==phase
        assert int(self.env._sim_step_counter)==self.initial_counter+self.calls+1
        target=self.term.processed_actions
        assert target.shape==(self.n,6) and target.device==self.device and target.dtype==self.start.dtype
        if phase==0:
            self.start.copy_(torch.where(self.valid[:,None],self.last_applied,
                                        self.robot.data.joint_pos[:,self.ids]))
            self.desired.copy_(target)
            self.valid.fill_(True)
        else:
            self.fault.logical_or_((self.desired!=target).any(-1))
        command=self.desired if phase==7 else self.start+(self.desired-self.start)*((phase+1)/8.)
        self.fault.logical_or_(~torch.isfinite(command).all(-1))
        previous=self.start if phase==0 else self.last_applied
        self.max_increment.copy_(torch.maximum(self.max_increment,torch.linalg.vector_norm(command-previous,dim=-1).amax()))
        self.robot.set_joint_position_target(command,joint_ids=self.ids)
        applied=self.robot.data.joint_pos_target[:,self.ids]
        self.fault.logical_or_((applied!=command).any(-1))
        self.window_targets[:,phase]=applied
        self.last_applied.copy_(applied)
        self.calls+=1

    def audit(self):
        self.attached()
        assert self.calls%8==0 and self.calls==self.owner.total_samples
        assert not bool(self.fault.any().item()),'Nonfinite, changed held target or native application mismatch'
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.sources.items())
        expected=self.start[:,None]+(self.desired-self.start)[:,None]*(torch.arange(1,9,device=self.device)[None,:,None]/8.)
        expected[:,-1]=self.desired
        torch.testing.assert_close(self.window_targets,expected,rtol=0,atol=0)
        assert torch.equal(self.last_applied,self.desired)
        return dict(mode=MODE,physical_applications=self.calls,environment_physical_applications=self.calls*self.n,
            complete_policy_windows=self.calls//8,num_envs=self.n,device=str(self.device),
            finite_and_held_target_and_native_application_health=True,exact_window_endpoint=True,
            reset_start='measured native joint position on first physical application after each per-world reset',
            reset_calls=self.resets,reset_world_events=int(self.reset_world_count.item()),
            maximum_physical_leg_target_increment_l2_rad=float(self.max_increment.item()),
            source_sha256=self.sources,extra_environment_or_scene_step_owner=False,
            host_transfers_in_physical_hot_path=False,per_world_Python_loop=False,
            raw_action_observation_reward_mapping_changed=False,wheel_effort_interpolation=False,
            original_PD_and_DC_motor_limits_retained=True,
            last_window_first4_worlds=dict(start_rad=self.start[:4].cpu().tolist(),desired_rad=self.desired[:4].cpu().tolist(),
                actual_native_targets_rad=self.window_targets[:4].cpu().tolist()),task_success_claimed=False)

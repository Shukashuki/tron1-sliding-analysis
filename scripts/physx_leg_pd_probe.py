"""Explicit DC-motor gain intervention, with actual pre-physics compute records.

Training remains K/500/10. The optional40/2.5 execution uses the same DC model,
not the official configuration's implicit actuator. No solver drive is added.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
from types import MethodType
import numpy as np

PROFILES={'baseline_500_10':(500.,10.),'soft_40_2p5':(40.,2.5)}


class LegPDProbe:
    def __init__(self,env,profile,*,allow_cpu_fixture=False):
        import torch
        self.torch,self.env,self.robot,self.profile=torch,env,env.scene['robot'],profile
        assert profile in PROFILES and env.num_envs==1 and env.physics_dt==.0025 and env.step_dt==.02
        assert allow_cpu_fixture or torch.device(env.device).type=='cuda'
        self.legs,self.wheels=self.robot.actuators['legs'],self.robot.actuators['wheels']
        assert allow_cpu_fixture or type(self.legs).__name__=='DCMotor'
        self.ids=[int(i) for i in self.legs.joint_indices]
        wheel_ids=[int(i) for i in self.wheels.joint_indices]
        assert len(self.ids)==6 and len(wheel_ids)==2
        assert not set(self.ids)&set(wheel_ids)
        assert self.legs.stiffness.shape==self.legs.damping.shape==(1,6)
        assert bool((self.legs.stiffness==500.).all()) and bool((self.legs.damping==10.).all())
        self.initial_counter=int(env._sim_step_counter);self.rows=[]
        self.action_count=0;self.initialization_callbacks=[];self.gains_activated=False
        self.term=env.action_manager.get_term('joint_pos')
        action_ids=[int(i) for i in self.term._joint_ids]
        assert self.term._asset is self.robot and set(action_ids)==set(self.ids),('Joint asset/identity mismatch',self.term._asset is self.robot,action_ids,self.ids)
        self.original_apply=self.term.apply_actions
        self.original_compute=self.legs.compute
        self.sources={}
        if not allow_cpu_fixture:
            for f in (self.original_compute,self.original_apply,self.legs._clip_effort,self.robot._apply_actuator_model):
                p=Path(inspect.getsourcefile(f)).resolve();self.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        self.invariants={}
        for group,motor in (('legs',self.legs),('wheels',self.wheels)):
            expected={'effort_limit':80. if group=='legs' else 12.,
                'velocity_limit':15. if group=='legs' else 100.,
                '_saturation_effort':80. if group=='legs' else 12.}
            for field in ('effort_limit','velocity_limit','_saturation_effort','_vel_at_effort_lim'):
                value=getattr(motor,field)
                assert torch.is_tensor(value) or isinstance(value,(int,float))
                assert bool(torch.isfinite(torch.as_tensor(value)).all())
                if field in expected:assert bool((torch.as_tensor(value)==expected[field]).all())
                self.invariants[(group,field)]=value.detach().clone() if torch.is_tensor(value) else value
        self.wheel_gains=(self.wheels.stiffness.clone(),self.wheels.damping.clone())
        self.kp,self.kd=PROFILES[profile]
        self.action_wrapper=MethodType(lambda owner:self.apply_actions(),self.term)
        self.wrapper=MethodType(lambda owner,*a,**k:self.compute(*a,**k),self.legs)
        self.term.apply_actions=self.action_wrapper
        self.legs.compute=self.wrapper

    def plain(self,value):return value.detach().cpu().numpy().copy()

    def apply_actions(self):
        assert self.term.apply_actions is self.action_wrapper
        counter=int(self.env._sim_step_counter)
        assert counter==self.initial_counter+self.action_count+1 and self.action_count==len(self.rows),'Require consecutive physical action/motor pairs'
        if not self.gains_activated:
            if self.profile!='baseline_500_10':
                self.legs.stiffness.fill_(self.kp);self.legs.damping.fill_(self.kd)
                self.legs.cfg.stiffness,self.legs.cfg.damping=self.kp,self.kd
            self.gains_activated=True
        self.action_count+=1
        return self.original_apply()

    def compute(self,control_action,joint_pos,joint_vel):
        assert self.legs.compute is self.wrapper
        counter=int(self.env._sim_step_counter)
        if not self.gains_activated:
            assert self.action_count==0 and not self.rows and counter==self.initial_counter
            self.initialization_callbacks.append(counter)
            assert bool((self.legs.stiffness==500.).all()) and bool((self.legs.damping==10.).all())
            return self.original_compute(control_action,joint_pos,joint_vel)
        assert counter==self.initial_counter+self.action_count and self.action_count==len(self.rows)+1,'Require actual consecutive physical motor computations'
        row=dict(sim_counter=counter,joint_pos_rad=self.plain(joint_pos[0]),joint_vel_rad_s=self.plain(joint_vel[0]),
            target_pos_rad=self.plain(control_action.joint_positions[0]),target_vel_rad_s=self.plain(control_action.joint_velocities[0]),
            feedforward_effort_nm=self.plain(control_action.joint_efforts[0]),kp=self.plain(self.legs.stiffness[0]),kd=self.plain(self.legs.damping[0]))
        result=self.original_compute(control_action,joint_pos,joint_vel)
        row.update(computed_effort_nm=self.plain(self.legs.computed_effort[0]),applied_effort_nm=self.plain(self.legs.applied_effort[0]))
        self.rows.append(row)
        return result

    def gain_readback(self):
        assert self.gains_activated
        assert self.torch.equal(self.legs.stiffness,self.torch.full_like(self.legs.stiffness,self.kp))
        assert self.torch.equal(self.legs.damping,self.torch.full_like(self.legs.damping,self.kd))
        return dict(kp_range=[self.kp,self.kp],kd_range=[self.kd,self.kd])

    def save(self,path):
        assert len(self.rows)==self.action_count==2792 and self.legs.compute is self.wrapper
        self.gain_readback()
        for (group,name),value in self.invariants.items():
            current=getattr(self.robot.actuators[group],name)
            assert self.torch.equal(current,value) if self.torch.is_tensor(value) else current==value
        for now,before in zip((self.wheels.stiffness,self.wheels.damping),self.wheel_gains,strict=True):assert self.torch.equal(now,before)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in self.sources.items())
        arrays={name:np.asarray([row[name] for row in self.rows]) for name in self.rows[0]}
        assert all(np.isfinite(v).all() for v in arrays.values())
        assert all(arrays[k].shape==(2792,6) for k in arrays if k!='sim_counter')
        np.testing.assert_array_equal(arrays['kp'],np.full((2792,6),self.kp))
        np.testing.assert_array_equal(arrays['kd'],np.full((2792,6),self.kd))
        expected=(arrays['kp']*(arrays['target_pos_rad']-arrays['joint_pos_rad'])
            +arrays['kd']*(arrays['target_vel_rad_s']-arrays['joint_vel_rad_s'])+arrays['feedforward_effort_nm'])
        np.testing.assert_allclose(arrays['computed_effort_nm'],expected,rtol=2e-6,atol=2e-4)
        arrays['time_before_step_s']=np.arange(2792)*.0025
        arrays['leg_joint_names']=np.asarray([self.robot.joint_names[i] for i in self.ids])
        np.savez_compressed(path,**arrays)
        return dict(profile=self.profile,training_leg_kp=500.,training_leg_kd=10.,
            actual_gain_readback=self.gain_readback(),actual_actuator_class='DCMotor',
            physical_motor_computations=2792,policy_decisions=349,
            gain_activation='first_actual_policy_physics_action',
            initialization_compute_callbacks_at_original_gains=len(self.initialization_callbacks),
            PD_reconstruction_max_abs_error_nm=float(np.max(np.abs(expected-arrays['computed_effort_nm']))),
            explicit_control_only_no_solver_drive_added=True,torque_speed_limits_and_wheel_gains_unchanged=True,
            reward_reference_actor_observation_and_action_mapping_unchanged=True,
            physical_contact_geometry_material_and_mass_properties_unchanged=True,
            solver_clock_initial_state_and_command_hold_unchanged=True,
            official_implicit_actuator_reproduction_claimed=False,SDK_source_sha256=self.sources,
            fixed_motor_envelopes={group:{name:(self.plain(value).tolist() if self.torch.is_tensor(value) else value)
                for (g,name),value in self.invariants.items() if g==group} for group in ('legs','wheels')},
            inference_only=True,retrained_for_modified_gains=False,task_success_claimed=False)

"""H owner reuses G force/clock/reset lifecycle, adding paired native COM-vZ.

G's single owner marker and reward-manager factory intentionally remain in use:
this is one subclassed owner, not stacked reward/simulator hooks. Only H reward
math and one reward-only sampled signal are new; all14 other terms are intact.
"""
from __future__ import annotations
import hashlib
import importlib
import inspect
import math
from numbers import Real
from pathlib import Path
import ppo_ground_contact_v2g as parent
import ground_contact_reward_v2h as definition
from ground_contact_velocity_contract import (VELOCITY_SOURCE,SOURCE_PROPERTIES,
    verify_velocity_data_source,verify_velocity_source_file)

PARENT_COLLECTOR_SHA256="ed331d47aebe6016077645cfdfaa30aebdbc3cc6940ff7bbb5779c073e7f935a"
require,clock=parent.require,parent.clock
configure_sensors,reward_term=parent.configure_sensors,parent.reward_term
MARKER,SENSORS,WHEELS,GROUND=parent.MARKER,parent.SENSORS,parent.WHEELS,parent.GROUND
SUBSTEPS,PHYSICS_DT,POLICY_DT=parent.SUBSTEPS,parent.PHYSICS_DT,parent.POLICY_DT

class GroundContactV2H(parent.GroundContactV2G):
    def __init__(self,env,settings=None,*,allow_cpu_fixture=False):
        require(hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest()==PARENT_COLLECTOR_SHA256,
                "Frozen G owner source differs")
        definition.verify_base()
        self.h_settings=definition.validate_settings(settings)
        self.torch=importlib.import_module("torch")
        self.device=self.torch.device(env.device);self.num_envs=int(env.num_envs)
        require((self.device.type=="cuda" and self.device.index is not None)
                or (allow_cpu_fixture and self.device.type=="cpu"), "Explicit native device required")
        self.robot=env.scene["robot"];self.velocity_data=self.robot.data
        self.velocity_class=type(self.velocity_data)
        self.velocity_descriptors=tuple(inspect.getattr_static(self.velocity_class,name,None)
                                        for name in SOURCE_PROPERTIES)
        self.velocity_source_info=(None if self.device.type=="cpu"
                                   else verify_velocity_data_source(self.velocity_data))
        # Validate the new data channel BEFORE G attaches any lifecycle hooks.
        self._velocity()
        self.com_vz=self.torch.zeros((self.num_envs,SUBSTEPS),device=self.device)
        self.velocity_fault=self.torch.zeros(self.num_envs,device=self.device,dtype=self.torch.bool)
        self.velocity_samples=self.velocity_unique_reads=0
        self.velocity_last_clock=None
        self.velocity_penalty_sum=self.torch.zeros((),device=self.device)
        self.vz_min=self.torch.full((),float("inf"),device=self.device)
        self.vz_max=self.torch.full((),-float("inf"),device=self.device)
        super().__init__(env,allow_cpu_fixture=allow_cpu_fixture)

    def _data_timestamp(self):
        value=self.velocity_data._sim_timestamp
        require(isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(float(value)),
                "Native COM cache timestamp is invalid")
        return float(value)

    def _velocity(self):
        require(self.robot.data is self.velocity_data and type(self.velocity_data) is self.velocity_class,
                "Native COM data identity changed")
        require(all(inspect.getattr_static(self.velocity_class,name,None) is descriptor
                    for name,descriptor in zip(SOURCE_PROPERTIES,self.velocity_descriptors)),
                "Native COM velocity getter changed")
        value=self.velocity_data.root_com_lin_vel_w
        require(self.torch.is_tensor(value) and tuple(value.shape)==(self.num_envs,3)
                and value.device==self.device and value.dtype==self.torch.float32,
                "Require native GPU float32 base COM velocity[N,3]")
        cache=self.velocity_data._root_com_vel_w.timestamp
        require(isinstance(cache,Real) and not isinstance(cache,bool) and math.isfinite(float(cache))
                and float(cache)==self._data_timestamp(), "COM cache is stale for this scene refresh")
        return value[:,2]

    def update(self,*args,**kwargs):
        before=self.total_samples
        timestamp=self._data_timestamp()
        result=super().update(*args,**kwargs)
        if self.total_samples==before:
            return result
        require(self.total_samples==before+1 and self.inside_step and 1<=self.samples<=SUBSTEPS,
                "H velocity cannot invent an extra physical sample")
        after=self._data_timestamp()
        tolerance=self.dt_rounding+8*math.ulp(max(abs(timestamp),abs(after),PHYSICS_DT))
        require(abs(after-timestamp-PHYSICS_DT)<=tolerance,
                "Native COM epoch did not refresh with the physical scene update")
        vz=self._velocity()
        require(clock(self.env)==self.last_clock, "Physics advanced while sampling COM velocity")
        self.com_vz[:,self.samples-1]=vz
        self.velocity_fault.logical_or_(~self.torch.isfinite(vz))
        safe=self.torch.nan_to_num(vz,nan=0.,posinf=0.,neginf=0.)
        self.vz_min.copy_(self.torch.minimum(self.vz_min,safe.amin()))
        self.vz_max.copy_(self.torch.maximum(self.vz_max,safe.amax()))
        self.velocity_samples+=1
        self.velocity_last_clock=self.last_clock
        return result

    def reward(self):
        self._complete()
        require(self.velocity_samples==self.total_samples and self.velocity_last_clock==self.last_clock,
                "H reward requires velocity paired with all eight native force samples")
        require(not self.window_reset, "Reward read after autoreset")
        from height_smoothness import anchor_xy_tracking
        require(importlib.import_module("training.tron1_drift") is self.beta_module
                and self.beta_module.sideslip_tracking is self.beta_function,
                "Existing beta reward implementation changed")
        phase_good = self._phase() == self.held_phase
        anchor = anchor_xy_tracking(self.env, "motion", .3)
        speed = self.beta_module.speed_tracking(self.env, std=.35)
        beta_quality = self.beta_function(self.env)
        reference_velocity = self.motion_command.body_lin_vel_w
        require(self.torch.is_tensor(reference_velocity) and reference_velocity.ndim == 3
                and reference_velocity.shape[0] == self.num_envs
                and reference_velocity.shape[1] > 0 and reference_velocity.shape[2] == 3
                and reference_velocity.device == self.device
                and reference_velocity.dtype == self.torch.float32,
                "Require actual held-reference body velocities")
        reference_speed = self.torch.linalg.vector_norm(reference_velocity[:, 0, :2], dim=-1)
        phase_good = phase_good & (self._phase() == self.held_phase)
        require(clock(self.env) == self.last_clock, "Physics advanced while reading reference beta")
        self.phase_fault.logical_or_(~phase_good)
        components = definition.reward_torch(self.forces, self.mg, anchor, speed,
                                             beta_quality, reference_speed, self.com_vz,
                                             settings=self.h_settings, validate=False)
        value = self.torch.where(phase_good, components["raw_reward"],
                                self.torch.full_like(components["raw_reward"], float("nan")))
        self.reward_fault.logical_or_(~self.torch.isfinite(value))
        if self.last_read != self.steps:
            moving, low = reference_speed > .3, reference_speed <= .3
            self.low_reference_gate_fault.logical_or_(low & (components["beta_contact_gate"] != 1.))
            self.beta_counts.add_(self.torch.stack((moving.sum(), low.sum())))
            self.beta_gate_sum.add_(components["beta_contact_gate"].sum())
            self.beta_unique_reads += 1
            self.velocity_unique_reads += 1
            self.velocity_penalty_sum.add_(self.torch.nan_to_num(components["loaded_com_vz_penalty"]).sum())
            self.unique_reads += 1
            self.last_read = self.steps
            self.raw_sum.add_(self.torch.nan_to_num(value).sum())
        return value

    def audit(self,health=False):
        result=super().audit(health=health)
        result.update(reward_definition_version="ground_contact_v2h_any_wheel",
            settings=self.h_settings,
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            definition_sha256=hashlib.sha256(Path(definition.__file__).read_bytes()).hexdigest(),
            immutable_v2g_definition_sha256=definition.BASE_G_DEFINITION_SHA256,
            immutable_v2g_collector_sha256=PARENT_COLLECTOR_SHA256,
            velocity_source=VELOCITY_SOURCE,
            velocity_source_path=None if self.velocity_source_info is None else self.velocity_source_info["path"],
            velocity_source_sha256=None if self.velocity_source_info is None else self.velocity_source_info["sha256"],
            velocity_source_verified=self.velocity_source_info is not None,
            velocity_signal="base_COM_worldZ_m_s",velocity_num_envs=self.num_envs,
            velocity_all8_actual_substeps=True,velocity_is_finite_difference=False,
            velocity_end_state_repeated=False,velocity_cache_epoch_same_refresh_not_absolute_sim_time=True,
            velocity_physics_samples=self.velocity_samples,
            velocity_environment_samples=self.velocity_samples*self.num_envs,
            velocity_unique_reward_windows=self.velocity_unique_reads,
            base_owner_marker=MARKER)
        if health:
            if self.velocity_source_info is not None:
                require(verify_velocity_source_file(self.velocity_source_info["path"])==self.velocity_source_info,
                        "Native COM source changed during training")
            result.update(
                velocity_cache_refresh_health=self.velocity_samples==self.total_samples
                    and self.velocity_samples>0 and self.velocity_last_clock==self.last_clock,
                device_finite_com_vz_health=not bool(self.velocity_fault.any().item()),
                observed_com_vz_min_m_s=float(self.vz_min.item()),
                observed_com_vz_max_m_s=float(self.vz_max.item()),
                mean_added_velocity_penalty=float(self.velocity_penalty_sum.item())/
                    max(1,self.velocity_unique_reads*self.num_envs))
        return result

def install(env,settings=None):
    return GroundContactV2H(env,settings)


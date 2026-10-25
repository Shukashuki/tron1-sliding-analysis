"""Extend the frozen M owner with synchronous actual joint-velocity samples.

One owner, unchanged physical clocks/force sensors/M formula. The M ownership
marker intentionally remains: the R object is its subclass, so stacking with
M or the older collectors is still rejected. The audited reward version is R.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import ground_contact_reward_v2r as definition
import ppo_ground_contact_v2m as parent

PARENT_SHA = "1ff4d39e46be8b19e12db13143401a22cc933a2f1834a2eabbb947d1454c6add"
if hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest()!=PARENT_SHA:
    raise ValueError("Frozen M lifecycle changed")
configure_sensors, reward_term = parent.configure_sensors, parent.reward_term
MARKER = parent.MARKER


class GroundContactV2R(parent.GroundContactV2M):
    def __init__(self,env,settings=None,*,allow_cpu_fixture=False):
        self.r_settings = definition.validate_settings(settings)
        torch = importlib.import_module("torch")
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        parent.require(len(names)==len(set(names))==8 and all(n in names for n in definition.LEGS),
                       "Require actual eight unique joint names and all six legs")
        values = robot.data.joint_vel
        parent.require(torch.is_tensor(values) and values.shape==(env.num_envs,8)
                       and values.device==torch.device(env.device) and values.dtype==torch.float32,
                       "Require actual native joint velocities[N,8] in rad/s")
        self.leg_indices = torch.tensor([names.index(n) for n in definition.LEGS],device=env.device,dtype=torch.long)
        self.leg_velocities = torch.zeros((env.num_envs,8,6),device=env.device)
        self.leg_velocity_fault = torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.velocity_samples = self.total_velocity_samples = self.velocity_unique_reads = 0
        self.velocity_square_sum = torch.zeros((),device=env.device)
        self.velocity_cost_sum = torch.zeros((),device=env.device)
        super().__init__(env,parent.definition.validate_settings(),allow_cpu_fixture=allow_cpu_fixture)

    def step(self,*args,**kwargs):
        self.velocity_samples = 0
        self.leg_velocities.zero_()
        return super().step(*args,**kwargs)

    def update(self,*args,**kwargs):
        before = self.samples
        result = super().update(*args,**kwargs)
        if self.inside_step and self.samples==before+1:
            parent.require(self.velocity_samples==before,"Velocity/force physical sample clocks differ")
            values = self.robot.data.joint_vel
            parent.require(values.shape==(self.num_envs,8) and values.device==self.device and values.dtype==self.torch.float32,
                           "Actual joint velocity interface changed")
            value = values.index_select(-1,self.leg_indices)
            self.leg_velocities[:,before] = value
            self.leg_velocity_fault.logical_or_(~self.torch.isfinite(value).all(-1))
            self.velocity_samples += 1
            self.total_velocity_samples += 1
            parent.require(parent.clock(self.env)==self.last_clock,"Physics advanced while reading joint velocities")
        return result

    def _complete(self):
        super()._complete()
        parent.require(self.velocity_samples==8 and self.total_velocity_samples==self.total_samples,
                       "R requires all8 synchronized actual velocity/force samples")

    def reward(self):
        previous_read = self.last_read
        base = super().reward()
        mean_square = (self.leg_velocities/self.r_settings["leg_velocity_scale_rad_s"]).square().mean(dim=(-2,-1))
        cost = self.r_settings["leg_velocity_penalty_weight"]*mean_square
        result = base-cost
        self.reward_fault.logical_or_(~self.torch.isfinite(result))
        if previous_read!=self.steps:
            self.raw_sum.sub_(self.torch.nan_to_num(cost).sum())
            self.velocity_square_sum.add_(self.torch.nan_to_num(mean_square).sum())
            self.velocity_cost_sum.add_(self.torch.nan_to_num(cost).sum())
            self.velocity_unique_reads += 1
        return result

    def audit(self,health=False):
        result = super().audit(health=health)
        result.update(reward_definition_version="ground_contact_v2r_any_wheel",settings=self.r_settings,
            immutable_v2m_module_sha256=PARENT_SHA,immutable_v2m_definition_sha256=definition.BASE_M_SHA,
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            definition_sha256=hashlib.sha256(Path(definition.__file__).read_bytes()).hexdigest(),
            added_actual_leg_velocity_cost=True,leg_joint_names=definition.LEGS,
            leg_velocity_physics_samples=self.total_velocity_samples,leg_velocity_unique_reads=self.velocity_unique_reads,
            sampled_leg_environment_substeps=self.total_velocity_samples*self.num_envs,
            leg_velocity_cost_ungated=True,wheel_spin_penalized=False,extra_environment_steps=0)
        if health:
            result.update(device_finite_leg_velocity_health=not bool(self.leg_velocity_fault.any()),
                mean_actual_leg_velocity_square_rad2_s2=float(self.velocity_square_sum)/max(1,self.velocity_unique_reads*self.num_envs),
                mean_leg_velocity_cost=float(self.velocity_cost_sum)/max(1,self.velocity_unique_reads*self.num_envs))
        return result


def install(env,settings=None):
    return GroundContactV2R(env,settings)

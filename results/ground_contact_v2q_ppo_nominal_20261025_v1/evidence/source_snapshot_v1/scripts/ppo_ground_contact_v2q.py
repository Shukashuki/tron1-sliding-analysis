"""Q adds synchronous authored wheel-gap samples to immutable K lifecycle.

K's ownership marker is deliberately reused: Q cannot coexist with K/I owners.
No physical, action, observation, reset, or reference state is written here.
"""
from __future__ import annotations
import hashlib
import importlib
from pathlib import Path
import ppo_ground_contact_v2k as parent
import ground_contact_reward_v2q as definition
from wheel_ground_geometry_torch import WheelGroundGap

PARENT_RUNTIME_SHA256 = '3fb2ce270ed5783101bc558cc615e311dd9f586c3096a28184e9174e3d595f2d'
if hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() != PARENT_RUNTIME_SHA256:
    raise ValueError('Frozen K runtime differs')
require, configure_sensors = parent.require, parent.configure_sensors
MARKER, SUBSTEPS = parent.MARKER, parent.SUBSTEPS
WHEELS, SENSORS, GROUND = parent.WHEELS, parent.SENSORS, parent.GROUND


class GroundContactV2Q(parent.GroundContactV2K):
    def __init__(self, env, settings=None, *, allow_cpu_fixture=False):
        settings = definition.validate_settings(settings)
        # Read-only geometry audit before installing lifecycle wrappers.
        geometry = WheelGroundGap(env, allow_cpu_fixture=allow_cpu_fixture)
        super().__init__(env, allow_cpu_fixture=allow_cpu_fixture)
        self.settings, self.geometry = settings, geometry
        self.geometry_gaps = self.torch.zeros_like(self.forces)
        self.geometry_fault = self.torch.zeros_like(self.fault)
        self.geometry_samples = self.geometry_unique_reads = 0
        self.geometry_cost_sum = self.torch.zeros((), device=self.device)
        self.geometry_positive_windows = self.torch.zeros((), device=self.device, dtype=self.torch.long)

    def step(self, *args, **kwargs):
        self.geometry_gaps.zero_()
        return super().step(*args, **kwargs)

    def update(self, *args, **kwargs):
        before = self.total_samples
        result = super().update(*args, **kwargs)
        if self.total_samples != before:
            require(self.total_samples == before+1, 'Geometry must follow one actual force sample')
            gap = self.geometry.gaps()
            self.geometry_gaps[:, self.samples-1] = gap
            self.geometry_fault.logical_or_(~self.torch.isfinite(gap).all(-1))
            self.geometry_samples += 1
        return result

    def reward(self):
        self._complete()
        require(self.geometry_samples == self.total_samples, 'Missing synchronous wheel geometry sample')
        require(not self.window_reset, 'Reward read after autoreset')
        from height_smoothness import anchor_xy_tracking
        require(importlib.import_module('training.tron1_drift') is self.beta_module
                and self.beta_module.sideslip_tracking is self.beta_function,
                'Existing beta reward implementation changed')
        phase_good = self._phase() == self.held_phase
        anchor = anchor_xy_tracking(self.env, 'motion', .3)
        speed = self.beta_module.speed_tracking(self.env, std=.35)
        beta_quality = self.beta_function(self.env)
        reference_velocity = self.motion_command.body_lin_vel_w
        require(self.torch.is_tensor(reference_velocity) and reference_velocity.ndim == 3
                and reference_velocity.shape[0] == self.num_envs
                and reference_velocity.shape[1] > 0 and reference_velocity.shape[2] == 3
                and reference_velocity.device == self.device
                and reference_velocity.dtype == self.torch.float32,
                'Require actual held-reference body velocities')
        reference_speed = self.torch.linalg.vector_norm(reference_velocity[:, 0, :2], dim=-1)
        phase_good = phase_good & (self._phase() == self.held_phase)
        require(parent.clock(self.env) == self.last_clock, 'Physics advanced while reading reference beta')
        self.phase_fault.logical_or_(~phase_good)
        components = definition.reward_torch(self.forces, self.mg, anchor, speed,
            beta_quality, reference_speed, geometry_gap_m=self.geometry_gaps,
            settings=self.settings, validate=False)
        value = self.torch.where(phase_good, components['raw_reward'],
                                self.torch.full_like(components['raw_reward'], float('nan')))
        self.reward_fault.logical_or_(~self.torch.isfinite(value))
        if self.last_read != self.steps:
            moving, low = reference_speed > .3, reference_speed <= .3
            self.low_reference_gate_fault.logical_or_(low & (components['beta_contact_gate'] != 1.))
            self.beta_counts.add_(self.torch.stack((moving.sum(), low.sum())))
            self.beta_gate_sum.add_(components['beta_contact_gate'].sum())
            self.beta_unique_reads += 1
            self.unique_reads += 1
            self.last_read = self.steps
            self.raw_sum.add_(self.torch.nan_to_num(value).sum())
            cost = components['unsupported_geometry_gap_cost']
            self.geometry_cost_sum.add_(self.torch.nan_to_num(cost).sum())
            self.geometry_positive_windows.add_((cost > 0).sum())
            self.geometry_unique_reads += 1
        return value

    def audit(self, health=False):
        result = super().audit(health=health)
        result.update(reward_definition_version='ground_contact_v2q_any_wheel',
            settings=self.settings, module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            definition_sha256=hashlib.sha256(Path(definition.__file__).read_bytes()).hexdigest(),
            immutable_v2k_definition_sha256=definition.BASE_K_DEFINITION_SHA256,
            immutable_v2k_runtime_sha256=PARENT_RUNTIME_SHA256,
            authored_geometry_audit=self.geometry.audit_result,
            geometry_physics_samples=self.geometry_samples,
            geometry_unique_reward_reads=self.geometry_unique_reads,
            geometry_force_sampling_synchronous=True,
            geometry_gap_scale_m=.02, geometry_gap_scale_is_acceptance_tolerance=False,
            unsupported_geometry_cost_only_below5N=True)
        if health:
            result.update(device_finite_geometry_health=not bool(self.geometry_fault.any().item()),
                mean_unsupported_geometry_gap_cost=float(self.geometry_cost_sum.item())/max(1,self.unique_reads*self.num_envs),
                positive_geometry_cost_environment_windows=int(self.geometry_positive_windows.item()))
        return result


def install(env, settings=None):
    return GroundContactV2Q(env, settings)


def reward_term(weight=.4):
    require(weight == .4, 'Q retains external grounded weight +0.4')
    from isaaclab.managers import ManagerTermBase, RewardTermCfg
    class ContactReward(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.env = env
        def __call__(self, env):
            require(env is self.env, 'Different reward environment')
            owner = getattr(env, MARKER, None)
            require(isinstance(owner, GroundContactV2Q), 'Install synchronous Q collector before stepping')
            return owner.reward()
        def reset(self, env_ids=None):
            owner = getattr(self.env, MARKER, None)
            if owner is not None: owner.reset(env_ids)
    return RewardTermCfg(func=ContactReward, weight=weight)

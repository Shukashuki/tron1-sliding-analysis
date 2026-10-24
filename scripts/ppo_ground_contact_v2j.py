"""J single owner: I lifecycle plus current-start and eight native omegaXY samples.

Only reward-owned observations are added. Actor/critic inputs, actions, physics,
DR, reference beta and the other14 reward terms are unchanged.
"""
from __future__ import annotations
import hashlib
import importlib
import inspect
import math
from numbers import Real
from pathlib import Path
import ppo_ground_contact_v2i as parent
import ground_contact_reward_v2j as definition
from ground_contact_angular_increment_contract import (
    ANGULAR_SOURCE, SOURCE_PROPERTIES, verify_angular_data_source, verify_angular_source_file)

PARENT_COLLECTOR_SHA256 = "ef31c3ee196c96d3f2b8511ffe65b56ea3c21349d70dddfaff870680a1a73415"
require, clock = parent.require, parent.clock
configure_sensors, reward_term = parent.configure_sensors, parent.reward_term
MARKER, SENSORS, WHEELS, GROUND = parent.MARKER, parent.SENSORS, parent.WHEELS, parent.GROUND
SUBSTEPS, PHYSICS_DT, POLICY_DT = parent.SUBSTEPS, parent.PHYSICS_DT, parent.POLICY_DT

class GroundContactV2J(parent.GroundContactV2I):
    def __init__(self, env, settings=None, *, allow_cpu_fixture=False):
        require(hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() == PARENT_COLLECTOR_SHA256,
                "Frozen I collector source differs")
        definition.verify_base()
        self.j_settings = definition.validate_settings(settings)
        self.torch = importlib.import_module("torch")
        self.env, self.robot = env, env.scene["robot"]
        self.device, self.num_envs = self.torch.device(env.device), int(env.num_envs)
        require((self.device.type == "cuda" and self.device.index is not None)
                or (allow_cpu_fixture and self.device.type == "cpu"), "Explicit native device required")
        self.angular_data = self.robot.data
        self.angular_class = type(self.angular_data)
        self.angular_descriptors = tuple(inspect.getattr_static(self.angular_class, n, None) for n in SOURCE_PROPERTIES)
        self.angular_source_info = None if self.device.type == "cpu" else verify_angular_data_source(self.angular_data)
        self._angular_velocity()  # Validate before any lifecycle attachment.
        self.angular = self.torch.zeros((self.num_envs, 9, 2), device=self.device)
        self.angular_fault = self.torch.zeros(self.num_envs, device=self.device, dtype=self.torch.bool)
        self.angular_samples = self.angular_starts = self.angular_unique_reads = 0
        self.angular_last_clock = self.angular_start_clock = None
        self.angular_penalty_sum = self.torch.zeros((), device=self.device)
        super().__init__(env, allow_cpu_fixture=allow_cpu_fixture)

    def _angular_epoch(self):
        value = self.angular_data._sim_timestamp
        require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)),
                "Native angular cache epoch invalid")
        return float(value)

    def _angular_velocity(self):
        require(self.env.scene["robot"] is self.robot and self.robot.data is self.angular_data
                and type(self.angular_data) is self.angular_class, "Native angular data identity changed")
        require(all(inspect.getattr_static(self.angular_class, n, None) is d
                    for n, d in zip(SOURCE_PROPERTIES, self.angular_descriptors)), "Native angular getter changed")
        value = self.angular_data.root_com_ang_vel_w
        require(self.torch.is_tensor(value) and tuple(value.shape) == (self.num_envs, 3)
                and value.device == self.device and value.dtype == self.torch.float32,
                "Require native float32 world angular velocity[N,3]")
        cache = self.angular_data._root_com_vel_w.timestamp
        require(isinstance(cache, Real) and not isinstance(cache, bool) and math.isfinite(float(cache))
                and float(cache) == self._angular_epoch(), "Native angular cache stale")
        return value[:, :2]

    def step(self, *args, **kwargs):
        self._attached()
        require(not self.inside_step and not self.inside_update, "Reentrant angular window")
        before = clock(self.env)
        start = self._angular_velocity()
        require(clock(self.env) == before, "Physics advanced during start angular read")
        self.angular.zero_()
        self.angular[:, 0].copy_(start)
        self.angular_fault.logical_or_(~self.torch.isfinite(start).all(-1))
        self.angular_start_clock = before
        self.angular_starts += 1
        return super().step(*args, **kwargs)

    def update(self, *args, **kwargs):
        before, epoch = self.total_samples, self._angular_epoch()
        result = super().update(*args, **kwargs)
        if self.total_samples == before:
            return result
        require(self.total_samples == before+1 and self.inside_step and 1 <= self.samples <= 8,
                "Angular sampling cannot invent a physical step")
        after = self._angular_epoch()
        tolerance = self.dt_rounding+8*math.ulp(max(abs(epoch), abs(after), PHYSICS_DT))
        require(abs(after-epoch-PHYSICS_DT) <= tolerance, "Angular epoch did not advance with actual physical refresh")
        value = self._angular_velocity()
        require(clock(self.env) == self.last_clock, "Physics advanced during angular sampling")
        self.angular[:, self.samples].copy_(value)
        self.angular_fault.logical_or_(~self.torch.isfinite(value).all(-1))
        self.angular_samples += 1
        self.angular_last_clock = self.last_clock
        return result

    def reward(self):
        self._complete()
        require(self.angular_samples == self.total_samples and self.angular_last_clock == self.last_clock
                and self.angular_start_clock == self.start_clock and self.angular_starts == self.steps,
                "J reward requires the true start and8 paired native angular samples")
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
                                             beta_quality, reference_speed, self.angular,
                                             settings=self.j_settings, validate=False)
        value = self.torch.where(phase_good, components["raw_reward"],
                                self.torch.full_like(components["raw_reward"], float("nan")))
        self.reward_fault.logical_or_(~self.torch.isfinite(value))
        if self.last_read != self.steps:
            moving, low = reference_speed > .3, reference_speed <= .3
            self.low_reference_gate_fault.logical_or_(low & (components["beta_contact_gate"] != 1.))
            self.beta_counts.add_(self.torch.stack((moving.sum(), low.sum())))
            self.beta_gate_sum.add_(components["beta_contact_gate"].sum())
            self.beta_unique_reads += 1
            self.angular_unique_reads += 1
            self.angular_penalty_sum.add_(self.torch.nan_to_num(components["loaded_angular_increment_penalty"]).sum())
            self.unique_reads += 1
            self.last_read = self.steps
            self.raw_sum.add_(self.torch.nan_to_num(value).sum())
        return value

    def audit(self, health=False):
        result = super().audit(health=health)
        result.update(reward_definition_version="ground_contact_v2j_any_wheel", settings=self.j_settings,
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            definition_sha256=hashlib.sha256(Path(definition.__file__).read_bytes()).hexdigest(),
            immutable_v2i_definition_sha256=definition.BASE_I_DEFINITION_SHA256,
            immutable_v2i_collector_sha256=PARENT_COLLECTOR_SHA256,
            angular_source=ANGULAR_SOURCE,
            angular_source_path=None if self.angular_source_info is None else self.angular_source_info["path"],
            angular_source_sha256=None if self.angular_source_info is None else self.angular_source_info["sha256"],
            angular_source_verified=self.angular_source_info is not None,
            angular_signal="root_COM_worldXY_rad_s", angular_num_envs=self.num_envs,
            angular_all8_actual_substeps=True, angular_start_is_current_physical_state=True,
            angular_velocity_is_pose_finite_difference=False, angular_end_state_repeated=False,
            angular_cache_same_refresh_not_absolute_time=True,
            angular_physics_samples=self.angular_samples, angular_boundary_samples=self.angular_starts,
            angular_environment_physics_samples=self.angular_samples*self.num_envs,
            angular_environment_boundary_samples=self.angular_starts*self.num_envs,
            angular_unique_reward_windows=self.angular_unique_reads, base_owner_marker=MARKER)
        if health:
            if self.angular_source_info is not None:
                require(verify_angular_source_file(self.angular_source_info["path"]) == self.angular_source_info,
                        "Native angular source changed during execution")
            result.update(angular_cache_refresh_health=self.angular_samples == self.total_samples
                and self.angular_samples > 0 and self.angular_last_clock == self.last_clock
                and self.angular_starts == self.completed_steps,
                device_finite_angular_health=not bool(self.angular_fault.any().item()),
                mean_added_angular_penalty=float(self.angular_penalty_sum.item())/
                    max(1, self.angular_unique_reads*self.num_envs))
        return result

def install(env, settings=None):
    return GroundContactV2J(env, settings)

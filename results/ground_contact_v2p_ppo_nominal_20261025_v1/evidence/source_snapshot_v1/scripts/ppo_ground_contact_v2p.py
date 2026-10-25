"""P retains K support shaping; adds an independent bounded peak-impact cost.

Same8 native samples, original3mg threshold, unchanged actors/references/clocks
and other14 rewards. No outcome claim before completed dual-engine validation.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
import struct
from types import MethodType

import ground_contact_reward_v2p as definition
from ground_contact_beta_contract import BETA_QUALITY_SOURCE, verify_beta_source

PHYSICS_DT = .0025
POLICY_DT = .02
SUBSTEPS = 8
WHEELS = ("wheel_L_Link", "wheel_R_Link")
SENSORS = ("wheel_L_ground_v2", "wheel_R_ground_v2")
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
MARKER = "_tron1_ground_contact_v2p"
_MISSING = object()


def require(value, message):
    if not value:
        raise ValueError(message)


def configure_sensors(scene):
    from isaaclab.sensors import ContactSensorCfg
    for name, body in zip(SENSORS, WHEELS):
        require(getattr(scene, name, None) is None, "Do not overwrite an existing sensor")
        setattr(scene, name, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/"+body, update_period=0., history_length=0,
            filter_prim_paths_expr=[GROUND], track_contact_points=False,
            track_air_time=False, track_pose=False, max_contact_data_count_per_prim=64,
            debug_vis=False))


def clock(env):
    require(env.cfg.decimation == SUBSTEPS, "Require eight actual physical substeps")
    for value in (env.physics_dt, env.cfg.sim.dt, env.sim.get_physics_dt()):
        require(math.isclose(float(value), PHYSICS_DT, rel_tol=0, abs_tol=1e-12),
                "Require native 2.5 ms physics, not evaluation relabeling")
    require(math.isclose(float(env.step_dt), POLICY_DT, rel_tol=0, abs_tol=1e-12),
            "Require unchanged 50 Hz reward/action clock")
    return (int(env._sim_step_counter), int(env.sim.current_time_step_index), float(env.sim.current_time))


def _restore(owner, name, value):
    if value is _MISSING:
        delattr(owner, name)
    else:
        setattr(owner, name, value)


class GroundContactV2P:
    def __init__(self, env, settings=None, *, allow_cpu_fixture=False):
        self.settings = definition.validate_settings(settings)
        self.env, self.scene, self.robot = env, env.scene, env.scene["robot"]
        self.torch = importlib.import_module("torch")
        self.device = self.torch.device(env.device)
        require((self.device.type == "cuda" and self.device.index is not None)
                or (allow_cpu_fixture and self.device.type == "cpu"),
                "Production collector requires an explicit CUDA device")
        self.cpu_fixture = self.device.type == "cpu"
        self.num_envs = int(env.num_envs)
        require(self.num_envs > 0 and not hasattr(env, MARKER) and not hasattr(env.scene, MARKER)
                and all(not hasattr(owner, '_tron1_ground_contact_v2'+name)
                    for owner in (env, env.scene) for name in ('k', 'l', 'm', 'n', 'o'))
                and not hasattr(env, "_tron1_ground_contact_v2i")
                and not hasattr(env.scene, "_tron1_ground_contact_v2i")
                and not hasattr(env, "_tron1_ground_contact_v2j")
                and not hasattr(env.scene, "_tron1_ground_contact_v2j")
                and not hasattr(env, "_tron1_ground_contact_v2")
                and not hasattr(env.scene, "_tron1_ground_contact_v2")
                and not hasattr(env, "_tron1_ground_contact_v2b")
                and not hasattr(env.scene, "_tron1_ground_contact_v2b")
                and not hasattr(env, "_tron1_ground_contact_v2c")
                and not hasattr(env.scene, "_tron1_ground_contact_v2c")
                and not hasattr(env, "_tron1_ground_contact_v2d")
                and not hasattr(env.scene, "_tron1_ground_contact_v2d")
                and not hasattr(env, "_tron1_ground_contact_v2e")
                and not hasattr(env.scene, "_tron1_ground_contact_v2e")
                and not hasattr(env, "_tron1_ground_contact_v2f")
                and not hasattr(env.scene, "_tron1_ground_contact_v2f")
                and not hasattr(env, "_tron1_ground_contact_v2g")
                and not hasattr(env.scene, "_tron1_ground_contact_v2g"),
                "Require one fresh collector owner")
        self.motion_command = env.command_manager.get_term("motion")
        self.beta_module = importlib.import_module("training.tron1_drift")
        self.beta_function = self.beta_module.sideslip_tracking
        self.beta_source = (None if self.cpu_fixture else verify_beta_source(self.beta_module.__file__))
        self.held_phase = self._phase().clone()
        self.phase_fault = self.torch.zeros(self.num_envs, device=self.device, dtype=self.torch.bool)
        self.low_reference_gate_fault = self.torch.zeros_like(self.phase_fault)
        self.beta_counts = self.torch.zeros(2, device=self.device, dtype=self.torch.long)
        self.beta_gate_sum = self.torch.zeros((), device=self.device)
        self.beta_unique_reads = 0
        self.last_clock = clock(env)
        self.sensors = tuple(env.scene.sensors.get(name) for name in SENSORS)
        require(all(sensor is not None for sensor in self.sensors), "Ground sensors must be configured first")
        if not self.cpu_fixture:
            import omni.usd
            from pxr import UsdPhysics
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(GROUND)
            require(prim and prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI),
                    "Actual floor collision prim is missing")
            require(env.cfg.scene.terrain.terrain_type == "plane", "This reward is verified for flat floor only")
        gravity = tuple(float(v) for v in env.cfg.sim.gravity)
        require(gravity[0] == gravity[1] == 0. and gravity[2] < 0., "Require downward world-Z gravity")
        masses = self.robot.root_physx_view.get_masses().detach().clone()
        require(tuple(masses.shape) == (self.num_envs, len(self.robot.body_names)), "Unexpected live mass array")
        require(bool(self.torch.isfinite(masses).all()) and bool((masses >= 0).all()), "Invalid live body masses")
        self.mg = (masses.sum(-1)*abs(gravity[2])).to(device=self.device, dtype=self.torch.float32)
        require(bool((self.mg > 0).all()) and bool(self.torch.isfinite(self.mg).all()), "Invalid live weight")
        self.mass_range = [float(masses.sum(-1).min()), float(masses.sum(-1).max())]
        target = self.mg*self.settings["support_load_fraction"]
        require(bool((target > self.settings["support_force_threshold_n"]).all()),
                "K requires actual .5mg load target above5N")
        self.load_margin_target_range_n = [float(target.min()), float(target.max())]
        self.load_margin_gravity_w_m_s2 = list(gravity)
        self.forces = self.torch.zeros((self.num_envs, SUBSTEPS, 2), device=self.device)
        self.fault = self.torch.zeros(self.num_envs, device=self.device, dtype=self.torch.bool)
        self.reward_fault = self.torch.zeros_like(self.fault)
        self.bearing = self.torch.zeros(4, device=self.device, dtype=self.torch.long)
        self.force_min = self.torch.full((2,), float("inf"), device=self.device)
        self.force_max = self.torch.full((2,), -float("inf"), device=self.device)
        self.raw_sum = self.torch.zeros((), device=self.device)
        self.peak_impact_cost_sum = self.torch.zeros((), device=self.device)
        self.peak_impact_positive_windows = self.torch.zeros((), device=self.device, dtype=self.torch.long)
        self.peak_impact_fault = self.torch.zeros_like(self.fault)
        self.steps = self.completed_steps = self.total_samples = self.unique_reads = self.reset_calls = 0
        self.samples = 0
        self.last_read = None
        self.inside_step = self.inside_update = self.window_reset = self.closed = False
        self.dt_rounding = abs(struct.unpack("f", struct.pack("f", PHYSICS_DT))[0]-PHYSICS_DT)
        self.original_step, self.original_update = env.step, env.scene.update
        self.previous_step, self.previous_update = vars(env).get("step", _MISSING), vars(env.scene).get("update", _MISSING)
        # Validate sensors before mutating any environment lifecycle owner.
        self._matrix(0)
        self._matrix(1)
        self.step_wrapper = MethodType(lambda owner, *a, **k: self.step(*a, **k), env)
        self.update_wrapper = MethodType(lambda owner, *a, **k: self.update(*a, **k), env.scene)
        env.step, env.scene.update = self.step_wrapper, self.update_wrapper
        setattr(env, MARKER, self)
        setattr(env.scene, MARKER, self)

    def _phase(self):
        require(self.env.command_manager.get_term("motion") is self.motion_command,
                "Held motion command identity changed")
        phase = self.motion_command.time_steps
        require(self.torch.is_tensor(phase) and tuple(phase.shape) == (self.num_envs,)
                and phase.device == self.device and phase.dtype == self.torch.long,
                "Require per-environment native reference time_steps")
        return phase

    def _attached(self):
        require(not self.closed and getattr(self.env, MARKER, None) is self
                and getattr(self.scene, MARKER, None) is self
                and self.env.step is self.step_wrapper and self.scene.update is self.update_wrapper,
                "Ground-contact owner/lifecycle changed")

    def _matrix(self, side):
        sensor = self.sensors[side]
        require(self.scene.sensors.get(SENSORS[side]) is sensor, "Ground sensor replaced")
        require(tuple(sensor.body_names) == (WHEELS[side],)
                and tuple(sensor.cfg.filter_prim_paths_expr) == (GROUND,)
                and sensor.cfg.update_period == 0. and sensor.cfg.history_length == 0,
                "Wheel/floor filter identity differs")
        require(math.isclose(float(sensor._sim_physics_dt), PHYSICS_DT, rel_tol=0, abs_tol=1e-12),
                "Contact impulses must convert using 2.5 ms, not 20 ms")
        value = sensor.data.force_matrix_w
        require(value is not None and tuple(value.shape) == (self.num_envs, 1, 1, 3)
                and value.device == self.device and value.dtype == self.torch.float32,
                "Expected GPU float32 ground-filtered normal force[N,1,1,3]")
        return value[:, 0, 0, 2]

    def step(self, *args, **kwargs):
        self._attached()
        require(not self.inside_step, "Reentrant control step")
        self.held_phase.copy_(self._phase())
        self.start_clock = self.last_clock = clock(self.env)
        self.samples = 0
        self.forces.zero_()
        self.window_reset = False
        self.steps += 1
        self.inside_step = True
        try:
            result = self.original_step(*args, **kwargs)
            self._complete()
            require(self.last_read == self.steps, "New reward must be consumed before reset/end")
            self.completed_steps += 1
            return result
        finally:
            self.inside_step = False

    def update(self, *args, **kwargs):
        self._attached()
        require(not self.inside_update, "Reentrant scene update")
        self.inside_update = True
        try:
            result = self.original_update(*args, **kwargs)
            if not self.inside_step:
                return result
            current = clock(self.env)
            if current[0] == self.last_clock[0]:
                require(current == self.last_clock, "Native clock advanced without physical counter")
                return result
            require(self.samples < SUBSTEPS and current[0] == self.last_clock[0]+1
                    and current[1] == self.last_clock[1]+1, "Need eight consecutive actual samples")
            tolerance = self.dt_rounding+8*math.ulp(max(abs(current[2]), abs(self.last_clock[2]), PHYSICS_DT))
            require(abs(current[2]-self.last_clock[2]-PHYSICS_DT) <= tolerance, "Native physics time differs")
            require(not (args and "dt" in kwargs), "Duplicate scene dt")
            dt = args[0] if args else kwargs.get("dt")
            require(math.isclose(float(dt), PHYSICS_DT, rel_tol=0, abs_tol=1e-12), "Only physical refreshes contribute")
            fz = self.torch.stack((self._matrix(0), self._matrix(1)), dim=-1)
            self.forces[:, self.samples] = fz
            valid = self.torch.isfinite(fz).all(-1) & (fz >= -1e-4).all(-1)
            self.fault.logical_or_(~valid)
            safe = self.torch.nan_to_num(fz, nan=0., posinf=0., neginf=0.)
            load = safe > 5.
            self.bearing.add_(self.torch.stack((load[:, 0].sum(), load[:, 1].sum(), load.any(-1).sum(), load.all(-1).sum())))
            self.force_min.copy_(self.torch.minimum(self.force_min, safe.amin(0)))
            self.force_max.copy_(self.torch.maximum(self.force_max, safe.amax(0)))
            self.samples += 1
            self.total_samples += 1
            self.last_clock = current
            return result
        finally:
            self.inside_update = False

    def _complete(self):
        self._attached()
        require(self.inside_step and self.samples == SUBSTEPS and clock(self.env) == self.last_clock
                and self.last_clock[0] == self.start_clock[0]+SUBSTEPS
                and self.last_clock[1] == self.start_clock[1]+SUBSTEPS,
                "Reward requires all eight actual physics samples")

    def reward(self):
        self._complete()
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
                                             beta_quality, reference_speed,
                                             settings=self.settings, validate=False)
        cost = components['ungated_peak_impact_cost']
        self.peak_impact_fault.logical_or_(~self.torch.isfinite(cost) | (cost < 0.) | (cost > 1.))
        value = self.torch.where(phase_good, components["raw_reward"],
                                self.torch.full_like(components["raw_reward"], float("nan")))
        self.reward_fault.logical_or_(~self.torch.isfinite(value))
        if self.last_read != self.steps:
            moving, low = reference_speed > .3, reference_speed <= .3
            self.low_reference_gate_fault.logical_or_(low & (components["beta_contact_gate"] != 1.))
            self.beta_counts.add_(self.torch.stack((moving.sum(), low.sum())))
            self.beta_gate_sum.add_(components["beta_contact_gate"].sum())
            self.beta_unique_reads += 1
            self.unique_reads += 1
            self.last_read = self.steps
            self.raw_sum.add_(self.torch.nan_to_num(value).sum())
            self.peak_impact_cost_sum.add_(self.torch.nan_to_num(cost).sum())
            self.peak_impact_positive_windows.add_((cost > 0.).sum())
        return value

    def reset(self, env_ids=None):
        self._attached()
        if self.inside_step:
            self._complete()
            require(self.last_read == self.steps, "Autoreset occurred before reward")
            self.window_reset = True
        self.reset_calls += 1

    def audit(self, health=False):
        result = {"reward_definition_version": "ground_contact_v2p_any_wheel",
            "settings": self.settings, "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "definition_sha256": hashlib.sha256(Path(definition.__file__).read_bytes()).hexdigest(),
            "immutable_v2a_definition_sha256": definition.base.BASE_DEFINITION_SHA256,
            "immutable_v2b_definition_sha256": definition.base.BASE_B_DEFINITION_SHA256,
            "immutable_v2c_definition_sha256": definition.base.BASE_C_DEFINITION_SHA256,
            "immutable_v2d_definition_sha256": definition.base.BASE_D_DEFINITION_SHA256,
            "immutable_v2f_definition_sha256": definition.BASE_F_DEFINITION_SHA256,
            "immutable_v2g_definition_sha256": definition.BASE_G_DEFINITION_SHA256,
            "support_gap_aggregation": "max8",
            "support_gap_is_window_event_not_duration_fraction": False,
            "support_gap_is_continuous_load_margin": True,
            "support_gap_cost_mode": self.settings["support_gap_cost_mode"],
            "immutable_v2i_definition_sha256": definition.BASE_I_DEFINITION_SHA256,
            "load_margin_target_range_n": self.load_margin_target_range_n,
            "load_margin_gravity_w_m_s2": self.load_margin_gravity_w_m_s2,
            "beta_quality_source": BETA_QUALITY_SOURCE,
            "beta_quality_source_sha256": None if self.beta_source is None else self.beta_source["sha256"],
            "beta_quality_source_path": None if self.beta_source is None else self.beta_source["path"],
            "beta_quality_source_verified": self.beta_source is not None,
            "beta_quality_std_rad": .12, "beta_low_reference_speed_m_s": .3,
            "beta_num_envs": self.num_envs, "beta_unique_reward_reads": self.beta_unique_reads,
            "beta_quality_is_unweighted_existing_term": True,
            "reference_beta_not_zero_beta_is_intended": True,
            "physics_dt_s": PHYSICS_DT, "policy_dt_s": POLICY_DT, "substeps": SUBSTEPS,
            "device": str(self.device), "cpu_fixture_only": self.cpu_fixture,
            "floor_filter": GROUND, "force_units": "normal force N; impulse converted with physical dt2.5ms",
            "mass_kg_range": self.mass_range, "steps_started": self.steps,
            "steps_completed": self.completed_steps, "physics_samples": self.total_samples,
            "unique_reward_windows": self.unique_reads, "reset_calls": self.reset_calls,
            "single_wheel_support_allowed": True, "bilateral_constraint": False,
            "height_or_zero_slip_target": False, "observation_action_or_physics_change": False,
            "immutable_v2k_definition_sha256": definition.BASE_K_DEFINITION_SHA256,
            "ungated_peak_impact_weight": 1., "ungated_peak_impact_aggregation": "max8_maxwheel",
            "peak_impact_tracking_or_support_gated": False,
            "closed": self.closed}
        if health:
            healthy = not bool(self.fault.any().item())
            result.update(
                device_held_reference_phase_health=not bool(self.phase_fault.any().item()),
                device_low_reference_speed_gate_health=not bool(self.low_reference_gate_fault.any().item()),
                beta_moving_reference_samples=int(self.beta_counts[0].item()),
                beta_low_reference_samples=int(self.beta_counts[1].item()),
                beta_environment_reward_samples=self.num_envs*self.beta_unique_reads,
                beta_gate_mean=float(self.beta_gate_sum.item())/max(1, self.num_envs*self.beta_unique_reads),
                device_finite_normal_force_health=healthy,
                device_finite_bounded_peak_impact_health=not bool(self.peak_impact_fault.any().item()),
                mean_ungated_peak_impact_cost=float(self.peak_impact_cost_sum.item())/max(1, self.unique_reads*self.num_envs),
                positive_peak_impact_environment_windows=int(self.peak_impact_positive_windows.item()),
                device_finite_reward_health=not bool(self.reward_fault.any().item()),
                ground_fz_min_N=self.force_min.cpu().tolist(), ground_fz_max_N=self.force_max.cpu().tolist(),
                force_bearing_samples_L_R_any_both=self.bearing.cpu().tolist(),
                sampled_environment_substeps=self.total_samples*self.num_envs,
                mean_raw_reward=float(self.raw_sum.item())/max(1, self.unique_reads*self.num_envs))
        return result

    def close(self):
        if self.closed:
            return self.audit()
        self._attached()
        require(not self.inside_step and not self.inside_update, "Do not detach within a step")
        _restore(self.env, "step", self.previous_step)
        _restore(self.scene, "update", self.previous_update)
        delattr(self.env, MARKER)
        delattr(self.scene, MARKER)
        self.closed = True
        return self.audit()


def install(env, settings=None):
    return GroundContactV2P(env, settings)


def reward_term(weight=.4):
    require(weight == .4, "V2k replaces only the definition; grounded weight stays +0.4")
    from isaaclab.managers import ManagerTermBase, RewardTermCfg
    class ContactReward(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.env = env

        def __call__(self, env):
            require(env is self.env, "Different reward environment")
            owner = getattr(env, MARKER, None)
            require(isinstance(owner, GroundContactV2P), "Install the eight-substep owner before stepping")
            return owner.reward()

        def reset(self, env_ids=None):
            owner = getattr(self.env, MARKER, None)
            if owner is not None:
                owner.reset(env_ids)
    return RewardTermCfg(func=ContactReward, weight=weight)

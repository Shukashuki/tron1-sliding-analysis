"""Prepared GPU-local mean-of-four wheel/ground support reward.

Raw reward = 1 - mean_(four physics samples, two wheels)
exp(-clamp(F_ground_normal_world_z, min=0)/(0.20*mg)). External weight is +0.4.
This is a bounded contact-shaping change, not an absolute-height constraint or
a zero-sideslip objective. Continuous contact/stability is not certified.

Isaac/Torch imports occur only after explicit configuration/installation. The
installed 2.3/PhysX sensor contract needs an actual loaded-contact Isaac smoke
before training adoption: CPU fixtures cannot verify force direction or units.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import math
from pathlib import Path
from types import MethodType

import ppo_leg_torque_smoothness as clock


WHEEL_NAMES = ("wheel_L_Link", "wheel_R_Link")
SENSOR_NAMES = ("wheel_L_ground", "wheel_R_ground")
GROUND_COLLIDER = "/World/ground/terrain/GroundPlane/CollisionPlane"
ENV_MARKER = "_tron1_substep_ground_support_accumulator"
SCENE_MARKER = "_tron1_substep_ground_support_owner"
CONFLICT_MARKERS = (clock.ENV_MARKER, clock.SCENE_MARKER, *clock.CONFLICT_MARKERS,
                    "_tron1_ground_load_reserve")
SCALE_FRACTION = .20
NEGATIVE_FORCE_TOLERANCE_N = 1.e-4
POSITIVE_FORCE_TELEMETRY_N = 1.e-5
CLOCK_HELPER_SHA256 = "2649dc39f08f8b540430617f7a2c8182527d895b62d5c515672e66c7c3e03aa7"
ATTRIBUTED_REFERENCE_SOURCES = {
    "training/tron1_ground_load_reserve.py": "761eccede3c0d90d744a4d9c45c9671b318426ab9bc6f9762ee7cacb3c14496d",
    "training/tron1_substep_rewards.py": "aeec31437226984a326a263109aaadebca004a6ef333f25a1d3377688aa276a5",
    "training/tron1_ground_load_substeps.py": "4df069ea80e37cf7575346201687be934927434db45ea143f5ce797d4979ecd2",
}
require = clock.require


def _scale(value):
    require(clock._number(value, "scale_fraction") == SCALE_FRACTION,
            "This prepared support reward fixes scale_fraction=0.20")
    return SCALE_FRACTION


def configure_ground_support_sensors(scene_cfg):
    """Two single-body/one-ground filters; no point-recording CPU path."""
    require(all(getattr(scene_cfg, name, None) is None for name in SENSOR_NAMES),
            "Refusing to overwrite wheel-ground sensors")
    from isaaclab.sensors import ContactSensorCfg
    values = [ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/"+body, update_period=0., history_length=0,
        filter_prim_paths_expr=[GROUND_COLLIDER], track_contact_points=False,
        track_air_time=False, track_pose=False, max_contact_data_count_per_prim=32,
        debug_vis=False) for body in WHEEL_NAMES]
    for name, value in zip(SENSOR_NAMES, values):
        setattr(scene_cfg, name, value)


def _native_gravity():
    manager = importlib.import_module("isaacsim.core.simulation_manager").SimulationManager
    return manager.get_physics_sim_view().get_gravity()


def _ground_audit():
    import omni.usd
    from pxr import UsdPhysics
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(GROUND_COLLIDER)
    require(prim and prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI)
            and prim.GetTypeName() in ("Plane", "Mesh"),
            "Ground filter must resolve to the declared actual flat-ground collider")
    return {"path": GROUND_COLLIDER, "usd_type": prim.GetTypeName(),
            "collision_api_verified": True, "loaded_normal_direction_runtime_smoke_required": True}


def _gravity(value):
    require(len(value) == 3, "Gravity must have three components")
    result = tuple(clock._number(v, "gravity") for v in value)
    require(result[0] == result[1] == 0. and result[2] < 0.,
            "Support normalization requires fixed downward world-Z gravity")
    return result


def _event_audit(env):
    """Allow exactly the inspected historical BeyondMimic recipe, not mass DR."""
    tracking = importlib.import_module("training.tron1_tracking")
    mdp = tracking.mdp
    expected = {
        "physics_material": ("startup", mdp.randomize_rigid_body_material),
        "legacy_joint_friction": ("startup", tracking.clear_legacy_joint_friction),
        "add_joint_default_pos": ("startup", mdp.randomize_joint_default_pos),
        "base_com": ("startup", mdp.randomize_rigid_body_com),
        "push_robot": ("interval", mdp.push_by_setting_velocity)}
    audit, seen = [], set()
    for mode, names in env.event_manager.active_terms.items():
        for name in names:
            require(name in expected and name not in seen, "Uninspected or duplicate event: "+name)
            term = env.event_manager.get_term_cfg(name)
            expected_mode, function = expected[name]
            identity = term.func is function or (isinstance(function, type) and type(term.func) is function)
            require(mode == expected_mode and term.mode == expected_mode and identity,
                    "Event identity/mode differs: "+name)
            seen.add(name)
            audit.append({"name": name, "mode": mode,
                          "function": function.__module__+"."+function.__name__})
    require(seen == set(expected), "Missing historical BeyondMimic event terms")
    return audit


class GroundSupportAccumulator:
    """One current-window GPU statistic; no reward history crosses a reset."""
    def __init__(self, env, *, scale_fraction=SCALE_FRACTION, _allow_cpu_fixture=False):
        require(hashlib.sha256(Path(clock.__file__).read_bytes()).hexdigest() == CLOCK_HELPER_SHA256,
                "Frozen clock/lifecycle helper changed")
        clock._clock(env)
        require(not any(hasattr(owner, name) for owner in (env, env.scene)
                        for name in (ENV_MARKER, SCENE_MARKER, *CONFLICT_MARKERS)),
                "Conflicting strict substep accumulator/cache owner")
        self.env, self.scene, self.robot = env, env.scene, env.scene["robot"]
        self.torch = importlib.import_module("torch")
        self.device, self.dtype = self.torch.device(env.device), self.torch.float32
        require(self.device.type == "cuda" and self.device.index is not None or
                (_allow_cpu_fixture and self.device.type == "cpu"),
                "Production ground support requires an explicit CUDA device")
        self.cpu_fixture = _allow_cpu_fixture and self.device.type == "cpu"
        self.num_envs = clock._integer(env.num_envs, "environment count")
        require(self.num_envs > 0, "Need a nonempty vector environment")
        self.scale_fraction = _scale(scale_fraction)
        self.sensors = tuple(self.scene.sensors.get(name) for name in SENSOR_NAMES)
        require(all(sensor is not None for sensor in self.sensors), "Configure both ground sensors before env construction")
        self.event_audit, self.ground_audit = _event_audit(env), _ground_audit()
        self.gravity = _gravity(env.cfg.sim.gravity)
        native_gravity = _gravity(_native_gravity())
        require(all(math.isclose(a, b, rel_tol=1.e-6, abs_tol=1.e-7)
                    for a, b in zip(self.gravity, native_gravity)), "Native and configured gravity differ")
        masses = self.robot.root_physx_view.get_masses()
        require(masses.device.type == "cpu" and tuple(masses.shape) == (self.num_envs, len(self.robot.body_names))
                and self.torch.is_floating_point(masses), "Live masses must be CPU floating [N,bodies]")
        mass = masses.detach().to(dtype=self.torch.float64).clone()
        require(bool(self.torch.isfinite(mass).all()) and bool((mass >= 0.).all()), "Invalid live body masses")
        total = mass.sum(dim=1)
        require(bool((total > 0.).all()), "Every environment needs positive total mass")
        self.mass_range = [float(total.min()), float(total.max())]
        mg = total*abs(native_gravity[2])
        self.mg_range = [float(mg.min()), float(mg.max())]
        self.mg = mg.to(device=self.device, dtype=self.dtype)
        require(bool(self.torch.isfinite(self.mg).all()) and bool((self.mg > 0.).all()), "Invalid cached mg")
        legs = self.robot.actuators["legs"]
        kp, kd = legs.stiffness, legs.damping
        require(bool(self.torch.isfinite(kp).all()) and bool(self.torch.isfinite(kd).all())
                and bool((kp == 500.).all()) and bool((kd == 10.).all()),
                "Contact-only experiment preserves actual Kp500/Kd10")
        self.gain_readback = {"kp_range": [float(kp.min()), float(kp.max())],
                              "kd_range": [float(kd.min()), float(kd.max())]}
        self.fault = self.torch.zeros(self.num_envs, device=self.device, dtype=self.torch.bool)
        self.sum_cost = self.torch.zeros(self.num_envs, device=self.device)
        self.cached_reward = self.torch.zeros_like(self.sum_cost)
        self.positive_samples = self.torch.zeros(2, device=self.device, dtype=self.torch.long)
        self.negative_samples = self.torch.zeros((), device=self.device, dtype=self.torch.long)
        self.min_force = self.torch.full((2,), float("inf"), device=self.device)
        self.max_force = self.torch.full((2,), -float("inf"), device=self.device)
        self.force_xy_peak = self.torch.zeros((), device=self.device)
        for body, sensor in zip(WHEEL_NAMES, self.sensors):
            matrix = self._matrix(sensor, body)
            require(bool(self.torch.isfinite(matrix).all()) and
                    not bool((matrix[..., 2] < -NEGATIVE_FORCE_TOLERANCE_N).any()),
                    "Invalid startup filtered normal force; loaded-contact runtime smoke remains required")
        self.closed = self.inside_step = self.inside_update = self.window_reset = False
        self.samples = self.total_samples = self.steps_started = self.steps_completed = self.steps_failed = 0
        self.cost_reads = self.unique_cost_windows = self.reset_calls = 0
        self.last_cost_window = None
        self.last_clock = clock._native_clock(env)
        import struct
        self.dt_rounding = abs(struct.unpack("f", struct.pack("f", .005))[0]-.005)
        self.original_step, self.original_update = env.step, self.scene.update
        self.previous_step, self.previous_update = vars(env).get("step", clock._MISSING), vars(self.scene).get("update", clock._MISSING)
        def step(owner, *args, **kwargs):
            require(owner is self.env, "Wrong step owner")
            return self._step(*args, **kwargs)
        def update(owner, *args, **kwargs):
            require(owner is self.scene, "Wrong scene owner")
            return self._update(*args, **kwargs)
        self.step_wrapper, self.update_wrapper = MethodType(step, env), MethodType(update, self.scene)
        env.step, self.scene.update = self.step_wrapper, self.update_wrapper
        setattr(env, ENV_MARKER, self); setattr(self.scene, SCENE_MARKER, self)

    def _matrix(self, sensor, body):
        cfg, view = sensor.cfg, sensor.contact_physx_view
        require(self.scene.sensors.get(SENSOR_NAMES[WHEEL_NAMES.index(body)]) is sensor,
                "Ground sensor replaced after installation")
        require(cfg.update_period == 0. and cfg.history_length == 0 and not cfg.track_contact_points
                and not cfg.track_air_time and not cfg.track_pose
                and tuple(cfg.filter_prim_paths_expr) == (GROUND_COLLIDER,), "Unexpected ground-sensor contract")
        require(tuple(sensor.body_names) == (body,) and sensor.num_bodies == 1,
                "Each filtered sensor must select exactly its named single wheel")
        paths = ("{ENV_REGEX_NS}/Robot/"+body, self.scene.env_regex_ns+"/Robot/"+body)
        require(cfg.prim_path in paths and view.filter_count == 1 and view.sensor_count == self.num_envs
                and cfg.max_contact_data_count_per_prim >= 32
                and view.max_contact_data_count >= 32*self.num_envs, "Ground filter/view identity or capacity differs")
        require(math.isclose(sensor._sim_physics_dt, .005, rel_tol=0, abs_tol=1.e-12),
                "Contact impulses must convert using 0.005 s physics dt, not policy dt")
        matrix = sensor.data.force_matrix_w
        require(matrix is not None and tuple(matrix.shape) == (self.num_envs, 1, 1, 3)
                and matrix.device == self.device and matrix.dtype == self.dtype,
                "Ground force must remain float32 [N,1,1,3] on the simulation device")
        return matrix

    def _attached(self):
        require(not self.closed and getattr(self.env, ENV_MARKER, None) is self
                and getattr(self.scene, SCENE_MARKER, None) is self
                and self.env.step is self.step_wrapper and self.scene.update is self.update_wrapper,
                "Ground-support owner changed; close in LIFO order")

    def _step(self, *args, **kwargs):
        self._attached(); clock._clock(self.env)
        require(not self.inside_step, "Reentrant ground-support step")
        require(self.env.num_envs == self.num_envs and self.env.scene["robot"] is self.robot,
                "Robot or environment count changed after startup mass read")
        self.start_clock = self.last_clock = clock._native_clock(self.env)
        self.samples = 0; self.sum_cost.zero_(); self.cached_reward.zero_()
        self.window_reset = False; self.steps_started += 1; self.inside_step = True
        try:
            result = self.original_step(*args, **kwargs)
            self._complete()
            require(self.last_cost_window == self.steps_started, "Ground reward was not consumed before reset/end")
            self.steps_completed += 1
            return result
        except BaseException:
            self.steps_failed += 1
            raise
        finally:
            self.inside_step = False

    def _update(self, *args, **kwargs):
        self._attached()
        require(not self.inside_update, "Reentrant scene update")
        self.inside_update = True
        try:
            result = self.original_update(*args, **kwargs)
            if not self.inside_step:
                return result
            current = clock._native_clock(self.env)
            if current[0] == self.last_clock[0]:
                require(current == self.last_clock, "Native physics advanced without environment counter")
                return result
            require(self.samples < 4 and current[0] == self.last_clock[0]+1
                    and current[1] == self.last_clock[1]+1, "Need four consecutive native physics samples")
            tolerance = self.dt_rounding+8*math.ulp(max(abs(current[2]), abs(self.last_clock[2]), .005))
            require(abs(current[2]-self.last_clock[2]-.005) <= tolerance, "Native physics time differs")
            require(not (args and "dt" in kwargs), "Duplicate scene-update dt")
            dt = args[0] if args else kwargs.get("dt")
            require(math.isclose(clock._number(dt, "scene-update dt"), .005, rel_tol=0, abs_tol=1.e-12),
                    "Only actual physics-dt refreshes may contribute")
            matrices = [self._matrix(sensor, body) for sensor, body in zip(self.sensors, WHEEL_NAMES)]
            forces = self.torch.stack([matrix[:, 0, 0] for matrix in matrices], dim=1)
            fz = forces[..., 2]
            finite = self.torch.isfinite(forces).all(dim=(1, 2))
            negative = fz < -NEGATIVE_FORCE_TOLERANCE_N
            self.fault.logical_or_(~finite | negative.any(dim=1))
            # Keep invalid values fatal in the returned reward. Sanitization is
            # only for bounded telemetry/math; no invalid data is called support.
            safe = self.torch.nan_to_num(fz, nan=0., posinf=0., neginf=0.)
            self.sum_cost.add_(self.torch.exp(-safe.clamp(min=0.)/(self.scale_fraction*self.mg[:, None])).sum(dim=1))
            self.positive_samples.add_((safe > POSITIVE_FORCE_TELEMETRY_N).sum(dim=0))
            self.negative_samples.add_(negative.sum())
            self.min_force.copy_(self.torch.minimum(self.min_force, safe.amin(dim=0)))
            self.max_force.copy_(self.torch.maximum(self.max_force, safe.amax(dim=0)))
            self.force_xy_peak.copy_(self.torch.maximum(self.force_xy_peak, self.torch.nan_to_num(forces[..., :2]).abs().amax()))
            self.samples += 1; self.total_samples += 1; self.last_clock = current
            if self.samples == 4:
                value = 1.-self.sum_cost/8.
                self.cached_reward.copy_(self.torch.where(self.fault, self.torch.full_like(value, float("nan")), value))
            return result
        finally:
            self.inside_update = False

    def _complete(self):
        self._attached()
        require(self.inside_step and self.samples == 4 and clock._native_clock(self.env) == self.last_clock
                and self.last_clock[0] == self.start_clock[0]+4 and self.last_clock[1] == self.start_clock[1]+4,
                "Ground reward requires all four native substeps before reset")

    def reward(self, scale_fraction=SCALE_FRACTION):
        self._complete(); _scale(scale_fraction)
        require(not self.window_reset, "Ground reward must be read before environment reset")
        self.cost_reads += 1
        if self.last_cost_window != self.steps_started:
            self.unique_cost_windows += 1; self.last_cost_window = self.steps_started
        return self.cached_reward

    def reset(self, env_ids=None):
        """No history to invalidate; enforce reward-before-reset order only."""
        self._attached()
        if self.inside_step:
            self._complete()
            require(self.last_cost_window == self.steps_started, "Environment reset preceded ground reward")
            self.window_reset = True
        self.reset_calls += 1

    def audit(self, *, check_device_health=False):
        result = {"schema_version": 1, "mode": "all_substeps_mean_ground_support",
                  "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "clock_helper_sha256": CLOCK_HELPER_SHA256,
                  "attributed_reference_sources": copy.deepcopy(ATTRIBUTED_REFERENCE_SOURCES),
                  "formula": "1-mean_4_substeps_2_wheels(exp(-clamp(F_ground_normal_world_z_N,min=0)/(.20*mg_N)))",
                  "scale_fraction": self.scale_fraction, "scale_is_minimum_load_requirement": False,
                  "raw_reward_bounds": [0., 1.], "external_reward_weight": .4,
                  "sensor_names": list(SENSOR_NAMES), "wheel_body_names": list(WHEEL_NAMES),
                  "force_matrix_shape_each_sensor": [self.num_envs, 1, 1, 3],
                  "force_units": "normal force N, impulse conversion uses physics dt=.005",
                  "ground_filter": copy.deepcopy(self.ground_audit),
                  "physics_dt_s": .005, "policy_dt_s": .02, "decimation": 4,
                  "mass_kg_range": self.mass_range, "mg_N_range": self.mg_range,
                  "gravity_m_s2": list(self.gravity), "mass_gravity_reads": 1,
                  "cached_mg_device": str(self.mg.device), "cached_mg_dtype": str(self.mg.dtype),
                  "gain_readback": copy.deepcopy(self.gain_readback),
                  "event_identity_audit": copy.deepcopy(self.event_audit),
                  "steps_started": self.steps_started, "steps_completed": self.steps_completed,
                  "steps_failed": self.steps_failed, "physics_samples_total": self.total_samples,
                  "physics_environment_substeps": self.total_samples*self.num_envs,
                  "samples_last_window": self.samples, "unique_cost_windows": self.unique_cost_windows,
                  "cost_reads": self.cost_reads, "reset_calls": self.reset_calls, "closed": self.closed,
                  "num_envs": self.num_envs, "device": str(self.device), "dtype": str(self.dtype),
                  "history_carried_across_windows": False, "cpu_fixture_only": self.cpu_fixture,
                  "bulk_cpu_copies_per_step": 0, "explicit_health_scalar_reads_per_step": 0,
                  "isaac_lazy_sensor_sync_free_claimed": False,
                  "continuous_ground_contact_certified": False, "absolute_height_constraint": False,
                  "zero_sideslip_target": False, "device_health_checked": check_device_health}
        if check_device_health:
            result.update(device_health_finite=not bool(self.fault.any().item()),
                          positive_normal_force_samples_left_right=self.positive_samples.cpu().tolist(),
                          negative_normal_force_samples=int(self.negative_samples.item()),
                          observed_ground_fn_z_min_N=self.min_force.cpu().tolist() if self.total_samples else None,
                          observed_ground_fn_z_max_N=self.max_force.cpu().tolist() if self.total_samples else None,
                          observed_normal_force_xy_abs_peak_N=float(self.force_xy_peak.item()))
        return result

    def close(self):
        if self.closed:
            return self.audit()
        self._attached()
        require(not self.inside_step and not self.inside_update, "Cannot detach during a step/update")
        clock._restore(self.env, "step", self.previous_step); clock._restore(self.scene, "update", self.previous_update)
        delattr(self.env, ENV_MARKER); delattr(self.scene, SCENE_MARKER)
        self.closed = True
        return self.audit()


def install_ground_support_accumulator(env, *, scale_fraction=SCALE_FRACTION):
    previous = getattr(env, ENV_MARKER, None)
    if previous is not None:
        require(type(previous) is GroundSupportAccumulator, "Unexpected ground-support owner")
        previous._attached(); _scale(scale_fraction)
        return previous
    return GroundSupportAccumulator(env, scale_fraction=scale_fraction)


def substep_ground_support_reward(env, scale_fraction=SCALE_FRACTION):
    handler = getattr(env, ENV_MARKER, None)
    require(type(handler) is GroundSupportAccumulator and handler.env is env,
            "Ground-support reward requires its explicitly installed accumulator")
    return handler.reward(scale_fraction)


def make_reward_term(*, weight=.4, scale_fraction=SCALE_FRACTION):
    """Return the same-name replacement term; no observer is installed in init."""
    require(clock._number(weight, "support weight") == .4, "Prepared one-term comparison retains weight +0.4")
    _scale(scale_fraction)
    from isaaclab.managers import ManagerTermBase, RewardTermCfg
    class GroundSupportReward(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.env = env

        def reset(self, env_ids=None):
            handler = getattr(self.env, ENV_MARKER, None)
            if handler is not None:
                handler.reset(env_ids)

        def __call__(self, env, scale_fraction=SCALE_FRACTION):
            require(env is self.env, "Ground reward called for another environment")
            return substep_ground_support_reward(env, scale_fraction)
    return RewardTermCfg(func=GroundSupportReward, weight=weight,
                         params={"scale_fraction": scale_fraction})

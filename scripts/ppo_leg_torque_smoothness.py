"""Prepared GPU-local, all-physics-substep leg applied-torque-rate reward.

No simulator imports occur until installation. This module changes no actions,
observations, motor model, references, resets, or physics. Runtime integration
still requires an explicitly authorized Isaac smoke before training adoption.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from numbers import Integral, Real
from pathlib import Path
import struct
from types import MethodType


LEG_JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint")
ENV_MARKER = "_tron1_leg_applied_torque_rate_accumulator"
SCENE_MARKER = "_tron1_leg_applied_torque_rate_owner"
CONFLICT_MARKERS = ("_tron1_substep_wheel_velocity_accumulator", "_tron1_substep_wheel_velocity_owner",
                    "_tron1_ground_load_substep_accumulator", "_tron1_ground_load_substep_owner")
PHYSICS_DT = .005
POLICY_DT = .02
DECIMATION = 4
RATED_LEG_TORQUE_NM = 80.
TORQUE_RATE_SCALE_NM_S = RATED_LEG_TORQUE_NM/POLICY_DT
DELTA_TORQUE_SCALE_NM = PHYSICS_DT*TORQUE_RATE_SCALE_NM_S
_MISSING = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    require(not isinstance(value, bool) and isinstance(value, Integral) and value >= 0,
            name+" must be a nonnegative integer")
    return int(value)


def _number(value, name):
    require(not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value),
            name+" must be finite")
    return float(value)


def _clock(env):
    require(_integer(env.cfg.decimation, "decimation") == DECIMATION, "Require fixed four-substep decimation")
    for value in (env.physics_dt, env.cfg.sim.dt, env.sim.get_physics_dt()):
        require(math.isclose(_number(value, "physics dt"), PHYSICS_DT, rel_tol=0, abs_tol=1e-12),
                "Require fixed 0.005 s physics dt")
    require(math.isclose(_number(env.step_dt, "policy dt"), POLICY_DT, rel_tol=0, abs_tol=1e-12),
            "Require fixed 0.02 s policy dt")


def _native_clock(env):
    time = _number(env.sim.current_time, "native physics time")
    require(time >= 0., "Native physics time must be nonnegative")
    return (_integer(env._sim_step_counter, "environment physics counter"),
            _integer(env.sim.current_time_step_index, "native physics index"),
            time)


def _restore(owner, name, previous):
    if previous is _MISSING:
        delattr(owner, name)
    else:
        setattr(owner, name, previous)


class LegAppliedTorqueRateAccumulator:
    """Own one scene-update/step chain; GPU history is reset per environment.

    Every actual physics sample contributes mean_legs((delta applied tau/20 Nm)^2).
    The reward is its mean over exactly four substeps. The first actual substep
    after each environment reset contributes zero but remains in denominator 4.
    """
    def __init__(self, env):
        _clock(env)
        require(not any(hasattr(owner, name) for owner in (env, env.scene)
                        for name in (ENV_MARKER, SCENE_MARKER, *CONFLICT_MARKERS)),
                "Conflicting strict substep accumulator owner")
        self.env, self.scene, self.robot = env, env.scene, env.scene["robot"]
        self.torch = importlib.import_module("torch")
        self.num_envs = _integer(env.num_envs, "environment count")
        require(self.num_envs > 0, "Need a nonempty vector environment")
        ids, names = self.robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
        require(tuple(names) == LEG_JOINT_NAMES and len(set(ids)) == 6, "Require six distinct named leg joints")
        self.leg_ids = tuple(_integer(index, "leg joint ID") for index in ids)
        torque = self.robot.data.applied_torque
        require(torque.ndim == 2 and torque.shape[0] == self.num_envs and max(self.leg_ids) < torque.shape[1],
                "Applied torque must have shape [environments, physical joints]")
        require(self.torch.is_floating_point(torque) and torque.device == self.torch.device(env.device),
                "Applied torque must remain floating point on the environment device")
        self.device, self.dtype, self.torque_shape = torque.device, torque.dtype, tuple(torque.shape)
        self.leg_index = self.torch.tensor(self.leg_ids, device=self.device, dtype=self.torch.long)
        self.previous = self.torch.zeros((self.num_envs, 6), device=self.device, dtype=self.dtype)
        self.valid = self.torch.zeros(self.num_envs, device=self.device, dtype=self.torch.bool)
        self.fault = self.torch.zeros(self.num_envs, device=self.device, dtype=self.torch.bool)
        self.sum_cost = self.torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.cached_cost = self.torch.zeros_like(self.sum_cost)
        self.closed = self.inside_step = self.inside_update = False
        self.samples = self.total_samples = self.steps_started = self.steps_completed = self.steps_failed = 0
        self.cost_reads = self.unique_cost_windows = self.reset_calls = 0
        self._last_cost_window = None
        self.last_clock = _native_clock(env)
        self._dt_rounding = abs(struct.unpack("f", struct.pack("f", PHYSICS_DT))[0]-PHYSICS_DT)
        self.original_step, self.original_update = env.step, env.scene.update
        self.previous_step = vars(env).get("step", _MISSING)
        self.previous_update = vars(env.scene).get("update", _MISSING)
        def step(owner, *args, **kwargs):
            require(owner is self.env, "Wrong step owner")
            return self._step(*args, **kwargs)
        def update(owner, *args, **kwargs):
            require(owner is self.scene, "Wrong scene owner")
            return self._update(*args, **kwargs)
        self.step_wrapper, self.update_wrapper = MethodType(step, env), MethodType(update, env.scene)
        env.step, env.scene.update = self.step_wrapper, self.update_wrapper
        setattr(env, ENV_MARKER, self); setattr(env.scene, SCENE_MARKER, self)

    def _attached(self):
        require(not self.closed and getattr(self.env, ENV_MARKER, None) is self and
                getattr(self.scene, SCENE_MARKER, None) is self and
                self.env.step is self.step_wrapper and self.scene.update is self.update_wrapper,
                "Substep accumulator ownership changed; close in LIFO order")

    def reset(self, env_ids=None):
        """Invalidate history only; never splice reset torques into a difference."""
        self._attached()
        self.valid[slice(None) if env_ids is None else env_ids] = False
        self.reset_calls += 1

    def _step(self, *args, **kwargs):
        self._attached(); _clock(self.env)
        require(not self.inside_step, "Reentrant environment step")
        self.start_clock = self.last_clock = _native_clock(self.env)
        self.samples = 0; self.sum_cost.zero_(); self.cached_cost.zero_()
        self.steps_started += 1; self.inside_step = True
        try:
            result = self.original_step(*args, **kwargs)
            self._complete()
            require(self._last_cost_window == self.steps_started, "Reward was not consumed before reset/end of control step")
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
            current = _native_clock(self.env)
            if current[0] == self.last_clock[0]:
                require(current == self.last_clock, "Native physics advanced without an environment counter")
                return result
            require(self.samples < DECIMATION and current[0] == self.last_clock[0]+1 and
                    current[1] == self.last_clock[1]+1, "Need exactly consecutive native physics samples")
            tolerance = self._dt_rounding+8*math.ulp(max(abs(current[2]), abs(self.last_clock[2]), PHYSICS_DT))
            require(abs(current[2]-self.last_clock[2]-PHYSICS_DT) <= tolerance, "Native physics time differs")
            require(not (args and "dt" in kwargs), "Duplicate scene-update dt")
            dt = args[0] if args else kwargs.get("dt")
            require(math.isclose(_number(dt, "scene-update dt"), PHYSICS_DT, rel_tol=0, abs_tol=1e-12),
                    "Only actual physics-dt updates may contribute")
            torque = self.robot.data.applied_torque
            require(tuple(torque.shape) == self.torque_shape and torque.device == self.device and torque.dtype == self.dtype,
                    "Applied-torque tensor shape/device/dtype changed")
            selected = torque.index_select(1, self.leg_index)
            self.fault.logical_or_(~self.torch.isfinite(selected).all(dim=1))
            delta_cost = ((selected-self.previous)/DELTA_TORQUE_SCALE_NM).square().mean(dim=1)
            self.sum_cost.add_(self.torch.where(self.valid, delta_cost, self.torch.zeros_like(delta_cost)))
            self.previous.copy_(selected); self.valid[:] = True
            self.samples += 1; self.total_samples += 1; self.last_clock = current
            if self.samples == DECIMATION:
                value = self.sum_cost/DECIMATION
                self.cached_cost.copy_(self.torch.where(self.fault, self.torch.full_like(value, float("nan")), value))
            return result
        finally:
            self.inside_update = False

    def _complete(self):
        self._attached()
        require(self.inside_step and self.samples == DECIMATION and _native_clock(self.env) == self.last_clock and
                self.last_clock[0] == self.start_clock[0]+DECIMATION and
                self.last_clock[1] == self.start_clock[1]+DECIMATION,
                "Reward requires all four actual substeps, before reset")

    def cost(self):
        """Idempotent current-window read; RewardManager supplies weight * .02."""
        self._complete()
        self.cost_reads += 1
        if self._last_cost_window != self.steps_started:
            self.unique_cost_windows += 1; self._last_cost_window = self.steps_started
        return self.cached_cost

    def audit(self, *, check_device_health=False):
        """Only explicit final health auditing synchronizes one GPU scalar."""
        result = {"schema_version": 1, "mode": "all_substeps_mean_leg_applied_torque_rate",
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "formula": "mean_4_substeps(mean_6_legs((delta_applied_tau_Nm/20_Nm)^2))",
            "leg_joint_names": list(LEG_JOINT_NAMES), "leg_joint_ids": list(self.leg_ids),
            "physics_dt_s": PHYSICS_DT, "policy_dt_s": POLICY_DT, "decimation": DECIMATION,
            "rated_leg_torque_nm": RATED_LEG_TORQUE_NM, "torque_rate_scale_nm_s": TORQUE_RATE_SCALE_NM_S,
            "delta_torque_scale_nm": DELTA_TORQUE_SCALE_NM,
            "first_actual_substep_after_reset_cost": 0., "denominator_after_reset": DECIMATION,
            "torque_source": "robot.data.applied_torque; explicit actuator effort, not computed/unclipped torque or contact constraint force",
            "wheel_cost_included": False, "history_carried_across_control_windows": True,
            "device": str(self.device), "dtype": str(self.dtype), "num_envs": self.num_envs,
            "steps_started": self.steps_started, "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed, "physics_samples_total": self.total_samples,
            "samples_last_window": self.samples, "unique_cost_windows": self.unique_cost_windows,
            "cost_reads": self.cost_reads, "reset_calls": self.reset_calls, "closed": self.closed,
            "bulk_cpu_copies_per_step": 0, "gpu_health_scalar_reads_per_step": 0,
            "device_health_checked": check_device_health}
        if check_device_health:
            result["device_health_finite"] = not bool(self.fault.any().item())
        return result

    def close(self):
        if self.closed:
            return self.audit()
        self._attached()
        require(not self.inside_step and not self.inside_update, "Cannot detach during step/update")
        _restore(self.env, "step", self.previous_step); _restore(self.scene, "update", self.previous_update)
        delattr(self.env, ENV_MARKER); delattr(self.scene, SCENE_MARKER)
        self.closed = True
        return self.audit()


def install_accumulator(env):
    previous = getattr(env, ENV_MARKER, None)
    if previous is not None:
        require(type(previous) is LegAppliedTorqueRateAccumulator, "Unexpected existing torque-rate owner")
        previous._attached()
        return previous
    return LegAppliedTorqueRateAccumulator(env)


def make_reward_term():
    """Create a reset-aware Isaac term without installing an observer in init."""
    from isaaclab.managers import ManagerTermBase
    class LegAppliedTorqueRate(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.env = env

        def reset(self, env_ids=None):
            handler = getattr(self.env, ENV_MARKER, None)
            if handler is not None:
                handler.reset(env_ids)

        def __call__(self, env):
            handler = getattr(env, ENV_MARKER, None)
            require(type(handler) is LegAppliedTorqueRateAccumulator and handler.env is self.env,
                    "Torque-rate reward requires its explicitly installed accumulator")
            return handler.cost()
    return LegAppliedTorqueRate

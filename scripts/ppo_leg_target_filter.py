"""Single-factor, zero-shot 50 Hz low-pass of six *processed* leg targets.

This diagnostic does not wrap environment stepping or modify observations,
requested actions, wheel actions, PD gains, resets, rewards, or physics.  Isaac
imports are unnecessary: installation checks the already-loaded runtime APIs.
The separate telemetry describes software position targets, not PhysX drives.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import math
from numbers import Integral, Real
from pathlib import Path
from types import MethodType


LEG_JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint")
WHEEL_JOINT_NAMES = ("wheel_L_Joint", "wheel_R_Joint")
ENV_MARKER = "_tron1_leg_target_filter"
TERM_MARKER = "_tron1_leg_target_filter_owner"
PHYSICS_DT, POLICY_DT, DECIMATION = .005, .02, 4
SOURCE_PINS = {
    "isaaclab.managers.action_manager": "a95d7c45048b08d8c7b526ba5ab88d40087b4bcbacb248d6e9b039c5f3b9afb9",
    "isaaclab.envs.mdp.actions.joint_actions": "1b3dcb55d969d886cee500e660f12331e49f8638fec98cde30d2b818a1ca1692",
    "isaaclab.envs.mdp.observations": "ac9662dac074c66e91fd18bb707f8a5efc820d33e115fdea638bc91c037bb41b",
    "isaaclab.envs.utils.io_descriptors": "8212b30c65d28e92c25097d66d93e0edb1f8db5ed91235147b86c314b31b8154",
    "isaaclab.envs.manager_based_rl_env": "8ec2759541c8320ed725411d49564f5617b3ab6b9d340c860bededee13d557d4",
}
_MISSING = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    require(not isinstance(value, bool) and isinstance(value, Integral) and value >= 0,
            name + " must be a nonnegative integer")
    return int(value)


def _number(value, name):
    require(not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value), name + " must be finite")
    return float(value)


def _clock(env):
    require(_integer(env.cfg.decimation, "decimation") == DECIMATION, "Require four physical substeps")
    for value in (env.physics_dt, env.cfg.sim.dt, env.sim.get_physics_dt()):
        require(math.isclose(_number(value, "physics dt"), PHYSICS_DT, rel_tol=0, abs_tol=1e-12),
                "Require unchanged 0.005 s physics dt")
    require(math.isclose(_number(env.step_dt, "policy dt"), POLICY_DT, rel_tol=0, abs_tol=1e-12),
            "Require unchanged 0.02 s policy dt")


def _native_clock(env):
    value = _number(env.sim.current_time, "native time")
    require(value >= 0, "Native time must be nonnegative")
    return (_integer(env._sim_step_counter, "environment counter"),
            _integer(env.sim.current_time_step_index, "native counter"), value)


def _restore(owner, name, previous):
    if previous is _MISSING:
        delattr(owner, name)
    else:
        setattr(owner, name, previous)


def _observation_wrapper(observation):
    """Validate the installed one-layer descriptor, not a misleading module tag.

    functools.wraps preserves last_action's module/name while its callable code
    lives in io_descriptors.py. Both full files are independently SHA-pinned by
    _runtime_sources below; following __wrapped__ alone is not sufficient.
    """
    module = "isaaclab.envs.mdp.observations"
    require(inspect.isfunction(observation) and observation.__module__ == module
            and observation.__name__ == "last_action", "Actor last_action must remain the requested ActionManager.action")
    original = getattr(observation, "__wrapped__", None)
    require(inspect.isfunction(original) and original is not observation
            and not hasattr(original, "__wrapped__"), "Require exactly one last_action descriptor wrapper")
    require(inspect.unwrap(observation) is original, "Last-action unwrap identity differs")
    require(observation.__qualname__ == original.__qualname__ == "last_action"
            and original.__module__ == module and original.__name__ == "last_action"
            and original.__code__.co_name == "last_action" and original.__code__.co_firstlineno == 653,
            "Original pinned last_action identity/code boundary differs")
    require(observation.__code__.co_name == "wrapper" and observation.__code__.co_firstlineno == 192
            and observation.__code__.co_qualname == "generic_io_descriptor.<locals>._apply.<locals>.wrapper"
            and tuple(observation.__code__.co_freevars) == ("descriptor", "func", "inspect_hooks", "sig"),
            "Require exact generic_io_descriptor wrapper code boundary/closure")
    closure = dict(zip(observation.__code__.co_freevars, observation.__closure__ or ()))
    require(closure["func"].cell_contents is original, "Descriptor closure does not call the unwrapped last_action")
    return original, {"exact_wrapper_count": 1, "wrapper_code_firstlineno": 192,
        "wrapper_code_qualname": observation.__code__.co_qualname, "original_code_firstlineno": 653,
        "closure_func_is_unwrapped_original": True, "unwrap_identity_verified": True}


def _runtime_sources(env, manager, term, *, fixture):
    if fixture:
        return {"cpu_fixture_only": True, "source_pins_checked": False}
    require(type(manager).__module__ == "isaaclab.managers.action_manager" and type(manager).__name__ == "ActionManager",
            "Require native pinned ActionManager")
    require(type(term).__module__ == "isaaclab.envs.mdp.actions.joint_actions" and type(term).__name__ == "JointPositionAction",
            "Require native pinned JointPositionAction")
    for owner, names in ((manager, ("process_action", "apply_action", "reset")),
                         (term, ("process_actions", "apply_actions", "reset"))):
        require(all(getattr(getattr(owner, name), "__func__", None) is getattr(type(owner), name) for name in names),
                "Runtime action API was monkeypatched before filter installation")
    native_bases = [base for base in type(env).__mro__
                    if base.__module__ == "isaaclab.envs.manager_based_rl_env" and base.__name__ == "ManagerBasedRLEnv"]
    require(len(native_bases) == 1 and getattr(env.step, "__func__", None) is native_bases[0].step,
            "Require unchanged native ManagerBasedRLEnv.step (terminal-capture subclass permitted)")
    observation = env.cfg.observations.policy.actions.func
    original_observation, wrapper_audit = _observation_wrapper(observation)
    objects = {type(manager).__module__: type(manager), type(term).__module__: type(term),
               native_bases[0].__module__: native_bases[0], original_observation.__module__: original_observation,
               "isaaclab.envs.utils.io_descriptors": observation}
    result = {}
    for module, expected in SOURCE_PINS.items():
        path = Path(inspect.getsourcefile(objects[module]) or "")
        require(path.is_file(), "Runtime API source file missing: " + module)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, "Runtime API source SHA256 mismatch: " + module)
        result[module] = {"path": str(path), "sha256": actual}
    return {"cpu_fixture_only": False, "source_pins_checked": True, "files": result,
            "observation_wrapper_chain": wrapper_audit}


class LegTargetFilter:
    """One owner of process/reset/apply on the existing joint_pos term only.

    A private CPU-fixture switch exists exclusively for unit tests. Production
    installation requires one CUDA environment and exact installed source pins.
    No filter state is exposed to the unchanged actor: this is zero-shot delay.
    """
    def __init__(self, env, *, alpha, _allow_cpu_fixture=False):
        self.alpha = _number(alpha, "alpha")
        require(self.alpha in (1., .5), "Only alpha=1 or alpha=0.5 is authorized")
        require(isinstance(_allow_cpu_fixture, bool), "Fixture flag must be boolean")
        _clock(env)
        self.env, self.manager, self.robot = env, env.action_manager, env.scene["robot"]
        self.term = self.manager.get_term("joint_pos")
        self.wheel_term = self.manager.get_term("wheel_effort")
        require(not hasattr(env, ENV_MARKER) and not hasattr(self.term, TERM_MARKER), "Filter already installed or conflicting owner")
        self.torch = importlib.import_module("torch")
        self.num_envs = _integer(env.num_envs, "environment count")
        require(self.num_envs == 1, "This fixed diagnostic requires exactly one environment")
        require(self.term is not self.wheel_term and self.term.action_dim == 6 and self.wheel_term.action_dim == 2,
                "Require separate six-leg position and two-wheel effort terms")
        require(tuple(self.manager._terms) == ("joint_pos", "wheel_effort"), "Action term order must remain joint_pos, wheel_effort")
        ids, names = self.robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
        self.leg_ids = tuple(_integer(index, "leg joint ID") for index in ids)
        require(tuple(names) == LEG_JOINT_NAMES and len(set(self.leg_ids)) == 6, "Require six distinct ordered named legs")
        require(not isinstance(self.term._joint_ids, slice) and tuple(self.term._joint_ids) == self.leg_ids,
                "Position action joint order must match six named legs")
        wheel_ids, wheel_names = self.robot.find_joints(list(WHEEL_JOINT_NAMES), preserve_order=True)
        require(tuple(wheel_names) == WHEEL_JOINT_NAMES and tuple(self.wheel_term._joint_ids) == tuple(wheel_ids)
                and set(wheel_ids).isdisjoint(self.leg_ids), "Wheel action mapping changed")
        require(self.term._asset is self.robot and self.wheel_term._asset is self.robot, "Require original robot action asset")
        q = self.robot.data.joint_pos
        require(q.ndim == 2 and q.shape[0] == 1 and max(self.leg_ids) < q.shape[1]
                and self.torch.is_floating_point(q), "Joint positions must have shape [1, physical joints]")
        self.device, self.dtype = q.device, q.dtype
        self.fixture = _allow_cpu_fixture
        require(self.device == self.torch.device(env.device), "Joint positions moved off environment device")
        require(self.device.type == "cuda" or self.fixture, "Production filter requires CUDA, not a CPU fallback")
        self.source_provenance = _runtime_sources(env, self.manager, self.term, fixture=self.fixture)
        self.leg_index = self.torch.tensor(self.leg_ids, device=self.device, dtype=self.torch.long)
        self.previous = self.torch.zeros((1, 6), device=self.device, dtype=self.dtype)
        self.valid = self.torch.zeros(1, device=self.device, dtype=self.torch.bool)
        self.finite = self.torch.ones((), device=self.device, dtype=self.torch.bool)
        self.action_history_ok = self.finite.clone(); self.wheel_ok = self.finite.clone()
        self.targets_ok = self.finite.clone(); self.identity_ok = self.finite.clone()
        self.control_records, self.physics_records = [], []
        self.control_steps = self.physics_apply_calls = self.reset_calls = self.failures = 0
        self.samples = 0; self.closed = self.inside_hook = False
        self.start_clock = self.last_start_clock = None
        self.expected_action = self.expected_previous_action = self.requested = None
        self.originals, self.previous_attributes, self.wrappers = {}, {}, {}
        self.wheel_methods = {name: getattr(self.wheel_term, name) for name in ("process_actions", "apply_actions", "reset")}
        for name in ("process_actions", "reset", "apply_actions"):
            self.originals[name] = getattr(self.term, name)
            self.previous_attributes[name] = vars(self.term).get(name, _MISSING)
            def wrapper(owner, *args, _name=name, **kwargs):
                require(owner is self.term, "Wrong action-term owner")
                return self._call(_name, *args, **kwargs)
            self.wrappers[name] = MethodType(wrapper, self.term)
        # Validate all tensor shapes before attaching any hook.
        self._tensor(self.manager.action, (1, 8), "ActionManager.action")
        self._tensor(self.manager.prev_action, (1, 8), "ActionManager.prev_action")
        self._tensor(self.term.processed_actions, (1, 6), "processed position actions")
        self._tensor(self.robot.data.joint_pos_target, tuple(q.shape), "software position target buffer")
        for name, method in self.wrappers.items():
            setattr(self.term, name, method)
        setattr(env, ENV_MARKER, self); setattr(self.term, TERM_MARKER, self)

    def _tensor(self, value, shape, name):
        require(tuple(value.shape) == shape and value.dtype == self.dtype and value.device == self.device,
                name + " shape/device/dtype changed")
        self.finite.logical_and_(self.torch.isfinite(value).all())
        return value

    def _attached(self):
        require(not self.closed and getattr(self.env, ENV_MARKER, None) is self
                and getattr(self.term, TERM_MARKER, None) is self
                and all(getattr(self.term, key) is value for key, value in self.wrappers.items()),
                "Filter ownership changed; restore hooks in LIFO order")
        require(self.manager.get_term("joint_pos") is self.term and self.manager.get_term("wheel_effort") is self.wheel_term,
                "Action terms replaced while filter attached")
        require(all(getattr(self.wheel_term, key) == value for key, value in self.wheel_methods.items()),
                "Wheel methods changed while filter attached")

    def _call(self, name, *args, **kwargs):
        self._attached(); _clock(self.env)
        require(not self.inside_hook, "Reentrant filter hook")
        self.inside_hook = True
        try:
            return getattr(self, "_" + name)(*args, **kwargs)
        except BaseException:
            self.failures += 1
            raise
        finally:
            self.inside_hook = False

    def _unchanged(self):
        action = self._tensor(self.manager.action, (1, 8), "ActionManager.action")
        previous = self._tensor(self.manager.prev_action, (1, 8), "ActionManager.prev_action")
        equal = (action == self.expected_action).all() & (previous == self.expected_previous_action).all()
        self.action_history_ok.logical_and_(equal)
        return equal

    def _process_actions(self, actions):
        require(self.control_steps == 0 or self.samples == DECIMATION, "Previous control must contain exactly four applies")
        current = _native_clock(self.env)
        if self.last_start_clock is not None:
            require(current[:2] == (self.last_start_clock[0]+4, self.last_start_clock[1]+4)
                    and math.isclose(current[2]-self.last_start_clock[2], POLICY_DT, rel_tol=0, abs_tol=2e-6),
                    "Control decisions must be consecutive 50 Hz native windows")
        self.expected_action = self.manager.action.detach().clone()
        self.expected_previous_action = self.manager.prev_action.detach().clone()
        self._tensor(actions, (1, 6), "requested leg actions")
        self.action_history_ok.logical_and_((actions == self.expected_action[:, :6]).all())
        result = self.originals["process_actions"](actions)
        # Original processing rebinds _processed_actions. NEVER retain its old pointer.
        target = self._tensor(self.term.processed_actions, (1, 6), "new processed position actions")
        requested = target.detach().clone()
        measured = self._tensor(self.robot.data.joint_pos.index_select(1, self.leg_index), (1, 6), "measured named leg positions")
        initialized = ~self.valid.clone()
        self.previous.copy_(self.torch.where(self.valid[:, None], self.previous, measured))
        if self.alpha == 1.:
            filtered = requested.clone()  # Exact identity, no subtract/add rounding.
        else:
            filtered = self.previous*.5 + requested*.5
        target.copy_(filtered)
        self.previous.copy_(filtered); self.valid[:] = True
        self.requested = requested
        self._tensor(filtered, (1, 6), "filtered targets")
        self.identity_ok.logical_and_((target == requested).all() if self.alpha == 1. else self.torch.ones_like(self.identity_ok))
        equal = self._unchanged().clone()
        self.start_clock = self.last_start_clock = current
        self.samples = 0; self.control_steps += 1
        self.control_records.append((current, self.expected_action.clone(), requested, filtered.clone(), equal,
                                     initialized.clone()))
        return result

    def _apply_actions(self):
        require(self.control_steps > 0 and self.samples < DECIMATION, "Apply requires one active control and at most four substeps")
        current = _native_clock(self.env)
        index = self.samples+1
        require(current[:2] == (self.start_clock[0]+index, self.start_clock[1]+index-1)
                and math.isclose(current[2]-self.start_clock[2], (index-1)*PHYSICS_DT, rel_tol=0, abs_tol=2e-6),
                "Apply must occur once per 200 Hz substep, before native stepping")
        self._unchanged()
        wheel_raw = self._tensor(self.wheel_term.raw_actions, (1, 2), "unfiltered wheel requests")
        self.wheel_ok.logical_and_((wheel_raw == self.expected_action[:, 6:]).all())
        processed = self._tensor(self.term.processed_actions, (1, 6), "held processed targets")
        self.targets_ok.logical_and_((processed == self.previous).all())
        result = self.originals["apply_actions"]()
        applied = self._tensor(self.robot.data.joint_pos_target.index_select(1, self.leg_index), (1, 6),
                               "explicit software position target readback")
        self.targets_ok.logical_and_((applied == self.previous).all())
        self._unchanged()
        self.physics_records.append((current, self.control_steps-1, index, applied.detach().clone(), self.previous.clone()))
        self.samples += 1; self.physics_apply_calls += 1
        return result

    def _reset(self, env_ids=None):
        require(self.control_steps == 0 or self.samples == DECIMATION, "Reset cannot interrupt a four-substep target hold")
        result = self.originals["reset"](env_ids)
        # The original reset may precede the robot reset. Read measured q ONLY
        # during the next original process_actions, after reset has completed.
        self.valid[slice(None) if env_ids is None else env_ids] = False
        self.reset_calls += 1
        return result

    def control_arrays(self):
        """Copies are explicit post-rollout exports; no per-step host copies."""
        np = importlib.import_module("numpy")
        rows = self.control_records
        def stack(index, shape, dtype=float):
            return (self.torch.stack([row[index] for row in rows]).detach().cpu().numpy().reshape((-1,)+shape)
                    if rows else np.empty((0,)+shape, dtype=dtype))
        return {"time_s": np.asarray([row[0][2] for row in rows], dtype=np.float64),
                "sim_step_counter": np.asarray([row[0][0] for row in rows], dtype=np.int64),
                "native_step_index": np.asarray([row[0][1] for row in rows], dtype=np.int64),
                "requested_actions_clipped": stack(1, (8,)), "requested_targets_rad": stack(2, (6,)),
                "filtered_targets_rad": stack(3, (6,)), "last_action_unchanged": stack(4, (), bool),
                "initialized_from_measured": stack(5, (), bool)}

    def physics_arrays(self):
        np = importlib.import_module("numpy")
        rows = self.physics_records
        def stack(index):
            return (self.torch.stack([row[index] for row in rows]).detach().cpu().numpy().reshape((-1, 6))
                    if rows else np.empty((0, 6), dtype=np.float32))
        return {"time_s": np.asarray([row[0][2] for row in rows], dtype=np.float64),
                "sim_step_counter": np.asarray([row[0][0] for row in rows], dtype=np.int64),
                "native_step_index": np.asarray([row[0][1] for row in rows], dtype=np.int64),
                "control_index": np.asarray([row[1] for row in rows], dtype=np.int64),
                "substep_index": np.asarray([row[2] for row in rows], dtype=np.int64),
                "applied_targets_rad": stack(3), "filtered_targets_rad": stack(4)}

    def audit(self, *, check_device_health=False):
        result = {"schema_version": 1, "kind": "zero_shot_processed_leg_position_target_lowpass",
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "alpha": self.alpha,
            "formula": "q_filtered[k]=(1-alpha)*q_filtered[k-1]+alpha*q_requested[k]; alpha1 exact copy",
            "first_after_reset": "initialize previous from measured named leg q at first process, then filter requested target",
            "physics_dt_s": PHYSICS_DT, "policy_dt_s": POLICY_DT, "decimation": DECIMATION,
            "leg_joint_names": list(LEG_JOINT_NAMES), "leg_joint_ids": list(self.leg_ids),
            "action_term_keys": ["joint_pos", "wheel_effort"], "num_envs": self.num_envs,
            "device": str(self.device), "dtype": str(self.dtype), "source_api_provenance": self.source_provenance,
            "control_steps": self.control_steps, "physics_apply_calls": self.physics_apply_calls,
            "samples_last_control": self.samples, "reset_calls": self.reset_calls, "failures": self.failures,
            "closed": self.closed, "cpu_fixture_only": self.fixture,
            "actor_last_action": "unchanged requested ActionManager.action, not filtered position target",
            "target_readback": "robot.data.joint_pos_target named six legs; explicit software actuator target, NOT native PhysX drive",
            "filter_update_rate_hz": 50, "apply_rate_hz": 200, "wheel_outputs_filtered": False,
            "alpha_one_identity_checked": self.alpha == 1.,
            "runtime_api_calls": ["joint_pos.process_actions(original first, then reacquire processed_actions)",
                                  "joint_pos.reset(original first, then invalidate private history)",
                                  "joint_pos.apply_actions(original setter, then software target readback)"],
            "additional_physics_steps": 0, "additional_reset_calls": 0, "additional_training_updates": 0,
            "bulk_cpu_copies_per_control": 0, "gpu_health_scalar_reads_per_control": 0,
            "physics_telemetry_time_semantics": "pre-step native time/index; sim_step_counter already incremented for pending apply",
            "zero_shot_distribution_shift": "unobserved target-filter state and delay; not a retrained policy or safety qualification",
            "device_health_checked": check_device_health}
        if check_device_health:
            result.update(device_health_finite=bool(self.finite.item()),
                          requested_action_history_unchanged=bool(self.action_history_ok.item()),
                          wheel_requests_unchanged=bool(self.wheel_ok.item()),
                          applied_targets_match_filtered=bool(self.targets_ok.item()),
                          alpha_one_exact_identity=bool(self.identity_ok.item()))
        return result

    def close(self):
        if self.closed:
            return self.audit()
        self._attached()
        require(not self.inside_hook, "Cannot detach inside action hook")
        require(self.control_steps == 0 or self.samples == DECIMATION, "Cannot detach incomplete control window")
        for name, previous in self.previous_attributes.items():
            _restore(self.term, name, previous)
        delattr(self.env, ENV_MARKER); delattr(self.term, TERM_MARKER)
        self.closed = True
        return self.audit()


def install_leg_target_filter(env, *, alpha, _allow_cpu_fixture=False):
    return LegTargetFilter(env, alpha=alpha, _allow_cpu_fixture=_allow_cpu_fixture)

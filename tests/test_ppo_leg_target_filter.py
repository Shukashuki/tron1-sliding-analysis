"""CPU fixtures only: no simulator imports, inference, stepping jobs or GPU use."""
import ast
import functools
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ppo_leg_target_filter as module


class PositionTerm:
    action_dim = 6

    def __init__(self, robot):
        self._asset = robot
        self._joint_ids = list(robot.leg_ids)
        self._raw_actions = torch.zeros((1, 6), dtype=torch.float64)
        self._processed_actions = self._raw_actions.clone()
        self.calls = {"process": 0, "apply": 0, "reset": 0}

    @property
    def processed_actions(self):
        return self._processed_actions

    @property
    def raw_actions(self):
        return self._raw_actions

    def process_actions(self, actions):
        self.calls["process"] += 1
        self._raw_actions[:] = actions
        # Deliberately rebind exactly as pinned Isaac does.
        self._processed_actions = self._raw_actions*1. + .25

    def apply_actions(self):
        self.calls["apply"] += 1
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        self.calls["reset"] += 1
        self._raw_actions[slice(None) if env_ids is None else env_ids] = 0.


class WheelTerm:
    action_dim = 2

    def __init__(self, robot):
        self._asset, self._joint_ids = robot, list(robot.wheel_ids)
        self.raw_actions = torch.zeros((1, 2), dtype=torch.float64)
        self.effort_targets = self.raw_actions.clone()
        self.calls = {"process": 0, "apply": 0, "reset": 0}

    def process_actions(self, actions):
        self.calls["process"] += 1
        self.raw_actions[:] = actions

    def apply_actions(self):
        self.calls["apply"] += 1
        self.effort_targets.copy_(self.raw_actions*12.)

    def reset(self, env_ids=None):
        self.calls["reset"] += 1
        self.raw_actions[slice(None) if env_ids is None else env_ids] = 0.


class Robot:
    def __init__(self, ids=(7, 5, 3, 1, 0, 2)):
        self.leg_ids, self.wheel_ids = ids, (4, 6)
        self.data = SimpleNamespace(joint_pos=torch.full((1, 8), 2., dtype=torch.float64),
                                    joint_pos_target=torch.zeros((1, 8), dtype=torch.float64))
        self.setter_calls = 0

    def find_joints(self, names, preserve_order):
        assert preserve_order
        if tuple(names) == module.LEG_JOINT_NAMES:
            return list(self.leg_ids), list(module.LEG_JOINT_NAMES)
        assert tuple(names) == module.WHEEL_JOINT_NAMES
        return list(self.wheel_ids), list(module.WHEEL_JOINT_NAMES)

    def set_joint_position_target(self, targets, joint_ids):
        self.setter_calls += 1
        self.data.joint_pos_target[:, joint_ids] = targets


class Manager:
    def __init__(self, robot):
        self._terms = {"joint_pos": PositionTerm(robot), "wheel_effort": WheelTerm(robot)}
        self.action = torch.zeros((1, 8), dtype=torch.float64)
        self.prev_action = self.action.clone()

    def get_term(self, name):
        return self._terms[name]

    def process_action(self, actions):
        self.prev_action[:] = self.action
        self.action[:] = actions
        for name, section in (("joint_pos", slice(0, 6)), ("wheel_effort", slice(6, 8))):
            self._terms[name].process_actions(actions[:, section])

    def apply_action(self):
        for term in self._terms.values():
            term.apply_actions()

    def reset(self, env_ids=None):
        self.action.zero_(); self.prev_action.zero_()
        for term in self._terms.values():
            term.reset(env_ids)


class Env:
    def __init__(self):
        self.num_envs, self.device = 1, "cpu"
        self.physics_dt, self.step_dt = .005, .02
        self.cfg = SimpleNamespace(decimation=4, sim=SimpleNamespace(dt=.005))
        self._sim_step_counter = 0
        self.sim = SimpleNamespace(current_time=0., current_time_step_index=0, get_physics_dt=lambda: .005)
        self.robot = Robot()
        self.scene = {"robot": self.robot}
        self.action_manager = Manager(self.robot)
        self.step_calls = self.reset_calls = 0

    def step(self, actions):
        self.step_calls += 1
        self.action_manager.process_action(actions)
        for _ in range(4):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.sim.current_time_step_index += 1
            self.sim.current_time += .005

    def reset(self, measured=4.):
        self.reset_calls += 1
        self.action_manager.reset()
        # Actual measured q becomes available AFTER term.reset.
        self.robot.data.joint_pos.fill_(measured)


@pytest.fixture
def env():
    return Env()


def install(env, alpha=.5):
    return module.install_leg_target_filter(env, alpha=alpha, _allow_cpu_fixture=True)


def action(value=1., wheels=(.2, -.3)):
    result = torch.full((1, 8), value, dtype=torch.float64)
    result[:, 6:] = torch.tensor(wheels, dtype=torch.float64)
    return result


def test_half_filter_uses_measured_initial_q_and_postprocessed_offset(env):
    handler = install(env)
    env.step(action(1.))
    expected = torch.full((1, 6), 1.625, dtype=torch.float64)  # (2 + (1+.25))/2
    assert torch.equal(env.action_manager.get_term("joint_pos").processed_actions, expected)
    env.step(action(-1.))
    assert torch.equal(handler.previous, torch.full((1, 6), .4375, dtype=torch.float64))
    arrays = handler.control_arrays()
    np.testing.assert_array_equal(arrays["initialized_from_measured"], [True, False])
    np.testing.assert_array_equal(arrays["requested_targets_rad"][:, 0], [1.25, -.75])


def test_alpha_one_is_bitwise_identity_not_arithmetic_reconstruction(env):
    handler = install(env, 1.)
    values = action()
    values[:, :6] = torch.tensor([[-1e-12, 1e-12, .123456789, -.123456789, .999999999, -.999999999]])
    env.robot.data.joint_pos.fill_(1e18)
    env.step(values)
    arrays = handler.control_arrays()
    np.testing.assert_array_equal(arrays["filtered_targets_rad"], arrays["requested_targets_rad"])
    assert handler.audit(check_device_health=True)["alpha_one_exact_identity"]


def test_rebound_processed_tensor_is_reacquired_every_decision(env):
    handler = install(env)
    term = env.action_manager.get_term("joint_pos")
    first = term.processed_actions
    env.step(action())
    second = term.processed_actions
    env.step(action())
    assert first is not second and second is not term.processed_actions
    assert torch.equal(first, torch.zeros_like(first))
    assert handler.control_steps == 2


def test_requested_action_history_and_wheels_are_unchanged(env):
    handler = install(env)
    first, second = action(.8), action(-.4, wheels=(-.5, .7))
    env.step(first); env.step(second)
    manager = env.action_manager
    assert torch.equal(manager.action, second) and torch.equal(manager.prev_action, first)
    assert torch.equal(manager.get_term("joint_pos").raw_actions, second[:, :6])
    assert torch.equal(manager.get_term("wheel_effort").raw_actions, second[:, 6:])
    assert torch.equal(manager.get_term("wheel_effort").effort_targets, second[:, 6:]*12.)
    audit = handler.audit(check_device_health=True)
    assert audit["requested_action_history_unchanged"] and audit["wheel_requests_unchanged"]
    np.testing.assert_array_equal(handler.control_arrays()["last_action_unchanged"], [True, True])


def test_four_original_apply_calls_and_actual_software_target_readbacks(env):
    handler = install(env)
    env.step(action())
    term = env.action_manager.get_term("joint_pos")
    assert term.calls == {"process": 1, "apply": 4, "reset": 0}
    assert env.robot.setter_calls == 4 and env.step_calls == 1 and env.reset_calls == 0
    physics = handler.physics_arrays()
    np.testing.assert_array_equal(physics["substep_index"], [1, 2, 3, 4])
    np.testing.assert_array_equal(physics["sim_step_counter"], [1, 2, 3, 4])
    np.testing.assert_array_equal(physics["native_step_index"], [0, 1, 2, 3])
    np.testing.assert_allclose(physics["time_s"], [0, .005, .01, .015])
    np.testing.assert_array_equal(physics["applied_targets_rad"], physics["filtered_targets_rad"])
    assert handler.audit(check_device_health=True)["applied_targets_match_filtered"]


def test_named_mapping_preserves_other_joint_target_columns(env):
    handler = install(env)
    env.robot.data.joint_pos_target[:, env.robot.wheel_ids] = 912.
    env.step(action())
    assert handler.leg_ids == (7, 5, 3, 1, 0, 2)
    assert torch.equal(env.robot.data.joint_pos_target[:, env.robot.wheel_ids], torch.full((1, 2), 912., dtype=torch.float64))


def test_reset_calls_original_once_invalidates_then_reads_after_reset(env):
    handler = install(env)
    env.step(action())
    previous = handler.previous.clone()
    env.reset(measured=8.)
    assert torch.equal(handler.previous, previous) and not handler.valid.any()
    env.step(action())
    assert torch.equal(handler.previous, torch.full((1, 6), 4.625, dtype=torch.float64))
    assert handler.reset_calls == 1 and env.reset_calls == 1
    assert env.action_manager.get_term("joint_pos").calls["reset"] == 1


def test_initial_reset_without_control_is_valid(env):
    handler = install(env)
    env.reset(measured=4.)
    env.step(action())
    assert handler.reset_calls == 1
    assert torch.equal(handler.previous, torch.full((1, 6), 2.625, dtype=torch.float64))


def test_exported_arrays_are_detached_copies_not_live_tensors(env):
    handler = install(env)
    env.step(action())
    first = handler.control_arrays()
    first["filtered_targets_rad"].fill(-99.)
    assert (handler.control_arrays()["filtered_targets_rad"] == 1.625).all()


def test_complete_349_decisions_and_1396_native_applies(env):
    handler = install(env, 1.)
    for _ in range(349):
        env.step(action())
    handler.close()
    audit = handler.audit(check_device_health=True)
    assert audit["control_steps"] == 349 and audit["physics_apply_calls"] == 1396
    assert audit["samples_last_control"] == 4 and audit["closed"]
    assert audit["failures"] == 0 and audit["device_health_finite"]
    assert handler.control_arrays()["requested_actions_clipped"].shape == (349, 8)
    assert handler.physics_arrays()["applied_targets_rad"].shape == (1396, 6)
    np.testing.assert_array_equal(handler.control_arrays()["sim_step_counter"], np.arange(0, 1396, 4))
    np.testing.assert_array_equal(handler.physics_arrays()["control_index"], np.repeat(np.arange(349), 4))


def test_close_restores_class_methods_and_is_idempotent(env):
    term = env.action_manager.get_term("joint_pos")
    originals = {name: getattr(term, name).__func__ for name in ("process_actions", "reset", "apply_actions")}
    handler = install(env)
    env.step(action()); handler.close(); handler.close()
    assert not hasattr(env, module.ENV_MARKER) and not hasattr(term, module.TERM_MARKER)
    assert all(name not in vars(term) and getattr(term, name).__func__ is method for name, method in originals.items())
    assert handler.audit(check_device_health=True)["closed"]


def test_close_restores_preexisting_instance_bound_method(env):
    term = env.action_manager.get_term("joint_pos")
    previous = term.process_actions
    term.process_actions = previous
    handler = install(env)
    handler.close()
    assert term.process_actions is previous


def test_duplicate_install_and_changed_hook_ownership_rejected(env):
    handler = install(env)
    with pytest.raises(ValueError, match="already installed"):
        install(env)
    env.action_manager.get_term("joint_pos").reset = lambda ids=None: None
    with pytest.raises(ValueError, match="ownership"):
        handler.close()


def test_wheel_hook_replacement_rejected(env):
    handler = install(env)
    env.action_manager.get_term("wheel_effort").apply_actions = lambda: None
    with pytest.raises(ValueError, match="Wheel methods"):
        env.step(action())


@pytest.mark.parametrize("alpha", [0, .2, .75, 2, float("nan"), True])
def test_only_two_predeclared_alphas_allowed(env, alpha):
    with pytest.raises(ValueError):
        install(env, alpha)


def test_cpu_fallback_not_allowed_in_production(env):
    with pytest.raises(ValueError, match="requires CUDA"):
        module.install_leg_target_filter(env, alpha=.5)


def test_unsupported_action_order_and_joint_mapping_rejected(env):
    env.action_manager._terms = dict(reversed(list(env.action_manager._terms.items())))
    with pytest.raises(ValueError, match="term order"):
        install(env)
    env.action_manager._terms = dict(reversed(list(env.action_manager._terms.items())))
    env.action_manager.get_term("joint_pos")._joint_ids = list(range(6))
    with pytest.raises(ValueError, match="joint order"):
        install(env)


@pytest.mark.parametrize("field,value", [("decimation", 2), ("dt", .01)])
def test_physics_clock_changes_rejected(env, field, value):
    if field == "dt":
        env.cfg.sim.dt = value
    else:
        env.cfg.decimation = value
    with pytest.raises(ValueError):
        install(env)


def test_apply_without_process_rejected(env):
    handler = install(env)
    with pytest.raises(ValueError, match="active control"):
        env.action_manager.get_term("joint_pos").apply_actions()
    assert handler.failures == 1


def test_duplicate_process_and_partial_reset_close_fail_closed(env):
    handler = install(env)
    env.action_manager.process_action(action())
    with pytest.raises(ValueError, match="four applies"):
        env.action_manager.process_action(action())
    with pytest.raises(ValueError, match="interrupt"):
        env.action_manager.get_term("joint_pos").reset()
    with pytest.raises(ValueError, match="incomplete"):
        handler.close()


def test_duplicate_apply_and_wrong_native_order_rejected(env):
    handler = install(env)
    env.action_manager.process_action(action())
    with pytest.raises(ValueError, match="before native stepping"):
        env.action_manager.get_term("joint_pos").apply_actions()
    env._sim_step_counter += 1
    env.action_manager.apply_action()
    with pytest.raises(ValueError, match="before native stepping"):
        env.action_manager.get_term("joint_pos").apply_actions()


def test_nonfinite_targets_latch_health_failure(env):
    handler = install(env)
    values = action(); values[0, 0] = float("nan")
    env.step(values)
    assert not handler.audit(check_device_health=True)["device_health_finite"]


def test_requested_history_mutation_is_not_silently_certified(env):
    handler = install(env)
    env.action_manager.process_action(action())
    env.action_manager.prev_action.fill_(99.)
    for _ in range(4):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.sim.current_time_step_index += 1; env.sim.current_time += .005
    assert not handler.audit(check_device_health=True)["requested_action_history_unchanged"]


def test_actual_software_target_mismatch_not_silently_certified(env):
    original = env.robot.set_joint_position_target
    def wrong(targets, joint_ids):
        original(targets, joint_ids)
        env.robot.data.joint_pos_target[:, joint_ids] += 1.
    env.robot.set_joint_position_target = wrong
    handler = install(env)
    env.step(action())
    assert not handler.audit(check_device_health=True)["applied_targets_match_filtered"]


def test_no_extra_step_reset_forward_or_native_drive_setter_in_source():
    tree = ast.parse(Path(module.__file__).read_text())
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not {"step", "forward", "reset", "write_joint_state_to_sim", "set_dof_position_targets"}.intersection(calls)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and any(alias.name.startswith("isaac") for alias in node.names)
                   for node in ast.walk(tree))


def test_runtime_api_source_pin_rejection_is_not_bypassed():
    fake = SimpleNamespace()
    with pytest.raises(ValueError, match="pinned ActionManager"):
        module._runtime_sources(fake, object(), object(), fixture=False)
    assert module._runtime_sources(fake, object(), object(), fixture=True)["source_pins_checked"] is False


def test_empty_telemetry_shapes_and_source_provenance_are_explicit(env):
    handler = install(env)
    assert handler.control_arrays()["requested_actions_clipped"].shape == (0, 8)
    assert handler.physics_arrays()["applied_targets_rad"].shape == (0, 6)
    audit = handler.audit()
    assert audit["source_api_provenance"]["cpu_fixture_only"] is True
    assert not audit["source_api_provenance"]["source_pins_checked"]
    assert "device_health_finite" not in audit
    assert audit["additional_physics_steps"] == audit["additional_reset_calls"] == 0


@pytest.fixture
def descriptor_runtime(tmp_path, monkeypatch):
    """Five isolated source files; no Isaac imports or native app required.

    Fixture SHA pins are explicitly substituted only inside this test. The
    installed wrapper code boundary and closure layout match the audited SDK.
    """
    modules = {}
    def source_module(name, source):
        path = tmp_path/(name.rsplit(".", 1)[-1]+".py")
        path.write_text(source)
        current = ModuleType(name); current.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, current)
        modules[name] = current
        return current, path
    descriptor, descriptor_path = source_module("isaaclab.envs.utils.io_descriptors", "\n"*186+'''def generic_io_descriptor(func):
    def _apply(func):
        descriptor = object()
        inspect_hooks = []
        sig = inspect.signature(func)
        @functools.wraps(func)
        def wrapper(env, *args, **kwargs):
            inspect_flag = kwargs.pop("inspect", False)
            out = func(env, *args, **kwargs)
            if inspect_flag:
                for hook in inspect_hooks:
                    hook(output=out, descriptor=descriptor, signature=sig)
            return out
        return wrapper
    return _apply(func)
''')
    descriptor.__dict__.update(functools=functools, inspect=inspect)
    exec(compile(descriptor_path.read_text(), str(descriptor_path), "exec"), descriptor.__dict__)
    observations, observation_path = source_module("isaaclab.envs.mdp.observations", "\n"*652+'''def last_action(env, action_name=None):
    if action_name is None:
        return env.action_manager.action
    return env.action_manager.get_term(action_name).raw_actions
''')
    exec(compile(observation_path.read_text(), str(observation_path), "exec"), observations.__dict__)
    original = observations.last_action
    wrapped = descriptor.generic_io_descriptor(original)
    observations.last_action = wrapped
    class_sources = {
        "isaaclab.managers.action_manager": "class ActionManager:\n def process_action(self): pass\n def apply_action(self): pass\n def reset(self): pass\n",
        "isaaclab.envs.mdp.actions.joint_actions": "class JointPositionAction:\n def process_actions(self): pass\n def apply_actions(self): pass\n def reset(self): pass\n",
        "isaaclab.envs.manager_based_rl_env": "class ManagerBasedRLEnv:\n def step(self): pass\n",
    }
    for name, source in class_sources.items():
        current, path = source_module(name, source)
        exec(compile(source, str(path), "exec"), current.__dict__)
    manager = modules["isaaclab.managers.action_manager"].ActionManager()
    term = modules["isaaclab.envs.mdp.actions.joint_actions"].JointPositionAction()
    env = modules["isaaclab.envs.manager_based_rl_env"].ManagerBasedRLEnv()
    env.cfg = SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(actions=SimpleNamespace(func=wrapped))))
    pins = {name: hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest() for name, current in modules.items()}
    monkeypatch.setattr(module, "SOURCE_PINS", pins)
    return SimpleNamespace(env=env, manager=manager, term=term, wrapped=wrapped, original=original,
        observation_path=observation_path, descriptor_path=descriptor_path, modules=modules, pins=pins)


def test_one_descriptor_wrapper_and_unwrapped_original_are_separately_sha_pinned(descriptor_runtime):
    fixture = descriptor_runtime
    assert inspect.getsourcefile(fixture.wrapped) == str(fixture.descriptor_path)
    assert inspect.getsourcefile(inspect.unwrap(fixture.wrapped)) == str(fixture.observation_path)
    proof = module._runtime_sources(fixture.env, fixture.manager, fixture.term, fixture=False)
    assert proof["source_pins_checked"] and len(proof["files"]) == 5
    assert proof["files"]["isaaclab.envs.mdp.observations"]["sha256"] == fixture.pins["isaaclab.envs.mdp.observations"]
    assert proof["files"]["isaaclab.envs.utils.io_descriptors"]["sha256"] == fixture.pins["isaaclab.envs.utils.io_descriptors"]
    assert proof["observation_wrapper_chain"]["exact_wrapper_count"] == 1
    assert proof["observation_wrapper_chain"]["closure_func_is_unwrapped_original"]
    requested, named = object(), object()
    fake_env = SimpleNamespace(action_manager=SimpleNamespace(action=requested, get_term=lambda name: SimpleNamespace(raw_actions=named)))
    assert fixture.wrapped(fake_env) is requested and fixture.wrapped(fake_env, action_name="joint_pos") is named


@pytest.mark.parametrize("path_name,module_name", [
    ("observation_path", "isaaclab.envs.mdp.observations"),
    ("descriptor_path", "isaaclab.envs.utils.io_descriptors"),
])
def test_either_original_or_descriptor_file_mutation_rejects_full_hash(descriptor_runtime, path_name, module_name):
    fixture = descriptor_runtime
    path = getattr(fixture, path_name)
    path.write_text(path.read_text()+"\n# changed source\n")
    with pytest.raises(ValueError, match="Runtime API source SHA256 mismatch: "+module_name):
        module._runtime_sources(fixture.env, fixture.manager, fixture.term, fixture=False)


def test_plain_unwrapped_last_action_is_not_accepted_as_installed_callable(descriptor_runtime):
    with pytest.raises(ValueError, match="exactly one"):
        module._observation_wrapper(descriptor_runtime.original)


def test_two_wrappers_or_recursive_chain_fail_closed(descriptor_runtime):
    fixture = descriptor_runtime
    twice = fixture.modules["isaaclab.envs.utils.io_descriptors"].generic_io_descriptor(fixture.wrapped)
    with pytest.raises(ValueError, match="exactly one"):
        module._observation_wrapper(twice)
    fixture.wrapped.__wrapped__ = fixture.wrapped
    with pytest.raises(ValueError, match="exactly one"):
        module._observation_wrapper(fixture.wrapped)


def test_forged_wrapped_pointer_not_matching_real_closure_is_rejected(descriptor_runtime):
    fixture = descriptor_runtime
    forged = FunctionType(fixture.original.__code__, fixture.original.__globals__, fixture.original.__name__, fixture.original.__defaults__)
    fixture.wrapped.__wrapped__ = forged
    with pytest.raises(ValueError, match="closure does not call"):
        module._observation_wrapper(fixture.wrapped)


@pytest.mark.parametrize("which,firstline", [("wrapped", 193), ("original", 654)])
def test_wrapper_or_original_code_boundary_mismatch_rejected(descriptor_runtime, which, firstline):
    fixture = descriptor_runtime
    function = getattr(fixture, which)
    function.__code__ = function.__code__.replace(co_firstlineno=firstline)
    with pytest.raises(ValueError, match="boundary"):
        module._observation_wrapper(fixture.wrapped)


def test_preinstalled_action_monkeypatch_still_rejected(descriptor_runtime):
    fixture = descriptor_runtime
    fixture.manager.process_action = lambda: None
    with pytest.raises(ValueError, match="monkeypatched"):
        module._runtime_sources(fixture.env, fixture.manager, fixture.term, fixture=False)


def test_installed_complete_sha_pins_not_weakened():
    assert module.SOURCE_PINS == {
        "isaaclab.managers.action_manager": "a95d7c45048b08d8c7b526ba5ab88d40087b4bcbacb248d6e9b039c5f3b9afb9",
        "isaaclab.envs.mdp.actions.joint_actions": "1b3dcb55d969d886cee500e660f12331e49f8638fec98cde30d2b818a1ca1692",
        "isaaclab.envs.mdp.observations": "ac9662dac074c66e91fd18bb707f8a5efc820d33e115fdea638bc91c037bb41b",
        "isaaclab.envs.utils.io_descriptors": "8212b30c65d28e92c25097d66d93e0edb1f8db5ed91235147b86c314b31b8154",
        "isaaclab.envs.manager_based_rl_env": "8ec2759541c8320ed725411d49564f5617b3ab6b9d340c860bededee13d557d4",
    }

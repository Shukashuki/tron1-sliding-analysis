"""CPU Torch fixtures for native-rate reward lifecycle; no Isaac/GPU jobs."""
from pathlib import Path
import ast
import sys
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ppo_leg_torque_smoothness as reward


class Scene:
    def __init__(self, robot):
        self.robot = robot
        self.calls = 0

    def __getitem__(self, key):
        assert key == "robot"
        return self.robot

    def update(self, dt):
        self.calls += 1


class Env:
    def __init__(self):
        self.num_envs, self.device = 2, "cpu"
        self.cfg = SimpleNamespace(decimation=4, sim=SimpleNamespace(dt=.005))
        self.physics_dt, self.step_dt = .005, .02
        self._sim_step_counter = 0
        self.sim = SimpleNamespace(current_time_step_index=0, current_time=0., get_physics_dt=lambda: .005)
        self.robot = SimpleNamespace(data=SimpleNamespace(applied_torque=torch.zeros(2, 8, dtype=torch.float64)),
            find_joints=lambda names, preserve_order: (list(range(6)), list(reward.LEG_JOINT_NAMES)))
        self.scene = Scene(self.robot)
        self.sequence = [torch.full((2, 8), float(value), dtype=torch.float64) for value in (0, 20, 40, 60)]
        self.read_twice = self.extra_refresh = False
        self.reset_after_reward = None
        self.close_calls = 0

    def step(self, actions=None):
        for tau in self.sequence:
            self.robot.data.applied_torque.copy_(tau)
            self._sim_step_counter += 1
            self.sim.current_time_step_index += 1
            self.sim.current_time += .005
            self.scene.update(.005)
            if self.extra_refresh:
                self.scene.update(.005)
        handler = getattr(self, reward.ENV_MARKER)
        value = handler.cost().clone()
        if self.read_twice:
            assert torch.equal(value, handler.cost())
        if self.reset_after_reward is not None:
            handler.reset(self.reset_after_reward)
            self.robot.data.applied_torque.fill_(999.)
            self.scene.update(.005)
        return value

    def close(self):
        self.close_calls += 1


@pytest.fixture
def env():
    return Env()


def test_first_substep_zero_and_fixed_denominator_four(env):
    handler = reward.install_accumulator(env)
    value = env.step()
    assert torch.equal(value, torch.full((2,), .75, dtype=torch.float64))
    assert handler.samples == 4 and handler.total_samples == 4
    assert handler.unique_cost_windows == handler.steps_completed == 1
    assert handler.audit(check_device_health=True)["device_health_finite"] is True


def test_history_carries_across_control_windows(env):
    handler = reward.install_accumulator(env)
    env.step()
    env.sequence = [torch.full((2, 8), 80., dtype=torch.float64)]*4
    assert torch.equal(env.step(), torch.full((2,), .25, dtype=torch.float64))
    assert handler.total_samples == 8 and handler.steps_completed == 2


def test_partial_reset_does_not_splice_history_or_reset_other_env(env):
    handler = reward.install_accumulator(env)
    env.step(); handler.reset(torch.tensor([0]))
    env.sequence = [torch.full((2, 8), 80., dtype=torch.float64)]*4
    assert torch.equal(env.step(), torch.tensor([0., .25], dtype=torch.float64))


def test_same_counter_refresh_does_not_duplicate_samples(env):
    env.extra_refresh = True
    handler = reward.install_accumulator(env)
    assert torch.equal(env.step(), torch.full((2,), .75, dtype=torch.float64))
    assert env.scene.calls == 8 and handler.total_samples == 4


def test_reset_refresh_after_reward_is_not_new_physics(env):
    env.reset_after_reward = [0]
    handler = reward.install_accumulator(env)
    assert torch.equal(env.step(), torch.full((2,), .75, dtype=torch.float64))
    assert handler.samples == 4 and handler.total_samples == 4
    env.reset_after_reward = None
    env.sequence = [torch.full((2, 8), 80., dtype=torch.float64)]*4
    assert torch.equal(env.step(), torch.tensor([0., .25], dtype=torch.float64))


def test_forward_outside_step_does_not_advance_reward_history(env):
    handler = reward.install_accumulator(env)
    env.robot.data.applied_torque.fill_(80.)
    env.scene.update(.005)
    assert handler.total_samples == 0 and not handler.valid.any()
    with pytest.raises(ValueError, match="all four actual"):
        handler.cost()


def test_repeated_reward_reads_are_idempotent(env):
    env.read_twice = True
    handler = reward.install_accumulator(env)
    env.step()
    assert handler.cost_reads == 2 and handler.unique_cost_windows == 1


def test_wheel_efforts_are_not_included(env):
    handler = reward.install_accumulator(env)
    env.sequence = [torch.zeros(2, 8, dtype=torch.float64) for _ in range(4)]
    for index, sample in enumerate(env.sequence):
        sample[:, 6:] = index*100000.
    assert torch.equal(env.step(), torch.zeros(2, dtype=torch.float64))
    assert handler.audit()["wheel_cost_included"] is False


def test_named_leg_mapping_does_not_assume_first_six_columns(env):
    ids = [7, 5, 3, 1, 0, 2]
    env.robot.find_joints = lambda names, preserve_order: (ids, list(reward.LEG_JOINT_NAMES))
    env.sequence = [torch.zeros(2, 8, dtype=torch.float64) for _ in range(4)]
    for number, sample in enumerate(env.sequence):
        sample[:, ids] = 20.*number
        sample[:, [4, 6]] = 10000.*number
    handler = reward.install_accumulator(env)
    assert torch.equal(env.step(), torch.full((2,), .75, dtype=torch.float64))
    assert handler.leg_ids == tuple(ids)


def test_torque_is_read_after_original_scene_update(env):
    before = env.scene.update
    def update(dt):
        before(dt)
        env.robot.data.applied_torque[:, :6].add_(10.)
    env.scene.update = update
    reward.install_accumulator(env)
    assert torch.equal(env.step(), torch.full((2,), .75, dtype=torch.float64))


def test_source_arrays_and_actions_are_not_modified(env):
    values = [sample.clone() for sample in env.sequence]
    actions = torch.tensor([[1., -1.], [2., -2.]])
    original = actions.clone()
    reward.install_accumulator(env); env.step(actions)
    assert torch.equal(actions, original)
    assert all(torch.equal(a, b) for a, b in zip(values, env.sequence))


def test_nonfinite_applied_leg_effort_latches_fault(env):
    handler = reward.install_accumulator(env)
    env.sequence[0][0, 0] = float("nan")
    value = env.step()
    assert torch.isnan(value[0]) and torch.isfinite(value[1])
    handler.reset()
    assert handler.audit(check_device_health=True)["device_health_finite"] is False


@pytest.mark.parametrize("count", [3, 5])
def test_incomplete_or_extra_physics_samples_fail_closed(env, count):
    env.sequence = [torch.zeros(2, 8, dtype=torch.float64)]*count
    handler = reward.install_accumulator(env)
    with pytest.raises(ValueError):
        env.step()
    assert handler.steps_failed == 1


def test_changed_native_index_fails_closed(env):
    previous = env.scene.update
    def bad_update(dt):
        previous(dt)
        env.sim.current_time_step_index += 1
    env.scene.update = bad_update
    reward.install_accumulator(env)
    with pytest.raises(ValueError, match="consecutive"):
        env.step()


def test_bad_physics_clock_rejected(env):
    env.physics_dt = .01
    with pytest.raises(ValueError, match="physics dt"):
        reward.install_accumulator(env)


@pytest.mark.parametrize("value", [-.005, float("nan"), float("inf")])
def test_invalid_native_time_rejected_before_install(env, value):
    env.sim.current_time = value
    with pytest.raises(ValueError):
        reward.install_accumulator(env)


@pytest.mark.parametrize("marker", reward.CONFLICT_MARKERS)
def test_old_strict_owner_is_rejected(env, marker):
    setattr(env.scene, marker, object())
    with pytest.raises(ValueError, match="Conflicting"):
        reward.install_accumulator(env)


def test_installation_is_idempotent_for_exact_same_owner(env):
    first = reward.install_accumulator(env)
    assert reward.install_accumulator(env) is first


def test_ownership_change_rejected_and_close_restores_methods(env):
    original_step, original_update = env.step, env.scene.update
    handler = reward.install_accumulator(env)
    own_update = env.scene.update
    env.scene.update = lambda dt: None
    with pytest.raises(ValueError, match="ownership"):
        env.step()
    env.scene.update = own_update
    handler.close()
    assert env.step == original_step and env.scene.update == original_update
    assert not hasattr(env, reward.ENV_MARKER)
    assert handler.close()["closed"] is True


def test_no_device_to_host_copy_in_hot_paths():
    tree = ast.parse(Path(reward.__file__).read_text())
    hot = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in ("_step", "_update", "cost", "reset")]
    methods = {node.func.attr for function in hot for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not methods.intersection({"cpu", "numpy", "tolist", "item"})

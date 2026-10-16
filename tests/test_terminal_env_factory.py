"""CPU stubs exercise terminal capture ordering, not actual Isaac physics."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from terminal_observation import make_terminal_observation_env_class


class FakeCommand:
    def __init__(self):
        self.time_steps = torch.tensor([3, 4, 9], dtype=torch.long)
        self.motion = SimpleNamespace(time_step_total=10)
        self.refreshed_frames = []

    def _refresh_relative_targets(self):
        self.refreshed_frames.append(self.time_steps.clone())


class FakeObservations:
    def __init__(self, env):
        self.env = env
        self.history_updates = []
        self.fail_capture = False

    def compute(self, update_history=False):
        self.history_updates.append(update_history)
        if self.fail_capture and not update_history:
            raise RuntimeError("deliberate final observation failure")
        e = self.env
        policy = torch.stack((e.position, e.command.time_steps.float(), e.action[:, 0]), dim=-1)
        critic = torch.cat((policy, (2 * e.position)[:, None]), dim=-1)
        return {"policy": policy, "critic": critic}


class FakeManagerBasedRLEnv:
    """Only reproduce the v2.3 physics/reward/reset/command/observation order."""

    def __init__(self, cfg=None, terminated=None, truncated=None):
        self.cfg = cfg or SimpleNamespace(observations=SimpleNamespace())
        self.device = "cpu"
        self.num_envs = 3
        self.position = torch.tensor([10.0, 20.0, 30.0])
        self.action = torch.zeros(3, 1)
        self.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self.command = FakeCommand()
        self.command_manager = SimpleNamespace(get_term=lambda name: self.command)
        self.observation_manager = FakeObservations(self)
        self.next_terminated = torch.tensor([True, False, False]) if terminated is None else terminated.clone()
        self.next_truncated = torch.tensor([False, False, True]) if truncated is None else truncated.clone()
        self.reset_calls = []
        self.extras = {"unrelated_extra": 321}

    def _reset_idx(self, env_ids):
        self.reset_calls.append(env_ids.clone())
        self.position[env_ids] = -100.0 - env_ids.float()
        self.action[env_ids] = 0.0
        self.command.time_steps[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs))
        return self.observation_manager.compute(update_history=True), {}

    def step(self, actions):
        self.action = actions.clone()
        self.position += 1.0 + actions[:, 0]
        self.episode_length_buf += 1
        # Reward uses the HELD pre-action command, unlike successor observations.
        reward = (self.position + self.command.time_steps).clone()
        terminated, truncated = self.next_terminated.clone(), self.next_truncated.clone()
        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids):
            self._reset_idx(done_ids)
        # TRON1 skips phase advancement for newly reset environments.
        self.command.time_steps = (self.command.time_steps + (self.episode_length_buf > 0)).clamp(max=9)
        self.command._refresh_relative_targets()
        observations = self.observation_manager.compute(update_history=True)
        return observations, reward, terminated, truncated, self.extras


@pytest.fixture
def env_class(monkeypatch):
    package = ModuleType("isaaclab")
    module = ModuleType("isaaclab.envs")
    module.ManagerBasedRLEnv = FakeManagerBasedRLEnv
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", module)
    return make_terminal_observation_env_class()


@pytest.mark.parametrize("both_flags", [False, True])
def test_final_physical_state_phase_and_flags_survive_automatic_reset(env_class, both_flags):
    terminated = torch.tensor([True, False, both_flags])
    truncated = torch.tensor([False, False, True])
    env = env_class(terminated=terminated, truncated=truncated)
    actions = torch.tensor([[0.25], [0.5], [-0.25]])
    observations, reward, actual_terminated, actual_truncated, extras = env.step(actions)

    # Returned done rows belong to reset episodes, but bootstrap rows must not.
    torch.testing.assert_close(observations["policy"], torch.tensor([
        [-100.0, 0.0, 0.0], [21.5, 5.0, 0.5], [-102.0, 0.0, 0.0]]))
    torch.testing.assert_close(extras["bootstrap_observation"]["policy"], torch.tensor([
        [11.25, 4.0, 0.25], [21.5, 5.0, 0.5], [30.75, 9.0, -0.25]]))
    # Privileged final observations are captured too, not the privileged reset.
    torch.testing.assert_close(extras["bootstrap_observation"]["critic"][:, -1],
                               torch.tensor([22.5, 43.0, 61.5]))
    torch.testing.assert_close(extras["final_reference_frames"], torch.tensor([4, 9]))
    torch.testing.assert_close(extras["final_observation_env_ids"], torch.tensor([0, 2]))
    # In particular, final clip frame 9 is clamped rather than wrapping to 0/10.
    assert env.command.time_steps.tolist() == [0, 5, 0]
    assert env.command.refreshed_frames[0].tolist() == [4, 4, 9]
    assert env.command.refreshed_frames[1].tolist() == [3, 4, 9]
    torch.testing.assert_close(reward, torch.tensor([14.25, 25.5, 39.75]))
    assert actual_terminated.dtype == actual_truncated.dtype == torch.bool
    torch.testing.assert_close(actual_terminated, terminated)
    torch.testing.assert_close(actual_truncated, truncated)
    torch.testing.assert_close(env.next_terminated, terminated)
    torch.testing.assert_close(env.next_truncated, truncated)
    assert env.observation_manager.history_updates == [False, True]
    assert extras["unrelated_extra"] == 321
    assert "bootstrap_observation" not in env.extras


def test_no_done_rows_return_ordinary_successor_observations(env_class):
    none = torch.zeros(3, dtype=torch.bool)
    env = env_class(terminated=none, truncated=none)
    observations, _, terminated, truncated, extras = env.step(torch.zeros(3, 1))
    assert extras["bootstrap_observation"] is observations
    assert extras["final_observation_env_ids"].numel() == 0
    assert extras["final_reference_frames"].numel() == 0
    assert not terminated.any() and not truncated.any()
    assert env.observation_manager.history_updates == [True]
    assert env.reset_calls == []
    assert env._captured_final is None
    assert env.command.time_steps.tolist() == [4, 5, 9]


def test_explicit_reset_does_not_attempt_terminal_capture(env_class):
    env = env_class()
    observations, _ = env.reset()
    assert env._capture_final_observation is False
    assert env._captured_final is None
    assert env.observation_manager.history_updates == [True]
    assert env.command.refreshed_frames == []
    assert len(env.reset_calls) == 1
    torch.testing.assert_close(observations["policy"][:, 0], torch.tensor([-100.0, -101.0, -102.0]))


def test_capture_failure_restores_reference_phase_and_capture_flag(env_class):
    env = env_class()
    before = env.command.time_steps.clone()
    env.observation_manager.fail_capture = True
    with pytest.raises(RuntimeError, match="deliberate"):
        env.step(torch.zeros(3, 1))
    torch.testing.assert_close(env.command.time_steps, before)
    assert env._capture_final_observation is False
    assert env._captured_final is None
    assert env.reset_calls == []


@pytest.mark.parametrize("group_history,corruption,term_history,modifiers", [
    (2, False, 0, None), (None, True, 0, None),
    (None, False, 2, None), (None, False, 0, [object()]),
])
def test_factory_fails_closed_on_history_noise_or_stateful_modifiers(
        env_class, group_history, corruption, term_history, modifiers):
    term = SimpleNamespace(func=lambda env: None, history_length=term_history, modifiers=modifiers)
    group = SimpleNamespace(enable_corruption=corruption, history_length=group_history, term=term)
    cfg = SimpleNamespace(observations=SimpleNamespace(policy=group))
    with pytest.raises(ValueError, match="history|uncorrupted|modifiers"):
        env_class(cfg=cfg)

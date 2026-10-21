"""Torch CPU task-clock/event tests; these doubles are not physical evidence."""
import ast
import json
from pathlib import Path
import sys

import pytest
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from mjlab_gpu_ppo_env import GpuTron1PPOEnv, TensorPhaseSampler
from tron1_torch_task import TorchHeightSmoothObjective


class TensorPlantDouble:
    def __init__(self, batch):
        self.qpos = torch.zeros(batch, 15)
        self.qpos[:, 2], self.qpos[:, 3] = .75, 1.
        self.qvel = torch.zeros(batch, 14)
        self.time = torch.zeros(batch)
        self.events = []

    def measure_tensors(self):
        p = self.qpos[:, None, :3].expand(-1, 9, -1).clone()
        q = self.qpos[:, None, 3:7].expand(-1, 9, -1).clone()
        return {"qpos": self.qpos, "qvel": self.qvel, "time_s": self.time,
                "body_pos_w": p, "body_quat_w": q,
                "body_lin_vel_w": self.qvel[:, None, :3].expand(-1, 9, -1).clone(),
                "body_ang_vel_w": self.qvel[:, None, 3:6].expand(-1, 9, -1).clone()}

    def step_tensors(self, action):
        self.time += .02
        result = self.measure_tensors()
        force = torch.zeros(len(self.qpos), 4, 9, 3)
        force[:, :, -2:, 2] = 6.
        result["body_net_forces_w"] = force
        return result

    def reset_tensors(self, mask, qpos, qvel):
        self.events.append("reset")
        self.qpos.copy_(torch.where(mask[:, None], qpos, self.qpos))
        self.qvel.copy_(torch.where(mask[:, None], qvel, self.qvel))
        self.time.masked_fill_(mask, 0.)

    def apply_link_velocity_deltas(self, mask, deltas):
        self.events.append("push")
        self.qvel[:, :6] += torch.where(mask[:, None], deltas, torch.zeros_like(deltas))


def task_double(training=False, frames=349, batch=2):
    env = object.__new__(GpuTron1PPOEnv)
    env.device, env.dtype, env.num_envs = torch.device("cpu"), torch.float32, batch
    env.frames, env.max_episode_length = frames, frames+2
    env.training, env.dr_mode, env.seed = training, "mjlab_v1p4" if training else "none", 42
    env.generator = torch.Generator().manual_seed(42)
    env.sampler = TensorPhaseSampler(frames, batch, env.generator, env.device)
    env.qids, env.vids = torch.arange(7, 15), torch.arange(6, 14)
    env.q0s, env.encoder_bias = torch.zeros(batch, 8), torch.zeros(batch, 8)
    env.soft_limits = torch.tensor([[-1., 1.]]*8)
    env.com_local = torch.zeros(batch, 3)
    env.batch = TensorPlantDouble(batch)
    base = env.batch.measure_tensors()
    env.reference_data = {key: value[:1].expand(frames, *value.shape[1:]).clone()
                          for key, value in base.items() if key.startswith("body_")}
    env.reference_data.update(joint_pos=torch.zeros(frames, 8), joint_vel=torch.zeros(frames, 8))
    env.profile = json.loads((ROOT/"config/car_drift_height_smooth_task.json").read_text())
    env.objective = TorchHeightSmoothObjective(env.profile, env.soft_limits)
    env.phases = torch.zeros(batch, dtype=torch.long)
    env.episode_length_buf = torch.zeros(batch, dtype=torch.long)
    env.previous_actions = torch.zeros(batch, 8)
    env.previous_vz = torch.zeros(batch)
    env.first_reward = torch.ones(batch, dtype=torch.bool)
    env.push_timer = torch.full((batch,), 2.)
    env.push_bounds = torch.tensor([[-.5, .5], [-.5, .5], [-.2, .2],
                                    [-.52, .52], [-.52, .52], [-.78, .78]])
    env.gravity = torch.tensor([0., 0., -1.])
    env.episode_rewards = torch.zeros(batch)
    env.total_transitions = env.total_physical_steps = 0
    env.total_resets, env.total_pushes = torch.tensor(batch), torch.tensor(0)
    env._closed = False
    return env


def test_sampler_failure_ema_and_distribution_matches_original_constants():
    sampler = TensorPhaseSampler(349, 10000, torch.Generator().manual_seed(3), "cpu")
    assert sampler.bin_count == 35
    phases = sampler.sample()
    assert 0 <= phases.min() and phases.max() < 349
    assert .18 < (phases == 0).float().mean() < .23
    sampler.record_failures(torch.tensor([0, 348, 348]), torch.tensor([True, True, False]))
    sampler.update()
    assert sampler.failed[0] == pytest.approx(.001)
    assert sampler.failed[-1] == pytest.approx(.001)
    assert sampler.current_failed.count_nonzero() == 0
    assert sampler.probabilities().sum() == pytest.approx(1.)


def test_held_phase_dt_reward_and_no_unnecessary_forward_event():
    env = task_double()
    obs, reward, term, trunc, extra = env.step(torch.zeros(2, 8))
    assert not term.any() and not trunc.any()
    assert env.batch.events == []
    torch.testing.assert_close(env.phases, torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(extra["reference_frames_for_action"], torch.zeros(2, dtype=torch.long))
    assert obs["policy"].shape == (2, 51) and obs["critic"].shape == (2, 120)
    weighted = sum(env.objective.weights[k]*v for k, v in extra["raw_reward_terms"].items())
    torch.testing.assert_close(reward, .02*weighted)
    assert extra["raw_reward_terms"]["height_vertical_acceleration"].count_nonzero() == 0


def test_motion_end_autoreset_preserves_raw_timeout_and_reset_history():
    env = task_double()
    env.phases[:] = 348
    env.previous_vz[:] = 20.
    env.first_reward[:] = False
    _, _, term, trunc, extra = env.step(torch.zeros(2, 8))
    assert trunc.all() and not term.any()
    assert env.batch.events == ["reset"]
    assert env.phases.count_nonzero() == 0 and env.first_reward.all()
    assert env.previous_vz.count_nonzero() == 0 and env.previous_actions.count_nonzero() == 0
    torch.testing.assert_close(extra["time_outs"], trunc)
    _, _, _, _, next_extra = env.step(torch.zeros(2, 8))
    assert next_extra["raw_reward_terms"]["height_vertical_acceleration"].count_nonzero() == 0


def test_reset_resamples_timer_before_push_and_ema_uses_failed_held_phase():
    env = task_double(training=True)
    env.phases[:] = torch.tensor([100, 200])
    env.push_timer[:] = 0.
    # First world is upside down, so it resets and must NOT receive old push.
    env.batch.qpos[0, 3:7] = torch.tensor([0., 1., 0., 0.])
    _, _, term, _, _ = env.step(torch.zeros(2, 8))
    assert term.tolist() == [True, False]
    assert env.batch.events == ["reset", "push"]
    assert env.push_timer.min() >= .98 and env.push_timer.max() <= 3.
    assert env.sampler.failed[100*35//349] == pytest.approx(.001)
    assert env.total_pushes == 1
    assert env.batch.qvel[0].count_nonzero() == 0
    assert env.batch.qvel[1, :6].count_nonzero() > 0


def test_large_xy_and_height_offsets_are_not_termination_gates():
    env = task_double()
    env.batch.qpos[:, :3] += torch.tensor([100., -100., 3.])
    _, _, terminated, _, _ = env.step(torch.zeros(2, 8))
    assert not terminated.any()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_raw_action_is_not_hidden_by_saturation(value):
    env = task_double()
    action = torch.zeros(2, 8)
    action[0, 0] = value
    with pytest.raises(FloatingPointError):
        env.step(action)


def test_metadata_identifies_device_reward_without_mutating_startup_contract():
    env = task_double()
    env._metadata_startup = {"reward_contract": {"kind": "numpy_oracle"}}
    env.batch.metadata = {"physics": "test_double"}
    report = env.metadata()
    assert report["reward_contract"]["kind"] == "torch_device_exact_height_smooth_tracking_objective"
    assert env._metadata_startup["reward_contract"]["kind"] == "numpy_oracle"
    assert report["host_task_adapter"] is False
    assert "not NumPy draw-for-draw" in report["random_stream"]


def test_gpu_task_has_only_tiny_event_readback_not_bulk_numpy_or_nonzero():
    tree = ast.parse((ROOT/"scripts/mjlab_gpu_ppo_env.py").read_text())
    task_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GpuTron1PPOEnv")
    methods = {n.name: n for n in task_class.body if isinstance(n, ast.FunctionDef)}
    for name in ("_reset_mask", "_push_mask", "observations", "_state", "_reference"):
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr in {"cpu", "numpy", "item", "tolist", "nonzero"}
                       for n in ast.walk(methods[name]))
    # One three-byte payload for event branching, never a state/action tensor.
    calls = [n.func.attr for n in ast.walk(methods["step"])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.count("cpu") == 1 and calls.count("tolist") == 1
    assert "numpy" not in calls and "nonzero" not in calls

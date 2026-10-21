"""CPU-only RSL3 interface test; not a physical or GPU speed benchmark."""
import copy
from importlib import metadata
from pathlib import Path
import sys

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from mjlab_gpu_ppo_env import TensorRslWrapper
from test_mjlab_gpu_ppo_env import task_double
from train_ppo_backend import ppo_config


def test_wrapper_keeps_all_rollout_tensors_on_the_environment_device():
    env = task_double(batch=8)
    wrapped = TensorRslWrapper(env)
    obs, _ = wrapped.reset()
    assert obs["policy"].shape == (8, 51) and obs["critic"].shape == (8, 120)
    action = torch.full((8, 8), 2.)
    obs, reward, done, extra = wrapped.step(action)
    assert all(value.device == env.device for value in
               (obs["policy"], obs["critic"], reward, done, extra["time_outs"]))
    assert done.dtype == torch.long
    torch.testing.assert_close(env.previous_actions, torch.ones_like(action))


def test_actual_unmodified_rsl3_runs_one_cpu_iteration(tmp_path):
    pytest.importorskip("rsl_rl")
    if metadata.version("rsl-rl-lib") != "3.0.1":
        pytest.skip("This integration contract is pinned to RSL3.0.1")
    from rsl_rl.runners import OnPolicyRunner
    env = task_double(batch=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        runner = OnPolicyRunner(TensorRslWrapper(env), copy.deepcopy(ppo_config()),
                                 log_dir=str(tmp_path), device="cpu")
        before = {key: value.clone() for key, value in runner.alg.policy.state_dict().items()}
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        assert runner.tot_timesteps == 8*24
        assert any(not torch.equal(before[key], value)
                   for key, value in runner.alg.policy.state_dict().items())
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()

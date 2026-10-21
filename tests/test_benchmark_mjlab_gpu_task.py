"""Asset/GPU-free checks for the opt-in real-CUDA parity/timing harness."""
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from benchmark_mjlab_gpu_task import (action_trace, capture_trace, compare_arrays,
                                     compare_traces, main, time_step_windows)


class FakeEnv:
    num_envs = 2

    def __init__(self, *, terminate_on=None):
        self.batch = self
        self.steps = 0
        self.terminate_on = terminate_on

    def observations(self):
        return {"policy": np.full((2, 51), self.steps, dtype=np.float32),
                "critic": np.full((2, 120), self.steps, dtype=np.float32)}

    def measure(self):
        return {"qpos": np.full((2, 15), self.steps, dtype=np.float32),
                "qvel": np.full((2, 14), self.steps, dtype=np.float32)}

    def step(self, actions):
        assert np.asarray(actions).shape == (2, 8)
        self.steps += 1
        return (self.observations(), np.ones(2, dtype=np.float32),
                np.full(2, self.steps == self.terminate_on), np.zeros(2, dtype=bool),
                {"raw_reward_terms": {"anchor": np.ones(2, dtype=np.float32)}})


def test_trace_actions_are_reproducible_finite_and_small():
    actions = action_trace(24, 512, 42)
    assert actions.dtype == np.float32
    assert actions.shape == (24, 512, 8)
    np.testing.assert_array_equal(actions, action_trace(24, 512, 42))
    assert not np.array_equal(actions, action_trace(24, 512, 43))
    assert np.max(np.abs(actions[..., :6])) <= .002
    assert np.max(np.abs(actions[..., 6:])) <= .02
    with pytest.raises(ValueError):
        action_trace(0, 4, 42)


def test_identical_checked_trace_passes_and_has_exact_initial_physics():
    actions = action_trace(8, 2, 42)
    host = capture_trace(FakeEnv(), actions)
    gpu = capture_trace(FakeEnv(), actions)
    report = compare_traces(host, gpu)
    assert report["passed"]
    assert report["complete_requested_trace"]
    assert report["uninterrupted_no_auto_resets"]
    assert report["comparisons"]["initial/qpos"]["atol"] == 0.
    assert report["comparisons"]["terminated"]["atol"] == 0.
    assert set(host["arrays"]) == {"qpos", "qvel", "policy", "critic", "reward",
                                   "terminated", "truncated", "raw/anchor"}


def test_auto_reset_invalidates_even_identical_traces():
    actions = action_trace(8, 2, 42)
    host = capture_trace(FakeEnv(terminate_on=3), actions)
    gpu = capture_trace(FakeEnv(terminate_on=3), actions)
    report = compare_traces(host, gpu)
    assert host["steps_completed"] == 3
    assert report["host_auto_reset_steps"] == [2]
    assert not report["passed"]
    assert not report["uninterrupted_no_auto_resets"]


def test_physics_difference_outside_tolerance_is_not_hidden_by_task_parity():
    actions = action_trace(2, 2, 42)
    host, gpu = capture_trace(FakeEnv(), actions), capture_trace(FakeEnv(), actions)
    gpu["arrays"]["qpos"][0, 0, 0] += .01
    report = compare_traces(host, gpu)
    assert not report["passed"]
    assert report["comparisons"]["qpos"]["elements_outside_tolerance"] == 1
    assert report["comparisons"]["reward"]["passed"]


def test_reward_term_name_mismatch_is_a_failure():
    actions = action_trace(2, 2, 42)
    host, gpu = capture_trace(FakeEnv(), actions), capture_trace(FakeEnv(), actions)
    del gpu["arrays"]["raw/anchor"]
    report = compare_traces(host, gpu)
    assert not report["passed"]
    assert report["missing_from_gpu"] == ["raw/anchor"]


def test_nonfinite_and_shape_mismatch_never_pass():
    assert compare_arrays([1.], [1.], atol=0., rtol=0.)["passed"]
    assert not compare_arrays([np.nan], [np.nan], atol=1., rtol=1.)["passed"]
    assert not compare_arrays([1.], [[1.]], atol=1., rtol=1.)["passed"]


def test_timing_warms_up_and_syncs_only_at_window_boundaries():
    env, syncs = FakeEnv(), []
    times = iter([10., 12., 20., 24., 30., 33.])
    prepared = []

    def prepare(action):
        prepared.append(1)
        return action

    report = time_step_windows(env, action_trace(24, 2, 42), warmup_steps=8, repeats=3,
                               synchronize=lambda: syncs.append(env.steps),
                               prepare_action=prepare, timer=lambda: next(times))
    assert env.steps == 8+3*24
    assert len(prepared) == 24
    assert syncs == [8, 32, 32, 56, 56, 80]
    assert report["seconds_per_window"] == [2., 4., 3.]
    assert report["median_transitions_per_s"] == 16.
    assert report["cuda_synchronizations_in_timing_harness"] == 6
    assert report["median_environment_step_ms"] == 125.


def test_existing_directory_is_not_overwritten_before_loading_gpu_packages(tmp_path):
    with pytest.raises(FileExistsError, match="Never overwrite"):
        main(["--model", "unavailable.xml", "--motion", "unavailable.npz",
              "--output-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("keyword,value", [("--num-envs", "0"), ("--steps", "0"),
                                            ("--warmup-steps", "-1"), ("--mu", "nan")])
def test_invalid_counts_and_mu_are_rejected_without_cuda(keyword, value, tmp_path):
    with pytest.raises(SystemExit):
        main(["--model", "unavailable.xml", "--motion", "unavailable.npz",
              "--output-dir", str(tmp_path/"new"), keyword, value])

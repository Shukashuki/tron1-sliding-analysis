"""No Torch, CUDA or assets required for the optional profiling harness tests."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from profile_mjlab_gpu_task import (MethodEventRecorder, STAGES, instrument_methods,
                                   main, profile_window)


class FakeEvent:
    def __init__(self):
        self.streams = []

    def record(self, stream):
        self.streams.append(stream)

    def elapsed_time(self, other):
        assert self.streams and other.streams
        return 2.


def recorder():
    ticks = iter(np.arange(2000)*.001)
    return MethodEventRecorder(FakeEvent, lambda: "current-stream", timer=lambda: next(ticks))


def make_env():
    class Target:
        def step(self, *args):
            return "original"

    batch, objective, env = SimpleNamespace(sim=Target()), SimpleNamespace(), SimpleNamespace(num_envs=2)
    env.batch, env.objective = batch, objective
    for _, target_name, method in STAGES:
        target = {"sim": batch.sim, "batch": batch, "objective": objective, "env": env}[target_name]
        if not hasattr(target, method):
            setattr(target, method, lambda *args, **kwargs: "original")
    env.calls = 0

    def step(action):
        env.calls += 1
        batch.sim.step()
        batch.sim.forward()
        objective.evaluate()
        env.observations()

    env.step = step
    return env


def test_recorder_records_current_stream_and_inclusive_cpu_wall_without_sync():
    records = recorder()
    fn = records.wrap("test", lambda value: value+1)
    assert fn(5) == 6
    result = records.summarize_after_synchronization()["test"]
    assert result["calls"] == 1
    assert result["device_event_elapsed_ms"] == 2.
    assert result["cpu_submission_wall_ms"] == pytest.approx(1.)
    assert result["inclusive_non_additive"]
    begin, end, _ = records.samples["test"][0]
    assert begin.streams == end.streams == ["current-stream"]


def test_instrumentation_is_reversible_including_inherited_methods_after_error():
    env, records = make_env(), recorder()
    original_owned = env.objective.evaluate
    assert "step" not in env.batch.sim.__dict__
    with pytest.raises(RuntimeError, match="test"):
        with instrument_methods(env, records):
            assert env.batch.sim.step() == "original"
            assert env.objective.evaluate() == "original"
            raise RuntimeError("test")
    assert env.objective.evaluate is original_owned
    assert "step" not in env.batch.sim.__dict__
    assert env.batch.sim.step() == "original"


def test_profile_excludes_warmup_and_only_adds_two_boundary_syncs():
    env, records, syncs = make_env(), recorder(), []
    ticks = iter([10., 12.])
    report = profile_window(env, [np.zeros((2, 8))]*8, records, warmup_steps=2,
                            synchronize=lambda: syncs.append(env.calls), timer=lambda: next(ticks))
    assert env.calls == 10
    assert syncs == [2, 10]
    assert report["window_wall_s"] == 2.
    assert report["stages"]["physics.step"]["calls"] == 8
    assert report["stages"]["task.evaluate"]["calls"] == 8
    assert not report["stage_totals_are_additive"]
    assert "not GPU kernel busy time" in report["timing_semantics"]["device_event_elapsed_ms"]
    assert "earlier physics" in report["timing_semantics"]["status_read"]


def test_recorder_keeps_exception_and_records_final_event():
    records = recorder()

    def fail():
        raise ValueError("physics failure")

    with pytest.raises(ValueError, match="physics failure"):
        records.wrap("failed", fail)()
    assert records.summarize_after_synchronization()["failed"]["calls"] == 1


def test_existing_evidence_is_preserved_without_loading_gpu_packages(tmp_path):
    with pytest.raises(FileExistsError, match="Never overwrite"):
        main(["--model", "missing.xml", "--motion", "missing.npz", "--output-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("flag,value", [("--steps", "0"), ("--num-envs", "-1"),
                                        ("--warmup-steps", "-1"), ("--mu", "nan")])
def test_invalid_cli_counts_or_friction_do_not_start_cuda(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        main(["--model", "missing.xml", "--motion", "missing.npz",
              "--output-dir", str(tmp_path/"new"), flag, value])

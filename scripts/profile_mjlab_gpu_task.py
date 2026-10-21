"""Opt-in bounded CUDA-event profile of the unchanged GPU-resident TRON1 task.

This is diagnostic instrumentation, not a performance optimization or PPO run.
CUDA events use the current stream inside each method: plant calls execute on
the existing Warp/Torch physics stream, task calls on their calling stream.
Nested/inclusive stage measurements MUST NOT be summed. Event intervals can
include GPU queue idle gaps while the CPU dispatches work; they are NOT kernel
busy-time measurements. CPU wall intervals include blocking synchronization,
and a status read can wait for earlier solver work rather than consume that
time checking status. Event allocation/recording itself also perturbs timing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from benchmark_mjlab_gpu_task import action_trace


ROOT = Path(__file__).resolve().parents[1]
MJLAB_PIN = "4360096b49c1bcc478668c11ce7d458db747e49d"
STAGES = (
    ("physics.step", "sim", "step"),
    ("physics.forward", "sim", "forward"),
    ("motor.torques", "batch", "_motor_torques"),
    ("health.record_physics_status", "batch", "_record_physics_status"),
    ("contact.forces", "batch", "_body_contact_forces_into"),
    ("health.flag_nonfinite", "batch", "_flag_nonfinite"),
    ("health.raise_if_status_failed", "batch", "_raise_if_status_failed"),
    ("task.hold_reference", "objective", "hold_reference"),
    ("task.evaluate", "objective", "evaluate"),
    ("task.observations", "env", "observations"),
)


class MethodEventRecorder:
    """Record without synchronizing; resolve events only after a boundary sync."""

    def __init__(self, event_factory, current_stream, *, timer=time.perf_counter):
        self.event_factory = event_factory
        self.current_stream = current_stream
        self.timer = timer
        self.samples = {}

    def wrap(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            begin, end = self.event_factory(), self.event_factory()
            stream = self.current_stream()
            begin.record(stream)
            started = self.timer()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = (self.timer()-started)*1000.
                end.record(self.current_stream())
                self.samples.setdefault(name, []).append((begin, end, elapsed))
        return measured

    def summarize_after_synchronization(self):
        report = {}
        for name, values in self.samples.items():
            device_ms = np.array([start.elapsed_time(end) for start, end, _ in values])
            wall_ms = np.array([wall for _, _, wall in values])
            if (not np.isfinite(device_ms).all() or not np.isfinite(wall_ms).all()
                    or np.any(device_ms < 0.) or np.any(wall_ms < 0.)):
                raise ValueError("Invalid nonfinite or negative profile duration: "+name)
            report[name] = {
                "calls": len(values),
                "device_event_elapsed_ms": float(device_ms.sum()),
                "device_event_mean_ms": float(device_ms.mean()),
                "device_event_min_ms": float(device_ms.min()),
                "device_event_max_ms": float(device_ms.max()),
                "cpu_submission_wall_ms": float(wall_ms.sum()),
                "cpu_submission_wall_mean_ms": float(wall_ms.mean()),
                "cpu_submission_wall_max_ms": float(wall_ms.max()),
                "inclusive_non_additive": True,
            }
        return report


@contextmanager
def instrument_methods(env, recorder):
    """Patch only this instance graph, and restore it even if a step fails."""
    objects = {"sim": env.batch.sim, "batch": env.batch,
               "objective": env.objective, "env": env}
    patched = []
    try:
        for name, target_name, method in STAGES:
            target = objects[target_name]
            original = getattr(target, method)
            namespace = getattr(target, "__dict__", {})
            owned = method in namespace
            original_owned = namespace.get(method)
            setattr(target, method, recorder.wrap(name, original))
            patched.append((target, method, original_owned if owned else original, owned))
        yield
    finally:
        for target, method, original, owned in reversed(patched):
            if owned:
                setattr(target, method, original)
            else:
                delattr(target, method)


def profile_window(env, actions, recorder, *, warmup_steps, synchronize,
                   timer=time.perf_counter):
    """No extra step-level synchronizations; the task's own checks remain intact."""
    if warmup_steps < 0 or len(actions) < 1:
        raise ValueError("Nonnegative warmup and positive profile length required")
    for index in range(warmup_steps):
        env.step(actions[index % len(actions)])
    synchronize()
    with instrument_methods(env, recorder):
        started = timer()
        for action in actions:
            env.step(action)
        synchronize()
        elapsed = timer()-started
    stages = recorder.summarize_after_synchronization()
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Positive finite profile wall time required")
    return {"window_wall_s": float(elapsed), "steps": len(actions),
            "num_envs": env.num_envs, "warmup_steps_excluded": warmup_steps,
            "explicit_device_synchronizations": 2,
            "stages": stages,
            "stage_totals_are_additive": False,
            "timing_semantics": {
                "cpu_submission_wall_ms": "Inclusive CPU method wall time, INCLUDING blocking waits for previously queued device work; not CPU compute-only",
                "device_event_elapsed_ms": "Inclusive current-stream event interval, INCLUDING queue idle gaps caused by host dispatch; not GPU kernel busy time",
                "nested_stages": "Overlapping/nested stage intervals MUST NOT be summed or interpreted as exclusive percentages",
                "event_overhead": "Per-call CUDA event allocation/recording perturbs dispatch and is included in whole-window wall time",
                "status_read": "Large CPU wall time for status read may be waiting on earlier physics; do not attribute that wait to status-check computation"}}


def _write_report(output, report):
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "initializing", "training_run": False,
              "training_speedup_claimed": False, "optimization_applied": False,
              "physics_or_task_parameters_changed": False,
              "profile_kind": "instrumented bounded environment-only diagnostic",
              "device": args.device, "seed": args.seed, "fixed_mu": args.mu,
              "dr_mode": "none", "training": False,
              "physics_schedule": "unchanged four step+full-forward pairs per interval",
              "mjlab_commit": MJLAB_PIN}
    _write_report(output, report)
    env = None
    try:
        report["input_sha256"] = {
            name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
            for name in ("model", "motion", "contract", "task_profile")}
        report["source_sha256"] = {
            name: hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest()
            for name in ("profile_mjlab_gpu_task.py", "mjlab_gpu_ppo_env.py",
                         "mjlab_tron1_gpu.py", "mjlab_tron1_env.py", "tron1_torch_task.py")}
        actual_pin = subprocess.check_output(
            ["git", "-C", str(args.mjlab_source), "rev-parse", "HEAD"], text=True).strip()
        if actual_pin != MJLAB_PIN:
            raise ValueError(f"Expected pinned mjlab {MJLAB_PIN}, found {actual_pin}")
        sys.path.insert(0, str(Path(args.mjlab_source).resolve()/"src"))
        import torch
        if not args.device.startswith("cuda:") or not torch.cuda.is_available():
            raise RuntimeError("Explicit real CUDA device required; no CPU fallback")
        from mjlab_gpu_ppo_env import GpuTron1PPOEnv
        env = GpuTron1PPOEnv(args.model, args.motion, args.contract, args.task_profile,
                             num_envs=args.num_envs, seed=args.seed, training=False,
                             dr_mode="none", device=args.device, fixed_mu=args.mu)
        actions = [torch.as_tensor(value, device=args.device)
                   for value in action_trace(args.steps, args.num_envs, args.seed)]
        recorder = MethodEventRecorder(
            lambda: torch.cuda.Event(enable_timing=True),
            lambda: torch.cuda.current_stream(args.device))
        report["status"] = "profiling"
        _write_report(output, report)
        report["profile"] = profile_window(
            env, actions, recorder, warmup_steps=args.warmup_steps,
            synchronize=lambda: torch.cuda.synchronize(args.device))
        report["environment_metadata"] = env.metadata()
        report["runtime"] = {
            "gpu_name": torch.cuda.get_device_name(args.device),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "packages": {name: package_metadata.version(name) for name in
                         ("torch", "mujoco", "mujoco-warp", "warp-lang")}}
        report["status"] = "completed_instrumented_diagnostic_not_training_benchmark"
        _write_report(output, report)
        return report
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        _write_report(output, report)
        raise
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=ROOT/"config/nominal_actuator_contract.json")
    parser.add_argument("--task-profile", type=Path, default=ROOT/"config/car_drift_height_smooth_task.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mjlab-source", type=Path, default=ROOT/"upstream/mjlab")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu", type=float, default=.7)
    args = parser.parse_args(argv)
    if min(args.num_envs, args.steps) < 1 or args.warmup_steps < 0:
        parser.error("Positive environment/step counts and nonnegative warmup required")
    if not np.isfinite(args.mu) or args.mu <= 0:
        parser.error("Finite positive fixed friction required")
    if args.output_dir.exists():
        raise FileExistsError("Never overwrite existing diagnostic evidence: "+str(args.output_dir))
    report = run(args)
    print(json.dumps({"status": report["status"],
                      "window_wall_s": report["profile"]["window_wall_s"],
                      "training_speedup_claimed": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

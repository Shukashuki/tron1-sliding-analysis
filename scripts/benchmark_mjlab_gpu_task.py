"""Compare the host task bridge with the GPU-resident task on real mjlab physics.

Both branches use the SAME pinned MuJoCo-Warp solver, not CPU-vs-GPU physics.
A small fixed-mu deterministic trace verifies task/physics agreement first.
Separate larger timing windows measure environment stepping only: construction,
JIT, trace copies, PPO updates and initial warmup are excluded. No training-speed
claim can be inferred from this environment-only microbenchmark.

CUDA work is opt-in through this CLI; importing the module requires no Torch,
Warp, mjlab, robot asset or GPU. Every run uses a fresh output directory.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MJLAB_PIN = "4360096b49c1bcc478668c11ce7d458db747e49d"
# Host task arithmetic is float64, while GPU task arithmetic is float32. These
# are explicit bounded-trace tolerances, not a claim of bitwise equivalence.
TOLERANCES = {
    "qpos": (2.e-5, 1.e-5), "qvel": (2.e-4, 1.e-5),
    "policy": (5.e-4, 1.e-5), "critic": (5.e-4, 1.e-5),
    "reward": (1.e-4, 1.e-5), "raw_reward_terms": (4.e-3, 1.e-5),
}


def as_numpy(value):
    """Detach a diagnostic snapshot explicitly, outside timed GPU step loops."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def action_trace(steps, num_envs, seed):
    """Small reproducible applied actions, shared verbatim between branches."""
    if min(steps, num_envs) < 1:
        raise ValueError("Positive trace length and environment count required")
    rng = np.random.default_rng(seed)
    result = np.zeros((steps, num_envs, 8), dtype=np.float32)
    result[..., :6] = rng.uniform(-.002, .002, (steps, num_envs, 6))
    result[..., 6:] = rng.uniform(-.02, .02, (steps, num_envs, 2))
    return result


def _snapshot(env, obs):
    measured = env.batch.measure()
    return {"qpos": as_numpy(measured["qpos"]), "qvel": as_numpy(measured["qvel"]),
            "policy": as_numpy(obs["policy"]), "critic": as_numpy(obs["critic"])}


def capture_trace(env, actions, prepare_action=lambda value: value):
    """Capture a short checked trace; any auto-reset invalidates its parity use."""
    initial = _snapshot(env, env.observations())
    samples = {name: [] for name in ("qpos", "qvel", "policy", "critic", "reward",
                                      "terminated", "truncated")}
    raw_terms, reset_steps = {}, []
    for index, action in enumerate(actions):
        obs, reward, terminated, truncated, extras = env.step(prepare_action(action))
        current = _snapshot(env, obs)
        current.update(reward=as_numpy(reward), terminated=as_numpy(terminated),
                       truncated=as_numpy(truncated))
        for name, value in current.items():
            samples[name].append(value)
        terms = extras["raw_reward_terms"]
        if raw_terms and set(terms) != set(raw_terms):
            raise ValueError("Raw reward term names changed during the trace")
        for name, value in terms.items():
            raw_terms.setdefault(name, []).append(as_numpy(value))
        if np.any(current["terminated"] | current["truncated"]):
            reset_steps.append(index)
            # The env already auto-reset. Do not claim later states have the
            # same uninterrupted solver history or adaptive reset RNG stream.
            break
    arrays = {name: np.asarray(value) for name, value in samples.items()}
    arrays.update({"raw/"+name: np.asarray(value) for name, value in raw_terms.items()})
    return {"initial": initial, "arrays": arrays, "auto_reset_steps": reset_steps,
            "steps_completed": len(samples["reward"]), "steps_requested": len(actions)}


def compare_arrays(expected, actual, *, atol, rtol):
    """Finite shape-strict comparison with inspectable error magnitudes."""
    left, right = np.asarray(expected), np.asarray(actual)
    result = {"expected_shape": list(left.shape), "actual_shape": list(right.shape),
              "atol": float(atol), "rtol": float(rtol)}
    if left.shape != right.shape:
        return {**result, "passed": False, "reason": "shape_mismatch"}
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return {**result, "passed": False, "reason": "nonfinite"}
    delta = np.abs(left.astype(np.float64)-right.astype(np.float64))
    limit = atol+rtol*np.abs(left.astype(np.float64))
    return {**result, "passed": bool(np.all(delta <= limit)),
            "max_abs_error": float(np.max(delta, initial=0.)),
            "rms_error": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.,
            "elements_outside_tolerance": int(np.count_nonzero(delta > limit))}


def compare_traces(host, gpu):
    comparisons = {}
    for name in ("qpos", "qvel", "policy", "critic"):
        # Both branches must begin at the exact same uploaded float32 state.
        tolerance = (0., 0.) if name in ("qpos", "qvel") else TOLERANCES[name]
        comparisons["initial/"+name] = compare_arrays(host["initial"][name],
                                                     gpu["initial"][name],
                                                     atol=tolerance[0], rtol=tolerance[1])
    host_names, gpu_names = set(host["arrays"]), set(gpu["arrays"])
    for name in sorted(host_names & gpu_names):
        if name in ("terminated", "truncated"):
            atol, rtol = 0., 0.
        else:
            atol, rtol = TOLERANCES["raw_reward_terms" if name.startswith("raw/") else name]
        comparisons[name] = compare_arrays(host["arrays"][name], gpu["arrays"][name],
                                           atol=atol, rtol=rtol)
    uninterrupted = not host["auto_reset_steps"] and not gpu["auto_reset_steps"]
    complete = (host["steps_completed"] == host["steps_requested"]
                and gpu["steps_completed"] == gpu["steps_requested"])
    passed = (uninterrupted and complete and host_names == gpu_names
              and all(value["passed"] for value in comparisons.values()))
    return {"passed": bool(passed), "same_recorded_fields": host_names == gpu_names,
            "missing_from_gpu": sorted(host_names-gpu_names),
            "extra_from_gpu": sorted(gpu_names-host_names),
            "uninterrupted_no_auto_resets": uninterrupted,
            "host_auto_reset_steps": host["auto_reset_steps"],
            "gpu_auto_reset_steps": gpu["auto_reset_steps"],
            "complete_requested_trace": complete, "comparisons": comparisons,
            "precision_note": "Host task float64 vs GPU task float32; physics is MuJoCo-Warp float32 in BOTH branches"}


def time_step_windows(env, actions, *, warmup_steps, repeats, synchronize,
                      prepare_action=lambda value: value, timer=time.perf_counter):
    """Bounded same-shape step windows, synchronizing only at window boundaries."""
    if warmup_steps < 0 or repeats < 1 or len(actions) < 1:
        raise ValueError("Nonnegative warmup, positive repeats and steps required")
    prepared = [prepare_action(action) for action in actions]
    for index in range(warmup_steps):
        env.step(prepared[index % len(prepared)])
    elapsed = []
    for _ in range(repeats):
        synchronize()
        started = timer()
        for action in prepared:
            # Deliberately do not inspect or copy returned device tensors here.
            env.step(action)
        synchronize()
        elapsed.append(float(timer()-started))
    median = float(np.median(elapsed))
    transitions = int(env.num_envs*len(actions))
    if not np.isfinite(elapsed).all() or min(elapsed) <= 0:
        raise ValueError("Invalid nonpositive/nonfinite measured duration")
    return {"seconds_per_window": elapsed, "median_window_s": median,
            "environment_steps_per_window": len(actions), "num_envs": env.num_envs,
            "transitions_per_window": transitions,
            "median_environment_step_ms": median/len(actions)*1000.,
            "median_transitions_per_s": transitions/median,
            "warmup_environment_steps_excluded": warmup_steps,
            "cuda_synchronizations_in_timing_harness": repeats*2,
            "step_internal_synchronizations": "Implementation dependent; included in elapsed time"}


def _new_environment(args, implementation, num_envs, dr_mode):
    if implementation == "host":
        from mjlab_tron1_env import MjlabWarpBatch
        from mujoco_ppo_env import MujocoPPOEnv
        return MujocoPPOEnv(args.model, args.motion, args.contract, args.task_profile,
                            num_envs=num_envs, seed=args.seed, training=dr_mode != "none",
                            dr_mode=dr_mode, fixed_mu=args.mu,
                            batch_factory=lambda *values: MjlabWarpBatch(*values, device=args.device))
    from mjlab_gpu_ppo_env import GpuTron1PPOEnv
    return GpuTron1PPOEnv(args.model, args.motion, args.contract, args.task_profile,
                          num_envs=num_envs, seed=args.seed, training=dr_mode != "none",
                          dr_mode=dr_mode, fixed_mu=args.mu, device=args.device)


def _write_report(output, report):
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "initializing", "performance_claimed": False,
              "training_speedup_claimed": False, "device": args.device,
              "seed": args.seed, "parity_fixed_mu": args.mu,
              "same_physics_both_branches": "Pinned mjlab MuJoCo-Warp CUDA",
              "order": "host parity, GPU parity, host timing, GPU timing; one environment alive at a time"}
    _write_report(output, report)
    try:
        report["input_sha256"] = {
            name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
            for name in ("model", "motion", "contract", "task_profile")}
        report["source_sha256"] = {
            name: hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest()
            for name in ("benchmark_mjlab_gpu_task.py", "mjlab_gpu_ppo_env.py",
                         "mjlab_tron1_env.py", "mujoco_ppo_env.py")}
        actual_pin = subprocess.check_output(
            ["git", "-C", str(args.mjlab_source), "rev-parse", "HEAD"], text=True).strip()
        if actual_pin != MJLAB_PIN:
            raise ValueError(f"Expected pinned mjlab {MJLAB_PIN}, found {actual_pin}")
        sys.path.insert(0, str(Path(args.mjlab_source).resolve()/"src"))
        import torch
        if not args.device.startswith("cuda:") or not torch.cuda.is_available():
            raise RuntimeError("A real explicitly selected CUDA device is required; no CPU fallback")
        report["runtime"] = {
            "gpu_name": torch.cuda.get_device_name(args.device),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "environment_threads": {name: os.environ.get(name) for name in
                                     ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "packages": {name: package_metadata.version(name) for name in
                         ("torch", "mujoco", "mujoco-warp", "warp-lang")}}
        synchronize = lambda: torch.cuda.synchronize(args.device)
        gpu_action = lambda action: torch.as_tensor(action, device=args.device)
        traces = {}
        actions = action_trace(args.parity_steps, args.parity_num_envs, args.seed)
        for implementation in ("host", "gpu"):
            report["status"] = "running_"+implementation+"_parity"
            _write_report(output, report)
            env = _new_environment(args, implementation, args.parity_num_envs, "none")
            try:
                synchronize()
                traces[implementation] = capture_trace(
                    env, actions, gpu_action if implementation == "gpu" else lambda value: value)
                report[implementation+"_metadata"] = env.metadata()
                synchronize()
            finally:
                env.close()
        arrays = {"shared_actions": actions}
        for implementation, trace in traces.items():
            arrays.update({implementation+"/initial/"+name: value
                           for name, value in trace["initial"].items()})
            arrays.update({implementation+"/"+name: value for name, value in trace["arrays"].items()})
        np.savez_compressed(output/"parity_arrays.npz", **arrays)
        report["parity"] = compare_traces(traces["host"], traces["gpu"])
        if not report["parity"]["passed"]:
            report["status"] = "failed_parity_timing_not_run"
            _write_report(output, report)
            return report
        timings = {}
        perf_actions = action_trace(args.steps, args.num_envs, args.seed+1)
        for implementation in ("host", "gpu"):
            report["status"] = "running_"+implementation+"_timing"
            _write_report(output, report)
            env = _new_environment(args, implementation, args.num_envs, args.performance_dr_mode)
            try:
                timings[implementation] = time_step_windows(
                    env, perf_actions, warmup_steps=args.warmup_steps, repeats=args.repeats,
                    synchronize=synchronize,
                    prepare_action=gpu_action if implementation == "gpu" else lambda value: value)
            finally:
                synchronize()
                env.close()
        report.update(status="passed_parity_and_measured_environment_timing",
                      environment_timing=timings, performance_claimed=True,
                      measured_environment_step_speedup=(timings["host"]["median_window_s"]/
                                                         timings["gpu"]["median_window_s"]),
                      timing_scope="Same shaped step traces; no PPO, no initialization/JIT, no diagnostic device-to-host trace copies",
                      timing_dr_mode=args.performance_dr_mode,
                      timing_task_trace_parity_claimed=False)
        _write_report(output, report)
        return report
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        _write_report(output, report)
        raise


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
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parity-num-envs", type=int, default=4)
    parser.add_argument("--parity-steps", type=int, default=8)
    parser.add_argument("--performance-dr-mode", choices=("none", "mjlab_v1p4"), default="mjlab_v1p4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu", type=float, default=.7)
    args = parser.parse_args(argv)
    if min(args.num_envs, args.steps, args.repeats, args.parity_num_envs, args.parity_steps) < 1:
        parser.error("Use positive environment, step and repeat counts")
    if args.warmup_steps < 0 or not np.isfinite(args.mu) or args.mu <= 0:
        parser.error("Use nonnegative warmup count and finite positive friction")
    if args.output_dir.exists():
        raise FileExistsError("Never overwrite existing benchmark evidence: "+str(args.output_dir))
    report = run(args)
    print(json.dumps({name: report[name] for name in ("status", "performance_claimed",
                                                     "training_speedup_claimed",
                                                     "measured_environment_step_speedup")
                      if name in report}), flush=True)
    return 0 if report["status"] == "passed_parity_and_measured_environment_timing" else 1


if __name__ == "__main__":
    raise SystemExit(main())

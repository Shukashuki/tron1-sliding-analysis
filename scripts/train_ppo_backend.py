"""Train the same RSL-RL 3.0.1 TRON1 task on explicitly named physics backends.

mjlab supplies MuJoCo-Warp simulation, not its RSL5 runner or G1 task defaults.
The host NumPy task bridge is deliberately unoptimized; timings are not a GPU
throughput benchmark. Use a new output directory for every run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MJLAB_PIN = "4360096b49c1bcc478668c11ce7d458db747e49d"


def ppo_config(seed=42, save_interval=100):
    return {
        "seed": seed, "num_steps_per_env": 24, "save_interval": save_interval,
        "logger": "tensorboard", "experiment_name": "tron1_backend_comparison",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCritic", "init_noise_std": .5, "noise_std_type": "log",
            "actor_obs_normalization": True, "critic_obs_normalization": True,
            "actor_hidden_dims": [256, 128, 128], "critic_hidden_dims": [256, 128, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO", "value_loss_coef": 1., "use_clipped_value_loss": True,
            "clip_param": .2, "entropy_coef": .005, "num_learning_epochs": 5,
            "num_mini_batches": 4, "learning_rate": .001, "schedule": "adaptive",
            "gamma": .99, "lam": .95, "desired_kl": .01, "max_grad_norm": 1.,
        },
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_state_hash(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class NumpyRslWrapper:
    """Minimal VecEnv protocol; official RSL timeout bootstrap stays unchanged."""
    def __init__(self, env, device):
        import torch
        self.env, self.device = env, torch.device(device)
        self.num_envs, self.num_actions = env.num_envs, env.num_actions
        self.max_episode_length = env.max_episode_length

    def _obs(self, values):
        import torch
        from tensordict import TensorDict
        return TensorDict({key: torch.as_tensor(value, device=self.device)
                           for key, value in values.items()}, batch_size=[self.num_envs])

    @property
    def episode_length_buf(self):
        import torch
        return torch.as_tensor(self.env.episode_length_buf, device=self.device)

    def get_observations(self):
        return self._obs(self.env.observations())

    def reset(self):
        return self._obs(self.env.reset()), {}

    def step(self, actions):
        import torch
        obs, rew, terminated, truncated, extras = self.env.step(actions.detach().cpu().numpy())
        return (self._obs(obs), torch.as_tensor(rew, device=self.device),
                torch.as_tensor(terminated | truncated, device=self.device, dtype=torch.long),
                {"time_outs": torch.as_tensor(truncated, device=self.device), "log": extras["log"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjlab", "physx"), required=True)
    parser.add_argument("--mjlab-task-implementation", choices=("gpu", "host"), default="gpu",
                        help="gpu keeps task tensors on CUDA; host is the frozen calibration oracle")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=ROOT/"config/nominal_actuator_contract.json")
    parser.add_argument("--task-profile", type=Path, default=ROOT/"config/car_drift_height_smooth_task.json")
    parser.add_argument("--mjlab-source", type=Path, default=ROOT/"upstream/mjlab")
    parser.add_argument("--tron1-source", type=Path, default=ROOT/"upstream/tron1-learning-from-human")
    parser.add_argument("--beyondmimic-source", type=Path)
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--smoke-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.smoke_steps < 0:
        parser.error("Positive environment/iteration counts required")
    if args.output_dir.exists():
        parser.error("Use a nonexistent output directory; never overwrite an experiment")
    if importlib.metadata.version("rsl-rl-lib") != "3.0.1":
        parser.error("This matched algorithm adapter requires rsl-rl-lib==3.0.1")
    args.output_dir.mkdir(parents=True)
    started, env, app, accumulator = time.monotonic(), None, None, None
    report = {"status": "initializing", "backend": args.backend,
              "performance_claimed": False, "requested_iterations": args.iterations}
    try:
        if args.backend == "physx":
            if args.asset_path is None or args.beyondmimic_source is None:
                raise ValueError("PhysX requires explicit existing USD and BeyondMimic source")
            os.environ["TRON1_BEYONDMIMIC_ROOT"] = str(args.beyondmimic_source.resolve())
            sys.path.insert(0, str(args.tron1_source.resolve()))
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=True, device=args.device).app
        import torch
        from rsl_rl.runners import OnPolicyRunner
        if args.backend == "mjlab":
            actual_pin = subprocess.check_output(
                ["git", "-C", str(args.mjlab_source), "rev-parse", "HEAD"], text=True).strip()
            if actual_pin != MJLAB_PIN:
                raise ValueError(f"mjlab checkout is not pinned v1.4.0: {actual_pin}")
            sys.path.insert(0, str(args.mjlab_source.resolve()/"src"))
            if args.mjlab_task_implementation == "gpu":
                from mjlab_gpu_ppo_env import GpuTron1PPOEnv, TensorRslWrapper
                env = GpuTron1PPOEnv(args.model, args.motion, args.contract, args.task_profile,
                                     num_envs=args.num_envs, seed=args.seed, training=True,
                                     dr_mode="mjlab_v1p4", device=args.device)
                wrapped = TensorRslWrapper(env)
            else:
                from mjlab_tron1_env import MjlabWarpBatch
                from mujoco_ppo_env import MujocoPPOEnv
                env = MujocoPPOEnv(args.model, args.motion, args.contract, args.task_profile,
                                   num_envs=args.num_envs, seed=args.seed, training=True,
                                   dr_mode="mjlab_v1p4", batch_factory=lambda *a:
                                   MjlabWarpBatch(*a, device=args.device))
                wrapped = NumpyRslWrapper(env, args.device)
        else:
            from isaaclab.envs import ManagerBasedRLEnv
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from training.tron1_tracking import make_env_cfg
            from training.tron1_drift import load_task_profile, install_drift_reward_sampling
            from physx_mjlab_recipe import apply_mjlab_recipe, recipe_audit
            task = load_task_profile(args.task_profile)
            cfg = make_env_cfg(args.motion, args.asset_path, args.num_envs, args.device,
                               task_profile=task)
            cfg.seed = args.seed
            from height_smoothness import install_height_smoothness
            height_contract = install_height_smoothness(
                cfg, json.loads(args.task_profile.read_text())["height_regularization"])
            if cfg.terminations.anchor_pos is not None or cfg.terminations.ee_body_pos is not None:
                raise AssertionError("Absolute position/height termination must remain disabled")
            apply_mjlab_recipe(cfg)
            env = ManagerBasedRLEnv(cfg=cfg)
            accumulator = install_drift_reward_sampling(env, task)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        obs, _ = wrapped.reset()
        for _ in range(args.smoke_steps):
            obs, rew, done, extra = wrapped.step(torch.zeros(args.num_envs, 8, device=args.device))
            if not all(torch.isfinite(obs[k]).all() for k in ("policy", "critic")) or not torch.isfinite(rew).all():
                raise FloatingPointError("Nonfinite observation/reward in physics smoke")
        if tuple(obs["policy"].shape) != (args.num_envs, 51) or tuple(obs["critic"].shape) != (args.num_envs, 120):
            raise AssertionError("Observation contract changed")
        wrapped.reset()
        config = ppo_config(args.seed, args.save_interval)
        write_json(args.output_dir/"runner_config.json", config)
        metadata = env.metadata() if args.backend == "mjlab" else recipe_audit(env)
        if args.backend == "physx":
            metadata["height_regularization"] = height_contract
            metadata["reward_terms"] = {name: float(env.reward_manager.get_term_cfg(name).weight)
                                         for name in env.reward_manager.active_terms}
            metadata["termination_terms"] = list(env.termination_manager.active_terms)
            if len(metadata["reward_terms"]) != 15 or "height_vertical_acceleration" not in metadata["reward_terms"]:
                raise AssertionError("PhysX must resolve the same 15-term height-smooth objective")
        manifest = {"backend": args.backend, "mjlab_commit": MJLAB_PIN,
                    "task_implementation": args.mjlab_task_implementation if args.backend == "mjlab" else "isaaclab_gpu",
                    "algorithm": "unmodified RSL-RL3.0.1 PPO; custom task wrapper, not mjlab's RSL5 runner",
                    "seed": args.seed, "host": platform.node(), "python": sys.version,
                    "gpu": torch.cuda.get_device_name(args.device),
                    "versions": {n: importlib.metadata.version(n) for n in
                                 ("torch", "rsl-rl-lib", "mujoco", "mujoco-warp", "warp-lang")},
                    "environment": metadata, "smoke_steps": args.smoke_steps,
                    "inputs": {k: {"path": str(v.resolve()), "sha256": source_hash(v)}
                               for k, v in {"model": args.model, "motion": args.motion,
                                            "contract": args.contract, "task": args.task_profile}.items()},
                    "iterations": args.iterations, "num_envs": args.num_envs,
                    "transitions_requested": args.num_envs*24*args.iterations,
                    "source_sha256": {p.name: source_hash(p) for p in
                                      [Path(__file__), ROOT/"scripts/mujoco_ppo_env.py",
                                       ROOT/"scripts/mjlab_tron1_env.py", ROOT/"scripts/physx_mjlab_recipe.py",
                                       ROOT/"scripts/mjlab_gpu_ppo_env.py", ROOT/"scripts/mjlab_tron1_gpu.py",
                                       ROOT/"scripts/tron1_torch_task.py"]
                                      if p.is_file()},
                    "evaluation_protocol": "DR disabled; both surfaces fixed to each of17mu; XY failure only; no airborne failure"}
        manifest["height_regularization"] = {
            **json.loads(args.task_profile.read_text())["height_regularization"],
            "absolute_height_reward": False, "height_termination": False}
        manifest["height_adapter_sha256"] = source_hash(ROOT/"scripts/height_smoothness.py")
        manifest["task_profile"] = json.loads(args.task_profile.read_text())
        manifest["contract"] = json.loads(args.contract.read_text())
        if args.backend == "physx":
            manifest["inputs"]["asset"] = {"path": str(args.asset_path.resolve()),
                                          "sha256": source_hash(args.asset_path)}
            manifest["tron1_training_source"] = {str(p.relative_to(args.tron1_source)): source_hash(p)
                for p in sorted((args.tron1_source/"training").glob("*.py"))}
        import inspect
        import rsl_rl.runners.on_policy_runner as runner_source
        import rsl_rl.algorithms.ppo as ppo_source
        manifest["algorithm_source_sha256"] = {
            key: source_hash(inspect.getfile(module))
            for key, module in (("runner", runner_source), ("ppo", ppo_source))}
        # Seed AFTER all environment construction and smoke draws on both engines.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        runner = OnPolicyRunner(wrapped, copy.deepcopy(config), log_dir=str(args.output_dir), device=args.device)
        manifest["initial_policy_state_sha256"] = policy_state_hash(runner.alg.policy)
        write_json(args.output_dir/"manifest.json", manifest)
        report.update(status="training", smoke_steps_passed=args.smoke_steps,
                      initial_policy_state_sha256=manifest["initial_policy_state_sha256"])
        write_json(args.output_dir/"run_status.json", report)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.save(str(args.output_dir/"model_final.pt"))
        actor = torch.nn.Sequential(copy.deepcopy(runner.alg.policy.actor_obs_normalizer).cpu(),
                                    copy.deepcopy(runner.alg.policy.actor).cpu()).eval()
        exported = torch.jit.script(actor)
        exported.save(str(args.output_dir/"policy.pt"))
        probe = torch.linspace(-2., 2., 51).repeat(2, 1)
        with torch.inference_mode():
            expected = runner.alg.policy.act_inference({"policy": probe.to(args.device)})
            if not torch.allclose(exported(probe), expected.cpu(), rtol=1.e-5, atol=1.e-6):
                raise AssertionError("Exported actor normalization/inference mismatch")
        write_json(args.output_dir/"policy_contract.json", json.loads(args.contract.read_text()))
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        if runner.tot_timesteps != args.num_envs*24*args.iterations:
            raise AssertionError("Runner transition accounting mismatch")
        report.update(status="training_completed", elapsed_seconds=time.monotonic()-started,
                      training_environment_steps=runner.tot_timesteps,
                      checkpoint_sha256=source_hash(args.output_dir/"model_final.pt"),
                      exported_policy_sha256=source_hash(args.output_dir/"policy.pt"))
        write_json(args.output_dir/"environment_final.json",
                   env.metadata() if args.backend == "mjlab" else recipe_audit(env))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        write_json(args.output_dir/"run_status.json", report)
        if accumulator is not None:
            accumulator.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close(skip_cleanup=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

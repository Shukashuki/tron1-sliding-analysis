"""Asymmetric SAC on the same TRON1 task and BeyondMimic DR as PPO.

Default invocation is NOT a smoke: choose --smoke-only for a short integration
check. The deterministic raw-observation actor is exported separately from its
optimizer checkpoint. Simulator state is not checkpointed for exact continuation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from controller_references import (DEFAULT_CATALOG, PRIMARY_REFERENCE_ID,
                                   SELECTED_REFERENCES, verify_motion_identity)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "motion-file", "asset-path", "task-profile", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-id", choices=tuple(SELECTED_REFERENCES), default=PRIMARY_REFERENCE_ID)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--policy-steps", type=int, default=24000)
    parser.add_argument("--warmup-transitions", type=int, default=10000)
    parser.add_argument("--replay-capacity", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--updates-per-policy-step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=2400)
    parser.add_argument("--ground-friction", type=float, default=1.)
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-episode-steps", type=int, default=4,
                        help="Smoke-only short timeout to exercise pre-reset bootstrap capture")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (min(args.num_envs, args.policy_steps, args.replay_capacity, args.batch_size,
            args.updates_per_policy_step, args.log_interval, args.save_interval) < 1
            or args.warmup_transitions < 0 or not 0. < args.gamma < 1. or args.ground_friction <= 0.):
        parser.error("Invalid SAC collection settings")
    if args.smoke_only and args.policy_steps > 128:
        parser.error("A smoke must be explicitly short (<=128 policy steps)")
    if args.smoke_episode_steps < 2:
        parser.error("Smoke timeout must allow at least two physical control intervals")
    reference_spec = verify_motion_identity(args.motion_file, args.reference_id)
    reference_catalog_sha256 = sha(DEFAULT_CATALOG)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Preserving previous SAC output: " + str(output))
    output.mkdir(parents=True)
    app = env = None
    status = {"kind": "asymmetric_tron1_sac", "status": "initializing",
              "smoke_only": args.smoke_only, "trained_policy_claimed": False}
    start = time.monotonic()
    try:
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from sac_tracking import SACConfig, SACAgent, VectorReplay
        from terminal_observation import make_terminal_observation_env_class
        from height_smoothness import install_height_smoothness
        from train_beyondmimic_style import install_beyondmimic_events, audit_material_draw
        sys.path.insert(0, str(args.source_root.resolve()))
        from training.tron1_tracking import make_env_cfg, observation_action_contract
        from training.tron1_drift import load_task_profile, apply_task_profile_to_contract
        from training.tron1_runtime_device import validate_simulation_device
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        profile = load_task_profile(args.task_profile)
        if "height_regularization" not in profile or "wheel_ground_load_reserve" in profile:
            raise ValueError("SAC comparison requires the new explicit height-smooth drift profile")
        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device,
                           task_profile=profile)
        cfg.seed = args.seed
        recipe = install_beyondmimic_events(cfg, ground_friction=args.ground_friction)
        height_contract = install_height_smoothness(cfg, profile["height_regularization"])
        if args.smoke_only:
            cfg.episode_length_s = args.smoke_episode_steps * cfg.sim.dt * cfg.decimation
        env = make_terminal_observation_env_class()(cfg=cfg)
        device_audit = validate_simulation_device(env, requested_device=args.device)
        material_audit = audit_material_draw(env, recipe)
        contract = apply_task_profile_to_contract(observation_action_contract(), profile)
        contract.update(training_domain_randomization=recipe, height_regularization=height_contract,
                        policy_algorithm="SAC", actor_format="torchscript_raw_observation_deterministic_tanh")
        contract["domain_randomization"] = "BeyondMimic-style training events enabled; disabled for fixed-friction validation"
        learner_cfg = SACConfig(actor_dim=contract["actor_dim"], critic_dim=contract["critic_dim"],
                                action_dim=contract["action_dim"], initial_alpha=.2*env.step_dt)
        agent = SACAgent(learner_cfg, env.device)
        replay = VectorReplay(args.replay_capacity, learner_cfg.actor_dim, learner_cfg.critic_dim,
                              learner_cfg.action_dim, env.device)
        obs, _ = env.reset()
        if obs["policy"].shape != (args.num_envs, learner_cfg.actor_dim) or obs["critic"].shape != (args.num_envs, learner_cfg.critic_dim):
            raise ValueError("SAC environment observation dimensions differ from PPO contract")
        source = args.source_root.resolve()
        source_files = ["training/tron1_tracking.py", "training/tron1_drift.py",
                        "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py"]
        adapter_files = ["sac_tracking.py", "terminal_observation.py", "height_smoothness.py",
                         "train_beyondmimic_style.py", Path(__file__).name]
        manifest = {"algorithm": "SAC", "smoke_only": args.smoke_only,
                    "reference_id": args.reference_id, "reference_spec": reference_spec,
                    "reference_catalog_sha256": reference_catalog_sha256,
                    "pooled_success_statistics": False,
                    "source_training_modules_sha256": {name: sha(source/name) for name in source_files},
                    "comparison_modules_sha256": {name: sha(Path(__file__).with_name(name)) for name in adapter_files},
                    "height_adapter_sha256": sha(Path(__file__).with_name("height_smoothness.py")),
                    "height_regularization": height_contract, "recipe": recipe,
                    "task_profile": profile, "task_profile_sha256": sha(args.task_profile),
                    "motion_sha256": sha(args.motion_file), "asset_sha256": sha(args.asset_path),
                    "motion_file": str(args.motion_file.resolve()), "asset_file": str(args.asset_path.resolve()),
                    "contract": contract, "policy_contract": contract,
                    "sac_config": asdict(learner_cfg), "seed": args.seed, "num_envs": args.num_envs,
                    "policy_steps_requested": args.policy_steps,
                    "environment_transitions_requested": args.num_envs*args.policy_steps,
                    "batch_size": args.batch_size, "gamma": args.gamma,
                    "warmup_transitions": args.warmup_transitions,
                    "updates_per_policy_step": args.updates_per_policy_step,
                    "replay_capacity": replay.capacity, "replay_allocated_bytes": replay.allocated_bytes,
                    "observation_normalization": "fixed identity affine (embedded in networks/export); PPO uses online RSL normalization",
                    "reward_scaling": "unchanged Isaac RewardManager dt-scaled rewards; no SAC-only multiplier",
                    "initial_entropy_temperature_units": "0.2 * control_dt for dt-scaled rewards; automatic entropy tuning thereafter",
                    "smoke_episode_timeout_steps": args.smoke_episode_steps if args.smoke_only else None,
                    "bootstrap": "terminated masks target; truncation uses captured pre-reset physical successor and advanced/clamped reference phase, not reset state",
                    "initial_material_audit": material_audit, "device_audit": device_audit,
                    "resume_limitations": "learner/replay checkpoint available; no complete simulator/episode-state continuation",
                    "evaluation": "freeze deterministic actor, disable random events, fixed validation material grid"}
        write_json(output/"manifest.json", manifest)
        write_json(output/"policy_contract.json", contract)
        write_json(output/"runner_config.json", {"algorithm": "SAC", "sac": asdict(learner_cfg)})
        status.update(status="smoke_running" if args.smoke_only else "training", policy_steps_completed=0)
        write_json(output/"run_status.json", status)
        transition_count = terminated_count = truncated_count = final_observation_count = 0
        history = []
        last_metrics = None
        for step in range(args.policy_steps):
            before_actor = obs["policy"].clone()
            before_critic = obs["critic"].clone()
            if transition_count < args.warmup_transitions:
                action = 2.*torch.rand((args.num_envs, learner_cfg.action_dim), device=env.device)-1.
            else:
                action = agent.act(before_actor)
            obs, reward, terminated, truncated, extras = env.step(action)
            bootstrap = extras["bootstrap_observation"]
            replay.add(actor_obs=before_actor, critic_obs=before_critic, actions=action,
                       rewards=reward, bootstrap_actor_obs=bootstrap["policy"],
                       bootstrap_critic_obs=bootstrap["critic"], terminated=terminated.bool(),
                       truncated=truncated.bool(), discounts=torch.full_like(reward, args.gamma))
            transition_count += args.num_envs
            terminated_count += int(terminated.sum())
            truncated_count += int(truncated.sum())
            final_observation_count += len(extras["final_observation_env_ids"])
            if replay.size >= args.batch_size and transition_count >= args.warmup_transitions:
                for _ in range(args.updates_per_policy_step):
                    last_metrics = agent.update(replay.sample(args.batch_size))
            if step == 0 or (step+1) % args.log_interval == 0 or step+1 == args.policy_steps:
                row = {"policy_steps": step+1, "environment_transitions": transition_count,
                       "updates": agent.updates, "reward_mean": float(reward.mean()),
                       "terminated_total": terminated_count, "truncated_total": truncated_count,
                       "captured_final_observations": final_observation_count,
                       "elapsed_s": time.monotonic()-start, "losses": last_metrics}
                history.append(row)
                status.update(policy_steps_completed=step+1, environment_transitions=transition_count,
                              learner_updates=agent.updates, elapsed_s=row["elapsed_s"])
                write_json(output/"run_status.json", status)
                write_json(output/"training_log.json", history)
                print(json.dumps(row), flush=True)
            if not args.smoke_only and (step+1) % args.save_interval == 0:
                agent.save(output/f"model_step_{step+1:06d}.pt")
        if agent.updates == 0:
            raise RuntimeError("No SAC update occurred; this is not a learner integration test")
        if args.smoke_only and (truncated_count == 0 or final_observation_count < truncated_count):
            raise RuntimeError("Smoke did not exercise captured time-limit bootstrap transitions")
        checkpoint = output/("model_smoke.pt" if args.smoke_only else "model_final.pt")
        actor_path = output/("actor_smoke.pt" if args.smoke_only else "actor_normalized.pt")
        agent.save(checkpoint, replay=replay if args.save_replay else None)
        agent.export_actor(actor_path)
        exported = torch.jit.load(str(actor_path), map_location=env.device)
        with torch.no_grad():
            export_error = float((agent.act(obs["policy"], deterministic=True)-exported(obs["policy"])).abs().max())
        if export_error > 1e-6:
            raise AssertionError("Exported SAC actor differs from deterministic collection actor")
        status.update(status="smoke_passed" if args.smoke_only else "training_completed",
                      trained_policy_claimed=not args.smoke_only,
                      policy_steps_completed=args.policy_steps, environment_transitions=transition_count,
                      learner_updates=agent.updates, terminated_total=terminated_count,
                      truncated_total=truncated_count, captured_final_observations=final_observation_count,
                      export_max_abs_error=export_error, checkpoint_sha256=sha(checkpoint),
                      actor_sha256=sha(actor_path), elapsed_s=time.monotonic()-start)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_s=time.monotonic()-start)
        raise
    finally:
        write_json(output/"run_status.json", status)
        if env is not None:
            env.close()
        if app is not None:
            app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()

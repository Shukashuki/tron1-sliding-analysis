"""Train a TRON1 motion with the pinned BeyondMimic startup/interval DR recipe.

Requires the TRON1 source checkout, Isaac Lab 2.3, RSL-RL 3, robot USD, and
an exported 50 Hz TRON1 motion. This is a new policy, not the archived baseline.
No evaluation friction sweep is performed during training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BM_COMMIT = "cd65172032893724b445448818c34165846d847d"
VELOCITY_RANGE = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_revision(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def load_source_ppo_config(source_root: Path, seed: int, save_interval: int) -> dict:
    path = source_root / "scripts/train_tracking.py"
    spec = importlib.util.spec_from_file_location("tron1_original_training_entrypoint", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source PPO configuration from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ppo_config(seed, save_interval)


def install_beyondmimic_events(cfg, *, ground_friction: float = 1.0) -> dict:
    """Map the official G1 event recipe to TRON1's base_Link and all 8 joints.

    The exact source ranges are copied from the pinned upstream EventCfg. The
    robot/ground contact law is still TRON1/PhysX-specific, not G1-equivalent.
    """
    from whole_body_tracking.tasks.tracking import mdp
    from isaaclab.managers import EventTermCfg
    from isaaclab.managers import SceneEntityCfg
    from training.tron1_tracking import LEG_JOINT_NAMES

    material = cfg.events.physics_material
    if material.mode != "startup" or material.func is not mdp.randomize_rigid_body_material:
        raise ValueError("Unexpected TRON1 material event; refusing to override it")
    material.params.update({
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.3, 1.6),
        "dynamic_friction_range": (0.3, 1.2),
        "restitution_range": (0.0, 0.5),
        "num_buckets": 64,
    })
    cfg.scene.terrain.physics_material.static_friction = ground_friction
    cfg.scene.terrain.physics_material.dynamic_friction = ground_friction
    cfg.scene.terrain.physics_material.restitution = 0.0
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    cfg.events.add_joint_default_pos = EventTermCfg(
        func=mdp.randomize_joint_default_pos, mode="startup",
        # Upstream helper also writes the joint_pos action offset. TRON1 has
        # only six position-action joints; its two wheels are velocity actions.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
                "pos_distribution_params": (-0.01, 0.01), "operation": "add"},
    )
    cfg.events.base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}},
    )
    cfg.events.push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )
    return {
        "upstream_commit": BM_COMMIT,
        "friction_static": [0.3, 1.6], "friction_dynamic": [0.3, 1.2],
        "restitution": [0.0, 0.5], "material_buckets": 64,
        "joint_default_position_offset_rad": [-0.01, 0.01],
        "base_com_offset_m": {"x": [-0.025, 0.025], "y": [-0.05, 0.05], "z": [-0.05, 0.05]},
        "push_interval_s": [1.0, 3.0], "push_velocity_range": VELOCITY_RANGE,
        "ground_friction": ground_friction,
        "robot_mapping": "G1 torso_link -> TRON1 base_Link; joint offset applies to six position-controlled legs, not two velocity-controlled wheels",
    }


def audit_material_draw(env, expected: dict) -> dict:
    import numpy as np

    material = env.scene["robot"].root_physx_view.get_material_properties().cpu().numpy()
    if material.ndim < 3 or material.shape[-1] != 3:
        raise ValueError(f"Unexpected PhysX material shape: {material.shape}")
    ranges = (expected["friction_static"], expected["friction_dynamic"], expected["restitution"])
    for index, bounds in enumerate(ranges):
        values = material[..., index]
        if not np.isfinite(values).all() or values.min() < bounds[0] - 1e-5 or values.max() > bounds[1] + 1e-5:
            raise ValueError(f"Runtime material channel {index} outside BeyondMimic range")
    if env.num_envs > 1 and np.ptp(material[..., :2], axis=0).max() < 1e-3:
        raise ValueError("No friction diversity observed across training environments")
    return {"shape": list(material.shape),
            "static_range": [float(material[..., 0].min()), float(material[..., 0].max())],
            "dynamic_range": [float(material[..., 1].min()), float(material[..., 1].max())],
            "restitution_range": [float(material[..., 2].min()), float(material[..., 2].max())]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--asset-path", type=Path, required=True)
    parser.add_argument("--task-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-steps", type=int, default=8)
    parser.add_argument("--ground-friction", type=float, default=1.0)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (min(args.num_envs, args.iterations, args.save_interval) < 1
            or args.smoke_steps < 0 or args.ground_friction <= 0):
        parser.error("Invalid environment/iteration/ground material settings")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Preserving nonempty training output")
    output.mkdir(parents=True, exist_ok=True)
    app = env = accumulator = None
    status = {"status": "initializing", "training_recipe": "BeyondMimic event mapping to TRON1"}
    start = time.monotonic()
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        source_root = args.source_root.resolve()
        sys.path.insert(0, str(source_root))
        from training.tron1_tracking import make_env_cfg, observation_action_contract
        from training.tron1_drift import (load_task_profile, apply_task_profile_to_contract,
                                          install_drift_reward_sampling)
        profile = load_task_profile(args.task_profile)
        if "wheel_ground_load_reserve" in profile:
            raise ValueError("Cached ground-load-reserve reward requires its historical fixed event set; use a noncached drift profile for BeyondMimic DR")
        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs,
                           args.device, task_profile=profile)
        cfg.seed = args.seed
        recipe = install_beyondmimic_events(cfg, ground_friction=args.ground_friction)
        env = ManagerBasedRLEnv(cfg=cfg)
        accumulator = install_drift_reward_sampling(env, profile)
        material_audit = audit_material_draw(env, recipe)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        contract = apply_task_profile_to_contract(observation_action_contract(), profile)
        contract["training_domain_randomization"] = recipe
        obs, _ = wrapped.reset()
        if (tuple(obs["policy"].shape) != (args.num_envs, contract["actor_dim"])
                or env.action_manager.total_action_dim != contract["action_dim"]):
            raise ValueError("Observation/action contract mismatch")
        for _ in range(args.smoke_steps):
            obs, rewards, _, _ = wrapped.step(torch.zeros(args.num_envs, contract["action_dim"], device=args.device))
            if not torch.isfinite(rewards).all() or not all(torch.isfinite(x).all() for x in obs.values()):
                raise FloatingPointError("Non-finite training smoke sample")
        wrapped.reset()
        ppo = load_source_ppo_config(source_root, args.seed, args.save_interval)
        (output / "runner_config.json").write_text(json.dumps(ppo, indent=2) + "\n")
        revision = source_revision(source_root)
        source_files = ["scripts/train_tracking.py", "training/tron1_tracking.py",
                        "training/tron1_drift.py", "training/tron1_domain_randomization.py",
                        "training/tron1_contact_slip.py"]
        manifest = {"status": "training_configured", "source_commit": revision,
                    "source_checkout_kind": "git" if revision else "snapshot_without_git_metadata",
                    "source_training_modules_sha256": {name: sha(source_root / name) for name in source_files},
                    "trainer_sha256": sha(Path(__file__)), "motion_sha256": sha(args.motion_file),
                    "asset_sha256": sha(args.asset_path), "task_profile_sha256": sha(args.task_profile),
                    "motion_file": str(args.motion_file.resolve()),
                    "asset_file": str(args.asset_path.resolve()),
                    "task_profile": profile, "terrain": None, "contract": contract,
                    "seed": args.seed, "num_envs": args.num_envs, "iterations": args.iterations,
                    "recipe": recipe, "material_draw_audit": material_audit,
                    "policy_contract": contract,
                    "evaluation": "Freeze final actor; material randomization disabled; sweep fixed coefficients."}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        status["status"] = "training"
        (output / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        runner = OnPolicyRunner(wrapped, copy.deepcopy(ppo), log_dir=str(output), device=args.device)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.save(str(output / "model_final.pt"))
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        status.update(status="training_completed", elapsed_seconds=time.monotonic() - start,
                      iterations_requested=args.iterations,
                      learning_iteration_index=runner.current_learning_iteration,
                      final_material_draw_audit=audit_material_draw(env, recipe))
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        (output / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        if accumulator is not None:
            accumulator.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()

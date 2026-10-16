"""Short runtime check for the new task. No policy training or success claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "motion-file", "asset-path", "task-profile", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=2)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserving previous smoke output")
    args.output_dir.mkdir(parents=True)
    status = {"kind": "height_smooth_task_runtime_smoke", "status": "initializing",
              "trained_policy": False, "closed_loop_tracking_success_claimed": False}
    app = env = None
    start = time.monotonic()
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        sys.path.insert(0, str(args.source_root.resolve()))
        from training.tron1_tracking import make_env_cfg
        from training.tron1_drift import load_task_profile
        from training.tron1_runtime_device import validate_simulation_device
        from train_beyondmimic_style import install_beyondmimic_events, audit_material_draw
        from height_smoothness import install_height_smoothness
        profile = load_task_profile(args.task_profile)
        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device,
                           task_profile=profile)
        recipe = install_beyondmimic_events(cfg)
        contract = install_height_smoothness(cfg, profile["height_regularization"])
        env = ManagerBasedRLEnv(cfg=cfg)
        audit = validate_simulation_device(env, requested_device=args.device)
        active = env.termination_manager.active_terms
        if "anchor_pos" in active or "ee_body_pos" in active:
            raise AssertionError("Height stopping rule remains active")
        term = env.reward_manager.get_term_cfg("height_vertical_acceleration").func
        material = audit_material_draw(env, recipe)
        obs, _ = env.reset()
        rewards, samples = [], []
        for index in range(args.steps):
            if index == args.steps // 2:
                env.reset()  # Exercise the managed term's history reset.
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(args.num_envs, 8, device=env.device))
            if not torch.isfinite(reward).all() or not all(torch.isfinite(x).all() for x in obs.values()):
                raise FloatingPointError("Nonfinite smoke observations/rewards")
            if not torch.isfinite(term.cost).all() or not ((term.cost >= 0.) & (term.cost <= 1.)).all():
                raise AssertionError("Invalid bounded vertical smoothness cost")
            rewards.append(reward.detach().cpu().tolist())
            samples.append(term.cost.detach().cpu().tolist())
        status.update(status="smoke_passed", steps=args.steps, num_envs=args.num_envs,
                      elapsed_s=time.monotonic()-start, device_audit=audit,
                      active_termination_terms=active, height_contract=contract,
                      material_draw_audit=material, reward_samples=rewards,
                      smoothness_cost_samples=samples,
                      observation_shapes={key: list(value.shape) for key, value in obs.items()})
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_s=time.monotonic()-start)
        raise
    finally:
        (args.output_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        if env is not None:
            env.close()
        if app is not None:
            app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()

"""Run a small real-CUDA mjlab/TRON1 task verification, never a training job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mjlab_tron1_env import MjlabWarpBatch
from mujoco_ppo_env import MujocoPPOEnv


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = MujocoPPOEnv(args.model, args.motion, args.contract, args.task_profile,
                       num_envs=args.num_envs, training=args.dr_mode != "none",
                       dr_mode=args.dr_mode, seed=args.seed,
                       batch_factory=lambda *values: MjlabWarpBatch(*values, device=args.device))
    try:
        initial, initial_obs = env.batch.measure(), env.observations()
        rewards, clocks, actions = [], [], []
        for index in range(args.steps):
            action = np.zeros((args.num_envs, 8), dtype=np.float32)
            # Nonzero wheel efforts exercise both original torque actuators.
            action[:, 6:] = .025*(index % 3-1)
            obs, reward, terminated, truncated, extras = env.step(action)
            current = env.batch.measure()
            if (obs["policy"].shape != (args.num_envs, 51)
                    or obs["critic"].shape != (args.num_envs, 120)
                    or not np.isfinite(reward).all()):
                raise AssertionError("Observation/reward contract failed")
            rewards.append(reward.copy())
            clocks.append(current["time_s"].copy())
            actions.append(action.copy())
        final = env.batch.measure()
        # Contact forces are taken from the actual GPU solver, not a CPU rerun.
        physical = env.batch.step(np.zeros((args.num_envs, 8), dtype=np.float32))
        forces = physical["body_net_forces_w"]
        if forces.shape != (args.num_envs, 4, 9, 3) or not np.isfinite(forces).all():
            raise AssertionError("Physical contact-force history failed")
        before_subset_reset = physical["time_s"].copy()
        env.batch.reset(np.array([0]), initial["qpos"][:1], initial["qvel"][:1])
        after_subset_reset = env.batch.measure()
        if not np.isclose(after_subset_reset["time_s"][0], 0., atol=1.e-7):
            raise AssertionError("Selected-world reset did not reset its physical clock")
        np.testing.assert_allclose(after_subset_reset["time_s"][1:], before_subset_reset[1:],
                                   atol=1.e-7, rtol=0.)
        np.savez_compressed(output / "smoke_arrays.npz", initial_qpos=initial["qpos"],
                            initial_qvel=initial["qvel"], initial_actor=initial_obs["policy"],
                            initial_critic=initial_obs["critic"], final_qpos=final["qpos"],
                            final_qvel=final["qvel"], rewards=np.asarray(rewards),
                            actions=np.asarray(actions), physical_times=np.asarray(clocks),
                            contact_force_history_w=forces,
                            applied_torque=physical["applied_torque"],
                            before_subset_reset_time=before_subset_reset,
                            after_subset_reset_time=after_subset_reset["time_s"])
        report = {"status": "passed_smoke_not_full_training", "steps": args.steps,
                  "num_envs": args.num_envs, "elapsed_s": time.monotonic()-started,
                  "actual_contact_force_norm_N": float(np.linalg.norm(forces)),
                  "nonzero_contact_samples": int(np.count_nonzero(np.linalg.norm(forces, axis=-1) > 0.)),
                  "subset_reset_other_world_clock_preserved": True,
                  "task_metadata": env.metadata(), "physics_metadata": env.batch.metadata,
                  "inputs": {name: {"path": str(Path(value).resolve()),
                                    "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest()}
                             for name, value in (("model", args.model), ("motion", args.motion),
                                                 ("contract", args.contract), ("task_profile", args.task_profile))}}
        (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({key: report[key] for key in
                          ("status", "steps", "num_envs", "elapsed_s", "actual_contact_force_norm_N")}), flush=True)
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--task-profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dr-mode", choices=("none", "mjlab_v1p4"), default="none")
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 1:
        parser.error("Use at least two worlds and one step for reset-isolation checks")
    run(args)


if __name__ == "__main__":
    main()

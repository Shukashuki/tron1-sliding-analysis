# TRON1 PPO physics backend comparison

The new paired experiment uses the BeyondMimic tracking randomization implemented in **mjlab v1.4.0**, adapted to TRON1. PPO remains RSL-RL 3.0.1 with the existing network and optimization settings. Historical PhysX-trained results remain a separate cohort because their material randomization differs.

BeyondMimic's [official repository](https://github.com/HybridRobotics/whole_body_tracking#alternative-implementations) directs its MuJoCo-Warp implementation to mjlab. The pinned [tracking configuration](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/tasks/tracking/tracking_env_cfg.py) is the source of the new DR recipe, not a claim that PhysX and MuJoCo share identical contact laws.

## Fixed task and algorithm

The primary reference is traj_4. Both training arms retain eight motor actions, a 51D actor observation and a 120D critic observation. Six leg motors track `q0 + action`; two wheels receive scaled torque. Encoder bias changes measured leg position by `+b` and the position target by `-b`; the privileged critic receives unbiased positions.

The control rate is 50 Hz with four 5 ms physics steps. Rewards retain XY tracking, speed, yaw rate, sideslip, contact, stability and action regularization. Vertical acceleration is penalized for smoothness; absolute height and airborne duration are not failure criteria. Tilt safety and motion-end timeouts remain enabled.

Each full training arm uses 512 environments, 24 steps per rollout and 1,000 PPO iterations: **12,288,000 transitions**, seed 42. Network initialization is seeded after environment construction and smoke testing, with its state hash recorded. This does not imply identical random disturbance realizations across engines.

## Randomization

| Quantity | New paired recipe |
| --- | --- |
| Wheel sliding friction | Continuous U(0.3, 1.2), one startup draw per environment, shared by both wheels |
| PhysX material mapping | Static = dynamic = draw; ground μ = 1 with multiply combination |
| MuJoCo material mapping | Wheel geom priority = 1 so the wheel draw overrides ground friction |
| Encoder position bias | U(−0.01, 0.01) rad; actor biased, critic unbiased, opposite target correction |
| Base COM offset | x ±0.025 m, y/z ±0.05 m |
| Additive root link velocity pushes | Every 1–3 s; linear x/y ±0.5, z ±0.2 m/s; angular roll/pitch ±0.52, yaw ±0.78 rad/s |
| Restitution | No coefficient randomization; native contact-law differences remain |

Wheel-only material selection follows mjlab's foot selection. Nonwheel coefficients remain fixed: MuJoCo's original 0.6 and ground 1.0 combine to an effective plane-contact coefficient of 1.0; PhysX uses nonwheel 1.0 with unit-ground multiplication to match this effective coefficient. MuJoCo's single tangential friction coefficient is not a one-to-one implementation of independent PhysX static/dynamic coefficients. The adapter also corrects v1.4.0's function-based interval-event timer reset behavior by resampling push timers at every episode reset.

## Backend implementation

The repository pins mjlab as [a submodule](../upstream/mjlab). Its `Simulation` supplies persistent, float32 MuJoCo-Warp CUDA physics; a custom TRON1 bridge recomputes the original DC motor limits each physical step and extracts actual GPU contact forces. The verified NumPy task computes observations and rewards on the host. Host transfers limit throughput, so this first implementation is not a GPU performance benchmark.

The adapter intentionally does not import mjlab's RSL 5 runner or use its G1 action, noise, reward or reset defaults. The existing RSL 3 algorithm and TRON1 task are retained to reduce confounding changes. MuJoCo-Warp training and native CPU MuJoCo validation remain distinct implementations and precisions.

## Evaluation

Each newly trained policy will be evaluated from frame zero in both PhysX and native MuJoCo, with perturbations disabled. The [locked protocol](../config/ppo_mjlab_backend_comparison_v1.json) retains all 17 friction values from 0.01 to 2.0. Report core XY RMSE and peak error, speed retention, actual course change and body/wheel/velocity headings. XY screening thresholds remain 0.15 m RMSE or 0.30 m peak error. Intended sideslip is a behavior metric, not a mismatch penalty. Flight and contact discontinuities are diagnostics only.

Closed-loop differences include policy feedback, task adaptation and numerical/contact implementation. Matched-state/action probes are needed to isolate transition differences; closed-loop tracking error alone does not identify which simulator is more accurate.

## Reproduction

Initialize `upstream/mjlab` at the pinned revision and use an environment containing MuJoCo 3.8.1, MuJoCo-Warp 3.8.1, Warp 1.12.1, Torch 2.7.0 and RSL-RL 3.0.1. Do not install mjlab's default dependency set into the existing Isaac environment: this tag requests a newer RSL runner that this comparison deliberately does not use.

```bash
python scripts/preflight_mjlab_runtime.py --mjlab-source upstream/mjlab --output outputs/mjlab_preflight.json
PYTHONPATH=upstream/mjlab/src python scripts/smoke_mjlab_tron1.py \
  --model /path/to/mujoco_model.xml --motion /path/to/motion.npz \
  --contract config/nominal_actuator_contract.json \
  --task-profile config/car_drift_height_smooth_task.json \
  --dr-mode mjlab_v1p4 --output-dir outputs/mjlab_tron1_smoke
python scripts/train_ppo_backend.py --backend mjlab \
  --model /path/to/mujoco_model.xml --motion /path/to/motion.npz \
  --num-envs 512 --iterations 1000 --output-dir outputs/ppo_mjlab_full
```

The PhysX arm uses the same trainer with `--backend physx`, explicit `--asset-path`, `--tron1-source` and `--beyondmimic-source`. Every run requires a new output directory. Source snapshots are generated by `scripts/package_ppo_backend.py`; model assets, checkpoints and credentials are excluded.

## Verification status

The [asset-free runtime preflight](ppo_mjlab_backend_preflight_20261021_v1/runtime_preflight.json) passed on the RTX host without package or driver changes: mjlab manager imports, a compiled Warp CUDA kernel and a falling sphere with ground contact. Driver CUDA 12.2 requires the non-captured graph fallback, which ran successfully.

The [actual TRON1 smoke](ppo_mjlab_backend_preflight_20261021_v1/tron1_physics_smoke.json) passed with four environments, eight control steps, nonzero GPU contact forces and isolated per-world reset. Both [MuJoCo-Warp](ppo_mjlab_backend_preflight_20261021_v1/mjlab_training_smoke_status.json) and [PhysX](ppo_mjlab_backend_preflight_20261021_v1/physx_training_smoke_status.json) then completed two PPO iterations with 32 environments: **1,536 transitions each**, including normalized TorchScript export. Their initial policy state hashes are identical. The PhysX readback confirms all 15 reward terms, verified materials and only tilt/motion-end/time-limit termination. These checks establish an executable learning pipeline, not successful high-speed tracking.

Exact executed source snapshots and their file hashes accompany the smoke evidence. The new adapter tests pass independently. A full working-tree test run found 27 historical source-hash failures against pre-existing modified MPC runner files; those files and frozen historical expectations were not changed for this PPO work.

## Full training launch

The [sequential dispatcher](../scripts/run_ppo_backend_pair.py) launched the full MuJoCo-Warp arm on physical GPU 0, with the PhysX arm queued afterward. Each arm requests 12,288,000 transitions. The [launch status](ppo_mjlab_backend_preflight_20261021_v1/full_pair_launch_status.json) and [initial full-run manifest](ppo_mjlab_backend_preflight_20261021_v1/full_mjlab_launch_manifest.json) are snapshots, not live completion reports. Full training and the subsequent 17-point cross-engine evaluations are not yet completed.

The dispatcher verifies an idle GPU before each child, immutable source hashes, exact transition counts, exported checkpoint hashes and matching initial network states. It stops safely on a busy GPU or failed child; it does not tune parameters, retry automatically or terminate other users' processes. The initial measured collection rate was approximately 1,155 transitions/s, implying about three hours for the MuJoCo-Warp arm if that rate persists.

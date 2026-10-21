# GPU-resident TRON1 task: validation and timing

The task hot path now keeps bulk state, actions, contact forces, observations, rewards, resets and pushes on the GPU. An RTX 4090 D short test passed numerical parity. The measured environment-only improvement was **1.080× throughput**, not a demonstrated PPO training speedup. The existing full training on GPU 0 remains on its immutable host-task snapshot; GPU 1 was used only for the approved diagnostics and released afterward.

## What changed and what did not

[GPU physics bridge](../scripts/mjlab_tron1_gpu.py), [Torch task mathematics](../scripts/tron1_torch_task.py) and [GPU environment/RSL wrapper](../scripts/mjlab_gpu_ppo_env.py) replace bulk NumPy round trips with device tensors. Fixed-capacity Warp kernels reduce actual contact forces; CUDA events order the Torch and Warp streams. Each control interval still reads one fatal-status scalar and one three-byte reset/push/fault payload. This is not zero CPU synchronization.

The [paired protocol](2026-10-21-ppo-mjlab-backend-protocol.md) is unchanged: 51D actor, 120D critic, eight DC motor actions, 15 reward terms, matched mjlab randomization and unmodified RSL-RL 3.0.1 PPO. Physics remains four 5 ms **step + full-forward pairs** per 20 ms control interval, including the existing contact/warm-start semantics. There is no new height, XY or airborne termination. Subsequent phase/push draws use a private Torch RNG with the same distributions, not the NumPy draw sequence.

## Measured results

The [v9 report](mjlab_gpu_task_20261021_v1/benchmark_report.json) compares the host and GPU tasks on the **same pinned MuJoCo-Warp CUDA physics**, not CPU MuJoCo versus GPU physics.

At fixed μ = 0.7, four environments and eight identical-action steps had identical qpos/qvel. Maximum absolute observation error was 1.19e−7, reward error 4.47e−8 and raw reward-term error 7.71e−8. There were no episode resets in this parity trace; it does not establish long randomized rollout equivalence.

Timing used 512 environments, 24 control steps/window, three measured windows after warm-up, and training randomization enabled. Construction, JIT, parity copies and PPO updates were excluded.

| Task | Median seconds / window | Transitions / second |
| --- | ---: | ---: |
| Host NumPy task | 10.511 | 1,169 |
| GPU-resident task | 9.732 | 1,263 |

This is about 8.0% higher throughput or 7.4% less elapsed time. Three fixed-order windows, different reset RNG histories and a concurrently running job on another GPU limit the interpretation; there is no confidence interval or end-to-end training claim.

## Remaining cost

A separate [v10 instrumented diagnostic](mjlab_gpu_task_20261021_v1/profile_report.json), with 512 environments, μ = 0.7, no DR and eight measured steps, recorded mean inclusive CUDA-event intervals of 43.51 ms per physics `step`, 43.39 ms per full `forward`, 2.41 ms per reward evaluation and 1.57 ms per observation build. Stage intervals are non-additive and include host-dispatch gaps; they are not GPU-kernel busy-time percentages. This shorter, different workload must not be compared directly with the timing table.

The physical call path remains expensive in this implementation. Driver 535.230.02 exposes CUDA 12.2, below the pinned mjlab conditional-graph requirement of 12.4; `cuda_graphs=false` was recorded in both branches. Neither the driver nor the four post-step forwards were changed. These measurements do not show that MuJoCo is inherently slow or that removing forwards would preserve the experiment.

## Reproduction and evidence

Use mjlab v1.4.0 at `4360096b49c1bcc478668c11ce7d458db747e49d`, MuJoCo/MuJoCo-Warp 3.8.1, Warp 1.12.1, Torch 2.7.0+cu128 and RSL-RL 3.0.1. Model/motion hashes are in the reports. On the measured host, logical `cuda:0` was physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`.

```bash
export PYTHONPATH=upstream/mjlab/src
export OMP_NUM_THREADS=4
python scripts/benchmark_mjlab_gpu_task.py \
  --model /path/to/mujoco_model.xml --motion /path/to/motion.npz \
  --device cuda:0 --num-envs 512 --steps 24 --repeats 3 \
  --output-dir outputs/gpu_task_benchmark_new
python scripts/profile_mjlab_gpu_task.py \
  --model /path/to/mujoco_model.xml --motion /path/to/motion.npz \
  --device cuda:0 --num-envs 512 --steps 8 --warmup-steps 2 \
  --output-dir outputs/gpu_task_profile_new
```

For a separately authorized new training run, use `train_ppo_backend.py --backend mjlab --mjlab-task-implementation gpu` with the existing model/motion and training arguments. No full GPU-task PPO training was launched here.

The [evidence directory](mjlab_gpu_task_20261021_v1/) preserves exact executed v9/v10 source archives, manifests, reports and parity arrays; it excludes robot assets, weights and credentials. Current code additionally rejects nonfinite raw actions before saturation and labels Torch reward metadata correctly; these defensive changes were unit-tested after the measured snapshots and do not change the reported finite-action benchmark. On remote v10, 77 tests passed, including an actual RSL-RL 3 CPU update on a task double, not a GPU learning run. Current local checks passed 75 tests plus 10 Warp-bridge tests, with one RSL dependency skip. Full GPU-task learning throughput and tracking quality remain unmeasured.

# TRON1 MPC: Isaac physics-backend comparison preflight (2026-10-21)

This is an experiment protocol and environment audit, **not a new simulator
result**. The aim is to measure what changes when an MPC controller's
prediction physics and validation physics use Isaac Lab's PhysX or
Newton/MuJoCo-Warp backends. PPO is contextual; this protocol does not retrain
or rank PPO. Isaac Lab's Newton path may run without the Isaac Sim Kit runtime,
but it remains within the same Isaac Lab task framework.

## Current evidence and environment

- The existing full-turn Isaac CEM runner uses a **PhysX plant and native
  MuJoCo CPU predictor**. Its nominal `drift_drl_traj4`, μ=0.70, β-on historical
  episode has core XY RMSE 0.0674 m and speed retention 102.25%; its median
  solve time is 638 ms against a 20 ms control interval. It is not an Isaac
  PhysX predictor or a Newton/MJWarp result.
- The RTX host has Isaac Sim 5.1, Isaac Lab 2.3, and standalone
  `mujoco-warp==3.8.1`, but no installed NVIDIA Newton physics package or
  located TRON1 Newton/MJWarp result. Both previously used hosts are reachable
  with the explicitly selected user SSH identity. No remote jobs were started
  for this audit.
- The RTX and A800 hosts have NVIDIA driver 535.230.02 and 535.54.03,
  respectively; the local RTX 3060 Ti has 572.70 and 8 GB VRAM. The
  [Isaac Lab 3.0 installation requirements](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/isaaclab/installation.html)
  specify Linux driver **580.65.06 or newer** for its supported CUDA 13 stack.
  Upgrading a shared host's kernel driver would affect other jobs and was not
  attempted. The current hosts therefore cannot be claimed as validated
  Isaac Lab 3.0 multi-backend runtimes.

## Locked experiment

Use a fresh, pinned Isaac Lab 3.0 task that exposes
`physics=isaacsim_physx` and `physics=newton_mjwarp` via the same task config.
Reproduce the PhysX cell **in that same framework revision** before comparing
it with Newton; the old Isaac Lab 2.3 result is historical only. Keep the
`drift_drl_traj4` trajectory, μ=0.70, 5 ms physics/20 ms control period,
seed 42, actuator/action contract, reference phases, CEM budget, β-on reward,
wheel-XY tracking, and evaluation metrics fixed. Audit joint/body ordering,
inertia, wheel geometry, motor limits, material mapping, and contact sensing
before any controller conclusion. Equal numerical μ does not imply equal
wheel-ground slip across solvers.

| MPC predictor | Validation plant: PhysX | Validation plant: Newton/MJWarp |
| --- | --- | --- |
| PhysX | PP | PN |
| Newton/MJWarp | NP | NN |

Run short matched-action, one-step and multi-step prediction checks first;
then run the four full-turn closed-loop CEM cells. For cross-engine cells,
the candidate-rollout simulator must be isolated from the validation plant
and reset from the same measured state. Record both runtime backends rather
than inferring them from file names. Report core XY RMSE/peak, speed retention,
body/velocity/wheel headings, contact-conditioned lateral motion, fallback
count and per-decision latency. Do not label a 20 ms deadline as met unless
measured, or interpret wheel-heading offset alone as tire slip/failure.

The [matrix manifest](../config/mpc_isaac_backend_matrix_traj4_mu0p7.json)
and [read-only preflight](../scripts/preflight_mpc_backend_matrix.py) currently
accept the archived MuJoCo CPU→PhysX episode only as a historical baseline;
**PP, PN, NP and NN are all missing**. The preflight locks source-data and
controller semantics, requires matching Isaac Lab revision across new cells,
and validates report/artifact hashes. It does not independently certify a
self-reported backend audit. After nominal completion, analyze SA29 separately
and only then consider a fixed-controller friction sweep. CEM is the first
controller because it already has a full Isaac closed-loop runner. Existing
MPPI and SQP runs are MuJoCo-only pilots; extend their predictor/plant adapters
and repeat the same backend matrix before comparing those optimizer families.

References: [Isaac Lab physics backends](https://isaac-sim.github.io/IsaacLab/main/source/concepts/physics_backends.html),
[PhysX/Newton transfer and ordering checks](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html),
[Newton asset preparation](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/how-to/prepare_asset_for_newton.html).

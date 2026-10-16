# Nominal turning: tracking and control burden

The existing μ=0.70 episodes show different actuator use, not just different
XY error. PPO maintains accurate tracking with relatively large, rapidly varying
leg commands and small wheel torques. NMPC H30/B10 uses more effort than B5 in
Isaac while reducing XY error, but does not improve MuJoCo tracking. These are
descriptive observations from eight existing episodes, **not a controller ranking**.
No new physics, policy inference, training or friction cases were run.

## Core results

Torque RMS pools the six leg or two wheel components separately, using recorded
actual joint torque. Target-rate RMS pools the six leg command differences.
All values below use the original primary turn at seed 42 and μ=0.70.

| Controller / engine | XY RMSE (m) | Mean COM-speed retention | Actual torque RMS: legs / wheels (Nm) | Leg target-rate RMS (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| PPO / Isaac | 0.0565 | 99.41% | 53.41 / 1.52 | 14.84 |
| PPO / MuJoCo | 0.0498 | 99.74% | 54.91 / 1.28 | 15.83 |
| SAC / Isaac | 0.5748 | 100.23% | 57.75 / 9.15 | 24.82 |
| SAC / MuJoCo | 0.7543 | 119.61% | 51.97 / 5.07 | 20.38 |
| NMPC H30/B5 / Isaac | 0.2318 | 95.84% | 25.12 / 3.36 | 2.23 |
| NMPC H30/B5 / MuJoCo | 0.0916 | 101.16% | 30.86 / 5.67 | 3.21 |
| NMPC H30/B10 / Isaac | 0.0674 | 102.25% | 37.01 / 6.46 | 4.57 |
| NMPC H30/B10 / MuJoCo | 0.1700 | 95.68% | 33.76 / 5.50 | 3.91 |

SAC MuJoCo includes fallen states: maximum full-episode tilt is 138.07°.
Its speed and effort therefore do not describe successful upright turning.
Low effort alone is not better control, and high effort does not guarantee
small XY error. No causal claim about effort, sliding or engine differences is made.

The [aggregate JSON](nominal_control_burden_20261016_v1/summary.json) also records
per-joint/group torque RMS, absolute P95/peak, group-L2 statistics, normalized
wheel-action rates and requested wheel torque, for both the core and full clip.
It preserves the original XY, speed, heading, measured-contact-slip and posture
aggregates without changing their definitions. No raw motion, states, action
arrays, models or policy weights are redistributed.

## What is measured

The normalized action has six position-controlled legs and two torque-controlled
wheels. With constant `q0` and unit leg scale, `Δa_leg / 0.02` is the commanded
leg-target rate, **not measured joint velocity**. Wheel `Δa / 0.02` is in normalized
units/s, not wheel angular acceleration. These are 50 Hz finite-difference slew
proxies for stepwise targets, not continuous physical target velocities.
Requested wheel torque is `12 a_wheel`
at 50 Hz, before motor speed-dependent clipping; recorded actual torque remains
authoritative. For example, SAC's core requested wheel-torque RMS is 9.74 Nm in
Isaac and 9.13 Nm in MuJoCo, versus actual RMS 9.15 and 5.07 Nm. This difference
is not presented as a measured clipping rate or a calibrated motor diagnosis.

Post-step torque samples describe the preceding 5 ms physics substep. Initial
`t=0` torque is excluded: Isaac can retain cached reset effort before any audited
action. Rates use differences between actual successive commands, without an
invented first reset delta. The aligned burden core uses command issue times
`[1.44,4.00)` and their post-step torques `(1.44,4.00]`: 128 commands, 512 torque
samples and 128 boundary-aware command differences. Full-clip counts are 349,
1,396 and 348 respectively. The unchanged canonical XY/heading/contact core
remains the closed interval `[1.44,4.00]` with 513 physical samples.

There is **no power or energy** estimate, torque–velocity product, DC-envelope
occupancy or inferred command-clipping statistic. The recording phases do not
justify silently identifying post-step joint velocity with the velocity used
inside each actuator update. Torque magnitudes are simulation effort, not
verified hardware limits or battery consumption.

## Scope and existing top-down views

Use the original three-panel views; no effort plots are added:

- [PPO: reference / Isaac / MuJoCo](controller_primary_ppo_nominal_pair_v1/topdown.png).
- [SAC: reference / Isaac / MuJoCo](controller_primary_sac_nominal_pair_v2/topdown.png).
- [NMPC H30/B5: reference / Isaac / MuJoCo](controller_primary_nmpc_residual_h30_b5_nominal_pair_v1/topdown.png).
- [NMPC H30/B10: reference / Isaac / MuJoCo](controller_primary_nmpc_residual_h30_b10_nominal_pair_v1/topdown.png).

PPO/SAC each used 12,288,000 training transitions, but normalization, learner reuse
and compute differ. Their deployed actors use 51 inputs; NMPC has full physical
and contact-memory state plus 0.60 s reference preview. Matching stage terms
does not equalize RL discount/bootstrap/orientation-termination semantics or
computation. NMPC B5/B10 also change residual-block duration and search dimension.
Different historical controller sources/native builds are retained with their
actual hashes; this is not an identical-source frozen benchmark.

This audit excludes the six-iteration runtime-failure diagnostic and subsequent
experiments. It does not freeze NMPC or start a friction grid. Height, flight,
tilt and nonzero heading offsets do not become new failure gates.

## Offline reproduction

The [analyzer](../scripts/analyze_nominal_control_burden.py) requires the original
eight raw episodes and the four public canonical pairs. It verifies their byte
bindings, named action/torque axes, 349 decisions, 1,397 physical samples and
float32 four-substep action holding before computing aggregates. Historical
MuJoCo policy traces retain the canonical capture-manifest binding where their
original report did not itself hash the rollout; this is not upgraded to an
independent execution proof. The old contract's task-profile field is ignored:
only the explicit shared actuator fields are used here.

```sh
python scripts/analyze_nominal_control_burden.py \
  --actuator-contract /path/to/shared/policy_contract.json \
  --case PPO results/controller_primary_ppo_nominal_pair_v1 \
    /path/to/ppo/isaac/physics_trajectory.npz /path/to/ppo/isaac/report.json \
    /path/to/ppo/mujoco/rollout.npz /path/to/ppo/mujoco/report.json \
  --case SAC results/controller_primary_sac_nominal_pair_v2 \
    /path/to/sac/isaac/physics_trajectory.npz /path/to/sac/isaac/report.json \
    /path/to/sac/mujoco/rollout.npz /path/to/sac/mujoco/report.json \
  --case NMPC_H30_B5 results/controller_primary_nmpc_residual_h30_b5_nominal_pair_v1 \
    /path/to/b5/isaac/rollout.npz /path/to/b5/isaac/report.json \
    /path/to/b5/mujoco/rollout.npz /path/to/b5/mujoco/report.json \
  --case NMPC_H30_B10 results/controller_primary_nmpc_residual_h30_b10_nominal_pair_v1 \
    /path/to/b10/isaac/rollout.npz /path/to/b10/isaac/report.json \
    /path/to/b10/mujoco/rollout.npz /path/to/b10/mujoco/report.json \
  --output-dir outputs/NEW_nominal_control_burden
```

Existing outputs are never overwritten. Asset-free synthetic tests validate the
clock masks, reset-cache exclusion, rate boundaries and statistic definitions.

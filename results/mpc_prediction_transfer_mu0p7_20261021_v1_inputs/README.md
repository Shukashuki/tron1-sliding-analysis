# Executed CEM physics-transfer inputs

These files preserve evidence for the nominal `traj_4`, μ=0.70, CEM
predictor-by-plant experiment. They contain generated simulation data and
source/configuration snapshots, not robot assets, learned weights or credentials.

| Directory | Prediction model | Validation plant | Origin |
| --- | --- | --- | --- |
| `mujoco_isaac` | Native MuJoCo CPU | Isaac/PhysX | Archived nominal episode |
| `mujoco_mujoco` | Native MuJoCo CPU | MuJoCo | Archived nominal episode |
| `physx_isaac` | Isolated Isaac/PhysX candidate worker | Isaac/PhysX | New full episode |
| `physx_mujoco` | Isolated Isaac/PhysX candidate worker | MuJoCo | New full episode |

Each directory contains the original, unedited `report.json` and generated
`rollout.npz`. The report's rollout SHA-256 binds it to the recording used to
create the published compact pair.
The original two pairs retain native 200 Hz world-coordinate traces and
recomputed wheel geometry; the combined bundle does not time-warp or register
their trajectories.

## Exact executed source

`executed_snapshot_v5.tar.gz` contains 20 source, configuration and test files.
Its SHA-256 is
`632b6e27fc11daef82c96b9d5971804c524c7d8b2af789f40d9a6edf8dc2500e`.
Both new full episodes and the published calibration probe use this snapshot.
Runtime reports also hash the existing remote task source, asset and motion.
Those dependencies are not redistributed in this archive.

The archive preserves the exact executed runner files, including pre-existing
friction-oracle support in the working tree. Neither episode enables that
option: prediction μ=0.70 was predeclared, not read from the validation plant.
The repository change adds PhysX backend selection without committing the
unrelated working-tree oracle changes. Use the archive when reproducing exact
runner source hashes.

The machine-specific worker configuration records the actual installation
paths. For another machine, create a new configuration with the matching task,
motion, asset, Python and BeyondMimic source paths. Preserve explicit `cuda:N`,
64 candidates and `environment_spacing_m: 0.0`. Co-located candidates are
isolated by PhysX environment collision filtering; differing grid offsets
otherwise introduce slot-dependent numerical differences.

## Calibration evidence

`probe_h30` contains the original probe report and three generated candidate
recordings at reference frames 0, 100 and 150. Each tests 64 candidates over
30 control intervals, with four physical substeps per interval. The probe
checks zero-step state reconstruction, first-substep actuator torque, scalar
versus batch stage cost, duplicate candidates, A/B/A reset repeatability,
candidate permutation and changing only one candidate. These are empirical
checks on the tested states and plans, not proof that internal PhysX contact
caches are serialized or that all states will repeat identically.

The matrix analyzer rechecks the saved array shapes, finite bounded controls,
stage-cost sums, candidate-cost identities and source bindings. Repeat,
torque, clock and scalar-cost parity diagnostics remain measurements reported
by the original probe, not independently replayed hidden engine state.

## Scope

The optimizer, objective and action contract are fixed. There is no PPO
initialization, action replay, prescribed root trajectory, external root force
or controller retuning. Height/flight and heading offset are diagnostics, not
failure labels. This is an offline closed-loop comparison; simulated time
does not advance while each control solution is computed.
The current PhysX worker steps all 64 environments even for small rescoring
batches, discarding padded slots from returned costs. Recorded latency includes
this implementation overhead and IPC; it is not an intrinsic engine-speed test.

The historical MuJoCo predictor cold-resets integration state every 20 ms.
The PhysX predictor reconstructs once at candidate start and then integrates
continuously over the horizon. Both retain the same MuJoCo bias-force nominal
seed. Thus this compares predictor implementations/backends, not an isolated
change to a numerical solver or a claim about physical accuracy.

Candidate arithmetic also differs: MuJoCo uses float64 whereas the PhysX
worker quantizes candidate controls to float32 before simulation and scoring;
both validation plants consume float32 actions. Contact reward channels use
post-step `mj_forward` constraint forces in MuJoCo and completed-step contact
impulses divided by the time step in PhysX. Matching reward formulas and
thresholds does not make these measurement operators identical. XY and speed
are the primary outcome comparisons.

# Hockey-stop learning success and increased speed

Status: **explicitly approved, uploaded and launched** in tmux `hockey_speed_success_20261028_v1` on the authorized RTX host. The original-recipe, original-speed, seed-42 smoke passed; its formal training is running. No success-rate result exists yet. Previous single-rollout successes do not establish a success rate. [Launch evidence](preflight/launch.json) · [Upload receipt](preflight/upload_receipt.json) · [CPU checks](preflight/local_tests.json).

The requested baseline is the earlier **TRON1 BeyondMimic-style PPO tracking recipe**, rather than the previous hockey experiment's zero-support-weight arm. The second method is the directly trained hockey tracking PPO, without the additional continuous support penalty. The full recipes differ in observations, rewards, learning-rate schedule and episode-end semantics; this is not a single-reward causal ablation.

| Shared reference condition | Original speed | 1.5× speed | 2× speed |
| --- | ---: | ---: | ---: |
| Designed base-link entry speed | 1.307 m/s | 1.961 m/s | 2.615 m/s |
| Total designed XY displacement, including approach | 1.438 m | 2.157 m | 2.876 m |
| Translational kinetic energy at equal mass | 1× | 2.25× | 4× |
| Braking interval | 0.40–1.80 s | Same | Same |
| Pose clock / final reference time | Unchanged / 2.76 s | Same | Same |

Speed scaling changes world translation and its contribution to every body's COM velocity. It preserves GMR joint positions/velocities, body and wheel orientations, angular velocities, height and phase timing. A fixed braking duration means the target stopping distance grows with speed. These are designed robot commands, not measured human skating speeds. The original-speed motion bytes are unchanged.

Wheel spin remains the same zero placeholder at reset in every condition, excluded from imitation. The higher-speed evaluations therefore include the initial wheel spin-up transient. A failed training run means failure within the fixed budget and reset convention, not proof that a physical hockey stop is impossible at that speed.

| Recipe | Earlier BeyondMimic-style tracking | Direct hockey PPO |
| --- | --- | --- |
| Actor / critic inputs | 51 / 120 | 56 / 125 |
| Imitation objective | Six exponential global/relative body tracking terms | XY, velocity vector, heading, yaw rate, leg pose, wheel axes, loaded β |
| Explicit velocity / braking / holding input | No | Yes |
| Terminal supported-stop reward | No | Yes |
| Extra continuous support-deficit penalty | No | No |
| Learning rate / schedule | 0.001 / adaptive | 0.0003 / fixed |
| Reference-end termination | Historical timeout semantics | Finite-task terminal |
| Training orientation termination | Historical reference-orientation error, 0.6 rad | Absolute tilt, 45° |

Both use the same PhysX robot/actuators, 400 Hz physics, 50 Hz control, action-rate weight −0.5, network widths, PPO clip/epochs/minibatches, and training friction DR over [0.3, 1.2]. Encoder bias, COM offsets and velocity pushes are also shared. Absolute height tracking and height termination remain disabled, as previously requested. The original recipe therefore retains the existing TRON1 free-height and simulation-clock adaptations; it is not an unmodified official humanoid task. Its wheel-relative XY tracking remains active; wheel axial orientation/spin imitation remains excluded.

The hockey arm retains the documented v1 sampler and float32 phase-boundary limitations. The undeployed v2 correction is not introduced during this comparison. Those implementation differences remain part of the full-recipe comparison and cannot be attributed solely to β or the stopping reward.

## Success denominators

- **Learning success:** three independent training seeds (42, 43, 44) per method and speed, each starting from scratch with 12,288,000 transitions. Report nominal full-start passes out of three, including unsuccessful seeds. Three seeds provide only a preliminary reliability estimate.
- **Conditional rollout success:** each trained policy runs the same 32 IID initial-state perturbations at μ = 0.7, in addition to one exact nominal episode. Report successes/32 and a 95% Wilson interval for each policy. Report the mean and min/max across training seeds separately; do not pool 96 correlated trials into an IID binomial confidence claim.
- **Paired failures:** retain original-only, hockey-only, both-success and neither-success outcomes for matching perturbation IDs. Report nonexclusive failure counts for XY, speed retention, heading, support, tilt, terminal stop and corroborated sliding.

The perturbations are uniform initial XY ±0.02 m, yaw ±2°, forward COM speed ±5%, lateral COM velocity ±0.05 m/s, and world yaw rate ±0.1 rad/s. They are drawn once with seed 20261028 and paired across methods and training seeds. Height, joint pose, reference commands, friction and subsequent disturbances are unchanged. The evaluator observes all 139 controls and 1,112 physical substeps, without resets after failure. The nominal case is excluded from the random-trial denominator.

All **eleven existing physical stop-and-slide gates** remain fixed, including approach-speed retention, XY RMSE ≤0.15 m and peak ≤0.30 m, continuous at-least-one-wheel support, supported terminal stop, heading/tilt limits, and ≥50 ms of same-wheel loaded high-β material sliding corroborated by near-limit aggregate friction and dissipative force. No height gate is added. Friction evidence remains wheel-aggregate, not a pointwise cone certificate.

## Frozen execution scope

[Plan and case draws](source_snapshot_v1/plan.json) · [Payload manifest](source_snapshot_v1/manifest.json) · [Runtime preflight](preflight/remote_runtime.json) · [Upstream source parity](preflight/upstream_remote_parity.json)

The payload contains 41 files, 736,575 bytes, plus its manifest. Archive SHA-256: `f8a3d60a547094df98f0290a8a81da3a8f8820d15ed86ce485eb39f765b8b1ac`. It includes code/configuration and derived robot references; no robot models, weights, credentials, video or SMPL assets.

The prepared scope is 18 formal trainings, 18 short smoke checks, and 594 single-environment episodes. Each smoke uses 32 environments × 2 iterations; each formal run uses 512 × 1,000 iterations × 24 steps = 12,288,000 transitions. Total formal budget is 221,184,000 transitions. At the previous 1,341 s per training, training alone would take approximately 6.7 hours; evaluation and resource waiting are additional.

The dispatcher uses one freshly audited, completely idle RTX at a time. It runs seed 42 across all speeds first, then seeds 43 and 44. It stops on infrastructure/audit failure, retains unsuccessful policies, does not retry or tune automatically, and never stops other processes. Idle polling has a six-hour cumulative limit. The approved runtime and assets are reused without installing packages.

## Reproduction

Run from the repository root. Build a **new** payload from current source with:

```bash
python scripts/prepare_hockey_speed_study.py \
  --output outputs/hockey_speed_study_reproduction/source \
  --archive outputs/hockey_speed_study_reproduction/source.tar.gz
```

To reproduce this exact study, use the committed immutable snapshot instead. On the authorized compute host, after verifying the manifest and supplying the existing runtime paths:

```bash
"$HOCKEY_ISAAC_PYTHON" "$HOCKEY_BUNDLE/dispatch_hockey_speed_study.py" \
  --bundle "$HOCKEY_BUNDLE" --prior-launch "$HOCKEY_PRIOR_LAUNCH" \
  --audit-helper "$HOCKEY_GPU_AUDIT_HELPER" --output "$HOCKEY_NEW_RUN_DIRECTORY"
```

The dispatcher launches training, the full episode suite, and a separate CPU analysis that reconstructs every observation/action and native contact before qualification. Its status file records process identity, GPU audit, arguments, return codes and report hashes. Once collected, summarize only complete audited cells:

```bash
python scripts/summarize_hockey_speed_study.py \
  --plan results/hockey_speed_success_20261028_v1/source_snapshot_v1/plan.json \
  --runs "$HOCKEY_COLLECTED_RUNS" --output outputs/hockey_speed_success_summary
```

Missing cells remain explicitly pending; they are never counted as successes or silent failures. Rendered examples must be selected after reporting every trial, with representative failure cases retained alongside successful nominal runs.

Read-only live progress and evidence retrieval:

```bash
python scripts/collect_hockey_speed_study.py --status-only
python scripts/collect_hockey_speed_study.py --representative-traces \
  --output outputs/hockey_speed_success_collection
```

The collector downloads completed reports and optionally nominal / worst failed-XY traces. Actor weights and checkpoints remain on the authorized host; no remote process is started, stopped or modified by collection.

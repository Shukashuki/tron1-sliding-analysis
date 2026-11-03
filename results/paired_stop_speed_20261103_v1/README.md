# Direct braking versus hockey stop at higher speed

This study increases initial speed while keeping the **1.438207 m total
forward-distance budget**. Both behaviors train from scratch at each tested
speed, with paired seeds, the same simulator, training budget and evaluation
perturbations. The question is whether either behavior can stop more reliably,
with continuous bilateral contact and less body jitter.

The queue is launched, initially waiting for the current low-speed training
stage to finish. Remaining low-speed work is deferred, and existing results
are preserved. See [collected progress](launch_evidence/progress.json) for the
timestamped status; this document does not claim a completed speed result.

| Speed factor | Common initial COM XY speed | Brake phase used for scoring | Evaluation duration |
|---|---:|---:|---:|
| 1.5 | 1.955 m/s | 0.28–1.20 s | 2.16 s |
| 2.0 | 2.607 m/s | 0.20–0.90 s | 1.86 s |
| 2.5 | 3.258 m/s | 0.16–0.72 s | 1.68 s |
| 3.0 | 3.910 m/s | 0.14–0.60 s | 1.56 s |

The original GMR joint/root sequence is uniformly time-compressed, then held
stationary to preserve 0.96 s after the designed end of braking. Forward
distance is unchanged. Root orientation uses a rotation spline, scalar
coordinates use shape-preserving cubic interpolation, and rigid-body poses
and COM velocities are recomputed with forward kinematics. No new IK or
physics rollout generates these references. FK consistency does not establish
dynamic feasibility. Control stays at 50 Hz and PhysX at 400 Hz. The brake
reward/scoring start is rounded up to the next control sample (at most one
20 ms interval); the exact kinematic phase is also recorded.

Direct braking preserves the initial body orientation and leg posture as soft
targets, with beta and positive lateral-sliding reward weights set to zero.
Hockey retains its GMR pose/heading targets and beta/sliding rewards. Both
have exactly the same reference base-link XY at every control frame. This is
a comparison of complete behavior/reward recipes, not an isolated beta test.
Wheel angles remain imitation placeholders; policies actively command wheel
torque. There is no absolute body-height or equal-load requirement.

The common evaluation reset scales every archived velocity field together,
leaving pose unchanged. The existing zero wheel-speed initialization is
retained; it is not asserted to be a force-balanced rolling state. Both
behaviors receive the same physical reset and paired perturbation, with
readback checks. Their reference-state initialization during training differs
with their own targets. Actual brake-entry speed is reported: matching the
whole maneuver's initial state does not guarantee equal realized speed at the
brake phase, or isolate intrinsic braking efficiency.

Each policy receives 12,288,000 parent transitions, 3,072,000 contact-refinement
transitions and 9,216,000 optimizer-continuation transitions, totaling
**24,576,000**. Each stage first runs a 1,536-transition smoke check; the parent
also receives a single-world physical-reset probe. Formal training uses 512
environments. Contact refinement retains the existing bilateral-contact,
smooth XYZ-stop and angular-acceleration terms. The prior action-rate weight
of −0.5, optimizer settings and DR remain frozen. Training retains effective
friction U[0.3, 1.2] and the existing encoder/COM/push randomization; this speed
study uses equal static/dynamic coefficients, with nominal evaluation at
mu_s = mu_d = 0.7.

Seed 43 explores ascending speeds. Both policies must finish all training and
all 33 evaluations before the next decision. The first level where either
policy passes fewer than 30/33 common task gates is selected; otherwise the
highest declared level is selected. Both behaviors then train afresh with
seeds 44 and 45 at that level. Selection is symmetric: hockey winning is not
required, and a level where both fail is retained. Seed-43 selection results
are exploratory; the two additional paired seeds are confirmation, not 33
independent training seeds. The maximum formal budget is 294,912,000
transitions. There is no outcome-dependent reward tuning or replacement of
failed seeds.

Common gates require bilateral contact at every native 2.5 ms sample, no
unknown contact, no fall or non-wheel support, XYZ speed at most 0.1 m/s held
for 0.5 s, terminal/stop position within 5 cm of the distance target, forward
overshoot at most 5 cm, lateral excursion at most 10 cm, and the existing
XY/heading/approach-speed gates. Contact uses the same 0.1 mm geometric
tolerance. Slip is diagnostic rather than necessary for either behavior's
success. The report also preserves individual failed gates, contact fraction,
stopping time/distance, actual brake-entry speed, lateral excursion, tilt,
action variation and 10–100 Hz body angular RMS; a binary score alone is not
an explanation of failure.

Reproduce reference generation and the frozen source using the existing
licensed/local robot asset and archived source files:

```bash
python scripts/prepare_paired_stop_speed.py \
  --output outputs/paired_stop_speed_reproduction \
  --handoff-sha256 693ef7b120088a6afaf62198881af1bf65bdf01367ca0d4c8b824335ebbac20e
python -m pytest -q tests/test_paired_stop_speed.py \
  tests/test_direct_brake_training.py tests/test_hockey_equal_distance.py
```

The deployment plan is pinned to this run's remote workspace and process
handoff. A new deployment must declare its own workspace/handoff, rather than
reusing historical PIDs. The builder refuses to overwrite an existing study.
The archive contains 108 program/configuration/derived-reference files plus
its manifest; no robot models, learned weights, credentials or source video
are uploaded. Existing remote assets and upstream code are hash-checked.

- Workspace: `/data/zengxiong/workspaces/tron1-paired-stop-speed-20261103-v1/`
- tmux session: `tron1-paired-stop-speed-20261103-v1`
- Queue status: `runs_v1/status.json`; log: `dispatcher_v1.log`
- [Frozen plan](source_snapshot_v1/plan.json), [manifest](source_snapshot_v1/manifest.json), [payload](payload.json)
- [Validation record](verification.json)

Only one GPU worker runs at a time. The queue checks for no compute or graphics
processes, zero utilization and less than 256 MiB used on three consecutive
samples, then uses a cooperative device lock. The preceding training worker
is never signalled. After its verified normal exit and checkpoint completion,
only its paused CPU dispatcher is retired, releasing the old queue's device
lock; remaining low-speed stages stay deferred. Infrastructure failures stop
the queue without automatic retries.

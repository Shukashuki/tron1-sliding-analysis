# Direct braking at the same forward stopping distance

This experiment trains a new direct-braking PPO policy in Isaac Lab / PhysX.
The primary comparison with hockey stop is **stable completed stopping with
less jitter**, at the same initial physical state and forward-distance budget.
Training completion is not evidence of task success.

**2026-11-03 update:** seed 43's equal-friction lineage and 33 evaluations are
complete. See the [measured comparison and queue recovery](../direct_brake_comparison_20261103_v1/README.md).
The user subsequently prioritized the [higher-speed paired comparison](../paired_stop_speed_20261103_v1/README.md).
The current split-friction continuation is allowed to finish; subsequent
low-speed stages and seeds are deferred, with existing artifacts preserved.

**Historical launch status:** seed 43's parent was formally training on RTX 4090 D GPU 2.
The 1,536-transition smoke test and 139-control-step shared-reset probe passed.
See [recorded progress](launch_evidence/progress.json) for the timestamp and
actual optimization counters; this is a collected snapshot, not a live page.

The reference keeps the initial body orientation and leg posture as soft
targets. Its base-link XY translation matches the hockey reference at every
frame: designed entry speed 1.30746 m/s, 0.52298 m approach and 0.91522 m braking
distance, totaling 1.43821 m. The policy controls both wheel torques; wheel
angles in the reference are placeholders, and neither wheel is locked.
The shared measured evaluation reset has base COM speed about 1.30329 m/s.

β tracking and the positive lateral-sliding reward have weight zero. No
friction-boundary bonus or absolute body-height constraint is introduced.
Incidental physical slip is measured; it is neither prohibited nor required
for direct-braking success. Joint-pose and wheel-axis targets remain soft.
This is a behavior-recipe comparison, not an isolated causal test of a human
posture prior.

## Frozen training sequence

The declared seeds are 43–47. Every seed starts from scratch, without a hockey
actor, then branches into equal and split friction. No seed is dropped or
selected based on performance.

| Stage | Transitions per seed | Initialization |
|---|---:|---|
| Direct-braking parent | 12,288,000 | Random actor/critic |
| Equal-friction contact refinement | 3,072,000 | That seed's completed direct parent; new optimizer |
| Equal-friction continuation | 9,216,000 | Its contact refinement, including optimizer |
| Split-friction contact refinement | 3,072,000 | The same seed's completed direct parent; new optimizer |
| Split-friction continuation | 9,216,000 | Its contact refinement, including optimizer |

Each final policy has a 24,576,000-transition lineage. Sharing the parent gives
36,864,000 unique formal transitions per seed, or **184,320,000 total** across
five seeds. This corrects the ambiguous aggregate budget in the older
replication draft. Each training stage first runs 32 environments × 24 steps ×
2 iterations = 1,536 smoke transitions; formal training uses 512 environments.
Physics runs at 400 Hz and policy control at 50 Hz.

The parent uses the frozen original parent code. Contact stages use the frozen
v7/v8 refinement recipe: bilateral contact fraction/loss/clearance shaping,
XYZ stop smoothing and angular-acceleration smoothing. These stronger contact
terms start in the refinement stage, not the from-scratch parent. Actor,
critic, PPO settings, DR and stage budgets otherwise follow their historical
counterparts. Evaluation-only source insertions add the common reset and
archive actuator torque and body contact telemetry; the training branches
are unchanged.

The reused trainer retains its internal `hockey_*` names. The direct-braking
protocol, reference hash and explicit `direct_brake` metadata identify this
new behavior; those legacy labels do not imply a hockey policy initialization.

Training retains effective wheel-contact dynamic friction U[0.3, 1.2], together
with the original encoder, COM and push randomization. The equal branch uses
μs/μd = 1; the split branch uses μs/μd = 0.9/0.7. These are exploratory simulator
materials, not measured tire properties. Nominal evaluations use 0.7/0.7 and
0.9/0.7, respectively. Each final policy records the same 33 nominal/perturbed
trials at its own material and two additional cross-material nominal episodes
from {0.7/0.7, 0.9/0.7, 0.9/0.9}; the duplicate own-material nominal is reused.

## Evaluation and evidence

All evaluations write the same archived base-link pose, COM twist and named
joint state before applying the paired perturbation. Readback must agree to
5e-5 and previous actions must be zero. Training retains reference-state
initialization for each behavior; equal physical resets are an evaluation
contract, not a claim of identical RSI distributions.

The prospective forward-distance tolerance is 5 cm, the maximum allowed
forward overshoot is 5 cm, and the lateral corridor is 10 cm. These are
declared engineering thresholds, not calibrated physical measurement limits.
Stopping requires speed ≤0.1 m/s held for at least 0.5 s; bilateral continuous
contact remains mandatory, with native contact/geometry checked at 400 Hz and
unknown contact not counted as success. No equal-load requirement is imposed.
Falls and non-wheel support disqualify a trial. Slip and wheel-heading gates
specific to hockey stop are excluded from the direct-braking success test.

The queue archives raw states, actions, actuator effort, body contact forces,
native wheel contact buffers, material readbacks and initialization audits.
Independent qualification and a paired stability report remain separate work;
the queue does not turn training rewards into success rates. Compare raw
stopping distance/time, lateral excursion, contact interruptions, angular
acceleration, 10–100 Hz body angular RMS and action changes. Archived seed-42
hockey results are descriptive; prospective matched hockey seeds still need
their own full lineages.

## Reproduction and remote status

Build the frozen source from the repository's archived sources:

```bash
python scripts/prepare_direct_brake_training.py --output outputs/direct_brake_reproduction
python -m pytest -q tests/test_direct_brake_training.py tests/test_hockey_equal_distance.py
```

The builder is pinned to this experiment's remote workspace. The source-file
hashes are reproducible; tarball metadata is not intended to be byte-identical.
The existing remote assets and upstream sources are reused and hash-checked;
the upload includes no model, checkpoint, credential or original video.

- Host: `zengxiong@14.103.233.29`
- Workspace: `/data/zengxiong/workspaces/tron1-direct-brake-20261102-v1/`
- Session: `tron1-direct-brake-20261102-v1`
- Queue status: `runs_v1/status.json`
- Dispatcher log: `dispatcher.log`
- First training log: `runs_v1/seed43/parent/smoke.log`
- Manifest: [source_snapshot_v1/manifest.json](source_snapshot_v1/manifest.json)
- Frozen plan: [source_snapshot_v1/plan.json](source_snapshot_v1/plan.json)

The queue requires three consecutive observations of an RTX with zero
utilization, less than 256 MiB allocated, and no compute or graphics processes.
Only one GPU job runs at a time. It waits up to 24 cumulative hours for capacity
and never terminates or replaces another workload. A short physical reset
probe follows each parent's smoke training before its formal training starts.
Infrastructure failures stop the queue without an automatic retry; all results
are retained. Check the collected launch/status record for whether training
has actually started.

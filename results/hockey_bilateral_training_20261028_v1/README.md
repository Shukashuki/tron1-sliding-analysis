# Bilateral-contact hockey PPO retraining

The requested objective is continuous **two-wheel geometric contact**, free left/right load transfer, and useful dissipative lateral sliding while tracking and stopping stably. The four previous nominal recordings all fail the new bilateral gate; none is a success baseline under this definition. This is a new optimization recipe, not a one-factor causal ablation.

Status: **GPU 1 formal retraining is running**, using `source_snapshot_v4` and `run_v4` in tmux `hockey_bilateral_20261028_v4`. The 32-environment smoke completed 1,536 transitions without a native-contact fault; maximum authored/native geometry error was 3.56 µm over 17,824 measured wheel samples. The verified formal log reached 184,320 / 12,288,000 transitions (15 completed PPO iterations), around 3,464 transitions/s. Its early ETA was approximately 63 minutes; this is a training estimate, not a completion or success claim. [Smoke report](remote_evidence/run_v4/smoke/report.json) · [Execution status snapshot](remote_evidence/run_v4/status.json) · [Launch record](launch_v4.json).

The user explicitly paused the older speed study and friction-boundary queue. The old `hockey_s100_seed44/formal` trainer and its speed-study scheduler are stopped with SIGSTOP, preserving in-memory progress and GPU allocations. The childless friction-boundary waiting dispatcher was disabled with SIGTERM after its initial SIGSTOP did not persist; it had started no GPU work. [Initial pause receipt](execution_v2/old_training_paused.json) · [Boundary queue disabled](execution_v2/boundary_wait_queue_disabled.json). The old studies must not be resumed without a new user instruction.

The original waiting-only dispatcher was superseded. Before launch, GPU 1 had no active workload; its small inactive CUDA context belongs to the verified old hockey trainer whose main device is GPU 3. The capacity dispatcher checks ownership, exact command, process start identity, memory and utilization before each stage. Other users' processes are ineligible.

The first smoke failed its geometry audit before learning. Diagnostic revision v3 localized intermittent batched native separation mismatches, up to 18.35 mm, while buffer bounds and finite-value checks passed. Revision v4 independently copies normal and friction buffers with explicit GPU synchronization; this eliminated the observed mismatch in smoke and allowed formal training to start. This isolates the problem to the contact-data acquisition path but does not establish which vendor buffer/stream mechanism caused it. Failed reports are retained in `remote_evidence/run_v2` and `run_v3`. The dispatcher now also checks the report's failure status because Isaac shutdown can return exit code zero after a caught training exception. Reward weights, the 0.1 mm touch tolerance, and the 1 mm alignment rejection threshold are unchanged.

## Training scope

One original-speed policy, seed 42, trained from scratch: 32 environments × 2 iterations for smoke validation, then 512 × 1,000 iterations × 24 steps = **12,288,000 transitions**. Physics stays at 400 Hz and actions at 50 Hz. The original reference, robot asset, PPO optimizer settings, friction DR and actor/critic dimensions 56/125 are retained. No contact force, friction coefficient or sliding label is added to the actor. Existing models and reference files are reused remotely, rather than uploaded again.

Independent validation runs the nominal start and the same 32 declared perturbed starts at mu=0.7, each for the full 139 control steps, without early reset. The evaluator reconstructs native contacts, verifies recorded actor inputs/actions and checks whole-episode bilateral contact first, including startup and terminal samples. Failed-contact cases remain diagnostic. Stop, tracking, heading, speed and corroborated lateral-slide gates follow; the slip calculation uses pre-step contact geometry and material velocities, plus the displacement of the same material points across the step.

## Reward changes

| Component | Weight / change | Intended behavior |
| --- | --- | --- |
| Whole-window bilateral contact | +2 | Both wheels have native geometric contact throughout all eight physical substeps; the authored cylinder also remains at ground level after each step |
| Contact interruption | −4 | Half worst-substep loss plus half mean loss, bounded to [0,1] |
| Wheel-ground clearance | −1 | Bounded distance penalty beyond the 0.1 mm contact tolerance, reaching full scale at an additional 1 cm |
| Useful boundary sliding | +0.5 | Bounded lateral motion, dissipative/opposing friction and braking near the wheel-level friction limit, gated by bilateral continuity and tracking/uprightness |
| Body angular acceleration | −0.05 | Bounded squared angular acceleration, scale 50 rad/s² |
| Leg imitation | 1 → 0.25 | Allow posture and load-transfer adaptation to retain both contacts |
| Existing positive task rewards | Additional bilateral gate | XY, velocity, heading, yaw rate, leg pose, wheel axis, beta and stop rewards are zero for a control window with a contact interruption |

A geometric contact requires a valid native point with separation ≤0.1 mm; no force threshold or equal-load target defines touching. Unequal load and a nearly unloaded touching wheel are permitted. Friction quality still depends on actual force: the sliding bonus has a bounded load factor reaching one at 0.25 mg, which prevents negligible-capacity contacts from earning full friction credit. The existing beta/stop terms retain their original load weighting inside the new bilateral gate. At least one wheel may provide useful sliding; both need not slide simultaneously.

The sliding bonus is available only during 0.4–1.8 s braking. Its slip and dissipation ramps saturate, so unbounded speed or energy does not yield unbounded reward. Native force is averaged over the following step; pre-state anchor power is corroboration, not exact integrated work or a per-contact friction-cone certificate. Geometry and velocity are captured before each physics step, fixing the earlier one-step alignment issue. An online authored-cylinder check rejects native/pre-state separation discrepancies above 1 mm before formal training proceeds.

Action-rate −0.5, vertical-acceleration −0.1, the original tracking terms, DR and free body height remain. There is no wheel-XY/spin imitation or imposed absolute body height. Full-clip reset probability rises from 0.2 to 0.5; the explicit phase command uses integer frame boundaries and adaptive reference sampling responds to falls. Contact loss receives penalties and reward gating without an added early termination, so PPO can learn recovery and validation still sees the entire motion.

These incentives do not mathematically guarantee episode-wide contact. The strict independent evaluator is the acceptance test.

## Checks and reproduction

- 40 tests pass: 7 new reward/collector counterexamples, 7 boundary regressions, 12 contact/timing/priority checks, and 14 existing phase/sampling/task regressions.
- Replay agrees with independent aligned evidence over 8,896 wheel/physics samples from four recorded motions. Maximum lateral-velocity discrepancy is below 7.3e−7 m/s; authored/native separation discrepancy is below 5.8e−7 m. No contact-failed control window receives a sliding bonus.
- A separate independent qualification replay still marks the original-speed hockey policy as failing bilateral contact, despite passing its other task gates. Historical success labels are not imported.

[Protocol](source_snapshot_v4/protocol.json) · [Original execution plan, scheduling superseded](source_snapshot_v1/plan.json) · [Current source manifest](source_snapshot_v4/manifest.json) · [Offline sampler/reward audit](offline_audit_v1/report.json) · [Local validation](preflight/local_validation.json) · [Upload receipt](preflight/upload_receipt.json)

The original v1 source snapshot contains 23 program/configuration files, 226,894 bytes, plus its manifest. The current v4 snapshot adds the capacity dispatcher and the contact-reader/report-handling fixes; its exact contents and hashes are in the linked manifest. It contains no robot assets, policy weights, credentials, videos, SMPL assets or reference NPZ. Archive SHA-256: `0a73a730f120b918d5e19ecca48e6b37facf0c886684ac0ec1a1e9d98f6ae96b`.

Remote root: `/data/zengxiong/workspaces/tron1-hockey-bilateral-20261028-v1/`. The current capacity dispatcher checks the source/runtime before each stage, stops on failure without automatic retries, and has a six-hour capacity wait. The original deferred plan is retained as history and is superseded by the user’s request to use GPU 1 immediately. The smoke stage must pass native timing/geometry checks before the complete training starts.

```bash
python scripts/prepare_hockey_bilateral_training.py \
  --output outputs/hockey_bilateral_source \
  --archive outputs/hockey_bilateral_source.tar.gz
python -m pytest -q tests/test_hockey_contact_priority.py tests/test_hockey_contact_timing.py tests/test_hockey_ground_contact.py
python -m unittest discover -s tests -p test_hockey_bilateral_training.py -v
```

Current remote status and progress:

```bash
ssh zengxiong@14.103.233.29 cat /data/zengxiong/workspaces/tron1-hockey-bilateral-20261028-v1/run_v4/status.json
ssh zengxiong@14.103.233.29 tail -60 /data/zengxiong/workspaces/tron1-hockey-bilateral-20261028-v1/run_v4/formal.log
```

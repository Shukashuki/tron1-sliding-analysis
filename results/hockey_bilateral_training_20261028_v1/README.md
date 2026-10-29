# Bilateral-contact hockey PPO retraining

The requested objective is continuous **two-wheel geometric contact**, free left/right load transfer, and useful dissipative lateral sliding while tracking and stopping stably. The four previous nominal recordings all fail the new bilateral gate; none is a success baseline under this definition. This is a new optimization recipe, not a one-factor causal ablation.

Status: **uploaded, runtime verified, and deferred queue running** in tmux `hockey_bilateral_20261028_v1` (PID 4032578). The verified state is `waiting_for_existing_studies`, with zero GPU stages started. [Launch record](preflight/launch.json) · [Verified queue status](preflight/verified_queue_status.json) · [Remote preflight](preflight/remote_preflight.json). No new learning result is claimed here. The job is serialized after the already approved speed study and boundary ablation, and selects a completely idle RTX before each GPU stage. It does not stop existing processes.

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

[Protocol](source_snapshot_v1/protocol.json) · [Execution plan](source_snapshot_v1/plan.json) · [23-file manifest](source_snapshot_v1/manifest.json) · [Offline sampler/reward audit](offline_audit_v1/report.json) · [Local validation](preflight/local_validation.json) · [Upload receipt](preflight/upload_receipt.json)

The source snapshot contains 23 program/configuration files, 226,894 bytes, plus its manifest. It contains no robot assets, policy weights, credentials, videos, SMPL assets or reference NPZ. Archive SHA-256: `0a73a730f120b918d5e19ecca48e6b37facf0c886684ac0ec1a1e9d98f6ae96b`.

Remote root: `/data/zengxiong/workspaces/tron1-hockey-bilateral-20261028-v1/`. The dispatcher checks the source/runtime before each stage, stops on failure without automatic retries, and has bounded waits of 24 hours for dependencies and six hours for an idle GPU. The smoke stage must pass native timing/geometry checks before the complete training starts.

```bash
python scripts/prepare_hockey_bilateral_training.py \
  --output outputs/hockey_bilateral_source \
  --archive outputs/hockey_bilateral_source.tar.gz
python -m pytest -q tests/test_hockey_contact_priority.py tests/test_hockey_contact_timing.py tests/test_hockey_ground_contact.py
python -m unittest discover -s tests -p test_hockey_bilateral_training.py -v
```

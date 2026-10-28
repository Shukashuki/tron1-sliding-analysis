# Hockey MPPI: prepared control search, awaiting upload approval

The [completed conditioning probes](../hockey_stop_conditioning_physx_20261027_v1/README.md)
improved support but did not reproduce the hockey stop. This snapshot prepares
an actual PhysX search against the **unchanged hockey target and stop criteria**.
It has not run. Automatic review rejected the attempted upload because prior
approvals covered other specific payloads; approval for this new archive is pending.

Two limitations in the previous generic car-search interface are corrected:

- The old residual envelope forced control corrections to zero at the end of the
  stop. A smooth initial 100 ms ramp now preserves the first action while retaining
  a free final control knot during the stop hold.
- Previously, `clip(raw_PPO + residual)` could ignore a small correction when the
  raw actor was far outside [-1,1]. The new decoder is
  `clip(clip(raw_PPO) + residual)`. With zero residual it exactly matches the old
  clipped controller, and a correction can now move a saturated prior action.

The PPO keeps the `reference_dq` input intervention that had the best existing XY
and support results. All measured state and previous-action fields remain intact.
This is **PPO-assisted trajectory generation**, not an independent model-only
controller or a newly trained hockey policy.

The new cost follows XY, **planar velocity vector**, body heading, loaded wheel
plane headings, GMR leg posture/body up axis/wheel camber, and valid loaded wheel
beta. It penalizes support gaps, excessive tilt, action changes and terminal speed.
A separate deficit penalizes the absence of a 50 ms loaded lateral-slip event
with aggregate friction utilization in [0.9,1.05] and negative anchor power.
The full [configuration](source_snapshot_v1/config.json) exposes all weights.
There is no absolute height or continuous wheel-spin tracking term. One-wheel
support is allowed; beta is masked below 0.25 m/s and is not imposed at rest.

The planning slip screen uses aggregate proxies. Final qualification still
reconstructs raw contacts and requires opposing lateral force and velocity at the
same friction anchor. A low search cost cannot declare task success. The locked
XY, speed, support, heading, pre-instability sliding and terminal-stop gates remain
unchanged, and the final plan must be rerun in a single environment.

The initial leg residual is fitted to the archived failed `reference_dq` states.
It reduces **fixed-state joint-target error** from 0.725 to 0.053 rad. This is an
initialization calculation, not improved physical tracking: the new closed-loop
states will differ. The unchanged GMR joint targets require at most 0.267 rad
offset from q0 and fit within the existing +/-1 action range. Wheel residuals
start at zero; MPPI determines subsequent wheel torques.

The Gaussian prior is fixed around that initial fit. Each of six updates uses 63
Gaussian samples plus one diagnostic mean rollout, with fixed standard deviations
0.20 for leg actions and 0.35 for wheel actions. All finite sampled candidates
participate with the proper likelihood correction in latent space. The updated
weighted mean is deployed; this is not best-sample selection. The complete
2.78 s rollout includes the 2.76 s reference endpoint and its final control hold.

The bounded execution scope is:

1. One 16-control-step, single-environment smoke check.
2. One 64-environment search: zero-residual baseline, six MPPI population updates,
   and final-mean replay (eight passes total, 139 control steps per pass).
3. One separate 139-step, single-environment final-mean validation with raw native
   contact buffers. Independent assessment and fixed-camera rendering follow.

All stages are serial at mu=0.7, 400 Hz physics / 50 Hz control, each on an RTX
verified completely idle immediately before launch. The dispatcher stops on a
failed stage or lack of an idle GPU. It does not retry jobs, stop other processes,
train a policy, or modify models, weights, the target, or the physics force law.

Eight focused Torch tests passed, including saturated-action correction, terminal
control authority, one-wheel support, low-speed beta masking, root-height and
wheel-spin invariance, force/slip conjunction, terminal speed, and independent
64-world costs. Another 31 evaluator/source-admission tests passed. The cost
functions were also evaluated on all 1,112 recorded steps of the existing dq run
and remained finite. These are CPU checks; native PhysX execution is unverified.

[The manifest](source_snapshot_v1/manifest.json) lists **12 files, 189,195 bytes**
plus the manifest: source, configuration and derived reference/initialization
data. No models, weights, credentials or original video are included. Archive:

`outputs/hockey_physx_mppi_bundle_20261027_v1.tar.gz`

SHA-256: `222097504a25093c8f3181f746211f8ddf3fb154cf78aa3df77fb5e8706c2edd`.

Proposed extraction directory:
`zengxiong@14.103.233.29:/data/zengxiong/workspaces/tron1-hockey-mppi-20261027-v1/source_snapshot_v1/`.

```bash
# Local preparation only; does not upload or run physics.
python scripts/prepare_hockey_physx_mppi.py --output /tmp/hockey-mppi-snapshot

# Torch environment (CPU).
CUDA_VISIBLE_DEVICES='' python tests/test_hockey_mppi_math.py

# Existing scientific Python environment.
python -m pytest -q tests/test_evaluate_hockey_physx.py tests/test_hockey_mppi_source.py

# After a completed single-world validation, first replay controls on CPU
# in the Torch environment. The actor remains local and outside Git.
CUDA_VISIBLE_DEVICES='' python scripts/audit_hockey_control_replay.py \
  --run-dir /path/to/returned/validation \
  --actor /path/to/actor_normalized.pt \
  --search-source results/hockey_stop_mppi_preflight_20261027_v1/source_snapshot_v1 \
  --conditioning-table results/hockey_stop_mppi_preflight_20261027_v1/source_snapshot_v1/nominal_conditioning.npz \
  --command-dir results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2 \
  --output /tmp/hockey-mppi-action-replay

# Scientific Python environment; same locked hockey target and stop gates.
python scripts/evaluate_hockey_physx.py \
  --run-dir /path/to/returned/validation \
  --search-source results/hockey_stop_mppi_preflight_20261027_v1/source_snapshot_v1 \
  --action-replay /tmp/hockey-mppi-action-replay/report.json \
  --conditioning-table results/hockey_stop_mppi_preflight_20261027_v1/source_snapshot_v1/nominal_conditioning.npz \
  --command-dir results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2 \
  --output /tmp/hockey-mppi-qualified
```

[status.json](status.json) records the approval block. The previous goal turn made
progress by completing and reviewing the two probes. This turn adds the concrete
search implementation and independent qualification support; no new physical
result or successful motion is claimed.

## Applied-action replay

The subsequent local audit replays all **417 control steps** in the three already
completed physical trajectories using the exact frozen actor. Its saved normalizer
is applied once, and all 51 observed fields are first reconstructed from the
recorded pre-action states. All cases reproduce the applied actions:

| Existing run | Maximum action error | Raw actor elements clipped, whole rollout | Clipped before first sampled 45-degree tilt |
|---|---:|---:|---:|
| Original hockey | 1.05e-6 | 41.5% | 28.6% (14 control samples) |
| Reference dq replacement | 1.07e-6 | 38.1% | 22.0% (59 control samples) |
| Reference q/dq/phase replacement | 8.94e-7 | 24.7% | 10.2% (75 control samples) |

These percentages count individual action components; their windows differ and
they are not independent performance estimates. The replay confirms the recorded
controller outputs, not motor torque accuracy or the cause of the subsequent
falls. No new physics or improvement in stop performance is claimed.

For the pending MPPI experiment, the new local verifier reconstructs the saved
plan independently with SciPy cubic Hermite interpolation. It preserves the full
139-frame clock even for a 16-step smoke prefix and checks the two-stage action
clipping. Native MPPI qualification now requires this replay report, bound to the
exact actor, reference, trace, mean and source snapshot. Altered controls, a changed
plan or a falsely low summary error cannot pass using stale replay arrays.

Four replay contract tests and 33 evaluator/source/binding tests passed. The
[replay preflight](control_replay_preflight.json) and [three actual CPU audit
reports](control_replay/) record the evidence. The 12-file remote archive and its
SHA remain unchanged: all new replay tooling runs locally after results return.

# Drift candidates after excluding lateral/crossover steps

## Decision

Lateral and crossover steps are **excluded from active drift selection**. Existing
GMR animations and rate sweeps remain historical geometric-offset controls.
Running/jogging and skating remain excluded. No new policy or sim2sim run was started.

The preferred motion families are sustained circular drifting and continuous
left/right drift transitions (figure-eight), with translation, substantial heading
offset and explicitly measured speed retention **throughout each turn**. A large
offset in straight sideways travel or an almost stationary spin is insufficient.
Evaluate individual turns with unwrapped course: endpoint heading changes cancel
over complete circles and figure-eights. The inherited exploratory 80% speed
retention limit is not a claim of zero deceleration or a physical drift definition.

## What was actually found

| Source | Available evidence | Selection status |
| --- | --- | --- |
| [XCAR individual-wheel-drive drifting](https://github.com/zhou-yh19/xcar-rlgpu) | Open circular/figure-eight/continuous-drift tasks; [paper](https://arxiv.org/abs/2507.23339) reports simulation and real-car demonstrations | Best matching **motion families**; no ready-made numeric measured trajectory found in inspected tree |
| [SFU `0018_DanceTurns002`](https://mocap.cs.sfu.ca/) | Downloaded BVH with global root translation and bilateral ankle/toe pivots | New non-running/non-skating human candidate, but not a qualified high-momentum drift |
| SFU `0018_DanceTurns001` | Downloaded on-the-spot dance spins | Low-translation negative control, not selected |
| [Alpine skiing GNSS + IMU supplement](https://zenodo.org/records/18468439) | Downloaded 14.17 MB XLSX; 12 runs, each with COM XYZ and midpoint-between-ankles XYZ | Fast translation, but released data lack separate ankles and foot headings; insufficient for bilateral GMR |
| [Ski-Pose PTZ](https://www.epfl.ch/labs/cvlab/data/ski-poseptz-dataset/) | Calibrated multi-view skiing pose dataset | Requires an institutional access request; none sent |
| [Transient car NMPC supplement](https://api.figshare.com/v2/articles/25263323) | API lists two MP4 files only | Not numeric sensor trajectories; video reconstruction would be a separate method |

XCAR is pinned as [a submodule](../upstream/xcar-rlgpu) at
`eeeb289d8b700b3c9137ef40614988deeff31392`; no upstream implementation is copied.
Its fixed-circle task uses radius 1 m and a beta target of −1 rad (−57.3°).
These are **desired targets, not achieved states**. Its low-speed penalty
saturates at 0.5 m/s and does not enforce constant speed. The platform also has
steering and four wheels: independent drive alone does not establish TRON1
feasibility. A controller rollout/state export and source/robot-scale speed audit
would be needed before using this as a demonstrated reference.

The available Holden HDM05 archive was inventoried via bounded HTTP ranges:
**2,344 processed BVH members**. The inventory includes walking circles, turns,
shuffle, hops, kicks and cartwheels, but provides no demonstrated qualifying drift.
This is a filename/class inventory, not a quantitative rejection of all 2,344
motions and not proof that inaccessible original full takes lack suitable actions.

## Downloaded human-source checks

SFU checks use the pinned GMR vendor reader for **forward kinematics only**.
No IK/retarget call is made. Uniform leg-length normalization uses the previously
audited TRON1 hip–knee–ankle chain length, 0.603841 m; speeds below are therefore
robot-scale **source root** speeds, not anatomical human speed or robot CoM speed.

| BVH | Duration | Root speed P95 / peak | Longest bilateral per-foot window: speed ≥0.8 m/s, axial offset ≥20°, near-ground proxy |
| --- | ---: | ---: | ---: |
| `DanceTurns002` | 15.55 s | 0.950 / 1.268 m/s | L 0.383 s; R 0.300 s |
| `DanceTurns001` | 6.04 s | 0.560 / 0.948 m/s | L/R 0.058 s |

Neither has samples above 1.5 m/s at its native clock. These angle windows do
not yet require course change or speed retention; **they are not drift passes**.
Foot headings use ankle-to-toe-base pivots, and ground proximity uses the same
pooled 5th-percentile low-foot height +0.10 m proxy as the HDM05 source checks.
No force/contact-patch measurement is present. Artificially speeding up playback
is not evidence that the source contains sustained inertial turning.

The [alpine study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0347192)
describes global GNSS/IMU alignment and resampling to 240 Hz. Assuming that
sampling rate for the timestamp-free workbook, its 12 **human** COM tracks have
horizontal speed P95 values of 11.12–14.28 m/s. This addresses translation speed,
not bilateral offset: COM-to-ankle-midpoint direction must **not** be substituted
for either foot's heading. Skiing is an optional domain expansion, not silently
selected in place of HDM05; slope-powered motion cannot be flattened into a
physically equivalent flat-ground TRON1 reference without a new adaptation.

## Reproduce

Candidate decisions are in [the selection manifest](../config/drift_candidate_selection.json).
[The machine-readable audit](drift_candidate_search/source_audit.json) contains
source URLs, SHA-256 hashes, bounded archive inventory, released workbook schemas
and native-clock FK statistics. Raw third-party files remain in ignored `outputs/`.

```bash
python scripts/inspect_drift_candidates.py \
  --include-hdm-index \
  --expected-audit results/drift_candidate_search/source_audit.json \
  --audit outputs/replayed_drift_source_audit.json
python -m pytest -q
```

This is **source discovery**, not a new GMR result, controlled-drift demonstration
or friction failure-boundary experiment. No newly qualified human drift is claimed.

# Hockey-stop reference audit and prepared PhysX probe

**Follow-up:** the [evaluation and geometry audit](../hockey_stop_evaluation_audit_20261026_v1/README.md)
found a 10 mm / 50 mm wheel-width mismatch and prepared a corrected v2 command.
The v1 upload request below is historical and superseded. Independent evaluation
also replaces the earlier inline success screen; no hockey physics result exists yet.

The user-provided SMPL-X models enabled the first GVHMR reconstruction and GMR
preview. This follow-up corrects the foot-frame and ground-geometry mapping.
**No hockey-stop physics result is available yet.** The reference is not admitted
for policy training, and neither loaded sliding nor friction saturation is claimed.

## What the reference supports

Heel-to-toe surface frames differ from the original ankle-axis wheel headings by
8.0 / 5.4 degrees RMS. These anatomical frames are estimates, not measured blade
edges. The frozen raw GVHMR output was reproduced before extracting them.

The raw reference has a peak speed of only 0.338 m/s at robot leg scale. A separate
camera-rotation-compensated estimate, assuming a fixed camera center, produces a
13.85 m/s peak and 8.27 m path versus the raw estimate's 0.36 m path. That
alternative is diagnostic only; neither estimate supplies calibrated stopping
distance or speed. See [geometry audit](report.json).

GMR is extended with world-coordinate position tasks, a modest nominal-posture
task, and a weighted convergence check. The upstream solve/update loop is retained.
World Z has zero pelvis-position cost. Wheel XY fitting is relaxed, while normal
geometry takes priority. Optional support projection subtracts the lowest foot's
height at each frame, preserving relative foot lifts and every horizontal target.
This is an explicit at-least-one-wheel support assumption, **not** detected load,
static-foot locking, or a proof of feasible contact.

| Diagnostic | Original preview | Support-adapted GMR |
|---|---:|---:|
| Root XY RMSE | 0.143 m | 0.003 m |
| Left / right wheel position RMSE | 0.0046 / 0.0035 m | 0.269 / 0.088 m |
| Left / right wheel heading RMSE | 20.3 / 12.9 deg | 25.2 / 20.7 deg |
| Minimum wheel-bottom height, left / right | +0.050 / -0.053 m | -0.0015 / -0.0011 m |
| Maximum wheel-bottom height, left / right | +0.222 / +0.007 m | +0.022 / +0.021 m |

The lower root error comes with substantially larger wheel XY errors; this is a
tradeoff, not an overall fidelity claim. The human feet's median heading separation
is 45.46 degrees, whereas the realized robot wheels have only 0.41 degrees median
and 2.83 degrees maximum separation. The source's independent foot yaw is largely
lost. TRON1's leg chain lacks a yaw steering joint; this reference mostly retains
whole-body turning rather than independent skate steering.

The selected output is [gmr_support_projection_v1](gmr_support_projection_v1/report.json).
[gmr_world_corrected_v3](gmr_world_corrected_v3/report.json) retains the original
vertical estimate without support projection. Earlier local attempts remain
catalogued in [attempt_audit.json](attempt_audit.json), including rejected attempts
where Mink's fused QP path bypassed subclass error/Jacobian overrides. The final
tests check both the analytic Jacobian and the QP actually assembled by Mink.

## Explicitly designed stop command

The separate [command](physx_command_v1/report.json) combines the support-adapted
GMR posture with a designed root path. It preserves the existing car PPO's exact
initial joint offsets, blends posture for 0.3 s, then follows the source posture
timing. Entry speed is 1.307 m/s, matching the existing physical seed's COM speed;
braking runs from 0.4 to 1.8 s, followed by a hold through 2.76 s. The designed root
link stopping distance is 1.438 m. **These values are test settings, not human
measurements.** Root height follows wheel geometry and is not fixed.

[Four-panel command diagnostic](physx_command_v1/command_beta_4x1.png) shows the
path, headings, geometric beta, speed and clearance. Body beta uses directed
velocity course minus forward heading; each wheel additionally uses an axial
[-90,90) plane angle, so reverse rolling at 180 degrees is not labeled large lateral
slip. Angles are masked below 0.25 m/s. Wheel velocities are their own COM velocities,
not contact-material velocities. Body absolute beta peaks at about 85 degrees.

## Prepared physical check

The fixed probe reuses the previously deployed, SHA-locked native PhysX runner,
400 Hz physics, 50 Hz control, rigid cylindrical wheels and mu=0.7. It uses the
existing car PPO as an out-of-distribution stabilizer with zero residual. It is
not a hockey-trained policy or a completed MPPI search. The old actor has no
explicit velocity/braking command; failure of this probe will not establish that
the robot cannot perform the task. The legacy turn cost is not used for hockey
success or controller ranking.

The requested run is one 16-step smoke followed by one 139-step baseline, each
on a freshly verified completely idle RTX GPU and a single environment. No
training, search, external force or state overwrite inside a rollout is allowed.
The six-file payload is fixed in the local manifest:
`outputs/hockey_physx_probe_bundle_20261026_v1/manifest.json`.
Its 109,321 bytes contain two Python files, two JSON files and two derived NPZs;
no licensed models, neural weights, credentials or source RGB are included.
Archive SHA-256:
`73ba20afce682c58fe1b888655202aa13e52b7b87727a6e5cbb17687de2d5b6a`.

**Upload was rejected before execution by automatic approval review.** The stated
reason was that prior approval of other snapshots or the host did not authorize
this exact payload/destination combination. An explicit approval question is
pending. This is separate from the older, also-pending car-speed sweep archive.
Nothing from either rejected archive is fetched indirectly onto the host.

Physical evaluation must combine continuous normal load, actual normal-contact
point lateral velocity, aggregate friction utilization and friction-anchor power
on the same wheel. The prepared gate uses Fz > 5 N, lateral RMS > 0.05 m/s,
aggregate utilization >= 0.9, negative anchor power below -0.05 W, and a continuous
event of at least 0.05 s. The aggregate ratio is
`norm(sum(F_anchor_XY)) / (mu * sum(F_normal_Z))`; PhysX normal points and friction
anchors are different buffers. It is not a pointwise cone-saturation certificate.
Stopping while fallen or unsupported is not successful braking. The nominal task
also requires small XY error and an upright supported final stop.

## Reproduce locally

Use the existing runtimes and model inputs documented in the
[first preview](../hockey_stop_video_20261026_v1/README.md). Run
`audit_hockey_reference_geometry.py` with the frozen local GVHMR run and weights.
Then run `retarget_hockey_gvhmr_gmr.py` with:

```text
--foot-geometry results/hockey_stop_reference_audit_20261026_v1/foot_geometry.npz
--task-profile sole_world_geometry
--nominal-posture-weight 5
--weighted-convergence
--anywheel-support-geometry
```

Use a new output directory and the original `--world`, `--smplx-neutral` and
`--model` arguments. Omit the last flag for the unprojected vertical diagnostic.
`prepare_hockey_physx_reference.py` records hashes of the retarget, existing actor
reference, physical seed and robot XML. `analyze_hockey_reference.py` reproduces
the command plot and bilateral geometric beta. Thirteen relevant tests pass.
Licensed body models, robot assets, RGB video and pretrained networks stay local.

The fixed-relative-camera, four-column source/human/TRON1/XY animation is local:
`outputs/hockey_stop_source_discovery_20261026_v1/render_gmr_support_projection_v1/source_human_gmr_xy_4x1.mp4`.

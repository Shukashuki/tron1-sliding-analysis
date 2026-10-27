# Hockey-stop evaluation and wheel-geometry correction

The next physical probe is prepared, but **no hockey-stop PhysX rollout has run**.
The remote upload requires explicit approval of the corrected v2 payload.
The original v1 request is superseded; neither payload has been uploaded to the
PhysX host or indirectly deployed from Git.

## Contact geometry corrected before the physical probe

The GMR MJCF has 10 mm wide wheels, while the SHA-locked PhysX USD uses 50 mm wide
wheels. Their collider centers also differ by 0.93 mm along the axle. A read-only
USD/FK audit found that v1's apparent ground clearance becomes **up to 6.78 mm of
authored penetration** when evaluated using the PhysX wheel geometry. This is a
reference-geometry difference, not an observed physical penetration or solver error.

The [corrected command](physx_command_v2/report.json) uses the USD cylinder
dimensions and local transforms for support projection. Joint positions, joint
velocities, all body orientations and all XY coordinates remain exactly unchanged.
Only the vertical translation and its derived velocity change; maximum root-Z
correction is 6.78 mm. Height still follows geometry rather than an absolute target.
Neither model file, mass properties nor physical simulator settings were edited.

The independent [corrected geometry audit](corrected_command_geometry/report.json)
finds no authored penetration beyond 1 mm. Its smallest gaps are at floating-point
roundoff, about -2.5e-16 m. This verifies the support calculation only; cooked PhysX
collision shapes, normal loads and continuous contact still require the real run.

## Independent qualification, separate from inline diagnostics

The earlier inline screen could accept a stationary episode without checking
entry speed, and could shorten the required terminal hold to a truncated clip.
It is now explicitly diagnostic-only and cannot claim task success. The new local
[evaluator](../../scripts/evaluate_hockey_physx.py) requires:

- Full reference coverage, entry speed >= 0.8 m/s, and approach speed ratio 0.95–1.10.
- XY RMSE <= 0.15 m and peak error <= 0.30 m.
- At least one wheel continuously loaded after the initial 20 ms; lifting the other is allowed.
- A complete final 0.50 s with speed <= 0.10 m/s, wheel support and tilt <= 30 degrees.
- Braking body-heading RMSE <= 15 degrees and loaded wheel-plane heading RMSE <= 20 degrees.
- A same-wheel event lasting >= 0.05 s during the declared braking phase: normal load > 5 N,
  normal-point lateral RMS > 0.05 m/s, aggregate utilization within [0.9,1.05], anchor power < -0.05 W,
  a dissipative laterally slipping friction anchor, and axial wheel beta >= 20 degrees
  at speed >= 0.25 m/s. Tilt must remain below 45 degrees for this task screen.

The utilization upper bound is a prospective 5% tolerance for the aggregate
Coulomb screen. Larger ratios are flagged and do not count as near-edge evidence;
they are not automatically declared SDK constraint violations.
These additional heading/tilt limits are prospective protocol choices, not measured
hardware limits. No absolute height threshold is used. Reversal at 180 degrees is
zero axial wheel beta; beta is undefined below its speed threshold. Missing samples
cannot be joined into a support/sliding event. Stop holds use elapsed state time;
contact-duration counts use completed physical-step intervals. Early stopping,
later restarting, falling, wrong headings and cross-wheel combinations fail the
relevant gates. The command is motion-inspired with designed metric speed, not a
calibrated human stopping trajectory.

## Native-buffer verification

The evaluator independently reconstructs each wheel's forces, contact-material
velocity, friction-anchor power and aggregate utilization from archived native
buffers. Force and velocity are paired at the same friction anchor spatially.
They describe completed-substep force and post-step velocity; their product is a
power proxy, not exact substep work or a total-energy certificate.

Two existing car trajectories tested the measurement implementation; they are
**instrumentation preflights, not hockey-stop trials**:

| Archived case | Physical steps checked | Max normal-force difference | Max lateral-slip difference |
|---|---:|---:|---:|
| [Nominal car reference](historical_nominal/report.json) | 2,792 | 1.53e-5 N | 1.25e-7 m/s |
| [Fast car failure](historical_fast_failure/report.json) | 1,864 | 1.53e-5 N | 4.36e-6 m/s |

Each case includes a horizontal four-panel plot. Loaded bilateral beta is reported
separately before the first 45-degree tilt diagnostic, so large post-fall angles do
not become evidence of successful sliding.

PhysX's patch model applies friction at selected anchors, and TGS and PGS handle
the tangent constraints differently. See the [PhysX 5.4 friction documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/struct_px_friction_type.html).
This source explains the distinction; it does not identify the deployed runtime's
exact solver build. The exposed normal-point and friction-anchor buffers do not
provide a one-to-one normal-load assignment. Consequently the evaluator reports
**wheel-level corroboration**, not pointwise friction-cone saturation. Low aggregate
force can also hide cancellation between anchors; both the resultant-force ratio
and the sum-of-anchor-magnitudes ratio are retained.

## Reproduce and run status

Thirty-one relevant tests pass, including 18 new evaluator cases. The wheel-geometry
audit uses local MuJoCo and USD Python bindings on CPU. No new GPU process,
physics rollout, training or MPPI search ran in this update.

```bash
python scripts/evaluate_hockey_physx.py \
  --run-dir /path/to/completed/hockey/baseline \
  --command-dir results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2 \
  --output /path/to/new/qualification
```

Use `--instrumentation-only` instead of `--command-dir` for the two historical
car cases. The evaluator validates the frozen engine, wrapper, command and artifact
hashes for a hockey qualification. It runs locally after retrieval and need not
change the remote payload.

The **v2** upload contains six payload files (109,866 bytes), plus its manifest;
no models, weights, RGB, binaries or credentials. Archive SHA-256:
`fc87c10983c54ca74900fab66e86d498eb0e66def04e7b6a4cfffa5b12d3642c`.
Local manifest: `outputs/hockey_physx_probe_bundle_20261026_v2/manifest.json`.
The pending request covers only one 16-step smoke and one 139-step baseline on a
freshly idle RTX, one environment at a time. Automatic review previously rejected
the SSH upload because older approvals did not cover this payload/destination;
the replacement approval question explicitly names v2.

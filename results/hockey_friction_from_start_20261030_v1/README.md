# Continuous friction evolution from reset

[Play the continuous force/cone animation](analysis/force_evolution_from_start_8x_4x1.mp4).
The four horizontal panels show growing left/right force histories followed by
the left/right wheel-resultant friction sections. Each frame updates normal
force, tangential force, static/dynamic capacity, contact and slip readouts.
The full **0–2.76 s** recording is shown at **8× slow motion**, retaining all
**1,104 physical intervals at 400 Hz**. There is no force smoothing, pose
interpolation, retraining or new physics rollout.

[Animated preview](preview/force_evolution.gif) ·
[native contact points / friction-anchor animation, also starting at zero](animation/case0_contact_friction_8x_4x1.mp4) ·
[full force curves](analysis/force_history_4x1.png) ·
[first 100 ms](analysis/startup_0p1s_4x1.png) ·
[friction sections](analysis/cone_sections_from_start_4x1.png).

The primary video has growing histories with no future force curves. Its cone
vector is the current resultant, with a faint trail of up to 40 samples and a
star marking the first interval. The companion contact-point animation retains
the old display convention of a faint whole-trajectory background. Both videos
show every physical interval; the lightweight GIF preview samples every fourth
video frame and is not the full-resolution analysis.

## What becomes visible at the beginning

The selected v9 nominal policy is the same successful **static/dynamic 0.9/0.7**
recording used in the aligned human-to-PPO display. The old contact animation
started at 0.4 s; the underlying archived analysis already contained time zero.
This update includes that beginning instead of treating braking onset as the
start of the experiment.

In the first **[0, 0.0025] s** interval, left/right normal forces are
**130.02 / 102.21 N**, and tangential resultant magnitudes are **91.01 / 71.55 N**.
Both are approximately `mu_d * Fn` (`rho_d ≈ 1`). The initial body XY speed is
**1.303 m/s**, while both wheel joint velocities are **0 rad/s**. Initial
tangential point speeds are **1.293 / 1.297 m/s**, but lateral speeds are only
**0.0166 / 0.0065 m/s**. This is predominantly longitudinal startup sliding,
consistent with the nonzero body velocity and zero initial wheel spin; it is
distinct from the later lateral sliding during the turn. Most of this first
slip transient subsides as wheel spin rises over roughly the first 10–20 ms.

| Phase | Time [s] | Mean normal force L/R [N] | Mean tangential magnitude L/R [N] |
| --- | --- | --- | --- |
| Approach, including startup | 0–0.4 | 109.57 / 109.73 | 44.38 / 35.76 |
| Braking | 0.4–1.8 | 93.31 / 125.02 | 57.59 / 64.59 |
| Stop / hold | 1.8–2.76 | 105.91 / 112.64 | 38.27 / 37.90 |

These are means of individual wheel-force magnitudes, not net braking force.
Normal-force redistribution changes the available capacity even though both
material coefficients stay fixed throughout this recording. Local friction and
slip persist in parts of the hold phase; small body speed does not imply zero
contact-point speed. Both wheels remain touching at every checked sample.

## Definitions and timing

`Fn` is the summed normal load for a wheel on the verified horizontal plane;
`|Ft|` is the norm of its summed tangential friction-anchor force. The plotted
capacities are `0.9 Fn` and `0.7 Fn`. Cone-section coordinates are
`F_long / (0.9 Fn)` and `F_lat / (0.9 Fn)`, so the static circle has radius 1 and
the dynamic circle radius `0.7 / 0.9`. These are **wheel-resultant diagnostics**:
the archive does not establish the normal-load assignment to each friction
anchor, and no individual contact-point cone is claimed.

Geometry and pre-step velocities are sampled at `t`; force is impulse divided
by `dt` over **[t, t + 2.5 ms]**. The first native interval is paired with trace
force slot 1. The unstepped reset state's zero-filled force slot 0 is not an
observed zero force and is never plotted. The last force interval is
[2.7575, 2.7600] s; there is no invented force measurement beyond the task.

`TOUCH` in the animation requires contact at both recorded endpoints. Sliding
requires both pre-step and interval tangential RMS speeds above 0.05 m/s;
`LOW SLIP` means both below 0.02 m/s. Other combinations are labeled mixed or
transient. These labels describe kinematics, **not an observed PhysX
static/dynamic mode flag**. Reaching the dynamic circle therefore does not, by
itself, prove a newly observed static-to-dynamic transition. Full task success
criteria remain unchanged.

## Reproduction and verification

```bash
python scripts/plot_hockey_friction_from_start.py \
  --analysis results/hockey_static_dynamic_transition_20261029_v1/trained_contact_points_v1 \
  --nominal results/hockey_static_dynamic_transition_20261029_v1/continuation_evidence/run_v9/native_split_smooth_stop_xyz_huber_continuity/nominal \
  --output <fresh-analysis-directory>
python scripts/render_hockey_friction_contacts.py \
  --analysis results/hockey_static_dynamic_transition_20261029_v1/trained_contact_points_v1 \
  --start-time 0 --output <fresh-contact-animation-directory>
python -m pytest -q tests/test_hockey_contact_cone_relation.py \
  tests/test_hockey_contact_timing.py tests/test_hockey_braking_contacts.py
```

The previous renderer's default remains 0.4 s for historical reproduction.
Both outputs are decoded completely to verify frame count, dimensions and
50 fps encoding. Ten existing timing/contact checks pass. The data and actor
identities are hash checked against the existing qualification; normal and
tangential forces are also compared with the archived trace at the correct
one-step offset. No source RGB, policy weights, body models or robot meshes
are included here. See the [interval CSV](analysis/force_intervals.csv),
[analysis provenance](analysis/report.json),
[contact-animation provenance](animation/provenance.json), and
[GIF generation record](preview/provenance.json).

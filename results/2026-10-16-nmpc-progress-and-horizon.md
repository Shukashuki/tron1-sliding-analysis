# Nominal NMPC: progress, model audit and finite-scenario tests

This follow-up uses **only the primary turn at μ=0.70**. No NMPC friction
grid, controller freeze, new training or PPO initialization is reported.
The reference, 15-term height-free objective and actuator mapping stay fixed.

## The previous Isaac error is not all lateral sliding

At the same physical timestamp, project root-origin position error onto the
reference root-origin tangent and its left normal. Negative along-track error
means behind the reference. There is no nearest-path fit, time warp, spatial
registration or COM-course substitution. Derivatives use demonstrated
reference samples only, excluding synthetic recovery/tail.

| Core RMS error (m) | Total XY | Along tangent | Along normal |
| --- | ---: | ---: | ---: |
| Absolute H15, Isaac | 0.3244 | 0.1569 | 0.2839 |
| Residual H15, Isaac | 0.2590 | 0.2208 | 0.1354 |
| Absolute H15, MuJoCo | 0.2803 | 0.2198 | 0.1739 |
| Residual H15, MuJoCo | 0.1154 | 0.0878 | 0.0749 |

Residual Isaac's tangent component is behind throughout the core and accounts
for **72.65% of squared XY error**. Normal error improves, but same-time
progress remains inadequate. These components do not measure contact slip or
establish a constant controller delay. Root-origin speed diagnostics also
differ from the previously reported base-COM speed; do not mix those points.
The original total-XY screen remains unchanged.

## Bounded observed-action model audit (residual H15 only)

Sixteen intervals were selected from fixed time strata and measured contact
changes **before prediction**, then predicted independently for one 20 ms
action interval each: 64 physical substeps, no CEM or new closed loop.
Inputs are the exact recorded 111-D controller states, actual float32 actions,
runtime joint limits and original nonzero contact history. Local and original
remote workers match C++ source, MuJoCo headers/library and compiled model
snapshot; their host-specific native binary hashes remain distinct.

| Maximum prediction difference | Value | Scope |
| --- | ---: | --- |
| Root XY distance | 0.780 mm | All 64 substeps; maximum occurs at 20 ms |
| Root COM-velocity component | 0.0296 m/s | All substeps/components |
| Root angular-velocity component | 0.703 rad/s | All substeps/components |
| Joint torque, first substep | 0.0000155 Nm | Same measured initial state |
| Joint torque, all substeps | 21.28 Nm | Subsequent predicted states evolve independently |

No sizeable action-mapping error was found in this sample. Subsequent contact,
angular velocity and torque differences remain, so small short-interval XY
error does **not** certify long-horizon/counterfactual accuracy or explain the
whole tracking discrepancy. Contact-force channels have different engine
semantics. Nonroot body channels reconstructed by FK are not extra measured
Isaac snapshots; the logged actual objective remains authoritative.
The [sanitized diagnostics](nmpc_nominal_diagnostics_20261016_v1/summary.json)
record exact input/source hashes, sample selection and component definitions.
This is not an audit of H30 states or a causal explanation of its results.

## Longer look-ahead development

Residual H30/B5 predicts 0.60 s instead of 0.30 s, retaining five blocks.
Block duration therefore also changes from 0.06 to 0.12 s: this is not a
horizon-only causal experiment. Population 64, three CEM iterations, eight
elites, seed 42, 20 ms control and nominal prediction μ=0.70 remain unchanged.
Both plants independently replan using MuJoCo prediction, not action replay.

![Residual H30: reference, Isaac and MuJoCo](controller_primary_nmpc_residual_h30_b5_nominal_pair_v1/topdown.png)

The [strict H30 pair](controller_primary_nmpc_residual_h30_b5_nominal_pair_v1/summary.json)
completes 349 decisions / 1,397 physical samples in each engine:

| H30 core diagnostic | Isaac | MuJoCo |
| --- | ---: | ---: |
| XY RMSE / peak | 0.2318 / 0.3350 m | 0.0916 / 0.1252 m |
| Mean COM-speed retention | 95.84% | 101.16% |
| Net COM-course turn | 60.38° | 36.48° |
| Maximum full-episode tilt | 15.81° | 18.44° |
| Median replanning time | 659 ms | 969 ms |

Neither engine records nonwheel-ground contact or solver failure. Isaac improves
on H15 but still exceeds both XY limits, so H30 is **not frozen or qualified
for a friction grid**. Net-course values are endpoint velocity diagnostics,
not path-turn equivalence; reference COM-course turn is 59.255°.
The simulator waits for solves: these are not 20 ms real-time results or
controlled cross-host timing comparisons.

## Three fixed prediction models did not fix nominal tracking

The [finite-scenario pair](controller_primary_nmpc_residual_scenario3_nominal_pair_v1/summary.json)
returns to residual H15/B5, with the previously declared prediction coefficients
`[0.3, 0.7, 1.6]`. Each candidate uses one shared action sequence; CEM minimizes
the maximum **summed horizon cost** over those constant-coefficient models.
Both actual plants remain fixed at μ=0.70. This is not a validation-friction
sweep, a switching-uncertainty controller or a formal robustness guarantee.

![Three-model NMPC: reference, Isaac and MuJoCo](controller_primary_nmpc_residual_scenario3_nominal_pair_v1/topdown.png)

| Three-model core diagnostic | Isaac | MuJoCo |
| --- | ---: | ---: |
| XY RMSE / peak | 0.2745 / 0.3760 m | 0.1282 / 0.2008 m |
| Mean COM-speed retention | 109.04% | 96.56% |
| Maximum full-episode tilt | 13.68° | 98.51° |

All 349 decisions and 1,397 physical samples are complete, with zero solver
failures. Isaac is worse than both single-model residual variants. MuJoCo's
core XY screen passes despite its large tilt and first nonwheel-ground contact
at 3.180 s; this is not stable controlled execution. Posture remains a separate
diagnostic, not a replacement height/flight failure gate. All 68,054 candidate
evaluations are finite, with 204,162 scenario evaluations and 6,279 batch calls
per episode: three-model prediction also spends more computation.

Following the explicit experiment decision, **nominal XY must pass before any
controller freeze or friction sweep**. These completed failures are preserved;
the next improvement is not selected using held-out validation coefficients.

Height, flight and nonzero heading offsets are not failure gates. Mean speed,
coverage, a small one-step prediction error or an XY pass alone do not prove
a controlled high-speed turn, continuous-uncertainty robustness or hardware
readiness. Current reference speed remains approximately 1 m/s.

## Model-only initialization and narrower leg sampling did not improve tracking

The [local search-development summary](nmpc_nominal_search_development_20261016_v1/summary.json)
preserves five MuJoCo variants, including the earlier unseeded H15 baseline.
All five complete 349 decisions / 1,397 physical samples at plant and prediction
μ=0.70, seed 42, with zero solver failures. These are nominal development
results, not a frozen benchmark or a validation-friction grid.

![Local initializer and sampling ablations](nmpc_nominal_search_development_20261016_v1/topdown.png)

| MuJoCo variant | Core XY RMSE / peak (m) | Mean COM-speed retention | Maximum full-clip tilt |
| --- | ---: | ---: | ---: |
| Unseeded residual H30/B5 | 0.091639 / 0.125206 | 101.16% | 18.44° |
| Model-only initializer, H15/B5 | 0.203165 / 0.297071 | 94.69% | 10.43° |
| Model-only initializer, H30/B5 | 0.153742 / 0.192481 | 98.58% | 94.31° |
| H30/B5, lower initial leg standard deviation | 0.205810 / 0.396978 | 95.54% | 13.39° |

The initializer contains only 349×8 applied actions generated by the earlier
MuJoCo residual H30/B5 controller: no PPO, learned actor or human demonstration
is used as its donor. It is an **additional** initial candidate, alongside
previous-action, shifted-plan and bias-only candidates, rescored from the
current measured state under every declared prediction model. CEM can reject
or correct it; it does not replay actions or prescribe robot state. The feature
is default OFF, with exact default action/plan/cost/RNG golden regression tests.
Donor generation already spent 8,166,480 prediction substeps plus 1,396 donor
plant substeps; subsequent online CEM and antecedent development are additional.

The narrower-search variant has no initializer and changes only the six initial
leg sampling standard deviations from 0.04 to 0.02. Minimum standard deviations
and both wheel sampling widths remain unchanged. Neither search change improves
on unseeded H30. The H30 initializer variant first records nonwheel-ground
contact at 4.240 s, outside the core, and reaches 94.31° full-clip tilt; the other
four variants record no nonwheel-ground contact. Posture/contact measurements
remain diagnostics, not new height or flight failure gates.

Neither initializer nor lower-leg-width variant has been evaluated in Isaac;
the prepared initializer snapshot has not been uploaded. No controller is
frozen and no NMPC friction grid has started: **both engines must first achieve
core XY RMSE ≤0.15 m and peak ≤0.30 m**. Separate local and Isaac H30/B10 checks
are in progress; no outcome from them is claimed here.

## Reproduction

The vector-only diagnostic needs the existing public compact pairs, not models:

```sh
python scripts/analyze_xy_error_components.py \
  --baseline-dir results/controller_primary_nmpc_nominal_pair_v1 \
  --residual-dir results/controller_primary_nmpc_residual_nominal_pair_v1 \
  --output-dir outputs/NEW_nominal_xy_components
```

The bounded [prediction probe](../scripts/probe_nmpc_isaac_prediction.py) requires
the hash-locked original Isaac episode, licensed model/motion, matching helper
snapshot, contract and an explicitly built native library; its CLI documents
those inputs. It is bound to this particular H15 episode, not a generic model
certificate. Use historical checkout `3faf08d` for this probe: the later optional
initializer changes runtime CEM/runner hashes, and the probe deliberately rejects
those different source bytes. Reproduce the H30 episode using the shared
[residual runner command](2026-10-16-nmpc-nominal-development.md#audited-artifacts-and-offline-reproduction)
plus `--horizon-steps 30 --control-blocks 5` and a fresh output directory.

For the optional initializer, inspect
`python scripts/export_model_nominal_plan.py --help` and first export a local,
hash-locked action artifact from the completed unseeded residual H30 donor.
Add `--model-nominal-plan /path/to/NEW_artifact/plan.json` to that residual H30
command (or set `--horizon-steps 15` for the initializer H15 case). The raw donor
and action NPZ are not published; the public search-development summary retains
their portable metadata, source hashes and offline generation budget.
For the narrower-leg H30 case, use
`--settings config/nmpc_height_smooth_residual_h30_low_leg_std.json` in the
same base command, retain `--horizon-steps 30 --control-blocks 5`, and leave
`--model-nominal-plan` unset. These commands generate new physics episodes;
they are not asset-free replays of the public summary.

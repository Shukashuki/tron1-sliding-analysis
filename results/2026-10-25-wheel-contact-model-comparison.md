# Wheel contact models for the high-speed turning task

The repository already contains a nine-slice rigid-wheel prototype and an external nonlinear tire-force prototype. Their completed nominal tests are recorded below. Neither currently satisfies the complete stable-contact turning task. Introducing these models is a contact-model comparison, not evidence that either is a validated real tire or a successful controller.

## Separate geometry from the force law

| Model | Geometry | Normal contact | Tangential force | Intended question |
|---|---|---|---|---|
| Original rigid wheel | One authored cylinder per wheel | Native PhysX | Native material friction | Baseline rigid contact |
| Nine-slice rigid wheel | Nine adjacent cylinders along the same axle | Native PhysX | Same native material law | Does the collision representation change contact continuity and closed-loop behavior? |
| Nonlinear tire prototype | Original authored cylinder | Native PhysX | External combined-slip force; native wheel friction set to zero | Does a slip-dependent tangential law change the task? |
| Nine slices plus nonlinear force | Not executed | Not yet validated | Not yet validated | Future interaction between geometry and force law |

Nine slices retain the original 0.127 m radius and total 0.05 m width, original wheel mass/inertia and joints. Each slice is rigid and attached to the same wheel body. This is axial subdivision, not a polygonal approximation around the circumference. It introduces additional collision shapes and possible contact seams; it does not introduce elastic tread, independent slice suspension, or an identified contact-pressure distribution. More shapes do not guarantee continuous contact or more accurate physical behavior.

The nonlinear prototype retains native normal collision response and replaces wheel-ground tangential friction with a custom force. It is not the PhysX Vehicle extension. The official [PhysX vehicle documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Vehicles.html) separately describes tire slip, force generation and low-speed handling; those features are not automatically present in a rigid cylinder or this custom prototype.

## Implemented nonlinear law

The implementation computes contact-point velocity from rigid-body COM velocity, angular velocity and measured contact offset. It projects this into wheel longitudinal/lateral directions. The normalized slip vector uses the larger magnitude of hub longitudinal speed, circumferential speed, and a 0.2 m/s regularization speed as its denominator.

For slip magnitude `s`, the implemented force utilization is:

```text
u(s) = sin(C atan(B s - E (B s - atan(B s))))
B = 4.5, C = 1.8, E = 1.0
F_t = -mu F_z u(s) * normalized_slip_direction
tau_COM = contact_offset cross F_t
```

This is a simplified isotropic combined-slip Magic-Formula-style sensitivity model. Its parameters have not been identified from a real TRON1 wheel. It does not provide separate fitted longitudinal/lateral curves, elastic tire deformation, relaxation-length dynamics, explicit static adhesion, or experimentally identified load/camber sensitivity. The force uses the preceding physics sample's normal load and contact offset: a declared 2.5 ms measurement delay, not an implicit coupled tire/contact solve. Reset starts with zero external tire force. Native tangential friction is disabled to avoid counting friction twice. These assumptions matter when interpreting low-speed behavior and support transitions.

## Completed nominal evidence

All rows below use `traj_4`, intended nonzero beta, 400 Hz physics, 50 Hz policy actions, action-rate weight −0.5 and validation mu 0.7. The original, sliced and nonlinear policies were each trained from scratch for 12,288,000 transitions with the K contact-reward recipe and the retained friction/encoder/COM/push DR. This is a single-seed comparison. Rows explicitly distinguish training from evaluation conditions.

| Training model → evaluation model | Evaluation engine | Core XY RMSE (m) | Core beta RMSE (deg) | Any-wheel support (%) | Longest support gap (ms) |
|---|---|---:|---:|---:|---:|
| Original → original | PhysX | 0.0816 | 3.77 | 89.52 | 45.0 |
| Original → nine slices | PhysX | 0.0878 | 4.22 | 89.12 | 55.0 |
| Nine slices → nine slices | PhysX | 0.1296 | 4.61 | 88.19 | 50.0 |
| Nine slices → original | PhysX | 0.0793 | 2.03 | 89.77 | 45.0 |
| Nonlinear → nonlinear | PhysX | 0.1872 | 20.65 | 93.27 | 50.0 |
| Nonlinear → original | PhysX | 0.0444 | 3.88 | 90.93 | 60.0 |
| Nonlinear → native original | CPU MuJoCo | 0.1778 | 18.81 | 84.95 | 62.5 |

Core tracking is measured over 1.44–4.00 s. Support is measured over 0.02–6.96 s at every actual 2.5 ms physics sample: at least one wheel must have ground-normal force strictly above 5 N. The other wheel may lift. A support gap is an absence of measured wheel load, not by itself a geometric flight measurement. No absolute root-height constraint is added.

Every row fails at least the retained continuous-support criterion. The nonlinear same-model episode also fails the core XY and beta thresholds. Its higher aggregate support percentage does not establish continuous contact. The nonlinear-trained actor's better native-PhysX tracking is a transfer observation; it does not prove the nonlinear training law is more physically accurate or superior across engines. Native CPU MuJoCo in the final row uses its original contact law, so that row changes both engine and tangential law.

- [Nine-slice report and four-cell animation](ppo_k_native_slices9_20261025_v1/report.html), [native metrics](ppo_k_native_slices9_20261025_v1/four_arm_analysis.json).
- [Nonlinear training, transfer and full-body animation](ppo_k_nonlinear_tire_20261025_v2/completed_report.html), [native metrics](ppo_k_nonlinear_tire_20261025_v2/analysis/analysis.json).
- [Separate K/M warm-refinement negative result](contact_warm_refinement_20261025_v1/report.html). This uses additional training budget and is not included as a matched-budget wheel-model arm.

## Subsequent controlled comparisons

Keep geometry and tangential law as independent experimental axes. First retain the existing original/native, sliced/native and original/nonlinear cells. A combined sliced/nonlinear model is a separate future cell and has not been executed. Before running it, define how measured contact loads are aggregated into tire force, how forces/torques are distributed across slices, and how shared friction capacity avoids multiplying `mu F_z` by the number of slices. Preserve mass/inertia and actuator limits. Distinguish the force model's measurement delay from an intentionally modeled physical relaxation time.

Use two distinct protocols: frozen-policy transfer to expose sensitivity to the changed plant, and separately trained policies with matched budgets to measure adaptation. Preserve one actor per variant across evaluation engines. Closed-loop results include the policy's response to changing observations and do not isolate contact-solver error. Short, identical-initial-state/action recordings are needed for that separate diagnosis.

Record reference and actual XY path, speed retention, body heading, each wheel heading, intended beta error, wheel-local slip, normal/tangential force, friction utilization, support interruptions, contact switches and body chatter. For native contact, saturation is an engine-specific friction diagnostic. For the external law, `|F_t|/(mu F_z)` is the implemented force utilization. Neither a nonzero body beta alone nor utilization alone establishes sustained loaded tire sliding: inspect wheel-local tangential motion while loaded as well.

Retain the existing complete nominal gates: core/whole XY RMSE ≤0.15 m and peak ≤0.30 m; core speed ratio 0.95–1.10; intended beta RMSE ≤10 degrees; angular-XY 10–100 Hz RMS ≤0.7 of the original same-engine PPO; and continuous any-wheel support. Inspect full-body integrity and the within-turn speed profile. Keep the existing fixed-relative-distance camera, opaque full-body replay and common-scale overhead paths. Nominal mu 0.7 must qualify before any new friction sweep. No combined-model job or new friction sweep was launched for this consolidation.

This evidence covers the two-wheel TRON1 task. It does not supply a calibrated four-wheel car comparison or establish the value of human motion retargeting/AMP; those remain separate experiments.

## Reproduction entry points

- Geometry: [authoring and slice placement](../scripts/tron1_wheel_geometry.py), [all-environment native geometry audit](../scripts/ppo_native_slices9_model.py), [slice trainer](../scripts/train_ppo_k_native_slices9.py), [protocol](../config/ppo_k_native_slices9_400hz_v1.json).
- Force law: [NumPy reference](../scripts/nonlinear_tire_contact.py), [batched GPU kernel](../scripts/nonlinear_tire_torch.py), [trainer](../scripts/train_ppo_k_nonlinear_tire.py), [protocol and limitations](../config/ppo_k_nonlinear_tire_400hz_v1.json).
- Exact existing invocation arguments, dependency/asset locations, source hashes and native startup audits are retained in each linked report's evidence collection. The public records exclude model assets, neural weights and credentials.

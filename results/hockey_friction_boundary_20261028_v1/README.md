# PPO for supported sliding near the friction boundary

Status: **uploaded, hashes verified, and queued after the existing 18-cell study**. The user explicitly approved this exact payload, destination and deferred two-arm training/validation, resolving the initial automatic-review rejection. The new tmux session `hockey_friction_boundary_20261028_v1` is alive with status `waiting_for_parent`; no new GPU stage has started. The parent study has completed six cells and is training the original recipe at 1× speed / seed 43. [Upload receipt](preflight/upload_receipt.json) · [Runtime preflight](preflight/remote_preflight.json) · [Launch identity](preflight/launch.json) · [Verified queue status](preflight/verified_queue_status.json). This candidate has no new learning result yet.

The objective is to let PPO coordinate wheel camber, leg posture, wheel torque and load transfer while producing a supported, dissipative sliding phase. The friction-cone boundary is a condition in force space, not a geometric path that the robot follows. At each wheel we measure

\[
\rho_i=\frac{\|\sum_a F_{t,ia}\|}{\mu_i\sum_j F_{n,ij}}.
\]

## Control objective: allocate friction capacity

The intended mechanism is coordinated friction-capacity allocation through independent wheel actuation, wheel camber/contact geometry, and whole-body load transfer. The available actions are six leg joint-position targets and two independent wheel-torque commands. Wheel speed and normal load are resulting states, not directly prescribed independent controls. The robot has no independent wheel-yaw steering joint; achievable wheel directions follow from its leg and body kinematics.

For a flat, isotropic Coulomb approximation, write `C_i = mu_i F_ni` and `F_long,i² + F_lat,i² <= C_i²`. Three coupled effects must be distinguished:

| Mechanism | Physical role | Evidence to report separately for each wheel |
| --- | --- | --- |
| Wheel torque and resulting speed | Change longitudinal contact slip and force; longitudinal and lateral demands share the same tangential capacity | Joint speed/torque, loaded contact-point longitudinal and lateral slip, `F_long / C` and `F_lat / C` |
| Wheel camber and body/leg posture | Change contact locations, effective moment arms and material-point motion | Camber, projected wheel direction, native contact/anchor positions and resulting force direction |
| COM motion and leg support | Redistribute normal load and available capacity between wheels, while satisfying whole-body force and moment balance | `F_nL`, `F_nR`, their load fractions, `C_L`, `C_R`, COM motion and angular-momentum response |

With equal coefficients, only wheel contacts, negligible vertical COM acceleration and no other vertical external forces, `C_L + C_R ≈ mu mg`: shifting load redistributes capacity rather than creating additional total grip. In dynamic motion, normal loads also depend on vertical acceleration, angular-momentum change and contact geometry; COM position alone does not determine them.

The current rigid-cylinder Coulomb model does not contain a nonlinear tire camber-thrust law. Its ground-normal friction cone does not simply rotate with wheel camber, nor does camber automatically increase the material coefficient. Any measured camber benefit must therefore be attributed to the modeled contact geometry and dynamics. A tire model with camber stiffness would be a separate modeling experiment; see the [PhysX Vehicle tire-force documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Vehicles.html#pxvehicletireforceparams).

The frozen boundary bonus takes the better-scoring wheel at each substep. It permits learned coordination but does **not** explicitly optimize a left/right force allocation or establish that such allocation is optimal. Existing tracking and stopping rewards provide the task objective. Evidence of effective allocation must additionally show the required braking/turning response, stable support and a low-slip terminal phase; a high friction-utilization ratio by itself is insufficient. Interventions restricting wheel actuation, camber or load transfer would be needed to test each mechanism's causal contribution, with any loss of task feasibility reported.

This clarification changes the interpretation and requested analysis, not the approved immutable payload, reward weights, observation inputs, success gates or training queue. Tests with unequal static/dynamic coefficients remain a separate extension: use the corresponding coefficient for the relevant force-limit diagnostic and distinguish observed slip-state changes from an internal solver mode flag.

Only the new reward weight differs between the two arms: 0 versus +0.5. Both use identical native contact instrumentation, 1.5× reference speed, seed 42, 400 Hz physics, 50 Hz actions and fresh 12,288,000-transition PPO training. All existing soft imitation, velocity, stopping and action-smoothness rewards remain unchanged, as do the 56/125 actor/critic inputs and friction DR [0.3, 1.2]. No absolute height constraint, wheel-XY target, bilateral-support requirement or additional support-deficit penalty is introduced. The actor does not receive privileged contact forces or the sampled friction coefficient; those are used by the training reward.

## Bounded boundary-sliding reward

For each completed physical substep and wheel, the candidate multiplies:

- A load gate: normal force >5 N, with a bounded load factor reaching one at 0.25 mg. Losing load cannot preserve a full bonus merely by keeping the force ratio near one.
- A boundary factor `exp(-((rho - 1) / 0.1)^2)`, screened out above rho=1.05.
- A lateral material-point slip ramp: zero through 0.05 m/s, reaching one at 0.15 m/s. Wheel COM speed alone cannot satisfy it.
- A bounded dissipative-power factor, and at least one friction anchor with lateral slip, opposing lateral force and negative force–velocity product at that same anchor.
- A bounded braking factor requiring the wheel's net tangential friction force to oppose the actual base COM velocity.

The better-scoring wheel is used at each substep. XY/velocity tracking and uprightness gate the score; it vanishes at 45° tilt. The bonus is available only during the reference braking phase, 0.4–1.8 s. **Every one of the eight physical substeps must have at least one wheel above the load threshold** for that control interval to earn any bonus. Single-wheel support and handoffs are allowed. The resulting unweighted reward lies in [0, 1]; increasing slip speed or dissipative power without bound gives no further benefit. Isaac applies its usual control-dt scaling to the weighted term.

This is a reward, not a hard support constraint. It does not guarantee a successful stop or exclude failures outside braking. The unchanged independent evaluator still checks the whole episode, including continuous support, stop/hold, XY tracking and at least 50 ms of qualifying high-beta sliding. Unsuccessful episodes stay in the denominator.

The actual per-world randomized coefficient is used during training, rather than assuming evaluation mu=0.7. Static and dynamic coefficients remain equal in this first ablation. The design does not identify an internal static-to-dynamic solver transition or impose a coefficient drop.

## Measurement limits and checks

PhysX computes friction at contact patches and anchors. The available buffers do not provide a normal-load allocation to each friction anchor, so rho is a **wheel-aggregate proxy**, not pointwise cone certification. Contact-point velocities use `v_COM + omega × (point − COM)` relative to the stationary flat ground. Completed-substep forces paired with post-step velocities provide a dissipative-power proxy, not exact integrated work. See the [PhysX friction documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#friction-and-restitution).

[Local validation](preflight/local_validation.json) covers seven new tests and 21 existing hockey regression tests. Counterexamples include saturated static contact, pure rolling, absent lateral opposition, load collapse, a single missing support substep, positive contact power, stopping, falling and off-path motion. A collector lifecycle test checks per-world randomized mu, eight physical samples per action and reset-refresh exclusion.

[Offline replay](offline_audit_v2/report.json) reconstructs 8,896 wheel/substep samples from four historical original-speed PhysX episodes. The sampler agrees with the independent native-buffer reconstruction within the existing force/velocity tolerances. Mean candidate bonuses are 0.188 / 0.139 for the earlier recipe's nominal / trial 022 recordings and 0.210 / 0.228 for hockey PPO. These are objective sanity checks, not new learning results or replacement success scores: an eventual failure can still have a valid earlier sliding phase. [Replayed reward curves](offline_audit_v2/replayed_bonus.npz).

## Prepared execution

The immutable [21-file manifest](source_snapshot_v1/manifest.json) covers 314,831 bytes of code, configuration and the already-used 1.5× derived robot reference. It excludes models, policy weights, credentials, video and SMPL assets. Archive SHA-256: `51af4d5764fc81174879bfa7be02a0a0e166cbbfb6cf556b32a8a1aa90725ac1`.

Intended destination: `zengxiong@14.103.233.29:/data/zengxiong/workspaces/tron1-hockey-friction-boundary-20261028-v1/source_snapshot_v1/`.

The running dependency wrapper waits until all 18 cells of the existing study complete. It then runs the two arms serially, each with a 32-environment / 2-iteration smoke check, a 512-environment / 1,000-iteration training, and 33 single-environment mu=0.7 validation episodes. Total formal budget: 24,576,000 transitions; validation: 66 episodes. It selects only a freshly confirmed completely idle RTX, never changes another process, and stops on infrastructure failure without retry. Waiting for the parent is bounded at 24 hours; subsequent idle-GPU waiting is bounded at six hours. Deployment is at the intended destination above; the current wait consumes no GPU allocation.

The [protocols and plan](source_snapshot_v1/plan.json) retain the frozen physical evaluator byte-for-byte. This experiment tests one reward weight at one speed and training seed; any successful result would still need replication.

Recreate a fresh local payload from the repository:

```bash
python scripts/prepare_hockey_boundary_study.py \
  --output outputs/hockey_boundary_reproduction/source_snapshot_v1 \
  --archive outputs/hockey_boundary_reproduction.tar.gz
```

The remote entry point is `defer_hockey_boundary_study.py`, with explicit `--parent-status`, `--bundle`, `--prior-launch`, `--audit-helper`, `--output` and `--status` paths. The parent manifest is pinned to the existing 18-cell study, and both protocols are checked to differ only in the new reward weight before any GPU stage starts.

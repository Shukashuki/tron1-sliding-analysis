# Read-only audit: stage-reward parity is not return parity

The height-smooth PPO/SAC training and nominal NMPC use the same 15 stage
reward components, but **not the same episodic, discount or terminal-value
semantics**. This is a comparison limitation, not evidence explaining the
observed controller differences. No change or experiment was made by this audit;
the current ablation changes search coordinates only.

## What training actually terminates

[TRON1 configuration](../upstream/tron1-learning-from-human/training/tron1_tracking.py)
inherits the [tracking termination definitions](../upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py).
The [height adapter](../scripts/height_smoothness.py) removes `anchor_pos`
and `ee_body_pos` entirely: they were **Z-only** tests, not surviving XY gates.
The remaining terms are:

| Term | Exact condition | Classification |
| --- | --- | --- |
| `anchor_ori` | `abs(g_reference_body_z - g_robot_body_z) > 0.6`, using unit gravity projected into each anchor frame | True termination |
| `motion_end` | Held command frame ≥ last reference frame; 348 for primary | Timeout/truncation |
| `time_out` | Episode control-step count ≥ maximum; configured primary duration 7.02 s, 351 steps | Timeout/truncation |

The orientation threshold is **not 0.6 radians**. For the upright, yaw-only
car reference, it is `1 - cos(tilt) > 0.6`: tilt greater than approximately
66.422°. It is independent of yaw and geometric sideslip. Motion end normally
precedes the general time limit; random reference-state initialization can
shorten training episodes further.

There is no XY-distance, height, flight, contact-loss or heading-offset
termination after this adapter, and no separate termination penalty among the
15 rewards. The [recorded runtime smoke](2026-10-15-controller-integration.json)
reported exactly `time_out`,
`anchor_ori`, `motion_end`; its recorded status SHA-256 is
`33308b8bc3349bf2a6454666018e3b3df3d70c7b70106035980b05608a9182bb`.
Formal PPO/SAC manifests use the same height-adapter SHA-256
`e133aca58f671fc22ae1dfc97a03461e7724e4072bf8b74f58a2c14281d198b8`.

## Bootstrap behavior

Isaac computes the terminal flag and the terminal step's reward on the
post-action physical state, before reset and command advancement. Thus the
terminal step's reward is retained; termination affects continuation value.

- PPO uses γ=0.99 and GAE λ=0.95. Installed RSL-RL 3.0.1 resets GAE
  continuation on `done`. For timeouts, its official installed implementation
  first adds `γ V(pre-action observation)` to reward—an approximation, not
  evaluation of the actual final successor.
- [SAC](../scripts/sac_tracking.py) uses γ=0.99. True termination masks its
  target continuation; truncation alone bootstraps. Its
  [terminal-observation adapter](../scripts/terminal_observation.py) captures
  the actual pre-reset successor with advanced/clamped reference phase.
- If both flags coincide, this installed PPO timeout augmentation still
  applies; SAC's true-termination mask dominates. This corner case is not
  claimed to explain any measured performance difference.

Read-only inspection confirmed `is_finite_horizon=False` and the wrapper's
timeout forwarding. Installed source hashes, without private paths:

| Component | SHA-256 |
| --- | --- |
| RSL-RL 3.0.1 `algorithms/ppo.py` | `deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c` |
| RSL-RL `storage/rollout_storage.py` | `10510c1c0e7b250a343ba923a4747a5c337b78c286c1dd5d684128305ee0afa1` |
| IsaacLab RSL wrapper | `96fcd3a849a2b0996f16db15b38408b2fed3cc17be201f557b22df735421f1be` |

## Difference from current NMPC

The [NMPC objective](../scripts/nmpc_tracking_objective.py) explicitly excludes
tilt safety termination. Its [native rollout objective](../scripts/nmpc_batch_objective.py)
sums undiscounted stages over the whole planning horizon, with no terminal
value, including predicted states after orientation collapse. Fifteen matching
stage terms therefore do not reproduce RL's lost continuation value. Survival
is only implicitly favorable when that continuation value is positive; there
is no explicit survival bonus or guaranteed safety constraint.

A future, separately labeled **termination-consistent objective ablation**
could retain the terminal step's reward and assign zero future reward after
the same orientation event. This must not silently add height/flight/slip
gates or treat motion-end/time-limit truncations as true failures.

Crucially, current cost is `dt * (R_max - reward)`, with `R_max=12.4` for
this profile. In a fixed-horizon absorbing zero-reward state, future cost must
remain `dt * R_max` (0.248 per control step), **not zero**. Simply stopping
the cost sum would make early failure artificially cheap. Matching RL more
closely would also require explicit discount and terminal-value decisions;
event masking alone is not full equivalence. No such alternative is implemented.

Full-horizon validation continues both learned and model-based controllers
without pose-triggered resets; posture/contact remain diagnostics. The
asymmetry identified here concerns **training returns versus planning returns**,
not a hidden height-based validation failure rule. No policy action data,
physics rollouts, new constraints or reward changes were used in this audit.

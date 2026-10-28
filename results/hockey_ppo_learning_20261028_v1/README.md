# Direct PPO learning of the hockey stop

Status: both 32-environment smoke runs passed; the first 512-environment training arm has started. No learned hockey stop has passed physical qualification yet.

The previous experiments used a car-turn policy, with command substitutions or MPPI residuals, to attempt this motion. This experiment trains a new policy directly on the locked hockey reference. It uses the GMR posture and the designed global stopping path, whose speed and distance are **not measured human motion**. The failed MPPI trajectory is not a training target.

The actor now has 56 inputs: the existing 51 observations, followed by target planar velocity in the actual body's yaw frame, target world yaw rate, braking-phase indicator, and stop-hold indicator. The critic has 125 inputs. Both actors are trained from scratch; their observation normalizers are learned independently. Existing car checkpoints do not initialize either arm.

## First controlled comparison

| Setting | Tracking | Support |
| --- | --- | --- |
| Reference, reset distribution, network, seed | Identical | Identical |
| Physics / control rate | 400 / 50 Hz | 400 / 50 Hz |
| Action-rate weight | −0.5 | −0.5 |
| Continuous any-wheel load-deficit weight | 0 | +0.4 on a negative deficit |
| Training | 512 environments, 1,000 updates | Same |
| Transitions | 12,288,000 | 12,288,000 |
| Nominal evaluation | μ = 0.7, one environment, 139 controls | Same |

The only treatment difference is the continuous support-deficit weight. Both arms already condition the β reward on actual wheel loading and the terminal hold reward on any-wheel support. Thus this is an ablation of the **additional continuous load-margin penalty**, not an experiment with all contact information removed.

PPO uses RSL-RL 3.0.1, 24 steps per environment, fixed learning rate 3e−4, clip parameter 0.2, entropy coefficient 0.005, initial action standard deviation 0.5, 5 learning epochs, 4 mini-batches, γ = 0.99 and λ = 0.95. These are initial experiment settings, not optimized results. The historical 1e−4 / clip 0.1 recipe fine-tuned an existing car actor; it is not a matched from-scratch hockey baseline.

The existing shared tracking DR remains active: wheel friction uniformly drawn from [0.3, 1.2] once per environment, shared by both wheels; unit-friction ground and multiply combination; equal static/dynamic friction; zero restitution; leg encoder bias, base COM offsets, and velocity pushes. Evaluation disables these random disturbances and fixes effective wheel/ground μ at 0.7. Native rigid-cylinder Coulomb contact is retained.

## Reward and episode semantics

Approach tracking uses the reference velocity vector. During braking that vector decreases to zero; there is no constant-speed requirement during braking. After 1.8 s, a separate reward favors a supported, upright stop. XY, heading, yaw rate, six leg angles, spin-invariant wheel axes, and valid loaded per-wheel β are tracked throughout their applicable phases.

The additional support cost reads all eight actual physical substeps. For substep s,

`d_s = clip(1 − max(Fz_left, Fz_right)/(0.5 m g), 0, 1)`

and the support arm adds `−0.4 × [max_s(d_s) + mean_s(d_s)]/2`. A single wheel may carry the load, and the supporting side may change. Extra impact force above the load target gives no extra credit. Isaac multiplies reward rates by the 0.02 s control interval.

No absolute root-height target or termination is used. Height regularization penalizes bounded vertical acceleration only. Wheel XY position and wheel spin are excluded from imitation. Training episodes terminate at 45° tilt or clip completion; the clip end is a true finite-task terminal. Validation runs the complete horizon without resetting after a fall, so pre-fall and post-fall effects remain distinguishable.

Training resets use the existing adaptive reference-state initialization, including 20% starts at frame zero. Reference wheel spin is a zero placeholder and initializes the wheels, but is never a tracking target. Validation initializes directly from the locked reference, rather than injecting the old car rollout's seed state. This initialization difference must be retained when displaying old MPPI failures as historical context.

Friction saturation is not rewarded directly. Independent evaluation still requires actual material-point lateral motion, sustained same-wheel load, an aggregate friction-force ratio near one, and dissipative force/velocity pairing. No pointwise friction-cone certificate is claimed.

## Qualification and reproduction

The existing hockey task gates are unchanged: entry speed, approach-speed retention, XY error, body/wheel headings, uninterrupted any-wheel support, no fall, supported terminal stop, and a ≥50 ms loaded high-β sliding event corroborated by friction and dissipation. Pose fidelity is reported separately. Smaller XY error alone is not task success.

The [snapshot manifest](source_snapshot_v1/manifest.json) records the exact nine uploaded files. The approved archive SHA-256 is `ee2d90d6e58b6f68e4b1bddbc40659d9c824382dda908149be70820c6e16ee7e`. Models, weights, credentials, SMPL assets and source video are excluded.

Entry points:

- [`prepare_hockey_ppo.py`](../../scripts/prepare_hockey_ppo.py): build a fresh source/reference bundle.
- [`dispatch_hockey_ppo.py`](../../scripts/dispatch_hockey_ppo.py): serial smoke, training and nominal validation, with a fresh fully-idle RTX check before every stage.
- [`train_hockey_ppo.py`](../../scripts/train_hockey_ppo.py): training or full-horizon native-force recording.
- [`analyze_hockey_ppo.py`](../../scripts/analyze_hockey_ppo.py): independently reconstruct all 56 observations, replay the exported actor, reconstruct native contacts and apply the unchanged physical qualification.
- [`collect_hockey_ppo.py`](../../scripts/collect_hockey_ppo.py): retrieve finished evidence and keep actor weights under ignored `outputs/`.
- [`render_hockey_physx_probe.py`](../../scripts/render_hockey_physx_probe.py): fixed-relative-camera GMR / physical motion / XY / contact display.

The source snapshot is immutable. Any later hyperparameter or reward change needs a new version and must retain this comparison's results. CPU validation passed thirteen tests covering support sampling/reset isolation, missing-substep detection, single-wheel/alternating support, spin/height invariance, endpoint reference timing, zero-speed β masking, command coordinates and phase boundaries. The two PhysX smoke runs additionally verified actual 56/125 observation dimensions, native material readback, finite reward/forces, actor export and 1,536 transitions each. Their initial network state hashes and friction draw ranges match; the only reward-weight difference is support 0 versus 0.4. These checks establish executable plumbing, not learned task performance.

# Current priority: contact models and Isaac/PhysX training

The user narrowed the current stage to contact-model analysis, with training primarily in Isaac. Here **Isaac means the existing Isaac Lab + PhysX setup**. The wider backend and RL-algorithm comparison is deferred; its completed evidence remains available.

The previously active PhysX SAC learner completed its unchanged **12,288,000-transition** budget. Its exported actor and checkpoint passed the frozen cohort verifier. The dispatcher was held while that learner finished, then terminated after the learner exited; the obsolete full-matrix analysis waiter was also stopped. No signal was sent to the learner, and the MuJoCo-Warp formal learner never started. This completed SAC run is supplementary evidence, not a replacement for the PPO contact-model baseline or proof that a contact model is correct. No extra training budget is introduced by this priority change. See the [finalization audit](finalization.json).

A single μ = 0.7 PhysX deployment also completed and its recording was hash-verified. Its core XY RMSE is **0.365 m**, speed retention **0.947**, and the bilateral XML-cylinder gap-band fraction is **6.64%** (−5 to +1 mm). This is a retained negative result, not contact-stable task completion or an algorithm ranking. The recording lacks per-point native normals, so it does not certify friction-cone saturation. [Descriptive metrics](nominal_descriptive_metrics.json) / [deployment proof](nominal_status.json).

Reproduce those offline metrics:

```bash
python results/isaac_contact_focus_20261031_v1/analyze_physx_nominal.py \
  --result-dir results/isaac_contact_focus_20261031_v1 \
  --model /path/to/frozen/mujoco_model.xml \
  --reference /path/to/frozen/motion.npz \
  --legacy-evaluator /path/to/frozen_cpu_source/scripts/eval_tracking_mujoco.py
```

The MuJoCo XML in this command is used only for hash-locked forward kinematics and the analytic envelope. It does not simulate or change the recorded PhysX trajectory.

## Order of work

1. **Verify the contact readout in PhysX.** Keep native normal directions, normal/friction vectors, contact positions, separation, and their physical timestamps. Check force/momentum consistency and distinguish native contact geometry from an analytic cylinder-envelope estimate.
2. **Compare contact-model factors while holding the policy fixed.** Separate single-cylinder versus sliced geometry from native versus external tangential laws. Match mass, inertia, actuator settings, entry state, material settings and physical timestep. Report external-force delay as a separate factor.
3. **Use PhysX-trained PPO to study adaptation to the selected models.** Change the contact model alone in the first comparison; keep the reference, reward, observation/action contract, DR and training budget fixed. Inspect both-wheel contact before interpreting sliding or tracking gains. Existing model-specific PPO runs are historical evidence until their contact metrics pass the current audit.
4. **Only after these checks**, revisit algorithm ranking, training-backend transfer and the hockey-prior benefit experiment.

Start from nominal μ = 0.7. The static/dynamic split μs = 0.9, μd = 0.7 is a separately labeled condition. A simulator's two material coefficients do not by themselves establish the instant of a static-to-dynamic transition.

## Evaluation priorities

- **Both-wheel contact continuity:** native separations/contact reports and geometry together; force > 5 N alone is not the contact definition. Report interruptions and tolerance sensitivity, including deep penetration separately.
- **Contact-conditioned sliding:** material-point velocity relative to the ground, with pre-step/post-step/interval timing identified. Velocity during separation is not counted as ground sliding.
- **Friction capacity:** native normal load and tangential force, in their actual contact frame. Do not invent a one-to-one normal-load allocation to friction anchors when the API does not expose that association. Aggregate capacity is a necessary check, not a per-point saturation certificate.
- **Force/slip consistency:** force opposition, mechanical power and momentum residuals. A large β or utilization ratio alone is insufficient.
- **Task and smoothness:** XY error, speed retention or stopping performance, and body/action oscillation, reported alongside contact continuity.

## Evidence to use

| Evidence | Status and use |
|---|---|
| [Native-force reconstruction](../wheel_native_force_check_20261031_v1/README.md) | Completed force/momentum and ground-mesh sensitivity checks. |
| [Controlled wheel factorial](../sliding_foundation_completion_20261031_v1/wheel_factor_analysis/report.json) | 18 formal + 6 smoke cases; isolate geometry/law/timestep effects without a policy. |
| [Frozen-policy full-body model study](../physx_tron1_wheel_models_20261025_v1/analysis.json) | Historical six-cell study; retain its actor and model provenance. |
| [Model-specific PPO geometry training](../ppo_k_native_slices9_20261025_v1/four_arm_analysis.json) | Completed PhysX training/evaluation, single cylinder versus nine slices. Reassess with the current bilateral criterion. |
| [Nonlinear-model PPO training](../ppo_k_nonlinear_tire_20261025_v2/analysis/analysis.json) | Existing PhysX-trained policy; external law parameters are assumptions, not identified tire parameters. Historical one-wheel/load screens do not meet the current bilateral criterion. |
| [Hockey native contact audit](../hockey_contact_model_framework_20261031_v1/report.json) | Existing full-body native contact, force and material-point velocity analysis. |

The new 400 Hz PPO matrix and SAC learner use the **turning** reference. Hockey-stop results belong to a distinct task cohort and must not be pooled with them. Real-robot measurements remain pending; no hardware motion is requested or scheduled here.

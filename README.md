# TRON1 sliding-angle / XY failure analysis

This repository separates **reference geometry**, **BeyondMimic-style policy
training**, and **fixed-friction Isaac↔MuJoCo validation**. The question is:
for a high course-to-wheel/foot heading offset, at which validation contact
frictions does the learned TRON1 wheel-foot policy develop a large horizontal
position error? Views are top-down only. Height/flight is not a failure gate.

## Experiment contract

1. Select car or human demonstrations with a sustained, nontrivial signed
   course-to-wheel/foot angle. `γ = wrap(atan2(v_y,v_x) − wheel/foot heading)`.
   Do not equate this geometric angle with measured tire contact-patch slip.
2. Car: retain the source XY/yaw trajectory, then measure the **retargeted**
   TRON1 wheel rolling-plane headings. The pinned open
   [`drift_drl/traj_4` source](upstream/drift_drl/code/ref_trajectory/traj_4.csv) has
   vehicle slip state but does **not** provide measured individual wheel steer
   angles. Thus its robot wheel headings are retarget output, not source sensor
   readings. Use this clip first; DOA `sa_traj29` is a second high-angle
   candidate, still without a trained tracking policy. The selected reference
   is hash-locked in [`config/car_reference_lock.json`](config/car_reference_lock.json):
   at moving source frames its retargeted left-wheel offset has mean absolute
   11.27° and peak absolute 27.22°.
3. Human: use pinned [GMR](upstream/GMR) to retarget pelvis plus both ankles to
   TRON1. Add left/right human heel-to-toe heading offsets as an **auxiliary
   sidecar** using `scripts/attach_human_wheel_offset.py`; compare each foot
   independently and mask low-speed / near-ground-proxy samples. TRON1 WF has no
   independent wheel-yaw joint, so a human foot heading offset cannot be
   silently imposed as a realized joint command. A skating-marker-to-GMR clip
   with valid wheel-center fit and a corresponding trained policy is still
   required; the existing skating IK/interaction-mesh previews are **not** GMR
   results and are not substituted here. The
   [FS-Jump3D source](upstream/FS-Jump3D) is pinned as a submodule; its
   separately licensed C3D archive is not copied into this repository.
4. Train a new tracking PPO with the official BeyondMimic event ranges, mapped
   to TRON1 (`scripts/train_beyondmimic_style.py`). At startup, sample robot
   static friction U[0.3,1.6], dynamic friction U[0.3,1.2], restitution
   U[0,0.5] in 64 buckets. Also use the upstream ±0.01 rad joint default
   offset on the six position-controlled leg joints, base CoM range (mapped
   from G1 torso to TRON1 `base_Link`), and
   1–3 s velocity pushes. These are **training** randomizations. TRON1's body,
   action, wheel handling, motion and drift rewards remain adaptations, not an
   exact G1 reproduction. The ground material is fixed at 1.0 during training,
   matching the upstream nominal ground material. The earlier worst-substep
   ground-load reward is excluded: its audited mass/cache implementation only
   permits the historical fixed event set and is incompatible with new CoM
   randomization. The selected compatible reward profile is
   [`config/car_drift_beyondmimic_task.json`](config/car_drift_beyondmimic_task.json).
5. Freeze the trained actor and evaluate single deterministic episodes per
   coefficient in Isaac/PhysX and MuJoCo. Turn off random events for validation.
   First test equal static/dynamic `µ = 0.30, 0.45, 0.60, 0.70, 0.80, 1.00`
   in both engines. `µ=1.20` is pending the current evaluator's upper-bound
   extension; subsequently test `(µ_static,µ_dynamic)=(1.4,1.0)` and
   `(1.6,1.2)` once both evaluators support and read back distinct components.
   This is a **grid of fixed evaluation cases**, not a second policy training
   distribution. Equal numeric coefficients are not equal calibrated tire laws.
6. For every matched case, plot reference/Isaac/MuJoCo root XY with an equal
   XY aspect ratio. Report reference-tracking error in each engine and the
   Isaac↔MuJoCo XY difference separately. An early stop is right-censored;
   never extrapolate or splice after reset. Screen complete 1.44–4.00 s turn
   cores with reference-XY RMSE >0.15 m or peak >0.30 m. These are inherited
   *exploratory* limits, not physical drift thresholds. Report post-core drift
   separately. The numeric specification is in
   [`config/evaluation_friction.json`](config/evaluation_friction.json).

The pinned original source is [BeyondMimic's official EventCfg](upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py).
Its robot-material randomization runs at startup; it is not an evaluation
event. See also the [official BeyondMimic paper](https://arxiv.org/abs/2508.08241)
and [GMR implementation](https://github.com/YanjieZe/GMR).

## Available result: historical fixed-friction-policy baseline

The old TRON1 actor was trained at `µ=0.15` with motor randomization, **without
BeyondMimic friction randomization**. Three previously recorded evaluation
pairs are published here only as a baseline for the new protocol. This does
not establish a threshold across the BeyondMimic training range.

| Fixed validation µ | Isaac core reference XY RMSE | MuJoCo core reference XY RMSE | Matched sim2sim core XY RMSE | Status |
| ---: | ---: | ---: | ---: | --- |
| 0.05 | unavailable | unavailable | unavailable | Both stop before core end; observed overlap peak engine gap 0.232 m |
| 0.70 | 0.0985 m | 0.1030 m | 0.0153 m | Full core; no large reference XY screen |
| 1.00 | 0.1502 m | 0.1510 m | 0.0455 m | Full core; both narrowly exceed 0.15 m reference XY screen |

![Top-down XY path comparison](results/car_baseline/topdown.png)

The plot ends each engine path at its actual final observed state. Compact
[XY-only native traces](results/car_baseline/xy_trace.npz),
[machine-readable metrics](results/car_baseline/summary.json), and
[input SHA-256/readback provenance](results/car_baseline/xy_trace.provenance.json)
support independent re-analysis. No third-party raw mocap or vehicle CSV is
redistributed here.

## Reproduce what is currently public

```bash
git clone --recurse-submodules https://github.com/Shukashuki/tron1-sliding-analysis.git
cd tron1-sliding-analysis
python -m pip install -r requirements.txt
python -m pytest -q tests
python scripts/analyze_saved_xy.py \
  --trace results/car_baseline/xy_trace.npz \
  --output-dir outputs/replayed_topdown
```

The trace-replay command recreates `summary.json` and `topdown.png` without
Isaac or MuJoCo. With a local full source checkout **at the archived experiment
revision**, `--source-root /path/to/tron1-learning-from-human` imports the
original physics traces, verifies friction/readback and hashes, and regenerates
the compact trace too. The `upstream/tron1-learning-from-human` submodule is
pinned to its already-public commit; it does **not** yet contain all unpublished
historical evaluator changes or the raw checkpoints/assets. Therefore a fresh
physics rerun is not presently reproducible from public files alone. The
unpublished source branch was **not** bulk-pushed here; publishing it requires
separate review/approval because it includes many additional historical
outputs. The new standalone analysis and training scripts are kept in this
repository for narrow review.

For a new car-policy run, supply the source checkout, the licensed TRON1 USD,
the 50 Hz retargeted drift motion, and the drift task profile to
`scripts/train_beyondmimic_style.py` inside an Isaac Lab 2.3 / RSL-RL 3
environment. A representative invocation is:

```bash
python scripts/train_beyondmimic_style.py \
  --source-root /path/to/tron1-learning-from-human \
  --motion-file /path/to/drift/motion.npz \
  --asset-path /path/to/WF_TRON1A.usda \
  --task-profile config/car_drift_beyondmimic_task.json \
  --output-dir outputs/car_bm_dr_seed42 \
  --num-envs 512 --iterations 1000 --seed 42 --device cuda:0 --headless
```

Do not label a one-iteration smoke test as a trained policy. Export and validate
the final actor's observation/action contract before running the fixed-friction
grid. The original source's friction evaluators presently support a single
equal static/dynamic µ up to 1.0; `µ=1.2` and off-diagonal pairs need explicit
evaluator support and material readback before they count as results.

```bash
python scripts/run_car_validation.py \
  --source-root /path/to/full/tron1-learning-from-human \
  --motion-file /path/to/drift/motion.npz \
  --reference-file /path/to/drift/robot_reference.npz \
  --asset-path /path/to/WF_TRON1A.usda \
  --mujoco-model /path/to/tron1_model.xml \
  --checkpoint outputs/car_bm_dr_seed42/model_final.pt \
  --runner-config outputs/car_bm_dr_seed42/runner_config.json \
  --task-profile config/car_drift_beyondmimic_task.json \
  --output-dir outputs/car_fixed_friction_grid --device cuda:0
```

The driver requires a completed training status, checks the BeyondMimic
material recipe and source-motion hashes, runs both engines at each fixed µ,
verifies Isaac policy-rate observations before MuJoCo playback, then generates
`analysis/topdown.png` and `analysis/summary.json`. It will not overwrite an
existing grid. For human motion, first supply
one valid GMR retarget plus time-aligned bilateral foot markers; then run:

```bash
python scripts/attach_human_wheel_offset.py \
  --gmr-reference /path/to/gmr/robot_reference.npz \
  --foot-markers /path/to/aligned_foot_markers.npz \
  --model /path/to/WF_TRON1A/mujoco/robot.xml \
  --source-to-robot-yaw-deg 0 \
  --output-dir outputs/human_heading_sidecar
```

The marker NPZ must provide `time_s`, `heel_xyz_m[T,2,3]`, and
`toe_xyz_m[T,2,3]`, with L/R order and the exact GMR sample clock. Supply the
actual coordinate alignment angle instead of using the illustrative zero if
the source axes differ. Human and new car policy results are **pending**; no
sim2sim claim is made for either.

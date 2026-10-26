# Reproduction

Run from the repository root. NumPy is sufficient for preparation and audits;
rendering also needs the existing MuJoCo/OSMesa, Pillow, Matplotlib and ffmpeg
environment. Rendering performs zero dynamics steps. Remote physics reuses the
existing Isaac installation, TRON1 asset and cached S actor; none of those
models, weights or credentials is uploaded by these drivers.

```bash
python scripts/prepare_physx_speed_references.py \
  --motion results/camber_kinematic_reference_20261025_v1/independent/fit_differential.npz \
  --original-motion results/contact_wheel_axis_20261025_v1/reference.npz \
  --base-config config/physx_policy_mppi_reference_v1.json \
  --output-dir /tmp/physx_speed_references

python scripts/analyze_physx_speed_ladder.py \
  --probes results/physx_speed_reference_20261025_v1/evidence/probes_v1 \
  --references results/physx_speed_reference_20261025_v1/references \
  --original-reference results/contact_wheel_axis_20261025_v1/reference.npz \
  --output /tmp/physx_speed_analysis.json

python scripts/analyze_physx_speed_rescue.py \
  --rescue results/physx_speed_reference_20261025_v1/evidence/rescue_v1 \
  --baseline results/physx_speed_reference_20261025_v1/evidence/probes_v1/speed1p5 \
  --reference results/physx_speed_reference_20261025_v1/references/speed1p5/car_reference.npz \
  --output /tmp/physx_speed_rescue_analysis.json

python scripts/render_physx_speed_ladder.py \
  --probes results/physx_speed_reference_20261025_v1/evidence/probes_v1 \
  --references results/physx_speed_reference_20261025_v1/references \
  --analysis results/physx_speed_reference_20261025_v1/analysis_v3.json \
  --model outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml \
  --output-dir /tmp/physx_speed_render

python scripts/render_physx_policy_mppi_reference.py \
  --search results/physx_speed_reference_20261025_v1/evidence/rescue_v1/search_v1 \
  --validation results/physx_speed_reference_20261025_v1/evidence/rescue_v1/validation_v1 \
  --baseline-validation results/physx_speed_reference_20261025_v1/evidence/probes_v1/speed1p5 \
  --reference results/physx_speed_reference_20261025_v1/references/speed1p5/car_reference.npz \
  --model outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml \
  --speed-scale 1.5 --output-dir /tmp/physx_speed_rescue_render
```

Original host/root:
`zengxiong@14.103.233.29:/data/zengxiong/workspaces/tron1-physx-speed-reference-20261025-v1/`.
Physics used immutable `source_snapshot_v1` for the probes and
`rescue_source_snapshot_v1` for the rescue. Both snapshots and manifests are
archived in `evidence/`; the corresponding top-level code may have additional
analysis improvements. Exact command arrays, input hashes, process identities
and fresh idle-GPU snapshots are in the driver statuses/launch records.

To rerun physics on that already configured host, use the frozen driver
`scripts/run_physx_speed_ladder.py` with `--source`, `--prior-launch`,
`--audit-helper` and a **fresh** `--output` directory. It runs a16-step smoke test
then all four single-world probes, checking a completely idle GPU before each.
The serial rescue driver `scripts/run_physx_speed_rescue.py` takes
`--source`, a completed `--probe-status`, `--audit-helper` and a fresh `--output`.
It runs a smoke test, bounded full search and separate final-mean validation.
The recorded original launch files provide the exact existing source/asset/actor
paths. Do not overwrite completed evidence or replace the final weighted mean
with a best sample. Drivers signal no other process and do not retry failures.

Both videos are50FPS with native-time indices. The speed-ladder panels share
actual simulation time; shorter episodes explicitly hold their terminal state
with an END label. Camera distance1.9m, world orientation, target offset and XY
scale are fixed. Gray keypoints show the retimed kinematic base/wheel reference;
blue shows actual position/body heading, green velocity direction, and red/purple
segments the actual wheel planes. Mesh FK matches recorded wheel origins within
20μm. Graphics use MuJoCo only as an offline renderer, not as a predictor/plant.

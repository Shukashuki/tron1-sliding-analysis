# Video-based hockey-stop reference: first GMR preview

The subsequent [foot-frame and ground-geometry audit](../hockey_stop_reference_audit_20261026_v1/README.md)
retains this original preview and adds a support-adapted GMR comparison. No
hockey-stop physical success has been established.

The car-turn task remains separate. This task targets approach, foot rotation,
loaded skid-braking and a stable stop. Constant speed is required during approach,
not throughout braking. The [protocol](../../config/hockey_stop_imitation_v1.json)
retains per-wheel contact-point slip and friction-utilization evaluation for the
later PhysX stage.

## Completed result and limits

The public Kinetics-700-2020 validation set provides 50 `hockey stop` candidates.
All were downloaded and visually screened; two were processed with MediaPipe
Heavy on CPU. Those results are 2D and hip-relative 3D, not global trajectories.

A subsequent frame-by-frame review found crossfades and slow-motion replay in
OeJuSvEcY_o. The final selected episode is **VVYG0XKA0YA frames [40,105)**,
approximately 1.335–3.504 seconds within its downloaded clip. It starts after a
hard cut at frame 35 and the affected tracking-smoothing boundary. The 29.970-fps
source is explicitly normalized to 30 fps, producing 65 frames. See the
[episode audit](episode_selection_v1.json).

The pinned GVHMR pipeline completed on CPU using its original YOLO, ViTPose flip
testing, HMR2 features and **SimpleVO**. No alternative camera approximation was
used in the final result. Both raw `postproc=False` and standard `postproc=True`
outputs are retained. The standard correction changes root XY by **0.068 m RMS**
in model-estimated coordinates, so the raw motion is the primary skating preview.
The standard static-joint correction must not silently remove sliding.

Raw GVHMR was passed to the existing pinned **GMR**, with pelvis and bilateral
ankle targets, and wheel-axis orientation costs. GMR internally uses Mink; this
is not an independent standalone IK implementation. Actual realized wheel
headings/camber are recorded. Wheel spin is a zero placeholder, not an observed
human motion or a control command. No physics step or policy training ran.

| Diagnostic | First preview |
|---|---:|
| Episode length | 65 frames / 2.167 s |
| Both ViTPose ankle confidences >= 0.5 | 100% of frames |
| Reference peak speed, estimated at robot leg scale | **0.338 m/s** |
| GMR root XY fit RMSE | 0.143 m |
| Left / right wheel-position fit RMSE | 4.6 / 3.5 mm |
| Left / right axial wheel-heading fit RMSE | 20.3 / 12.9 degrees |
| Minimum left / right cylinder-bottom geometry proxy | +0.050 / -0.053 m |

**This reference is not admitted for training.** Its estimated reference speed
fails the prospective 0.8 m/s entry-speed requirement. GMR's own peak root speed
is 0.823 m/s, but that includes retargeting corrections and cannot replace the
source reference-speed requirement. Small ankle-position fitting errors also do
not imply accurate wheel headings or feasible ground contact. The speed is a
monocular model estimate with no external calibration; confidence is not accuracy.
No stopping distance, loaded sliding, friction saturation or task success is
claimed from these kinematic outputs.

The local four-column animation shows source video, reconstructed human at robot
leg scale, actual TRON1 mesh FK, and reference/realized XY with wheel headings.
Robot rendering uses software OSMesa and fixed camera distance/orientation while
following the base; no zoom changes are applied.

- `outputs/hockey_stop_source_discovery_20261026_v1/render_gmr_VVYG_v1/source_human_gmr_xy_4x1.mp4`
- `outputs/hockey_stop_source_discovery_20261026_v1/render_gvhmr_VVYG_v1/gvhmr_raw_vs_postproc_4x1.mp4`

Original RGB and RGB-containing previews remain local. Git retains derived pose
features, frozen estimated references, GMR traces, metrics and reproduction code.
SMPL/SMPL-X models, archives and pretrained neural weights remain local.

## Reproduce

Use Python 3.10 and the pinned GVHMR requirements. Exact installed packages are
recorded in [gvhmr_runtime_requirements.txt](gvhmr_runtime_requirements.txt).
The crop used imageio-ffmpeg 0.6.0 with its bundled FFmpeg 7.0.2. Rendering GMR
requires MuJoCo, Mink 1.2.0 and DAQP, as in the existing HDM05 GMR experiment.
The audited runtime used MuJoCo 3.8.1 and DAQP 0.8.7 for GMR.
The existing licensed TRON1 WF XML/mesh assets are also separate user inputs;
the XML SHA-256 is recorded in the GMR report, matching the earlier HDM05 study.

```bash
python scripts/download_hockey_stop_videos.py \
  --output outputs/hockey_stop_source_discovery_20261026_v1

python scripts/prepare_hockey_body_models.py \
  --smpl data/SMPL_python_v.1.1.0.zip --smplx data/models_smplx_v1_1.zip \
  --output outputs/hockey_stop_source_discovery_20261026_v1/body_models \
  --inventory /tmp/hockey_body_model_inventory.json

python scripts/download_hockey_gvhmr_weights.py \
  --output outputs/hockey_stop_source_discovery_20261026_v1/checkpoints

python scripts/select_hockey_stop_episode.py \
  --source outputs/hockey_stop_source_discovery_20261026_v1/clips/VVYG0XKA0YA_000241_000251.mp4 \
  --output outputs/hockey_stop_source_discovery_20261026_v1/clips/VVYG_stop_episode_40_105_30fps.mp4

CUDA_VISIBLE_DEVICES='' python scripts/reconstruct_hockey_stop_gvhmr.py \
  --video outputs/hockey_stop_source_discovery_20261026_v1/clips/VVYG_stop_episode_40_105_30fps.mp4 \
  --checkpoints outputs/hockey_stop_source_discovery_20261026_v1/checkpoints \
  --body-models outputs/hockey_stop_source_discovery_20261026_v1/body_models \
  --reuse-preprocess results/hockey_stop_video_20261026_v1/preprocess_cache_40_105 \
  --output /tmp/hockey_gvhmr --threads 4 --batch-size 1

python scripts/retarget_hockey_gvhmr_gmr.py \
  --world /tmp/hockey_gvhmr/world_raw.npz \
  --smplx-neutral outputs/hockey_stop_source_discovery_20261026_v1/body_models/smplx/SMPLX_NEUTRAL.npz \
  --model /path/to/tron1-learning-from-human/assets/robots/WF_TRON1A/mujoco/robot.xml \
  --output /tmp/hockey_gmr

MUJOCO_GL=osmesa python scripts/render_hockey_gmr.py \
  --video outputs/hockey_stop_source_discovery_20261026_v1/clips/VVYG_stop_episode_40_105_30fps.mp4 \
  --retarget /tmp/hockey_gmr --output /tmp/hockey_gmr_render --ffmpeg /path/to/ffmpeg
```

Omit `--reuse-preprocess` to recompute features. The stored cache was sliced from
completed per-frame features; all 65 decoded frames were verified pixel-identical
to the selected lossless crop. Cache, source, body-model and checkpoint hashes
are recorded. Upstream camera RANSAC is not seeded by NumPy; rerunning camera
estimation is not claimed bit-identical. Use the frozen reference arrays when
comparing downstream methods.

User-supplied body models are necessary; the preparation script does not download
or upload them. See the [inventory](body_model_inventory_v1.json). The weight
mirror is the one used by the demo linked from upstream GVHMR's README; downloads
are pinned by size and SHA-256 and exclude body models.

Eight stopping/geometry tests pass. They check stop-hold duration, missing-frame
rejection, uncalibrated input rejection, bilateral headings, and spin-independent
wheel-plane headings. Synthetic smoke tests are labeled separately from the
actual video result in [runtime preflight](gvhmr_runtime_preflight_v1.json).

## Sources and attribution

[CVDF distribution](https://github.com/cvdfoundation/kinetics-dataset),
[MediaPipe pose API](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python),
[pinned GVHMR installation](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md),
[static-joint correction](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/model/gvhmr/utils/postprocess.py).

The CPU reconstruction wrapper adapts GVHMR code by the 3D Vision Group, State
Key Lab of CAD&CG, Zhejiang University. Retain its
[copyright and noncommercial research license](GVHMR_LICENSE.txt) and cite the
upstream publications. GVHMR and GMR are pinned submodules.

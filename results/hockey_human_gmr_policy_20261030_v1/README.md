# Time-aligned human → GMR → trained PPO display

The horizontal four-panel video shows the same elapsed time in the human RGB
episode, estimated human skeleton, source GMR retarget, and a successful recorded
PhysX PPO rollout. The selected policy is the v9 split-friction continuation,
with **static friction 0.9 and dynamic friction 0.7**. This is an existing result;
the renderer performs no training or physics simulation.

| Panel | Displayed data |
| --- | --- |
| Human video | The original 65-frame hockey-stop episode, normalized to 30 fps, with unchanged framing |
| Human skeleton | The corresponding GVHMR estimate stored with GMR, uniformly scaled by 0.78327264 to robot leg scale |
| GMR retarget | The pinned `gmr_support_projection_v1` output, including its existing support-height projection; source global XY retained |
| Trained PPO | The v9 **nominal** PhysX recording, with named-joint mesh replay and native contact markers |

The policy uses **GMR posture plus a separately designed stop path**. It does not
track the human's estimated global XY. Its reference blends the initial posture
over 0–0.30 s, brakes over 0.40–1.80 s, and continues through 2.76 s. The third
panel intentionally shows the source GMR result rather than substituting this
training reference. Wheel spin in source GMR is a zero placeholder. The source
model's cylinder width is 10 mm; the training/PhysX collision width is 50 mm.
Both robot panels use the original visual meshes, with collision proxies hidden.

## Alignment and camera

All panels share elapsed seconds from the first episode frame. RGB, skeleton and
GMR use the **same preceding 30 Hz source frame**; there is no duration
normalization or pose interpolation. Their last pose is at 2.133333 s (65 frames
encode to 2.166667 s at 30 fps). After that pose timestamp, all three source
panels show an explicit frozen-frame banner while PPO continues to 2.76 s.
The designed task-phase bar is not an inferred event annotation for the video.

The real-time video samples the recording every 20 ms, at 50 fps. The 8× slow
video displays **all 1,105 physical states at 2.5 ms spacing**, also at 50 fps;
substep motion is not smoothed away. The source-frame hold lag is below 33.34 ms
before its endpoint. The final pose occupies one encoded frame, so video
durations are 2.78 s and 22.10 s. The CSV records the exact frame mapping.

All three 3D views use a fixed 2.0 m camera distance, azimuth 135°, elevation
−18°, and look-at offset `[0, 0, −0.24]` from each subject's pelvis/root. There
is no automatic zoom or rotation. The RGB view retains the original camera.
The human reconstruction is uncalibrated; its floor is a visualization proxy.

## Selected result and checks

The selected nominal recording passes all 11 frozen task gates: body XY RMSE
**0.02308 m**, both wheels touching at every checked 400 Hz sample, and 0.50 s
of qualified terminal stop hold. The same actor passes the nominal plus 32
prescribed initial-state perturbations (**33/33**). This is one trained seed and
one material setting, not a new generalization estimate or a claim of zero
jitter. See the [existing robustness audit](../hockey_static_dynamic_transition_20261029_v1/robustness_audit_v1/report.json).

Green markers use native contact geometry at the matching pre-step pose:
raw row `i`, pose `i`, reconstructed contact slot `i+1`. Contact eligibility is
separation ≤0.1 mm with a valid normal; no load threshold is imposed. The
renderer does not infer static/dynamic solver transitions from these markers.
It verifies every recorded wheel origin against mesh forward kinematics, input
hashes and qualification identities, and decodes both complete output videos.

## Reproduce locally

From the repository root, with the separately obtained robot meshes and original
source episode at the paths in [selection.json](selection.json):

```bash
python -m pytest tests/test_hockey_video_alignment.py -q
MUJOCO_GL=osmesa python scripts/render_hockey_human_to_policy.py \
  --config results/hockey_human_gmr_policy_20261030_v1/selection.json \
  --output outputs/hockey_human_gmr_policy_20261030_v1
```

Dependencies: NumPy, OpenCV, MuJoCo with OSMesa, Pillow and `imageio-ffmpeg`;
pytest for the three time-alignment checks. The output directory must be new.
The original episode and human reconstruction are existing local inputs; this
command neither downloads them nor loads SMPL models or policy weights.

Local outputs are `human_gmr_ppo_realtime_4x1.mp4`,
`human_gmr_ppo_slow8x_4x1.mp4`, and four stills in the output directory above.
These RGB-containing media remain local. Git contains the renderer, selection,
frame mapping and render provenance, without copying the source RGB, body
models, policy weights or robot meshes. The [provenance](provenance.json) records
the local media hashes and complete encoding checks.

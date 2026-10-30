# Left and right friction cones only

[Full animation](render/friction_cones_8x_2x1.mp4) ·
[GIF preview](render/friction_cones_preview.gif).

This layout keeps only the two triangular wheel-resultant friction cones from
the [previous display](../hockey_friction_triangle_20261030_v1/README.md).
The shared time, phase, COM speed, force values, capacity ratios and contact/slip
readouts remain visible. The material is static/dynamic **0.9/0.7**.

The full video contains every 400 Hz interval from **0 to 2.76 s**: 1,104 frames
at 50 fps, giving 8× slow motion. The 276-frame GIF samples every fourth video
frame for a smaller preview. This is a layout change with no physics, force,
contact or policy changes. Forces and geometry keep their existing interval
alignment; the radial cone uses the full tangential magnitude on the right half.

```bash
python scripts/render_hockey_friction_pair.py \
  --analysis results/hockey_static_dynamic_transition_20261029_v1/trained_contact_points_v1 \
  --output <fresh-output-directory>
```

The renderer verifies input hashes, decodes the entire MP4 and checks the GIF's
frame count and duration. [Provenance](render/provenance.json).

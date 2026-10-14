# Wider fixed-friction validation: µ=0.01–2.00

The same frozen TRON1 WF actor from the [initial six-case experiment](2026-10-14-bm-friction-grid.md) was evaluated at 17 equal static/dynamic friction coefficients, with one seed-42 episode per simulator and coefficient. The checkpoint, reference, training material randomization, controller and 50/200 Hz clocks were unchanged. An isolated evaluation source extended only the coefficient-validation bounds from 1.0 to 2.0; the [source audit](car_bm_dr_wide_seed42/evaluation_source_audit.json) records input and generated hashes. Both robot and ground coefficients were checked from Isaac's material readbacks, and MuJoCo's compiled sliding-friction coefficients were checked. The original six cases reproduced exactly the previously published XY metrics.

![Top-down XY trajectories and actual observation durations for 17 friction coefficients](car_bm_dr_wide_seed42/topdown.png)

## What early termination means

The reference lasts 6.98 s and its turning core is 1.44–4.00 s. Every early stop in this sweep was triggered by the inherited `anchor_pos` termination. Despite the name, its implementation is `bad_anchor_pos_z_only`: it stops an episode when the absolute vertical anchor error exceeds **0.35 m**. The recorded paths end at the actual first-episode terminal state; no reset or extrapolated continuation is appended.

This stopping rule is distinct from the requested XY assessment. A Z-triggered stop is recorded as missing later data, rather than counted as a large-XY failure. In particular, a clip stopped before 1.44 s cannot establish how it would track the drifting turn. Training's termination rules remain unchanged in this experiment.

| µ | Isaac end time | MuJoCo end time | Isaac observed XY peak | MuJoCo observed XY peak | Core coverage |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.01 | 0.66 s | 0.64 s | 0.2860 m | 0.2733 m | Neither reaches core |
| 0.03 | 0.68 s | 0.66 s | 0.3001 m | 0.2966 m | Neither reaches core |
| 0.05 | 0.70 s | 0.68 s | 0.3125 m | 0.3162 m | Neither reaches core |
| 0.10 | 0.82 s | 0.76 s | 0.4273 m | 0.3938 m | Neither reaches core |
| 0.15 | 1.14 s | 6.76 s | 0.2085 m | 0.1738 m | MuJoCo alone completes core |

The observed XY peaks above are calculated only over the available portion of each clip. At µ=0.05 and 0.10, both engines already show reference-XY error above 0.30 m during the initial segment, before the turn core. This establishes an observed horizontal tracking error, but does not establish a complete-turn failure boundary. At µ=0.15, MuJoCo's complete-core reference XY RMSE is 0.0568 m and peak is 0.0924 m; Isaac has no complete-core verdict. The μ=0.15 MuJoCo stop occurs after the core, at 6.76 s.

## Complete-turn measurements

Both engines completed the entire 6.98 s reference at every sampled µ from 0.20 through 2.00. None exceeded the exploratory core reference-XY screens: RMSE >0.15 m or peak >0.30 m.

| µ | Isaac core reference XY RMSE | MuJoCo core reference XY RMSE | Isaac–MuJoCo core XY RMSE |
| ---: | ---: | ---: | ---: |
| 0.20 | 0.0588 m | 0.0312 m | 0.0489 m |
| 0.30 | 0.0436 m | 0.0323 m | 0.0212 m |
| 0.45 | 0.0335 m | 0.0340 m | 0.0169 m |
| 0.60 | 0.0340 m | 0.0354 m | 0.0142 m |
| 0.70 | 0.0253 m | 0.0372 m | 0.0249 m |
| 0.80 | 0.0247 m | 0.0303 m | 0.0138 m |
| 1.00 | 0.0336 m | 0.0426 m | 0.0272 m |
| 1.20 | 0.0360 m | 0.0373 m | 0.0163 m |
| 1.40 | 0.0268 m | 0.0296 m | 0.0209 m |
| 1.60 | 0.0389 m | 0.0462 m | 0.0291 m |
| 1.80 | 0.0293 m | 0.0484 m | 0.0373 m |
| 2.00 | 0.0293 m | 0.0408 m | 0.0332 m |

Increasing validation friction to 2.00 did not produce a large core XY deviation for this reference and policy. The largest paired core XY RMSE in a complete pair was 0.0489 m at µ=0.20. Low-friction extrapolation exposes missing turn data and early horizontal errors; no precise full-turn friction failure boundary has been measured. XY tracking error also includes along-path lag and should not automatically be interpreted as lateral contact-patch slip.

The trained TRON1 configuration uses the `average` ground material combine rule, while the official BeyondMimic ground configuration uses `multiply`. The policy uses the upstream randomization event ranges, with TRON1's retained material pairing. Equal coefficients in PhysX and MuJoCo are still not a calibrated tire model. This sweep uses one seed, one reference and equal static/dynamic pairs; unequal pairs and full-horizon low-friction diagnostic rollouts are separate experiments.

## Public replay

The [summary](car_bm_dr_wide_seed42/summary.json), [native-clock XY traces](car_bm_dr_wide_seed42/xy_trace.npz), and [material/source/termination provenance](car_bm_dr_wide_seed42/xy_trace.provenance.json) are public. Recompute the complete 17-case table and figure without either simulator:

```bash
python scripts/analyze_friction_grid.py \
  --trace results/car_bm_dr_wide_seed42/xy_trace.npz \
  --expected-summary results/car_bm_dr_wide_seed42/summary.json \
  --output-dir outputs/replayed_wide_grid
```

The top-level README gives the isolated source-generation and full paired physics-rerun commands. Licensed robot assets and the checkpoint remain on the authorized remote host.

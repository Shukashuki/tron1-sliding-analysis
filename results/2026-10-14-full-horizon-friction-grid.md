# Full-horizon low-friction XY diagnostic

The [wider friction sweep](2026-10-14-wide-friction-grid.md) stopped several low-friction episodes because of the inherited vertical-anchor predicate. This follow-up records the same frozen policy for the complete 6.98 s reference. It retains the original height, gravity-orientation and wheel-height predicates as diagnostics, while motion-end/time limits remain active. No training, reset state, controller, material coefficient or reference is changed. Nonfinite states or an unexpected MuJoCo time reset abort the run. An isolated source snapshot and its [hash audit](car_bm_dr_full_horizon_seed42/evaluation_source_audit.json) make these evaluation changes explicit.

![Complete root XY paths at seven validation friction coefficients](car_bm_dr_full_horizon_seed42/topdown.png)

All fourteen episodes reached the motion end. The core remains 1.44–4.00 s, with reference-XY RMSE >0.15 m or peak >0.30 m as exploratory horizontal-error screens. A pose-condition flag does not define an XY failure here.

| Equal static/dynamic µ | Isaac core reference XY RMSE | MuJoCo core reference XY RMSE | Isaac–MuJoCo core XY RMSE | Large reference-XY screen |
| ---: | ---: | ---: | ---: | --- |
| 0.01 | 0.9764 m | 1.0858 m | 0.1214 m | Both |
| 0.03 | 1.1222 m | 1.1907 m | 0.1040 m | Both |
| 0.05 | 1.6246 m | 1.6429 m | 0.0283 m | Both |
| 0.10 | 2.3618 m | 2.4261 m | 0.1234 m | Both |
| 0.15 | 1.8932 m | 0.0568 m | 1.9376 m | Isaac only |
| 0.20 | 0.0588 m | 0.0312 m | 0.0489 m | Neither |
| 0.70 | 0.0253 m | 0.0372 m | 0.0249 m | Neither |

The µ=0.15 pair is the clearest sampled sim2sim failure: MuJoCo follows the core while Isaac develops a large horizontal error. At µ=0.05, both simulators fail to follow the reference but remain close to each other; a small sim2sim gap therefore does not imply successful imitation. The µ=0.20 and 0.70 control cases reproduce the previous XY metrics exactly, supporting that the diagnostic affects continuation after pose predicates rather than the preceding physics.

For this seed and clip, the sampled transition from large to small core reference-XY error is bracketed by µ=0.15 and 0.20 in Isaac, and µ=0.10 and 0.15 in MuJoCo. This is a coarse empirical bracket, not a calibrated physical or statistical friction threshold. Pose errors are substantial after the first would-stop event: the vertical-anchor predicate is active during roughly 84–91% of policy steps in Isaac for µ≤0.15. These continuations include failed posture and states beyond the actor's normal training rollout; they do not demonstrate actively controlled drifting or measured contact-patch slip.

The first recorded vertical-anchor flag times agree with the earlier stops: Isaac 0.66, 0.68, 0.70, 0.82 and 1.14 s at µ=0.01, 0.03, 0.05, 0.10 and 0.15; MuJoCo 0.64, 0.66, 0.68, 0.76 and 6.76 s. Both engines have no pose-predicate flags at µ=0.20 and 0.70. Per-condition active fractions and onset times are in the [provenance](car_bm_dr_full_horizon_seed42/xy_trace.provenance.json).

## Reproduce

Recompute the figure and metrics from the [compact trace](car_bm_dr_full_horizon_seed42/xy_trace.npz):

```bash
python scripts/analyze_friction_grid.py \
  --trace results/car_bm_dr_full_horizon_seed42/xy_trace.npz \
  --expected-summary results/car_bm_dr_full_horizon_seed42/summary.json \
  --output-dir outputs/replayed_full_horizon
```

For a physics rerun, first create a separate source from the pinned public checkout (apply the formal-source compatibility patch in the README when exact historical source hashes are required):

```bash
python scripts/materialize_full_horizon_source.py \
  --source-root upstream/tron1-learning-from-human \
  --output-dir outputs/full_horizon_source
```

Use the top-level `run_car_validation.py` command with `--source-root outputs/full_horizon_source`, a new output directory, `--mu 0.01 0.03 0.05 0.10 0.15 0.20 0.70` and `--full-horizon-diagnostic`. The flag must agree with the source audit. Licensed model assets and the unchanged checkpoint are required for simulation.

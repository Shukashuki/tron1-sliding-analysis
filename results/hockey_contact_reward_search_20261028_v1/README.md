# Hockey contact-model and reward comparison

**9/9 screens completed; 0 nominal successes.** The fixed nine-cell sweep is complete; further reward learning remains necessary. Every completed actor stays upright, but none preserves bilateral contact over the complete reference. Tracking and sliding metrics below are diagnostic; they do not override this failure. No robustness success rate is available.

## Measured results

Friction values are static/dynamic. “Interruption” means at least one wheel loses contact; “Both airborne” is the subset with neither wheel touching. Stop hold must reach 0.5 s.

| Contact model / friction | Policy | Both touching | Interruption (ms) | Both airborne (ms) | XY RMSE (m) | Stop hold (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cylinder 0.9/0.7 | parent | 92.75% | 200.0 | 0.0 | 0.0176 | 0.5000 |
| Nine slices 0.9/0.7 | parent | 94.38% | 155.0 | 0.0 | 0.0175 | 0.5000 |
| Cylinder 0.7/0.7 | parent | 92.75% | 200.0 | 0.0 | 0.0177 | 0.5000 |
| Cylinder 0.9/0.7 | dense | 97.64% | 65.0 | 17.5 | 0.0406 | 0.4425 |
| Nine slices 0.9/0.7 | dense | 95.83% | 115.0 | 10.0 | 0.0349 | 0.4450 |
| Cylinder 0.7/0.7 | dense | 99.73% | 7.5 | 0.0 | 0.0287 | 0.5000 |
| Cylinder 0.9/0.7 | tight | 95.38% | 127.5 | 17.5 | 0.0502 | 0.3400 |
| Nine slices 0.9/0.7 | tight | 95.11% | 135.0 | 67.5 | 0.0364 | 0.4800 |
| Cylinder 0.7/0.7 | tight | 92.57% | 205.0 | 100.0 | 0.0648 | 0.4925 |
| Cylinder 0.9/0.7 | load_free | 96.74% | 90.0 | 0.0 | 0.0304 | 0.4125 |
| Nine slices 0.9/0.7 | load_free | 97.92% | 57.5 | 35.0 | 0.0191 | 0.5000 |
| Cylinder 0.7/0.7 | load_free | 99.55% | 12.5 | 5.0 | 0.0327 | 0.5000 |

[Verified summary](final_summary_v2/report.json) · [CSV](final_summary_v2/completed_cases.csv) · [Contact-event audit](final_failure_audit_v2/report.json)

The closest screen is `native_equal_dense`: every other task gate passes, but the right wheel loses contact for 7.5 ms at 0.8225–0.8300 s, reaching 0.198 mm clearance. Native normal and tangential forces are zero during the first 5 ms of this event ([force diagnostic](reward_signal_audit_v1/native_equal_dense_near_gap.json)). The fixed 0.1 mm tolerance is unchanged.

Increasing contact penalties with the `tight` recipe worsens continuity in all three contact models. The equal-friction `load_free` screen also passes the stop gate, but interrupts contact for 12.5 ms, including 5 ms with neither wheel touching. The `load_free` recipe changes several terms together; it is not a single-factor ablation. With nine slices it passes the stop gate, but still has 35 ms of simultaneous contact loss. These single-seed screens do not establish a causal effect of friction alone.

## Fixed-follow animations

Each animation has four horizontal panels: frozen parent, `dense`, `tight`, `load_free`. The camera remains at the same relative pose and 2.4 m distance. Slow motion preserves every 400 Hz physical sample. MuJoCo supplies graphics/FK only; the recorded physics is PhysX, without smoothing or pose correction. Contact markers use native geometry without a force threshold.

| Contact configuration | Full motion | Contact slow motion |
| --- | --- | --- |
| Cylinder, 0.9/0.7 | [4×1 replay](native_reward_comparison_render_v1/native_rewards_fullbody_4x1.mp4) | [8× excerpt](native_reward_comparison_render_v1/native_rewards_contact_excerpt_8x_4x1.mp4) |
| Nine slices, 0.9/0.7 | [4×1 replay](slices_reward_comparison_render_v1/slices_rewards_fullbody_4x1.mp4) | [8× excerpt](slices_reward_comparison_render_v1/slices_rewards_contact_excerpt_8x_4x1.mp4) |
| Cylinder, 0.7/0.7 | [4×1 replay](equal_reward_comparison_render_v1/equal_rewards_fullbody_4x1.mp4) | [8× excerpt](equal_reward_comparison_render_v1/equal_rewards_contact_excerpt_8x_4x1.mp4) |

Additional controlled views: [cylinder friction × reward](cylinder_friction_comparison_render_v1/cylinder_friction_fullbody_4x1.mp4), [cylinder/slices × reward](split_friction_comparison_render_v1/split_friction_fullbody_4x1.mp4). Each render directory contains source hashes, FK checks and camera settings in `provenance.json`.

## Training and qualification

All screens start from the same existing 12,288,000-transition PPO actor, critic and normalizers, then use a fresh optimizer for **3,072,000 additional transitions** (512 environments × 24 steps × 250 iterations). They are warm-started screens, not cold-start or equal-budget comparisons against the parent. Learning rate is 1e-4, initial exploration standard deviation 0.1, entropy coefficient 0.001, full-clip reset probability 0.5, with integer reference phases and fall-only adaptive sampling.

All models use 400 Hz physics / 50 Hz control, 1 mm contact offsets and zero rest offsets. The nine slices are rigid adjacent cylinders with unchanged outer dimensions, mass and inertia; they are not an elastic tire. Dynamic friction retains U(0.3, 1.2) randomization, with the stated static/dynamic ratio applied to the same draw. Ground friction is 1 with multiply combination. Existing encoder/COM/push randomization remains. No body-height or equal-load target is introduced.

| Reward recipe | Contact fraction | Contact loss | Clearance | Sliding | Angular acceleration | Leg imitation | Gap scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | +2 | −4 | −2 | +0.5 | −0.05 | +1 | 10 mm |
| `tight` | +4 | −8 | −4 | +0.5 | −0.05 | +1 | 3 mm |
| `load_free` | +2 | −4 | −2 | +1 | −0.1 | +0.25 | 10 mm |

Contact shaping reads all eight physical substeps. Dense XY/speed/heading/beta feedback remains available after contact loss; sliding credit additionally requires a complete bilateral window and opposing, dissipative friction near dynamic capacity. The fall cost is −8 per event (configured weight −400 × 0.02 s). Aggregate friction utilization is corroborating evidence, not a pointwise friction-cone certificate or an observed solver stick/slip state.

Independent qualification checks all 56 observations, normalized actor replay, material readback, raw contact buffers and the entire 2.76 s reference. It requires bilateral geometric contact at every physical sample plus the endpoint, followed by the unchanged speed, XY, heading, tilt, 0.5 s terminal hold and ≥0.05 s corroborated sliding gates. Unequal wheel loads are allowed. A nominal pass triggers 33 single-environment initial-state trials; it is not sufficient to claim the research goal is achieved.

## Next reward candidates

An [offline signal audit](final_reward_signal_audit_v2/report.json) of all nine screens found that the 50 Hz stop reward can miss an intermediate speed violation, while 28–37% of physical samples saturate the angular-acceleration penalty at 50 rad/s². This identifies reward limitations, not a proven failure mechanism. The legacy stop replay uses a native-force proxy, not the unarchived training filtered force matrix.

The follow-up [refinement specification](../../config/hockey_contact_refinement_v1.json) first replaces capped angular cost with a non-saturating pseudo-Huber cost, then optionally adds bilateral geometric stop scoring at both ends of every substep (half mean plus half worst score). Numeric weights, original initialization, training budget, contact models, reference and success gates stay fixed. It starts with the closest equal-friction cylinder configuration. The six-candidate queue has now started after all nine screens failed. The [v3 source manifest](source_snapshot_v3/manifest.json), [verified upload receipt](upload_receipt_v3.json) and [launch record](remote_evidence_v3/launch_v3.json) pin its code and settings. A dispatcher guard enforces completion of the predecessor; the original v2 snapshot and results remain unchanged. The first candidate passed its 32-environment/two-iteration smoke audit and is in formal training on GPU 7. No completed refinement policy result is available yet. [Offline candidate scores](final_refinement_scores_v2/report.json) are not predicted policy improvements.

The [reference geometry audit](../hockey_reference_contact_gap_20261029_v1/README.md) also found one-wheel grounding in the source reference. Its proposed bilateral geometric correction has not been dynamically validated and is not part of this sweep. External nonlinear tire and elastic brush models remain outside this comparison; see the [model limitations](../hockey_split_friction_20261028_v1/README.md).

## Reproduction and execution record

[Specification](../../config/hockey_contact_reward_search_v1.json) · [Snapshot preparation](../../scripts/prepare_hockey_contact_search.py) · [Trainer](../../scripts/train_hockey_contact_search.py) · [Reward/model adapter](../../scripts/hockey_contact_search.py) · [Dispatcher](../../scripts/dispatch_hockey_contact_search.py) · [Nominal qualifier](../../scripts/qualify_hockey_contact_search.py) · [33-case qualifier](../../scripts/analyze_hockey_contact_robustness.py)

The [public recording subset](publication_v2/manifest.json) contains unchanged robot trajectories, native contact buffers and qualification reports for all twelve recordings (three parents and nine screens). Only process/device status metadata is reduced. For offline analysis or rendering, use `publication_v2` as the evidence directory; the locked reference and open robot asset are still required. No weights, licensed body-model assets or source video are included.

The immutable [v2 manifest](source_snapshot_v2/manifest.json) pins 41 code/config/test files, 440,876 bytes. [Upload receipt](upload_receipt_v2.json) and [launch record](remote_evidence/launch_v2.json) identify the authorized remote snapshot. Existing assets, reference and parent weights are reused by SHA-256. Jobs run serially on a verified available RTX; old paused jobs remain paused. The dispatcher stops on runtime/audit errors and never displaces another workload.

Read-only collection and analysis:

```bash
python scripts/collect_hockey_contact_search.py --output results/hockey_contact_reward_search_20261028_v1/remote_evidence --include-rollouts
python scripts/summarize_hockey_contact_search.py --bundle results/hockey_contact_reward_search_20261028_v1/source_snapshot_v2 --evidence results/hockey_contact_reward_search_20261028_v1/remote_evidence --output <fresh-summary-directory>
python scripts/analyze_hockey_contact_screen_failures.py --bundle results/hockey_contact_reward_search_20261028_v1/source_snapshot_v2 --evidence results/hockey_contact_reward_search_20261028_v1/remote_evidence --reference results/hockey_speed_success_20261028_v1/source_snapshot_v1/references/s100 --output <fresh-audit-directory>
```

Runtime status: `/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1/run_v2/status.json`, tmux `hockey_contact_reward_search_20261029_v2`. Local reports are timestamped observations. Successful source, collector and mathematical checks do not imply successful robot behavior.

Refinement status: `run_v3/status.json` under the same remote root, tmux `hockey_contact_reward_refinement_20261029_v3`. Collect it with `--version v3` into `remote_evidence_v3`; pass `--run-name run_v3` to the analysis/render tools.

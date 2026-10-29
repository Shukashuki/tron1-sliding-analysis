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

## Smoothness and stop refinements

**4/6 refinement screens completed; 0 nominal successes.** The first three replace only the capped angular-acceleration cost with a non-saturating pseudo-Huber cost. The remaining three also replace endpoint stop scoring with bilateral geometric stop scoring at both ends of every physical substep (half mean plus half worst score). The equal-friction combined-reward arm has completed; its unequal-friction cylinder counterpart is training. Numeric weights, original warm parent, 3,072,000-transition budget, contact models, reference and success gates remain fixed. [Specification](../../config/hockey_contact_refinement_v1.json), [immutable source manifest](source_snapshot_v3/manifest.json), [verified summary](refinement_summary_v4/report.json).

An [offline audit of the nine earlier screens](final_reward_signal_audit_v2/report.json) found that the 50 Hz stop score can miss intermediate speed violations and that 28–37% of physical samples saturate the former angular penalty at 50 rad/s². These are reward limitations, not proven failure mechanisms. The legacy stop replay uses a native-force proxy, not the unarchived training filtered force matrix.

Each entry below is `dense → smooth`, within the same contact model:

| Contact model / friction | Interruption (ms) | Both airborne (ms) | Angular RMS (rad/s²) | XY RMSE (m) | Stop hold (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cylinder 0.7/0.7 | 7.5 → 2.5 | 0 → 0 | 110.22 → 78.95 | 0.02866 → 0.03287 | 0.5000 → 0.4825 |
| Cylinder 0.9/0.7 | 65.0 → 57.5 | 17.5 → 25.0 | 87.20 → 85.49 | 0.04058 → 0.03881 | 0.4425 → 0.5000 |
| Nine slices 0.9/0.7 | 115.0 → 57.5 | 10.0 → 2.5 | 101.64 → 68.79 | 0.03493 → 0.02670 | 0.4450 → 0.4750 |

All six entries fail the overall task. Angular RMS covers the 2.78 s recording; contact and stop gates cover the 2.76 s reference. These are single-seed matched screens, not robustness estimates. Comparing different friction models also changes the training friction ratio; it is not a same-policy friction sweep.

The remaining failures differ:

- **Cylinder 0.7/0.7:** the right wheel loses contact at 0.6250–0.6275 s (native separation 0.124 mm, zero interval-averaged normal force). Terminal contact is bilateral, but speed exceeds 0.1 m/s from 2.745 s and ends at 0.1215 m/s. [Aligned force check](refinement_failure_audit_v1/near_gap_force_check.json).
- **Cylinder 0.9/0.7:** every gate except bilateral continuity passes. Both wheels lose contact at 1.725–1.750 s, with maximum gaps 0.755 mm left and 1.734 mm right. Less total interruption therefore coexists with more simultaneous contact loss.
- **Nine slices 0.9/0.7:** angular RMS decreases by 32.3%, but contact and stop gates still fail. Right-wheel interruptions occur at 0.5825–0.6100 s and 0.9375–0.9450 s. Left contact is lost from 2.7375 s, and both contacts are absent during the final 2.5 ms and at the endpoint. Speed stays below 0.1 m/s throughout the final 0.5 s; this stop failure is caused by missing contact, not excessive speed. Including the endpoint, maximum gaps are 2.767 mm left and 1.328 mm right.

[Contact-event audit](refinement_failure_audit_v3/report.json) · [unchanged recordings](refinement_publication_v3/manifest.json) · [formula-aware offline replay](refinement_reward_audit_v4/report.json). Offline scores are computed from validation recordings; they are not logged training reward or predicted learning improvements. No robustness suite has run because no nominal candidate has passed.

The fourth screen, **cylinder 0.7/0.7 `smooth_stop`**, passes every gate except continuous bilateral contact. Relative to `smooth`, stop hold improves from 0.4825 to 0.5000 s and XY RMSE from 0.03287 to 0.02335 m, but contact interruption increases from 2.5 to **147.5 ms**, including 27.5 ms with neither wheel touching. Angular RMS is 81.67 rad/s² versus 78.95. Its contact interruptions occur at 0.5850–0.6325, 0.7025–0.7250, 0.9050–0.9100 and 1.7450–1.8175 s. This single-seed result demonstrates a tradeoff in the trained actor, not a general causal effect of stop rewards. [Four-screen audit](refinement_failure_audit_v4/report.json), [recordings](refinement_publication_v4/manifest.json), [selected-formula replay](refinement_reward_audit_v5/report.json).

Fixed-follow comparisons retain individual source/qualification hashes for every panel:

- [Equal-friction parent / dense / load-free / smooth](refinement_equal_render_v1/equal_smooth_fullbody_4x1.mp4), with [8× contact excerpt](refinement_equal_render_v1/equal_smooth_contact_excerpt_8x_4x1.mp4).
- [Equal versus unequal friction, dense / smooth pairs](refinement_friction_render_v1/friction_smooth_fullbody_4x1.mp4), with [8× interruption excerpt](refinement_friction_render_v1/friction_smooth_contact_excerpt_8x_4x1.mp4).
- [Cylinder versus nine slices at 0.9/0.7, dense / smooth pairs](refinement_geometry_render_v1/geometry_smooth_fullbody_4x1.mp4), with [8× terminal excerpt](refinement_geometry_render_v1/geometry_smooth_contact_excerpt_8x_4x1.mp4), [panel sources](refinement_geometry_render_inputs_v1.json) and [provenance](refinement_geometry_render_v1/provenance.json).
- [Equal-friction parent / dense / smooth / smooth-stop](refinement_stop_render_v1/stop_refinement_fullbody_4x1.mp4), with [8× contact-loss excerpt](refinement_stop_render_v1/stop_refinement_contact_excerpt_8x_4x1.mp4), [panel sources](refinement_stop_render_inputs_v1.json) and [provenance](refinement_stop_render_v1/provenance.json).

A [normal-motion audit](refinement_liftoff_audit_v3/report.json) identifies a separate stopping issue in the earlier dense/smooth pairs. The equal-friction and nine-slice `smooth` actors finish with base COM vertical speeds of **1.430 and 1.324 m/s**, rising **0.115 and 0.205 m** during the final 0.5 s. Position differences corroborate the recorded velocities to below 0.001 m/s RMS. Existing stop rewards constrain XY speed and tilt, while vertical regularization penalizes acceleration without requiring final vertical velocity to settle. A three-dimensional stop-speed reward is therefore a candidate for a later sweep; it would not introduce an absolute-height target.

The same audit **does not support vertical-penalty saturation as the explanation**: none of the four recordings reaches a cost of 0.999, and none has a control-mean acceleration below 5 m/s² while a physical substep exceeds 50 m/s². This contrasts with the earlier angular-penalty saturation finding. Normal clearance changes are distinct from tangential sliding; the audit leaves all contact/sliding criteria unchanged and establishes no causal controller mechanism. [Four-column plot with shared vertical scales](refinement_liftoff_audit_v3/normal_motion_4x1.png), [liftoff events](refinement_liftoff_audit_v3/liftoff_events.csv), [code](../../scripts/audit_hockey_contact_liftoff.py). Three checks cover time averaging, contact-loss indexing and incomplete windows.

The [reference geometry audit](../hockey_reference_contact_gap_20261029_v1/README.md) found one-wheel grounding in the source reference. Its bilateral geometric correction has not been dynamically validated and is not used here. A separate [RSI audit](rsi_reference_audit_v1/report.json) finds that only reference frames 0, 1, 2 and 35 are bilateral. The frozen sampler chooses frame 0 with probability 0.5; the remainder uses 14 adaptive bins and frames 0–137. Uniform bins would give 51.45% bilateral selected reference poses. Actual adaptive weights and reset contact traces were not archived, so this is **not a measured training contact rate**; it excludes joint-limit clipping and physics settling. It identifies a possible later initialization ablation, not a demonstrated cause. [Reproduction code](../../scripts/audit_hockey_rsi_reference_geometry.py) verifies source/reference hashes, with three bin-boundary tests.

External nonlinear tire and elastic brush models remain outside this comparison; see the [model limitations](../hockey_split_friction_20261028_v1/README.md).

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

The [robustness launcher](../../scripts/dispatch_hockey_contact_robustness.py) requalifies a passed nominal recording on CPU before starting its frozen 33-case suite. It rejects unfinished/failed screens, changed actors/recordings and incomplete budgets. The auxiliary code is prepared separately; it does not modify the active training snapshot.

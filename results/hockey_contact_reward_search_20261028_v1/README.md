# Hockey contact-model and reward comparison

**Sixteen nominal screens are complete, with 0 task successes.** Every completed actor stays upright, but none maintains bilateral contact over the complete reference. This includes the initial nine screens, six smoothness/planar-stop refinements, and the first three-dimensional stop-reward screen. The unequal-friction XYZ cylinder arm is in formal training; the nine-slice arm remains pending. No robustness success rate is available.

## What the result means

Contact is checked at every 2.5 ms physical sample over the 2.76 s reference, including the terminal state. Both wheels need valid native contact geometry within the unchanged 0.1 mm separation tolerance; a single missed or unknown sample fails. Contact does not depend on a normal-force threshold, equal wheel loads or an absolute body-height target. This is sampled simulation evidence, not a claim about unobserved time between steps.

The remaining gates retain approach/entry speed, XY tracking, body/wheel heading, tilt, stopping and at least 0.05 s of corroborated lateral sliding. Sliding requires both contacts, tangential-velocity and displacement evidence, near-capacity aggregate friction and dissipative opposing forces. This is not a pointwise friction-cone certificate or an observed PhysX stick/slip flag. [Frozen independent qualifier](source_snapshot_v3/qualify_hockey_contact_search.py), [task/sliding gates](source_snapshot_v3/analyze_hockey_bilateral.py).

The frozen **stop-hold** gate means 0.5 s of bilateral contact, XY speed at most 0.1 m/s and tilt at most 30°. It does not certify that vertical motion has settled. Failed-contact tracking and sliding metrics below remain diagnostic and cannot override the primary failure. A nominal pass requires a separate 33-case initial-state robustness suite before completion can be considered.

## Completed results

Friction is static/dynamic. “Interruption” means at least one wheel is missing; “Neither” is the subset with both contacts missing. `smooth` changes only the capped angular penalty to pseudo-Huber; `smooth_stop` additionally replaces the endpoint planar stop reward with geometric contact and speed/tilt scoring at both ends of all eight physical substeps.

| Contact model / friction | Recipe | Interruption (ms) | Neither (ms) | XY RMSE (m) | XY stop hold (s) | Angular RMS (rad/s²) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cylinder 0.7/0.7 | dense | 7.5 | 0.0 | 0.0287 | 0.5000 | 110.22 |
| Cylinder 0.7/0.7 | smooth | 2.5 | 0.0 | 0.0329 | 0.4825 | 78.95 |
| Cylinder 0.7/0.7 | smooth_stop | 147.5 | 27.5 | 0.0233 | 0.5000 | 81.67 |
| Cylinder 0.7/0.7 | smooth_stop_xyz | 212.5 | 62.5 | 0.0225 | 0.2625 | 90.43 |
| Cylinder 0.9/0.7 | dense | 65.0 | 17.5 | 0.0406 | 0.4425 | 87.20 |
| Cylinder 0.9/0.7 | smooth | 57.5 | 25.0 | 0.0388 | 0.5000 | 85.49 |
| Cylinder 0.9/0.7 | smooth_stop | 100.0 | 10.0 | 0.0277 | 0.4250 | 88.81 |
| Nine slices 0.9/0.7 | dense | 115.0 | 10.0 | 0.0349 | 0.4450 | 101.64 |
| Nine slices 0.9/0.7 | smooth | 57.5 | 2.5 | 0.0267 | 0.4750 | 68.79 |
| Nine slices 0.9/0.7 | smooth_stop | 142.5 | 60.0 | 0.0320 | 0.5000 | 80.58 |

All ten rows fail the task. Angular RMS covers the 2.78 s recording; task/contact gates cover the 2.76 s reference. These are single-seed matched screens, not statistical robustness estimates. Cross-model comparisons also change the training geometry or friction ratio.

The closest contact result remains the equal-friction `smooth` actor: 2.5 ms right-wheel loss at 0.6250–0.6275 s, reaching 0.124 mm native separation with zero interval-averaged normal force. It also misses the planar stop gate. [Aligned force check](refinement_failure_audit_v1/near_gap_force_check.json).

Adding the substep stop reward increases contact interruption in all three models in this sweep. It improves stop hold for the equal cylinder and nine-slice actors, while the unequal cylinder speeds up again during the last 0.5 s and ends at 0.1535 m/s in XY. The final nine-slice actor has 142.5 ms interruption, including 60 ms with neither wheel touching; it passes the other gates. Better stopping or lower angular RMS therefore does not establish successful contact control.

[Six-screen summary](refinement_summary_v6/report.json) · [contact-event audit](refinement_failure_audit_v6/report.json) · [recordings](refinement_publication_v6/manifest.json) · [frozen-formula replay](refinement_reward_audit_v7/report.json).

The initial nine-screen sweep also included stronger `tight` contact penalties and a multi-term `load_free` recipe. None passed; `tight` worsened continuity relative to `dense` in all three models. The complete parent controls and all initial results remain in the [initial summary](final_summary_v2/report.json), [CSV](final_summary_v2/completed_cases.csv), [failure audit](final_failure_audit_v2/report.json) and [recordings](publication_v2/manifest.json). The legacy recipe name `load_free` is not a contact definition; unequal loads are allowed throughout.

## Fixed-follow animations

Each animation has four horizontal panels and a fixed 2.4 m camera offset. MuJoCo supplies graphics and forward kinematics only; trajectories and contacts come from recorded PhysX. No poses are corrected, smoothed or interpolated. Native geometry is aligned to the pre-step pose; force is averaged over the following physical step. New renders label planar speed `vXY` and vertical speed `vZ` separately; older renders' `speed` field is XY speed.

- [Nine slices: parent / dense / smooth / smooth-stop](refinement_slices_stop_render_v1/slices_stop_fullbody_4x1.mp4), [8× contact-loss excerpt](refinement_slices_stop_render_v1/slices_stop_contact_excerpt_8x_4x1.mp4), [provenance](refinement_slices_stop_render_v1/provenance.json).
- [Equal cylinder: dense / smooth / planar-stop / XYZ-stop](xyz_equal_render_v1/equal_xyz_fullbody_4x1.mp4), [8× terminal-contact excerpt](xyz_equal_render_v1/equal_xyz_contact_excerpt_8x_4x1.mp4), [provenance](xyz_equal_render_v1/provenance.json).
- [Equal and unequal cylinders: smooth / smooth-stop pairs](refinement_stop_friction_render_v1/stop_friction_fullbody_4x1.mp4), [8× excerpt](refinement_stop_friction_render_v1/stop_friction_contact_excerpt_8x_4x1.mp4).
- [Equal cylinder: parent / dense / smooth / smooth-stop](refinement_stop_render_v1/stop_refinement_fullbody_4x1.mp4), [8× excerpt](refinement_stop_render_v1/stop_refinement_contact_excerpt_8x_4x1.mp4).
- Earlier controlled comparisons: [geometry × smoothness](refinement_geometry_render_v1/geometry_smooth_fullbody_4x1.mp4), [friction × smoothness](refinement_friction_render_v1/friction_smooth_fullbody_4x1.mp4), [equal-friction refinements](refinement_equal_render_v1/equal_smooth_fullbody_4x1.mp4).

The initial four-column reward comparisons are retained for the [unequal cylinder](native_reward_comparison_render_v1/native_rewards_fullbody_4x1.mp4), [nine slices](slices_reward_comparison_render_v1/slices_rewards_fullbody_4x1.mp4) and [equal cylinder](equal_reward_comparison_render_v1/equal_rewards_fullbody_4x1.mp4).

## Current three-dimensional stop experiment

The two planar cylinder `smooth_stop` actors end with upward COM speeds of **1.228 and 1.513 m/s**, despite low XY speed ([recorded terminal velocities](stop_xyz_preflight_v1/terminal_speed_context.json)). Position differences independently corroborate substantial terminal upward motion in the [normal-motion audit](refinement_liftoff_audit_v3/report.json). That audit does **not** support vertical-penalty saturation as the explanation: all four analyzed recordings have zero windows at a replayed vertical cost of 0.999 or higher. This differs from the previously observed angular-penalty saturation.

The [new specification](../../config/hockey_contact_stop_xyz_v1.json) changes only stop-speed scoring from XY to XYZ during the same stop phase. It retains bilateral geometric substep scoring, pseudo-Huber angular cost, numerical weights, original warm parent, DR and independent gates. There is no absolute-height target. The [v5 snapshot](source_snapshot_v5/manifest.json) contains 41 code/config/test files, 375,480 bytes; the unused v4 candidate was withdrawn to fix first-reset handling before any GPU job.

All six v3 screens failed before v5 started. [Runtime preflight](xyz_runtime_preflight_v1/report.json) confirms the first 32-environment, two-iteration Isaac smoke test completed 1,536 transitions without native-collector faults, with the XYZ formula selected. Its parent qualification is identical to v3. This validates execution, not learned task success. [29 local checks and matched-protocol audit](stop_xyz_preflight_v1/report.json), [upload receipt](upload_receipt_v5.json), [launch record](remote_evidence_v5/launch_v5.json). The offline auditor executes each snapshot's verified reward source; earlier row values remain unchanged.

The first XYZ actor completed its full 3,072,000-transition screen and fails both continuous contact and terminal stop hold. Against the matched equal-friction planar-stop actor, interruption rises from 147.5 to 212.5 ms, while XY RMSE changes from 0.02335 to 0.02249 m. The last 62.5 ms has neither wheel touching. Its terminal upward COM speed falls from 1.228 to 0.040 m/s, but this is **not a settled stop**: during the last 0.5 s the base COM rises 0.200 m, peaks at 1.352 m/s upward, and finishes with geometric wheel clearances of 41.5/32.9 mm. These clearance values come from authored wheel geometry; no native contact manifold exists at that terminal pose. Low endpoint vertical speed alone would misclassify this airborne state.

[First XYZ summary](xyz_summary_v1/report.json) · [contact-event audit](xyz_failure_audit_v1/report.json) · [unchanged recordings](xyz_publication_v1/manifest.json) · [matched normal-motion audit](xyz_normal_motion_audit_v1/report.json) · [XYZ reward replay](xyz_reward_audit_v1/report.json). The replay confirms the XYZ formula without a force-based support proxy; it is validation-only scoring, not a training-return measurement or proof of a causal mechanism. This first result does not justify generalizing to the two unfinished models.

A separate [RSI reference-phase audit](rsi_reference_audit_v1/report.json) finds that only reference frames 0, 1, 2 and 35 are bilaterally grounded. Under hypothetical uniform adaptive bins, the current 50% full-start mixture would select bilateral reference geometry 51.45% of the time. Actual bin weights/reset contacts were not archived, so this is not a measured training contact rate or a proven cause. The [reference correction diagnostic](../hockey_reference_contact_gap_20261029_v1/README.md) remains geometric only and is not used in these runs.

## Training, physics and reproduction

Each screen warm-starts the same 12,288,000-transition PPO actor/critic and normalizers, then uses a fresh optimizer for **512 × 24 × 250 = 3,072,000 new transitions**, seed 42. Learning rate is 1e-4, initial exploration standard deviation 0.1, entropy coefficient 0.001, physics 400 Hz and policy 50 Hz. These are fine-tuning screens, not cold-start or full-budget retraining comparisons.

Training randomizes dynamic wheel friction over 0.3–1.2, with static friction tied to the declared ratio. Ground friction is 1 with multiply combination; other existing randomization remains fixed. Nominal evaluation uses dynamic friction 0.7. Nine rigid axial slices retain the cylinder's outer dimensions, mass and inertia, with 1 mm contact offset and zero rest offset. They are not an elastic tire model. External nonlinear tire/brush models are outside this batch; see [model limitations](../hockey_split_friction_20261028_v1/README.md).

[Snapshot builder](../../scripts/prepare_hockey_contact_search.py) · [trainer](../../scripts/train_hockey_contact_search.py) · [reward adapter](../../scripts/hockey_contact_refinement.py) · [serial dispatcher](../../scripts/dispatch_hockey_contact_search.py) · [33-case robustness launcher](../../scripts/dispatch_hockey_contact_robustness.py).

```bash
python scripts/collect_hockey_contact_search.py --version v5 --output results/hockey_contact_reward_search_20261028_v1/remote_evidence_v5 --include-rollouts
python scripts/summarize_hockey_contact_search.py --bundle results/hockey_contact_reward_search_20261028_v1/source_snapshot_v3 --evidence results/hockey_contact_reward_search_20261028_v1/remote_evidence_v3 --run-name run_v3 --output <fresh-summary-directory>
```

For offline analysis, use the corresponding public recording subset as `--evidence`; manifests preserve raw trajectory/contact hashes. Locked references, robot assets and parent weights are required separately. No weights, licensed body-model files, credentials or source video are published here.

Remote root: `/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1`. The active run is `run_v5/status.json`, tmux `hockey_contact_reward_xyz_20261029_v5`. Jobs run serially on verified available RTX devices, old jobs remain paused, and a runtime error stops the dispatcher without automatic retries. Runtime status observations are timestamped; code/tests alone do not establish robot success.

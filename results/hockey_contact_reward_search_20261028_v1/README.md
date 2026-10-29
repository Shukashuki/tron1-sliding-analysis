# Hockey contact-model and reward screens

Status on 2026-10-29: **prepared and locally tested; no new GPU experiment started**. Automatic approval review rejected upload of the first archive because its specific private source/config payload and destination were not covered by the earlier snapshot approvals. The corrected, complete v2 payload is awaiting explicit approval. The old speed study remains paused. No new success rate is available.

The last cold-start bilateral reward produced 0/33 successful validations and fell at 0.1375 s in the nominal run. Its high training contact fraction was biased by short episodes. The earlier original-speed hockey policy is a more useful initialization: under the same current qualification it has nominal XY RMSE 0.01788 m, no 45-degree fall, and 93.48% bilateral contact, but still fails the strict contact-continuity gate. See the [independent parent requalification](../hockey_bilateral_training_20261028_v1/previous_nominal_comparison.json). Neither policy is a task success.

This is a **warm-started PPO reward screen**, not a new cold-start learning baseline. Every cell starts from the same existing actor, critic and observation normalizers. It resets the optimizer and exploration standard deviation; it does not continue the failed bilateral actor. Parent training cost is 12,288,000 transitions, followed by 3,072,000 new transitions per screen. The screen is shorter than the earlier full training budget and is not presented as an equal-budget final comparison.

| Contact configuration | Nominal static / dynamic friction | Geometry and force law |
| --- | --- | --- |
| `native_split` | 0.9 / 0.7 | Authored cylinder; native PhysX Coulomb friction |
| `slices9_split` | 0.9 / 0.7 | Nine adjacent axial cylinders; native friction, same mass/inertia and outer dimensions |
| `native_equal` | 0.7 / 0.7 | Authored cylinder; equal-coefficient control |

All arms use 1 mm contact offsets, zero rest offsets, 400 Hz physics and 50 Hz control. Runtime checks inspect every cloned wheel's native shape count and offsets. Dynamic friction retains the existing training draw U(0.3, 1.2); static friction is the same draw times the declared ratio. Ground friction remains one with multiply combination. Each wheel's slices share its material draw. No mass, actuator, body-height target, actor input or reference-speed change is introduced.

| Reward variant | Contact fraction | Contact loss | Wheel clearance | Sliding | Angular acceleration | Leg imitation | Gap scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | +2 | -4 | -2 | +0.5 | -0.05 | +1 | 10 mm |
| `tight` | +4 | -8 | -4 | +0.5 | -0.05 | +1 | 3 mm |
| `load_free` | +2 | -4 | -2 | +1 | -0.1 | +0.25 | 10 mm |

Contact terms inspect every physical substep without a load threshold or equal-load target. Clearance measures wheel-ground separation, not absolute body height. Base XY, speed, heading, wheel-axis and beta rewards retain their dense feedback even after a contact interruption. Positive sliding credit still requires a complete bilateral control window, opposing dissipative friction, near-dynamic-capacity force, braking phase and useful task tracking. The different reward variants are candidate recipes, not single-factor causal ablations.

All variants use an explicit fall cost of -8 per event: Isaac multiplies reward weights by the 0.02 s control interval, so the configured fall weight is -400. The previous -2 weight contributed only -0.04 per event. This scaling change is a hypothesis to test, not evidence that early falling has been solved. Other common fine-tuning changes are learning rate 0.0001, exploration standard deviation 0.1, entropy coefficient 0.001, full-clip reset probability 0.5, integer phase boundaries and fall-only adaptive sampling.

The prepared queue first validates the unchanged parent on each contact model, then runs a 32-environment/two-iteration smoke test, a 512-environment/250-iteration screen and a 139-control-step nominal recording. All GPU jobs are serial. A card is eligible only after repeated zero-utilization checks with no unknown process; small contexts belonging to identity-verified old jobs on another main device are allowed. The old paused main device is not eligible. The dispatcher never stops or resumes another process and stops on a failed runtime audit without automatic retries.

Qualification independently replays all 56 actor observations and actions, checks material readback and archived contact buffers, and reuses the unchanged full-horizon bilateral geometric gate (separation at most 0.1 mm, no missed samples), followed by speed, XY, heading, tilt, terminal stop and corroborated sliding gates. Training contact averages cannot pass. A nominal pass stops the screening queue for a subsequent 33-case robustness suite; it does not complete the overall research goal. If all screens fail, further learning/model work remains.

The external nonlinear tire prototype is not silently counted as a fourth tested arm. It currently uses a preceding-step normal load, lacks static adhesion, and requires its applied tangential force to be recorded separately from native friction before the hockey qualifier can use it. An elastic brush model is still unimplemented. Nine rigid slices alone do not create tread compliance or brush memory. See the [contact-model audit](../hockey_split_friction_20261028_v1/README.md).

Reproduction source: [experiment specification](../../config/hockey_contact_reward_search_v1.json), [preparation](../../scripts/prepare_hockey_contact_search.py), [trainer](../../scripts/train_hockey_contact_search.py), [reward/model adapter](../../scripts/hockey_contact_search.py), [dispatcher](../../scripts/dispatch_hockey_contact_search.py), [independent qualifier](../../scripts/qualify_hockey_contact_search.py). The existing remote reference, robot asset and parent checkpoint are reused by pinned SHA-256; no model assets or weights are uploaded.

The [v2 manifest](source_snapshot_v2/manifest.json) contains 41 files, 440,876 bytes. Manifest SHA-256: `047ffbb98b769d6b6366e8b4dde7e0eca9b685178de555729a7f2629576090e5`. Archive SHA-256: `78a3bdf2f572095f7cd12b1d1f3a84eeee970c7d3a47d3cf07bf15998bdabe50`. Destination requested: `zengxiong@14.103.233.29:/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1/source_snapshot_v2/`.

Local validation: **24 tests passed** across the new contact/warm-start counterexamples, historical bilateral collector, separate static/dynamic friction and GPU ownership eligibility. All snapshot Python files compile, and local transitive Python imports are present. Live Isaac validation of this new reward/model adapter is pending approval and execution.

A subsequent [reference/contact geometry audit](../hockey_reference_contact_gap_20261029_v1/README.md) found that the original reference uses one-wheel grounding. The stable parent's remaining failure comprises 20 ms left-wheel and 160 ms right-wheel interruptions. Small combined root-roll and vertical corrections admit bilateral reference geometry with unchanged joints, but their dynamics have not been validated. This diagnostic does not modify the v2 payload or its requested execution scope.

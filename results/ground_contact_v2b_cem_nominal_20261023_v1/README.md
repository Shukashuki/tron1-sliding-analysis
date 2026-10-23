# Actual CEM V2b nominal comparison — contact criterion failed

One deterministic episode per controller, μ=0.7, 400 Hz physics / 50 Hz control. V2b changes only the V2a temporal support aggregation from `0.75 mean + 0.25 min` to `min` over eight physical substeps. A single supporting wheel is allowed; no absolute-height or zero-β constraint is added.

| Recorded CEM result | Original | V2a | V2b |
| --- | ---: | ---: | ---: |
| Core XY RMSE (m) | 0.0850 | 0.0571 | 0.1014 |
| Core speed retention | 101.20% | 99.16% | 95.46% |
| Core reference-relative β RMSE (°) | 5.44 | 8.27 | 5.22 |
| Core 10–100 Hz angular XY RMS | 1.2995 | 0.7359 | 0.4482 |
| Any-wheel support, 0.02–6.96 s | 74.68% | 70.47% | 63.74% |
| Longest measured no-support interval (ms) | 222.5 | 257.5 | 257.5 |

V2b reduces core angular HF RMS by 65.5% relative to the original controller, but does not achieve continuous support. Its full-contact support fraction is worse than V2a, while core support improves from 58.93% to 65.37%. Whole-reference XY RMSE is 0.1226 m and peak error 0.2859 m. Only the two contact checks fail the unchanged numerical acceptance screen. This is not overall task completion.

There are zero solver failures and zero plant warnings. Median measured solve time is 1,294.6 ms, including IPC: this is an offline experiment, not real-time control.

PPO V2b formal training was reported running on GPU 3, PID 3583707, started 2026-10-23 18:16:55 UTC, with a declared budget of 12,288,000 transitions from scratch. This dated bundle contains no PPO V2b performance result. See `status.json` for provenance and limitations.

## Reproduce recorded metrics without robot assets

From the repository root:

```sh
python results/ground_contact_v2b_cem_nominal_20261023_v1/reproduce.py
```

The replay verifies artifact/helper hashes and exactly recomputes all three cases' full/core state metrics, the fixed V2b screen, and its contact-window diagnostics from the compact NPZ files. It does not step a simulator, run a policy, perform FK, or read robot/reference assets. Python numerical dependencies used by the repository are required.

All 349 × 15 actual V2b stage rewards were separately reproduced exactly from the source-bound full raw trace and solver log before packaging. Those full all-body reward histories are not in this compact bundle, so replay does not claim to reconstruct PPO returns or every stage reward. Reward definitions differ: episode-return sums are not a fair original/A/B performance comparison.

No weights, models, meshes, credentials, compiled libraries, or new rendering are included. `executed_source.tar.gz` contains only code/configuration and is not executed by replay. Extract only into a fresh isolated checkout. Original and V2a public bundles remain unchanged: [original comparison](../ppo_cem_matched_reward_400hz_20261022_v1/) and [V2a nominal](../ground_contact_v2_nominal_20261023_v1/).

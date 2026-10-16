# Nominal NMPC: block count and search budget

The H30/B10/I6 MuJoCo run reduces core XY RMSE to **0.0521 m**, with 102.05% mean speed retention and no solver failures. This is MuJoCo-only development evidence, not a qualified sim2sim controller: it is not frozen, not a friction sweep, and not a ranking.

[Five-panel top-down comparison](nmpc_nominal_block_budget_20261016_v1/topdown.png) · [Audited summary](nmpc_nominal_block_budget_20261016_v1/summary.json) · [Derived replay arrays](nmpc_nominal_block_budget_20261016_v1/development_trace.npz)

## Fixed task and observed results

All four episodes use the primary `drift_drl_traj4` reference, nominal plant friction μ=0.7 and the separately declared predictor scenario `[0.7]`. The source is CARLA simulation with reconstructed timing, not physical-car measurement. The 1.44–4.00 s evaluation core has reference planar base-COM speed approximately **1.03 m/s**; this is a relative high-speed-turn test, not a claim about a robot's maximum speed.

| Blocks / iterations | Core XY RMSE / peak (m) | Mean speed retained | Full-episode max tilt | Core XY screen |
|---|---:|---:|---:|---|
| B5 / I3 | 0.0916 / 0.1252 | 101.16% | 18.44° | Pass |
| B10 / I3 | 0.1700 / 0.3064 | 95.68% | 14.54° | Fail |
| B5 / I6 | 0.0673 / 0.1181 | 97.82% | 16.45° | Pass |
| B10 / I6 | 0.0521 / 0.0967 | 102.05% | 13.37° | Pass |

The inherited exploratory XY limits are RMSE ≤0.15 m and peak ≤0.30 m, measured against the reference at the same physical timestamps and world frame. There is no registration or time warp. All episodes finish 349 decisions / 1,397 physical samples over 6.98 s, with no fallback or failed candidate/scenario/batch prediction. No nonwheel ground contact is measured in these four episodes. Tilt and wheel contact/slip remain diagnostics; there is no absolute-height, flight, or zero-heading-offset failure gate. XY displacement is not itself evidence of lateral tire slip.

## What changed

Every effective configuration uses a 30-step / 0.60 s horizon, 64 candidates, 8 elites and seed 42. B5 and B10 give 0.12 s and 0.06 s residual blocks respectively. Three iterations use 68,054 candidate evaluations / 8,166,480 predicted physical substeps per episode; six use 135,062 / 16,207,440. The additional model-generated nominal plan is OFF in every case: there is no donor-episode budget or learned-policy initialization.

The bounded full-plan warm start, residual decoder, 15-term height-smooth objective and actuators are retained. Leg/wheel initial standard deviations are 0.04/0.35; minimum values are 0.005/0.03, with zero distribution smoothing. The raw settings preserve H15 defaults; the saved `effective_cem_config` records the actual H30 CLI overrides.

Actual source identities are preserved separately: B5/I3 uses CEM `8d2ac598…` and native wrapper `475ef6fe…`; B10/I3 and B5/I6 use `d7d50588…` / `475ef6fe…`; B10/I6 uses diagnostic CEM `7c667ef0…` / `bb19aeea…`. Full hashes are in the summary. Optional-plan-OFF and telemetry regressions pass, including decision/plan/RNG checks, but this is **not identical source** or proof of identical physical episodes. All four retain objective `ff22eebb…`, native C++ `deb3afe3…` and MuJoCo 3.8.1.

Recorded median solve times are approximately 0.97, 1.00, 2.10 and 2.04 s respectively. These are uncontrolled historical timings, **not real-time** results; all exceed the 20 ms control period. Simulation advances four 5 ms substeps per action without injecting that wall-clock delay.

## Qualification remains open

The earlier Isaac B10/I3 pass cannot be combined with this MuJoCo B10/I6 pass: these are **different controllers**. The same selected configuration and source snapshot must produce valid, passing nominal episodes in both engines before freezing it or starting a friction sweep. The diagnostic v3 remote upload remains pending approval; this bundle contains no Isaac B10/I6 result. The previous B5/I6 Isaac episode had 211 fallbacks and is not valid nominal qualification evidence.

This single-reference, single-seed 2×2 search-budget study suggests a useful next nominal candidate, not a causal contact-engine diagnosis or a statistical robustness claim. Geometric body/wheel heading offsets are features, not mismatch targets; measured contact-patch slip is reported separately. The top-down figure shows the demonstrated interval only; the compact arrays retain the full episode, including the recovery tail.

Replay the public derived data without licensed models, FK or simulation:

```sh
python -m pytest -q tests/test_nmpc_block_budget_publication.py
```

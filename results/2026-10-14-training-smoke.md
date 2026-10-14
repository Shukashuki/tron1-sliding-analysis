# BeyondMimic-style TRON1 drift training: bounded interface check

## Question and setup

Can the pinned BeyondMimic startup material / joint-position / CoM events and
interval pushes be mapped onto TRON1 WF, train one PPO iteration, and export a
checkpoint that the existing Isaac and MuJoCo evaluation interfaces accept?
This is an **interface check**, not a trained drift policy or friction-failure
experiment. The reference is the earlier `drift_drl/traj_4` high-sideslip
TRON1 motion. The new drift task profile deliberately omits the historical
worst-substep ground-load reward, whose cached mass/gravity audit only permits
the older fixed event set.

Remote setup: `zengxiong@14.103.233.29`, dedicated workspace
`/data/zengxiong/workspaces/tron1-sliding-analysis-20261014/`, RTX 4090 D
GPU 2, 2 environments, seed 42, 50 Hz policy / 200 Hz physics, one PPO
iteration. The source tree on that host is a snapshot without `.git`; the
trainer records source/module SHA-256 hashes and does not invent a commit id.

## Result

The final one-iteration smoke checkpoint was saved in
`smoke_bm_dr_gpu2_v6/model_final.pt`, with a manifest and runner config.
Live PhysX material readback over 2 environments × 9 bodies was:

| Material channel | Observed range | Configured training range |
| --- | ---: | ---: |
| Static friction | 0.3811–1.3919 | 0.3–1.6 |
| Dynamic friction | 0.4237–1.1599 | 0.3–1.2 |
| Restitution | 0.0146–0.4999 | 0–0.5 |

The checkpoint loaded into a one-environment Isaac fixed-`µ=0.70` evaluation;
the normalized TorchScript actor and policy contract were exported. The
MuJoCo evaluator reproduced the saved **policy-rate** Isaac observations and
accepted the actor/contract/model. The one-iteration actor terminated early
in MuJoCo at 0.60 s, which is expected for an untrained smoke checkpoint and
is **not** a meaningful drift-failure result. The Isaac physics-substep trace
must not be fed to the policy-rate observation verifier: doing so produced a
false row-2 mismatch, corrected by using `trajectory.npz`.

## Scope and subsequent result

This one-iteration smoke result remains an interface test only. A separate
1000-iteration training run and six-case fixed-friction sim2sim grid have since
completed; see the [formal car result](2026-10-14-bm-friction-grid.md). The
historical three-point XY figure uses a different fixed-friction-trained actor
and is labeled accordingly. The original evaluators support equal static/dynamic
coefficients up to 1.0. Higher equal-coefficient cases are supported by the
separate `materialize_wide_friction_source.py` evaluation snapshot;
off-diagonal pairs still require explicit support.

# NMPC I6: bounded reconstruction did not reproduce the prediction failure

This [sanitized diagnostic](nmpc_i6_prefix_reconstruction_20261016_v1.json)
reconstructs only the original Isaac I6 episode's planning prefix, using its
recorded measured states and previous applied actions. It starts no Isaac app
or validation plant. The original 15-file source snapshot, eight historical
helpers, native binary and compiled model are hash-locked; archived controller
code is unmodified. This is not a new controller evaluation or tuning trial.

Frames **0–136** match the available guards: raw selected and float32-applied
actions are bitwise equal, initial winners and counters match, and the largest
cost difference is 2.89×10⁻¹⁵. Cost tolerances were fixed at absolute/relative
1×10⁻¹² before execution; no tolerance was changed after seeing results.

At frame **137**, the reconstruction diverges and stops as required:

| Frame-137 diagnostic | Archived episode | Local reconstruction |
| --- | ---: | ---: |
| Finite / rejected candidates | 323 / 64 | 387 / 0 |
| Failed batch calls / total | 1 / 9 | 0 / 9 |
| Selected summed-horizon cost | 2.729990 | 2.632759 |

The reconstructed final iteration finds a lower-cost action; its maximum
normalized-action difference from the archived selection is 0.04660. Across
the bounded prefix, all 1,241 prediction calls complete without an exception:
53,405 candidates and 6,408,600 predicted physical substeps. The worker remains
alive before normal cleanup and exits normally afterward. No planning beyond
frame 137, retry, resume or new parameter search follows.

The historical failure is therefore **not reproduced**, and its cause remains
unknown. Original full warm plans, candidate scores and elite indices were not
saved: matching observable prefix evidence is not independent proof that the
historical population bytes were identical. Saved counters also do not identify
which original iteration rejected its population. These results do not prove
that the original worker died, nor explain the earlier simulator discrepancy.

An earlier preparation attempt stopped on a missing SciPy dependency before
worker construction, with zero planning; its artifacts remain untouched. The
second preparation uses aligned dependencies and is the **first actual planning
attempt** (Python 3.11.15, NumPy 1.26.0, SciPy 1.15.3, MuJoCo 3.8.1).
A contemporary runtime query is not proof of historical versions.
Public output contains aggregate metrics and hashes only, not states, action
plans, random-generator state, process identifiers or machine-specific paths.
Following the decision to stop further tuning, this closes the bounded
diagnostic; existing-controller comparisons remain separate.

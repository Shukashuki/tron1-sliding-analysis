# SA29: fixed-NMPC friction development

This is the separate `doa_sa29_scale2` S-avoidance supplement, not another
trial of the primary large-angle turn. Do not pool its statistics with
`drift_drl_traj4`. The completed local MuJoCo fixed-controller grid below is
unchanged. The remote follow-up has completed PPO and SAC training; the
matched three-controller grid is still in progress, not inferred from
the local results.

## Results

All 17 predeclared coefficients have complete recordings and zero solver
fallbacks. The nominal μ=0.7 pilot is reused byte-for-byte; the 16 new episodes
each exited 0 and passed full clock, source, action and common-score audits.
At most three local episodes ran at once, without retries or controller tuning.

The core XY screen is exceeded at **6/17** points: μ=0.01, 0.03, 0.05, 0.10,
0.15 and 1.40. The other 11 stay below both limits. This is not a monotone
friction threshold or a statistical success rate. In particular μ=1.40 fails
despite remaining upright (20.06° full maximum tilt), while 1.60 and 1.80 pass.

Speed and XY use the core; maximum tilt uses the entire 5.16 s playback.
Posture/overspeed are diagnostics, not extra criteria changing the XY column.

| μ | XY RMSE / peak [m] | Speed retained | Full maximum tilt | XY screen |
| ---: | ---: | ---: | ---: | --- |
| 0.01 | 0.2404 / 0.5154 | 109.5% | 113.69° | Exceeded |
| 0.03 | 0.2873 / 0.4476 | 94.7% | 113.54° | Exceeded |
| 0.05 | 0.3047 / 0.4646 | 92.9% | 113.75° | Exceeded |
| 0.10 | 0.2439 / 0.5706 | 109.5% | 109.17° | Exceeded |
| 0.15 | 0.7858 / 1.7219 | 139.2% | 95.57° | Exceeded |
| 0.20 | 0.1175 / 0.2430 | 99.0% | 108.88° | Below both limits |
| 0.30 | 0.0482 / 0.0808 | 101.2% | 18.80° | Below both limits |
| 0.45 | 0.0701 / 0.1353 | 100.7% | 18.92° | Below both limits |
| 0.60 | 0.0465 / 0.1217 | 100.2% | 30.16° | Below both limits |
| 0.70 | 0.1015 / 0.1602 | 99.1% | 21.73° | Below both limits |
| 0.80 | 0.0565 / 0.1135 | 99.3% | 15.80° | Below both limits |
| 1.00 | 0.0849 / 0.1584 | 96.3% | 16.80° | Below both limits |
| 1.20 | 0.0689 / 0.1096 | 98.8% | 14.74° | Below both limits |
| 1.40 | 0.1903 / 0.3283 | 97.8% | 20.06° | Exceeded |
| 1.60 | 0.0584 / 0.1102 | 100.9% | 13.03° | Below both limits |
| 1.80 | 0.0733 / 0.1633 | 97.1% | 19.33° | Below both limits |
| 2.00 | 0.0633 / 0.1012 | 99.0% | 14.73° | Below both limits |

![SA29 reference kinematics versus fixed NMPC at selected frictions](nmpc_sa29_friction_development_20261016_v1/topdown.png)

The [complete summary](nmpc_sa29_friction_development_20261016_v1/summary.json)
and [public signal archive](nmpc_sa29_friction_development_20261016_v1/development_trace.npz)
retain all 17 original clocks, measured COM velocity, heading geometry,
wheel-ground contact/slip, posture and both descriptive event windows.
Recorded optimization budgets total 855,253 candidate/scenario evaluations
and 102,630,360 requested 5 ms prediction substeps, including the reused pilot;
all 17 plant episodes contain 17,544 actual physics substeps.

### XY, contact and posture are distinct measurements

At μ=0.10 and 0.15, instantaneous XY error first exceeds 0.30 m at 1.475 s
and 1.815 s, while tilt is only 5.97° and 14.35° respectively. Tilt first
exceeds 60° at 2.065 s and 2.195 s. This is a temporal ordering, not a causal
identification or an added failure gate. At the first XY crossing, both wheels
have zero measured ground-contact count/force in these two recordings; this
single sample does not establish that an entire interval was airborne.

μ=0.20 stays below the core XY limits (0.117 / 0.243 m), despite a 92.18°
core maximum tilt and first nonwheel-ground contact at 3.050 s. Its first
0.30 m XY crossing is at 3.395 s, outside the 0–3.14 s core. The XY screen
therefore remains passed, but this is not controlled upright success.
Its net measured core COM-course change is 37.61°, versus the reference's
1.97°; retained mean speed alone is not exact direction tracking.

## Fixed experiment contract

The [separate grid declaration](../config/nmpc_sa29_friction_development.json)
binds 21 unchanged runtime sources, model, actuator contract, native library,
motion/reference and effective controller settings. It is not the earlier
one-episode pilot declaration. Fixed CEM remains H30/B10/C64/I3/E8, seed 42,
prediction μ=0.7, with the same full-body 15-term height-smooth score. Only
actual plant μ changes. The residual settings file retains H15/B5 defaults;
the effective H30/B10 values are explicit, unchanged command-line overrides.
No learned donor, model-plan initialization, adaptive selector or plant-μ
oracle is used.

Each full playback has 258 decisions / 1,033 physical samples / 5.16 s.
Evaluation keeps the source-demo core 0–3.14 s and world axes/original clocks.
The two windows 0.78–1.54 s and 1.88–3.04 s are descriptive, not extra gates.
The existing XY screen is RMSE >0.15 m OR peak >0.30 m. Speed retention,
heading offsets, contact slip, posture, height and flight do not add gates.
Synthetic recovery/tail remains visible but is not added to the core score.

COM speed uses recorded velocity, not a position derivative. Wheel-plane
geometry uses FK from measured states; actual contact-point slip remains the
recorded MuJoCo measurement. Contact eligibility requires wheel-ground count
>0 and upward net force >5 N. Ineligible zero slip is unavailable evidence,
not proof of rolling or flight; reference contacts are unmeasured.

The display coefficients are μ={0.01,0.10,0.70,1.60,2.00}; the lowest-friction
case is included to illustrate an observed failure, not to select a better
controller or exclude any of the 17 results. Thick curves mark the core;
arrows show COM velocity and ticks wheel planes. Pale ticks lack eligible
measured contact. No alignment, time warp, root correction or new retargeting.

## Reproduction and limits

Use the unchanged single-case `scripts/run_nmpc_tracking.py` with
`--reference-id doa_sa29_scale2`, matching locked input files, explicit native
backend/library and all five fixed H30/B10/C64/I3/E8 overrides. Repeat only for
the declared coefficients, preserving existing cases and reusing μ=0.7.
The [offline grid analyzer](../scripts/analyze_nmpc_sa29_friction_development.py)
accepts actual `--run MU=DIR` recordings and refuses relabeled primary data,
changed sources/controllers, a substituted nominal rerun or false completion.
It uses FK only and never steps dynamics or loads a learned policy.

The public-vector tests need no robot asset, policy weight or physics rollout:

```sh
python -m pytest -q tests/test_nmpc_sa29_friction_development.py
```

One seed and one episode per coefficient do not establish a statistical
success rate, universal friction boundary or controller-class ranking.
The simulator waits for each solve; wall-latency actuation delay is not
injected. No real-time, robust-stability, controlled-execution or hardware
claim is made. CARLA S-avoidance and reconstructed source timing do not
establish physical-car dynamic similarity.

The [completed primary three-controller paired grid](2026-10-16-primary-controller-friction-development.md)
and [original SA29 nominal pilot](2026-10-16-sa29-nmpc-pilot.md) remain unchanged.
Completing the original two-reference comparison still requires supplementary-
specific PPO/SAC training and both-engine fixed-friction validation. The user
subsequently approved the 43-file remote snapshot and full SA29 comparison.
The exact archive (339,297 bytes; SHA-256
`0065669a538f488eaaf619bc2b2c18d8d0c16cd7163d8c6788f48b2e8cba602e`)
was uploaded, extracted and all 43 member hashes verified. Existing runtime,
USD, XML, actuator contract and native-library identities passed preflight;
no new native build was performed.

### Approved follow-up: resource-scope interruption, not a policy result

On 2026-10-16, PPO started with the fixed 512-environment, seed-42,
1,000-iteration recipe on a freshly checked completely idle RTX GPU.
The actual PhysX backend and environment/robot tensors agreed on the selected
unmasked CUDA ordinal. Nevertheless, Isaac startup created approximately
346–378 MiB auxiliary CUDA contexts on each other visible GPU, including
GPUs with other users' work. Native/tensor ordinal consistency does **not**
prove memory/context exclusivity.

To respect the idle-GPU-only authorization, only this experiment's verified
process group was interrupted. All its processes and GPU contexts were
confirmed absent afterward; other users' processes were not signaled.
The last fully logged iteration was 199/1,000, accounting for 2,457,600
transitions. The exact post-interrupt collection total is unknown. There is
no `model_final.pt` and no valid completed SA29 PPO checkpoint. Despite the
interrupt, Kit returned child exit code 0 while `run_status.json` remained
stale at `training`; the outer supervisor correctly rejected completion.
This is an interrupted runtime attempt, **not** a physical XY failure or a
PPO/SAC/NMPC comparison result. Logs, partial metadata and termination evidence
are retained privately; no weights or credentials are published.

At that point, SAC and all new SA29 validation jobs remained unstarted. Continuing required either
verified single-GPU context isolation or explicit authorization for the
observed auxiliary allocations. The completed local MuJoCo grid above and
all prior primary-task recordings are unchanged; all 17 local SA29 NMPC
MuJoCo episodes can be reused without rerunning physics.

The user subsequently explicitly permitted the observed auxiliary allocations.
A fresh PPO run started on a newly checked completely idle RTX at 20:50 UTC,
with the same 512 environments, seed 42, 1,000 iterations, no setup smoke and
12,288,000 requested transitions. The interrupted first run is preserved, not
resumed or selected by performance. SAC and validation remain sequential,
with a fresh main-GPU guard before each application. The allowance concerns
auxiliary memory/context allocation, not running the main training/PhysX work
on an already busy GPU, retuning, new native builds or other users' processes.

PPO subsequently completed exactly 12,288,000 transitions / 24,000 policy
steps, with final checkpoint SHA-256
`bc9123c247ccfa2427d1ca6a42416592512e610db32e49ec8f2961769bef16fc`.
Its original manifest, status, runner and checkpoint bytes passed independent
budget/source/device checks. The μ=0.7 Isaac recording has all 1,033 physics
samples / 258 decisions / 5.16 s; it is not yet a completed sim2sim pair.
SAC subsequently completed the same 12,288,000 transitions / 24,000 policy
steps with the unchanged fixed recipe. Its final checkpoint SHA-256 is
`1620573e528cc5e0b0082153d8c2347e3731d133d700632312275b1b60773f6a`.
The original six selected training artifacts, full execution log and genuine
exit/owned-process closure passed independent checks; intermediate checkpoint
bytes have not been retrieved or claimed verified.

The archived DR `robot_mapping` sentence contains a legacy
"velocity-controlled wheels" label. Executed wheel control instead uses
direct effort commands, `tau_requested = 12 * clip(a, -1, 1)` Nm, followed by
the motor torque-speed/saturation limits; it is not a wheel-speed setpoint.
The frozen action implementation and numerical export contract establish
this behavior. The original metadata bytes are retained unchanged.

The first matching MuJoCo attempt stepped dynamics but failed at report
construction: an unconditional provenance hash read required
`training/tron1_terrain.py`, absent from the approved archive. It failed before
trace/report persistence, so its exact physical outcome is unavailable and
is not classified as an XY failure. The successful Isaac recording and the
failed MuJoCo log/receipt are preserved. An identical helper already on the
remote host (SHA-256 `dec819a3c3c526fce706fb6a5a6491388298884b32bfe73a5c331b166f976ebf`)
was copied into a new derived evaluation source; no new upload, native build,
controller or physics change was made. Historical provenance remains nine
originals / five changed outputs / twelve code files. The thirteenth code file
is separately bound as a reporting-only dependency; flat-ground and
no-visual-mesh checks ensure it is not used to augment the model.
The explicit MuJoCo-only retry used a fresh output after SAC's genuine closure,
reusing the original Isaac actor/contract/trajectory without rerunning Isaac.
Both original nominal recordings passed the offline first-pair I/O audit:
1,033 physics samples / 258 decisions / 5.16 s each, unchanged training and
source identities, verified exported observations and real clean terminal
captures. This opens the remaining declared validation cases; it is not a
performance-based selection or a completed-grid claim.

The new offline [policy-grid analyzer](../scripts/analyze_sa29_policy_friction_dynamics.py)
and [three-controller comparison adapter](../scripts/analyze_sa29_controller_comparison.py)
are prepared, with portable synthetic evidence tests. They require complete
reference-specific recordings and do not generate missing simulation results.
The comparison adapter also requires a portable `--execution-manifest` for
all 17 NMPC Isaac cases, binding the original terminal receipts and full logs
to each report, trace and solver log. Successful file parsing or process exit
alone does not establish a completed simulation recording.
No SA29 policy-grid or paired-controller result bundle has been published.

## Public capture and reproduction workflow

The new [capture utility](../scripts/sa29_capture.py) and
[portable assembler](../scripts/assemble_sa29_capture.py) remove the dependency
on private dispatch scripts. They do not replace the historical queue or
retroactively generate its receipts. Source preparation was tested on actual
local files; process/GPU tests use mocks, not additional physics episodes.

<details>
<summary>Source preparation, execution and capture-schema details</summary>

Use the existing asset/reference preparation instructions in the README and
the pinned runtime versions. Isaac needs the licensed robot USD, the locked
motion, and a **complete** BeyondMimic checkout through
`TRON1_BEYONDMIMIC_ROOT`. NMPC additionally needs the hash-matched XML, actuator
contract, existing native library and adjacent build JSON. These tools do not
download assets, install dependencies, compile native code or remove the
historical input-byte locks.

The original parameter-only [actuator contract](../config/nominal_actuator_contract.json)
is now included byte-for-byte (SHA-256 `6e105cbd...d09642f`). It contains no
weights or credentials. Its legacy export observation/task/checkpoint metadata
is retained for identity, not used as a learned controller: NMPC loads no actor,
and explicit plant-mu, prediction scenarios and the separate height-smooth task
configure the current experiment. In particular, the archived `0.18` field is
not SA29's validation friction or the fixed prediction mu.

Start from the fresh upstream source with `tron1_drift.py` hash `d5da59c7...`,
not the older historical `5eb40c9b...` patched source:

```sh
python scripts/sa29_capture.py prepare-source \
  --source-root upstream/tron1-learning-from-human \
  --output-dir outputs/sa29_capture_source
```

This creates `original_source/` with exactly the nine selected originals,
`evaluation_source/` with twelve materialized code files plus the separately
pinned reporting helper and unchanged materializer audit, and
`source_binding.json`. Do not materialize the entire upstream tree for this
locked analyzer: its historical baseline is exactly nine files.

Train with the unchanged `train_beyondmimic_style.py` and
`train_sac_tracking_isaac.py` entrypoints, the prepared **original** source,
`--reference-id doa_sa29_scale2`, locked motion/USD/task inputs, 512 environments,
seed 42 and ground friction 1.0. PPO uses 1,000 iterations, save interval 100
and `--smoke-steps 0`; SAC uses 24,000 policy steps, 10,000 warmup transitions,
500,000 replay capacity, batch 1,024, one update per step, gamma 0.99, log
interval 100 and save interval 2,400. Each collects 12,288,000 transitions.
Select a genuinely idle main RTX and use consistent unmasked CUDA and the
four explicit Kit settings for training as well as validation. The training
friction DR recipe remains unchanged; the wide grid is evaluation-only.

For each declared coefficient, provide a JSON **argv array** for one direct
evaluator, using canonical absolute input paths and a fresh output. The policy
Isaac entry is the prepared `eval_tracking_friction_isaac.py`; pass
`--motion-file`, `--asset-path`, the original `model_final.pt` and adjacent
`runner_config.json`, `--task-profile`, `--evaluation-contact-friction`,
`--output-dir`, `--num-envs 1 --seed 42 --contact-threshold-n 5.0`,
`--record-physics --verify-unmasked-device --device cuda:N --headless`, and
one `--kit_args` value containing exactly:

```text
--/renderer/activeGpu=N --/renderer/multiGpu/enabled=false --/physics/cudaDevice=N --/physics/sceneMultiGPUMode=0
```

Invoke the capture wrapper with the same algorithm/coefficient/device/output:

```sh
python scripts/sa29_capture.py run-engine \
  --algorithm PPO --engine isaac --mu 0.7 --device cuda:2 \
  --source-root /absolute/evaluation_source \
  --output-dir /absolute/new_case/isaac \
  --receipt /absolute/new_case/capture/isaac_terminal.json \
  --log /absolute/new_case/capture/isaac.log \
  --command-json /absolute/isaac_argv.json --allow-auxiliary-contexts
```

The paths/device are examples, not runnable access to private infrastructure.
Replace device `2` and every `N` consistently. Auxiliary-context consent is
explicit and does not allow sharing a busy main GPU; the guard still requires
less than 64 MiB, zero utilization and no attached compute PID. The wrapper
does not select another GPU, retry, train, use SSH or stop foreign processes.

Then capture the prepared `eval_tracking_mujoco.py` with the **same case's**
exported actor/contract and `--verify-isaac-trajectory` policy-rate trajectory,
locked motion/XML, fresh output and `--no-visual-mesh`. Use the wrapper's
`--engine mujoco --device cpu --isaac-receipt ORIGINAL --isaac-log ORIGINAL`;
it masks CUDA and verifies the actual prior Isaac evidence. NMPC capture uses
`--algorithm NMPC --engine isaac`, the prepared original nine-file source,
`run_nmpc_tracking_isaac.py`, fixed H30/B10/C64/I3/E8, singleton prediction
mu 0.7 and the existing native library. There is no automatic build path.

The wrapper observes the real child exit, full combined log, artifact hashes,
pre/post source/input hashes and owned-process/context closure. A traceback
with exit zero, missing files or an unclosed child produces an incomplete
receipt. An interrupted live child is reported for operator reconciliation,
not silently restarted or described as completed. This is trusted-capture
consistency, not cryptographic authenticity or a physical success claim.

Keep original raw states, weights, logs and receipts private. The assembler's
input schema is documented at the top of its source. Each file binding is
computed from an existing original, relative to an explicit `original_root`:

```python
from pathlib import Path
import hashlib
original_root = Path("originals").resolve()
def bind(path):
    p = Path(path).resolve()
    return {"path": str(p.relative_to(original_root)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
```

A policy plan (`kind: sa29_policy_capture_assembly_plan`) requires the three
inputs; the prepared source root/audit/hash/dependency mapping; original PPO
training files `manifest/status/checkpoint/runner_config`; SAC's same four plus
`policy_contract/actor`; and all 34 unique cases. Each case has seven original
artifact roles and both original receipt/log pairs. Rebase the prepared source
binding paths under `original_root`, but never rewrite the materializer audit.
Explicit paths also support preserving an original Isaac recording alongside
a separately documented MuJoCo-only retry. Never fill missing cases with
placeholders or generate terminal evidence from an old report.

```sh
python scripts/assemble_sa29_capture.py policy --plan actual_policy_plan.json \
  --output-dir outputs/NEW_private_policy_capture \
  --analysis-output-dir results/NEW_policy_bundle
python scripts/assemble_sa29_capture.py nmpc-execution --plan actual_nmpc_plan.json \
  --output-dir outputs/NEW_private_nmpc_capture \
  --analysis-output-dir results/NEW_joined_bundle
```

These commands print, but do not execute, the next analyzer argv. Append
`--policy-dynamics` with the completed policy bundle to the joined analyzer's
argv. NMPC assembly requires all 17 original report/trace/solver/receipt/log
sets, original source nine and the **published locked MuJoCo bundle**. It does
not accept an arbitrary newly generated MuJoCo bundle as the historical one.
Distinguish offline public-vector replay, a fresh physical rerun and byte-exact
historical raw replay: the last still requires the original private inputs.

</details>

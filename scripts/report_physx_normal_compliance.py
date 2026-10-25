"""Generate the English report from recomputed evidence; no physics or training."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shlex

import analyze_physx_normal_compliance as audit


def run(args):
    root = args.result
    data = audit.read(root/'analysis/analysis.json')
    assert data['analyzer_sha256'] == audit.sha(audit.__file__)
    assert data['collection_manifest_sha256'] == audit.verify_collection(root/'evidence')
    out = root/'report.html'
    assert not out.exists()
    rows = []
    for r, title in zip(data['rows'], audit.TITLES, strict=True):
        track = r['metrics']['windows']['core']['tracking']
        support = r['metrics']['any_wheel_ground_support']['full_contact']
        core_support = 1-r['metrics']['windows']['core']['contact']['measured_force_bearing_contact']['none']['sample_duty_fraction']
        rows.append('<tr>'+''.join('<td>'+v+'</td>' for v in (
            html.escape(title), f"{track['xy_rmse_m']*100:.2f}", f"{track['speed_retention']*100:.2f}",
            f"{track['beta_rmse_deg']:.2f}", f"{support['any_wheel_support_sample_fraction']*100:.2f}",
            f'{core_support*100:.2f}', f"{support['longest_no_support_observed_s']*1000:.1f}",
            f"{r['support_gap_diagnostics']['interruption_count']}", f"{r['peak_normal_force_n']:.0f}",
            f"{r['nominal_screen']['core_HF_ratio_to_original_same_engine_PPO']:.3f}"))+'</tr>')
    fixture_rows = []
    for b, title in zip(data['bench']['rows'], audit.TITLES, strict=True):
        p = b['profile']
        fixture_rows.append('<tr>'+''.join('<td>'+v+'</td>' for v in (
            html.escape(title), f"{p['stiffness_n_m']:.1f}", f"{p['damping_n_s_m']:.1f}",
            f"{1000*b['windows'][0]['mean_deflection_relative_to_rigid_m']:.3f}",
            f"{1000*b['windows'][1]['mean_deflection_relative_to_rigid_m']:.3f}",
            f"{b['peak_impulse_balance_error_n']:.6f}"))+'</tr>')
    launches = []
    for name in ('driver_v3_launch.json', 'material_roundtrip_v4_launch.json', 'transfer_driver_v2_launch.json'):
        launch = audit.read(root/'evidence'/name)
        launches.append(f'<p><a href="evidence/{name}">{html.escape(name)}</a></p><pre>'+html.escape(shlex.join(launch['argv']))+'</pre>')
    commands = '''python -B scripts/analyze_physx_normal_compliance.py \
  --collection results/physx_normal_compliance_20261025_v1/evidence \
  --output-dir /tmp/normal-compliance-analysis
python -B scripts/render_physx_normal_compliance.py \
  --collection results/physx_normal_compliance_20261025_v1/evidence \
  --analysis /tmp/normal-compliance-analysis/analysis.json \
  --output-dir /tmp/normal-compliance-render
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q \
  tests/test_physx_normal_compliance.py tests/test_analyze_physx_normal_compliance.py'''
    doc = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRON1: native normal-contact sensitivity in PhysX</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1450px;margin:30px auto;padding:0 20px;color:#202830}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cdd5dc;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f6}video,img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f6;padding:15px;font-size:12px}a{color:#0669a5}.note{padding:15px;border-left:4px solid #aa5226;background:#fff7ef}</style>
<h1>TRON1: native normal-contact sensitivity in PhysX</h1>
<p>The same frozen K PPO actor maintains the intended fast turn under all four contact models. Compliance slightly improves XY tracking and measured support fraction, but does not remove support interruptions. No candidate satisfies continuous any-wheel support. This is one deterministic rollout per model, with no training or parameter search.</p>
<p><a href="render/fullbody_comparison.mp4">Four horizontal full-body + world-XY panels</a> · <a href="analysis/analysis.json">Recomputed metrics and proof ledger</a> · <a href="evidence/collection_v1.manifest.json">132-file SHA-256 collection manifest</a></p>
<video controls loop playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<h2>Whole-robot result: nominal friction 0.7</h2>
<p>Core = 1.44–4.00 s. Whole support = 0.02–6.96 s. Support means max(Fz_left,Fz_right)&gt;5 N at a recorded 2.5 ms physics sample; single-wheel support is allowed. These are measured ground-normal loads, not a geometric flight classifier.</p>
<table><tr><th>Model</th><th>Core XY RMSE (cm)</th><th>Core speed retained (%)</th><th>Core beta RMSE (deg)</th><th>Whole support (%)</th><th>Core support (%)</th><th>Longest gap (ms)</th><th>Gaps</th><th>Peak single-wheel Fz (N)</th><th>HF / original PPO</th></tr>ROW_TABLE</table>
<p>All four pass the existing core/whole XY, speed, nonzero-reference beta, and high-frequency motion screens. All four fail the two continuity checks. HF is the native root angular-XY RMS in the common 10–100 Hz band, relative to the original same-engine PPO baseline; it is not normalized to the rigid K arm. Every compliant candidate has higher HF than the rigid K arm. The zero-damping model's higher support fraction accompanies an 85 ms maximum interruption. Its 9719 N whole-recording peak occurs at t=0, before the first applied policy action; its controlled/core peak is1217.51 N. The other three full-recording peaks occur during the turn. Do not interpret the9719 N initial readout as a turn impact; support fraction alone remains insufficient.</p>
<p>The rigid arm reproduces the original native physics recording byte for byte (SHA-256 <code>cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a</code>). All four share the exact initial mechanical state, original nonideal mass/COM/inertia, wheel collider frames, 1 mm contact offset, zero rest offset, PD gains 500/10, raw action mapping, 400 Hz physics and 50 Hz policy. Actor, reference and rewards are unchanged. Validation has no DR or observation noise. Closed-loop actions can subsequently differ because the physics differs.</p>
<img src="analysis/comparison.png" alt="Native fixture response, robot XY, ground loads, angular motion and support fractions">
<h2>What this model changes</h2>
<p>This study changes normal compliance, retaining native Coulomb tangential contact and the original single cylinder. It is separate from the <a href="../physx_tron1_wheel_models_20261025_v1/report.html">six-arm geometry × tangential force-law study</a> and the <a href="../ppo_k_native_slices9_20261025_v1/report.html">model-specific nine-slice PPO training study</a>. It does not add an identified rubber tire, a new nonlinear tangential law, or a deformable wheel.</p>
<p>PhysX provides an implicit normal spring/damper contact model. We select force-spring mode (accelerationSpring=False), with a rigid ground and compliant wheel materials. See the <a href="https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html">NVIDIA compliant-contact documentation</a>. The native three-column material tensor uses the negative restitution slot to expose compliant stiffness; see <a href="https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/class_px_material.html">PxMaterial</a>. Damping is authored and checked through USD; no direct native tensor damping readback is available here.</p>
<p>Predeclared profiles use m_ref = 22.27299977 / 2 = 11.13649989 kg, k=m_ref*g/delta_ref and c=2*zeta_ref*sqrt(k*m_ref). The 1 mm and 3 mm labels denote per-contact nominal parameters, not measured whole-wheel compression. These values have not been fitted to a real TRON1 tire.</p>
<h2>Loaded-wheel calibration fixture</h2>
<p>All four arms use the same half-robot-equivalent vertical mass, radius 0.127 m, width 0.05 m and ideal-cylinder inertia. XY and rotation are locked; vertical motion is free. This fixture is not the full robot and introduces no robot height constraint. Initial z=0.1271 m and vz=−0.3 m/s. A downward m_ref*g load is applied at 1–2 s and then removed. Means below use 0.75–1.00 s and 1.75–2.00 s; the undamped model is still oscillatory, so its window means are not a static stiffness identification.</p>
<table><tr><th>Model</th><th>k (N/m per contact)</th><th>c (Ns/m per contact)</th><th>Low-load mean compression vs rigid (mm)</th><th>High-load mean compression vs rigid (mm)</th><th>Peak momentum residual (N)</th></tr>FIXTURE_TABLE</table>
<p>All completed fixtures pass the unchanged 0.1 N vertical impulse-balance tolerance and non-tensile normal-force check. Momentum balance includes both normal and tangential contact-force components in world Z. The compliant manifold response differs from the nominal single-contact deflection; no contact-point count is inferred from this ratio.</p>
<h2>Material integration and preserved negative diagnostics</h2>
<p>A three-column native material write followed by restoration of the original negative stiffness did not reproduce the original compliant fixture response: maximum root-state component difference was 0.3241, and normal-force difference 840.16 N. This demonstrates that restoring the displayed stiffness alone is insufficient; it does not identify which hidden material field changed. The final full-robot implementation authors complete wheel materials before physics initialization and uses read-only fixed-friction startup auditing. Its rigid arm recovers the original trajectory exactly.</p>
<p>Earlier failed diagnostics are retained under <a href="evidence/bench_v1/status.json">bench_v1</a>, <a href="evidence/rigid_diagnostic_v2/impulse_balance_failure.json">normal-only momentum check</a>, and <a href="evidence/turn_models_v1/status.json">turn_models_v1</a>. These are not pooled into the four completed model results. The corrected benchmark includes the measured tangential Z impulse; no physical parameter or 0.1 N threshold was changed to pass it.</p>
<h2>Reproduction and limits</h2>
<p>Executed source snapshots, exact driver argv, native readbacks, per-case logs and raw NPZ recordings are in <a href="evidence/collection_v1.manifest.json">evidence</a>. Models, checkpoints, neural weights, credentials and binary executables are excluded. The renderer uses MuJoCo only for forward kinematics and graphics; all displayed dynamics come from the native PhysX recording. Camera distance/world angles are fixed and only camera translation follows the root; poses and heights are not smoothed, corrected or aligned.</p>
<pre>LOCAL_COMMANDS</pre>
<p>Remote commands below are the actual completed invocations. To reproduce, use the immutable snapshots and existing environment/assets; replace run IDs and output paths with fresh directories. The original drivers refuse existing result folders and audit a fully idle GPU before each case.</p>
REMOTE_COMMANDS
<p class="note">N=1 sensitivity results. No new PPO training was performed. Continuous any-wheel support remains unmet; nominal native CPU MuJoCo qualification for a selected new model is outstanding. This study does not establish real-tire fidelity or overall task completion. The next model should be evaluated with the same controls and native load/XY/beta diagnostics, rather than chosen by support fraction alone.</p>
</html>'''
    doc = doc.replace('ROW_TABLE', '\n'.join(rows)).replace('FIXTURE_TABLE', '\n'.join(fixture_rows))
    doc = doc.replace('LOCAL_COMMANDS', html.escape(commands)).replace('REMOTE_COMMANDS', '\n'.join(launches))
    out.write_text(doc)
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--result', type=Path, required=True)
    run(p.parse_args())

"""Generate a factual English K/P report only from completed native analyses."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import analyze_ppo_ground_contact_v2p as audit


def fmt(value, multiplier=1.):
    return 'n/a' if value is None else format(value*multiplier, '.2f')


def run(args):
    root = args.result
    data = audit.read(root/'analysis/analysis.json')
    assert data['kind'] == 'PPO_K_vs_P_ungated_peak_impact_dual_engine'
    assert data['status'] == 'completed_analysis' and data['analyzer_sha256'] == audit.sha(audit.__file__)
    assert not (root/'report.html').exists()
    rows = data['rows']; assert [r['id'] for r in rows] == list(audit.ORDER)
    table, diagnostics = [], []
    for r in rows:
        m = r['metrics']; c = m['windows']['core']['tracking']; w = m['windows']['whole_reference_overlap']['tracking']
        s = m['any_wheel_ground_support']['full_contact']; f = r['force_diagnostics']
        values = (r['id'], fmt(c['xy_rmse_m'], 100.), fmt(w['xy_rmse_m'], 100.),
            fmt(c['speed_retention'], 100.), fmt(c['beta_rmse_deg']),
            f"{100*s['any_wheel_support_sample_fraction']:.2f}", f"{1000*s['longest_no_support_observed_s']:.1f}",
            f"{r['nominal_screen']['core_HF_ratio_to_original_same_engine_PPO']:.3f}",
            'PASS' if r['nominal_screen']['numeric_nominal_episode_pass'] else 'FAIL')
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
        diagnostics.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (r['id'], round(f['whole_recording_peak_n'], 2), round(f['whole_recording_peak_time_s'], 4),
             round(f['policy_controlled_peak_n'], 2), round(f['core_peak_n'], 2), ', '.join(r['nominal_screen']['failed_checks'])))+'</tr>')
    p_rows = [r for r in rows if r['variant'] == 'P']
    numeric_pass = all(r['nominal_screen']['numeric_nominal_episode_pass'] for r in p_rows)
    lead = ('P passes all original nominal numerical screens in both engines; qualitative review and broader qualification remain separate.' if numeric_pass
        else 'P does not pass the original nominal task in both engines. Improvements in individual metrics do not establish continuous support or task completion.')
    px = next(r for r in rows if r['id'] == 'P_physx')
    departure = px['metrics']['persistent_xy_departure_gt0p30m']['whole_reference_overlap']['first_persistent_event']
    nonwheel = px['metrics']['nonwheel_ground_contact_diagnostic']['persistent_events']['first_persistent_event']
    timing = ('P PhysX has persistent XY error over0.30m from'+fmt(departure['onset_s'])+'s. ' if departure else '')
    timing += ('Persistent measured nonwheel-ground contact begins at'+fmt(nonwheel['onset_s'])+'s. ' if nonwheel else '')
    timing += 'The lower core high-frequency metric must be interpreted with the failed motion and recorded full-body replay; it does not establish stable contact. Contact/tilt are diagnostics, not substitutes for the XY criterion.'
    runtime = data['completed_training_proof']['contact_runtime_audit']
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>K versus P: independent peak-impact reward</title>
<style>body{font:16px/1.55 system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#202830}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #ccd4da;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f6}video,img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#f1f4f6;padding:15px}a{color:#0669a5}</style>
<h1>K versus P: independent peak-impact contact reward</h1><p>LEAD</p><p>TIMING</p>
<p><a href="render/fullbody_comparison.mp4">Four horizontal native-pose panels</a> · <a href="analysis/analysis.json">Full native metrics and completion proof</a> · <a href="evidence/collection_manifest.json">Collected evidence SHA-256 ledger</a></p>
<video controls loop playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<h2>Matched PPO comparison at nominal friction 0.7</h2>
<p>Both actors were trained from scratch in PhysX with512 environments,1000 iterations and12,288,000 transitions, seed42 and identical initial network. Each actor is validated in PhysX and native CPU MuJoCo. N=1 per cell. Core=1.44–4.00s; whole tracking=0–6.96s; support=0.02–6.96s. No trajectory alignment, reference extrapolation or post-reset samples are used.</p>
<table><tr><th>Cell</th><th>Core XY RMSE(cm)</th><th>Whole XY RMSE(cm)</th><th>Core speed retained(%)</th><th>Core beta RMSE(deg)</th><th>Any-wheel support(%)</th><th>Longest gap(ms)</th><th>HF/original PPO</th><th>All numerical gates</th></tr>TABLE</table>
<p>Support means at least one actual upward ground-normal wheel force&gt;5N at every native2.5ms sample. The other wheel may lift. The required fraction remains100%, with zero observed support gap. HF is root angular-XY RMS in the common10–100Hz band relative to the original same-engine PPO baseline, not relative to K. No absolute-height or geometric-flight gate is introduced.</p>
<h2>The only reward change</h2><pre>excess = max(Fz/(3mg)-1,0)
new_cost = 1-exp(-max_over_8_substeps_and_2_wheels(excess**2))
raw_P = raw_K-new_cost
weighted_grounded_reward = 0.4*raw_P   # existing policy-step dt multiplication retained</pre>
<p>K's original positive impact gate, reference-beta-gated support bonus and continuous worst-substep load-margin cost all remain. The new bounded cost is ungated by tracking/support quality and uses the existing3mg threshold with coefficient1. See the <a href="../ground_contact_peak_impact_diagnostic_20261025_v1/report.html">pre-training force-window diagnostic</a>. That offline arithmetic motivated the experiment but did not predict its learning outcome.</p>
<p>The other14 reward terms, beta+2/std0.12, action-rate−0.5,400Hz physics/50Hz policy, six position targets and two wheel torques, PD500/10, observations and original rigid-cylinder model are unchanged. Original training DR remains wheel friction[0.3,1.2], encoder bias, base COM and velocity pushes; the ground coefficient is1 with multiply combine in PhysX. Validation fixes the effective pair coefficient0.7 and disables DR/noise. This experiment does not train under the separate normal-compliance intervention.</p>
<p>Strict completed P training audit: raw new-cost mean COST, positive new-cost environment windows POSITIVE. All24008 reward windows have8 actual physical samples, including the original8 setup-smoke steps. The final export includes normalization once. The completed source, initial-network, DR, actuator and export proofs were verified on the execution host; no neural weights or physical assets are included in this result bundle.</p>
<h2>Measured force peaks and failed checks</h2><table><tr><th>Cell</th><th>Full-recording peak(N)</th><th>Peak time(s)</th><th>Policy-controlled peak(N)</th><th>Core peak(N)</th><th>Failed original checks</th></tr>DIAGNOSTICS</table>
<p>The initial native readout is retained in the full-recording peak but excluded from the controlled peak. Contact-model initialization must not be mistaken for a policy-controlled turn impact. Lower force peaks alone do not certify stable support.</p>
<h2>Reproduction</h2><pre>python -B scripts/analyze_ppo_ground_contact_v2p.py --collection RESULT/evidence --k-raw-root RESULT/baseline_K --output-dir /tmp/K-P-analysis
python -B scripts/render_ppo_ground_contact_v2p.py --collection RESULT/evidence --k-raw-root RESULT/baseline_K --analysis /tmp/K-P-analysis/analysis.json --output-dir /tmp/K-P-render</pre>
<p>Restore the pinned source from the executed source archive in a fresh checkout if local source hashes differ. Existing reference and visual assets are needed only for the original metric reference and FK/graphics; the renderer does not run dynamics or infer new actions. Camera distance and world angles are fixed across cases and time; camera translation alone follows the root. Opaque original poses/heights, body/wheel/velocity headings and all8 native contact samples per50Hz frame are retained.</p>
<p>This report is a single-seed contact-reward comparison. It does not establish population robustness, real-tire fidelity, hardware readiness or overall task completion. Friction-range qualification follows nominal success rather than replacing a failed nominal result.</p></html>'''
    template = template.replace('LEAD', html.escape(lead)).replace('TIMING', html.escape(timing)).replace('TABLE', '\n'.join(table)).replace('DIAGNOSTICS', '\n'.join(diagnostics))
    template = template.replace('COST', format(runtime['mean_ungated_peak_impact_cost'], '.6g')).replace('POSITIVE', str(runtime['positive_peak_impact_environment_windows']))
    (root/'report.html').write_text(template)
    print(root/'report.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--result', type=Path, required=True)
    run(p.parse_args())

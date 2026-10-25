"""Report completed K/Q native comparisons without weakening the original task."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import analyze_ppo_ground_contact_v2q as audit


def run(args):
    root=args.result.resolve();data=audit.read(root/'analysis/analysis.json')
    assert data['kind']=='PPO_K_vs_Q_unsupported_geometry_gap_dual_engine'
    assert data['status']=='completed_analysis' and data['analyzer_sha256']==audit.sha(audit.__file__)
    verification=audit.read(root/'artifact_verification.json')
    assert verification['status']=='verified' and verification['frames_decoded']==350
    assert verification['analysis_sha256']==audit.sha(root/'analysis/analysis.json')
    rows=data['rows'];assert [r['id'] for r in rows]==list(audit.ORDER)
    table=[]
    for row in rows:
        m=row['metrics'];c=m['windows']['core']['tracking'];w=m['windows']['whole_reference_overlap']['tracking']
        s=m['any_wheel_ground_support']['full_contact'];screen=row['nominal_screen']
        values=(row['id'],f"{100*c['xy_rmse_m']:.2f}",f"{100*w['xy_rmse_m']:.2f}",
            f"{100*c['speed_retention']:.2f}",f"{c['beta_rmse_deg']:.2f}",
            f"{100*s['any_wheel_support_sample_fraction']:.2f}",f"{1000*s['longest_no_support_observed_s']:.1f}",
            f"{screen['core_HF_ratio_to_original_same_engine_PPO']:.3f}",
            'PASS' if screen['numeric_nominal_episode_pass'] else ', '.join(screen['failed_checks']))
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    passed=all(r['nominal_screen']['numeric_nominal_episode_pass'] for r in rows if r['variant']=='Q')
    lead=('Q passes the original numerical nominal gates in both engines. Qualitative review and broader qualification remain separate.' if passed else
        'Q does not pass the original nominal task in both engines. Individual metric improvements do not establish continuous support or task completion.')
    runtime=data['completed_training_proof']['contact_runtime_audit']
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K versus Q: unsupported wheel-gap reward</title>
<style>body{font:16px/1.55 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #ccd4da;text-align:right}th{background:#eef3f6}video,img{max-width:100%}pre{background:#eef3f6;padding:16px;white-space:pre-wrap}</style>
<h1>K versus Q: add a geometric signal while wheel support is absent</h1><p>LEAD</p>
<video controls loop playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<p>Four actual episodes, same policy within each training arm: K→PhysX, Q→PhysX, K→native CPU MuJoCo, Q→native CPU MuJoCo. Original opaque poses/heights, common fixed camera distance and world angles, translation-only following, identical map bounds, body/wheel/velocity headings and all native support samples are retained. Rendering runs no dynamics or inference.</p>
<h2>Matched 12,288,000-transition PPO comparison at μ=0.7</h2>
<table><tr><th>Cell</th><th>Core XY RMSE(cm)</th><th>Whole XY RMSE(cm)</th><th>Core speed(%)</th><th>Core β RMSE(°)</th><th>Any-wheel support(%)</th><th>Longest gap(ms)</th><th>HF/original PPO</th><th>Original numerical gates</th></tr>TABLE</table>
<p>Core=1.44–4s; whole XY=0–6.96s; support=0.02–6.96s. At least one wheel must exceed5N at every native2.5ms sample; the other wheel may lift. Required support fraction remains100% and longest gap0. Body β tracks the intended nonzero reference. HF is angular-XY10–100Hz RMS relative to the original same-engine PPO baseline, not K. Single-wheel lift, geometric flight and body height are not substituted for the declared task gates. N=1 per cell.</p>
<h2>Only reward change from K</h2><pre>missing_i = max(Fz_left_i,Fz_right_i) &lt;= 5 N
gap_i = max(min(authored_gap_left_i,authored_gap_right_i),0)
new_cost = max8(missing_i * (1-exp(-gap_i/0.02 m)))
raw_Q = raw_K-new_cost
grounded_reward = 0.4*raw_Q   # unchanged policy-step dt multiplication follows</pre>
<p>The20mm exponential scale is reward shaping, not a contact acceptance tolerance or absolute torso-height target. Any supporting wheel disables the new cost at that substep. The original K force bonus, positive impact gate, reference-beta gate and worst-substep continuous force margin remain unchanged. No P-style independent impact penalty is added.</p>
<p>The GPU collector reads both authored finite-cylinder/flat-floor gaps and ground-filtered normal forces after the same eight actual physical updates, before reward/reset. All512 environments' original cylinders, local frames and native offsets are verified before learning. Authored geometry is a separate input from the native contact manifold; neither native point count nor a cached separation is used as the common support signal.</p>
<p>Both actors start from the identical seed42 network with512 environments ×24 steps ×1000 iterations. Original14 other reward terms, β+2/std0.12, action-rate−0.5,400Hz physics/50Hz policy, six position targets/two wheel efforts, PD500/10, actuator limits, observations, reference and rigid cylinders remain fixed. Original friction[0.3,1.2], encoder bias, base COM and velocity-push DR are preserved. Validation fixes effectiveμ=0.7 and disables DR/noise.</p>
<p>Completed Q audit:24008 windows,192064 paired native force/geometry samples per world, mean raw new cost COST and POSITIVE positive-cost world/windows. Both completed normalized-actor exports, identical initialization, exact budget, native2793-state/349-action recordings and full-horizon/no-reset proofs were verified on the execution host. Checkpoints, neural weights, physical assets, native libraries and credentials are omitted.</p>
<p><a href="analysis/analysis.json">All metrics and failed checks</a> · <a href="artifact_verification.json">350-frame decode/camera verification</a> · <a href="evidence/collection_manifest.json">Execution evidence</a> · <a href="../ground_contact_v2q_training_20261025_v1/report.html">Dated launch and smoke</a> · <a href="../native_contact_manifold_20261025_v1/report.html">Native geometry diagnosis</a></p>
<h2>Reproduce metrics and replay</h2><pre>python -B scripts/analyze_ppo_ground_contact_v2q.py --collection RESULT/evidence --k-raw-root RESULT/baseline_K --output-dir /tmp/K-Q-analysis
python -B scripts/render_ppo_ground_contact_v2q.py --collection RESULT/evidence --k-raw-root RESULT/baseline_K --analysis /tmp/K-Q-analysis/analysis.json --output-dir /tmp/K-Q-render</pre>
<p>Restore pinned source into a fresh checkout if hashes differ. Reference data and original visual assets are needed for metrics/FK and graphics; no policy weights are needed for offline replay. This single-seed comparison does not establish population robustness, real-tire fidelity, hardware readiness or overall task completion. Friction-range qualification follows nominal success.</p></html>'''
    text=text.replace('LEAD',html.escape(lead)).replace('TABLE','\n'.join(table))
    text=text.replace('COST',format(runtime['mean_unsupported_geometry_gap_cost'],'.6g')).replace('POSITIVE',str(runtime['positive_geometry_cost_environment_windows']))
    target=root/'report.html';assert not target.exists();target.write_text(text)
    print(json.dumps(dict(report=str(target),Q_numeric_nominal_both_engines_pass=passed,task_success_claimed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--result',type=Path,required=True);run(p.parse_args())

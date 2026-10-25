"""Publish measured nominal outcomes; training completion alone is not success."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import analyze_ppo_k_nonlinear_tire as audit


def run(root):
    root=Path(root);target=root/'completed_report.html'
    assert not target.exists()
    summary=audit.read(root/'analysis/analysis.json')
    verification=audit.read(root/'artifact_verification_v1.json')
    review=audit.read(root/'still_review_v1.json')
    assert summary['analyzer_sha256']==audit.sha(audit.__file__)
    assert verification['frames_decoded']==350 and verification['fixed_relative_camera_verified']
    assert review['inspected_frame_indices']==[112,150,200,348]
    assert review['full_manual_video_playback_claimed'] is False
    for name,digest in review['frame_sha256'].items():assert audit.sha(root/'render'/name)==digest
    rows=summary['rows'];assert [r['id'] for r in rows]==list(audit.ORDER)
    table=[]
    for row in rows:
        m=row['metrics'];core=m['windows']['core'];support=m['any_wheel_ground_support']['full_contact'];screen=row['nominal_screen']
        values=(row['title'],f"{core['tracking']['xy_rmse_m']:.4f}",f"{100*core['tracking']['speed_retention']:.2f}",
            f"{core['tracking']['beta_rmse_deg']:.2f}" if core['tracking']['beta_rmse_deg'] is not None else 'unavailable',
            f"{100*support['any_wheel_support_sample_fraction']:.2f}",f"{1000*support['longest_no_support_observed_s']:.1f}",
            f"{screen['core_HF_ratio_to_original_same_engine_PPO']:.3f}",f"{core['maximum_tilt_deg']:.2f}",
            ', '.join(screen['failed_checks']) or 'none')
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    paragraphs=[]
    for row in rows[1:]:
        event=row['metrics']['nonwheel_ground_contact_diagnostic']['persistent_events']['first_persistent_event']
        collision='No persistent nonwheel-ground force above 1 N was observed.' if event is None else f"Persistent nonwheel-ground force above 1 N starts at {event['onset_s']:.4f} s."
        paragraphs.append('<p><strong>'+html.escape(row['title'])+'.</strong> '+html.escape(collision+' '+review['case_notes'][row['id']])+'</p>')
    force=rows[1]['independent_tire_audit']
    passed=all(r['nominal_screen']['numeric_nominal_episode_pass'] for r in rows[1:])
    outcome=('All three new arms meet the retained numerical episode gates. Broader task completion requires a separate final audit.'
        if passed else 'The new policy does not qualify the complete original nominal task across these validation plants. Failed gates are retained in the table; training completion does not establish stable contact.')
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nonlinear-tire PPO: completed learning and model transfers</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#202936}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #ccd4df;text-align:right}th{background:#edf2f7}video,img{max-width:100%}.note{padding:16px;background:#fff3d6}code{overflow-wrap:anywhere}</style>
<h1>Nonlinear-tire PPO: completed model-specific learning and transfers</h1>
<p class="note">OUTCOME</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<p>Four actual 6.98 s recordings at μ=0.7. Left: published K PPO trained and evaluated with native PhysX friction. The other three panels share one new, fully trained normalized actor: matched nonlinear PhysX, native-friction PhysX transfer, and native-friction CPU MuJoCo transfer. Body, wheel rolling-plane and velocity headings are shown. One common camera distance and world angles are fixed across cases and time; only translation follows the robot. Original opaque poses/heights and a shared world-XY scale are retained.</p>
<h2>Original nominal gates, unchanged</h2>
<table><tr><th>Policy / validation plant</th><th>Core XY RMSE(m)</th><th>Core speed retained(%)</th><th>Core β RMSE(°)</th><th>Any-wheel support(%)</th><th>Longest gap(ms)</th><th>HF/original PPO</th><th>Core max tilt(°)</th><th>Failed gates</th></tr>TABLE</table>
<p>Core:1.44–4 s. Support:0.02–6.96 s. Every actual 2.5 ms sample must have at least one ground-filtered wheel normal force&gt;5 N; the other wheel may lift. Required support fraction100%, longest gap0. Core/whole XY RMSE≤0.15 m and peak≤0.30 m, core speed ratio0.95–1.10, intended nonzero reference β RMSE≤10°, and core10–100 Hz angular-XY RMS≤0.7 of the original same-engine baseline. All nine gates are independently recomputed. No height/flight gate or new zero-slip target is introduced. Body tilt and nonwheel contact are diagnostics; wheel loading after falling cannot demonstrate successful turning.</p>
BODY_REVIEW
<img src="analysis/comparison.png" alt="Recorded XY, speed, wheel normal force and body tilt">
<h2>What was trained</h2>
<p>The new policy completed512×24×1000=12,288,000 transitions from the same seed-42 initial network as K. The K contact reward and other14 reward terms, β+2, action-rate−0.5,500/10 DC-motor leg PD, limits, observations, reference,400 Hz physics and50 Hz control are unchanged. Original cylinders, masses, COM and inertia remain. The new variable is wheel-ground tangential force, implemented with the batched nonlinear combined-slip curve.</p>
<p>Native wheel static/dynamic friction is zero; native normal contact remains. The original startup μ draw[0.3,1.2], shared by both wheels in each world, supplies external force capacity μFz. Encoder, COM and push randomization retain their existing settings. Parameters B=4.5,C=1.8,E=1 and0.2 m/s regularization are sensitivity assumptions, not identified TRON1 tire data. Native static adhesion, tire elasticity and a full vehicle tire model are not claimed. The external law uses the preceding physical sample's load and contact point, with a2.5 ms coupling delay. Reset worlds have zero external force on their first physical application.</p>
<h2>Independent applied-force audit</h2>
<p>The retained NumPy law recomputes every loaded GPU force and COM moment from the recorded kinematics. Maximum differences: FORCE_ERROR N and MOMENT_ERROR Nm, within the declared float32 tolerances. Native tangential friction after the initial pre-substitution buffer remains below0.001 N. No-load external forces and moments are exactly zero. Contact energy evaluated at the preceding contact point is ENERGY J.</p>
<p>CORE_CARRIED core wheel applications retain external force&gt;0.001 N while the following physical sample has wheel Fz≤5 N. This explicitly exposes delayed force/contact coupling; it is not evidence of instantaneous contact-law equivalence. Used-load aggregate utilization and left/right loaded hub angles/contact velocities are in the JSON. They do not identify pointwise friction cones or sensor-measured tire slip angles. Body β, wheel-hub angle and patch sliding remain distinct.</p>
<h2>Scope and reproduction</h2>
<p>This is one seed and one nominal episode per plant. The native PhysX-to-MuJoCo transfer changes both the engine and tangential law; it cannot isolate the solver alone. The earlier frozen-policy external-Coulomb control remains a coupling sensitivity reference; a newly trained external-Coulomb control has not been added here. No friction sweep or weight tuning was performed during validation.</p>
<p>21 training/kernel/lifecycle/export tests and9 independent force/reference/collection guards passed. Completed formal training/export and native model readbacks were verified on the execution host. The collection excludes weights, model assets, credentials and native libraries. Numeric replay reproduces the reported metrics and all350 video frames decode. Stills2.24/3.00/4.00/6.96 s were inspected; full manual video playback is not claimed. All owned training and nominal evaluation jobs are terminal.</p>
<p>An earlier12,288,000-transition attempt was rejected at the CPU/GPU export comparison and never admitted to nominal validation. Its CPU export exactly matches its final checkpoint, but the live final contact-health audit was not persisted. The corrected repetition records native health before export, compares the export with independent CPU source modules, and separately checks full-FP32 GPU error against the existing absolute1e-4 cap. Training precision, rewards, hyperparameters and task gates are unchanged. This implementation repair adds12,288,000 discarded training transitions to the accepted attempt's12,288,000; it is disclosed compute overhead, not policy selection using nominal performance. See the <a href="../ppo_k_nonlinear_tire_20261025_v1/export_diagnostic_report.html">first attempt and numerical diagnosis</a>.</p>
<p><a href="analysis/analysis.json">All metrics and applied-force analysis</a> · <a href="completed_evidence_v1/formal_v1_strict_verification.json">Formal training/export proof</a> · <a href="completed_evidence_v1/nominal_v1/status.json">Three completed episodes</a> · <a href="completed_evidence_v1.manifest.json">Completed evidence hashes</a> · <a href="artifact_verification_v1.json">Video/camera/numerical verification</a> · <a href="still_review_v1.json">Inspected stills and notes</a>.</p>
<p><a href="../../config/ppo_k_nonlinear_tire_400hz_v1.json">Protocol</a> · <a href="../../scripts/train_ppo_k_nonlinear_tire.py">Trainer</a> · <a href="../../scripts/physx_batched_tire.py">Batched runtime</a> · <a href="../../scripts/analyze_ppo_k_nonlinear_tire.py">Read-only analyzer</a> · <a href="../../scripts/render_ppo_k_nonlinear_tire.py">Recorded-pose renderer</a>. Exact existing-environment commands are preserved in the collected launch records. Earlier <a href="report.html">dated startup evidence</a> and <a href="../ppo_k_native_slices9_20261025_v1/report.html">separate nine-slice training</a> remain unchanged.</p></html>'''
    values={'OUTCOME':html.escape(outcome),'TABLE':'\n'.join(table),'BODY_REVIEW':'\n'.join(paragraphs),
        'FORCE_ERROR':f"{force['force_max_abs_difference_n']:.7g}",'MOMENT_ERROR':f"{force['moment_max_abs_difference_nm']:.7g}",
        'ENERGY':f"{force['contact_energy_at_preceding_point_j']:.5g}",
        'CORE_CARRIED':str(force['core_wheel_applications_with_force_gt0p001N_and_next_Fz_le5N'])}
    for key,value in values.items():text=text.replace(key,value)
    target.write_text(text)
    print(json.dumps(dict(report=str(target),new_numeric_all_plants_pass=passed,task_success_claimed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--result',type=Path,required=True);run(p.parse_args().result)

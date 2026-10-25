"""Factual report: loaded wheels and low HF can coexist with a fallen robot."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import analyze_physx_leg_pd_probe as audit


def run(args):
    root=args.result.resolve();data=audit.read(root/'analysis/analysis.json')
    assert data['status']=='completed_analysis' and data['analyzer_sha256']==audit.sha(audit.__file__)
    assert audit.read(root/'artifact_verification.json')['frames_decoded']==350
    rows=data['rows'];assert [r['id'] for r in rows]==list(audit.PROFILES)
    table=[];servo=[]
    for row in rows:
        m=row['metrics'];c=m['windows']['core'];s=m['any_wheel_ground_support']['full_contact'];n=row['nominal_screen'];d=row['servo_diagnostics']
        values=(row['id'],f"{100*c['tracking']['xy_rmse_m']:.2f}",f"{100*c['tracking']['speed_retention']:.2f}",
            f"{c['tracking']['beta_rmse_deg']:.2f}",f"{100*s['any_wheel_support_sample_fraction']:.2f}",
            f"{1000*s['longest_no_support_observed_s']:.1f}",f"{n['core_HF_ratio_to_original_same_engine_PPO']:.3f}",
            f"{c['maximum_tilt_deg']:.2f}",', '.join(n['failed_checks']))
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
        values=(row['id'],f"{d['core_position_error_RMS_rad']:.4f}",f"{d['core_computed_effort_RMS_nm']:.2f}",
            f"{d['core_applied_effort_RMS_nm']:.2f}",f"{100*d['core_clipped_motor_sample_fraction']:.2f}",
            f"{row['force_diagnostics']['core_peak_n']:.2f}")
        servo.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    low=rows[1];event=low['metrics']['nonwheel_ground_contact_diagnostic']['persistent_events']['first_persistent_event']
    assert event is not None and low['metrics']['windows']['core']['maximum_tilt_deg']>90.
    assert not low['nominal_screen']['numeric_nominal_episode_pass']
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frozen K: explicit leg PD transfer</title>
<style>body{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}td,th{border:1px solid #ccd4da;padding:8px;text-align:right}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#eef3f6}video,img{max-width:100%;height:auto}pre{background:#eef3f6;padding:16px;white-space:pre-wrap}</style>
<h1>Explicit leg PD500/10 →40/2.5: a failed frozen-policy transfer</h1>
<p>The lower-gain execution has a higher wheel-force support fraction and lower core high-frequency angular motion, but the robot overturns and loses the path. It does not solve stable high-speed turning. K at500/10 remains the reference; no friction sweep or gain-specific PPO training was launched.</p>
<video controls loop playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<p>Two actual PhysX episodes at effectiveμ=0.7, same K normalized actor, same initial recorded mechanical state,400Hz physics/50Hz policy, original cylinders, masses, native contacts, motor torque/speed limits, reference and rewards. The same policy operates in closed loop, so its actions respond to different states. No physical simulation or policy inference is run by the native-pose renderer.</p>
<h2>Original task criteria, unchanged</h2><table><tr><th>Execution</th><th>Core XY RMSE(cm)</th><th>Core speed retained(%)</th><th>Core β RMSE(°)</th><th>Any-wheel support(%)</th><th>Longest gap(ms)</th><th>HF/original PPO</th><th>Core maximum tilt(°)</th><th>Failed original checks</th></tr>TABLE</table>
<p>Core=1.44–4s; support=0.02–6.96s. Wheel support remains at least one ground-filtered normal force&gt;5N at every actual2.5ms sample, with required fraction100% and maximum gap0. The other wheel may lift. β follows the intended nonzero reference. HF uses root angular-XY10–100Hz RMS relative to original same-engine PPO. Body height and geometric flight are not automatic failure gates.</p>
<p>The40/2.5 case has persistent nonwheel-ground force&gt;1N from EVENTs through the endpoint. Maximum core tilt exceeds90°. In inspected frames it is upright at2.24s and overturned at3.00,4.00 and6.96s. The99.17% wheel-force fraction includes continued wheel loading after the body has fallen. Likewise, lower full-core HF and motor effort cannot establish smoother successful turning. XY, speed, β and original support criteria fail independently of any height/flight shortcut.</p>
<h2>Measured actuator response</h2><table><tr><th>Execution</th><th>Core position error RMS(rad)</th><th>Core computed effort RMS(Nm)</th><th>Core applied effort RMS(Nm)</th><th>Core clipped motor samples(%)</th><th>Core peak wheel Fz(N)</th></tr>SERVO</table>
<img src="analysis/comparison.png" alt="Native XY, normal force, root angular motion and applied leg torque">
<p>Every actual DC-motor computation is recorded before its physics step: measured state, desired position/velocity, feedforward effort, gains, computed and clipped effort. Both arms contain2792 consecutive action/motor pairs. Independent PD reconstruction has zero maximum difference in the saved float32 records; applied effort matches the native post-step torque field exactly. The baseline observer reproduces the original K physics NPZ byte for byte.</p>
<pre>computed effort = Kp*(q_target-q) + Kd*(dq_target-dq) + feedforward
applied effort = original DC-motor torque/speed clipping(computed effort)
legs:80Nm continuous/stall,15rad/s; wheels:12Nm continuous/stall,100rad/s</pre>
<p>Initialization and wrapper-reset motor callbacks retain500/10 and are recorded separately. The selected gains activate on the first actual physical policy action. Each valid episode had one initialization compute callback and2792 recorded physical motor computations. No implicit solver drive is added. Runtime contracts explicitly distinguish training500/10 from execution40/2.5.</p>
<p>The numbers40/2.5 were checked against the <a href="https://github.com/limxdynamics/tron1-rl-isaaclab/blob/main/exts/bipedal_locomotion/bipedal_locomotion/assets/config/wheelfoot_cfg.py">official TRON1 wheel-foot configuration</a> (file blob014f37917b2bd5db8ad92afdcb478bb31e5381db). That source uses ImplicitActuatorCfg; this test retains our DC motor and its original limits. It is not a reproduction of the official controller or a test of a policy trained with those gains.</p>
<h2>Scope and reproducibility</h2><p>No PPO update, reward revision, contact-geometry change or pose interpolation was made. Both rollouts use PhysX, so this is an actuator-execution transfer diagnostic. Source snapshots and logs from two earlier setup-only failures are preserved separately; they are not task trajectories. The valid snapshot fixes initialization callback handling and CUDA tensor joint-index normalization.</p>
<p>All350 video frames were decoded; four saved stills were inspected. Camera distance/world angles are fixed across cases and time; only translation follows the root. Opaque original poses/heights, body/wheel/velocity headings, common XY bounds and native support dots remain. Full manual video playback and task completion are not claimed. N=1 per arm.</p>
<pre>python -B scripts/analyze_physx_leg_pd_probe.py --collection RESULT/evidence --reference RESULT/reference.npz --output-dir /tmp/leg-PD-analysis
python -B scripts/render_physx_leg_pd_probe.py --collection RESULT/evidence --reference RESULT/reference.npz --analysis /tmp/leg-PD-analysis/analysis.json --output-dir /tmp/leg-PD-render</pre>
<p>The repository checkout and original visual assets are needed for FK/graphics; existing Isaac/PhysX environments and original K actor are needed only to repeat dynamics. Neural weights, physical model files, native binaries and credentials are omitted from this evidence collection. All owned remote probe jobs have ended.</p>
<p><a href="analysis/analysis.json">Full native metrics and failed checks</a> · <a href="evidence/collection_manifest.json">Raw evidence SHA-256 ledger</a> · <a href="artifact_verification.json">Video/camera verification</a> · <a href="../physx_leg_command_probe_20261025_v1/report.html">Previous inference-only target-ramp probe</a> · <a href="../ground_contact_v2q_ppo_nominal_20261025_v1/report.html">Previous Q reward failure</a></p></html>'''
    text=text.replace('TABLE','\n'.join(table)).replace('SERVO','\n'.join(servo)).replace('EVENT',f"{event['onset_s']:.4f}")
    target=root/'report.html';assert not target.exists();target.write_text(text);print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--result',type=Path,required=True);run(p.parse_args())

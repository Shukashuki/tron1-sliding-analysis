"""Publish the matched task-space K/M contact-margin experiment."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import analyze_contact_taskspace_margin as analysis


def run(root):
    root = Path(root)
    assert not (root/"report.html").exists()
    d = analysis.read(root/"analysis/analysis.json")
    v = analysis.read(root/"artifact_verification_v1.json")
    review = analysis.read(root/"still_review_v1.json")
    assert v["frames_decoded"] == 350 and v["fixed_relative_camera_verified"]
    assert v["analysis_sha256"] == analysis.sha(root/"analysis/analysis.json")
    assert review["inspected_frame_indices"] == [112, 150, 200, 348]
    assert not review["full_manual_video_playback_claimed"]
    for n,h in review["frame_sha256"].items():
        assert analysis.sha(root/"render"/n) == h
    assert analysis.read(root/"final_process_audit_v1.json")["all_owned_processes_terminal"]
    assert "15 passed" in (root/"tests_v2.log").read_text()
    table = []
    for row in d["rows"]:
        c = row["metrics"]["windows"]["core"]
        s = row["metrics"]["any_wheel_ground_support"]["full_contact"]
        screen = row["nominal_screen"]
        values = [row["title"], f'{c["tracking"]["xy_rmse_m"]:.4f}', f'{c["tracking"]["speed_retention"]:.3f}',
                  f'{c["tracking"]["beta_rmse_deg"]:.2f}', f'{s["any_wheel_support_sample_fraction"]*100:.2f}',
                  f'{s["longest_no_support_observed_s"]*1000:.1f}', f'{screen["core_HF_ratio_to_original_same_engine_PPO"]:.3f}',
                  ", ".join(screen["failed_checks"]) or "none"]
        table.append("<tr>"+"".join("<td>"+html.escape(x)+"</td>" for x in values)+"</tr>")
    passed = all(row["nominal_screen"]["numeric_nominal_episode_pass"] for row in d["rows"] if row["variant"] == "margin4")
    outcome = ("Both new episodes pass the numerical gates; consult the independent full-body review before claiming task qualification."
               if passed else "The stronger-contact actor does not pass the complete nominal task across both engines. Better individual metrics do not establish stable-contact turning. No new friction sweep is launched.")
    notes = "".join("<p><b>"+html.escape(k)+".</b> "+html.escape(x)+"</p>" for k,x in review["case_notes"].items())
    text = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Task-space K/M contact-margin comparison</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}video,img{max-width:100%}.outcome{background:#fff1d2;padding:16px}</style>
<h1>Same task-space targets, stronger any-wheel load-margin penalty</h1>
<p class="outcome">OUTCOME</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<table><tr><th>Cell</th><th>Core XY RMSE (m)</th><th>Speed/reference</th><th>Beta RMSE (deg)</th><th>Any-wheel support (%)</th><th>Max gap (ms)</th><th>HF/original</th><th>Failed original checks</th></tr>TABLE</table>
<p>Core: 1.44–4.00 s. Support: 0.02–6.96 s, actual measured upward wheel/ground normal force strictly above 5 N at every 2.5 ms sample. Either wheel may support; bilateral contact is not required. Retained gates: core/whole XY RMSE ≤0.15 m and peak ≤0.30 m; core speed ratio 0.95–1.10; intended nonzero beta RMSE ≤10 degrees; core angular-XY 10–100 Hz RMS ≤0.7 of original same-engine PPO; support 100% and maximum gap zero. No absolute-height or geometric-flight failure gate is added. Body integrity and within-turn speed also matter; bearing after falling and lower chatter after stopping do not establish success. Single-seed episodes are not robustness qualification.</p>
REVIEW<img src="analysis/comparison.png" alt="Actual paths, speed, wheel-ground force and body tilt">
<h2>What the additional native diagnostics establish</h2>
<p>The total-normal-load residual RMS, normalized by the same audited 218.50 N robot weight, decreases from 1.362 to 0.856 in PhysX and 1.725 to 1.243 in MuJoCo. Root angular HF nevertheless increases. The named-leg-only velocity audit finds 10–100 Hz RMS increasing from 1.545 to 2.248 rad/s in PhysX and 1.667 to 2.238 rad/s in MuJoCo; wheel spin is excluded using the actual joint names. These observations do not establish causal attribution. A prospective total-load Huber statistic is computed offline only, and neither a load-balance nor a leg-velocity reward was installed in this experiment.</p>
<p><a href="normal_load_diagnostic/diagnosis.json">Total-load statistics</a> · <a href="leg_motion_diagnostic_v1.json">Named physical leg-motion audit and Parseval verification</a>.</p>
<img src="normal_load_diagnostic/normal_load.png" alt="Actual total wheel normal load with identical time and force scales">
<h2>Only one contact coefficient changes</h2>
<p>Both arms use the same free-intermediate-leg task-space definitions: base and wheel-center XY position/linear velocity, base angular velocity and both wheels' world-referenced axial plane headings. Root orientation, speed, yaw-rate, nonzero beta, free absolute height and observations are retained. The four task-space definitions are identical between these two actors; they are not an additional intervention in this comparison.</p>
<pre>F_any[s] = max(Fz_left[s], Fz_right[s], 0)
margin[s] = clip((0.5*m*g - F_any[s]) / (0.5*m*g - 5 N), 0, 1)
C = max over the 8 actual physical samples of margin[s]
r_contact = 0.4 * (G_gated_support_bonus - lambda * C)
K: lambda = 1; M: lambda = 4</pre>
<p>The positive support/tracking/reference-beta bonus G, force sampling, force threshold and margin mapping are unchanged. The expression is shown before the standard reward-manager multiplication by the 0.02 s policy interval. The external contact weight remains +0.4, all 15 reward weights and other 14 reward definitions stay unchanged. No new height target, motor gain, action filter, geometry, tire law or observation is introduced.</p>
<h2>Matched training and native validation</h2>
<p>The control is the completed task-space K actor from the preceding experiment. Each arm starts independently from the same original 12,288,000-transition K checkpoint 8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a. Actor, critic and normalizers are bit-identical at the start. Fresh Adam moments, learning rate 0.0001, clip 0.1 and 512×24×200 = 2,457,600 additional transitions are identical, giving each resulting policy 14,745,600 cumulative transitions. M does not resume the resulting task-space K actor. Runner configuration, task profiles, tracking source/parameters, asset/model/reference hashes, weights, initial physical states and original friction/encoder/COM/push DR are independently compared.</p>
<p>Training uses native PhysX, original held position targets and PD 500/10 with original motor torque/speed limits. Startup wheel-friction DR remains uniform [0.3,1.2]. Each actor is validated in native PhysX and native CPU MuJoCo at fixed mu 0.7 without DR/noise or hidden resets. One actor is used across both engines. Closed-loop engine transfer changes future observations/actions and is not a solver-only error measurement.</p>
<h2>Verification and reproduction</h2>
<p>The native smoke and its strict independent CPU checkpoint/export check passed before formal training. The full M refinement and final CPU check completed. Fifteen CPU tests pass, including rejection of relaxed support gates and wheel-spin-invariant heading tests. An unexecuted first source draft and failed CPU test log are retained: that draft expected the wrong exception type for an already rejected invalid setting. Only the test expectation changed before the executed archive; no production code, experiment parameter, training outcome or validation result was used for that revision.</p>
<p>All 350 video frames decode. Original numerical traces, policy identities, physical clocks and common fixed relative camera are independently verified. Stills at 2.24/3.00/4.00/6.96 s were inspected; full manual playback is not claimed. The 4×1 replay uses opaque actual native poses, fixed world angles, translation-only following at one common camera distance and shared world-XY scale. No pose, height or contact correction is applied. All owned compute is terminal. Weights, physical assets, credentials and native libraries are excluded from the evidence collection.</p>
<p><a href="analysis/analysis.json">Full metrics and policy proofs</a> · <a href="completed_evidence_v1.manifest.json">Collected evidence hashes</a> · <a href="artifact_verification_v1.json">Replay verification</a> · <a href="still_review_v1.json">Still review</a> · <a href="body_integrity_diagnostic_v1.json">Native body contact, initial sample separated</a> · <a href="tests_v2.log">CPU tests</a>.</p>
<p><a href="../../config/contact_taskspace_margin_400hz_v1.json">Protocol</a> · <a href="../../scripts/train_contact_taskspace_margin.py">Trainer</a> · <a href="../../scripts/verify_contact_taskspace_margin.py">Strict verifier</a> · <a href="../../scripts/eval_contact_taskspace_margin.py">Native evaluator</a> · <a href="../../scripts/analyze_contact_taskspace_margin.py">Analyzer</a> · <a href="../../scripts/render_contact_taskspace_margin.py">Renderer</a>. Existing dependency/asset locations and exact invocation arguments remain in collected launch records.</p></html>'''
    (root/"report.html").write_text(text.replace("OUTCOME",html.escape(outcome)).replace("TABLE","".join(table)).replace("REVIEW",notes))
    print(json.dumps({"report":str(root/"report.html"),"new_numeric_both_engines_pass":passed,"task_success_claimed":False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--result", type=Path, required=True)
    run(p.parse_args().result)

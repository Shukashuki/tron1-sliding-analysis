"""Publish the complete M/R negative result after verified native replay."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import analyze_contact_taskspace_leg_velocity as analysis


def run(root):
    root=Path(root);assert not (root/"report.html").exists()
    d=analysis.read(root/"analysis/analysis.json")
    verified=analysis.read(root/"artifact_verification_v1.json")
    review=analysis.read(root/"still_review_v1.json")
    assert verified["frames_decoded"]==350 and verified["fixed_relative_camera_verified"]
    assert verified["analysis_sha256"]==analysis.sha(root/"analysis/analysis.json")
    assert review["inspected_frame_indices"]==[112,150,200,348]
    for n,h in review["frame_sha256"].items():assert analysis.sha(root/"render"/n)==h
    assert not review["full_manual_video_playback_claimed"]
    assert analysis.read(root/"final_process_audit_v1.json")["all_owned_processes_terminal"]
    assert "63 passed" in (root/"tests_v1.log").read_text()
    table=[]
    for row in d["rows"]:
        c=row["metrics"]["windows"]["core"];s=row["metrics"]["any_wheel_ground_support"]["full_contact"];screen=row["nominal_screen"]
        values=[row["title"],f'{c["tracking"]["xy_rmse_m"]:.4f}',f'{c["tracking"]["speed_retention"]:.3f}',
                f'{c["tracking"]["beta_rmse_deg"]:.2f}',f'{s["any_wheel_support_sample_fraction"]*100:.2f}',
                f'{s["longest_no_support_observed_s"]*1000:.1f}',f'{screen["core_HF_ratio_to_original_same_engine_PPO"]:.3f}',
                ", ".join(screen["failed_checks"]) or "none"]
        table.append("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in values)+"</tr>")
    passing=all(r["nominal_screen"]["numeric_nominal_episode_pass"] for r in d["rows"] if r["variant"]=="leg_velocity")
    outcome=("Both R numerical episodes pass; separate whole-body and task qualification remains required."
             if passing else "R reduces the measured root high-frequency score but fails path and intended beta tracking, then collapses. It does not complete the stable-contact turning task; no new friction sweep follows.")
    notes="".join("<p><b>"+html.escape(k)+".</b> "+html.escape(v)+"</p>" for k,v in review["case_notes"].items())
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Actual leg-velocity regularization: M/R comparison</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}video,img{max-width:100%}.outcome{background:#fff1d2;padding:16px}</style>
<h1>M versus R: actual physical leg-velocity regularization</h1><p class="outcome">OUTCOME</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<table><tr><th>Cell</th><th>Core XY RMSE (m)</th><th>Mean speed/reference</th><th>Beta RMSE (deg)</th><th>Any-wheel support (%)</th><th>Max gap (ms)</th><th>HF/original</th><th>Failed retained checks</th></tr>TABLE</table>
<p>Core 1.44–4.00 s; support 0.02–6.96 s. All original nine numerical checks remain: core/whole XY RMSE ≤0.15 m and peak ≤0.30 m, core mean-speed ratio 0.95–1.10, intended nonzero beta RMSE ≤10 degrees, core angular-XY10–100Hz RMS ≤0.7 of original same-engine PPO, any-wheel ground-normal force strictly above5N at every actual2.5ms sample, support100% and maximum gap0. One wheel may lift. No absolute-height or geometric-flight failure gate is added. Full-body integrity and the within-turn speed profile matter: a passing mean speed or lower HF while losing the path, collapsing or bearing load after collapse does not establish success.</p>
REVIEW<img src="analysis/comparison.png" alt="Actual native path, speed, wheel load and body tilt">
<p>The native body-contact audit records the first post-initial non-wheel load above5N at3.7925s in PhysX and3.745s in MuJoCo. Whole-overlap peak tilt reaches69.25/71.12degrees. The raw PhysX t=0 force reading is retained separately. Root height is displayed as a measurement, not used as an absolute-height criterion.</p>
<h2>One added cost; unchanged execution</h2>
<pre>raw_R = raw_M - 0.1 * mean over8 physical samples and6 named legs of (dq_leg/(1rad/s))^2
reward-manager contribution = 0.02s * 0.4 * raw_R</pre>
<p>The added cost is ungated and uses actual joint velocities after each native scene update. Both wheel joints are excluded by their names, including when native joint order differs. This penalizes motion amplitude as well as chatter; it is not a frequency-selective filter. No filter changes actions, observations or physics. The frozen M contact formula, continuous load-margin coefficient4, force threshold, positive reference-beta support bonus, all15 reward weights and other14 reward definitions stay unchanged. The four free-intermediate-leg task-space objectives, beta+2, action-rate−0.5, free absolute height, held targets, PD500/10, motor limits, rigid cylinders and original DR are identical across the pair.</p>
<p>R subclasses the immutable M lifecycle. The same single owner records eight synchronized force/velocity samples, rejects incomplete physical windows and preserves reset handling. Duplicate scene refreshes and repeated reward reads add no samples or statistics. Invalid leg measurements poison the environment reward and latch a health failure. The composite cost is applied once per50Hz reward window, not eight times. Native startup/final velocity sample counts and finite-health flags are verified before export.</p>
<h2>Matched warm-start scope</h2>
<p>The control is the completed task-space M actor, not the original shorter-budget K policy. Both M and R independently load the same original K checkpoint8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a with bit-identical actor, critic and normalizers, retain their12,288,000 initial normalization samples, reset Adam moments and use fixed learning-rate0.0001/clip0.1. Each adds512×24×200=2,457,600 transitions, giving14,745,600 cumulative transitions. This is conservative warm adaptation, not from-scratch training. The result does not establish that a fresh policy with these objectives cannot learn the task.</p>
<p>Both actors are PhysX-trained. Each uses one fixed actor across native PhysX and native CPU MuJoCo validation atmu0.7, no DR/noise, identical observable initial states and a full6.98s horizon without hidden resets. Closed-loop transfer includes changing observations and actions; no solver-only attribution is claimed. Training frictionDR[0.3,1.2], encoder bias, COM perturbations and pushes retain the frozen recipe.</p>
<h2>Verification and reproduction</h2>
<p>63 CPU tests and38 subtests pass, including reordered wheel-spin exclusion, non-endpoint velocity samples, reset isolation, duplicate reads, invalid-state propagation and NumPy/Torch/runtime parity. The native smoke passed before the complete refinement; both exported checkpoints passed independent CPU tensor/normalizer equality and FP32 export checks. The collection rehashes actual final neural files on the execution host but excludes weights, assets, credentials and native libraries.</p>
<p>All350 video frames decode, all four native traces are recomputed and policy identities/cameras verified. Stills2.24/3.00/4.00/6.96s were inspected; full manual playback is not claimed. Actual native poses use opaque meshes, one common fixed relative camera distance, fixed world angles and translation-only following, with shared-scale world XY. No pose/height/contact correction is applied. All owned compute is terminal.</p>
<p><a href="analysis/analysis.json">Metrics and policy proofs</a> · <a href="completed_evidence_v1.manifest.json">Collected evidence hashes</a> · <a href="body_integrity_diagnostic_v1.json">Whole-horizon body contact</a> · <a href="artifact_verification_v1.json">Replay verification</a> · <a href="still_review_v1.json">Still review</a> · <a href="tests_v1.log">CPU tests</a>.</p>
<p><a href="../../config/contact_taskspace_leg_velocity_400hz_v1.json">Protocol</a> · <a href="../../scripts/ground_contact_reward_v2r.py">Arithmetic</a> · <a href="../../scripts/ppo_ground_contact_v2r.py">Native collector</a> · <a href="../../scripts/train_contact_taskspace_leg_velocity.py">Trainer</a> · <a href="../../scripts/verify_contact_taskspace_leg_velocity.py">Strict verifier</a> · <a href="../../scripts/eval_contact_taskspace_leg_velocity.py">Native evaluator</a> · <a href="../../scripts/analyze_contact_taskspace_leg_velocity.py">Analyzer</a> · <a href="../../scripts/render_contact_taskspace_leg_velocity.py">Renderer</a>. Exact existing dependency/asset paths and invocation arguments remain in the collected launch records.</p></html>'''
    (root/"report.html").write_text(text.replace("OUTCOME",html.escape(outcome)).replace("TABLE","".join(table)).replace("REVIEW",notes))
    print(json.dumps({"report":str(root/"report.html"),"new_numeric_both_engines_pass":passing,"task_success_claimed":False}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument("--result",type=Path,required=True)
    run(p.parse_args().result)

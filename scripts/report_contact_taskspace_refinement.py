"""Publish matched control/task-space results after native replay verification."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import analyze_contact_taskspace_refinement as analysis


def run(root):
    root = Path(root)
    target = root/"report.html"
    assert not target.exists()
    summary = analysis.read(root/"analysis/analysis.json")
    verified = analysis.read(root/"artifact_verification_v1.json")
    review = analysis.read(root/"still_review_v1.json")
    assert verified["frames_decoded"] == 350 and verified["fixed_relative_camera_verified"]
    assert verified["analysis_sha256"] == analysis.sha(root/"analysis/analysis.json")
    assert review["inspected_frame_indices"] == [112, 150, 200, 348]
    assert not review["full_manual_video_playback_claimed"]
    for name, digest in review["frame_sha256"].items():
        assert analysis.sha(root/"render"/name) == digest
    assert analysis.read(root/"final_process_audit_v1.json")["all_owned_processes_terminal"]
    assert "14 passed" in (root/"tests_v2.log").read_text()
    table = []
    for row in summary["rows"]:
        core = row["metrics"]["windows"]["core"]
        support = row["metrics"]["any_wheel_ground_support"]["full_contact"]
        screen = row["nominal_screen"]
        values = [row["title"], f'{core["tracking"]["xy_rmse_m"]:.4f}',
                  f'{core["tracking"]["speed_retention"]:.3f}', f'{core["tracking"]["beta_rmse_deg"]:.2f}',
                  f'{support["any_wheel_support_sample_fraction"]*100:.2f}',
                  f'{support["longest_no_support_observed_s"]*1000:.1f}',
                  f'{screen["core_HF_ratio_to_original_same_engine_PPO"]:.3f}',
                  ", ".join(screen["failed_checks"]) or "none"]
        table.append("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in values)+"</tr>")
    passed = all(row["nominal_screen"]["numeric_nominal_episode_pass"] for row in summary["rows"] if row["variant"] == "taskspace")
    outcome = ("Both new episodes pass the retained numerical gates; inspect the separate full-body review before task qualification."
               if passed else "The new task-space actor does not pass the complete nominal task in both engines. The stable-contact goal remains unmet; no new friction sweep is launched.")
    notes = "".join("<p><b>"+html.escape(k)+".</b> "+html.escape(v)+"</p>" for k,v in review["case_notes"].items())
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>K control versus free leg-link task-space tracking</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}img,video{max-width:100%}.outcome{background:#fff1d2;padding:16px}</style>
<h1>Release artificial intermediate-leg targets: matched warm-start ablation</h1>
<p class="outcome">OUTCOME</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<table><tr><th>Cell</th><th>Core XY RMSE (m)</th><th>Speed/reference</th><th>Beta RMSE (deg)</th><th>Any-wheel support (%)</th><th>Max gap (ms)</th><th>HF/original</th><th>Failed retained checks</th></tr>TABLE</table>
<p>Core 1.44–4.00 s; support 0.02–6.96 s. At least one wheel must exceed 5 N upward measured ground-normal force at every actual 2.5 ms sample. Single-wheel lift is allowed. The nine unchanged checks require core/whole XY RMSE ≤0.15 m and peak ≤0.30 m, core speed ratio 0.95–1.10, intended nonzero beta RMSE ≤10 degrees, core angular-XY 10–100 Hz RMS ≤0.7 of the original same-engine PPO, support 100% and maximum gap zero. No absolute-height or geometric-flight failure gate is introduced. Whole-body integrity and the within-turn speed profile also matter. Lower chatter after collapse or loading the wheels after a fall does not establish success.</p>
REVIEW<img src="analysis/comparison.png" alt="Native XY, speed, wheel load and body tilt">
<h2>Hypothesis and exact reward changes</h2>
<p>The locked 349-frame car-derived reference keeps all six leg joint positions constant and all six reference leg velocities zero. The old reward nevertheless tracks nine body positions/linear velocities and seven non-wheel orientations/angular velocities. These are artificial intermediate-leg imitation targets. Their presence is verified; their causal role in contact chatter is a hypothesis tested here, not a claim that the reference is physically infeasible.</p>
<p>The new actor retains base and both wheel-center XY position/linear-velocity objectives, restricts body angular-velocity tracking to the base, and replaces the non-wheel body-orientation objective with both wheels' world-referenced rolling-plane heading objective. Heading uses the authored local-Y axle, is invariant to wheel axial rotation and compares the plane modulo pi. It is not a zero-slip objective. Root global orientation, speed, yaw-rate and directed nonzero beta targets remain active. The heading scale is the original orientation scale 0.4 rad. Singular horizontal axle projections receive zero heading reward; nonfinite states remain nonfinite and are rejected.</p>
<p>Exactly four tracking definitions change; all 15 reward weights and the K physical-substep contact formula stay unchanged. The other eleven reward terms, original held position commands, PD 500/10, torque/speed limits, rigid cylinders, observations, clocks and original friction/encoder/COM/push DR remain unchanged. This is a task-space reward ablation, not a new tire-law experiment, stronger contact weight, motor-gain change or action filter. Validation records native physics without applying a training reward owner.</p>
<h2>Matched comparison and provenance</h2>
<p>The control is the completed warm K arm, not the original shorter-budget K policy. Both arms start from original K checkpoint 8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a with bit-identical actor/critic/normalizers, fresh Adam moments, fixed learning rate 0.0001 and clip 0.1. Both add 512×24×200 = 2,457,600 transitions; each resulting actor has 14,745,600 cumulative transitions. Runner configuration, initial state, reference, weights, contact settings and retained DR/acceptance are independently compared. Each variant uses one actor in both engines at fixed mu 0.7 without validation DR/noise. One-seed closed-loop transfer includes changing observations and actions; it does not isolate solver-only error.</p>
<p>The first setup revision incorrectly required a specific parameter-list order for the original leg names. It failed during configuration before environment construction or training; its logs, source and launch record are retained separately. The repaired guard validates the original unique name set, matching upstream membership selection. Reward targets, PPO settings and budgets were unchanged. The revised native smoke, strict CPU checkpoint/export verification and complete formal refinement passed. Fourteen CPU tests pass, including wheel-spin invariance and nonfinite-state handling. No failed setup is relabeled as training.</p>
<p>All 350 video frames decode, all four native metrics are independently recomputed and the common fixed relative camera is verified. Saved stills at 2.24, 3.00, 4.00 and 6.96 s were inspected. Full manual video playback is not claimed. Replays use original native poses with opaque meshes, translation-only following at one fixed common distance, fixed world angles and common-scale world XY. No pose, height or contact correction is applied. All owned jobs are terminal. Neural weights, physical assets, credentials and native libraries are excluded from the published evidence.</p>
<p><a href="analysis/analysis.json">Metrics and exact policy identities</a> · <a href="input_target_audit_v2.json">Source/reference target audit</a> · <a href="completed_evidence_v1.manifest.json">Collected file hashes</a> · <a href="artifact_verification_v1.json">Numerical/replay verification</a> · <a href="still_review_v1.json">Still review</a> · <a href="tests_v2.log">CPU tests</a> · <a href="../contact_taskspace_refinement_20261025_v1/failure_note.json">Preserved setup failure</a>.</p>
<p><a href="../../config/contact_taskspace_refinement_400hz_v1.json">Protocol</a> · <a href="../../scripts/car_taskspace_tracking.py">Four tracking definitions</a> · <a href="../../scripts/train_contact_taskspace_refinement.py">Trainer</a> · <a href="../../scripts/verify_contact_taskspace_refinement.py">Strict verifier</a> · <a href="../../scripts/eval_contact_taskspace_refinement.py">Native evaluator</a> · <a href="../../scripts/analyze_contact_taskspace_refinement.py">Analyzer</a> · <a href="../../scripts/render_contact_taskspace_refinement.py">Renderer</a>. Exact existing dependency/asset locations and invocation arguments remain in the collected launch records.</p></html>'''
    target.write_text(document.replace("OUTCOME", html.escape(outcome)).replace("TABLE", "".join(table)).replace("REVIEW", notes))
    print(json.dumps({"report": str(target), "new_numeric_both_engines_pass": passed, "task_success_claimed": False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--result", type=Path, required=True)
    run(p.parse_args().result)

"""Publish the completed paired warm-start experiment from verified evidence."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import analyze_contact_warm_refinement as analysis


def run(root):
    root = Path(root)
    target = root / "report.html"
    assert not target.exists()
    summary = analysis.read(root / "analysis/analysis.json")
    verified = analysis.read(root / "artifact_verification_v1.json")
    review = analysis.read(root / "still_review_v1.json")
    terminal = analysis.read(root / "final_process_audit_v1.json")
    assert verified["frames_decoded"] == 350
    assert verified["analysis_sha256"] == analysis.sha(root / "analysis/analysis.json")
    assert verified["fixed_relative_camera_verified"]
    assert terminal["all_owned_processes_terminal"]
    assert review["inspected_frame_indices"] == [112, 150, 200, 348]
    for name, digest in review["frame_sha256"].items():
        assert analysis.sha(root / "render" / name) == digest
    assert not review["full_manual_video_playback_claimed"]
    assert "10 passed" in (root / "tests_v1.log").read_text()
    rows = []
    for r in summary["rows"]:
        c = r["metrics"]["windows"]["core"]
        s = r["metrics"]["any_wheel_ground_support"]["full_contact"]
        n = r["nominal_screen"]
        values = [r["title"], f'{c["tracking"]["xy_rmse_m"]:.4f}',
                  f'{c["tracking"]["speed_retention"]:.3f}',
                  f'{c["tracking"]["beta_rmse_deg"]:.2f}',
                  f'{100*s["any_wheel_support_sample_fraction"]:.2f}',
                  f'{1000*s["longest_no_support_observed_s"]:.1f}',
                  f'{n["core_HF_ratio_to_original_same_engine_PPO"]:.3f}',
                  ", ".join(n["failed_checks"])]
        rows.append("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in values) + "</tr>")
    notes = "".join("<p><b>" + html.escape(k) + ".</b> " + html.escape(v) + "</p>"
                    for k, v in review["case_notes"].items())
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Paired K/M warm refinement: nominal failure audit</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}td,th{border:1px solid #ccd4df;padding:8px}table{border-collapse:collapse}img,video{max-width:100%}.outcome{background:#fff1d2;padding:16px}</style>
<h1>Paired contact-reward refinement from the same K checkpoint</h1>
<p class="outcome">Neither resulting actor passes the complete nominal task across both engines. Stronger M refinement falls in PhysX; its better MuJoCo episode still contains support gaps. No friction sweep follows these failed nominal checks.</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<table><tr><th>Cell</th><th>Core XY RMSE (m)</th><th>Speed/reference</th><th>Beta RMSE (deg)</th><th>Any-wheel support (%)</th><th>Max gap (ms)</th><th>HF/original</th><th>Failed retained gates</th></tr>TABLE</table>
<p>Core: 1.44–4.00 s. Support: 0.02–6.96 s, max wheel ground-normal force strictly above 5 N at each actual 2.5 ms sample. Single-wheel support is allowed. The nine retained checks also cover whole-trajectory XY error. Core/whole XY RMSE must be ≤0.15 m and peak ≤0.30 m; speed ratio 0.95–1.10; intended nonzero beta RMSE ≤10 degrees; core angular-XY 10–100 Hz RMS ≤0.7 of the original same-engine PPO; support 100% and gap zero. No absolute-height or geometric-airborne gate is imposed. Bearing load after falling does not establish recovery.</p>
REVIEW
<img src="analysis/comparison.png" alt="Native XY paths, speed, actual wheel load and body tilt">
<h2>Controlled warm-start design</h2>
<p>Both arms load the identical original K actor, critic and both observation normalizers from checkpoint 8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a. Both use fresh Adam moments, fixed learning rate 0.0001 and clip 0.1. Each adds 512×24×200 = 2,457,600 transitions to the original 12,288,000, so each resulting policy has 14,745,600 cumulative transitions. These are warm refinements, not from-scratch or total-budget-matched comparisons to the earlier 12M policies. Smoke transitions are recorded separately.</p>
<p>K retains its contact reward. M changes only the continuous any-wheel load-margin coefficient from 1 to 4; the external contact weight stays +0.4. The other 14 rewards, beta +2, action-rate −0.5, held leg commands, PD 500/10, motor limits, native rigid cylinders, reference, observations and original friction/encoder/COM/push randomization remain unchanged. There is no command ramp or gain change. Validation uses fixed mu 0.7 without DR or observation noise. Each arm uses the same actor in both engines; choosing one actor per evaluation engine would not establish a transferable policy.</p>
<p>The M PhysX trace first records persistent non-wheel ground force at 2.70 s (confirmed at 2.80 s). Its lower high-frequency score accompanies collapse and slowdown. The M MuJoCo episode stays upright in inspected stills, tracks the nonzero beta and meets the HF threshold, but still has 50 ms support gaps. K refinement retains tracking while failing the support and HF checks in both engines. These are single-seed episodes, not robustness qualification or solver-only attribution.</p>
<h2>Verification and reproduction</h2>
<p>The source/config archives, both native smokes, both full warm refinements, strict CPU policy exports, and all four nominal recordings are retained by hash. Actor/critic/normalizer load identity, cumulative normalization counts and actual physical force histories are verified. Ten CPU tests pass. All 350 video frames decode; numerical metrics and the common fixed camera were independently checked. Stills at 2.24, 3.00, 4.00 and 6.96 s were inspected; full manual playback is not claimed. Rendering uses saved native poses, opaque meshes, a common fixed 2.771 m relative camera and fixed world angles with translation-only following. No pose correction, height edits or contact inference are applied. All owned jobs are terminal.</p>
<p><a href="analysis/analysis.json">Full metrics and training proofs</a> · <a href="completed_evidence_v1.manifest.json">67 collected evidence hashes</a> · <a href="artifact_verification_v1.json">Replay verification</a> · <a href="still_review_v1.json">Still review</a> · <a href="final_process_audit_v1.json">Terminal process audit</a> · <a href="tests_v1.log">Tests</a>.</p>
<p><a href="../../config/contact_warm_refinement_400hz_v1.json">Protocol</a> · <a href="../../scripts/train_contact_warm_refinement.py">Trainer</a> · <a href="../../scripts/eval_contact_warm_refinement.py">Evaluator</a> · <a href="../../scripts/analyze_contact_warm_refinement.py">Analyzer</a> · <a href="../../scripts/render_contact_warm_refinement.py">Renderer</a>. Exact remote invocation arguments and existing asset paths remain in the collected launch records. Neural weights, model assets, credentials and native libraries are excluded from the published evidence.</p></html>'''
    target.write_text(document.replace("TABLE", "".join(rows)).replace("REVIEW", notes))
    print(json.dumps({"report": str(target), "task_success_claimed": False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--result", type=Path, required=True)
    run(p.parse_args().result)

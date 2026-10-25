"""Publish completed S/T native results with full support and body review."""
from __future__ import annotations
import argparse
import html
from pathlib import Path

import analyze_contact_duration_pair as analysis


def run(root):
    root=Path(root);assert not (root/"report.html").exists()
    d=analysis.read(root/"analysis/analysis.json");v=analysis.read(root/"artifact_verification_v1.json")
    review=analysis.read(root/"still_review_v1.json");camber=analysis.read(root/"camber_tracking_diagnostic_v1.json")
    assert v["frames_decoded"]==350 and v["native_numerical_recomputation_matches"] and v["formal_policy_identity_verified"]
    assert v["analysis_sha256"]==analysis.sha(root/"analysis/analysis.json")
    assert review["inspected_frame_indices"]==[112,150,200,348]
    for n,h in review["frame_sha256"].items():assert analysis.sha(root/"render"/n)==h
    assert analysis.read(root/"final_process_audit_v1.json")["all_owned_processes_terminal"]
    assert analysis.read(root/"CPU_checks_v3.json")["isolated_archive_tests_passed"]==23
    qualified=[variant for variant in ["S","T"] if all(r["nominal_screen"]["numeric_nominal_episode_pass"] for r in d["rows"] if r["variant"]==variant)]
    outcome=("Numerical nominal passes in both engines: "+", ".join(qualified)+". Whole-body/contact/camber review remains separately required."
        if qualified else "Neither S nor T passes the complete nominal screen in both engines. The task remains unfinished;no new friction sweep is launched.")
    table=[];details=[]
    for row in d["rows"]:
        c=row["metrics"]["windows"]["core"]["tracking"];s=row["metrics"]["any_wheel_ground_support"]["full_contact"]
        values=[row["title"],f'{c["xy_rmse_m"]:.4f}',f'{c["speed_retention"]:.4f}',f'{c["beta_rmse_deg"]:.2f}',
            f'{s["any_wheel_support_sample_fraction"]*100:.2f}',f'{s["longest_no_support_observed_s"]*1000:.1f}',
            f'{row["nominal_screen"]["core_HF_ratio_to_original_same_engine_PPO"]:.3f}',str(len(row["nominal_screen"]["failed_checks"]))]
        table.append("<tr>"+"".join("<td>"+html.escape(x)+"</td>" for x in values)+"</tr>")
        z=camber["rows"][row["id"]];integrity=z["body_integrity"]
        details.append("<p><b>"+html.escape(row["id"])+".</b> Failed gates: "+html.escape(", ".join(row["nominal_screen"]["failed_checks"]) or "none")+
            ". Maximum whole-episode body tilt: "+f'{integrity["max_whole_overlap_tilt_deg"]:.2f} deg'+
            ";postinitial nonwheel-contact samples above 5 N: "+str(integrity["postinitial_nonwheel_force_over5N_samples"])+".</p>")
        details.append("<p>Loaded camber target RMSE, left/right: "+" / ".join(f'{z["camber"][side]["loaded_camber_target_RMSE_deg"]:.2f}' for side in ["left","right"])+
            " deg. Loaded-point lateral RMS median, left/right: "+" / ".join(f'{z["loaded_point_slip"]["wheels"][side]["core_lateral_point_RMS_m_s_median"]:.4f}' for side in ["left","right"])+" m/s.</p>")
    notes="".join("<p><b>"+html.escape(k)+".</b> "+html.escape(n)+"</p>" for k,n in review["case_notes"].items())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Matched support duration PPO</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}video,img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}.status{background:#fff1d2;padding:14px}pre{white-space:pre-wrap}</style>
<h1>Worst-plus-mean versus twice-worst support cost</h1>
<p class="status">OUTCOME</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<table><tr><th>Native episode</th><th>Core XY RMSE(m)</th><th>Speed/reference</th><th>Beta RMSE(deg)</th><th>Any-wheel support(%)</th><th>Max gap(ms)</th><th>HF/original</th><th>Failed gates/9</th></tr>TABLE</table>
<p>All native validation is at mu=0.7. Core tracking is 1.44–4.00 s;whole tracking is 0–6.96 s. Support is sampled every 2.5 ms over 0.02–6.96 s. One actual upward wheel-floor normal force must exceed 5 N at every sample;the other wheel may lift. Original requirements remain:core/whole XY RMSE at most 0.15 m and peak at most 0.30 m;core speed ratio 0.95–1.10;nonzero-reference-beta RMSE at most 10 deg;core angular-XY 10–100 Hz RMS at most 0.7 of original same-engine PPO;100% support and zero longest gap. Height is free. Support after collapse or lower jitter after stopping cannot qualify the task.</p>
DETAILS
REVIEW
<img src="analysis/comparison.png" alt="Native XY paths,speeds,support forces and body tilts">
<h2>Matched intervention and initialization</h2>
<pre>d_j = clip((0.5*m*g - max(Fz_left,j,Fz_right,j,0))/(0.5*m*g - 5 N),0,1)
S cost = max8(d) + mean8(d)
T cost = 2*max8(d)
raw contact reward = unchanged gated positive G - cost
external contribution = 0.4 * 0.02 s * raw contact reward</pre>
<p>Both costs range from 0 to 2 and agree for every uniform-force window. S substitutes a mean-deficit component for one worst-deficit component. This provides an arithmetic improvement when a missing substep is repaired while another gap remains. The matched experiment tests whether that credit improves native learning;the offline observation alone does not prove a PPO gradient problem. The 0.5 mg margin shapes reward only;it does not replace the 5 N acceptance rule. The held-reference-beta gate,the positive support bonus and all other 14 reward terms are unchanged.</p>
<p>Both arms independently load the same original K actor,critic and normalizers from checkpoint 8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a. They use fresh Adam,fixed learning rate 0.0001,clip 0.1,seed 42,and 512*24*200=2,457,600 additional transitions each. Cumulative count is 14,745,600. Neither arm starts from the previous camber actor. This is matched warm refinement,not scratch learning.</p>
<p>Both use the same independent signed left/right camber reference 0c8f2cfc2876bbd103bad206e708d91bf6753f8faa75c3107b0ec37c56c448b9 and spin-invariant 3D axle reward. The motion aims at 110%/90% of the original signed kinematic lateral velocity on the left/right. Original car body path,speed and yaw are retained. Reconstructed wheel-center paths,leg poses and wheel spin are included in the reference. Geometric contact in that reference does not establish loaded support. Original rigid cylinders,motor limits,leg PD,DR,400 Hz physics and 50 Hz held actions remain unchanged. Historical K has a different cost bound [0,1] and is not the matched scale control.</p>
<p>Body tracking scores use the original locked car reference da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254. The generated motion's root poses and velocities agree within 1e-12. Its regenerated FP64 base streams differ slightly from the original FP32 arrays,by at most 2.34e-7 m in position. This scoring reference is distinct from the new wheel motion actually ingested by both actors. <a href="scoring_reference_binding_v1.json">Exact binding and precision differences</a>.</p>
<h2>Camber and contact-point slip</h2>
<img src="camber_tracking_diagnostic/comparison.png" alt="Signed wheel camber and actual loaded-point lateral RMS">
<p>Native signed camber comes from reconstructed wheel local+Y axles. Targets follow the actual held policy clock:samples 1–8 use reference frame 0,9–16 frame 1. Loaded camber statistics require wheel-summed upward Fz above 5 N. Slip additionally requires at least one actual individual ground point above 5 N. Recorded lateral slip is an unsigned,normal-force-weighted per-point RMS. An unloaded zero placeholder is excluded. Ideal single-point signed reference velocity and native multi-point unsigned RMS are different estimators. No signed slip tracking claim is inferred from these records.</p>
<p>Neither actor faithfully tracks the intended opposed camber. Actual loaded camber error remains about 11–24 deg while the kinematic reference peaks near 6.95 deg. The low loaded-point slip medians with intermittent spikes also do not establish sustained reference-like lateral sliding. The contact-duration change alone has not solved the conflict between support,tracking and the generated wheel motion.</p>
<h2>Execution and verification</h2>
<p>The first source snapshot omitted its frozen K runtime dependency and failed before environment construction. Driver,continuation and smoke handles were confirmed terminal and their failed records preserved. A new immutable source snapshot includes that dependency;all 23 tests pass from its extracted isolated directory. Remote admission loads all actual runtime-source ledger entries before launching Isaac. Both native short smokes and their strict CPU checkpoint/export verifiers precede formal training.</p>
<p>Launch records contain fresh full NVML compute/graphics,memory and PCI audits. One owned GPU job runs at a time on a completely idle device. Source snapshots are rehashed throughout. Six reference tensor streams are checked after named FP32 reordering at startup and final audit. Nominal single-environment evaluation disables DR/noise and captures native terminal states before reset. Full final actor/checkpoint hashes are checked on the host;their bytes are excluded from the evidence archive.</p>
<p>The four panels replay original opaque native poses with the same fixed camera distance and relative translation. No pose smoothing,height correction,alignment,yaw follow or zoom is applied. All 350 video frames decode. Selected stills 2.24,3.00,4.00 and 6.96 s and the diagnostic plots are reviewed separately;full manual video playback is not claimed. These single-seed nominal episodes do not establish robustness.</p>
<p><a href="analysis/analysis.json">All metrics and unchanged gates</a> · <a href="camber_tracking_diagnostic_v1.json">Camber,slip and body-integrity details</a> · <a href="artifact_verification_v1.json">Artifact verification</a> · <a href="still_review_v1.json">Still review</a> · <a href="final_process_audit_v1.json">Final process audit</a> · <a href="completed_evidence_v1.manifest.json">Evidence hashes</a> · <a href="CPU_checks_v3.json">Isolated CPU tests</a> · <a href="failed_snapshot_v2_manifest.json">Preserved preflight failure</a>.</p>
<p><a href="../contact_duration_credit_20261025_v1/report.html">Historical credit audit</a> · <a href="../contact_camber_reference_20261025_v1/report.html">Previous single-scale K camber test</a> · <a href="../../scripts/train_contact_duration_pair.py">Trainer</a> · <a href="../../scripts/run_contact_duration_pair.py">Serial driver</a> · <a href="../../scripts/continue_contact_duration_pair.py">Native continuation</a> · <a href="../../scripts/analyze_contact_duration_pair.py">Analyzer</a> · <a href="../../scripts/render_contact_duration_pair.py">Renderer</a>. Exact argv and existing asset paths are retained in the collected records. No model,weight,credential or native-library bytes are uploaded or committed.</p></html>'''
    (root/"report.html").write_text(page.replace("OUTCOME",html.escape(outcome)).replace("TABLE","".join(table)).replace("DETAILS","".join(details)).replace("REVIEW",notes))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);run(p.parse_args().result)

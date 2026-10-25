"""Publish the completed matched native camber-reference experiment."""
from __future__ import annotations
import argparse
import html
from pathlib import Path

import analyze_contact_camber_reference as analysis


def run(root):
    root=Path(root);assert not (root/"report.html").exists()
    d=analysis.read(root/"analysis/analysis.json");v=analysis.read(root/"artifact_verification_v1.json")
    review=analysis.read(root/"still_review_v1.json");camber=analysis.read(root/"camber_tracking_diagnostic_v1.json")
    assert v["frames_decoded"]==350 and v["native_numerical_recomputation_matches"]
    assert v["analysis_sha256"]==analysis.sha(root/"analysis/analysis.json")
    assert review["inspected_frame_indices"]==[112,150,200,348]
    for n,h in review["frame_sha256"].items():assert analysis.sha(root/"render"/n)==h
    assert analysis.read(root/"final_process_audit_v1.json")["all_owned_processes_terminal"]
    assert analysis.read(root/"CPU_checks_v4.json")["tests_passed"]==9
    assert camber["recorded_only"] and not camber["signed_actual_contact_slip_available"]
    passed=all(r["nominal_screen"]["numeric_nominal_episode_pass"] for r in d["rows"] if r["variant"]=="camber")
    outcome=("Both new episodes pass the numerical screen;the separate body/camber/contact review is still required for task qualification."
        if passed else "The new camber-reference actor does not pass the complete nominal task across both engines.No new friction sweep is launched.")
    table=[];detail=[]
    for r in d["rows"]:
        c=r["metrics"]["windows"]["core"]["tracking"];s=r["metrics"]["any_wheel_ground_support"]["full_contact"]
        vals=[r["title"],f'{c["xy_rmse_m"]:.4f}',f'{c["speed_retention"]:.3f}',f'{c["beta_rmse_deg"]:.2f}',
            f'{s["any_wheel_support_sample_fraction"]*100:.2f}',f'{s["longest_no_support_observed_s"]*1000:.1f}',
            f'{r["nominal_screen"]["core_HF_ratio_to_original_same_engine_PPO"]:.3f}',str(len(r["nominal_screen"]["failed_checks"]))]
        table.append("<tr>"+"".join("<td>"+html.escape(x)+"</td>" for x in vals)+"</tr>")
        z=camber["rows"][r["id"]]
        for side in ["left","right"]:
            q=z["camber"][side];slip=z["loaded_point_slip"]["wheels"][side]
            fmt=lambda v:"unavailable" if v is None else f"{v:.4f}"
            vals=[r["id"],side,fmt(q["loaded_camber_target_RMSE_deg"]),fmt(slip["core_lateral_point_RMS_m_s_median"]),
                str(q["core_loaded_samples"]),str(slip["core_identified_loaded_point_samples"])]
            detail.append("<tr>"+"".join("<td>"+html.escape(x)+"</td>" for x in vals)+"</tr>")
    notes="".join("<p><b>"+html.escape(k)+".</b> "+html.escape(n)+"</p>" for k,n in review["case_notes"].items())
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Native camber-reference comparison</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}video,img{max-width:100%}.status{background:#fff1d2;padding:14px}pre{white-space:pre-wrap}</style>
<h1>Original versus independent-camber reference:same3D wheel-axis reward</h1>
<p class="status">OUTCOME</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<table><tr><th>Cell</th><th>Core XY RMSE(m)</th><th>Speed/reference</th><th>Beta RMSE(deg)</th><th>Any-wheel support(%)</th><th>Max gap(ms)</th><th>HF/original</th><th>Failed checks/9</th></tr>TABLE</table>
<p>Core1.44–4.00s;whole tracking0–6.96s;support0.02–6.96s at2.5ms native samples.At least one actual upward wheel-ground normal force must exceed5N at every sample.One wheel may lift.All original nine gates remain:core/whole XY RMSE≤0.15m,peak≤0.30m;core speed ratio0.95–1.10;nonzero-beta RMSE≤10deg;core angular-XY10–100Hz RMS≤0.7 of the original same-engine PPO;100% support and zero longest gap.No absolute-height or camber gate is added.Bearing after a fall or quieting after stopping does not establish success.</p>
REVIEW
<img src="analysis/comparison.png" alt="Native paths,speed,normal support and body tilt">
<h2>Did the actor track wheel tilt,and did loaded points slide?</h2>
<img src="camber_tracking_diagnostic/comparison.png" alt="Native signed camber against held targets,and actual loaded-point lateral RMS">
<table><tr><th>Case</th><th>Wheel</th><th>Loaded camber target RMSE(deg)</th><th>Loaded-point lateral RMS median(m/s)</th><th>Loaded wheel samples</th><th>Identified strong-point samples</th></tr>DETAIL</table>
<p>Camber is asin(worldZ dot authored wheel local+Y axle).Targets use the actual held policy clock:native samples1–8 follow frame0,9–16 frame1,and so on.No target interpolation is used.Loaded camber uses summed upward wheel normal force above5N.Contact-slip statistics additionally require an individual real ground point with normal force above5N.The recorded slip is the normal-force-weighted RMS over such material points;it is unsigned.A zero-contact placeholder is never counted as rolling.Ideal reference point velocity and actual multi-point RMS are different estimators,and their signed errors cannot be reconstructed from these records.</p>
<p>The new actor remains upright in both inspected native replays.The PhysX core XY RMSE improves from0.5676 to0.1555m,but MuJoCo worsens from0.0772 to0.1669m.Both new episodes still have52.5ms maximum support gaps.The new loaded-camber target RMSE is17.71/9.45deg(left/right)in PhysX and17.25/10.45deg in MuJoCo,against reference peak camber about6.95deg.The candidate therefore does not faithfully reproduce the intended camber trajectories.Loaded-point lateral RMS medians are0.0352/0.0050m/s in PhysX and0.0055/0.0066m/s in MuJoCo;these measurements do not establish the intended sustained large side sliding.</p>
<h2>One prospective reference intervention</h2>
<p>The control is the completed3D-axis actor9604d4e75b176cbfb4c8812f9ced900c360ccb984f5325c90be1e1e81425b9cc from the preceding experiment.The candidate starts independently from the same original K full checkpoint8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a,with identical actor,critic and normalizers,fresh Adam,fixed learning rate0.0001 and clip0.1.Both use512×24×200=2,457,600 additional transitions,14,745,600 cumulative.This is warm refinement,not scratch learning and not a continuation of the failed3D control actor.</p>
<p>All15 reward definitions and weights,contact sampling and formula,DR,observations,actuators and clocks are unchanged.The original native cylindrical wheel colliders are retained;no new geometry or tire-force law is introduced.Only the complete motion NPZ is replaced by the independent-camber kinematic reference0c8f2cfc2876bbd103bad206e708d91bf6753f8faa75c3107b0ec37c56c448b9.The new reference includes independently tilted wheel planes,reconstructed hip/knee poses and velocities,wheel-center positions and near-rolling spin.The body XY,speed,yaw and nominal reference Z remain the original car mapping;no absolute-height policy reward is added.Wheel XY targets use the newly generated positions.</p>
<p>The reference targets approximately110% of the original signed surface lateral speed on the left and90% on the right over the turn core.Peak camber is about6.95deg with opposite signs between wheels.Initial poses agree to floating-point precision;the regenerated initial wheel velocities differ by about0.00064rad/s.This is disclosed as part of the motion intervention.Native initial observable states are separately compared between the new PhysX and MuJoCo evaluations.Geometric touch in the reference never substitutes for measured support.</p>
<h2>Kinematics do not establish force feasibility</h2>
<p>The offline fully-sliding isotropic Coulomb hypothesis at mu0.7 gives a best core force residual around0.654mg for both the zero-camber and independently fitted references,despite the small kinematic lateral-velocity target error.The original path asks for only about0.0466g median horizontal COM acceleration.This calculation allows any nonnegative wheel load split but omits angular-momentum constraints,static adhesion,tire elasticity and native solver behavior.It is not a PhysX force prediction or task-impossibility proof;policies can depart from reference wheel motions while tracking the body task.</p>
<img src="conditional_force_audit/comparison.png" alt="Conditional force consistency audit,not native simulation forces">
<p><a href="conditional_force_audit/analysis.json">Conditional force assumptions and statistics</a> · <a href="../camber_kinematic_reference_20261025_v1/report.html">Reference generator and kinematic-only results</a>.</p>
<h2>Execution and verification</h2>
<p>Nine CPU tests pass.Native smoke and its independent CPU checkpoint/export proof precede formal training.All six complete reference tensor streams are compared bit-for-bit after FP32 conversion and named native reordering,both at startup and final training audit.The new motion SHA is bound in the declared contract before simulator startup,then independently checked in the trained export.Old source files and old input pins remain unchanged;the native input-contract clone explicitly records its counted motion-pin replacement.</p>
<p>The initial3/5/4-only queue was confirmed to have launched no training or evaluation child.It was closed by verified signals to those two owned idle waiters only,with before/after process identities retained.A new work directory adds GPU0 as a prospective fallback after3/5/4.All GPU launches still require full fresh NVML compute/graphics,memory and PCI auditing.One owned GPU job runs at a time;no other user's process is stopped or shared.</p>
<p>Training and validation use native400Hz physics,50Hz held actions,original leg PD500/10 and motor limits.Training retains the same friction[0.3,1.2],encoder,COM and push DR.Native nominal validation fixes mu0.7 and disables DR/noise,with one actor across both engines and no hidden resets.All350 video frames decode;original opaque native poses and common fixed relative cameras are verified.Stills2.24/3.00/4.00/6.96s and the diagnostic plots were inspected;full manual playback is not claimed.Single-seed episodes do not establish robustness.</p>
<p><a href="analysis/analysis.json">Metrics,actors and matched protocol proof</a> · <a href="camber_tracking_diagnostic_v1.json">Camber,slip and body-integrity audit</a> · <a href="completed_evidence_v1.manifest.json">Completed evidence hashes</a> · <a href="artifact_verification_v1.json">Independent replay verification</a> · <a href="still_review_v1.json">Still review</a> · <a href="final_process_audit_v1.json">Final owned-process audit</a> · <a href="queue_migration_v1.json">Preserved queue migration</a> · <a href="CPU_checks_v4.json">CPU tests</a>.</p>
<p><a href="../../config/contact_camber_reference_400hz_v1.json">Protocol</a> · <a href="../../scripts/train_contact_camber_reference.py">Trainer</a> · <a href="../../scripts/verify_contact_camber_reference.py">Strict verifier</a> · <a href="../../scripts/eval_contact_camber_reference.py">Native evaluator</a> · <a href="../../scripts/analyze_contact_camber_reference.py">Analyzer</a> · <a href="../../scripts/render_contact_camber_reference.py">Renderer</a>.Exact existing asset/source paths and argv are preserved in completed launch records.No physical assets,weights,credentials or native libraries are uploaded or included in the collection.</p></html>'''
    (root/"report.html").write_text(text.replace("OUTCOME",html.escape(outcome)).replace("TABLE","".join(table)).replace("DETAIL","".join(detail)).replace("REVIEW",notes))
    print(str(root/"report.html"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);run(p.parse_args().result)

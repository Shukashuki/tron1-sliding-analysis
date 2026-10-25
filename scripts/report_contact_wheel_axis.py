"""Publish the completed matched3D-wheel-axis reward ablation,including failure."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path


def run(root):
    root=Path(root);assert not (root/"report.html").exists()
    read=lambda p:json.loads(p.read_text())
    d=read(root/"analysis/analysis.json");v=read(root/"artifact_verification_v1.json")
    review=read(root/"still_review_v1.json");s=read(root/"source_verification_v1.json")
    assert v["frames_decoded"]==350 and v["native_numerical_recomputation_matches"]
    assert s["collected_files_rehashed"]==74 and s["executed_sources_match_current_local_files"]==43
    assert read(root/"final_process_audit_v1.json")["all_owned_processes_terminal"]
    assert review["inspected_frame_indices"]==[112,150,200,348]
    table=[]
    for row in d["rows"]:
        c=row["metrics"]["windows"]["core"]["tracking"]
        contact=row["metrics"]["any_wheel_ground_support"]["full_contact"]
        values=[row["title"],f'{c["xy_rmse_m"]:.4f}',f'{c["speed_retention"]:.3f}',f'{c["beta_rmse_deg"]:.2f}',
            f'{100*contact["any_wheel_support_sample_fraction"]:.2f}',f'{1000*contact["longest_no_support_observed_s"]:.1f}',
            f'{row["nominal_screen"]["core_HF_ratio_to_original_same_engine_PPO"]:.3f}',str(len(row["nominal_screen"]["failed_checks"]))]
        table.append("<tr>"+"".join("<td>"+html.escape(x)+"</td>" for x in values)+"</tr>")
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>3D-wheel-axis reward ablation</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}video,img{max-width:100%}.status{background:#fff1d2;padding:16px}</style>
<h1>3D wheel-axis tracking alone does not solve continuous loaded contact</h1>
<p class="status">FAIL across the required nominal engines. The new actor falls and departs in PhysX;its MuJoCo transfer remains upright and tracks the turn but still fails contact continuity and the angular-HF gate. This result is not adopted as a successful controller. No new friction sweep is started.</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/fullbody_comparison.mp4"></video>
<table><tr><th>Cell</th><th>Core XY RMSE(m)</th><th>Speed/reference</th><th>Beta RMSE(deg)</th><th>Any-wheel support(%)</th><th>Max gap(ms)</th><th>HF/original</th><th>Failed checks/9</th></tr>TABLE</table>
<p>Core1.44–4.00s;support0.02–6.96s at every2.5ms physical sample,at least one wheel's measured upward floor-normal force strictly above5N. One wheel may lift. All nine original gates remain:core/whole XY RMSE≤0.15m and peak≤0.30m;core speed ratio0.95–1.10;nonzero-beta RMSE≤10deg;angular-XY10–100Hz RMS≤0.7 of the original same-engine PPO;100% any-wheel support and zero longest gap. No absolute-height or camber acceptance gate is added.</p>
<img src="analysis/comparison.png" alt="Native paths,speed,support and body tilt">
<h2>Single matched change</h2>
<p>The control is the preceding task-space K refinement. Both actors start independently from the same original K checkpoint8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a with identical actor,critic and normalizer tensors. Both use fresh Adam,fixed learning rate0.0001,clip0.1,and512×24×200=2,457,600 additional transitions;each has14,745,600 cumulative transitions. This is warm refinement,not scratch learning.</p>
<p>Only motion_body_ori changes:projected2D axial wheel-plane heading becomes the3D world axle direction,using theta=atan2(norm(cross(axis_ref,axis)),abs(dot(axis_ref,axis))) and exp(−mean_two_wheels(theta²)/0.4²). Wheel spin and axle sign remain irrelevant. Root orientation,nonzero beta,all15 reward weights,other14 reward definitions,contact formula,free-height treatment,DR,actuators,observations and reference are unchanged. No slip reward or camber-command reference is installed in this ablation.</p>
<h2>Native failure and evidence</h2>
<p>PhysX first records postinitial nonwheel-ground force above5N at2.55s and reaches about90deg body tilt. At3.00s the body is almost sideways with wheels above it;later it is low and off the route. Lower angular HF after this fall is not stable turning. In MuJoCo,no postinitial nonwheel-ground force above5N is recorded and inspected poses remain upright,but support gaps reach55ms and HF/original0.716 exceeds0.7. The same new normalized actor9604d4e75b176cbfb4c8812f9ced900c360ccb984f5325c90be1e1e81425b9cc is used in both engines. The initial PhysX nonwheel sample is separately recorded and not counted as a postinitial collision.</p>
<p>Native smoke,strict independent CPU checkpoint/export proof,formal refinement and both full-horizon evaluations completed. Training/source provenance retains the12 recorded CPU tests. All74 collected files were rehashed;all43 executed source/config/test copies match local bytes. All350 video frames decode and the fixed relative camera is verified. Original opaque poses are selected at50Hz from2793 native400Hz states;there is no interpolation,height correction or new physics. Stills2.24/3.00/4.00/6.96s and the native metrics plot were inspected;full manual playback is not claimed. All exact owned remote process handles are terminal and final actor bytes were rehashed. No model weights,assets or credentials are collected.</p>
<p><a href="analysis/analysis.json">Metrics and matched identity checks</a> · <a href="body_integrity_and_slip_v1.json">Native nonwheel forces and loaded-point slip</a> · <a href="completed_evidence_v1.manifest.json">Collection manifest</a> · <a href="artifact_verification_v1.json">Replay verification</a> · <a href="still_review_v1.json">Still review</a> · <a href="final_process_audit_v1.json">Final process check</a> · <a href="../../config/contact_wheel_axis_400hz_v1.json">Protocol</a> · <a href="../../scripts/train_contact_wheel_axis.py">Trainer</a> · <a href="../../scripts/analyze_contact_wheel_axis.py">Analyzer</a>.</p>
<p>This single-seed result cannot prove3D orientation tracking is generally ineffective or attribute engine differences to one solver parameter. The subsequent <a href="../camber_kinematic_reference_20261025_v1/report.html">independent-camber kinematic reference generator</a> is a separate intervention,not evidence that this actor learned controlled camber or stable sliding.</p></html>'''
    (root/"report.html").write_text(text.replace("TABLE","".join(table)))
    print(str(root/"report.html"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);run(p.parse_args().result)

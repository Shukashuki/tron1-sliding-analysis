"""Publish six actual TRON1 transfers, two four-panel videos and numeric replay."""
from __future__ import annotations
import argparse
import ast
import html
import json
from pathlib import Path
import tarfile

import analyze_physx_tron1_wheel_models as audit
import render_physx_tron1_wheel_models as renderer

ROOT=Path(__file__).resolve().parents[1]


def source_ledger(executed):
    todo=[Path(__file__).resolve(),Path(audit.__file__).resolve(),Path(renderer.__file__).resolve()]
    todo += [ROOT/n for n in executed if n.endswith(".py")]
    seen={ROOT/n for n in executed if not n.endswith(".py")}
    seen.add(ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    while todo:
        path=todo.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=([v.name for v in node.names] if isinstance(node,ast.Import)
                   else [node.module] if isinstance(node,ast.ImportFrom) else [])
            for n in names:
                candidate=ROOT/"scripts"/((n or "").split(".")[0]+".py")
                if candidate.is_file() and candidate not in seen:todo.append(candidate)
    return {str(p.relative_to(ROOT)):audit.sha(p) for p in sorted(seen)}


def run(a):
    root=a.input_root.resolve();output=a.output_dir.resolve();assert not output.exists()
    evidence=root/"collected_v1";summary=json.loads((root/"analysis_v2/analysis.json").read_text())
    assert summary["analyzer_sha256"]==audit.sha(audit.__file__)
    reference=audit.base.metrics.reference_data(a.reference)
    protocol=json.loads(a.protocol.read_text())
    assert audit.sha(a.reference)==summary["reference_sha256"] and audit.sha(a.protocol)==summary["nominal_protocol_sha256"]
    rows,_,_,_,_=audit.load_cases(evidence,reference,protocol,original_metrics=summary["original_baseline_metrics"])
    assert rows==summary["rows"]
    collection=json.loads((root/"matched_models_collection_v1.manifest.json").read_text())
    assert collection["kind"]=="six_completed_matched_offset_frozen_K_PhysX_model_transfers"
    assert collection["actor_sha256"]==audit.ACTOR and collection["formal_actor_verified_on_execution_host"]
    assert len(collection["files"])==48 and collection["models_weights_credentials_native_binaries_included"] is False
    assert all(audit.sha(evidence/n)==v["sha256"] for n,v in collection["files"].items())
    renderings={}
    for panel in ("primary","coupling"):
        folder=root/(panel+"_render_v1");p=json.loads((folder/"provenance.json").read_text())
        review=json.loads((root/(panel+"_video_review_v1")/"decode_verification.json").read_text())
        assert p["renderer_sha256"]==audit.sha(renderer.__file__) and p["analyzer_sha256"]==audit.sha(audit.__file__)
        assert p["analysis_sha256"]==audit.sha(root/"analysis_v2/analysis.json") and p["actor_sha256"]==audit.ACTOR
        assert p["camera_same_distance_all_cases_frames"] and not p["camera_yaw_follow"] and not p["camera_zoom"]
        assert p["robot_mesh_alpha"]==1. and not p["smoothing_interpolation_alignment_height_correction"]
        assert p["video_frames"]==review["frame_count"]==350 and p["video_size_pixels"]==[2560,850]
        assert review["video_sha256"]==audit.sha(folder/"fullbody_comparison.mp4")
        assert all(audit.sha(folder/n)==digest for n,digest in p["artifacts"].items())
        renderings[panel]=p
    assert renderings["primary"]["fixed_camera_distance_m"]==renderings["coupling"]["fixed_camera_distance_m"]
    assert renderings["primary"]["world_bounds"]==renderings["coupling"]["world_bounds"]
    output.mkdir(parents=True);files={}
    def save(path,name,digest=None):
        path=Path(path);data=path.read_bytes();value=__import__("hashlib").sha256(data).hexdigest()
        assert digest is None or value==digest
        target=output/name;target.parent.mkdir(exist_ok=True,parents=True)
        with target.open("xb") as f:f.write(data)
        files[name]=dict(sha256=value,bytes=len(data))
    for n,v in collection["files"].items():save(evidence/n,"evidence/"+n,v["sha256"])
    save(root/"matched_models_collection_v1.manifest.json","execution_collection_manifest.json")
    for n in ("analysis.json","comparison.png"):save(root/"analysis_v2"/n,n)
    save(a.reference,"reference.npz",summary["reference_sha256"]);save(a.protocol,"protocol.json",summary["nominal_protocol_sha256"])
    for panel,p in renderings.items():
        folder=root/(panel+"_render_v1")
        for n,digest in p["artifacts"].items():save(folder/n,panel+"/"+n,digest)
        save(folder/"provenance.json",panel+"/provenance.json")
        save(root/(panel+"_video_review_v1")/"decode_verification.json",panel+"/decode_verification.json")
        save(root/(panel+"_render_v1_launch.json"),panel+"/render_launch.json")
    executed={r["path"]:r["sha256"] for r in json.loads((evidence/"source_snapshot_v2.manifest.json").read_text())["files"]}
    assert all(audit.sha(ROOT/n)==v for n,v in executed.items())
    sources=source_ledger(executed)
    with tarfile.open(output/"executed_source.tar.gz","x:gz") as tar:
        for n in sorted(sources):tar.add(ROOT/n,arcname=n,recursive=False)
    files["executed_source.tar.gz"]=dict(sha256=audit.sha(output/"executed_source.tar.gz"),bytes=(output/"executed_source.tar.gz").stat().st_size)
    replay='''import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True);a=p.parse_args();r=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((r/"manifest.json").read_text())
for n,v in m["files"].items():assert sha(r/n)==v["sha256"],n
for n,v in m["source_sha256"].items():assert sha(a.repository/n)==v,n
sys.path.insert(0,str(a.repository/"scripts"));import analyze_physx_tron1_wheel_models as audit
summary=json.loads((r/"analysis.json").read_text());protocol=json.loads((r/"protocol.json").read_text())
reference=audit.base.metrics.reference_data(r/"reference.npz")
rows,_,_,_,_=audit.load_cases(r/"evidence",reference,protocol,original_metrics=summary["original_baseline_metrics"])
assert rows==summary["rows"]
print(json.dumps(dict(six_native_metric_sets_exact=True,all_external_wrenches_exact=True,frozen_nominal_screens_exact=True,physics_or_policy_execution=False,task_success_claimed=False)))
'''
    (output/"replay.py").write_text(replay);files["replay.py"]=dict(sha256=audit.sha(output/"replay.py"),bytes=(output/"replay.py").stat().st_size)
    page='''<!doctype html><meta charset="utf-8"><title>TRON1 wheel models: frozen policy</title><style>body{font:17px system-ui;max-width:1400px;margin:32px auto;line-height:1.5}video,img{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}</style>
<h1>TRON1 wheel geometry × force models: frozen-policy comparison</h1>
<p>Six PhysX episodes completed at μ=0.7 with the same declared K actor, same initial state,400 Hz physics/50 Hz policy, reference, gains and motors. Both wheels retain their original authored mass, COM, principal inertia and cylinder-to-body transforms. Every wheel shape has native-verified1 mm contact offset and0 rest offset. No policy was retrained.</p>
<p>The read-only native-offset replay and the explicit-offset authored/native arm are byte-identical to the original K recording. CPU USD tests verify unchanged envelope/volume and authored mass frames for both wheels. Native getters independently verify all-body masses, COM/inertia, armature/damping and shape counts in every episode.</p>
<h2>Primary geometry / force comparison</h2><video controls preload="metadata" poster="primary/comparison.png" src="primary/fullbody_comparison.mp4"></video>
<p>Authored/native, nine-slice/native, authored/nonlinear, nine-slice/nonlinear. Original recorded poses, opaque visual meshes, identical relative camera distance/world angles and XY bounds. Camera translates with the robot, without zoom, yaw following or pose smoothing. Collider subdivisions are not drawn; the unchanged visual mesh is shown. Body/wheel headings, COM direction and all native force-support samples within each displayed interval are annotated.</p>
<table><tr><th>Force model</th><th>Geometry</th><th>Core XY RMSE(m)</th><th>Whole XY RMSE(m)</th><th>Core speed ratio</th><th>β RMSE(°)</th><th>Any-wheel support</th><th>Longest gap(ms)</th><th>HF / original PPO</th><th>Numerical pass</th></tr>'''
    for row in rows:
        m=row["metrics"];core=m["windows"]["core"]["tracking"];whole=m["windows"]["whole_reference_overlap"]["tracking"]
        support=m["any_wheel_ground_support"]["full_contact"];screen=row["nominal_screen"]
        values=(row["law"],row["geometry"],f'{core["xy_rmse_m"]:.4f}',f'{whole["xy_rmse_m"]:.4f}',
            f'{core["speed_retention"]:.4f}',f'{core["beta_rmse_deg"]:.3f}',f'{support["any_wheel_support_sample_fraction"]:.2%}',
            f'{support["longest_no_support_observed_s"]*1000:.1f}',f'{screen["core_HF_ratio_to_original_same_engine_PPO"]:.3f}',screen["numeric_nominal_episode_pass"])
        page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    page+='''</table><p>Tracking, speed and intended nonzero-β criteria pass in all six cases. Continuous wheel support fails in all six: support fractions86.35–92.11%, longest gaps45–60 ms. Native cylinder arms pass the retained angular-HF reduction criterion; the four external-force arms do not. Neither slicing nor this nonlinear curve resolves the stable-contact task for the frozen policy. One deterministic episode per arm does not establish population robustness or physical fidelity.</p>
<h2>Force-application coupling control</h2><video controls preload="metadata" poster="coupling/comparison.png" src="coupling/fullbody_comparison.mp4"></video>
<p>External regularized Coulomb and nonlinear combined-slip arms use identical load/contact-point coupling. Native tangential friction is disabled in both, while native normal contact remains. Both external laws use the preceding physical sample's normal load and measured point, with2.5 ms coupling delay. Their applied forces, COM moments and dissipative contact power are independently recomputed from recorded kinematics. The first native-friction buffer precedes material replacement; every subsequent native-friction buffer is checked near zero. It is not counted as simultaneous native/external friction.</p>
<p>The nonlinear coefficients are sensitivity assumptions, not identified TRON1 tire parameters. This is an external wrench model, not built-in PhysX VehicleSDK. Aggregated friction patches and global-Z load do not identify pointwise friction cones. Full native normal vectors may include XY force; body β, wheel-hub angle and contact-point sliding are distinct. A larger support fraction alone is not continuous support, and no absolute-height/flight gate has been introduced.</p>
<p>Both videos have350 fully decoded frames at50 fps. Stills112/150/200/348 were inspected for each; this does not claim full manual playback. Every native2.5 ms state remains in evidence. Numeric replay reproduces all six metric sets, all external wrenches and unchanged nominal criteria without actor weights, physical models, inference or dynamics.</p>
<p><a href="analysis.json">Full metrics and failed checks</a> · <a href="replay.py">Read-only replay</a> · <a href="execution_collection_manifest.json">Execution evidence</a> · <a href="../physx_wheel_camber_controls_20261025_v1/report.html">Offset and free-camber controls</a> · <a href="../ground_contact_v2o_ppo_nominal_20261025_v1/report.html">Latest O reward comparison</a></p><pre>python replay.py --repository /path/to/tron1-sliding-analysis</pre>
<p>Original assets, neural weights, native libraries and credentials are omitted. Executed code/configuration snapshots and source hashes are included; restore source archives only into a fresh checkout. The first launch missing the existing BeyondMimic source environment path is preserved; it produced no task trajectory. Subsequent validated runs use the already installed, pinned upstream source. No installation, native compilation or other user's process intervention occurred.</p>
<p>Model-specific matched-budget learning remains a separate next phase. These transfer results do not show what a newly trained policy can achieve under each contact model and do not complete the stable-contact objective.</p>'''
    (output/"report.html").write_text(page);files["report.html"]=dict(sha256=audit.sha(output/"report.html"),bytes=(output/"report.html").stat().st_size)
    (output/"manifest.json").write_text(json.dumps(dict(kind=summary["kind"],files=files,source_sha256=sources,
        models_weights_credentials_native_binaries_included=False,model_specific_training=False,task_success_claimed=False),indent=2,allow_nan=False)+"\n")
    return dict(status="six_model_transfers_and_two_videos_published",output=str(output),task_success_claimed=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--input-root",type=Path,default=ROOT/"outputs/physx_tron1_wheel_models_20261025_v1")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--reference",type=Path,default=renderer.graphics.assets.paired.DEFAULT_REFERENCE)
    p.add_argument("--protocol",type=Path,default=ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    print(json.dumps(run(p.parse_args())))

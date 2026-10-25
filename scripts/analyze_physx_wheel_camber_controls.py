"""Publish measured native offsets and matched-offset free-wheel diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute(root):
    root=Path(root);rows=[];traces={}
    for folder in sorted((root/"cases").iterdir()):
        d=json.loads((folder/"status.json").read_text())
        assert d["status"]=="completed" and d["kind"]=="PhysX_single_wheel_offset_and_free_camber_controls"
        assert d["law"]=="native_coulomb" and d["physics_dt_s"]==.0025
        assert sha(folder/"native_trace.npz")==d["native_trace_sha256"]
        with np.load(folder/"native_trace.npz",allow_pickle=False) as a:raw={n:a[n].copy() for n in a.files}
        t=raw["time_s"];state=raw["root_state_w"];force=raw["ground_normal_force_w_n"]
        count=d["physical_steps"];assert count in (40,800) and state.shape==(count+1,13) and force.shape==(count+1,3)
        assert np.isfinite(state).all() and np.isfinite(force).all()
        np.testing.assert_allclose(t,np.arange(count+1)*.0025,rtol=0,atol=1e-12)
        np.testing.assert_allclose(state[0,:7],d["canonical_initial_pose_w"],rtol=0,atol=1e-7)
        np.testing.assert_allclose(state[0,7:],d["canonical_initial_velocity_w"],rtol=0,atol=1e-6)
        geometry=d["compound_geometry"];n=geometry["slices"]
        parser=json.loads((folder/"parser_audit.json").read_text());assert parser["body_max_shapes"]==n
        offsets=np.asarray(geometry["runtime_contact_offsets_m"])
        rest=np.asarray(geometry["runtime_rest_offsets_m"])
        assert offsets.shape==rest.shape==(1,n) and np.isfinite(offsets).all() and np.all(offsets>rest)
        np.testing.assert_allclose(rest,0.,rtol=0,atol=1e-8)
        if d["episode_rotation"]=="free":
            assert count==800 and d["axle_fixture_locked_world_rotation_bitmask_XZ"]==0
            np.testing.assert_allclose(offsets,.001,rtol=0,atol=1e-8)
        else:assert count==40 and d["axle_fixture_locked_world_rotation_bitmask_XZ"]==5
        np.testing.assert_array_equal(raw["applied_tangent_force_w_n"],np.zeros((count,3)))
        residual=raw["inferred_total_tangent_force_xy_n"]-force[1:,:2]-raw["native_sum_patch_tangent_force_w_n"][1:,:2]
        assert np.max(np.abs(residual))<.001,"Native impulse balance does not reproduce"
        q=state[:,3:7];w,x,y,z=np.moveaxis(q,-1,0)
        camber=np.rad2deg(np.arcsin(np.clip(2*(y*z+w*x),-1,1)))
        bearing=force[1:,2]>5
        runs=[];run=0
        for value in ~bearing:
            run=run+1 if value else 0;runs.append(run)
        row=dict(id=folder.name,slices=n,rotation=d["episode_rotation"],initial_camber_deg=d["initial_camber_deg"],
            duration_s=float(t[-1]),contact_offsets_m=offsets.tolist(),rest_offsets_m=rest.tolist(),
            support_fraction_above5N=float(bearing.mean()),longest_no_support_s=max(runs,default=0)*.0025,
            peak_normal_Fz_n=float(force[1:,2].max()),peak_abs_camber_deg=float(np.abs(camber).max()),
            final_camber_deg=float(camber[-1]),final_xy_m=state[-1,:2].tolist(),
            native_friction_patch_count_max=int(raw["native_friction_patch_count"].max()),
            XY_impulse_balance_peak_residual_n=float(np.abs(residual).max()),
            initial_state_w=state[0].tolist(),report_sha256=sha(folder/"status.json"),native_trace_sha256=sha(folder/"native_trace.npz"))
        rows.append(row);traces[folder.name]=dict(time=t,state=state,normal=force,camber=camber)
    assert len(rows)==8
    for phi in (0,-15,15):
        a,b=(next(r for r in rows if r["id"]==f"free_camber{phi:+d}_slices{n}") for n in (1,9))
        assert a["initial_state_w"]==b["initial_state_w"],"Paired initial mechanical state differs"
    return rows,traces


def run(a):
    root=a.input_root.resolve();evidence=root/"collected_v1";output=a.output_dir.resolve()
    assert not output.exists()
    manifest=json.loads((root/"controls_collection_v1.manifest.json").read_text())
    assert manifest["new_free_camber_episodes"]==6 and manifest["reused_completed_offset_measurements"]==2
    assert len(manifest["files"])==78
    assert all(sha(evidence/n)==v["sha256"] for n,v in manifest["files"].items())
    state=json.loads((evidence/"controls_v3/status.json").read_text())
    assert state["status"]=="all_eight_offset_and_free_camber_controls_completed_require_analysis"
    rows,traces=recompute(evidence)
    output.mkdir(parents=True);copied={}
    def save(path,name,digest=None):
        data=Path(path).read_bytes();s=hashlib.sha256(data).hexdigest();assert digest is None or s==digest
        p=output/name;p.parent.mkdir(exist_ok=True,parents=True)
        with p.open("xb") as stream:stream.write(data)
        copied[name]=dict(sha256=s,bytes=len(data))
    for n,v in manifest["files"].items():save(evidence/n,"evidence/"+n,v["sha256"])
    save(root/"controls_collection_v1.manifest.json","execution_collection_manifest.json")
    summary=dict(kind="native_offsets_and_matched_free_camber_diagnostics",rows=rows,new_episodes=6,reused_offset_measurements=2,
        initial_state_matched_between_slice_counts=True,contact_offsets_matched_for_free_episodes_m=.001,
        rest_offsets_m=0.,native_force_balance_includes_full_normal_vector=True,
        new_policy_training=False,TRON1_controller_result=False,task_success_claimed=False)
    p=output/"analysis.json";p.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n");copied[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey="row",layout="constrained")
    for col,phi in enumerate((0,-15,15)):
        for n,color in ((1,"#2271b2"),(9,"#d55e00")):
            tr=traces[f"free_camber{phi:+d}_slices{n}"]
            axes[0,col].plot(tr["time"],tr["camber"],color=color,label=f"{n} cylinder"+("s" if n>1 else ""))
            axes[1,col].plot(tr["time"],tr["normal"][:,2],color=color,alpha=.8)
        axes[0,col].set(title=f"Initial camber {phi:+d} degrees",ylabel="Measured axle camber (deg)");axes[0,col].legend()
        axes[1,col].set(xlabel="Time (s)",ylabel="Normal Fz (N)")
        for ax in axes[:,col]:ax.grid(alpha=.2)
    fig.suptitle("PhysX free single-wheel rolling | matched contact offset1mm, rest0 | initial side angle20deg")
    fig.savefig(output/"comparison.png",dpi=150);plt.close(fig)
    copied["comparison.png"]=dict(sha256=sha(output/"comparison.png"),bytes=(output/"comparison.png").stat().st_size)
    snapshot=json.loads((evidence/"source_snapshot_v3.manifest.json").read_text())
    sources={v["path"]:v["sha256"] for v in snapshot["files"]}
    assert all(sha(ROOT/n)==v for n,v in sources.items())
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha(__file__)
    with tarfile.open(output/"executed_source.tar.gz","x:gz") as tar:
        for n in sorted(sources):tar.add(ROOT/n,arcname=n,recursive=False)
    copied["executed_source.tar.gz"]=dict(sha256=sha(output/"executed_source.tar.gz"),bytes=(output/"executed_source.tar.gz").stat().st_size)
    replay='''import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True);a=p.parse_args();r=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((r/"manifest.json").read_text())
for n,v in m["files"].items():assert sha(r/n)==v["sha256"],n
for n,v in m["source_sha256"].items():assert sha(a.repository/n)==v,n
sys.path.insert(0,str(a.repository/"scripts"));from analyze_physx_wheel_camber_controls import recompute
rows,_=recompute(r/"evidence");assert rows==json.loads((r/"analysis.json").read_text())["rows"]
print(json.dumps(dict(eight_measurements_exact=True,native_impulse_balance_verified=True,physics_or_policy_execution=False,task_success_claimed=False)))
'''
    (output/"replay.py").write_text(replay);copied["replay.py"]=dict(sha256=sha(output/"replay.py"),bytes=(output/"replay.py").stat().st_size)
    page='''<!doctype html><meta charset="utf-8"><title>Wheel offsets and free camber</title><style>body{font:17px system-ui;max-width:1200px;margin:32px auto;line-height:1.5}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}</style><h1>Measured offsets and free-wheel camber controls</h1>
<p>Native readback: the single-cylinder automatic contact offset is1 mm; nine uniform axial cylinders use approximately0.122625 mm per shape. Rest offsets are0. The previous automatic-offset slicing study therefore combined geometry/manifold and automatic-offset changes. Two completed40-step measurements are explicitly reused; the six2-second free-wheel episodes below are new.</p>
<p>All six free-wheel cases use explicit1 mm contact /0 rest offsets, verified through native getters for every shape. Initial side angle20°, speed1 m/s, μ=0.7, physics400 Hz. Mass, COM and inertia match; corresponding one/nine-cylinder cases have identical initial poses and velocities. All rotation axes are free during these episodes, with spin initially along the tilted axle. This is a new fixture cohort, not an isolated camber-only change from the historical rotation-constrained bench.</p><img src="comparison.png"><table><tr><th>Camber</th><th>Slices</th><th>Support above5N</th><th>Longest gap(ms)</th><th>Peak Fz(N)</th><th>Peak absolute camber(°)</th></tr>'''
    for r in rows:
        if r["rotation"]!="free":continue
        page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (r["initial_camber_deg"],r["slices"],format(r["support_fraction_above5N"],".2%"),round(r["longest_no_support_s"]*1000,3),round(r["peak_normal_Fz_n"],3),round(r["peak_abs_camber_deg"],3)))+'</tr>'
    page+='''</table><p>Support is measured after each physical step; the initialization/warmup force sample is excluded. Slicing has mixed effects across these initial cambers. No body pose, height or heading is prescribed during an episode. There is no robot controller or learned policy here, and no claim of a completed TRON1 drift. Full normal-force vectors and native friction patches reproduce horizontal impulse balance within0.001 N.</p>
<p>Native getter semantics: <a href="https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html">Omni Physics tensor API</a>. The explicit model comparison must use native TRON1 offsets, not assume another asset's defaults.</p>
<p><a href="analysis.json">Metrics</a> · <a href="replay.py">Read-only replay</a> · <a href="../physx_wheel_geometry_20261025_v1/report.html">Previous automatic-offset bench</a></p><pre>python replay.py --repository /path/to/tron1-sliding-analysis</pre>'''
    (output/"report.html").write_text(page);copied["report.html"]=dict(sha256=sha(output/"report.html"),bytes=(output/"report.html").stat().st_size)
    (output/"manifest.json").write_text(json.dumps(dict(kind=summary["kind"],files=copied,source_sha256=sources,
        models_weights_credentials_native_binaries_included=False,task_success_claimed=False),indent=2)+"\n")
    print(json.dumps(dict(status="eight_measurements_published",output=str(output))))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--input-root",type=Path,default=ROOT/"outputs/physx_wheel_camber_controls_20261025_v1")
    p.add_argument("--output-dir",type=Path,required=True);run(p.parse_args())

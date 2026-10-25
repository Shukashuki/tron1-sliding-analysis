"""Recorded-only signed wheel-camber tracking and loaded-point RMS slip audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import analyze_contact_camber_reference as analysis
from analyze_contact_taskspace_scratch import slip_summary,body_integrity


def run(a):
    import mujoco
    root=a.result;target=root/"camber_tracking_diagnostic_v1.json";assert not target.exists()
    render=analysis.read(root/"render/provenance.json");verified=analysis.read(root/"artifact_verification_v1.json")
    assert verified["render_provenance_sha256"]==analysis.sha(root/"render/provenance.json")
    assert verified["formal_policy_identity_verified"] and verified["native_numerical_recomputation_matches"]
    assert analysis.sha(a.model)==render["source_model_sha256"]
    reference=analysis.metrics.metrics.reference_data(root/"reference.npz")
    rows,traces,_,_=analysis.load_cases(root/"completed_evidence_v1",reference)
    refpath=root/"completed_evidence_v1/source_snapshot_v4/references/fit_differential.npz"
    assert analysis.sha(refpath)=="0c8f2cfc2876bbd103bad206e708d91bf6753f8faa75c3107b0ec37c56c448b9"
    new=analysis.npz(refpath)
    replay=root/"render/replay_states.npz";assert analysis.sha(replay)==render["artifacts"]["replay_states.npz"]
    model=mujoco.MjModel.from_xml_path(str(a.model));data=mujoco.MjData(model)
    wheels=[model.body(n).id for n in ("wheel_L_Link","wheel_R_Link")]
    result={};arrays={}
    with np.load(replay,allow_pickle=False) as saved:
        for row in rows:
            key=row["id"];trace=traces[key];q=saved[key+"__qpos"]
            assert len(q)==2793
            gamma=[]
            for pose in q:
                data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
                axes=data.xmat[wheels].reshape(2,3,3)[:,:,1]
                gamma.append(np.arcsin(axes[:,2].clip(-1,1)))
                assert data.time==0.
            gamma=np.asarray(gamma)
            # Physical samples1..8 follow decision0;the recorded action/obs
            # hold is independently verified by the native recording contract.
            phase=np.maximum(0,(np.arange(2793)-1)//8)
            target_gamma=new["wheel_camber_rad"][phase] if row["variant"]=="camber" else np.zeros_like(gamma)
            core=(trace["time_s"]>=1.44-1e-9)&(trace["time_s"]<=4.+1e-9)
            loaded=trace["floor_normal_force"][:,:,2]>5.
            stats={}
            for i,name in enumerate(("left","right")):
                use=core&loaded[:,i]
                err=np.rad2deg(gamma[use,i]-target_gamma[use,i])
                stats[name]=dict(core_loaded_samples=int(use.sum()),
                    loaded_signed_camber_deg_median=None if not use.any() else float(np.median(np.rad2deg(gamma[use,i]))),
                    loaded_camber_target_RMSE_deg=None if not use.any() else float(np.sqrt(np.mean(err**2))),
                    all_core_camber_target_RMSE_deg=float(np.sqrt(np.mean(np.rad2deg(gamma[core,i]-target_gamma[core,i])**2))),
                    all_core_peak_abs_camber_deg=float(np.rad2deg(abs(gamma[core,i])).max()))
            base=analysis.CONTROL if row["variant"]=="control" else root/"completed_evidence_v1"
            folder=base/"nominal_v1"/("K_"+row["engine"])
            raw=analysis.npz(folder/("physics_trajectory.npz" if row["engine"]=="physx" else "rollout.npz"))
            result[key]=dict(camber=stats,loaded_point_slip=slip_summary(trace),body_integrity=body_integrity(raw,trace))
            arrays.update({key+"__actual_camber_rad":gamma,key+"__target_camber_rad":target_gamma,
                key+"__loaded":loaded,key+"__time_s":trace["time_s"],key+"__native_target_frame_index":phase,
                key+"__recorded_lateral_point_RMS_m_s":trace["contact_slip"],
                key+"__identified_loaded_point":loaded&(trace["contact_count"]>0)})
    output=root/"camber_tracking_diagnostic";output.mkdir(exist_ok=False)
    np.savez_compressed(output/"signals.npz",**arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(23,8),layout="constrained",sharex=True,sharey="row")
    for j,row in enumerate(rows):
        key=row["id"];t=arrays[key+"__time_s"]
        for i,name,color in [(0,"Left","#d62728"),(1,"Right","#9467bd")]:
            axes[0,j].plot(t,np.rad2deg(arrays[key+"__actual_camber_rad"][:,i]),color=color,label=name)
            axes[0,j].plot(t,np.rad2deg(arrays[key+"__target_camber_rad"][:,i]),'--',color=color,alpha=.6,label=name+" target")
            valid=arrays[key+"__identified_loaded_point"][:,i]
            slip=np.where(valid,arrays[key+"__recorded_lateral_point_RMS_m_s"][:,i],np.nan)
            axes[1,j].plot(t,slip,color=color,label=name)
        axes[0,j].set_title(row["title"])
        for ax in axes[:,j]:ax.axvspan(1.44,4,color="gray",alpha=.12);ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[1,j].set_xlabel("Native time [s]")
    axes[0,0].set_ylabel("Signed camber [deg]");axes[1,0].set_ylabel("Actual loaded-point lateral RMS [m/s]")
    fig.suptitle("Native poses /held reference | contact slip is unsigned per-point RMS;no geometric touch substitute")
    fig.savefig(output/"comparison.png",dpi=150);plt.close(fig)
    value=dict(kind="native_signed_camber_and_loaded_point_slip_diagnostic",rows=result,
        recorded_only=True,physics_or_inference_steps=0,
        target_clock="sample0=frame0;native samples1..8=decision/frame0,9..16=frame1;zero-order held,no target interpolation",
        camber_definition="asin(worldZ dot authored wheel localY axle),signed degrees",
        camber_load_mask="summed upward actual wheel-floor normal force>5N",
        slip_mask="wheel upwardFz>5N AND at least one individual ground point normal force>5N",
        target_single_geometric_point_and_actual_multi_point_RMS_are_different_estimators=True,
        signed_actual_contact_slip_available=False,acceptance_gate_changed=False,task_success_claimed=False,
        source_sha256=analysis.sha(__file__),new_reference_sha256=analysis.sha(refpath),replay_states_sha256=analysis.sha(replay),
        artifacts={p.name:analysis.sha(p) for p in output.iterdir()})
    target.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
    run(p.parse_args())

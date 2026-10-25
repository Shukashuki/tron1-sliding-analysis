"""Compare three completed PhysX tire laws using recorded wheel poses and forces.

No simulator, FK model, policy inference, or trajectory alignment is run.
Native-contact measurements and externally applied delayed forces are kept
separate. The existing tracking metrics are reused without changing screens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_backend_400hz_matrix as metrics
from analyze_cem_ground_contact_v2 import support_summary
from controller_metrics import heading_signals
from controller_trace import native_trace


CASES=("tron1_native_coulomb_v2","tron1_external_coulomb_v2","tron1_magic_formula_v1")
LABELS=("Native cylinder + Coulomb","External regularized Coulomb","Combined-slip Magic Formula")


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def run(a):
    assert not a.output_dir.exists()
    reference=metrics.reference_data(a.motion);original=load_npz(a.motion)
    rows=[];cases=[]
    for name,label in zip(CASES,LABELS):
        folder=a.collection/name;status=json.loads((folder/"tire_study_status.json").read_text())
        assert status["status"]=="completed_requires_analysis" and status["normalized_actor_sha256"]=="53c5a0f59428c5a5ba03092a19c269d64cdca0c9e96360d5d8abac195999dc69"
        for key in ("physics_trajectory.npz","tire_trajectory.npz"):assert sha(folder/key)==status[key+"_sha256"]
        raw=load_npz(folder/"physics_trajectory.npz");tire=load_npz(folder/"tire_trajectory.npz")
        trace=native_trace(raw,"policy_isaac")
        assert len(trace["time_s"])==2793 and len(tire["time_before_step_s"])==2792
        assert np.allclose(tire["time_before_step_s"],trace["time_s"][:-1],atol=1e-12,rtol=0)
        assert np.allclose(tire["preceding_normal_load_n"],trace["contact_force"][:-1,:,2],atol=1e-4,rtol=1e-6)
        short={k:(v[:-1] if isinstance(v,np.ndarray) and len(v)==2793 else v) for k,v in trace.items()}
        signals=heading_signals(short["quaternion"],short["com_velocity"],tire["wheel_quaternion_wxyz"],tire["COM_velocity_w_m_s"])
        core=metrics.window_metrics(short,reference,signals,(1.44,4.))
        whole=metrics.window_metrics(short,reference,signals,(0.,6.96))
        support=support_summary(trace["time_s"],trace["contact_force"][:,:,2],(.02,6.96))
        loaded=tire["preceding_normal_load_n"]>5.
        patch_velocity=tire["COM_velocity_w_m_s"]+np.cross(tire["angular_velocity_w_rad_s"],
            tire["preceding_contact_position_w_m"]-tire["COM_position_w_m"])
        external=status["law"]!="native_coulomb"
        force=tire["applied_tangent_force_w_n"] if external else tire["native_friction_force_w_n"]
        norm=np.linalg.norm(force[...,:2],axis=-1)
        ratio=np.divide(norm,.7*tire["preceding_normal_load_n"],out=np.full_like(norm,np.nan),where=loaded)
        mask=loaded & (tire["time_before_step_s"][:,None]>=1.44-1e-9)&(tire["time_before_step_s"][:,None]<=4.+1e-9)
        q=lambda value:np.quantile(value,[.5,.95,1.]).tolist()
        contact={"core_loaded_wheel_samples":int(mask.sum()),
            "core_contact_patch_speed_xy_m_s_quantiles_50_95_100":q(np.linalg.norm(patch_velocity[...,:2],axis=-1)[mask]),
            "core_force_over_used_muFn_quantiles_50_95_100":q(ratio[mask]),
            "core_fraction_loaded_wheel_samples_force_utilization_ge_0p9":float((ratio[mask]>=.9).mean()),
            "core_fraction_loaded_wheel_samples_patch_speed_ge_0p1_m_s":float((np.linalg.norm(patch_velocity[...,:2],axis=-1)[mask]>=.1).mean()),
            "force_time_semantics":"applied force divided by lagged load" if external else "native friction patch sum divided by same-sample normal load",
            "pointwise_cone_edge_identified":False,"drift_not_classified_from_utilization_alone":True}
        if external:
            next_load=trace["contact_force"][1:,:,2];nextmask=mask&(next_load>5)
            current=norm[nextmask]/(.7*next_load[nextmask])
            contact["core_applied_force_over_next_actual_muFn_quantiles_50_95_100"]=q(current)
        rows.append({"case":name,"label":label,"law":status["law"],"core":core,"whole":whole,
            "support":support,"contact":contact,"recording_sha256":status["physics_trajectory.npz_sha256"],
            "tire_recording_sha256":status["tire_trajectory.npz_sha256"]})
        cases.append({"trace":trace,"tire":tire,"signals":signals,"ratio":ratio})
    assert rows[0]["recording_sha256"]=="1927fd568e2b40684e54dad6546e5b0676668efff566f5ad10305094c2ff89fd"
    for c in cases[1:]:
        for key in ("position","quaternion","com_velocity","angular_velocity","joint_pos","joint_vel"):
            assert np.array_equal(c["trace"][key][0],cases[0]["trace"][key][0]),"Matched initial mechanical state required"
    a.output_dir.mkdir(parents=True)
    summary={"study":"TRON1_PhysX_three_tire_laws_frozen_M_PPO_transfer","status":"completed_three_full_horizon_episodes",
        "rows":rows,"actor_sha256":"53c5a0f59428c5a5ba03092a19c269d64cdca0c9e96360d5d8abac195999dc69",
        "baseline_instrumentation_matches_previous_raw_recording_exactly":True,"same_initial_mechanical_state_verified":True,
        "native_states_per_case":2793,"tire_pre_step_samples_per_case":2792,"policy_decisions":349,
        "physx_only":True,"dynamics_or_FK_or_inference_executed_by_analyzer":False,
        "tire_parameters_identified":False,"external_load_coupling_delay_physical_steps":1,
        "one_seed_and_one_episode_per_law":True,"policy_retrained_for_each_law":False,
        "limitations":["External Coulomb control measures the combined influence of load delay, regularization and force application.",
            "Difference between external Coulomb and Magic Formula isolates the chosen steady combined-slip curve within the same coupling.",
            "Neither external law includes static adhesion, camber thrust, tire relaxation, or a calibrated load-dependent tire model.",
            "Force utilization from aggregated patches is not per-contact friction-cone identification.",
            "Successful physical rollout completion does not establish stable support or task acceptance."],
        "analyzer_sha256":sha(__file__),"reference_sha256":sha(a.motion),"task_success_claimed":False}
    (a.output_dir/"analysis.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+chr(10))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4.7),dpi=150)
    axes[0].plot(reference["position"][:,0],reference["position"][:,1],color="gray",ls="--",label="Reference")
    for c,row,color in zip(cases,rows,("tab:blue","tab:orange","tab:green")):
        trace=c["trace"];t=trace["time_s"];kw={"label":row["label"],"color":color}
        axes[0].plot(trace["position"][:,0],trace["position"][:,1],**kw)
        axes[1].plot(t[:-1],np.rad2deg(c["signals"]["root_directed_beta_rad"]),**kw)
        axes[2].plot(t,np.linalg.norm(trace["com_velocity"][:,:2],axis=-1),**kw)
        for k,ls in ((0,"-"),(1,"--")):axes[3].plot(t[:-1],c["ratio"][:,k],color=color,ls=ls,alpha=.65)
    refyaw=np.arctan2(2*(reference["quaternion"][:,0]*reference["quaternion"][:,3]+reference["quaternion"][:,1]*reference["quaternion"][:,2]),1-2*np.sum(reference["quaternion"][:,2:]**2,axis=1))
    refbeta=(np.arctan2(reference["velocity"][:,1],reference["velocity"][:,0])-refyaw+np.pi)%(2*np.pi)-np.pi
    axes[1].plot(reference["time_s"],np.rad2deg(refbeta),color="gray",ls="--")
    axes[2].plot(reference["time_s"],np.linalg.norm(reference["velocity"][:,:2],axis=-1),color="gray",ls="--")
    axes[0].set(title="Recorded XY",xlabel="X (m)",ylabel="Y (m)",aspect="equal")
    axes[1].set(title="Body sideslip",xlabel="Time (s)",ylabel="Beta (deg)")
    axes[2].set(title="Speed",xlabel="Time (s)",ylabel="m/s")
    axes[3].set(title="Force / used mu Fn; L solid / R dashed",xlabel="Time (s)",ylabel="Aggregate utilization",ylim=(0,1.5))
    axes[0].legend(fontsize=6)
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle("Same frozen PPO, same PhysX geometry, mu=.7 | external tire coefficients are sensitivity assumptions")
    fig.tight_layout();fig.savefig(a.output_dir/"comparison.png");plt.close(fig)
    if a.render:render(a.output_dir,reference,original,cases,rows)
    print(json.dumps({"output":str(a.output_dir),"rows":[{"label":r["label"],"core":r["core"]["tracking"],"support":r["support"],"contact":r["contact"]} for r in rows]}))


def render(output,reference,original,cases,rows):
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    width,height,panel=2240,650,560
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",12)
    names=original["body_names"].tolist();wheel_ids=[names.index(n) for n in ("wheel_L_Link","wheel_R_Link")]
    allxy=np.concatenate([reference["position"][:,:2]]+[c["trace"]["position"][:,:2] for c in cases])
    center=(allxy.min(axis=0)+allxy.max(axis=0))/2;span=max(np.ptp(allxy,axis=0))+.8
    bounds=np.c_[center-span/2,center+span/2]
    def point(v,arm):return (arm*panel+30+int((v[0]-bounds[0,0])/span*500),590-int((v[1]-bounds[1,0])/span*500))
    def line(draw,origin,vector,arm,color,length,arrow=False):
        norm=np.linalg.norm(vector)
        if norm<1e-6:return
        vector=np.asarray(vector)/norm*length
        start=point(origin if arrow else origin-vector/2,arm);end=point(origin+vector if arrow else origin+vector/2,arm)
        draw.line([start,end],fill=color,width=3)
        if arrow:
            x,y=end;draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
    def body_forward(q):
        w,x,y,z=q;return np.array([1-2*(y*y+z*z),2*(x*y+w*z)])
    def rolling(q):
        w,x,y,z=q;return np.array([1-2*(x*x+z*z),-2*(x*y-w*z)])
    writer=imageio_ffmpeg.write_frames(str(output/"top_view.mp4"),(width,height),fps=50,codec="libx264",pix_fmt_in="rgb24",output_params=["-crf","20"],macro_block_size=2)
    writer.send(None)
    for frame in range(349):
        im=Image.new("RGB",(width,height),"white");draw=ImageDraw.Draw(im)
        for arm in range(4):
            title="Kinematic reference" if arm==0 else LABELS[arm-1]
            draw.text((arm*panel+20,12),title,fill="black",font=font)
            draw.text((arm*panel+20,36),f"t={frame*.02:.2f}s | fixed world bounds",fill="black",font=small)
            path=reference["position"][:,:2];draw.line([point(v,arm) for v in path],fill="#aaaaaa",width=2)
            if arm==0:
                pos=path[frame];q=reference["quaternion"][frame];vel=reference["velocity"][frame,:2]
                wp=original["body_pos_w"][frame,wheel_ids,:2];wq=original["body_quat_w"][frame,wheel_ids];Fn=None
            else:
                c=cases[arm-1];idx=frame*8;trace=c["trace"];tire=c["tire"]
                pos=trace["position"][idx,:2];q=trace["quaternion"][idx];vel=trace["com_velocity"][idx,:2]
                wp=tire["COM_position_w_m"][idx,:,:2];wq=tire["wheel_quaternion_wxyz"][idx];Fn=tire["preceding_normal_load_n"][idx]
                draw.line([point(v,arm) for v in trace["position"][:idx+1,:2]],fill="#15956c",width=3)
                score=rows[arm-1]["core"]["tracking"]
                draw.text((arm*panel+20,57),f"Core XY RMSE={score['xy_rmse_m']:.3f}m",fill="black",font=small)
            line(draw,pos,body_forward(q),arm,"#1464b4",.35,True)
            line(draw,pos,vel,arm,"#ed9518",.3,True)
            for k,color in ((0,"#d62728"),(1,"#9467bd")):
                line(draw,wp[k],rolling(wq[k]),arm,color,.254)
                x,y=point(wp[k],arm);draw.ellipse((x-4,y-4,x+4,y+4),fill=("#15956c" if Fn is not None and Fn[k]>5 else "#888888"))
            draw.text((arm*panel+20,605),f"Speed={np.linalg.norm(vel):.2f}m/s",fill="black",font=small)
            draw.text((arm*panel+20,628),"Blue: body | orange: velocity | red/purple: wheel planes",fill="black",font=small)
        writer.send(np.asarray(im).tobytes())
        if frame==150:im.save(output/"top_view.png")
    writer.close()
    (output/"render_provenance.json").write_text(json.dumps({"view":"fixed world orthographic2D schematic using measured root/wheel poses",
        "frames":349,"fps":50,"time_span_s":[0,6.96],"image_size":[width,height],"same_xy_bounds_all_panels_and_frames_m":bounds.tolist(),
        "wheel_lines_opaque":True,"root_and_wheel_states_smoothed_or_aligned":False,"simulator_or_FK_executed":False,
        "green_wheel_dots_mean_Fz_gt5N_at_display_sample":True,"substeps_per_display_frame":8,
        "unshown_substep_contacts_remain_in_numeric_trace":True,"reference_contact_is_not_measured":True,"video_sha256":sha(output/"top_view.mp4")},indent=2)+chr(10))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--collection",type=Path,required=True);p.add_argument("--motion",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--render",action="store_true")
    run(p.parse_args())

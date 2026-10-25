"""Analyze the unscaled source and six PhysX car-surrogate rollouts, without physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LAWS=("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula")
LABELS=("Native cylinder + Coulomb","External regularized Coulomb","Combined-slip Magic Formula")


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi


def yaw(q):
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def run(a):
    assert not a.output_dir.exists()
    assert sha(a.reference)=="1b4852178af7e1cd4d04aee4d42d86a7d12307f5ceb004c1b099b073683f298c"
    ref=load(a.reference);root=a.collection/"car_tire_sweep_v1"
    batch=json.loads((root/"status.json").read_text());assert batch["status"]=="six_car_surrogate_cases_completed_require_analysis"
    rows=[];cases={}
    initial=None;controls=None
    for case in batch["cases"]:
        folder=root/case["id"];status=json.loads((folder/"status.json").read_text())
        assert case["status"]=="completed_verified" and status["status"]=="completed_requires_analysis"
        assert status["native_trace_sha256"]==case["native_trace_sha256"]==sha(folder/"native_trace.npz")
        trace=load(folder/"native_trace.npz");t=trace["time_s"];state=trace["root_state_w"]
        assert len(t)==993 and np.allclose(t,np.arange(993)*.0025,atol=1e-12,rtol=0) and np.isfinite(state).all()
        mechanical={k:v[0].copy() for k,v in trace.items() if k in ("root_state_w","joint_pos_rad","joint_vel_rad_s","wheel_COM_position_w_m","wheel_COM_velocity_w_m_s","wheel_quaternion_wxyz")}
        if initial is None:initial=mechanical
        else:assert all(np.array_equal(mechanical[k],initial[k]) for k in initial),"All six car arms must have the same initial mechanical state"
        if controls is None:controls=trace["source_controls_and_wheel_torques"]
        else:assert np.array_equal(controls,trace["source_controls_and_wheel_torques"]),"Require identical source controls and applied wheel-torque targets"
        desired=np.stack([np.interp(t,ref["time_s_estimated"],ref["position_xy_m"][:,k]) for k in range(2)],axis=-1)
        desired_speed=np.interp(t,ref["time_s_estimated"],np.linalg.norm(ref["velocity_xy_m_s"],axis=-1))
        desired_beta=np.interp(t,ref["time_s_estimated"],np.unwrap(ref["beta_rad"]))
        speed=np.linalg.norm(state[:,7:9],axis=-1);heading=yaw(state[:,3:7]);beta=wrap(np.arctan2(state[:,8],state[:,7])-heading)
        course=np.unwrap(np.arctan2(state[:,8],state[:,7]));tilt=np.rad2deg(np.arccos(np.clip(1-2*np.sum(state[:,4:6]**2,axis=-1),-1,1)))
        bounds=ref["core_time_bounds_s_estimated"];core=(t>=bounds[0])&(t<=bounds[1])
        error=np.linalg.norm(state[:,:2]-desired,axis=-1)
        normal=trace["ground_normal_force_w_n"][:,:,2]
        Fn=normal[:-1];loaded=(Fn>5)&core[:-1,None]
        external=status["law"]!="native_coulomb"
        force=trace["applied_tangent_force_w_n"] if external else trace["native_friction_force_w_n"][:-1]
        ratio=np.divide(np.linalg.norm(force[...,:2],axis=-1),status["mu"]*Fn,out=np.full_like(Fn,np.nan),where=Fn>5)
        patch_velocity=trace["wheel_COM_velocity_w_m_s"][:-1]+np.cross(trace["wheel_angular_velocity_w_rad_s"][:-1],
            trace["contact_position_w_m"][:-1]-trace["wheel_COM_position_w_m"][:-1])
        row={"case":case["id"],"law":status["law"],"mu":status["mu"],"core_interval_s_estimated":bounds.tolist(),
            "core_XY_RMSE_m":float(np.sqrt(np.mean(error[core]**2))),"whole_XY_RMSE_m":float(np.sqrt(np.mean(error**2))),
            "core_XY_peak_m":float(error[core].max()),"core_speed_ratio":float(speed[core].mean()/desired_speed[core].mean()),
            "core_speed_mean_m_s":float(speed[core].mean()),"core_beta_RMSE_deg":float(np.rad2deg(np.sqrt(np.mean(wrap(beta[core]-desired_beta[core])**2)))),
            "core_course_change_deg":float(np.rad2deg(course[core][-1]-course[core][0])),
            "core_beta_range_deg":np.rad2deg([beta[core].min(),beta[core].max()]).tolist(),
            "core_any_wheel_support_fraction":float((normal[core]>5).any(axis=-1).mean()),
            "core_maximum_body_tilt_deg":float(tilt[core].max()),
            "core_contact_patch_speed_xy_m_s_quantiles_50_95_100":np.quantile(np.linalg.norm(patch_velocity[...,:2],axis=-1)[loaded],[.5,.95,1.]).tolist(),
            "core_force_over_used_muFn_quantiles_50_95_100":np.quantile(ratio[loaded],[.5,.95,1.]).tolist(),
            "reference_vehicle_plant_exactly_reproduced":False,"closed_loop_controller_used":False,"state_trace_sha256":sha(folder/"native_trace.npz")}
        rows.append(row);cases[(status["mu"],status["law"])]=dict(trace=trace,heading=heading,beta=beta,speed=speed,ratio=ratio,error=error)
    a.output_dir.mkdir(parents=True)
    result={"study":"unscaled_traj4_and_explicitly_assumed_PhysX_FWD_car_tire_law_sensitivity","status":"six_full_source_duration_surrogate_episodes_analyzed",
        "rows":rows,"same_initial_mechanical_state_all_six_verified":True,"same_control_and_torque_targets_all_six_verified":True,
        "reference_sha256":sha(a.reference),"analyzer_sha256":sha(__file__),
        "source_clock_is_reconstructed_estimate":True,"reference_is_CARLA_simulation_not_physical_capture":True,
        "exact_source_car_or_tires_identified":False,"source_wheel_orientations_observed":False,
        "limitations":["Surrogate wheelbase/track/radius/COM/inertia/steering and torque maps are declared assumptions.",
            "This compares open-loop model sensitivity, not an optimized car controller or exact source reproduction.",
            "Mu=.7 is not dynamically capable of the source lateral acceleration under a simple constant-mg flat-road traction budget; the source tire coefficient is not known.",
            "Mu=3.5 is the source code default, not verified capture metadata.",
            "External tire forces use a one-step-delayed normal load and no static-adhesion/camber/relaxation model.",
            "Aggregate horizontal tire force divided by vertical load is a diagnostic, not pointwise friction-cone identification."],
        "physx_only":True,"policy_training":False,"task_success_claimed":False}
    (a.output_dir/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+chr(10))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mu in (.7,3.5):
        fig,axes=plt.subplots(1,4,figsize=(20,4.7),dpi=150)
        for law,label,color in zip(LAWS,LABELS,("tab:blue","tab:orange","tab:green")):
            c=cases[(mu,law)];trace=c["trace"];t=trace["time_s"]
            axes[0].plot(trace["root_state_w"][:,0],trace["root_state_w"][:,1],label=label,color=color)
            axes[1].plot(t,np.rad2deg(c["beta"]),color=color)
            axes[2].plot(t,c["speed"],color=color)
            axes[3].plot(t,c["error"],color=color)
        axes[0].plot(ref["position_xy_m"][:,0],ref["position_xy_m"][:,1],color="gray",ls="--",label="Original simulated source")
        axes[1].plot(ref["time_s_estimated"],np.rad2deg(ref["beta_rad"]),color="gray",ls="--")
        axes[2].plot(ref["time_s_estimated"],np.linalg.norm(ref["velocity_xy_m_s"],axis=-1),color="gray",ls="--")
        axes[0].set(title="Unscaled car XY",xlabel="X (m)",ylabel="Y (m)",aspect="equal")
        axes[1].set(title="Body beta",xlabel="Time (s, source estimated)",ylabel="deg")
        axes[2].set(title="Speed",xlabel="Time (s)",ylabel="m/s")
        axes[3].set(title="XY error against source",xlabel="Time (s)",ylabel="m")
        axes[0].legend(fontsize=6)
        for ax in axes:ax.grid(alpha=.25)
        fig.suptitle(f"PhysX car surrogate, mu={mu} | identical source controls | declared plant assumptions, not exact CARLA reproduction")
        fig.tight_layout();fig.savefig(a.output_dir/(f"comparison_mu{str(mu).replace('.','p')}.png"));plt.close(fig)
    if a.render:render(a.output_dir,ref,cases,rows)
    print(json.dumps({"output":str(a.output_dir),"rows":rows}))


def render(output,ref,cases,rows):
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    panel,width,height=560,2240,650
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",12)
    allxy=np.concatenate([ref["position_xy_m"]]+[v["trace"]["root_state_w"][:,:2] for v in cases.values()])
    center=(allxy.min(axis=0)+allxy.max(axis=0))/2;span=max(np.ptp(allxy,axis=0))+10.;bounds=np.c_[center-span/2,center+span/2]
    def point(v,arm):return (arm*panel+30+int((v[0]-bounds[0,0])/span*500),590-int((v[1]-bounds[1,0])/span*500))
    def line(draw,pos,vector,arm,color,length,arrow=False):
        vector=np.asarray(vector);norm=np.linalg.norm(vector)
        if norm<1e-6:return
        delta=vector/norm*length
        start=point(pos if arrow else pos-delta/2,arm);end=point(pos+delta if arrow else pos+delta/2,arm)
        draw.line([start,end],fill=color,width=3)
        if arrow:
            x,y=end;draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
    for mu in (.7,3.5):
        name="top_view_mu"+str(mu).replace(".","p")
        writer=imageio_ffmpeg.write_frames(str(output/(name+".mp4")),(width,height),fps=50,codec="libx264",pix_fmt_in="rgb24",output_params=["-crf","20"],macro_block_size=2);writer.send(None)
        for frame in range(125):
            t=frame*.02;im=Image.new("RGB",(width,height),"white");draw=ImageDraw.Draw(im)
            for arm in range(4):
                title="Original CARLA source (kinematic)" if arm==0 else LABELS[arm-1]
                draw.text((arm*panel+18,12),title,fill="black",font=font)
                draw.text((arm*panel+18,37),f"t={t:.2f}s | mu={mu} surrogate | fixed world bounds",fill="black",font=small)
                draw.line([point(v,arm) for v in ref["position_xy_m"]],fill="#aaaaaa",width=2)
                if arm==0:
                    pos=np.array([np.interp(t,ref["time_s_estimated"],ref["position_xy_m"][:,k]) for k in range(2)])
                    angle=np.interp(t,ref["time_s_estimated"],ref["yaw_rad"])
                    velocity=np.array([np.interp(t,ref["time_s_estimated"],ref["velocity_xy_m_s"][:,k]) for k in range(2)])
                    draw.text((arm*panel+18,61),"Source wheel orientations / RPM unobserved",fill="black",font=small)
                else:
                    law=LAWS[arm-1];c=cases[(mu,law)];trace=c["trace"];idx=frame*8
                    pos=trace["root_state_w"][idx,:2];angle=c["heading"][idx];velocity=trace["root_state_w"][idx,7:9]
                    draw.line([point(v,arm) for v in trace["root_state_w"][:idx+1,:2]],fill="#15956c",width=3)
                    for k,color in enumerate(("#d62728","#9467bd","#d62728","#9467bd")):
                        q=trace["wheel_quaternion_wxyz"][idx,k];w,x,y,z=q
                        rolling=np.array([1-2*(x*x+z*z),-2*(x*y-w*z)])
                        p=trace["wheel_COM_position_w_m"][idx,k,:2];line(draw,p,rolling,arm,color,1.8)
                        px,py=point(p,arm);draw.ellipse((px-2,py-2,px+2,py+2),fill="#15956c" if trace["ground_normal_force_w_n"][idx,k,2]>5 else "#888888")
                    row=next(r for r in rows if r["mu"]==mu and r["law"]==law)
                    draw.text((arm*panel+18,61),f"Core XY RMSE={row['core_XY_RMSE_m']:.2f}m | assumed plant",fill="black",font=small)
                line(draw,pos,[np.cos(angle),np.sin(angle)],arm,"#1464b4",5.,True)
                line(draw,pos,velocity,arm,"#ed9518",4.,True)
                draw.text((arm*panel+18,606),f"Speed={np.linalg.norm(velocity):.1f}m/s",fill="black",font=small)
                draw.text((arm*panel+18,628),"Blue: body | orange: velocity | red/purple: wheel planes",fill="black",font=small)
            writer.send(np.asarray(im).tobytes())
            if frame==75:im.save(output/(name+".png"))
        writer.close()
    (output/"render_provenance.json").write_text(json.dumps({"frames_per_movie":125,"fps":50,"time_span_s":[0,2.48],
        "image_size":[width,height],"same_xy_bounds_all_panels_frames_and_mu_m":bounds.tolist(),
        "source_wheel_headings_invented_or_displayed":False,"root_and_wheel_states_smoothed_or_aligned":False,
        "view":"fixed orthographic2D schematic from native PhysX body/wheel poses",
        "videos_sha256":{p.name:sha(p) for p in output.glob("*.mp4")}},indent=2)+chr(10))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--collection",type=Path,required=True);p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--render",action="store_true")
    run(p.parse_args())

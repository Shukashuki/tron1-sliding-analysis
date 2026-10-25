"""Native Vehicle SDK telemetry versus the rigid-cylinder car diagnostics.

This is a component-model comparison, not a tire-curve-only ablation. Fixed
Y-up -> Z-up basis conversion is explicit. Wheel centers are reconstructed
from measured suspension jounce, the declared suspension frames and root pose;
rolling planes use measured steering and root attitude, not an IK solution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_physx_car_tires import load,wrap,yaw


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rotate(q,v):
    vec=q[...,1:];t=2*np.cross(vec,v)
    return v+q[...,:1]*t+np.cross(vec,t)


def run(a):
    assert not a.output_dir.exists()
    ref=load(a.reference);rows=[];cases={}
    for mu,name in ((.7,"vehicle_sdk_mu0p7_v1"),(3.5,"vehicle_sdk_mu3p5_v1")):
        folder=a.sdk_collection/name;status=json.loads((folder/"status.json").read_text())
        assert status["status"]=="completed_requires_analysis" and status["native_states"]==993 and status["mu"]==mu
        assert status["smoke_only"] is False and sha(folder/"native_trace.npz")==status["native_trace_sha256"]
        trace=load(folder/"native_trace.npz");t=trace["time_s"];state=trace["root_state_source_Zup"]
        assert np.allclose(t,np.arange(993)*.0025,atol=1e-12,rtol=0)
        desired=np.stack([np.interp(t,ref["time_s_estimated"],ref["position_xy_m"][:,i]) for i in range(2)],axis=-1)
        desired_beta=np.interp(t,ref["time_s_estimated"],np.unwrap(ref["beta_rad"]))
        desired_speed=np.interp(t,ref["time_s_estimated"],np.linalg.norm(ref["velocity_xy_m_s"],axis=-1))
        beta=wrap(np.arctan2(state[:,8],state[:,7])-yaw(state[:,3:7]));speed=np.linalg.norm(state[:,7:9],axis=-1)
        error=np.linalg.norm(state[:,:2]-desired,axis=-1);bounds=ref["core_time_bounds_s_estimated"]
        core=(t>=bounds[0])&(t<=bounds[1]);normal=trace["suspension_force"][:,:,1]
        loaded=(normal>5)&trace["is_on_ground"]&core[:,None]
        # Native tire_force is the Vehicle tire force, distinct from suspension.
        tangent=np.linalg.norm(trace["tire_force"][:,:,[2,0]],axis=-1)
        ratio=np.divide(tangent,trace["tire_friction"]*normal,out=np.full_like(normal,np.nan),where=normal>5)
        tilt=np.rad2deg(np.arccos(np.clip(1-2*np.sum(state[:,4:6]**2,axis=-1),-1,1)))
        row={"mu":mu,"core_XY_RMSE_m":float(np.sqrt(np.mean(error[core]**2))),
            "whole_XY_RMSE_m":float(np.sqrt(np.mean(error**2))),"core_XY_peak_m":float(error[core].max()),
            "core_speed_ratio":float(speed[core].mean()/desired_speed[core].mean()),
            "core_beta_RMSE_deg":float(np.rad2deg(np.sqrt(np.mean(wrap(beta[core]-desired_beta[core])**2)))),
            "core_any_wheel_native_on_ground_fraction":float(trace["is_on_ground"][core].any(axis=-1).mean()),
            "core_any_wheel_suspension_Fup_gt5N_fraction":float((normal[core]>5).any(axis=-1).mean()),
            "core_maximum_body_tilt_deg":float(tilt[core].max()),
            "core_loaded_wheel_samples":int(loaded.sum()),
            "core_native_tire_force_over_mu_suspension_force_quantiles_50_95_100":np.quantile(ratio[loaded],[.5,.95,1.]).tolist() if loaded.any() else None,
            "core_abs_native_tire_lateral_slip_rad_quantiles_50_95_100":np.quantile(np.abs(trace["tire_lateral_slip"])[loaded],[.5,.95,1.]).tolist() if loaded.any() else None,
            "native_trace_sha256":status["native_trace_sha256"],"physics_device":"CPU PhysX with native Vehicle SDK update",
            "external_measured_load_delay":False,"matched_tire_curve_only_ablation":False,"source_plant_identified":False}
        row["rollover_observed_core_tilt_gt90deg"]=bool(tilt[core].max()>90.)
        rows.append(row)
        positions=np.asarray(status["preflight"]["wheel_positions_relative_to_chassis_native_Yup"])
        relative=np.broadcast_to(positions,(993,4,3)).copy()
        relative[:,:,1]+=trace["suspension_jounce"]-450*9.81/45000
        wp=rotate(state[:,None,3:7],relative[:,:,[2,0,1]])+state[:,None,:3]
        angle=trace["steer_angle"];axle=np.stack((-np.sin(angle),np.cos(angle),np.zeros_like(angle)),axis=-1)
        axle=rotate(state[:,None,3:7],axle);direction=np.stack((axle[:,:,1],-axle[:,:,0]),axis=-1)
        cases[(mu,"sdk")]=dict(state=state,wheel_position=wp,rolling=direction,normal=normal,heading=yaw(state[:,3:7]))
        for law in ("native_coulomb","combined_slip_magic_formula"):
            source=a.rigid_collection/"car_tire_sweep_v1"/(law+"_mu"+str(mu).replace(".","p"))
            raw=load(source/"native_trace.npz");proof=json.loads((source/"status.json").read_text())
            assert proof["native_trace_sha256"]==sha(source/"native_trace.npz")
            assert np.array_equal(trace["source_controls"],raw["source_controls_and_wheel_torques"][:,:4])
            q=raw["wheel_quaternion_wxyz"];w,x,y,z=np.moveaxis(q,-1,0)
            direction=np.stack((1-2*(x*x+z*z),-2*(x*y-w*z)),axis=-1)
            cases[(mu,law)]=dict(state=raw["root_state_w"],wheel_position=raw["wheel_COM_position_w_m"],
                rolling=direction,normal=raw["ground_normal_force_w_n"][:,:,2],heading=yaw(raw["root_state_w"][:,3:7]))
    a.output_dir.mkdir(parents=True)
    result={"study":"native_PhysX_Vehicle_SDK_component_model_contrast","status":"two_full_car_reference_duration_episodes_analyzed",
        "rows":rows,"same_normalized_source_controls_as_rigid_surrogate_verified":True,
        "reference_is_not_real_car_measurement":True,"source_vehicle_calibration_not_identified":True,
        "native_Vehicle_default_tire_is_not_Pacejka":True,"native_Vehicle_has_no_external_Fn_sample_delay":True,
        "suspension_force_ratio_is_not_actual_tire_cone_utilization":True,
        "filtered_tire_normal_load_exposed_by_wheel_state_binding":False,
        "force_ratio_caution":"Vehicle tire load filtering/normalization and suspension-force projection differ from measured rigid contact Fn. A tire-force/(mu*suspension-up-force) ratio above1 is not proof of an SDK friction-cone violation.",
        "model_changes":["native suspension/road query and Vehicle tire update replace rigid wheel-ground contacts",
            "Vehicle lumped chassis inertia and virtual wheel mechanics differ from the multi-body rigid-axle articulation",
            "Vehicle SDK uses CPU PhysX here; rigid-axle arms use GPU PhysX, so this is not an execution-path-controlled tire-law ablation"],
        "wheel_render":"analytic composition of recorded root, steer angle and suspension jounce; no IK or optimization",
        "analyzer_sha256":sha(__file__),"task_success_claimed":False}
    (a.output_dir/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+chr(10))
    render(a.output_dir,ref,cases)
    print(json.dumps(result))


def render(output,ref,cases):
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    panel,width,height=560,2240,650
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",12)
    allxy=np.concatenate([ref["position_xy_m"]]+[v["state"][:,:2] for v in cases.values()])
    center=(allxy.min(axis=0)+allxy.max(axis=0))/2;span=max(np.ptp(allxy,axis=0))+10.;bounds=np.c_[center-span/2,center+span/2]
    def point(v,arm):return (arm*panel+30+int((v[0]-bounds[0,0])/span*500),590-int((v[1]-bounds[1,0])/span*500))
    def line(draw,pos,vector,arm,color,length,arrow=False):
        vector=np.asarray(vector);norm=np.linalg.norm(vector)
        if norm<1e-6:return
        d=vector/norm*length;start=point(pos if arrow else pos-d/2,arm);end=point(pos+d if arrow else pos+d/2,arm)
        draw.line([start,end],fill=color,width=3)
        if arrow:
            x,y=end;draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
    titles=("Original CARLA source (kinematic)","Rigid axle: native Coulomb","Rigid axle: external Magic Formula","Native Vehicle: suspension + tire")
    for mu in (.7,3.5):
        name="top_view_mu"+str(mu).replace(".","p")
        writer=imageio_ffmpeg.write_frames(str(output/(name+".mp4")),(width,height),fps=50,codec="libx264",pix_fmt_in="rgb24",output_params=["-crf","20"],macro_block_size=2);writer.send(None)
        for frame in range(125):
            t=frame*.02;idx=frame*8;im=Image.new("RGB",(width,height),"white");draw=ImageDraw.Draw(im)
            for arm in range(4):
                draw.text((arm*panel+16,12),titles[arm],fill="black",font=font)
                draw.text((arm*panel+16,37),f"t={t:.2f}s | surrogate mu={mu} | fixed bounds",fill="black",font=small)
                draw.line([point(v,arm) for v in ref["position_xy_m"]],fill="#aaaaaa",width=2)
                if arm==0:
                    pos=np.array([np.interp(t,ref["time_s_estimated"],ref["position_xy_m"][:,i]) for i in range(2)])
                    heading=np.interp(t,ref["time_s_estimated"],ref["yaw_rad"])
                    v=np.array([np.interp(t,ref["time_s_estimated"],ref["velocity_xy_m_s"][:,i]) for i in range(2)])
                    draw.text((arm*panel+16,61),"Source wheel orientations / forces unobserved",fill="black",font=small)
                else:
                    c=cases[(mu,("native_coulomb","combined_slip_magic_formula","sdk")[arm-1])]
                    pos=c["state"][idx,:2];heading=c["heading"][idx];v=c["state"][idx,7:9]
                    draw.line([point(p,arm) for p in c["state"][:idx+1,:2]],fill="#15956c",width=3)
                    for k,color in enumerate(("#d62728","#9467bd","#d62728","#9467bd")):
                        p=c["wheel_position"][idx,k,:2];line(draw,p,c["rolling"][idx,k],arm,color,1.8)
                        x,y=point(p,arm);draw.ellipse((x-2,y-2,x+2,y+2),fill="#15956c" if c["normal"][idx,k]>5 else "#888888")
                    draw.text((arm*panel+16,61),"Component-model diagnostic; not exact CARLA",fill="black",font=small)
                line(draw,pos,[np.cos(heading),np.sin(heading)],arm,"#1464b4",5.,True);line(draw,pos,v,arm,"#ed9518",4.,True)
                draw.text((arm*panel+16,606),f"Speed={np.linalg.norm(v):.1f}m/s",fill="black",font=small)
                draw.text((arm*panel+16,628),"Blue: body | orange: velocity | red/purple: wheel planes",fill="black",font=small)
            writer.send(np.asarray(im).tobytes())
            if frame==75:im.save(output/(name+".png"))
        writer.close()
    (output/"render_provenance.json").write_text(json.dumps({"frames_per_movie":125,"fps":50,"same_world_xy_bounds_all_panels_frames_and_mu_m":bounds.tolist(),
        "source_wheel_data_invented":False,"wheel_planes_opaque":True,"native_SDK_wheel_centers_reconstructed_from_root_steer_jounce":True,
        "IK_optimization_or_physics_executed":False,"videos_sha256":{p.name:sha(p) for p in output.glob("*.mp4")}},indent=2)+chr(10))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("sdk-collection","rigid-collection","reference","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())

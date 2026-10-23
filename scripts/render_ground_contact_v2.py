"""Generic SHA-bound recorded-pose400Hz replay; no dynamics or policy inference.

Defaults to original CEM versus V2 CEM. An explicit case manifest can later add
completed PPO PhysX/MuJoCo recordings without changing this graphics pipeline.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","osmesa")
import numpy as np

import analyze_cem_ground_contact_v2 as audit
import analyze_ppo_backend_400hz_matrix as metrics
from controller_trace import native_trace
from ppo_backend_400hz_contract import require,sha256,read_json,write_json
from render_physx_ppo_fullbody import build_qpos,create_runtime,mesh_provenance,_pose
from render_ppo_backend_400hz_nominal import diagnostics,replay_fk_audit
from render_ppo_beta_nominal import world_bounds
from render_ppo_fixed_follow import prepare_camera,apply_camera
import run_cem_ground_contact_v2 as runner

ROOT=Path(__file__).resolve().parents[1]
FPS,FRAMES=50,350


def native_indices():return np.arange(FRAMES,dtype=np.int64)*8


def load_case(spec):
    require(spec["kind"] in ("nmpc","policy_isaac","policy_mujoco"),"Select explicit native schema")
    raw_path,report_path=Path(spec["recording"]),Path(spec["report"])
    require(sha256(raw_path)==spec["recording_sha256"] and sha256(report_path)==spec["report_sha256"],"Case input SHA differs")
    report=read_json(report_path)
    require(report.get("status") in ("completed","evaluated") and not report.get("smoke_only",False),"Only completed formal records may be rendered")
    with np.load(raw_path,allow_pickle=False) as archive:raw={name:archive[name].copy() for name in archive.files}
    trace=native_trace(raw,spec["kind"])
    require(trace["time_s"].shape==(2793,) and trace["physics_dt_s"]==.0025
        and np.allclose(trace["time_s"],np.arange(2793)*.0025,rtol=0,atol=1e-10),"Require original full400Hz consecutive recording")
    for canonical,field in (("contact_force",spec.get("ground_force_key")),("contact_count",spec.get("ground_count_key"))):
        if field:
            value=np.asarray(raw[field]);value=value[:,0] if spec["kind"]=="policy_isaac" else value
            require(value.shape==trace[canonical].shape and np.isfinite(value).all(),"Invalid explicit ground channel")
            trace[canonical]=value.copy()
    require(sha256(raw_path)==spec["recording_sha256"] and sha256(report_path)==spec["report_sha256"],"Case inputs changed during load")
    return trace


def default_specs(input_dir):
    _,_,hashes,_=audit.load_completed(input_dir)
    baseline=read_json(audit.BASELINE_SUMMARY)
    old=next(row for row in baseline["rows"] if row["id"]=="cem_beta_on")
    return [{"id":"original_CEM","title":"Original CEM | original grounded +0.4 | beta ON","kind":"nmpc",
             "recording":str(audit.BASELINE_FOLDER/"rollout.npz"),"report":str(audit.BASELINE_FOLDER/"report.json"),
             "recording_sha256":old["recording_sha256"],"report_sha256":old["report_sha256"],
             "count_semantics":"legacy active ground points: normalforce>5N"},
            {"id":"V2_CEM","title":"V2 CEM | any-wheel ground support +0.4 | beta ON","kind":"nmpc",
             "recording":str(Path(input_dir)/"rollout.npz"),"report":str(Path(input_dir)/"report.json"),
             "recording_sha256":hashes["rollout.npz"],"report_sha256":hashes["report.json"],
             "ground_force_key":"ground_normal_force_w","ground_count_key":"ground_contact_count",
             "count_semantics":"V2 positive ground points: normalforce>=1e-6N"}]


def contact_window(trace,index):
    require(type(index) is int and 0<=index<2793,"Invalid original native index")
    indices=np.array([0]) if index==0 else np.arange(max(1,index-7),index+1)
    mask=np.max(trace["contact_force"][indices,:,2],axis=-1)>5.
    return {"native_indices":indices,"any_wheel_supported":mask,"supported_samples":int(mask.sum()),
            "samples":len(indices),"single_wheel_lift_allowed":True,"flight_inferred":False}


def draw_map(draw,trace,reference,index,box,bounds,font):
    low,high=np.asarray(bounds,dtype=float);left,top,right,bottom=box
    scale=min((right-left-22)/(high[0]-low[0]),(bottom-top-40)/(high[1]-low[1]));center=(low+high)/2
    pixel=lambda xy:((left+right)/2+(xy[0]-center[0])*scale,(top+bottom)/2-(xy[1]-center[1])*scale)
    draw.rectangle(box,fill="#f7f9fa",outline="#c9cdd1")
    draw.text((left+8,top+3),"World XY | ref gray, path green, body blue, COM dark green",font=font,fill="#333333")
    draw.line([pixel(p) for p in reference["position"][:,:2]],fill="#888888",width=2)
    if index>0:draw.line([pixel(p) for p in trace["position"][:index+1,:2]],fill="#15956c",width=2)
    signals=trace["signals"];root=trace["position"][index,:2]
    arrows=[(root,signals["body_heading_rad"][index],"#1464b4",trace["body_heading_geometry_valid"][index],False),
            (root,signals["course_rad"][index],"#13733c",signals["speed_m_s"][index]>=.25,False)]
    for side,color in enumerate(("#d62728","#9467bd")):
        bearing=trace["contact_count"][index,side]>0 and trace["contact_force"][index,side,2]>5.
        arrows.append((trace["wheel_pos"][index,side,:2],signals["wheel_rolling_plane_heading_rad"][index,side],
            color if bearing else "#d0c7ce",trace["wheel_heading_geometry_valid"][index,side],True))
    for point,angle,color,valid,axial in arrows:
        if valid:
            delta=.24*np.array([np.cos(angle),np.sin(angle)])
            start=point-delta/2 if axial else point;stop=point+delta/2 if axial else point+delta
            draw.line((pixel(start),pixel(stop)),fill=color,width=3)
            if not axial:
                tip=np.asarray(pixel(stop));direction=np.array([np.cos(angle),-np.sin(angle)]);normal=np.array([-direction[1],direction[0]])
                draw.polygon([tuple(tip),tuple(tip-7*direction+3*normal),tuple(tip-7*direction-3*normal)],fill=color)
    draw.text((left+8,bottom-18),f"Same fixed world axes | X [{low[0]:.2f},{high[0]:.2f}]m Y [{low[1]:.2f},{high[1]:.2f}]m",font=font,fill="#555555")


def render_frame(runtime,traces,qposes,reference,specs,plan,bounds,frame):
    from PIL import Image,ImageDraw
    require(type(frame) is int and 0<=frame<FRAMES,"Invalid original50Hz frame")
    index=int(native_indices()[frame]);canvas=Image.new("RGB",(640*len(specs),790),"#fafafa");draw=ImageDraw.Draw(canvas)
    draw.text((12,4),f"Recorded400Hz / replay50Hz | mu=0.7 | t={index*.0025:.2f}s | fixed camera distance {plan['distance_m']:.3f}m; no zoom/yaw-follow",font=runtime.small,fill="#111111")
    for column,spec in enumerate(specs):
        key=spec["id"];trace=traces[key];x=column*640
        _pose(runtime.model,runtime.data,qposes[key][index]);camera=apply_camera(runtime,plan,key,frame)
        runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
        canvas.paste(Image.fromarray(runtime.renderer.render().copy()),(x,58))
        draw.rectangle((x,30,x+640,57),fill="#eef1f4");draw.text((x+7,35),spec["title"],font=runtime.font,fill="#111111")
        current=diagnostics(trace,reference,index);error="n/a beyondreference" if current["xy_error_m"] is None else f"{current['xy_error_m']:.3f}m"
        beta=trace["signals"]["root_directed_beta_rad"][index];beta_text=f"{np.degrees(beta):.1f}deg" if trace["signals"]["root_beta_valid"][index] else "unavailable"
        draw.text((x+8,425),f"XY {error} | speed {current['speed_m_s']:.2f}m/s | beta {beta_text} (nonzero intended)",font=runtime.small,fill="#111111")
        draw.text((x+8,447),f"Original height {current['height_m']:.3f}m | tilt {current['tilt_deg']:.1f}deg | ground Fz L/R {current['fz'][0]:.0f}/{current['fz'][1]:.0f}N",font=runtime.small,fill="#111111")
        window=contact_window(trace,index)
        draw.text((x+8,470),f"Any-wheel Fz>5N | preceding native samples {window['supported_samples']}/{window['samples']} supported",font=runtime.small,fill="#111111")
        for j,supported in enumerate(window["any_wheel_supported"]):
            draw.ellipse((x+440+22*j,492,x+454+22*j,506),fill="#25804b" if supported else "#b74949")
        draw.text((x+8,489),"Dots: native2.5ms samples, NOT interpolated video contact",font=runtime.small,fill="#333333")
        draw_map(draw,trace,reference,index,(x+5,518,x+635,710),bounds,runtime.small)
    draw.text((12,721),"Contact goal NOT achieved: V2 full any-wheel support70.47%, longest no-load gap0.2575s. Tracking/HF improvement does not certify stable contact.",font=runtime.small,fill="#9b2929")
    draw.text((12,743),"At least one wheel may support; the other may lift. Red dot means no measured wheel Fz>5N, NOT a geometric flight label. N=1.",font=runtime.small,fill="#333333")
    draw.text((12,765),"Original opaque TRON1 meshes and all native poses/heights retained. Translation-only follow; fixed world orientation/distance. No new physics or policy inference.",font=runtime.small,fill="#333333")
    return canvas


def run(args):
    output=Path(args.output_dir);require(not output.exists(),"Preserve render; fresh output directory required")
    require(os.environ.get("MUJOCO_GL")=="osmesa","Require CPU OSMesa graphics")
    if getattr(args,"case_manifest",None):
        manifest=read_json(args.case_manifest);specs=manifest["cases"]
    else:specs=default_specs(args.input_dir)
    require(1<=len(specs)<=4 and len({s["id"] for s in specs})==len(specs),"Require1–4 unique recorded cases")
    traces={s["id"]:load_case(s) for s in specs};order=[s["id"] for s in specs]
    reference=metrics.reference_data(args.reference);kinematics,fk_source=metrics.load_fk(args.model,args.legacy_evaluator)
    for trace in traces.values():
        reconstruction=kinematics.reconstruct(trace)
        trace.update(signals=reconstruction["signals"],wheel_quaternion_wxyz=reconstruction["wheel_quaternion_wxyz"],
                     wheel_center_velocity_w=reconstruction["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"],trace["wheel_heading_geometry_valid"]=metrics.heading_geometry_masks(trace)
    runtime=create_runtime(args.model,width=640,height=360)
    try:
        qposes,audits={},{}
        for key in order:
            qposes[key],mapping=build_qpos(runtime.model,traces[key])
            audits[key]={**replay_fk_audit(runtime.model,traces[key],qposes[key]),"named_joint_qpos_addresses":mapping.tolist()}
        plan=prepare_camera(runtime,traces,qposes,order,native_indices())
        bounds=[value.tolist() for value in world_bounds(traces,reference)]
        output.mkdir(parents=True,exist_ok=False)
        replay={"native_indices":native_indices(),"fixed_distance_m":np.asarray(plan["distance_m"])}
        for key in order:
            replay.update({key+"__qpos":qposes[key],key+"__native_time_s":traces[key]["time_s"],
                key+"__camera_lookat":plan["lookat_by_case"][key],key+"__ground_normal_force_w":traces[key]["contact_force"],
                key+"__ground_count":traces[key]["contact_count"]})
        np.savez_compressed(output/"replay_states.npz",**replay)
        render_frame(runtime,traces,qposes,reference,specs,plan,bounds,145).save(output/"comparison.png")
        files=["replay_states.npz","comparison.png"]
        if not getattr(args,"skip_video",False):
            import imageio_ffmpeg
            stream=imageio_ffmpeg.write_frames(str(output/"fullbody_comparison.mp4"),(640*len(specs),790),fps=FPS,
                codec="libx264",quality=7,macro_block_size=2,output_params=["-pix_fmt","yuv420p","-movflags","+faststart"])
            stream.send(None)
            try:
                for frame in range(FRAMES):
                    stream.send(np.asarray(render_frame(runtime,traces,qposes,reference,specs,plan,bounds,frame)))
                    if frame%100==0:print(f"OSMesa recorded-only replay frame{frame}/{FRAMES}",flush=True)
            finally:stream.close()
            files.append("fullbody_comparison.mp4")
        result={"kind":"recorded_ground_contact_V2_fixed_relative_fullbody_replay","cases":specs,
            "native_states_per_case":2793,"native_dt_s":.0025,"video_frames":FRAMES,"video_fps":FPS,
            "selected_native_stride":8,"video_generated":not getattr(args,"skip_video",False),
            "reference_sha256":sha256(args.reference),"model_sha256":sha256(args.model),"renderer_sha256":sha256(__file__),
            "mesh_provenance":mesh_provenance(args.model),"FK_source":fk_source,"replay_FK_audit":audits,
            "camera":{key:value for key,value in plan.items() if key!="lookat_by_case"},"fixed_world_bounds":bounds,
            "camera_distance_same_for_all_cases_frames":True,"camera_yaw_follows_robot":False,"camera_zoom":False,
            "contact_overlay_native_substeps":8,"single_wheel_support_allowed":True,
            "stable_contact_or_goal_achieved_claimed":False,"robot_mesh_alpha":1.,"visual_reviewed":False,
            "new_physics_policy_inference_training":False,"interpolation_smoothing_alignment_height_correction":False,
            "artifacts":{name:sha256(output/name) for name in files}}
        write_json(output/"provenance.json",result);return result
    finally:runtime.renderer.close()


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-dir",type=Path,default=ROOT/"outputs/ground_contact_v2_20261023_v1/cem_mu0p7")
    result.add_argument("--case-manifest",type=Path,help="Explicit SHA-bound cases for later PPO/native recordings")
    result.add_argument("--reference",type=Path,default=runner.paired.DEFAULT_REFERENCE)
    result.add_argument("--model",type=Path,default=runner.paired.DEFAULT_MODEL)
    result.add_argument("--legacy-evaluator",type=Path,default=runner.paired.DEFAULT_LEGACY)
    result.add_argument("--output-dir",type=Path,required=True)
    result.add_argument("--skip-video",action="store_true")
    return result


if __name__=="__main__":run(parser().parse_args())

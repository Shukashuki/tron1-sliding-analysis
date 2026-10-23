"""NEW PPO-only original/V2 recorded-pose replay with actual per-case captions.

Default layout is four horizontal panels: old/new PhysX, then old/new MuJoCo.
Missing engines never become placeholder results; a two-panel partial render
requires --subset. Strict completed formal export/evaluation proofs are loaded
before any graphics runtime. No actors, physics, smoothing or pose correction.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","osmesa")
import numpy as np

import analyze_ppo_ground_contact_v2 as audit
import analyze_ppo_backend_400hz_matrix as metrics
from ppo_backend_400hz_contract import require,sha256,write_json
from render_physx_ppo_fullbody import build_qpos,create_runtime,mesh_provenance,_pose
from render_ppo_backend_400hz_nominal import diagnostics,replay_fk_audit
from render_ppo_beta_nominal import world_bounds
from render_ppo_fixed_follow import prepare_camera,apply_camera
import run_cem_ground_contact_v2 as assets

ROOT=Path(__file__).resolve().parents[1]
FRAMES,FPS=350,50
HEIGHT=850
COLORS={"body":"#1464b4","velocity":"#13733c","left":"#d62728","right":"#9467bd","unloaded":"#d1c9cf"}


def native_indices():return np.arange(FRAMES,dtype=np.int64)*8


def selected_engines(args):
    subset=getattr(args,"subset",None)
    require(subset in (None,"physx","mujoco"),"Unknown explicit subset")
    engines=(subset,) if subset else ("physx","mujoco")
    require(all(getattr(args,engine+"_dir",None) is not None for engine in engines),
        "Four panels require both NEW engine folders; use explicit --subset for a genuine two-panel partial comparison")
    return engines


def load_cases(args,reference):
    """Reuse strict analyzer provenance, not generic loose report labels."""
    engines=selected_engines(args);protocol=audit.load_protocol(args.protocol)
    traces,cases,evidence={},{},{}
    for engine in engines:
        new_raw,new_proof=audit.load_completed(getattr(args,engine+"_dir"),engine,args.protocol)
        old_raw,old_proof=audit.load_baseline(engine,args.baseline_root,args.baseline_summary)
        audit._compare_profiles(new_proof["contract"],old_proof["contract"])
        new_metrics,new_trace=audit.recorded_metrics(new_raw,engine,reference)
        old_metrics,old_trace=audit.recorded_metrics(old_raw,engine,reference)
        screen=audit.nominal_screen(new_metrics,old_metrics,protocol,engine)
        for prefix,trace,values,proof in (("baseline",old_trace,old_metrics,old_proof),("new",new_trace,new_metrics,new_proof)):
            key=prefix+"_"+engine
            # Visual support uses the same summed floor-normal-force operator as
            # the new acceptance rule. Legacy point-count thresholds never gate it.
            trace["contact_force"]=np.asarray(trace["floor_normal_force"]).copy()
            traces[key]=trace;evidence[key]=proof
            cases[key]={"id":key,"engine":engine,"variant":prefix,
                "title":("Original PPO" if prefix=="baseline" else "New contact-V2 PPO")+" | "+audit.BACKENDS[engine]+" | beta ON",
                "analysis":values,"new_nominal_screen":screen if prefix=="new" else None}
    if len(engines)==2:
        first,second=evidence["new_physx"],evidence["new_mujoco"]
        require(first["actor_sha256"]==second["actor_sha256"] and first["checkpoint_sha256"]==second["checkpoint_sha256"],
                "Both engines must evaluate the SAME new trained normalized actor")
    order=[prefix+"_"+engine for engine in engines for prefix in ("baseline","new")]
    return traces,cases,evidence,order


def contact_window(trace,index):
    require(type(index) is int and 0<=index<2793,"Invalid original native index")
    selected=np.array([0],dtype=int) if index==0 else np.arange(max(1,index-7),index+1)
    supported=np.max(trace["contact_force"][selected,:,2],axis=-1)>5.
    return {"indices":selected,"supported":supported,"supported_samples":int(supported.sum()),"samples":len(selected)}


def heading_glyphs(trace,index):
    signals=trace["signals"];root=trace["position"][index,:2]
    result=[{"point":root,"angle":signals["body_heading_rad"][index],"color":COLORS["body"],
             "valid":trace["body_heading_geometry_valid"][index],"axial":False,"kind":"body"},
            {"point":root,"angle":signals["course_rad"][index],"color":COLORS["velocity"],
             "valid":signals["speed_m_s"][index]>=.25,"axial":False,"kind":"COM_velocity"}]
    for side,name in enumerate(("left","right")):
        loaded=bool(trace["contact_force"][index,side,2]>5.)
        result.append({"point":trace["wheel_pos"][index,side,:2],
            "angle":signals["wheel_rolling_plane_heading_rad"][index,side],
            "color":COLORS[name] if loaded else COLORS["unloaded"],
            "valid":trace["wheel_heading_geometry_valid"][index,side],"axial":True,
            "kind":name+"_wheel","force_only_loaded":loaded})
    return result


def case_contact_caption(case):
    support=case["analysis"]["any_wheel_ground_support"]["full_contact"]
    return (f"Whole contact0.02-6.96s: any-wheel {100*support['any_wheel_support_sample_fraction']:.2f}% | "
            f"longest no-support {1000*support['longest_no_support_observed_s']:.1f}ms")


def outcome_caption(cases,order):
    pieces=[]
    for key in order:
        case=cases[key]
        if case["variant"]!="new":continue
        screen=case["new_nominal_screen"]
        verdict="PASS numerical screen" if screen["numeric_nominal_episode_pass"] else f"FAIL {len(screen['failed_checks'])} numerical checks"
        pieces.append(case["engine"]+": "+verdict)
    missing=[engine for engine in ("physx","mujoco") if "new_"+engine not in cases]
    if missing:pieces.append("NOT DISPLAYED: "+"/".join(missing))
    return "New PPO | "+" | ".join(pieces)+" | Overall task completion NOT claimed; qualitative review separate."


def draw_map(draw,trace,reference,index,box,bounds,font):
    low,high=np.asarray(bounds,dtype=float);left,top,right,bottom=box
    scale=min((right-left-22)/(high[0]-low[0]),(bottom-top-52)/(high[1]-low[1]));center=(low+high)/2
    pixel=lambda p:((left+right)/2+(p[0]-center[0])*scale,(top+bottom)/2-(p[1]-center[1])*scale)
    draw.rectangle(box,fill="#f7f9fa",outline="#c9cdd1")
    draw.text((left+8,top+2),"World XY: reference gray / path green; body blue / COM dark green",font=font,fill="#333333")
    draw.text((left+8,top+20),"Wheel axial heading: LEFT red / RIGHT purple; faint=Fz<=5N, not flight",font=font,fill="#333333")
    draw.line([pixel(p) for p in reference["position"][:,:2]],fill="#888888",width=2)
    if index>0:draw.line([pixel(p) for p in trace["position"][:index+1,:2]],fill="#15956c",width=2)
    for glyph in heading_glyphs(trace,index):
        if not glyph["valid"]:continue
        point,angle,axial=glyph["point"],glyph["angle"],glyph["axial"]
        delta=.24*np.array([np.cos(angle),np.sin(angle)])
        start=point-delta/2 if axial else point;stop=point+delta/2 if axial else point+delta
        draw.line((pixel(start),pixel(stop)),fill=glyph["color"],width=3)
        if not axial:
            tip=np.asarray(pixel(stop));direction=np.array([np.cos(angle),-np.sin(angle)]);normal=np.array([-direction[1],direction[0]])
            draw.polygon([tuple(tip),tuple(tip-7*direction+3*normal),tuple(tip-7*direction-3*normal)],fill=glyph["color"])
    draw.text((left+8,bottom-18),f"Same world scale X[{low[0]:.2f},{high[0]:.2f}]m Y[{low[1]:.2f},{high[1]:.2f}]m",font=font,fill="#555555")


def render_frame(runtime,traces,qposes,reference,cases,order,plan,bounds,frame):
    from PIL import Image,ImageDraw
    require(type(frame) is int and 0<=frame<FRAMES,"Invalid original50Hz frame")
    index=int(native_indices()[frame]);canvas=Image.new("RGB",(640*len(order),HEIGHT),"#fafafa");draw=ImageDraw.Draw(canvas)
    partial="EXPLICIT TWO-PANEL SUBSET" if len(order)==2 else "FOUR ACTUAL COMPLETED CASES"
    draw.text((12,4),f"PPO original vscontactV2 | mu=.7 | native400Hz / replay50Hz | t={index*.0025:.2f}s | {partial}",font=runtime.small,fill="#111111")
    for column,key in enumerate(order):
        trace,case,x=traces[key],cases[key],column*640
        _pose(runtime.model,runtime.data,qposes[key][index]);camera=apply_camera(runtime,plan,key,frame)
        runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
        canvas.paste(Image.fromarray(runtime.renderer.render().copy()),(x,58))
        draw.rectangle((x,30,x+640,57),fill="#eef1f4");draw.text((x+7,35),case["title"],font=runtime.font,fill="#111111")
        value=diagnostics(trace,reference,index);error="n/a afterreference" if value["xy_error_m"] is None else f"{value['xy_error_m']:.3f}m"
        signals=trace["signals"];beta=f"{np.degrees(signals['root_directed_beta_rad'][index]):.1f}deg" if signals["root_beta_valid"][index] else "unavailable"
        draw.text((x+8,425),f"XY {error} | COM speed {value['speed_m_s']:.2f}m/s | beta {beta} (nonzero intended)",font=runtime.small,fill="#111111")
        draw.text((x+8,447),f"Original height {value['height_m']:.3f}m | tilt {value['tilt_deg']:.1f}deg | ground FzL/R {value['fz'][0]:.0f}/{value['fz'][1]:.0f}N",font=runtime.small,fill="#111111")
        window=contact_window(trace,index)
        draw.text((x+8,469),f"Any-wheel Fz>5N: preceding native samples {window['supported_samples']}/{window['samples']} supported",font=runtime.small,fill="#111111")
        draw.text((x+8,491),"Dots = native2.5ms samples; one wheel may support",font=runtime.small,fill="#333333")
        for j,supported in enumerate(window["supported"]):
            draw.ellipse((x+440+22*j,493,x+454+22*j,507),fill="#25804b" if supported else "#b74949")
        draw.text((x+8,515),case_contact_caption(case),font=runtime.small,fill="#111111")
        draw_map(draw,trace,reference,index,(x+5,540,x+635,755),bounds,runtime.small)
    draw.text((12,765),outcome_caption(cases,order),font=runtime.small,fill="#8f2929")
    draw.text((12,787),"Contact fraction/gaps above are EACH case's actual wholecontact0.02-6.96s, not core1.44-4s. Single-wheel lift allowed; no point-count gate.",font=runtime.small,fill="#333333")
    draw.text((12,809),f"Opaque TRON1; fixedrelative camera {plan['distance_m']:.3f}m, fixedworldangles, no zoom/yawfollow. Original poses/heights; no physics/inference/interpolation.",font=runtime.small,fill="#333333")
    draw.text((12,831),"Red contact dots mean no measured wheel Fz>5N, not geometric flight. N=1. Improved tracking or lower HF alone does not establish stable contact.",font=runtime.small,fill="#333333")
    return canvas


def run(args):
    output=Path(args.output_dir);require(not output.exists(),"Preserve render; fresh output directory required")
    selected_engines(args)
    require(os.environ.get("MUJOCO_GL")=="osmesa","CPU OSMesa required")
    reference=metrics.reference_data(args.reference)
    traces,cases,evidence,order=load_cases(args,reference)
    kinematics,fk_source=metrics.load_fk(args.model,args.legacy_evaluator)
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
        replay={"native_indices":native_indices(),"fixed_camera_distance_m":np.asarray(plan["distance_m"])}
        for key in order:
            replay.update({key+"__qpos":qposes[key],key+"__native_time_s":traces[key]["time_s"],
                key+"__lookat":plan["lookat_by_case"][key],key+"__floor_normal_force_w":traces[key]["contact_force"]})
        np.savez_compressed(output/"replay_states.npz",**replay)
        render_frame(runtime,traces,qposes,reference,cases,order,plan,bounds,args.poster_frame).save(output/"comparison.png")
        files=["comparison.png","replay_states.npz"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream=imageio_ffmpeg.write_frames(str(output/"fullbody_comparison.mp4"),(640*len(order),HEIGHT),fps=FPS,
                codec="libx264",quality=7,macro_block_size=2,output_params=["-movflags","+faststart"])
            stream.send(None)
            try:
                for frame in range(FRAMES):
                    stream.send(np.asarray(render_frame(runtime,traces,qposes,reference,cases,order,plan,bounds,frame)))
                    if frame%100==0:print(f"PPO recorded-only OSMesa frame{frame}/{FRAMES}",flush=True)
            finally:stream.close()
            files.append("fullbody_comparison.mp4")
        result={"kind":"NEW_PPO_contact_V2_actual_case_caption_recorded_replay","case_order":order,"cases":cases,
            "input_evidence":evidence,"explicit_subset":args.subset,"full_four_panel_comparison":len(order)==4,
            "native_states_per_case":2793,"native_dt_s":.0025,"selected_stride":8,"video_frames":FRAMES,"video_fps":FPS,
            "video_generated":not args.skip_video,"video_size_pixels":[640*len(order),HEIGHT],
            "camera":{key:value for key,value in plan.items() if key!="lookat_by_case"},
            "camera_same_distance_all_cases_frames":True,"camera_yaw_follow":False,"camera_zoom":False,
            "world_bounds":bounds,"FK_source":fk_source,"replay_FK_audit":audits,"mesh_provenance":mesh_provenance(args.model),
            "model_sha256":sha256(args.model),"reference_sha256":sha256(args.reference),"protocol_sha256":sha256(args.protocol),
            "renderer_sha256":sha256(__file__),"contact_caption_source":"actual per-case analyzer metrics, never CEM constants",
            "contact_gate":"any wheel upward actual floor-normal Fz>5N, no pointcount gate",
            "contact_dot_native_samples_per_video_frame":8,"nonzero_beta_intended":True,"single_wheel_support_allowed":True,
            "robot_mesh_alpha":1.,"visual_reviewed":False,"goal_completion_claimed":False,
            "new_physics_policy_inference_training":False,"smoothing_interpolation_alignment_height_correction":False,
            "artifacts":{name:sha256(output/name) for name in files},
            "limitations":["N=1; numerical protocol screen and visual inspection are separate.",
                "50Hz poses are exact selections from400Hz native records; eight dots retain substep contact telemetry.",
                "All original poses/heights and later departures remain. No root/pose correction or geometric flight inference.",
                "Original and NEW policies are distinct trained actors; new SAME actor is checked across engines.",
                "Contact counts are not used for support coloring or acceptance; only actual summed upward floor-normal force.",
                "Reference ends6.96s; final6.98s XY error unavailable without extrapolation."]}
        write_json(output/"provenance.json",result);return result
    finally:runtime.renderer.close()


def parser():
    result=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    result.add_argument("--physx-dir",type=Path)
    result.add_argument("--mujoco-dir",type=Path)
    result.add_argument("--subset",choices=("physx","mujoco"),help="Explicit genuine two-panel partial comparison")
    result.add_argument("--protocol",type=Path,default=audit.DEFAULT_PROTOCOL)
    result.add_argument("--reference",type=Path,default=audit.DEFAULT_REFERENCE)
    result.add_argument("--baseline-root",type=Path,default=audit.BASELINE_ROOT)
    result.add_argument("--baseline-summary",type=Path,default=audit.BASELINE_SUMMARY)
    result.add_argument("--model",type=Path,default=assets.paired.DEFAULT_MODEL)
    result.add_argument("--legacy-evaluator",type=Path,default=assets.paired.DEFAULT_LEGACY)
    result.add_argument("--poster-frame",type=int,choices=range(FRAMES),default=150)
    result.add_argument("--skip-video",action="store_true")
    result.add_argument("--output-dir",type=Path,required=True)
    return result


if __name__=="__main__":run(parser().parse_args())

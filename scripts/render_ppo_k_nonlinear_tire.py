"""Four actual recorded-pose panels, fixed relative camera and common XY scale."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","osmesa")
import numpy as np

import analyze_ppo_k_nonlinear_tire as analysis
import render_ppo_ground_contact_v2 as graphics

GRAPHICS_SHA="0289c7da1764758cb79c184a7fbf9f5f0e9afd4adab23ef1ea700411be997efa"


def run(a):
    assert os.environ["MUJOCO_GL"]=="osmesa" and not a.output_dir.exists()
    assert analysis.sha(graphics.__file__)==GRAPHICS_SHA
    summary=analysis.read(a.analysis)
    assert summary["status"]=="completed_analysis" and summary["analyzer_sha256"]==analysis.sha(analysis.__file__)
    assert summary["kind"]=="K_nonlinear_trained_PPO_PhysX_law_and_MuJoCo_transfer"
    assert summary["reference_sha256"]==analysis.sha(a.reference)
    reference=graphics.metrics.reference_data(a.reference)
    rows,traces,_,proof,_=analysis.load_cases(a.collection,reference)
    assert rows==summary["rows"]
    kinematics,fk_source=graphics.metrics.load_fk(a.model,a.legacy_evaluator)
    cases={}
    for row in rows:
        key=row["id"];trace=traces[key];fk=kinematics.reconstruct(trace)
        trace.update(signals=fk["signals"],wheel_quaternion_wxyz=fk["wheel_quaternion_wxyz"],
                     wheel_center_velocity_w=fk["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"],trace["wheel_heading_geometry_valid"]=graphics.metrics.heading_geometry_masks(trace)
        cases[key]=dict(variant="baseline" if key=="K_native" else "new",engine=row["engine"],
            title=row["title"],analysis=row["metrics"],new_nominal_screen=row["nominal_screen"])
    runtime=graphics.create_runtime(a.model,width=640,height=360)
    try:
        qposes,checks={},{}
        for key in analysis.ORDER:
            qposes[key],mapping=graphics.build_qpos(runtime.model,traces[key])
            checks[key]=dict(graphics.replay_fk_audit(runtime.model,traces[key],qposes[key]),
                             named_joint_qpos_addresses=mapping.tolist())
        plan=graphics.prepare_camera(runtime,traces,qposes,list(analysis.ORDER),graphics.native_indices())
        bounds=[v.tolist() for v in graphics.world_bounds(traces,reference)]
        a.output_dir.mkdir(parents=True)
        from PIL import ImageDraw
        import imageio_ffmpeg
        writer=imageio_ffmpeg.write_frames(str(a.output_dir/"fullbody_comparison.mp4"),(2560,850),fps=50,
            codec="libx264",quality=7,macro_block_size=2,output_params=["-movflags","+faststart"])
        writer.send(None)
        try:
            for frame in range(350):
                image=graphics.render_frame(runtime,traces,qposes,reference,cases,list(analysis.ORDER),plan,bounds,frame)
                draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,2560,29),fill="#fafafa")
                draw.text((12,4),f"K reward / same PPO budget | nonlinear model learning and cross-law transfer | mu=.7 | beta ON | t={frame*.02:.2f}s",font=runtime.small,fill="#111111")
                draw.rectangle((0,763,2560,786),fill="#fafafa")
                message=" | ".join(row["title"]+": "+str(len(row["nominal_screen"]["failed_checks"]))+" failed numerical gates" for row in rows)
                draw.text((12,765),message+" | qualitative whole-body review required",font=runtime.small,fill="#8f2929")
                writer.send(np.asarray(image))
                if frame in (112,150,200,348):image.save(a.output_dir/(f"frame_{frame:03d}.png"))
                if frame==150:image.save(a.output_dir/"comparison.png")
                if frame%100==0:print(f"Nonlinear-trained PPO recorded replay frame={frame}/350",flush=True)
        finally:writer.close()
        states={"native_indices":graphics.native_indices(),"fixed_camera_distance_m":np.asarray(plan["distance_m"])}
        for key in analysis.ORDER:
            states[key+"__qpos"]=qposes[key]
            states[key+"__lookat"]=plan["lookat_by_case"][key]
            states[key+"__floor_normal_force_w"]=traces[key]["contact_force"]
        np.savez_compressed(a.output_dir/"replay_states.npz",**states)
        names=("fullbody_comparison.mp4","comparison.png","replay_states.npz",*(f"frame_{i:03d}.png" for i in (112,150,200,348)))
        result=dict(kind="four_panel_K_nonlinear_tire_trained_native_recorded_replay",case_order=list(analysis.ORDER),cases=cases,
            analysis_sha256=analysis.sha(a.analysis),analyzer_sha256=analysis.sha(analysis.__file__),renderer_sha256=analysis.sha(__file__),
            graphics_sha256=GRAPHICS_SHA,new_actor_sha256=proof["actor_sha256"],FK_source=fk_source,FK_readback_audits=checks,
            source_model_sha256=analysis.sha(a.model),reference_sha256=analysis.sha(a.reference),video_frames=350,video_fps=50,
            video_size_pixels=[2560,850],native_states_per_case=2793,world_bounds=bounds,
            fixed_camera_distance_m=plan["distance_m"],camera_same_distance_all_cases_frames=True,
            camera_yaw_follow=False,camera_zoom=False,camera_fixed_world_angles=True,
            smoothing_interpolation_alignment_height_correction=False,robot_mesh_alpha=1.,
            new_physics_policy_inference_training=False,visual_review_complete=False,task_success_claimed=False,
            artifacts={name:analysis.sha(a.output_dir/name) for name in names})
        (a.output_dir/"provenance.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        return result
    finally:runtime.renderer.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("collection","analysis","output-dir"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reference",type=Path,default=graphics.assets.paired.DEFAULT_REFERENCE)
    p.add_argument("--model",type=Path,default=graphics.assets.paired.DEFAULT_MODEL)
    p.add_argument("--legacy-evaluator",type=Path,default=graphics.assets.paired.DEFAULT_LEGACY)
    run(p.parse_args())

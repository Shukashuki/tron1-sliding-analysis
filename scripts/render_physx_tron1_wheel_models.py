"""Four horizontal fixed-relative-camera panels of measured K model transfers."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","osmesa")
import numpy as np
import analyze_physx_tron1_wheel_models as audit
import render_ppo_ground_contact_v2 as graphics

ROOT=Path(__file__).resolve().parents[1]
GRAPHICS="0289c7da1764758cb79c184a7fbf9f5f0e9afd4adab23ef1ea700411be997efa"


def run(a):
    assert os.environ["MUJOCO_GL"]=="osmesa" and not a.output_dir.exists()
    assert audit.sha(graphics.__file__)==GRAPHICS
    summary=json.loads(a.analysis.read_text())
    assert summary["status"]=="completed_analysis" and summary["analyzer_sha256"]==audit.sha(audit.__file__)
    reference=graphics.metrics.reference_data(a.reference);protocol=json.loads(a.protocol.read_text())
    assert audit.sha(a.reference)==summary["reference_sha256"] and audit.sha(a.protocol)==summary["nominal_protocol_sha256"]
    rows,traces,_,_,_=audit.load_cases(a.collection,reference,protocol,original_metrics=summary["original_baseline_metrics"])
    assert rows==summary["rows"]
    first="native_coulomb" if a.panel_set=="primary" else "external_coulomb_regularized"
    order=[g+"__"+law for law in (first,"combined_slip_magic_formula") for g in ("authored","slices9")]
    kinematics,fk_source=graphics.metrics.load_fk(a.model,a.legacy_evaluator)
    cases={}
    short={"native_coulomb":"Native Coulomb","external_coulomb_regularized":"External Coulomb","combined_slip_magic_formula":"Nonlinear tire"}
    for row in rows:
        key=row["id"];trace=traces[key];reconstruction=kinematics.reconstruct(trace)
        trace.update(signals=reconstruction["signals"],wheel_quaternion_wxyz=reconstruction["wheel_quaternion_wxyz"],
            wheel_center_velocity_w=reconstruction["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"],trace["wheel_heading_geometry_valid"]=graphics.metrics.heading_geometry_masks(trace)
        cases[key]=dict(variant="model_transfer",engine="PhysX",analysis=row["metrics"],
            title=short[row["law"]]+(" | authored cylinder" if row["geometry"]=="authored" else " | 9 slices"),
            new_nominal_screen=row["nominal_screen"])
    runtime=graphics.create_runtime(a.model,width=640,height=360)
    try:
        qposes={};fk={}
        for key,trace in traces.items():
            qposes[key],mapping=graphics.build_qpos(runtime.model,trace)
            fk[key]=dict(graphics.replay_fk_audit(runtime.model,trace,qposes[key]),named_joint_qpos_addresses=mapping.tolist())
        all_order=list(audit.NAMES)
        plan=graphics.prepare_camera(runtime,traces,qposes,all_order,graphics.native_indices())
        bounds=[v.tolist() for v in graphics.world_bounds(traces,reference)]
        a.output_dir.mkdir(parents=True)
        import imageio_ffmpeg
        from PIL import ImageDraw
        writer=imageio_ffmpeg.write_frames(str(a.output_dir/"fullbody_comparison.mp4"),(2560,850),fps=50,
            codec="libx264",quality=7,macro_block_size=2,output_params=["-movflags","+faststart"])
        writer.send(None)
        try:
            for frame in range(350):
                canvas=graphics.render_frame(runtime,traces,qposes,reference,cases,order,plan,bounds,frame)
                draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,2560,29),fill="#fafafa")
                draw.text((12,4),f"Same frozen K PPO | PhysX mu=.7 | matched contact offset1mm/rest0 | beta ON | {a.panel_set} | t={frame*.02:.2f}s",font=runtime.small,fill="#111111")
                draw.rectangle((0,763,2560,786),fill="#fafafa")
                label=" | ".join(cases[k]["title"]+": "+str(len(cases[k]["new_nominal_screen"]["failed_checks"]))+" failed checks" for k in order)
                draw.text((12,765),label+" | model transfer, no retraining; task completion not claimed",font=runtime.small,fill="#8f2929")
                writer.send(np.asarray(canvas))
                if frame==150:canvas.save(a.output_dir/"comparison.png")
                if frame%100==0:print(f"K wheel models {a.panel_set} recorded pose frame={frame}/350",flush=True)
        finally:writer.close()
        arrays={"native_indices":graphics.native_indices()}
        for key in order:
            arrays[key+"__qpos"]=qposes[key];arrays[key+"__lookat"]=plan["lookat_by_case"][key]
        np.savez_compressed(a.output_dir/"replay_states.npz",**arrays)
        result=dict(kind="four_panel_fixed_camera_frozen_K_wheel_model_replay",panel_set=a.panel_set,case_order=order,
            cases={k:cases[k] for k in order},analysis_sha256=audit.sha(a.analysis),analyzer_sha256=audit.sha(audit.__file__),
            renderer_sha256=audit.sha(__file__),graphics_sha256=GRAPHICS,actor_sha256=audit.ACTOR,
            video_frames=350,video_size_pixels=[2560,850],video_fps=50,camera_same_distance_all_cases_frames=True,
            camera_yaw_follow=False,camera_zoom=False,robot_mesh_alpha=1.,fixed_camera_distance_m=plan["distance_m"],
            camera_selected_over_all_six_recordings=True,world_bounds=bounds,FK_source=fk_source,FK_readback_audits=fk,
            source_model_sha256=audit.sha(a.model),reference_sha256=audit.sha(a.reference),
            smoothing_interpolation_alignment_height_correction=False,original_visual_meshes_retained_colliders_not_drawn=True,
            physics_policy_inference_training_executed=False,task_success_claimed=False,
            artifacts={n:audit.sha(a.output_dir/n) for n in ("comparison.png","fullbody_comparison.mp4","replay_states.npz")})
        (a.output_dir/"provenance.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    finally:runtime.renderer.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--collection",type=Path,required=True);p.add_argument("--analysis",type=Path,required=True)
    p.add_argument("--panel-set",choices=("primary","coupling"),required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--reference",type=Path,default=graphics.assets.paired.DEFAULT_REFERENCE)
    p.add_argument("--model",type=Path,default=graphics.assets.paired.DEFAULT_MODEL)
    p.add_argument("--legacy-evaluator",type=Path,default=graphics.assets.paired.DEFAULT_LEGACY)
    p.add_argument("--protocol",type=Path,default=ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    run(p.parse_args())

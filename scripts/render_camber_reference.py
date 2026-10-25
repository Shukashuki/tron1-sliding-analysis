"""Fixed-relative 4x1 kinematic camber replay;no inference or integration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","osmesa")
import numpy as np

import generate_camber_reference as reference
from render_physx_ppo_fullbody import create_runtime,_pose
from render_ppo_fixed_follow import prepare_camera,apply_camera


def load_cases(root):
    proof=json.loads((root/"generation.json").read_text());cases={}
    for row in proof["rows"]:
        p=root/(row["id"]+".npz")
        assert reference.sha(p)==row["reference_sha256"]
        with np.load(p,allow_pickle=False) as f:cases[row["id"]]={k:f[k].copy() for k in f.files}
    assert proof["frames"]==349 and len(cases)==4
    return proof,cases


def run(a):
    from PIL import Image,ImageDraw
    import imageio_ffmpeg
    assert not a.output_dir.exists() and os.environ["MUJOCO_GL"]=="osmesa"
    proof,cases=load_cases(a.references)
    assert reference.sha(a.model)==proof["model_sha256"]
    order=list(cases);indices=np.arange(349)
    traces={k:dict(position=c["qpos_mujoco"][:,:3]) for k,c in cases.items()}
    qposes={k:c["qpos_mujoco"] for k,c in cases.items()}
    runtime=create_runtime(a.model,width=640,height=360)
    try:
        plan=prepare_camera(runtime,traces,qposes,order,indices)
        cloud=np.concatenate([c["wheel_center_w"][...,:2].reshape(-1,2) for c in cases.values()])
        low,high=cloud.min(0)-.35,cloud.max(0)+.35
        for k in order:
            delta=plan["lookat_by_case"][k]-qposes[k][:,:3]
            np.testing.assert_allclose(delta,np.broadcast_to(delta[:1],delta.shape),atol=1e-12,rtol=0)
        a.output_dir.mkdir(parents=True)
        def frame(i):
            image=Image.new("RGB",(2560,820),"#fafafa");draw=ImageDraw.Draw(image)
            for j,row in enumerate(proof["rows"]):
                key=row["id"];c=cases[key];x=640*j
                _pose(runtime.model,runtime.data,qposes[key][i])
                runtime.renderer.update_scene(runtime.data,camera=apply_camera(runtime,plan,key,i),scene_option=runtime.option)
                image.paste(Image.fromarray(runtime.renderer.render().copy()),(x,35))
                draw.text((x+8,8),row["title"],font=runtime.font,fill="#111111")
                gamma=np.rad2deg(c["wheel_camber_rad"][i]);slip=c["wheel_point_lateral_velocity_m_s"][i]
                draw.text((x+8,402),f"t={i*.02:.2f}s | signed camber L/R {gamma[0]:+.2f}/{gamma[1]:+.2f} deg",font=runtime.small,fill="#111111")
                draw.text((x+8,425),f"Kinematic lateral speed L/R {slip[0]:+.3f}/{slip[1]:+.3f} m/s",font=runtime.small,fill="#111111")
                gap=c["wheel_geometric_gap_m"][i]*1000
                draw.text((x+8,448),f"Geometric wheel gaps L/R {gap[0]:.4f}/{gap[1]:.4f} mm; loads UNKNOWN",font=runtime.small,fill="#111111")
                draw.text((x+8,471),"Independent abduction;hip/knee contact reconstruction",font=runtime.small,fill="#333333")
                box=(x+8,501,x+632,750);l,t,r,b=box
                scale=min((r-l-30)/(high[0]-low[0]),(b-t-35)/(high[1]-low[1]));center=(low+high)/2
                pixel=lambda p:((l+r)/2+(p[0]-center[0])*scale,(t+b)/2-(p[1]-center[1])*scale)
                draw.rectangle(box,fill="#f3f6f8",outline="#c9cdd1")
                draw.line([pixel(p) for p in qposes[key][:,:2]],fill="#888888",width=2)
                draw.text((l+5,t+4),"Same world XY scale;body blue;L red;R purple",font=runtime.small,fill="#333333")
                for side,color in enumerate(("#d62728","#9467bd")):
                    draw.line([pixel(p) for p in c["wheel_center_w"][:i+1,side,:2]],fill=color,width=2)
                    p=c["wheel_center_w"][i,side,:2];axis=c["wheel_axle_w"][i,side]
                    forward=np.array([axis[1],-axis[0]]);forward/=np.linalg.norm(forward)
                    draw.line((pixel(p-.14*forward),pixel(p+.14*forward)),fill=color,width=4)
                root=qposes[key][i,:2];yaw=float(c["root_yaw_rad"][i])
                draw.line((pixel(root),pixel(root+.25*np.array([np.cos(yaw),np.sin(yaw)]))),fill="#1464b4",width=4)
            draw.text((12,768),"KINEMATIC REFERENCES ONLY | independent left/right camber | original body path,speed,yaw retained | no forces,physics rollout or policy training",font=runtime.small,fill="#8f2929")
            draw.text((12,792),f"Opaque original meshes | common fixed relative distance {plan['distance_m']:.3f}m | fixed world camera angles,no zoom | geometric contact does not prove bearing load",font=runtime.small,fill="#333333")
            return image
        writer=imageio_ffmpeg.write_frames(str(a.output_dir/"camber_reference_4x1.mp4"),(2560,820),fps=50,
            codec="libx264",quality=7,macro_block_size=2,output_params=["-movflags","+faststart"])
        writer.send(None)
        try:
            for i in indices:
                im=frame(int(i));writer.send(np.asarray(im))
                if i in (72,112,150,200,348):im.save(a.output_dir/f"frame_{i:03d}.png")
                if i==150:im.save(a.output_dir/"comparison.png")
                if i%100==0:print(f"kinematic reference frame={i}/349",flush=True)
        finally:writer.close()
        np.savez_compressed(a.output_dir/"camera_plan.npz",distance_m=plan["distance_m"],
            **{k+"__lookat":plan["lookat_by_case"][k] for k in order})
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,4,figsize=(22,8),layout="constrained",sharex=True,sharey="row")
        zero=cases[order[0]]["wheel_point_lateral_velocity_m_s"]
        ratios={f["id"]:f["target_signed_velocity_ratios_L_R"] for f in proof["fits"]}
        for j,row in enumerate(proof["rows"]):
            c=cases[row["id"]];t=c["time_s"]
            for side,name,color in ((0,"Left","#d62728"),(1,"Right","#9467bd")):
                axes[0,j].plot(t,np.rad2deg(c["wheel_camber_rad"][:,side]),color=color,label=name)
                axes[1,j].plot(t,c["wheel_point_lateral_velocity_m_s"][:,side],color=color,label=name)
                if row["id"] in ratios:
                    target=zero[:,side]*ratios[row["id"]][side]
                    core=(t>=1.44-1e-9)&(t<=4.+1e-9)
                    axes[1,j].plot(t[core],target[core],"--",color=color,alpha=.6,label=name+" target")
            axes[0,j].set_title(row["title"])
            for ax in axes[:,j]:ax.axvspan(1.44,4,color="gray",alpha=.12);ax.grid(alpha=.2);ax.legend(fontsize=7)
            axes[1,j].set_xlabel("Reference time [s]")
        axes[0,0].set_ylabel("Signed camber [deg]");axes[1,0].set_ylabel("Geometric point lateral velocity [m/s]")
        fig.savefig(a.output_dir/"camber_and_slip.png",dpi=150);plt.close(fig)
        artifacts={p.name:reference.sha(p) for p in a.output_dir.iterdir() if p.is_file()}
        result=dict(kind="independent_camber_kinematic_reference_replay",frames=349,fps=50,pixels=[2560,820],case_order=order,
            renderer_sha256=reference.sha(__file__),generation_sha256=reference.sha(a.references/"generation.json"),
            fixed_relative_camera_verified=True,fixed_distance_m=plan["distance_m"],fit_audit=plan["fit_audit"],
            source_model_sha256=reference.sha(a.model),opaque_meshes=True,physics_or_policy_steps=0,
            interpolation_or_pose_correction=False,manual_still_review_complete=False,task_success_claimed=False,artifacts=artifacts)
        (a.output_dir/"provenance.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    finally:runtime.renderer.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("references","model","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())

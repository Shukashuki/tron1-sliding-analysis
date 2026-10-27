#!/usr/bin/env python3
"""Render source / reconstructed human / actual TRON1 mesh FK / XY at fixed scale.

Software OSMesa rendering by default; no dynamics or contact forces are run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from render_hockey_gvhmr_comparison import skeleton, text, COLOR
from reconstruct_hockey_stop_gvhmr import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",type=Path,required=True)
    p.add_argument("--retarget",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--ffmpeg",default="ffmpeg")
    args=p.parse_args()
    os.environ.setdefault("MUJOCO_GL","osmesa")
    import mujoco
    r=json.loads((args.retarget/"report.json").read_text())
    trace=np.load(args.retarget/"retarget_trace.npz",allow_pickle=False)
    t=trace["time_s"];qpos=trace["qpos"];joints=trace["human_joints_zup_m"]
    model=mujoco.MjModel.from_xml_path(r["model_local_path"])
    if sha(Path(r["model_local_path"]))!=r["model_sha256"]:raise ValueError("Robot model changed")
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=440,width=640)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance=2.4;camera.azimuth=135.;camera.elevation=-18.
    xy=np.concatenate((qpos[:,:2],trace["root_target"][:,:2],trace["wheel_center"][...,:2].reshape(-1,2)))
    mid=(xy.min(0)+xy.max(0))/2;extent=max(np.ptp(xy,axis=0).max()*1.15,1.)
    def project(points):
        z=(np.asarray(points)-mid)*440/extent
        return np.stack((320+z[...,0],280-z[...,1]),axis=-1).astype(int)
    args.output.mkdir(parents=True,exist_ok=True)
    output=args.output/"source_human_gmr_xy_4x1.mp4"
    cap=cv2.VideoCapture(str(args.video))
    cmd=[args.ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","2560x640",
         "-r",str(1/np.median(np.diff(t))),"-i","pipe:0","-an","-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(output)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    floor=float(joints[0,[7,8,10,11],2].min())
    try:
        for i,now in enumerate(t):
            ok,frame=cap.read()
            if not ok:raise ValueError("Source shorter than motion")
            panels=[np.full((640,640,3),249,np.uint8) for _ in range(4)]
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];ratio=min(640/w,460/h)
            rgb=cv2.resize(rgb,(round(w*ratio),round(h*ratio)));y=70+(460-rgb.shape[0])//2;x=(640-rgb.shape[1])//2
            panels[0][y:y+rgb.shape[0],x:x+rgb.shape[1]]=rgb
            text(panels[0],"Source: uncut hockey-stop episode",(18,28),.6)
            text(panels[0],f"t={now:.3f} s / {len(t)} episode frames",(18,555),.54)
            text(panels[0],"Estimated motion; no measured contact force",(18,585),.49)
            skeleton(panels[1],joints[i],floor,"Raw GVHMR: human at robot leg scale")
            data.qpos[:]=qpos[i];mujoco.mj_forward(model,data)
            camera.lookat[:]=qpos[i,:3]
            renderer.update_scene(data,camera=camera)
            panels[2][70:510]=renderer.render()
            text(panels[2],"TRON1 WF: pinned GMR / mesh FK",(18,28),.6)
            text(panels[2],"Fixed follow distance / no physics step",(18,555),.54)
            text(panels[2],"Wheel spin placeholder = 0; not a controller",(18,585),.49)
            for name,c in (("root_target",(230,140,35)),("qpos",(40,40,40))):
                track=project(trace[name][:,:2]);cv2.polylines(panels[3],[track],False,c,2,cv2.LINE_AA)
                cv2.circle(panels[3],tuple(track[i]),5,c,-1)
            for side,c in enumerate(COLOR):
                center=trace["wheel_center"][i,side,:2];angle=np.radians(trace["wheel_heading_deg"][i,side])
                endpoints=project(np.array([center-.12*np.array([np.cos(angle),np.sin(angle)]),center+.12*np.array([np.cos(angle),np.sin(angle)])]))
                cv2.line(panels[3],tuple(endpoints[0]),tuple(endpoints[1]),c,4,cv2.LINE_AA)
            text(panels[3],"Reference / realized GMR XY + wheel headings",(18,28),.53)
            text(panels[3],f"Root XY fit RMSE: {r['root_XY_RMSE_m']:.3f} m",(18,540),.54)
            err=trace["wheel_heading_error_axial_deg"][i]
            text(panels[3],f"Wheel heading errors L/R: {err[0]:+.1f} / {err[1]:+.1f} deg",(18,568),.5)
            text(panels[3],"Near floor is geometry, not confirmed sliding",(18,598),.49)
            image=np.concatenate(panels,axis=1);proc.stdin.write(image.tobytes())
            if i==len(t)//2:cv2.imwrite(str(args.output/"comparison.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    finally:
        cap.release();renderer.close();proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError("FFmpeg failed")
    (args.output/"provenance.json").write_text(json.dumps({"renderer_sha256":sha(__file__),"retarget_sha256":sha(args.retarget/"retarget_trace.npz"),"source_sha256":sha(args.video),"frame_count":len(t),"physics_run":False,"software_rendering":os.environ.get("MUJOCO_GL"),"fixed_camera_distance_m":2.4,"fixed_camera_azimuth_deg":135,"fixed_camera_elevation_deg":-18},indent=2)+'\n')
    print(output)


if __name__=="__main__":main()

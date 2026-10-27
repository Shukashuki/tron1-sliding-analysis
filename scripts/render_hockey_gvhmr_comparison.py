#!/usr/bin/env python3
"""Fixed-scale 4x1 source/raw/postprocessed/XY preview of estimated GVHMR motion."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from reconstruct_hockey_stop_gvhmr import sha


EDGES = [(0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11),
         (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (13, 16),
         (16, 18), (18, 20), (9, 14), (14, 17), (17, 19), (19, 21)]
COLOR = [(220, 75, 55), (60, 110, 210)]


def text(img, message, xy, size=0.55, color=(30, 30, 30)):
    cv2.putText(img, message, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def skeleton(canvas, joints, floor, title):
    # Constant 170 pixels/meter, constant oblique camera direction; only follow
    # horizontal pelvis translation. Height is not re-centered every frame.
    root = joints[0]
    def project(points):
        p = np.asarray(points).copy()
        p[..., :2] -= root[:2]
        x = 320 + 170 * (0.8 * p[..., 0] - 0.6 * p[..., 1])
        y = 475 - 170 * (p[..., 2] - floor + 0.15 * p[..., 0] + 0.2 * p[..., 1])
        return np.stack((x, y), axis=-1).astype(int)
    for x in np.arange(-2, 2.01, 0.5):
        for points in ([[root[0]+x, root[1]-2, floor], [root[0]+x, root[1]+2, floor]],
                       [[root[0]-2, root[1]+x, floor], [root[0]+2, root[1]+x, floor]]):
            a, b = project(points)
            cv2.line(canvas, tuple(a), tuple(b), (222, 222, 222), 1)
    pixels = project(joints)
    for a, b in EDGES:
        c = COLOR[0] if a in (1,4,7,10,13,16,18,20) or b in (1,4,7,10,13,16,18,20) else COLOR[1]
        cv2.line(canvas, tuple(pixels[a]), tuple(pixels[b]), c, 3, cv2.LINE_AA)
    for i in range(22):
        cv2.circle(canvas, tuple(pixels[i]), 3, (30,30,30), -1)
    text(canvas, title, (18,28), .62)
    text(canvas, "Fixed follow offset / fixed scale", (18,555))
    text(canvas, "Estimated world; floor is a visual proxy", (18,582), .49)
    text(canvas, "No measured blade contact or forces", (18,608), .49)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable path (or ffmpeg on PATH)")
    args = p.parse_args()
    raw = np.load(args.reconstruction / "world_raw.npz", allow_pickle=False)
    pp = np.load(args.reconstruction / "world_standard_postproc.npz", allow_pickle=False)
    time = raw["time_s"]
    np.testing.assert_array_equal(time, pp["time_s"])
    conversion = np.array([[1,0,0], [0,0,-1], [0,1,0.]])
    joints = [r["joints_world_estimated_m"] @ conversion.T for r in (raw, pp)]
    floor = float(np.min(joints[0][0, [7,8,10,11], 2]))
    # One shared absolute XY box for both reconstructions and all frames.
    xy = np.concatenate([j[:, [0,7,8], :2].reshape(-1,2) for j in joints])
    mid = (xy.min(0)+xy.max(0))/2
    extent = max(float(np.ptp(xy,axis=0).max()), 1.)*1.15
    def project_xy(points):
        q = (np.asarray(points)-mid)/extent*450
        return np.stack((320+q[...,0], 300-q[...,1]),-1).astype(int)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "gvhmr_raw_vs_postproc_4x1.mp4"
    cap = cv2.VideoCapture(str(args.video))
    cmd = [args.ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
           "-s","2560x640","-r",str(1/np.median(np.diff(time))),"-i","pipe:0","-an",
           "-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, t in enumerate(time):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Video ends before reconstruction")
            panels = [np.full((640,640,3), 249, np.uint8) for _ in range(4)]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2]; ratio=min(640/w,460/h)
            rgb=cv2.resize(rgb,(int(w*ratio),int(h*ratio)))
            y,x=70+(460-rgb.shape[0])//2,(640-rgb.shape[1])//2
            panels[0][y:y+rgb.shape[0],x:x+rgb.shape[1]]=rgb
            text(panels[0],"Source video",(18,28),.65)
            text(panels[0],f"t={t:.3f} s / {len(time)} frames",(18,555))
            text(panels[0],"Monocular scale and camera need validation",(18,585),.49)
            skeleton(panels[1],joints[0][i],floor,"GVHMR raw: postproc=False")
            skeleton(panels[2],joints[1][i],floor,"GVHMR standard: postproc=True")
            for k,j in enumerate(joints):
                track=project_xy(j[:,0,:2]);cv2.polylines(panels[3],[track],False,COLOR[k],2,cv2.LINE_AA)
                for idx in (0,7,8):
                    cv2.circle(panels[3],tuple(project_xy(j[i,idx,:2])),5,COLOR[k],-1)
            text(panels[3],"Estimated world XY / common fixed scale",(18,28),.56)
            text(panels[3],"Red: raw   Blue: standard foot-lock",(18,555),.53)
            text(panels[3],"Dots: pelvis and both ankles",(18,582),.53)
            text(panels[3],f"Box span {extent:.2f} model-estimated m",(18,608),.49)
            image=np.concatenate(panels,axis=1)
            proc.stdin.write(image.tobytes())
            if i==len(time)//2:
                cv2.imwrite(str(args.output/"comparison.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    finally:
        cap.release();proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError("ffmpeg failed")
    report={"video":str(output),"source_sha256":sha(args.video),
            "renderer_sha256":sha(__file__),"frame_count":len(time),
            "raw_npz_sha256":sha(args.reconstruction/"world_raw.npz"),
            "postprocessed_npz_sha256":sha(args.reconstruction/"world_standard_postproc.npz"),
            "metric_world_trajectory_verified":False,"retarget_or_physics_render":False,
            "camera":"fixed oblique orientation and scale, follows XY only",
            "raw_pp_root_XY_difference_model_m_rms":float(np.sqrt(np.mean(np.sum((joints[0][:,0,:2]-joints[1][:,0,:2])**2,axis=-1))))}
    (args.output/"provenance.json").write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

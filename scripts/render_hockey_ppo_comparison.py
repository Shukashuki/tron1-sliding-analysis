"""GMR / two matched PPO rollouts / shared XY, in four horizontal panels.

Compose verified fixed-camera renders. Read native poses for the XY overlay;
perform no physics, smoothing, pose correction, retiming, or camera changes.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from evaluate_hockey_physx import load, sha

COLORS = ((185, 100, 35), (35, 115, 210))  # BGR: blue baseline, orange support.


def text(panel, value, y, *, scale=.52, color=(35, 35, 35)):
    cv2.putText(panel, value, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("tracking-render", "support-render", "tracking-run", "support-run", "command-dir", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--ffmpeg", required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    render_dirs = (a.tracking_render, a.support_render)
    run_dirs = (a.tracking_run, a.support_run)
    source = [d/"hockey_command_physx_xy_contact_4x1.mp4" for d in render_dirs]
    provenance = [json.loads((d/"provenance.json").read_text()) for d in render_dirs]
    traces = [load(d/"final_rollout.npz") for d in run_dirs]
    runs = [json.loads((d/"report.json").read_text()) for d in run_dirs]
    speed_study = all(r.get("kind") == "hockey_ppo_speed_success_v1" for r in runs)
    if speed_study and any(runs[0][k] != runs[1][k] for k in ("seed", "speed_scale", "case_id", "motion_sha256")):
        raise ValueError("Speed-study comparisons must pair the same seed, speed, case and reference")
    for side, (video, info, run_dir) in enumerate(zip(source, provenance, run_dirs)):
        run = json.loads((run_dir/"report.json").read_text())
        identity_ok = (run.get("method") == ("beyondmimic", "hockey")[side] if speed_study else
                       run["kind"] == "hockey_ppo_learning_v1" and run["variant"] == ("tracking", "support")[side])
        if not identity_ok or run["mode"] != "eval" or run["worlds"] != 1:
            raise ValueError("Require the two named direct-PPO validations")
        if sha(video) != info["video_sha256"] or sha(run_dir/"final_rollout.npz") != info["trace_sha256"]:
            raise ValueError("Changed render or physical trajectory")
        for key in ("fixed_camera", "frame_count", "fps", "command_diagnostics_sha256", "XY_center_m", "XY_extent_m"):
            if info[key] != provenance[0][key]:
                raise ValueError("Source display settings differ: "+key)
    if sha(a.command_dir/"command_diagnostics.npz") != provenance[0]["command_diagnostics_sha256"]:
        raise ValueError("Different GMR command")
    command = load(a.command_dir/"command_diagnostics.npz")
    count, fps = provenance[0]["frame_count"], provenance[0]["fps"]
    center = np.asarray(provenance[0]["XY_center_m"])
    extent = provenance[0]["XY_extent_m"]
    def project(points):
        v = (np.asarray(points)-center)*450/extent
        return np.stack((320+v[..., 0], 300-v[..., 1]), -1).astype(int)
    def arrow(panel, start, end, color, width=2):
        pixels = project([start, end])
        cv2.arrowedLine(panel, tuple(pixels[0]), tuple(pixels[1]), color, width, cv2.LINE_AA, tipLength=.25)
    background = np.full((640, 640, 3), 249, np.uint8)
    text(background, "Common-scale XY / body and wheel directions", 30, scale=.57)
    for axis in (0, 1):
        for tick in np.arange(np.floor((center[axis]-extent/2)*4)/4, center[axis]+extent/2, .25):
            line = np.array([center-extent/2, center+extent/2]); line[:, axis] = tick
            pixels = project(line)
            cv2.line(background, tuple(pixels[0]), tuple(pixels[1]), (229, 229, 229), 1)
    cv2.polylines(background, [project(command["qpos"][:, :2])], False, (70, 145, 35), 2, cv2.LINE_AA)
    text(background, "Green: designed reference", 535)
    text(background, "Blue: BM-style PPO | Orange: hockey PPO" if speed_study else
         "Blue: tracking PPO | Orange: support PPO", 560)
    text(background, "Arrows: body heading | bars: wheel planes", 585)
    text(background, "Black thin arrows: actual COM velocity direction", 610, scale=.48)
    videos = [cv2.VideoCapture(str(f)) for f in source]
    for video in videos:
        if not video.isOpened() or (int(video.get(cv2.CAP_PROP_FRAME_COUNT)), video.get(cv2.CAP_PROP_FPS),
            int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (count, fps, 2560, 640):
            raise ValueError("Unexpected source video dimensions or clock")
    a.output.mkdir(parents=True)
    path = a.output/"hockey_ppo_fullbody_xy_4x1.mp4"
    argv = [a.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "2560x640", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE)
    try:
        for frame_index in range(count):
            frames = []
            for video in videos:
                ok, image = video.read()
                if not ok:
                    raise ValueError("Source video ended early")
                frames.append(image)
            panel = background.copy()
            i = frame_index*8
            for trace, color in zip(traces, COLORS):
                cv2.polylines(panel, [project(trace["root_pos"][:i+1, :2])], False, color, 2, cv2.LINE_AA)
                root = trace["root_pos"][i, :2]
                w, x, y, z = trace["root_quat"][i]
                heading = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                arrow(panel, root, root+.22*np.array([np.cos(heading), np.sin(heading)]), color)
                v = trace["root_com_vel"][i, :2]; speed = np.linalg.norm(v)
                if speed >= .25:
                    arrow(panel, root, root+.3*v/speed, (45, 45, 45), 1)
                names = trace["body_names"].astype(str).tolist()
                for name in ("wheel_L_Link", "wheel_R_Link"):
                    j = names.index(name); c = trace["body_pos"][i, j, :2]
                    w, x, y, z = trace["body_quat"][i, j]
                    axle_xy = np.array([2*(x*y-w*z), 1-2*(x*x+z*z)])
                    length = np.linalg.norm(axle_xy)
                    if length <= 1e-8:
                        continue
                    direction = np.array([axle_xy[1], -axle_xy[0]])/length
                    pixels = project([c-.1*direction, c+.1*direction])
                    cv2.line(panel, tuple(pixels[0]), tuple(pixels[1]), color, 4, cv2.LINE_AA)
            text(panel, f"t = {frame_index/fps:.2f} s | grid = 0.25 m", 60)
            output = np.concatenate((frames[0][:, :640], frames[0][:, 640:1280], frames[1][:, 640:1280], panel), axis=1)
            proc.stdin.write(output.tobytes())
            if frame_index in (20, 40, 75, 115):
                cv2.imwrite(str(a.output/f"frame_{frame_index/fps:.2f}s.png"), output)
        if any(video.read()[0] for video in videos):
            raise ValueError("Unexpected trailing source frames")
    finally:
        for video in videos:
            video.release()
        proc.stdin.close()
    if proc.wait():
        raise RuntimeError("FFmpeg encoding failed")
    gif = a.output/"hockey_ppo_fullbody_xy_4x1.gif"
    subprocess.run([a.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-filter_complex",
        "fps=12.5,scale=1600:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer",
        "-loop", "0", str(gif)], check=True)
    check = cv2.VideoCapture(str(path)); decoded = 0
    while True:
        ok, image = check.read()
        if not ok:
            break
        if image.shape != (640, 2560, 3):
            raise ValueError("Output frame size changed")
        decoded += 1
    check.release()
    if decoded != count:
        raise ValueError("Output frame count mismatch")
    (a.output/"provenance.json").write_text(json.dumps(dict(script_sha256=sha(__file__),
        source_videos=[dict(path=str(f), sha256=sha(f)) for f in source],
        comparison="original_recipe_vs_hockey_recipe" if speed_study else "hockey_support_ablation",
        matched_case_id=runs[0].get("case_id"),matched_training_seed=runs[0].get("seed"),
        matched_speed_scale=runs[0].get("speed_scale"),
        frames_decoded=decoded, fps=fps, fixed_camera=provenance[0]["fixed_camera"],
        XY_center_m=center.tolist(), XY_extent_m=extent, time_remapping=False,
        pose_modification=False, physics_steps_performed=0, mp4_sha256=sha(path), gif_sha256=sha(gif)), indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()

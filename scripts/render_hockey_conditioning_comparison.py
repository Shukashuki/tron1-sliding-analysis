#!/usr/bin/env python3
"""Compose existing hockey probe renders; no simulation or pose modification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ffmpeg', required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cases = ('baseline', 'reference_dq', 'nominal_command')
    sources = [a.result / 'render' / c / 'hockey_command_physx_xy_contact_4x1.mp4' for c in cases]
    provenance = [json.loads((f.parent / 'provenance.json').read_text()) for f in sources]
    for f, info in zip(sources, provenance):
        if sha(f) != info['video_sha256']:
            raise ValueError(f'Changed source video: {f}')
        for key in ('fixed_camera', 'frame_count', 'fps', 'command_diagnostics_sha256',
                    'XY_center_m', 'XY_extent_m'):
            if info[key] != provenance[0][key]:
                raise ValueError(f'Incomparable source setting: {key}')
    count, fps = provenance[0]['frame_count'], provenance[0]['fps']
    videos = [cv2.VideoCapture(str(f)) for f in sources]
    for v in videos:
        if (not v.isOpened() or v.get(cv2.CAP_PROP_FRAME_COUNT) != count
                or v.get(cv2.CAP_PROP_FPS) != fps
                or v.get(cv2.CAP_PROP_FRAME_WIDTH) != 2560
                or v.get(cv2.CAP_PROP_FRAME_HEIGHT) != 640):
            raise ValueError('Unexpected source video dimensions or timing')
    a.output.mkdir(parents=True)
    out = a.output / 'hockey_conditioning_fullbody_4x1.mp4'
    cmd = [a.ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'bgr24', '-s', '2560x640', '-r', str(fps), '-i', 'pipe:0',
           '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    labels = ('GMR pose + designed stop (kinematics)',
              'PhysX: original hockey conditioning',
              'PhysX: replace only reference dq',
              'PhysX: replace reference q / dq / phase')
    try:
        for i in range(count):
            frames = []
            for v in videos:
                ok, frame = v.read()
                if not ok:
                    raise ValueError(f'Source ended early at frame {i}')
                frames.append(frame)
            panels = [frames[0][:, :640].copy()] + [f[:, 640:1280].copy() for f in frames]
            for panel, label in zip(panels, labels):
                panel[:55] = 249
                cv2.putText(panel, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            .60, (35, 35, 35), 1, cv2.LINE_AA)
            frame = np.concatenate(panels, axis=1)
            proc.stdin.write(frame.tobytes())
            if i in (round(.2 * fps), round(.8 * fps), round(1.5 * fps)):
                cv2.imwrite(str(a.output / f'frame_{i / fps:.2f}s.png'), frame)
        if any(v.read()[0] for v in videos):
            raise ValueError('Unexpected trailing frames')
    finally:
        for v in videos:
            v.release()
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('FFmpeg encoding failed')
    gif = a.output / 'hockey_conditioning_fullbody_4x1.gif'
    subprocess.run([a.ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(out),
                    '-filter_complex', 'fps=12.5,scale=1600:-1:flags=lanczos,split[a][b];'
                    '[a]palettegen[p];[b][p]paletteuse=dither=bayer', '-loop', '0', str(gif)], check=True)
    check = cv2.VideoCapture(str(out))
    decoded = 0
    while True:
        ok, frame = check.read()
        if not ok:
            break
        assert frame.shape == (640, 2560, 3)
        decoded += 1
    check.release()
    if decoded != count:
        raise ValueError('Output frame count mismatch')
    report = dict(script_sha256=sha(__file__), source_videos=[dict(case=c,
                  path=str(f), sha256=sha(f)) for c, f in zip(cases, sources)],
                  panel_order=list(labels), fps=fps, frames_decoded=decoded,
                  fixed_camera=provenance[0]['fixed_camera'], physics_steps_performed=0,
                  time_remapping=False, pose_modification=False, source_RGB_included=False,
                  mp4_sha256=sha(out), gif_sha256=sha(gif))
    (a.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Render only the left/right triangular friction cones on the original clock."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from evaluate_hockey_physx import load,sha
from plot_hockey_friction_from_start import animate_forces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads((args.analysis/'report.json').read_text())
    if sha(args.analysis/'signals.npz')!=report['artifact_sha256']['signals.npz']:
        raise ValueError('Audited signals changed')
    if len(report['cases'])!=1:raise ValueError('Select one identified policy')
    row=report['cases'][0]
    if not row['task_success'] or not np.allclose([row['mu_static'],row['mu_dynamic']],[.9,.7],atol=1e-9,rtol=0):
        raise ValueError('Require the selected successful split-friction case')
    raw=load(args.analysis/'signals.npz')
    s={k[6:]:v for k,v in raw.items() if k.startswith('case0_')}
    np.testing.assert_allclose(s['time_start_s'],np.arange(1104)*.0025,atol=1e-10,rtol=0)
    args.output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    video=animate_forces(s,row,args.output,'triangle',only_cones=True)
    import imageio_ffmpeg
    from PIL import Image
    gif=args.output/'friction_cones_preview.gif'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-threads','2',
        '-filter_complex_threads','1','-i',str(args.output/video['file']),'-filter_complex',
        "select='not(mod(n,4))',setpts=N/(12.5*TB),scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
        '-r','12.5','-loop','0',str(gif)]
    subprocess.run(command,check=True)
    with Image.open(gif) as im:
        duration=0
        for i in range(im.n_frames):im.seek(i);duration+=im.info['duration']
        if im.n_frames!=276 or duration!=22080:raise ValueError('Preview timing mismatch')
    result=dict(kind='hockey_triangle_pair_render_v1',source_sha256=sha(__file__),
        renderer_sha256=sha(Path(__file__).with_name('plot_hockey_friction_from_start.py')),
        analysis_report_sha256=sha(args.analysis/'report.json'),signals_sha256=sha(args.analysis/'signals.npz'),
        qualification_sha256=row['qualification_sha256'],raw_sha256=row['raw_sha256'],
        mu_static=row['mu_static'],mu_dynamic=row['mu_dynamic'],video=video,
        preview=dict(file=gif.name,sha256=sha(gif),frames=276,duration_ms=duration,
            selected_video_frames='0,4,8,...,1100',ffmpeg_command=command),
        change='Layout only; omit force-history panels, retain cone data, clocks and status readouts.',
        physics_steps=0,force_smoothing=False,
        artifact_sha256={p.name:sha(p) for p in sorted(args.output.iterdir())})
    (args.output/'provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(video),flush=True)


if __name__=='__main__':main()

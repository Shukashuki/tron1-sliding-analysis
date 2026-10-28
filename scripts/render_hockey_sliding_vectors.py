"""Fixed-ground view of temporally aligned wheel heading and material slip.

Render recorded data only. Velocity arrows share a fixed scale and use the
pre-step pose matching native contact geometry. Friction ratio is step-averaged.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from audit_hockey_contact_timing import CELLS
from evaluate_hockey_physx import load, q_matrix, sha


def text(image, value, y, x=14, scale=.48, color=(30, 30, 30)):
    cv2.putText(image,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    report=json.loads((a.audit/'report.json').read_text())
    assert report['script_sha256']==sha(Path(__file__).with_name('audit_hockey_contact_timing.py'))
    assert sha(a.audit/'aligned_signals.npz')==report['aligned_signals_sha256']
    archive=load(a.audit/'aligned_signals.npz')
    labels=('Earlier tracking PPO | 1x','Hockey PPO | 1x','Hockey PPO | 1.5x','Hockey PPO | 2x')
    cases=[];allxy=[]
    for name,label in zip(CELLS,labels):
        sig={k.split('__',1)[1]:v for k,v in archive.items() if k.startswith(name+'__')}
        row=next(c for c in report['cells'] if c['cell']==name)
        runpath=a.runs/name/'eval-suite/nominal/report.json'
        assert sha(runpath)==row['provenance']['run_report_sha256']
        motionpath=a.references/name.split('_')[1]/'motion.npz'
        assert sha(motionpath)==json.loads(runpath.read_text())['motion_sha256']
        ref=load(motionpath)['body_pos_w'][:,0,:2]
        sig['wheel_forward']=np.stack((q_matrix(sig['wheel_quat_pre'])[...,1,1],-q_matrix(sig['wheel_quat_pre'])[...,0,1]),axis=-1)
        sig['body_forward']=q_matrix(sig['root_quat_pre'])[...,:2,0]
        cases.append(dict(name=name,label=label,sig=sig,reference=ref))
        allxy.extend((sig['root_pos_pre'][:,:2],sig['wheel_pos_pre'][...,:2].reshape(-1,2),ref))
    xy=np.concatenate(allxy);center=(xy.min(0)+xy.max(0))/2
    extent=max(float(np.ptp(xy,axis=0).max()+1.8),4.)
    ppm=420/extent
    def project(xy):
        d=(np.asarray(xy)-center)*ppm
        return np.stack((320+d[...,0],295-d[...,1]),axis=-1).astype(int)
    for c in cases:
        bg=np.full((680,640,3),249,np.uint8)
        for axis in (0,1):
            lo=np.floor((center[axis]-extent/2)/.25)*.25
            for v in np.arange(lo,center[axis]+extent/2,.25):
                pts=np.tile(center,(2,1));pts[:,axis]=v;pts[:,1-axis]+=[-extent/2,extent/2]
                line=project(pts);cv2.line(bg,tuple(line[0]),tuple(line[1]),(228,228,228),1)
        cv2.polylines(bg,[project(c['reference'])],False,(155,155,155),2,cv2.LINE_AA)
        text(bg,c['label'],28,scale=.63)
        text(bg,'PhysX | mu=0.7 | seed42 | fixed ground / equal scale',53,scale=.43)
        text(bg,'Wheel planes: blue L / orange R | Body heading: red',544,scale=.42)
        text(bg,'Gray arrows: wheel COM v | Magenta: contact slip v',568,scale=.43)
        text(bg,'Arrow length: 0.25 m per 1 m/s | grid: 0.25 m',592,scale=.43)
        c['background']=bg
    def frame(index,slow):
        panels=[]
        for c in cases:
            s=c['sig'];panel=c['background'].copy();now=s['time_start_s'][index]
            for side,color in enumerate(((35,105,185),(205,105,30))):
                past=s['touching'][1:index+1,side]&s['lateral_sliding'][1:index+1,side]
                points=s['contact_point'][1:index+1,side,:2][past]
                for point in project(points):cv2.circle(panel,tuple(point),1,(190,70,145),-1)
                origin=s['wheel_pos_pre'][index,side,:2]
                direction=s['wheel_forward'][index,side];direction=direction/max(np.linalg.norm(direction),1e-8)
                pp=project([origin-.13*direction,origin+.13*direction])
                cv2.line(panel,tuple(pp[0]),tuple(pp[1]),color,4,cv2.LINE_AA)
                vc=s['wheel_com_velocity_pre'][index,side,:2]
                pp=project([origin,origin+.25*vc])
                cv2.arrowedLine(panel,tuple(pp[0]),tuple(pp[1]),(70,85,95),2,cv2.LINE_AA,tipLength=.2)
                if s['touching'][index,side]:
                    point=s['contact_point'][index,side,:2];vel=s['slip_velocity_pre'][index,side,:2]
                    pp=project([point,point+.25*vel])
                    cv2.arrowedLine(panel,tuple(pp[0]),tuple(pp[1]),(195,30,140),2,cv2.LINE_AA,tipLength=.2)
                    cv2.circle(panel,tuple(project(point)),4,(35,165,75),-1)
                    rho=s['rho'][index,side];rho=f'{rho:.2f}' if np.isfinite(rho) else 'n/a'
                    value=f"{'L' if side==0 else 'R'} TOUCH | lateral {s['lateral_rms_m_s'][index,side]:.2f} m/s | rho {rho}"
                else:
                    state='UNKNOWN' if s['contact_unknown'][index,side] else 'NO CONTACT'
                    value=f"{'L' if side==0 else 'R'} {state} | sliding not evaluated"
                text(panel,value,620+side*24,color=color,scale=.46)
            root=s['root_pos_pre'][index,:2];d=s['body_forward'][index];d=d/max(np.linalg.norm(d),1e-8)
            pp=project([root,root+.25*d]);cv2.arrowedLine(panel,tuple(pp[0]),tuple(pp[1]),(185,45,50),2,cv2.LINE_AA,tipLength=.3)
            text(panel,f"t={now:.4f} s | {'8x slow / every physical step' if slow else 'real time'}",80,scale=.46)
            text(panel,'Aligned contact/velocity; rho uses next-step force',669,scale=.42)
            panels.append(panel)
        return np.concatenate(panels,axis=1)
    a.output.mkdir(parents=True)
    def encode(path,indices,slow):
        process=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s','2560x680','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        try:
            for index in indices:
                image=frame(int(index),slow);process.stdin.write(image.tobytes())
                if index==329:cv2.imwrite(str(a.output/'aligned_sliding_0.82s.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        finally:process.stdin.close()
        if process.wait():raise RuntimeError('Encoding failed')
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==(680,2560,3);count+=1
        cap.release();assert count==len(indices) and fps==50.
        return dict(file=path.name,sha256=sha(path),frame_count=count,fps=fps)
    videos=[encode(a.output/'aligned_sliding_ground_view_4x1.mp4',np.arange(1,1106,8),False),
            encode(a.output/'aligned_sliding_braking_8x_slow_4x1.mp4',np.arange(161,722),True)]
    source=a.output/videos[1]['file'];gif=source.with_suffix('.gif')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i',str(source),'-filter_complex',
        'fps=12.5,scale=1600:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer',
        '-loop','0',str(gif)],check=True)
    provenance=dict(script_sha256=sha(__file__),audit_sha256=sha(a.audit/'report.json'),
        source_signals_sha256=sha(a.audit/'aligned_signals.npz'),physics_steps=0,
        common_fixed_ground_center_m=center.tolist(),common_fixed_ground_extent_m=extent,
        velocity_arrow_seconds=.25,contact_tolerance_m=.0001,
        velocity='Pre-step rigid material-point tangent velocity at native contact geometry; no smoothing',
        trail='Locations of reported touching lateral-sliding contact points; not a simulated tire mark or one material-point trajectory',
        slow_window_s=[.4,1.8],slowdown=8,slow_physical_stride=1,videos=videos,
        gif=dict(file=gif.name,sha256=sha(gif),fps=12.5))
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),videos=videos)))


if __name__=='__main__':main()

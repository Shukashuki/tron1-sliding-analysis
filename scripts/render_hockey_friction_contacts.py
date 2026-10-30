"""Render measured contact points/anchor forces and aggregate friction sections.

Every frame uses one archived 400 Hz interval; display at 50 fps gives 8x
slow motion. Positions/forces are never interpolated and no physics is run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np

from evaluate_hockey_physx import load,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report=json.loads((a.analysis/'report.json').read_text())
    if sha(a.analysis/'signals.npz')!=report['artifact_sha256']['signals.npz']:raise ValueError('Signals changed')
    s=load(a.analysis/'signals.npz');allpts=np.concatenate([s[f'case{k}_points'] for k in range(len(report['cases']))])
    allpts=allpts[allpts[:,0]>=.4-1e-9]
    bounds=(allpts[:,4:6].min(0)-.08,allpts[:,4:6].max(0)+.08)
    radius=max(1.1,max(float(np.nanmax(s[f'case{k}_rho_static'][s[f'case{k}_time_start_s']>=.4])) for k in range(len(report['cases']))))*1.02
    vlimit=max(float(s[f'case{k}_point_tangent_interval_rms'][s[f'case{k}_time_start_s']>=.4].max()) for k in range(len(report['cases'])))*1.08
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import cv2
    import imageio_ffmpeg
    a.output.mkdir(parents=True);outputs=[];colors=('#2779a5','#d37527')
    for k,row in enumerate(report['cases']):
        t=s[f'case{k}_time_start_s'];points=s[f'case{k}_points'];anchors=s[f'case{k}_anchors'];frames=np.flatnonzero(t>=.4-1e-9)
        fig,axs=plt.subplots(1,4,figsize=(20,4.5),dpi=128,constrained_layout=True)
        fig.suptitle(row['label']+' | 8x slow: every archived 2.5 ms interval\nNative normal points + friction anchors | Wheel-resultant sections; no per-point friction-cone assignment',fontsize=12)
        dots=[];segments=[];vectors=[];endpoints=[];texts=[];history=[]
        angle=np.linspace(0,2*np.pi,361)
        for side,color in enumerate(colors):
            p=points[(points[:,1]==side)&(points[:,0]>=.4)]
            axs[0].plot(p[:,4],p[:,5],color=color,alpha=.15,lw=.8)
            dots.append(axs[0].scatter([],[],c=color,s=40,animated=True,label=('L','R')[side]+' normal point'))
            collection=LineCollection([],colors=color,linewidths=2,animated=True);axs[0].add_collection(collection);segments.append(collection)
            ax=axs[side+1];ax.plot(np.cos(angle),np.sin(angle),'k-',label='Static boundary')
            ratio=row['mu_dynamic']/row['mu_static'];ax.plot(ratio*np.cos(angle),ratio*np.sin(angle),'k--',label='Dynamic boundary')
            vectors.append(ax.plot([],[],color=color,lw=2.5,animated=True)[0]);endpoints.append(ax.plot([],[],'o',color=color,ms=6,animated=True)[0])
            texts.append(ax.text(.03,.03,'',transform=ax.transAxes,fontsize=8,animated=True,
                                 bbox=dict(facecolor='white',alpha=.94,edgecolor='none',pad=2)))
            ax.set(title=('Left','Right')[side]+' wheel',xlabel='F_long / (mu_s Fn)',ylabel='F_lat / (mu_s Fn)',
                   xlim=(-radius,radius),ylim=(-radius,radius),aspect='equal');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='upper right')
            axs[3].plot(t,s[f'case{k}_point_tangent_interval_rms'][:,side],color=color,alpha=.12,lw=.8)
            history.append(axs[3].plot([],[],color=color,lw=1.1,animated=True,label=('Left','Right')[side])[0])
        axs[0].set(title='Touching points and anchor-force vectors',xlabel='World X [m]',ylabel='World Y [m]',
                   xlim=(bounds[0][0],bounds[1][0]),ylim=(bounds[0][1],bounds[1][1]),aspect='equal')
        axs[0].legend(fontsize=7,loc='upper left');axs[0].grid(alpha=.2)
        axs[0].text(.02,.02,'Force-vector scale: 0.0005 m/N\nBuffer identities are frame-local',transform=axs[0].transAxes,fontsize=7)
        axs[3].set(title='Material-point slip at touching points',xlabel='Time [s]',ylabel='Interval tangential RMS [m/s]',xlim=(.4,2.76),ylim=(0,vlimit));axs[3].grid(alpha=.2);axs[3].legend(fontsize=7)
        cursor=axs[3].axvline(.4,color='black',lw=1,animated=True)
        stamp=axs[3].text(.03,.95,'',transform=axs[3].transAxes,va='top',fontsize=9,animated=True)
        fig.canvas.draw();background=fig.canvas.copy_from_bbox(fig.bbox)
        width,height=fig.canvas.get_width_height();path=a.output/f'case{k}_contact_friction_8x_4x1.mp4'
        writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error',
            '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r','50','-i','pipe:0',
            '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        try:
            for j in frames:
                fig.canvas.restore_region(background)
                for side in (0,1):
                    p=points[(points[:,1]==side)&(points[:,2]==j)];f=anchors[(anchors[:,1]==side)&(anchors[:,2]==j)]
                    dots[side].set_offsets(p[:,4:6]);dots[side].set_sizes(35+np.clip(p[:,8],0,250)/3)
                    segments[side].set_segments(np.stack((f[:,4:6],f[:,4:6]+.0005*f[:,7:9]),axis=1))
                    valid=bool(s[f'case{k}_ratio_valid'][j,side]);fn=s[f'case{k}_normal_force_N'][j,side]
                    if valid:
                        cap=row['mu_static']*fn
                        x=s[f'case{k}_force_longitudinal_N'][j,side]/cap;y=s[f'case{k}_force_lateral_N'][j,side]/cap
                        vectors[side].set_data([0,x],[0,y]);endpoints[side].set_data([x],[y])
                        message=f"Fn={fn:.1f} N  |Ft|={s[f'case{k}_tangent_force_N'][j,side]:.1f} N\nrho_s={s[f'case{k}_rho_static'][j,side]:.3f}  rho_d={s[f'case{k}_rho_dynamic'][j,side]:.3f}"
                    else:
                        vectors[side].set_data([],[]);endpoints[side].set_data([],[]);message='Capacity ratio undefined'
                    texts[side].set_text(message)
                    selected=frames[frames<=j];history[side].set_data(t[selected],s[f'case{k}_point_tangent_interval_rms'][selected,side])
                    for artist,ax in [(dots[side],axs[0]),(segments[side],axs[0]),(vectors[side],axs[side+1]),(endpoints[side],axs[side+1]),(texts[side],axs[side+1]),(history[side],axs[3])]:
                        ax.draw_artist(artist)
                cursor.set_xdata([t[j],t[j]]);stamp.set_text(f't={t[j]:.4f} s → {t[j]+.0025:.4f} s\nCOM XY speed={s[f"case{k}_COM_speed"][j]:.3f} m/s')
                axs[3].draw_artist(cursor);axs[3].draw_artist(stamp)
                frame=cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()),cv2.COLOR_RGBA2BGR);writer.stdin.write(frame.tobytes())
                if j in (round(.5275/.0025),round(1.2025/.0025),round(2.5/.0025)):
                    cv2.imwrite(str(a.output/f'case{k}_{t[j]:.4f}s.png'),frame)
        finally:
            writer.stdin.close();encoder_status=writer.wait();plt.close(fig)
        if encoder_status!=0:raise RuntimeError('Video encoder failed')
        video=cv2.VideoCapture(str(path));count=0
        while True:
            ok,frame=video.read()
            if not ok:break
            if frame.shape[:2]!=(height,width):raise ValueError('Decoded frame size mismatch')
            count+=1
        fps=video.get(cv2.CAP_PROP_FPS);video.release()
        if count!=len(frames) or fps!=50:raise ValueError('Incomplete video')
        outputs.append(dict(file=path.name,sha256=sha(path),decoded_frames=count,width=width,height=height,fps=fps,
                            first_interval_s=float(t[frames[0]]),last_interval_end_s=float(t[frames[-1]]+.0025)))
        print(json.dumps(outputs[-1]),flush=True)
    metadata=dict(kind='hockey_native_contact_friction_animation_v1',source_sha256=sha(__file__),
        analysis_report_sha256=sha(a.analysis/'report.json'),signals_sha256=sha(a.analysis/'signals.npz'),
        interpolation=False,physics_steps=0,slowdown=8,force_arrow_scale_m_per_N=.0005,
        shared_world_bounds=[v.tolist() for v in bounds],shared_normalized_capacity_radius=radius,
        shared_slip_speed_limit_m_s=vlimit,videos=outputs,
        limitations=['Normal-point dots and friction-anchor vectors are separate measured objects; no point-to-anchor load matching.',
            'One frame shows pre-step geometry and the force averaged over the following interval.',
            'Faint footprint/trace includes the recorded future; colored current points and growing lines mark current time.'])
    (a.output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()

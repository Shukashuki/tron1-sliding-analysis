#!/usr/bin/env python3
"""Show all native force intervals, including the initial wheel-spin transient.

Uses the existing audited contact analysis. The first force is the average over
[0, 2.5 ms], not the zero-filled force slot at the unstepped reset state.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, q_matrix, sha


def section_coordinates(fn,ft,flong,flat,valid,mu_static,view):
    """The radial triangle uses the full tangent norm, never one projection."""
    fn,ft,flong,flat=np.broadcast_arrays(fn,ft,flong,flat)
    valid=np.asarray(valid,bool)&np.isfinite(fn)&(fn>1e-6)
    if view=='triangle':
        return np.where(valid,ft,np.nan),np.where(valid,fn,np.nan)
    if view!='circle':raise ValueError('Unknown friction section')
    cap=mu_static*fn
    return (np.divide(flong,cap,out=np.full(fn.shape,np.nan),where=valid),
            np.divide(flat,cap,out=np.full(fn.shape,np.nan),where=valid))


def draw_section(ax,mu_s,mu_d,view,fn_limit,radius=1.12):
    if view=='triangle':
        y=np.array([0.,fn_limit])
        ax.fill_betweenx(y,-mu_s*y,mu_s*y,color='#cfd9e4',alpha=.28)
        ax.fill_betweenx(y,-mu_d*y,mu_d*y,color='#d7eadb',alpha=.55)
        for mu,style,label in ((mu_s,'-','Static boundary'),(mu_d,'--','Dynamic boundary')):
            ax.plot([-mu*fn_limit,0,mu*fn_limit],[fn_limit,0,fn_limit],color='black',ls=style,lw=1.3,label=label)
        ax.set(xlim=(-1.1*mu_s*fn_limit,1.1*mu_s*fn_limit),ylim=(0,fn_limit),
               xlabel='Tangential magnitude |Ft| [N] (right half)',ylabel='Normal force Fn [N]',aspect='equal')
    else:
        angle=np.linspace(0,2*np.pi,361)
        ax.plot(np.cos(angle),np.sin(angle),'k-',lw=1.3,label='Static boundary')
        ax.plot(mu_d/mu_s*np.cos(angle),mu_d/mu_s*np.sin(angle),'k--',lw=1.3,label='Dynamic boundary')
        ax.set(xlim=(-radius,radius),ylim=(-radius,radius),
               xlabel='F_long / (mu_s Fn)',ylabel='F_lat / (mu_s Fn)',aspect='equal')


def contact_jump_summary(s,trace,geometry):
    points=s['points'];t=s['time_start_s'];n=len(t);rows=[]
    for side in (0,1):
        p=points[points[:,1]==side];counts=np.bincount(p[:,2].astype(int),minlength=n)
        center=np.stack([p[p[:,2]==i,4:7].mean(0) for i in range(n)])
        distance=np.linalg.norm(np.diff(center[:,:2],axis=0),axis=-1);j=int(np.argmax(distance))
        name='wheel_'+('L','R')[side]+'_Link';bid=list(trace['body_names']).index(name)
        rotations=q_matrix(trace['body_quat'][:,bid])
        local=np.stack([(center[i]-trace['body_pos'][i,bid])@rotations[i] for i in (j,j+1)])
        within=np.diff(p[:,2])==0;flat=np.linalg.norm(np.diff(p[:,4:6],axis=0),axis=-1)
        g=next(x for x in geometry['prototype'] if x['body'].endswith('/'+name))
        rows.append(dict(wheel=('L','R')[side],sampled_contact_all=bool(s['touching_states'][:,side].all()),
            point_count_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(counts,return_counts=True))},
            same_frame_point_separation_max_m=float(flat[within].max()) if within.any() else None,
            authored_width_m=g['total_width_m'],largest_centroid_jump=dict(
                interval_start_s=float(t[j]),interval_end_s=float(t[j+1]),world_XY_jump_m=float(distance[j]),
                wheel_center_XY_displacement_m=float(np.linalg.norm(trace['body_pos'][j+1,bid,:2]-trace['body_pos'][j,bid,:2])),
                point_counts=counts[j:j+2].tolist(),camber_deg=s['wheel_camber_deg'][j:j+2,side].tolist(),
                wheel_local_y_m=local[:,1].tolist(),wheel_local_y_change_m=float(local[1,1]-local[0,1]))))
    return dict(wheels=rows,contact_tolerance_m=.0001,
        old_background='Ragged contact samples were joined in buffer order, including different points within one time step.',
        corrected_background='Unconnected point cloud; no invented persistent point identities or cross-frame connecting lines.',
        centroid_is_material_point=False,force_or_slip_computation_uses_centroid_differences=False)


def animate_forces(s,row,output,cone_view='circle'):
    """Growing raw force histories and current wheel-resultant sections."""
    import subprocess
    import cv2
    import imageio_ffmpeg
    import matplotlib.pyplot as plt

    t=s['time_start_s'];end=s['time_end_s'];n=len(t);mu_s=row['mu_static'];mu_d=row['mu_dynamic']
    fn=s['normal_force_N'];ft=s['tangent_force_N'];colors=('#2079ac','#da7628')
    horizon=float(end[-1]);phi=np.linspace(0,2*np.pi,361)
    fig,axs=plt.subplots(1,4,figsize=(24,5.4),dpi=120)
    fig.subplots_adjust(left=.035,right=.992,bottom=.25,top=.79,wspace=.24)
    fig.suptitle('Continuous friction evolution | Successful PhysX PPO | mu_s=0.9, mu_d=0.7 | 8x slow, every 400 Hz interval',y=.985,fontsize=15)
    stamp=fig.text(.5,.9,'',ha='center',fontsize=12,animated=True)
    histories=[];cursors=[];force_dots=[];vectors=[];tips=[];tails=[];readouts=[];data=[]
    limit=max(1.1,float(np.nanmax(s['rho_static']))*1.05)
    fn_limit=float(np.ceil(fn.max()*1.08/20)*20)
    for side,color in enumerate(colors):
        ax=axs[side]
        for start,stop,c,label in ((0,.4,'#d8e6f3','Approach'),(.4,1.8,'#f7e6cd','Braking'),(1.8,horizon,'#dcefe1','Hold')):
            ax.axvspan(start,stop,color=c,alpha=.35)
            ax.text((start+stop)/2,.98,label,ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=8)
        curves=[];values=[]
        for value,c,ls,lw,label in ((fn[:,side],'#77818a','-',1.,'Normal Fn'),
            (ft[:,side],color,'-',1.6,'Tangential |Ft|'),(mu_s*fn[:,side],'#222222','--',1.,'Static capacity'),
            (mu_d*fn[:,side],'#458451',':',1.3,'Dynamic capacity')):
            curves.append(ax.plot([],[],color=c,ls=ls,lw=lw,drawstyle='steps-post',label=label,animated=True)[0]);values.append(value)
        histories.append(curves);data.append(values)
        cursors.append(ax.axvline(0,color='#9b3541',lw=.8,animated=True))
        force_dots.append(ax.plot([],[],'o',color=color,ms=4,animated=True)[0])
        ax.set(title=('Left','Right')[side]+' wheel: force history',xlim=(0,horizon),
               ylim=(0,float(fn.max())*1.08),xlabel='Time since reset [s]',ylabel='Force [N]')
        ax.legend(loc='lower left',bbox_to_anchor=(0,-.43),ncol=2,fontsize=9,frameon=False)
        ax.grid(alpha=.2)
        ax=axs[side+2]
        draw_section(ax,mu_s,mu_d,cone_view,fn_limit,limit)
        x,y=section_coordinates(fn[:,side],ft[:,side],s['force_longitudinal_N'][:,side],
            s['force_lateral_N'][:,side],s['ratio_valid'][:,side],mu_s,cone_view)
        ax.plot(x[0],y[0],'*',color='#ca3348',ms=9,label='First interval')
        tails.append((ax.plot([],[],color=color,alpha=.25,lw=1.,animated=True)[0],x,y))
        vectors.append(ax.plot([],[],color=color,lw=2.2,animated=True)[0])
        tips.append(ax.plot([],[],'o',color=color,ms=6,animated=True)[0])
        ax.set_title(('Left','Right')[side]+' wheel: '+('friction cone' if cone_view=='triangle' else 'friction section'))
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.13),ncol=3,fontsize=8,frameon=False);ax.grid(alpha=.2)
        position=ax.get_position()
        readouts.append(fig.text(position.x0,.048,'',fontsize=9,animated=True))
    fig.text(.5,.012,'Wheel-resultant diagnostics, not per-contact-point cones or observed solver stick/slip flags. Forces are interval averages; no smoothing.',ha='center',fontsize=8,color='#56616c')
    fig.canvas.draw();background=fig.canvas.copy_from_bbox(fig.bbox)
    width,height=fig.canvas.get_width_height();path=output/'force_evolution_from_start_8x_4x1.mp4'
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error',
        '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','50','-i','pipe:0',
        '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for j in range(n):
            fig.canvas.restore_region(background)
            phase='APPROACH' if t[j]<.4-1e-9 else ('BRAKING' if t[j]<1.8-1e-9 else 'STOP / HOLD')
            stamp.set_text(f'{phase}   |   Interval {t[j]:.4f}-{end[j]:.4f} s   |   System COM XY speed {s["COM_speed"][j]:.3f} m/s')
            fig.draw_artist(stamp)
            xs=np.r_[t[:j+1],end[j]]
            for side in (0,1):
                for artist,values in zip(histories[side],data[side]):
                    artist.set_data(xs,np.r_[values[:j+1],values[j]]);axs[side].draw_artist(artist)
                cursors[side].set_xdata([end[j],end[j]]);axs[side].draw_artist(cursors[side])
                force_dots[side].set_data([end[j]],[ft[j,side]]);axs[side].draw_artist(force_dots[side])
                trail,x,y=tails[side];trail.set_data(x[max(0,j-39):j+1],y[max(0,j-39):j+1])
                vectors[side].set_data([0,x[j]],[0,y[j]]);tips[side].set_data([x[j]],[y[j]])
                for artist in (trail,vectors[side],tips[side]):axs[side+2].draw_artist(artist)
                contact=bool(s['contact_at_both_endpoints'][j,side])
                pre=s['point_tangent_pre_rms'][j,side];interval=s['point_tangent_interval_rms'][j,side]
                state=('SLIDING' if min(pre,interval)>.05 else ('LOW SLIP' if max(pre,interval)<.02 else 'MIXED / TRANSIENT')) if contact else 'CONTACT INTERRUPTED'
                readouts[side].set_text(f'Fn={fn[j,side]:.1f} N   |Ft|={ft[j,side]:.1f} N   rho_s={s["rho_static"][j,side]:.3f}   rho_d={s["rho_dynamic"][j,side]:.3f}\n'
                    f'{"TOUCH at both endpoints" if contact else "NO persistent contact"} | {state}\n'
                    f'Tangential slip pre / interval: {pre:.3f} / {interval:.3f} m/s')
                fig.draw_artist(readouts[side])
            image=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();writer.stdin.write(image.tobytes())
            if j in (0,4,40,160,480,1000):
                cv2.imwrite(str(output/f'force_animation_{t[j]:.4f}s.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            if j%200==0:print(f'Force animation interval {j}/{n-1}',flush=True)
    finally:
        writer.stdin.close();status=writer.wait();plt.close(fig)
    if status:raise RuntimeError('Force-animation encoding failed')
    cap=cv2.VideoCapture(str(path));frames=0;fps=cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok,image=cap.read()
        if not ok:break
        if image.shape!=(height,width,3):raise ValueError('Wrong encoded frame size')
        frames+=1
    cap.release()
    if frames!=n or fps!=50:raise ValueError('Incomplete force animation')
    return dict(file=path.name,sha256=sha(path),decoded_frames=frames,fps=fps,slowdown=8,
                encoded_duration_s=frames/fps,first_interval_s=[float(t[0]),float(end[0])],
                last_interval_s=[float(t[-1]),float(end[-1])],frame_interpolation=False,
                force_history='Growing staircase at original 2.5 ms interval resolution',
                cone_trail_s=.1,force_smoothing=False,cone_view=cone_view,
                radial_triangle_uses_full_tangent_magnitude=cone_view=='triangle')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','nominal','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cone-view',choices=('circle','triangle'),default='circle')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads((args.analysis/'report.json').read_text())
    if sha(args.analysis/'signals.npz')!=report['artifact_sha256']['signals.npz']:
        raise ValueError('Audited analysis changed')
    if len(report['cases'])!=1:raise ValueError('Select one identified policy')
    row=report['cases'][0]
    checked=json.loads((args.nominal.parent/'nominal_qualification.json').read_text())
    run=json.loads((args.nominal/'report.json').read_text())
    if (row['qualification_sha256']!=sha(args.nominal.parent/'nominal_qualification.json')
        or checked['run_report_sha256']!=sha(args.nominal/'report.json')):
        raise ValueError('Qualification/recording mismatch')
    for name,digest in row['raw_sha256'].items():
        if sha(args.nominal/name)!=digest or run['artifacts'][name]!=digest:
            raise ValueError('Recording changed')
    raw=load(args.analysis/'signals.npz')
    s={key[len('case0_'):]:value for key,value in raw.items() if key.startswith('case0_')}
    trace=load(args.nominal/'final_rollout.npz')
    t=s['time_start_s'];dt=float(trace['physics_dt_s']);n=len(t)
    np.testing.assert_allclose(t,np.arange(n)*dt,atol=1e-10,rtol=0)
    np.testing.assert_allclose(s['time_end_s'],t+dt,atol=1e-10,rtol=0)
    np.testing.assert_allclose(s['normal_force_N'],trace['normal_fz'][1:n+1],atol=1e-4,rtol=0)
    np.testing.assert_allclose(s['tangent_force_N'],np.linalg.norm(trace['friction_force_w'][1:n+1,:,:2],axis=-1),atol=1e-4,rtol=0)
    mu_s,mu_d=row['mu_static'],row['mu_dynamic']
    if not np.allclose([mu_s,mu_d],[.9,.7],atol=1e-9,rtol=0) or not row['task_success']:
        raise ValueError('This presentation selects the successful static/dynamic 0.9/0.7 case')
    fn=s['normal_force_N'];ft=s['tangent_force_N'];horizon=float(t[-1]+dt)
    static=mu_s*fn;dynamic=mu_d*fn
    effective=np.full(fn.shape,np.nan)
    np.divide(ft,fn,out=effective,where=s['ratio_valid'])
    points=s['points']
    # RMS of individual material-point lateral displacements. Do not average
    # signed point velocities before squaring (opposing velocities can cancel).
    lat_interval=np.full(fn.shape,np.nan)
    for side in (0,1):
        p=points[points[:,1]==side];index=p[:,2].astype(int)
        count=np.bincount(index,minlength=n);total=np.bincount(index,weights=p[:,12]**2,minlength=n)
        np.sqrt(np.divide(total,count,out=np.full(n,np.nan),where=count>0),out=lat_interval[:,side])
    tangent_slip=(s['contact_at_both_endpoints'] & (s['point_tangent_pre_rms']>.05)
                  & (s['point_tangent_interval_rms']>.05))
    lateral_slip=(s['contact_at_both_endpoints'] & (s['pre_lateral_speed_m_s']>.05)
                  & (lat_interval>.05))
    joint_names=trace['joint_names'].astype(str).tolist()
    ids=[joint_names.index('wheel_'+side+'_Joint') for side in ('L','R')]
    spin=trace['joint_vel'][:n,ids]
    np.testing.assert_allclose(spin,s['wheel_spin_rad_s'],atol=1e-8,rtol=0)
    args.output.mkdir(parents=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('#2079ac','#da7628')
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'legend.fontsize':8})
    def phase_shading(ax):
        for start,end,color in ((0,.4,'#d8e6f3'),(.4,1.8,'#f7e6cd'),(1.8,horizon,'#dcefe1')):
            ax.axvspan(start,end,color=color,alpha=.35,zorder=-10)
        for edge in (.4,1.8):ax.axvline(edge,color='gray',ls=':',lw=.8)
    def force_panel(ax,side,end):
        mask=t<end-1e-10
        edges=np.r_[t[mask],min(end,float(s['time_end_s'][mask][-1]))]
        for values,color,ls,lw,label in ((fn,'#67727b','-',.8,'Normal Fn'),
            (ft,colors[side],'-',1.1,'Tangential |Ft|'),(static,'#222222','--',.9,'Static capacity: 0.9 Fn'),
            (dynamic,'#458451',':',1.1,'Dynamic capacity: 0.7 Fn')):
            ax.stairs(values[mask,side],edges,color=color,ls=ls,lw=lw,label=label,baseline=None)
        phase_shading(ax);ax.set(xlim=(0,end),ylim=(0,float(fn.max())*1.08),
            title=('Left','Right')[side]+' wheel: force and capacity',xlabel='Time since reset [s]',ylabel='Force [N]')
        ax.grid(alpha=.2);ax.legend(loc='upper right')
    fig,axs=plt.subplots(1,4,figsize=(24,5.4),constrained_layout=True)
    for side in (0,1):force_panel(axs[side],side,horizon)
    for side,color in enumerate(colors):
        axs[2].stairs(effective[:,side],np.r_[t,horizon],color=color,lw=.9,label=('Left','Right')[side],baseline=None)
        axs[3].stairs(s['point_tangent_pre_rms'][:,side],np.r_[t,horizon],color=color,lw=.9,label=('L','R')[side]+' tangential',baseline=None)
        axs[3].stairs(s['pre_lateral_speed_m_s'][:,side],np.r_[t,horizon],color=color,ls=':',lw=.75,alpha=.8,label=('L','R')[side]+' lateral',baseline=None)
    axs[2].axhline(mu_s,color='black',ls='--',lw=1,label='mu_s = 0.9')
    axs[2].axhline(mu_d,color='#458451',ls=':',lw=1.2,label='mu_d = 0.7')
    axs[2].set(title='Friction magnitude / normal force',ylabel='|Ft| / Fn',ylim=(0,max(mu_s*1.15,float(np.nanmax(effective))*1.08)))
    axs[3].axhline(.05,color='gray',ls='--',lw=.8,label='Slip threshold 0.05 m/s')
    axs[3].set(title='Material slip while touching',ylabel='Pre-step point RMS speed [m/s]',ylim=(0,float(s['point_tangent_pre_rms'].max())*1.12))
    for ax in axs[2:]:phase_shading(ax);ax.set(xlim=(0,horizon),xlabel='Time since reset [s]');ax.grid(alpha=.2);ax.legend(loc='upper right')
    fig.suptitle('Successful v9 PhysX PPO | mu_s=0.9, mu_d=0.7 | Complete 0-2.76 s record, no smoothing\n'
                 'Approach 0-0.4 s / braking 0.4-1.8 s / stop-hold 1.8-2.76 s | Both wheels touch at every recorded sample',fontsize=14)
    fig.savefig(args.output/'force_history_4x1.png',dpi=140);fig.savefig(args.output/'force_history_4x1.svg');plt.close(fig)

    fig,axs=plt.subplots(1,4,figsize=(24,5.2),constrained_layout=True)
    for side in (0,1):force_panel(axs[side],side,.1)
    for side,color in enumerate(colors):
        mask=t<.1;edges=np.r_[t[mask],.1]
        axs[2].stairs(s['point_tangent_pre_rms'][mask,side],edges,color=color,lw=1.2,label=('L','R')[side]+' tangential',baseline=None)
        axs[2].stairs(s['pre_lateral_speed_m_s'][mask,side],edges,color=color,ls=':',lw=1.2,label=('L','R')[side]+' lateral',baseline=None)
        axs[3].plot(np.r_[t[mask],.1],trace['joint_vel'][:mask.sum()+1,ids[side]],color=color,label=('Left','Right')[side])
    axs[2].axhline(.05,color='gray',ls='--',lw=.8)
    axs[2].set(title='Initial slip is mainly longitudinal',ylabel='Pre-step RMS point speed [m/s]')
    axs[3].set(title='Wheel spin starts at zero',ylabel='Joint velocity [rad/s]')
    for ax in axs[2:]:ax.set(xlim=(0,.1),xlabel='Time since reset [s]');ax.grid(alpha=.2);ax.legend(loc='upper right')
    fig.suptitle('First 100 ms | Initial body XY speed ~1.30 m/s, wheel joint speeds 0/0 rad/s\n'
        'First plotted force averages [0, 2.5 ms]; no pre-reset force or observed static-to-dynamic switch',fontsize=14)
    fig.savefig(args.output/'startup_0p1s_4x1.png',dpi=140);fig.savefig(args.output/'startup_0p1s_4x1.svg');plt.close(fig)

    fig,axs=plt.subplots(1,4,figsize=(24,5.2),constrained_layout=True)
    phi=np.linspace(0,2*np.pi,361)
    for j,(side,end) in enumerate(((0,.1),(1,.1),(0,horizon),(1,horizon))):
        ax=axs[j];valid=(t<end)&s['ratio_valid'][:,side]
        xx,yy=section_coordinates(fn[:,side],ft[:,side],s['force_longitudinal_N'][:,side],
            s['force_lateral_N'][:,side],s['ratio_valid'][:,side],mu_s,args.cone_view)
        x,y=xx[valid],yy[valid]
        draw_section(ax,mu_s,mu_d,args.cone_view,float(np.ceil(fn.max()*1.08/20)*20))
        sc=ax.scatter(x,y,c=t[valid],cmap='viridis',vmin=0,vmax=end,s=12,alpha=.8)
        ax.scatter(x[0],y[0],marker='*',s=150,color='#d9233a',edgecolor='white',label='First 2.5 ms interval',zorder=10)
        ax.set_title(('Left','Right')[side]+f' wheel | 0-{end:g} s')
        ax.grid(alpha=.2);ax.legend(loc='upper left' if args.cone_view=='triangle' else 'upper right',fontsize=7);fig.colorbar(sc,ax=ax,shrink=.65,label='Time [s]')
    fig.suptitle('Wheel-resultant friction sections from reset | mu_s=0.9, mu_d=0.7\n'
                 'Startup 0-0.1 s (first two panels) and full task (last two) | Not individual contact-point cones',fontsize=14)
    fig.savefig(args.output/'cone_sections_from_start_4x1.png',dpi=140);fig.savefig(args.output/'cone_sections_from_start_4x1.svg');plt.close(fig)

    animation=animate_forces(s,row,args.output,args.cone_view)

    phases=[]
    for label,start,end in [('startup',0.,.02),('approach',0.,.4),('braking',.4,1.8),('stop_hold',1.8,horizon)]:
        mask=(t>=start-1e-9)&(s['time_end_s']<=end+1e-9)
        phases.append(dict(phase=label,start_s=start,end_s=end,intervals=int(mask.sum()),
            mean_normal_N=fn[mask].mean(0).tolist(),mean_tangent_N=ft[mask].mean(0).tolist(),
            max_tangent_N=ft[mask].max(0).tolist(),mu_effective_p95=np.nanquantile(effective[mask],.95,axis=0).tolist(),
            geometric_contact_loss_intervals=(~s['contact_at_both_endpoints'][mask]).sum(0).tolist(),
            tangent_sliding_s=(tangent_slip[mask].sum(0)*dt).tolist(),lateral_sliding_s=(lateral_slip[mask].sum(0)*dt).tolist(),
            near_dynamic_force_s=(((s['rho_dynamic'][mask]>=.9)&(s['rho_dynamic'][mask]<=1.05)).sum(0)*dt).tolist()))
    with (args.output/'force_intervals.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['start_s','end_s','wheel','Fn_N','Ft_N','F_long_N','F_lat_N','mu_s_Fn_N','mu_d_Fn_N',
            'Ft_over_Fn','contact_start','contact_end','v_tangent_pre_RMS_m_s','v_tangent_interval_RMS_m_s',
            'v_lateral_pre_RMS_m_s','v_lateral_interval_RMS_m_s','wheel_spin_rad_s','tangent_sliding','lateral_sliding'])
        for i in range(n):
            for side in (0,1):
                writer.writerow([t[i],s['time_end_s'][i],('L','R')[side],fn[i,side],ft[i,side],
                    s['force_longitudinal_N'][i,side],s['force_lateral_N'][i,side],static[i,side],dynamic[i,side],
                    effective[i,side],int(s['touching_states'][i,side]),int(s['touching_states'][i+1,side]),
                    s['point_tangent_pre_rms'][i,side],s['point_tangent_interval_rms'][i,side],
                    s['pre_lateral_speed_m_s'][i,side],lat_interval[i,side],spin[i,side],
                    int(tangent_slip[i,side]),int(lateral_slip[i,side])])
    output=dict(kind='hockey_force_from_reset_v1',source_sha256=sha(__file__),
        analysis_report_sha256=sha(args.analysis/'report.json'),signals_sha256=sha(args.analysis/'signals.npz'),
        qualification_sha256=row['qualification_sha256'],raw_sha256=row['raw_sha256'],actor_sha256=run['actor_sha256'],
        mu_static=mu_s,mu_dynamic=mu_d,intervals=n,dt_s=dt,first_interval_s=[float(t[0]),float(t[0]+dt)],
        last_interval_s=[float(t[-1]),horizon],initial_time_discarded_s=0,physics_steps=0,smoothing=False,
        first_interval=dict(normal_N=fn[0].tolist(),tangent_N=ft[0].tolist(),static_capacity_N=static[0].tolist(),
            dynamic_capacity_N=dynamic[0].tolist(),rho_dynamic=s['rho_dynamic'][0].tolist(),
            tangent_pre_speed_m_s=s['point_tangent_pre_rms'][0].tolist(),lateral_pre_speed_m_s=s['pre_lateral_speed_m_s'][0].tolist(),
            wheel_spin_rad_s=spin[0].tolist(),body_XY_speed_m_s=float(np.linalg.norm(trace['root_com_vel'][0,:2]))),
        phases=phases,both_wheels_touch_all_samples=bool(s['touching_states'].all()),animation=animation,
        contact_path_diagnostics=contact_jump_summary(s,trace,run['geometry_audit']),
        force_time_convention='Geometry at t; force is impulse/dt over [t,t+2.5ms]. Reset force slot zero is not plotted.',
        interpretation='Startup sliding is predominantly longitudinal with zero initial wheel spin and nonzero body speed; not evidence of a newly observed static-to-dynamic switch.',
        solver_stick_slip_flag_observed=False,per_point_cone_certified=False,
        limitations=['Wheel-resultant forces only; normal-point/friction-anchor load correspondence is unavailable.',
            'The 0.05 m/s slip threshold is kinematic, not a solver-mode classifier.',
            'Phase slip durations are diagnostic, not the frozen task success qualifier.',
            'Contact is checked at recorded endpoints, not proven continuously between them.',
            'The startup window is included within approach; phase rows must not all be summed.'],
        artifact_sha256={p.name:sha(p) for p in sorted(args.output.iterdir())})
    (args.output/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output['first_interval'],indent=2),flush=True)


if __name__=='__main__':main()

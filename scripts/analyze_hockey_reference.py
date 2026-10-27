#!/usr/bin/env python3
"""Measure bilateral geometric beta for the explicitly designed stop command."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hdm05_gmr import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--command',type=Path,required=True)
    args=p.parse_args()
    z=np.load(args.command/'motion.npz',allow_pickle=False)
    d=np.load(args.command/'command_diagnostics.npz',allow_pickle=False)
    t=d['time_s'];names=z['body_names'].astype(str).tolist()
    ids=[names.index(x) for x in ('wheel_L_Link','wheel_R_Link')]
    q=z['body_quat_w'][:,0];v=z['body_lin_vel_w'][:,0]
    yaw=np.degrees(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
    wrap=lambda x:(x+180)%360-180
    speed=np.linalg.norm(v[:,:2],axis=-1)
    body_beta=wrap(np.degrees(np.arctan2(v[:,1],v[:,0]))-yaw)
    body_beta[speed<.25]=np.nan
    wheel_v=z['body_lin_vel_w'][:,ids,:2];wheel_speed=np.linalg.norm(wheel_v,axis=-1)
    wheel_directed_beta=wrap(np.degrees(np.arctan2(wheel_v[...,1],wheel_v[...,0]))-d['wheel_heading_deg'])
    # A wheel plane is axial: 180 degrees is reverse rolling, not lateral skid.
    wheel_beta=(wheel_directed_beta+90)%180-90
    wheel_beta[wheel_speed<.25]=np.nan
    near=(d['wheel_bottom_proxy_m']<=.03)&(d['wheel_bottom_proxy_m']>=-.02)
    mask=near&(wheel_speed>=.25)
    summary=dict(analysis_sha256=sha(__file__),motion_sha256=sha(args.command/'motion.npz'),
        angle_definition='body: directed course-minus-forward [-180,180); wheels: axial course-minus-rolling-plane [-90,90)',
        velocity_location='body COM and each wheel COM; not material contact-point slip',
        beta_min_speed_m_s=.25,body_abs_beta_peak_deg=float(np.nanmax(abs(body_beta))),
        wheel_abs_beta_peak_near_floor_deg=[float(np.max(abs(wheel_beta[mask[:,k],k]))) for k in (0,1)],
        wheel_near_floor_and_beta_valid_fraction=mask.mean(axis=0).tolist(),
        near_floor_is_contact=False,forces_available=False,friction_cone_utilization_available=False,
        designed_global_root_motion=True,measured_global_human_motion=False,
        physics_run=False,task_success_claimed=False)
    np.savez_compressed(args.command/'geometric_beta.npz',time_s=t,body_beta_deg=body_beta,
        wheel_beta_deg=wheel_beta,wheel_directed_beta_deg=wheel_directed_beta,
        wheel_beta_valid=wheel_speed>=.25,near_floor=near)
    (args.command/'geometric_beta.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axs=plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
    colors=['#dd573d','#3673bc'];xy=z['body_pos_w'][:,0,:2]
    axs[0].plot(*xy.T,color='black',label='Designed root path')
    for i in (0,20,40,60,80,110,138):
        vec=.15*np.array([np.cos(np.radians(yaw[i])),np.sin(np.radians(yaw[i]))])
        axs[0].arrow(*xy[i],*vec,width=.004,color='black',alpha=.55,length_includes_head=True)
        for k,color in enumerate(colors):
            center=z['body_pos_w'][i,ids[k],:2];h=np.radians(d['wheel_heading_deg'][i,k]);delta=.10*np.array([np.cos(h),np.sin(h)])
            axs[0].plot([center[0]-delta[0],center[0]+delta[0]],[center[1]-delta[1],center[1]+delta[1]],color=color,lw=2)
    axs[0].set_aspect('equal');axs[0].set(xlabel='X [m]',ylabel='Y [m]',title='Command XY / body + wheel headings')
    axs[1].plot(t,body_beta,color='black',label='Body COM')
    for k,c in enumerate(colors):axs[1].plot(t,wheel_beta[:,k],color=c,label=['Left wheel COM','Right wheel COM'][k])
    axs[1].set(title='Body beta / wheel axial beta; speed >= 0.25',xlabel='Time [s]',ylabel='Beta [deg]');axs[1].legend(fontsize=8)
    axs[2].plot(t,d['designed_root_link_speed_m_s'],label='Designed planar speed',color='black')
    axs[2].plot(t,speed,label='COM speed from moving posture',color='#e5a22e')
    axs[2].axvspan(.4,1.8,alpha=.1,color='gray');axs[2].set(title='Designed approach / brake / hold',xlabel='Time [s]',ylabel='Speed [m/s]');axs[2].legend(fontsize=8)
    for k,c in enumerate(colors):axs[3].plot(t,d['wheel_bottom_proxy_m'][:,k]*1000,color=c,label=['Left','Right'][k])
    axs[3].axhline(0,color='black',lw=.8);axs[3].set(title='Geometric wheel-bottom clearance',xlabel='Time [s]',ylabel='Clearance [mm]');axs[3].legend()
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle('Motion-inspired kinematic command only: no loaded contact or friction-cone measurement')
    fig.savefig(args.command/'command_beta_4x1.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

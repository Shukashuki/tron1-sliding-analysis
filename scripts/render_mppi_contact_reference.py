"""Fixed relative cameras and fixed world XY scale; replay recorded poses only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','osmesa')
import numpy as np
import generate_mppi_contact_reference as g


def run(a):
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from render_physx_ppo_fullbody import create_runtime,_pose
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    audit=json.loads((a.audit/'analysis.json').read_text())
    if g.sha(a.run/'rollout.npz')!=audit['rollout_sha256']:raise ValueError('Audit/rollout differs')
    with np.load(a.run/'rollout.npz',allow_pickle=False) as f:r={k:f[k].copy() for k in f.files}
    with np.load(g.ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz',allow_pickle=False) as f:s={k:f[k].copy() for k in f.files}
    runtime=create_runtime(g.ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml',width=640,height=400)
    # This group assignment affects graphics only, not simulation or source assets.
    runtime.model.geom_group[runtime.model.geom('floor').id]=1
    font=ImageFont.load_default(size=17);small=ImageFont.load_default(size=14)
    indices=np.arange(0,2785,8);t=r['time_s'];camber,heading=g.wheel_angles(r['body_quat_w'][:,-2:])
    root_heading=g.quaternion_heading(r['root_quaternion_wxyz']);speed=np.linalg.norm(r['base_com_velocity_w'][:,:2],axis=-1)
    xy_error=np.linalg.norm(r['root_pos'][:,:2]-np.column_stack([np.interp(t,s['time_s'],s['qpos_mujoco'][:,j]) for j in (0,1)]),axis=-1)
    points=np.concatenate((s['qpos_mujoco'][:,:2],r['root_pos'][:,:2],r['body_pos_w'][:,-2:,:2].reshape(-1,2)))
    low,high=points.min(0)-.4,points.max(0)+.4;center=(low+high)/2
    scale=min(590/(high[0]-low[0]),480/(high[1]-low[1]))
    pixel=lambda p:(320+(p[0]-center[0])*scale,300-(p[1]-center[1])*scale)
    a.output_dir.mkdir(parents=True)
    def frame(i):
        im=Image.new('RGB',(2560,640),'#fafafa');draw=ImageDraw.Draw(im)
        draw.text((12,10),'World XY | same scale throughout',font=font,fill='#111111')
        draw.line([pixel(p) for p in s['qpos_mujoco'][:,:2]],fill='#999999',width=3)
        draw.line([pixel(p) for p in r['root_pos'][:i+1,:2]],fill='#157ab5',width=3)
        p=r['root_pos'][i,:2];direction=np.array([np.cos(root_heading[i]),np.sin(root_heading[i])])
        draw.line((pixel(p),pixel(p+.3*direction)),fill='#174b9a',width=5)
        velocity=r['base_com_velocity_w'][i,:2];nv=np.linalg.norm(velocity)
        if nv>.01:draw.line((pixel(p),pixel(p+.3*velocity/nv)),fill='#2d9439',width=4)
        for side,color in enumerate(('#d62728','#9467bd')):
            center_w=r['body_pos_w'][i,-2+side,:2];direction=np.array([np.cos(heading[i,side]),np.sin(heading[i,side])])
            draw.line((pixel(center_w-.13*direction),pixel(center_w+.13*direction)),fill=color,width=5)
        draw.text((12,565),'gray ref | blue actual/body | green COM velocity',font=small,fill='#333333')
        draw.text((12,588),'red L / purple R: projected wheel planes',font=small,fill='#333333')
        for j,(azimuth,elevation,name) in enumerate(((135.,-20.,'Oblique fixed relative camera'),(225.,-8.,'Side fixed relative camera'))):
            _pose(runtime.model,runtime.data,r['qpos'][i]);camera=mujoco.MjvCamera()
            camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.azimuth=azimuth;camera.elevation=elevation;camera.distance=1.9
            camera.lookat[:]=r['qpos'][i,:3]+[0.,0.,-.25]
            runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
            offset=(j+1)*640;im.paste(Image.fromarray(runtime.renderer.render().copy()),(offset,42))
            draw.text((offset+10,10),name,font=font,fill='#111111')
            draw.text((offset+10,465),f't={t[i]:.2f}s | XY error {xy_error[i]:.3f}m | speed {speed[i]:.2f}m/s',font=small,fill='#111111')
            draw.text((offset+10,493),f'Camber L/R {np.rad2deg(camber[i,0]):+.1f}/{np.rad2deg(camber[i,1]):+.1f}deg',font=small,fill='#111111')
            offset_heading=np.rad2deg(g.angle(heading[i]-root_heading[i]))
            draw.text((offset+10,521),f'Wheel heading/body L/R {offset_heading[0]:+.1f}/{offset_heading[1]:+.1f}deg',font=small,fill='#111111')
            fz=r['ground_normal_force_w'][i,:,2]
            draw.text((offset+10,549),f'Actual upward Fz L/R {fz[0]:.1f}/{fz[1]:.1f}N',font=small,fill='#111111')
        x=1932;draw.text((x,10),'Actual contact and control diagnostics',font=font,fill='#111111')
        fz=r['ground_normal_force_w'][...,2];bound=max(300.,float(np.quantile(fz,.99))*1.1)
        box=(x,70,2540,340);draw.rectangle(box,fill='white',outline='#aaaaaa')
        def fp(time,value):return (x+(time/6.98)*608,340-270*min(value,bound)/bound)
        for side,color in enumerate(('#d62728','#9467bd')):
            draw.line([fp(float(t[k]),float(fz[k,side])) for k in range(0,len(t),2)],fill=color,width=2)
        px=fp(float(t[i]),0)[0];draw.line((px,70,px,340),fill='#111111',width=2)
        y5=fp(0,5)[1];draw.line((x,y5,2540,y5),fill='#999999',width=1)
        draw.text((x,355),f'Native400Hz force histories | range 0-{bound:.0f}N',font=small,fill='#333333')
        support=audit['support'];core=audit['windows']['core']['tracking']
        lines=[f"Core XY RMSE {core['xy_rmse_m']:.3f}m | beta RMSE {core['beta_rmse_deg']:.1f}deg",
            f"Core mean speed/reference {core['speed_retention']:.3f}",
            f"Any-wheel Fz>5N: {100*support['any_wheel_support_sample_fraction']:.2f}%",
            f"Longest no-support gap {1000*support['longest_no_support_observed_s']:.1f}ms",
            'Native MuJoCo rollout; independent L/R commands',
            'No height target; no wheel XY target; no PPO',
            'Candidate only: stable-contact feasibility not certified']
        for j,text in enumerate(lines):draw.text((x,390+j*28),text,font=small,fill='#7b2626' if j==6 else '#111111')
        return im
    try:
        writer=imageio_ffmpeg.write_frames(str(a.output_dir/'mppi_reference_4x1.mp4'),(2560,640),fps=50,codec='libx264',quality=7,macro_block_size=2,output_params=['-movflags','+faststart'])
        writer.send(None)
        try:
            for frame_index,i in enumerate(indices):
                im=frame(int(i));writer.send(np.asarray(im))
                if frame_index in (72,112,150,200,348):im.save(a.output_dir/f'frame_{frame_index:03d}.png')
                if frame_index==150:im.save(a.output_dir/'comparison.png')
                if frame_index%100==0:print(f'render {frame_index}/349',flush=True)
        finally:writer.close()
    finally:runtime.renderer.close()
    fig,axes=plt.subplots(1,4,figsize=(22,4.8),layout='constrained')
    axes[0].plot(s['qpos_mujoco'][:,0],s['qpos_mujoco'][:,1],'--',color='gray',label='Original reference')
    axes[0].plot(r['root_pos'][:,0],r['root_pos'][:,1],label='MPPI native rollout');axes[0].axis('equal');axes[0].set_title('World XY');axes[0].set_xlabel('X [m]');axes[0].set_ylabel('Y [m]')
    axes[1].plot(t,speed,label='Actual COM speed');axes[1].plot(s['time_s'],np.linalg.norm(s['body_lin_vel_w'][:,0,:2],axis=-1),'--',label='Reference COM speed');axes[1].set_title('Speed');axes[1].set_ylabel('[m/s]')
    for i,name,color in ((0,'Left','#d62728'),(1,'Right','#9467bd')):
        axes[2].plot(t,np.rad2deg(camber[:,i]),color=color,label=name+' camber')
        axes[2].plot(t,np.rad2deg(g.angle(heading[:,i]-root_heading)),color=color,ls='--',alpha=.65,label=name+' heading/body')
        axes[3].plot(t,r['ground_normal_force_w'][:,i,2],color=color,label=name+' Fz')
    axes[2].set_title('Independent wheel orientations');axes[2].set_ylabel('[deg]');axes[3].set_title('Actual native wheel-floor forces');axes[3].set_ylabel('[N]');axes[3].axhline(5,color='black',ls=':')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ax in axes[1:]:ax.set_xlabel('Time [s]');ax.axvspan(1.44,4,color='gray',alpha=.08)
    fig.savefig(a.output_dir/'overview_4x1.png',dpi=140);plt.close(fig)
    provenance=dict(kind='MPPI_actual_native_pose_replay',source_sha256=g.sha(__file__),run_rollout_sha256=g.sha(a.run/'rollout.npz'),audit_sha256=g.sha(a.audit/'analysis.json'),
        fps=50,frames=349,dimensions=[2560,640],camera_distance_m=1.9,lookat_offset_m=[0,0,-.25],
        camera_azimuths_deg=[135,225],camera_elevations_deg=[-20,-8],fixed_relative_camera=True,fixed_world_XY_scale=True,
        pose_corrections=0,physics_steps_during_render=0,nominal_task_success_claimed=False)
    provenance['artifacts']={p.name:g.sha(p) for p in a.output_dir.iterdir() if p.is_file()}
    g.write_json(a.output_dir/'provenance.json',provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('run','audit','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

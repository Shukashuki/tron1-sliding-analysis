"""Real-time4x1 speed comparison; fixed cameras; recorded PhysX mesh replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','osmesa')
import numpy as np
from prepare_physx_speed_references import load


def run(a):
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    from render_physx_ppo_fullbody import create_runtime,_pose,build_qpos
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert not a.output_dir.exists()
    analysis=json.loads(a.analysis.read_text());cases=analysis['cases']
    runtime=create_runtime(a.model,width=640,height=320)
    traces=[];refs=[];poses=[];wheel_ids=[];reference_wheels=[];errors=[]
    for c in cases:
        f=a.probes/c['id']/'final_rollout.npz';assert sha(f)==c['source_recording_sha256']
        r=load(f);ref=load(a.references/c['id']/'motion.npz');t=np.arange(len(r['root_pos']))*.0025
        native=dict(time_s=t,position=r['root_pos'],quaternion=r['root_quat'],joint_pos=r['joint_pos'],joint_names=r['joint_names'])
        q,_=build_qpos(runtime.model,native);poses.append(q);traces.append(r);refs.append(ref)
        names=r['body_names'].astype(str).tolist();wn=[names.index(k) for k in ('wheel_L_Link','wheel_R_Link')];wheel_ids.append(wn)
        rn=ref['body_names'].astype(str).tolist();reference_wheels.append([rn.index(k) for k in ('wheel_L_Link','wheel_R_Link')])
        worst=0.
        for i in range(0,len(q),8):
            _pose(runtime.model,runtime.data,q[i]);actual=runtime.data.xpos[[runtime.model.body(k).id for k in ('wheel_L_Link','wheel_R_Link')]]
            worst=max(worst,float(np.max(np.linalg.norm(actual-r['body_pos'][i,wn],axis=-1))))
        assert worst<2e-5;errors.append(worst)
    allxy=np.concatenate([r['root_pos'][:,:2] for r in traces]+[r['body_pos_w'][:,0,:2] for r in refs])
    low=allxy.min(0)-.35;high=allxy.max(0)+.35;center=(low+high)/2
    scale=min(580/(high[0]-low[0]),225/(high[1]-low[1]))
    def point(p,j):return (j*640+320+(p[0]-center[0])*scale,175-(p[1]-center[1])*scale)
    def yaw(q):return np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    font=ImageFont.load_default(size=18);small=ImageFont.load_default(size=15)
    end=max(len(r['actions']) for r in traces)
    a.output_dir.mkdir(parents=True)
    def frame(f):
        im=Image.new('RGB',(2560,832),'#fafafa');draw=ImageDraw.Draw(im)
        for j,(c,r,ref,q) in enumerate(zip(cases,traces,refs,poses)):
            i=min(f*8,len(q)-1);k=min(f,len(ref['joint_pos'])-1);offset=j*640
            ended=f>=len(r['actions']);t=i*.0025;v=r['root_com_vel'][i,:2];speed=float(np.linalg.norm(v))
            draw.text((offset+12,8),f"{c['requested_scale']:g}x reference | fixed PPO | PhysX mu=0.7",font=font,fill='#111111')
            draw.text((offset+12,34),f"t={t:.2f}s | actual {speed:.2f}m/s"+(' | END (held)' if ended else ''),font=small,fill='#b12b2b' if ended else '#111111')
            draw.line([point(p,j) for p in ref['body_pos_w'][:,0,:2]],fill='#999999',width=2)
            draw.line([point(p,j) for p in r['root_pos'][:i+1:4,:2]],fill='#157ab5',width=3)
            body=r['root_pos'][i,:2];angle=yaw(r['root_quat'][i]);d=np.array([np.cos(angle),np.sin(angle)])
            draw.line((point(body,j),point(body+.3*d,j)),fill='#174b9a',width=4)
            if speed>.01:draw.line((point(body,j),point(body+.3*v/speed,j)),fill='#228b22',width=3)
            for side,wid in enumerate(wheel_ids[j]):
                w,x,y,z=r['body_quat'][i,wid];axle=np.array([2*(x*y-w*z),1-2*(x*x+z*z)])
                d=np.array([axle[1],-axle[0]]);d/=np.linalg.norm(d)+1e-12;wp=r['body_pos'][i,wid,:2]
                draw.line((point(wp-.13*d,j),point(wp+.13*d,j)),fill=['#d62728','#9467bd'][side],width=5)
            kp=ref['body_pos_w'][k,[0,*reference_wheels[j]],:2]
            for p in kp:
                px,py=point(p,j);draw.ellipse((px-3,py-3,px+3,py+3),fill='#777777')
            draw.line([point(p,j) for p in kp[[1,0,2]]],fill='#888888',width=2)
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth=135.;camera.elevation=-20.;camera.distance=1.9;camera.lookat[:]=q[i,:3]+[0,0,-.25]
            _pose(runtime.model,runtime.data,q[i]);runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
            im.paste(Image.fromarray(runtime.renderer.render().copy()),(offset,310))
            m=c['metrics'];fz=r['normal_fz'][i]
            lines=[f"Core speed {m['actual_core_mean_speed_m_s']:.2f} / ref {c['reference_demand']['core_mean_speed_m_s']:.2f} m/s",
                f"Core XY {m['core']['xy_rmse_m']:.3f}m | beta RMSE {m['core']['beta_rmse_deg']:.1f}deg",
                f"Support {100*m['support_fraction']:.1f}% | longest gap {1000*m['longest_support_gap_s']:.1f}ms",
                f"Now Fz L/R {fz[0]:.1f}/{fz[1]:.1f}N | max tilt {m['maximum_root_tilt_deg']:.1f}deg",
                'Candidate failed task screens; slipping is not task success',
                'Gray: ref path/keypoints | blue: actual/body | green: velocity']
            for row,line in enumerate(lines):draw.text((offset+10,642+28*row),line,font=small,fill='#8b2222' if row==4 else '#222222')
        return im
    writer=imageio_ffmpeg.write_frames(str(a.output_dir/'speed_ladder_4x1.mp4'),(2560,832),fps=50,
        codec='libx264',quality=7,macro_block_size=2,output_params=['-movflags','+faststart']);writer.send(None)
    try:
        for f in range(end+1):
            im=frame(f);writer.send(np.asarray(im))
            if f in (50,100):im.save(a.output_dir/f'frame_{f:03d}.png')
            if f==100:im.save(a.output_dir/'comparison.png')
            if f%100==0:print('render',f,'/',end,flush=True)
    finally:writer.close();runtime.renderer.close()
    fig,axes=plt.subplots(1,4,figsize=(23,4.5),layout='constrained');colors=['#157ab5','#d57b17','#368a3a','#9256a3']
    for j,(c,r,ref) in enumerate(zip(cases,traces,refs)):
        t=np.arange(len(r['root_pos']))*.0025;label=f"{c['requested_scale']:g}x"
        axes[0].plot(r['root_pos'][:,0],r['root_pos'][:,1],color=colors[j],label=label)
        axes[1].plot(t,np.linalg.norm(r['root_com_vel'][:,:2],axis=-1),color=colors[j],label=label)
        axes[1].plot(ref['time_s'],np.linalg.norm(ref['body_lin_vel_w'][:,0,:2],axis=-1),'--',color=colors[j],alpha=.6)
        axes[2].plot(t,np.max(r['normal_fz'],axis=-1),color=colors[j],label=label)
        axes[3].plot(t,np.max(r['lateral_slip_rms'],axis=-1),color=colors[j],label=label)
    ref=refs[0];axes[0].plot(ref['body_pos_w'][:,0,0],ref['body_pos_w'][:,0,1],'--',color='gray',label='Reference')
    axes[0].axis('equal');axes[0].set_title('World XY [m]');axes[1].set_title('Actual speed / dashed target [m/s]')
    axes[2].set_title('Any-wheel max normal load [N]');axes[3].set_title('Loaded-point lateral slip RMS [m/s]')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ax in axes[1:]:ax.set_xlabel('Actual simulation time [s]')
    fig.savefig(a.output_dir/'overview_4x1.png',dpi=140);plt.close(fig)
    prov=dict(kind='real_time_PhysX_speed_ladder_mesh_replay',frames=end+1,fps=50,
        physical_time_shared_between_panels=True,completed_episode_panels_hold_final_state_with_END_label=True,
        fixed_camera_distance_m=1.9,fixed_camera_root_offset_m=[0,0,-.25],camera_zoom=False,
        camera_orientation_deg=[135.,-20.],world_XY_bounds=[low.tolist(),high.tolist()],
        ref_keypoints='base and left/right wheel origins; pure kinematic reference sampled at same actual time',
        red_purple_segments='current projected L/R wheel planes',maximum_wheel_FK_errors_m=errors,
        source_sha256=sha(__file__),analysis_sha256=sha(a.analysis),model_sha256=sha(a.model),
        graphics_backend='MuJoCo OSMesa only',dynamics_backend='PhysX',physics_steps_during_render=0,
        task_success_claimed=False,artifacts={})
    prov['artifacts']={p.name:sha(p) for p in a.output_dir.iterdir() if p.is_file()}
    (a.output_dir/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('probes','references','analysis','model','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())

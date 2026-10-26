"""Four horizontal panels of native PhysX recordings; graphics-only mesh replay."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import hashlib
os.environ.setdefault('MUJOCO_GL','osmesa')
import numpy as np
from physx_reference_math import trajectory_metrics,wrap_angle


def run(a):
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    import mujoco
    from render_physx_ppo_fullbody import create_runtime,_pose,build_qpos
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert not a.output_dir.exists()
    read=lambda p:{k:v for k,v in np.load(p,allow_pickle=False).items()}
    base=read(a.baseline_validation/'final_rollout.npz');final=read(a.validation/'final_rollout.npz')
    ref=read(a.reference)
    report=json.loads((a.validation/'report.json').read_text())
    steps=len(ref['joint_pos']);duration=steps*.02
    speed_scale=getattr(a,'speed_scale',1.)
    assert report['single_world_validation'] and report['steps']==steps and speed_scale>=1.
    a.output_dir.mkdir(parents=True)
    runtime=create_runtime(a.model,width=640,height=400)
    body_names=final['body_names'].astype(str).tolist()
    wheel_ids=[body_names.index(k) for k in ('wheel_L_Link','wheel_R_Link')]
    joint_names=final['joint_names'].astype(str)
    traces=[base,final];qpos=[];fk_errors=[]
    for trace in traces:
        native=dict(time_s=np.arange(len(trace['root_pos']))*.0025,position=trace['root_pos'],
            quaternion=trace['root_quat'],joint_pos=trace['joint_pos'],joint_names=joint_names)
        poses,_=build_qpos(runtime.model,native);qpos.append(poses)
        maximum=0.
        for i in range(0,len(poses),8):
            _pose(runtime.model,runtime.data,poses[i])
            actual=runtime.data.xpos[[runtime.model.body(k).id for k in ('wheel_L_Link','wheel_R_Link')]]
            maximum=max(maximum,float(np.max(np.linalg.norm(actual-trace['body_pos'][i,wheel_ids],axis=-1))))
        assert maximum<2e-5, f'Mesh FK disagreement {maximum}'
        fk_errors.append(maximum)
    allxy=np.concatenate([ref['body_pos_w'][:,0,:2]]+[r['root_pos'][:,:2] for r in traces])
    lo,hi=allxy.min(0)-.4,allxy.max(0)+.4;center=(lo+hi)/2;scale=min(570/(hi[0]-lo[0]),470/(hi[1]-lo[1]))
    xy=lambda p:(320+(p[0]-center[0])*scale,300-(p[1]-center[1])*scale)
    def yaw(q):return np.arctan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2]**2+q[...,3]**2))
    metrics=[trajectory_metrics(r,ref,core_interval_s=(1.44/speed_scale,4./speed_scale)) for r in traces]
    font=ImageFont.load_default(size=18);small=ImageFont.load_default(size=15)
    colors=['#b57b20','#157ab5'];time=np.arange(len(final['root_pos']))*.0025
    def frame(i):
        im=Image.new('RGB',(2560,640),'#fafafa');draw=ImageDraw.Draw(im)
        draw.text((12,10),'World XY | gray reference | fixed scale',font=font,fill='#111111')
        draw.line([xy(p) for p in ref['body_pos_w'][:,0,:2]],fill='#888888',width=3)
        for j,r in enumerate(traces):
            draw.line([xy(p) for p in r['root_pos'][:i+1:4,:2]],fill=colors[j],width=3)
            p=r['root_pos'][i,:2];y=yaw(r['root_quat'][i]);d=np.array([np.cos(y),np.sin(y)])
            draw.line((xy(p),xy(p+.3*d)),fill=colors[j],width=5)
        v=final['root_com_vel'][i,:2];p=final['root_pos'][i,:2]
        if np.linalg.norm(v)>.01:draw.line((xy(p),xy(p+.3*v/np.linalg.norm(v))),fill='#228b22',width=4)
        for side,k in enumerate(wheel_ids):
            w,x,y,z=final['body_quat'][i,k]
            axle=np.array([2*(x*y-w*z),1-2*(x*x+z*z)])
            d=np.array([axle[1],-axle[0]]);d/=(np.linalg.norm(d)+1e-12)
            p=final['body_pos'][i,k,:2]
            draw.line((xy(p-.12*d),xy(p+.12*d)),fill=['#d62728','#9467bd'][side],width=5)
        draw.text((12,550),'orange: PPO baseline | blue: PPO + MPPI',font=small,fill='#333333')
        draw.text((12,578),'green: velocity | red/purple: L/R wheel planes',font=small,fill='#333333')
        for j,r in enumerate(traces):
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth=135.;camera.elevation=-20.;camera.distance=1.9
            camera.lookat[:]=qpos[j][i,:3]+[0,0,-.25]
            _pose(runtime.model,runtime.data,qpos[j][i])
            runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
            offset=(j+1)*640;im.paste(Image.fromarray(runtime.renderer.render().copy()),(offset,45))
            draw.text((offset+12,10),['PhysX PPO | zero residual','PhysX PPO + MPPI | fresh single world'][j],font=font,fill='#111111')
            m=metrics[j];c=m['core']
            lines=[f"Core XY {c['xy_rmse_m']:.3f} m | speed ratio {c['speed_retention']:.3f}",
                f"Beta RMSE {c['beta_rmse_deg']:.1f} deg | max tilt {m['maximum_root_tilt_deg']:.1f} deg",
                f"Any-wheel support {100*m['support_fraction']:.2f}% | gap {1000*m['longest_support_gap_s']:.1f} ms",
                'Recorded PhysX poses; mesh replay has no dynamics',
                'Candidate reference; continuous support not certified']
            for k,line in enumerate(lines):draw.text((offset+12,460+29*k),line,font=small,fill='#333333')
        x0=1932;draw.text((x0,10),f'Native 400 Hz contact diagnostics | t={time[i]:.2f}s',font=font,fill='#111111')
        fields=[('normal_fz',500.,'Actual upward normal load [N]'),('lateral_slip_rms',2.,'Loaded-point lateral RMS [m/s]'),
                ('utilization_proxy',1.5,'Wheel aggregate utilization PROXY')]
        for row,(key,limit,title) in enumerate(fields):
            top=65+row*170;draw.text((x0,top),title,font=small,fill='#333333')
            for side,color in enumerate(['#d62728','#9467bd']):
                path=[(x0+600*time[k]/duration,top+140-110*np.clip(final[key][k,side],0,limit)/limit) for k in range(1,len(time),4)]
                draw.line(path,fill=color,width=2)
            tick=x0+600*time[i]/duration;draw.line((tick,top+25,tick,top+140),fill='#111111',width=1)
        draw.text((x0,590),'Proxy does not prove pointwise friction-cone saturation.',font=small,fill='#8b2222')
        return im
    writer=imageio_ffmpeg.write_frames(str(a.output_dir/'physx_reference_4x1.mp4'),(2560,640),fps=50,
        codec='libx264',quality=7,macro_block_size=2,output_params=['-movflags','+faststart'])
    writer.send(None)
    try:
        for f,i in enumerate(range(0,steps*8,8)):
            im=frame(i);writer.send(np.asarray(im))
            if f==min(150,steps-1):im.save(a.output_dir/'comparison.png')
            if f%100==0:print('render',f,flush=True)
    finally:writer.close();runtime.renderer.close()
    fig,axes=plt.subplots(1,4,figsize=(22,4.5),layout='constrained')
    axes[0].plot(ref['body_pos_w'][:,0,0],ref['body_pos_w'][:,0,1],'--',color='gray',label='Car-mapped reference')
    for j,r in enumerate(traces):
        label=['PPO zero residual','PPO + MPPI'][j]
        axes[0].plot(r['root_pos'][:,0],r['root_pos'][:,1],color=colors[j],label=label)
        axes[1].plot(time,np.linalg.norm(r['root_com_vel'][:,:2],axis=-1),color=colors[j],label=label)
        axes[2].plot(time,np.max(r['normal_fz'],axis=-1),color=colors[j],label=label)
    for side,name in enumerate(['Left','Right']):axes[3].plot(time,final['lateral_slip_rms'][:,side],label=name+' loaded-point slip')
    axes[0].axis('equal');axes[0].set_title('World XY [m]');axes[1].set_title('COM speed [m/s]')
    axes[2].set_title('Max wheel upward normal load [N]');axes[2].axhline(5,ls=':',color='black')
    axes[3].set_title('Lateral contact slip RMS [m/s]')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.savefig(a.output_dir/'overview_4x1.png',dpi=140);plt.close(fig)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    provenance=dict(kind='recorded_PhysX_policy_assisted_Mppi_mesh_replay',metrics=metrics,
        renderer_source_sha256=sha(__file__),model_sha256=sha(a.model),baseline_sha256=sha(a.baseline_validation/'final_rollout.npz'),
        validation_sha256=sha(a.validation/'final_rollout.npz'),reference_sha256=sha(a.reference),
        maximum_wheel_FK_error_m=fk_errors,FK_tolerance_m=2e-5,physics_steps_during_render=0,
        dynamics_backend='PhysX',graphics_backend='MuJoCo OSMesa mesh rendering only',
        camera_distance_m=1.9,camera_root_relative_offset_m=[0,0,-.25],camera_zoom=False,
        world_XY_scale_fixed=True,frames=steps,fps=50,speed_scale=speed_scale,artifacts={})
    provenance['artifacts']={p.name:sha(p) for p in a.output_dir.iterdir() if p.is_file()}
    (a.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('search','validation','baseline-validation','reference','model','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--speed-scale',type=float,default=1.)
    run(p.parse_args())

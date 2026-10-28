#!/usr/bin/env python3
"""Four horizontal panels: GMR command, PhysX poses, equal-scale XY, contact evidence.

MuJoCo supplies mesh graphics and FK only. No physics step, pose correction or
camera zoom. Original source RGB is deliberately absent from this public render.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from evaluate_hockey_physx import load,sha
from render_physx_ppo_fullbody import create_runtime,build_qpos,_pose

COLORS=((35,105,185),(205,105,30))


def text(image,value,y,x=16,scale=.51,color=(35,35,35)):
    cv2.putText(image,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--qualification',type=Path)
    p.add_argument('--case',help='Named episode in a speed-study qualification directory')
    p.add_argument('--xy-center',type=float,nargs=2)
    p.add_argument('--xy-extent',type=float)
    p.add_argument('--command-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    run_dir=a.run_dir or a.result/'evidence/probes_v2/baseline'
    qualification_dir=a.qualification or a.result/'qualification'
    trace=load(run_dir/'final_rollout.npz')
    raw=load(run_dir/'native_contact_buffers.npz')
    qualification_path=qualification_dir/(a.case+'.json' if a.case else 'report.json')
    signals=load(qualification_dir/(a.case+'.signals.npz' if a.case else 'signals.npz'))
    command=load(a.command_dir/'command_diagnostics.npz')
    reference=load(a.command_dir/'motion.npz')
    command_report=json.loads((a.command_dir/'report.json').read_text())
    qualification=json.loads(qualification_path.read_text())
    run=json.loads((run_dir/'report.json').read_text())
    label=run.get('conditioning_mode','old PPO / zero residual')
    if run.get('kind')=='hockey_native_physx_mppi_v1':label='PPO+MPPI (dq adapter)'
    if run.get('kind')=='hockey_ppo_learning_v1':label='Hockey-trained PPO / '+run['variant']
    if run.get('kind')=='hockey_ppo_speed_success_v1':
        method='BM-style PPO' if run['method']=='beyondmimic' else 'Hockey PPO'
        label=f'{method} x{run["speed_scale"]:g} seed{run["seed"]}'
        if not a.case or qualification['case']['id']!=a.case or run['case_id']!=a.case:
            raise ValueError('Speed-study case identity mismatch')
    first_fall=qualification.get('first_45deg_tilt_diagnostic_s',qualification.get('first_45deg_tilt_s'))
    stop_screen=qualification['qualification']['task_success_claimed']
    if sha(a.model)!=command_report['model_sha256']:raise ValueError('Visual model mismatch')
    for file,digest in qualification['input_sha256'].items():
        if sha(run_dir/file)!=digest:raise ValueError('Recording changed')
    if sha(a.command_dir/'motion.npz')!=command_report['motion_sha256']:raise ValueError('Reference changed')
    runtime=create_runtime(a.model,640,420);model=runtime.model
    import mujoco
    # Only opaque visual meshes and the ground are displayed; collision proxies hidden.
    qpos,_=build_qpos(model,dict(time_s=signals['time_s'],joint_names=trace['joint_names'],
        position=trace['root_pos'],quaternion=trace['root_quat'],joint_pos=trace['joint_pos']))
    names=trace['body_names'].astype(str).tolist();ids=[names.index('wheel_'+s+'_Link') for s in ('L','R')]
    mjids=[int(model.body('wheel_'+s+'_Link').id) for s in ('L','R')]
    maximum=0.
    for i,q in enumerate(qpos):
        _pose(model,runtime.data,q)
        maximum=max(maximum,float(np.linalg.norm(runtime.data.xpos[mjids]-trace['body_pos'][i,ids],axis=-1).max()))
    if maximum>2e-5:raise ValueError(f'PhysX wheel-origin FK mismatch: {maximum}')
    camera=runtime.cameras[0];camera.distance=2.4;camera.azimuth=135;camera.elevation=-18
    rt=command['time_s'];t=signals['time_s'];fps=50;times=np.arange(round(rt[-1]*fps)+1)/fps
    xy=np.concatenate((qpos[:,:2],command['qpos'][:,:2],trace['body_pos'][:,ids,:2].reshape(-1,2)))
    center=(xy.min(0)+xy.max(0))/2;extent=max(float(np.ptp(xy,axis=0).max()*1.3),2.)
    if a.xy_center is not None:center=np.asarray(a.xy_center,dtype=float)
    if a.xy_extent is not None:
        if not np.isfinite(a.xy_extent) or a.xy_extent<=0:raise ValueError('Invalid XY extent')
        extent=a.xy_extent
    def project(points):
        v=(np.asarray(points)-center)*450/extent
        return np.stack((320+v[...,0],300-v[...,1]),axis=-1).astype(int)
    backgrounds=[np.full((640,640,3),249,np.uint8) for _ in range(4)]
    xybg=backgrounds[2]
    for x in np.arange(np.floor(center[0]-extent/2),center[0]+extent/2+.001,.25):
        q=project([[x,center[1]-extent/2],[x,center[1]+extent/2]])
        cv2.line(xybg,tuple(q[0]),tuple(q[1]),(228,228,228),1)
    for y in np.arange(np.floor(center[1]-extent/2),center[1]+extent/2+.001,.25):
        q=project([[center[0]-extent/2,y],[center[0]+extent/2,y]])
        cv2.line(xybg,tuple(q[0]),tuple(q[1]),(228,228,228),1)
    cv2.polylines(xybg,[project(command['qpos'][:,:2])],False,(30,150,95),2,cv2.LINE_AA)
    cv2.polylines(xybg,[project(qpos[:,:2])],False,(80,80,80),2,cv2.LINE_AA)
    text(xybg,'XY paths + body / wheel headings',28,scale=.64)
    text(xybg,'Green: command | Gray: PhysX full path',555)
    text(xybg,'Blue / orange: left / right wheel plane',581)
    text(xybg,f'Fixed equal XY scale | grid = 0.25 m',607)
    chart=backgrounds[3]
    text(chart,'Native PhysX contact diagnostics',28,scale=.64)
    specs=[('normal_fz','Wheel normal load [N]',0,2500,5),
        ('normal_point_lateral_rms','Loaded lateral slip [m/s]',0,5,.05),
        ('aggregate_utilization','Loaded aggregate friction utilization',0,1.4,.9),
        ('wheel_axial_beta_deg','Loaded wheel axial beta [deg]',-90,90,20)]
    def chartpoint(tt,yy,row,lo,hi):
        return np.stack((60+np.asarray(tt)/rt[-1]*550,100+row*125+(hi-np.clip(yy,lo,hi))/(hi-lo)*75),axis=-1).astype(int)
    for row,(key,title,lo,hi,threshold) in enumerate(specs):
        y0=100+row*125
        text(chart,title,y0-15,scale=.51)
        cv2.rectangle(chart,(60,y0),(610,y0+75),(190,190,190),1)
        text(chart,str(hi),y0+5,x=8,scale=.4);text(chart,str(lo),y0+75,x=8,scale=.4)
        q=chartpoint(np.array([0,rt[-1]]),np.array([threshold,threshold]),row,lo,hi)
        cv2.line(chart,tuple(q[0]),tuple(q[1]),(170,170,170),1)
        for k,color in enumerate(COLORS):
            y=signals[key][:,k];valid=np.isfinite(y)&(t<=rt[-1])&signals['native_sample_valid']
            if key!='normal_fz':valid&=signals['normal_fz'][:,k]>5
            edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
            for start,end in zip(edges[::2],edges[1::2]):
                pts=chartpoint(t[start:end],y[start:end],row,lo,hi)
                if len(pts)>1:cv2.polylines(chart,[pts],False,color,1,cv2.LINE_AA)
                elif len(pts):cv2.circle(chart,tuple(pts[0]),1,color,-1)
        first=first_fall
        if first is not None:
            x=int(60+first/rt[-1]*550);cv2.line(chart,(x,y0),(x,y0+75),(190,50,60),1)
    for tick in (0,.7,1.4,2.1,2.76):text(chart,f'{tick:g}',572,x=int(60+tick/rt[-1]*550)-10,scale=.38)
    text(chart,'Time [s] | Red: first tilt >=45 deg',595,scale=.48)
    text(chart,'Wheel-level force ratio; no pointwise cone claim',615,scale=.47)
    a.output.mkdir(parents=True)
    video=a.output/'hockey_command_physx_xy_contact_4x1.mp4'
    argv=[a.ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','2560x640',
        '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    proc=subprocess.Popen(argv,stdin=subprocess.PIPE)
    try:
        for now in times:
            i=int(round(now/.0025));j=int(round(now/.02));panels=[x.copy() for x in backgrounds]
            for which,(panel,pose) in enumerate(zip(panels[:2],(command['qpos'][j],qpos[i]))):
                _pose(model,runtime.data,pose);camera.lookat[:]=pose[:3]+[0,0,-.24]
                runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
                if which==1 and i>0:
                    for side in (0,1):
                        row=i-1;count=int(raw[f'w{side}_normal_buffer_4'][row].item());start=int(raw[f'w{side}_normal_buffer_5'][row].item())
                        force=raw[f'w{side}_normal_buffer_0'][row,start:start+count,0]
                        points=raw[f'w{side}_normal_buffer_1'][row,start:start+count]-trace['environment_origin_w']
                        for point in points[force>5]:
                            scene=runtime.renderer.scene;geom=scene.geoms[scene.ngeom]
                            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012,.012,.012]),point,np.eye(3).ravel(),np.array([.1,.9,.25,1.]))
                            scene.ngeom+=1
                panel[65:485]=runtime.renderer.render()
            text(panels[0],'GMR pose + designed stop command',28,scale=.62)
            text(panels[0],f't = {now:.2f} s | designed speed {command["designed_root_link_speed_m_s"][j]:.2f} m/s',520)
            text(panels[0],'Braking 0.40-1.80 s | kinematics only',550)
            text(panels[0],'Human-inspired posture; global path designed',580,scale=.49)
            text(panels[0],'Fixed follow camera: 2.4 m / no zoom',610)
            text(panels[1],f'PhysX | {label} | mu=0.7',28,scale=.55)
            text(panels[1],f't = {now:.2f} s | COM speed {signals["body_speed_m_s"][i]:.2f} m/s',520)
            text(panels[1],f'Tilt {signals["root_tilt_deg"][i]:.1f} deg | stop gates: {"PASS" if stop_screen else "FAIL"}',550,
                color=(30,130,70) if stop_screen else (190,45,45))
            fz=signals['normal_fz'][i];text(panels[1],f'Wheel load L / R: {fz[0]:.1f} / {fz[1]:.1f} N',580)
            text(panels[1],'Green dots: native loaded contacts | graphics only',610,scale=.47)
            cv2.polylines(panels[2],[project(qpos[:i+1,:2])],False,(180,50,55),3,cv2.LINE_AA)
            cv2.circle(panels[2],tuple(project(command['qpos'][j,:2])),6,(30,150,95),-1)
            for k,color in enumerate(COLORS):
                c=trace['body_pos'][i,ids[k],:2];theta=np.radians(signals['wheel_heading_deg'][i,k]);direction=np.array([np.cos(theta),np.sin(theta)])
                pp=project([c-.12*direction,c+.12*direction]);cv2.line(panels[2],tuple(pp[0]),tuple(pp[1]),color,5,cv2.LINE_AA)
            c=qpos[i,:2];theta=np.radians(signals['body_heading_deg'][i]);pp=project([c,c+.23*np.array([np.cos(theta),np.sin(theta)])])
            cv2.arrowedLine(panels[2],tuple(pp[0]),tuple(pp[1]),(155,35,45),3,cv2.LINE_AA,tipLength=.3)
            cursor=int(60+now/rt[-1]*550)
            for row in range(4):cv2.line(panels[3],(cursor,100+row*125),(cursor,175+row*125),(50,50,50),1)
            image=np.concatenate(panels,axis=1);proc.stdin.write(image.tobytes())
            if np.isclose(now,.2) or np.isclose(now,1.5):
                cv2.imwrite(str(a.output/f'frame_{now:.2f}s.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    finally:
        runtime.renderer.close();proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('FFmpeg failed')
    provenance=dict(renderer_sha256=sha(__file__),model_sha256=sha(a.model),qualification_sha256=sha(qualification_path),
        case_id=a.case,run_kind=run.get('kind'),method=run.get('method'),seed=run.get('seed'),speed_scale=run.get('speed_scale'),
        trace_sha256=sha(run_dir/'final_rollout.npz'),conditioning_mode=run.get('conditioning_mode'),
        command_diagnostics_sha256=sha(a.command_dir/'command_diagnostics.npz'),video_sha256=sha(video),
        frame_count=len(times),fps=fps,physics_steps_performed=0,source_RGB_included=False,
        fixed_camera=dict(distance_m=2.4,azimuth_deg=135,elevation_deg=-18,lookat='current root + [0,0,-.24]'),
        maximum_wheel_origin_FK_error_m=maximum,all_1113_native_poses_checked=True,
        collision_proxies_displayed=False,visual_meshes_opaque=True,XY_extent_m=extent,XY_center_m=center.tolist(),
        chart_limits=dict(load_N=[0,2500],slip_m_s=[0,5],utilization=[0,1.4],axial_beta_deg=[-90,90]),
        native_normal_contact_markers='reported normal points with per-point load >5 N; 12 mm marker radius',
        chart_clipping_is_visual_only=True)
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(video)


if __name__=='__main__':main()

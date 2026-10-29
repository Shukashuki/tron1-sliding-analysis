"""Render four qualified PhysX recordings with aligned geometric contacts.

MuJoCo supplies mesh graphics and forward kinematics only. Native row i
describes the contact geometry at recorded pose i, before integration to i+1.
The video does not correct poses, interpolate physics or relax success gates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from evaluate_hockey_physx import derive_signals, load, sha
from render_physx_ppo_fullbody import build_qpos, create_runtime, _pose, mesh_provenance

W,H=640,720


def text(image,value,y,x=16,scale=.5,color=(35,35,35)):
    cv2.putText(image,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('bundle','evidence','audit'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--run-name',default='run_v2')
    selection_group=p.add_mutually_exclusive_group(required=True)
    selection_group.add_argument('--case',action='append',help='cell:parent_transfer or cell:nominal; exactly four')
    selection_group.add_argument('--cases-json',type=Path,help='Four source records: case, bundle, evidence, audit, run_name; paths relative to this JSON')
    p.add_argument('--slow-window',nargs=2,type=float,default=[.65,1.25])
    p.add_argument('--prefix',default='contact_screens')
    p.add_argument('--snapshot-times',nargs='+',type=float,default=[.825,1.165,1.5,2.75,2.76])
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not a.prefix or not a.prefix.replace('_','').isalnum():raise ValueError('Unsafe output prefix')
    read=lambda path:json.loads(path.read_text())
    if a.cases_json is not None:
        if any(getattr(a,k) is not None for k in ('bundle','evidence','audit')):
            raise ValueError('Mixed-source selection must declare each panel independently')
        sources=read(a.cases_json)
        for row in sources:
            if set(row)!={'case','bundle','evidence','audit','run_name'}:raise ValueError('Incomplete panel source')
            for key in ('bundle','evidence','audit'):row[key]=(a.cases_json.parent/row[key]).resolve()
    else:
        if any(getattr(a,k) is None for k in ('bundle','evidence','audit')):
            raise ValueError('A common source requires bundle, evidence and audit')
        sources=[dict(case=case,bundle=a.bundle,evidence=a.evidence,audit=a.audit,run_name=a.run_name) for case in a.case]
    identities={(str(row['bundle'].resolve()),row['run_name'],row['case']) for row in sources}
    if len(sources)!=4 or len(identities)!=4:raise ValueError('Four distinct completed cases required')
    refmeta=read(a.reference/'report.json')
    if sha(a.model)!=refmeta['model_sha256']:raise ValueError('Visual model mismatch')
    reference=load(a.reference/'motion.npz');horizon=(len(reference['joint_pos'])-1)/float(reference['fps'])
    snapshots=set()
    for time in a.snapshot_times:
        if not np.isfinite(time) or not 0<=time<=horizon+1e-9 or not np.isclose(time/.0025,round(time/.0025),atol=1e-9,rtol=0):
            raise ValueError('Snapshot time must be on the physical clock within the reference horizon')
        snapshots.add(round(time/.0025))
    runtime=create_runtime(a.model,W,420)
    import mujoco
    camera=runtime.cameras[0];camera.distance=2.4;camera.azimuth=135.;camera.elevation=-18.
    cases=[]
    for source in sources:
        selection=source['case'];bundle=source['bundle'];audit_path=source['audit']/'report.json'
        plan=read(bundle/'plan.json');audit=read(audit_path)
        if audit['source_manifest_sha256']!=sha(bundle/'manifest.json'):raise ValueError('Audit source changed')
        if sha(a.reference/'motion.npz')!=plan['reference_sha256']['motion.npz']:raise ValueError('Reference mismatch')
        cell,stage=selection.split(':')
        if stage not in ('parent_transfer','nominal'):raise ValueError('Require qualified evaluation')
        planrow=next(row for row in plan['cells'] if row['id']==cell)
        cfg=read(bundle/planrow['protocol']);policy='parent' if stage=='parent_transfer' else cell[len(planrow['model'])+1:]
        row=next(r for r in audit['cases'] if r['cell']==cell and r['policy']==policy)
        directory=source['evidence']/source['run_name']/cell/stage;qpath=directory.parent/(stage+'_qualification.json')
        if sha(qpath)!=row['qualification_sha256']:raise ValueError('Qualification changed')
        checked=read(qpath);run=read(directory/'report.json');q=checked['qualification']
        if checked['run_report_sha256']!=sha(directory/'report.json'):raise ValueError('Run identity mismatch')
        if checked['protocol_sha256']!=sha(bundle/planrow['protocol']):raise ValueError('Protocol changed')
        if checked['source_manifest_sha256']!=sha(bundle/'manifest.json'):raise ValueError('Qualification source changed')
        if run['motion_sha256']!=sha(a.reference/'motion.npz'):raise ValueError('Run reference mismatch')
        for name,digest in run['artifacts'].items():
            if sha(directory/name)!=digest or row['raw_sha256'][name]!=digest:raise ValueError('Recording changed')
        trace=load(directory/'final_rollout.npz');raw=load(directory/'native_contact_buffers.npz')
        dt=float(trace['physics_dt_s'])
        if not np.isclose(dt,.0025,atol=1e-12,rtol=0):raise ValueError('Expected 400 Hz source')
        times=np.arange(len(trace['root_pos']))*dt
        signals=derive_signals(trace,{})
        contact=contact_kinematics(previous_state_trace(trace),raw,.0001)
        qpos,_=build_qpos(runtime.model,dict(time_s=times,joint_names=trace['joint_names'],
            position=trace['root_pos'],quaternion=trace['root_quat'],joint_pos=trace['joint_pos']))
        ids=[trace['body_names'].astype(str).tolist().index('wheel_'+side+'_Link') for side in ('L','R')]
        mjids=[int(runtime.model.body('wheel_'+side+'_Link').id) for side in ('L','R')]
        error=0.
        for i,pose in enumerate(qpos):
            _pose(runtime.model,runtime.data,pose)
            error=max(error,float(np.linalg.norm(runtime.data.xpos[mjids]-trace['body_pos'][i,ids],axis=-1).max()))
        if error>2e-5:raise ValueError(f'Visual FK differs: {selection}: {error}')
        labels={'native_split':'Cylinder | static/dynamic 0.9/0.7',
                'slices9_split':'Nine rigid slices | 0.9/0.7','native_equal':'Cylinder | static/dynamic 0.7/0.7'}
        cases.append(dict(selection=selection,model=planrow['model'],policy=policy,label=labels[planrow['model']],
            manifest_sha256=sha(bundle/'manifest.json'),audit_sha256=sha(audit_path),
            protocol_sha256=checked['protocol_sha256'],run_name=source['run_name'],
            trace=trace,raw=raw,signals=signals,contact=contact,qpos=qpos,row=row,q=q,fk_error=error))

    def body_panel(case,i,slow):
        now=i*.0025;panel=np.full((H,W,3),249,np.uint8)
        pose=case['qpos'][i];raw=case['raw'];contact=case['contact'];ci=i+1
        # The contact result indexed i+1 was reconstructed from native row i
        # and pre-step pose i. No post-step displacement is applied to markers.
        _pose(runtime.model,runtime.data,pose);camera.lookat[:]=pose[:3]+[0.,0.,-.24]
        runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
        for side in (0,1):
            count=int(raw[f'w{side}_normal_buffer_4'][i].item());start=int(raw[f'w{side}_normal_buffer_5'][i].item())
            normal=raw[f'w{side}_normal_buffer_2'][i,start:start+count]
            separation=raw[f'w{side}_normal_buffer_3'][i,start:start+count].reshape(-1)
            valid=(np.linalg.norm(normal,axis=-1)>=1e-8)&(separation<=.0001)
            if valid.sum()!=contact['touching_point_count'][ci,side]:raise ValueError('Marker timing mismatch')
            points=raw[f'w{side}_normal_buffer_1'][i,start:start+count][valid]-case['trace']['environment_origin_w']
            for point in points:
                scene=runtime.renderer.scene
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3,.008),point,np.eye(3).ravel(),np.array([.1,.9,.25,1.]))
                scene.ngeom+=1
        panel[65:485]=runtime.renderer.render()
        text(panel,case['label'],25,scale=.59)
        text(panel,f"{'Frozen parent' if case['policy']=='parent' else case['policy']+' fine-tune'} | {'8x slow, every 2.5 ms' if slow else 'real time'}",49,scale=.51)
        text(panel,f"t={now:.4f}s | speed={case['signals']['body_speed_m_s'][i]:.3f}m/s | tilt={case['signals']['root_tilt_deg'][i]:.1f}deg",510)
        for side,color in enumerate(((35,105,185),(205,105,30))):
            state='TOUCH' if contact['touching'][ci,side] else ('UNKNOWN' if contact['contact_unknown'][ci,side] else 'NO CONTACT')
            gap=contact['min_separation_m'][ci,side]*1000
            gaptext=f'{gap:.3f} mm' if np.isfinite(gap) else 'unavailable'
            text(panel,f"{'L' if side==0 else 'R'}: {state} | native gap {gaptext}",536+side*24,color=color)
        slip=[]
        for side,events in zip(('L','R'),case['q']['corroborated_slide_events']):
            if any(e['start_s']-1e-9<=now<e['end_s']-1e-9 for e in events):slip.append(side)
        text(panel,'Force-corroborated lateral slide: '+('/'.join(slip) if slip else 'none'),586,scale=.46)
        for side in (0,1):
            for k in range(round(horizon/.0025)):
                state=contact['touching'][k+1,side];unknown=contact['contact_unknown'][k+1,side]
                color=(50,160,90) if state else ((215,160,35) if unknown else (205,65,65))
                x0=45+round(k*.0025/horizon*575);x1=45+round((k+1)*.0025/horizon*575)
                cv2.rectangle(panel,(x0,603+side*16),(max(x0,x1),613+side*16),color,-1)
            text(panel,('L','R')[side],612+side*16)
        cursor=45+round(now/horizon*575);cv2.line(panel,(cursor,601),(cursor,631),(20,20,20),2)
        success=case['q']['task_success_claimed']
        text(panel,f"Both touch: {case['row']['both_contact_percent']:.2f}% | whole-task {'PASS' if success else 'FAIL'}",654,
             color=(30,130,70) if success else (185,45,45),scale=.48)
        text(panel,'Green: touch | red: missing | amber: unknown',677,scale=.45)
        text(panel,'Fixed 2.4 m follow | recorded PhysX | no pose correction',700,scale=.42)
        return panel

    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();a.output.mkdir(parents=True)
    windows=[(a.prefix+'_fullbody_4x1',np.arange(round(horizon/.0025)+1)[::8],False),
             (a.prefix+'_contact_excerpt_8x_4x1',np.arange(round(a.slow_window[0]/.0025),round(a.slow_window[1]/.0025)+1),True)]
    artifacts=[]
    try:
        for name,indices,slow in windows:
            if len(indices)==0 or indices.min()<0 or indices.max()>round(horizon/.0025):raise ValueError('Invalid frame window')
            path=a.output/(name+'.mp4')
            proc=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                '-s',f'{4*W}x{H}','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast',
                '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
            try:
                for i in indices:
                    frame=np.concatenate([body_panel(case,int(i),slow) for case in cases],axis=1)
                    proc.stdin.write(frame.tobytes())
                    if i in snapshots:
                        cv2.imwrite(str(a.output/f'{name}_{i*.0025:.4f}s.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            finally:proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Video encoder failed')
            artifacts.append(dict(file=path.name,sha256=sha(path),frames=len(indices),fps=50,
                physical_window_s=[float(indices[0]*.0025),float(indices[-1]*.0025)],slowdown=8 if slow else 1))
    finally:runtime.renderer.close()
    audits={c['audit_sha256'] for c in cases}
    report=dict(kind='hockey_contact_screen_render_v2',renderer_sha256=sha(__file__),
        model_sha256=sha(a.model),meshes=mesh_provenance(a.model),audit_sha256=next(iter(audits)) if len(audits)==1 else None,
        reference_sha256=sha(a.reference/'motion.npz'),physics_steps=0,pose_corrections=0,source_RGB_included=False,
        native_contact_alignment='pose i / raw row i / reconstructed contact i+1',contact_force_threshold=None,
        marker_radius_m=.008,contact_separation_tolerance_m=.0001,collision_proxies_displayed=False,
        camera=dict(distance_m=2.4,azimuth_deg=135,elevation_deg=-18,lookat='recorded root + [0,0,-0.24]'),
        cases=[dict(selection=c['selection'],qualification_sha256=c['row']['qualification_sha256'],
            source_manifest_sha256=c['manifest_sha256'],audit_sha256=c['audit_sha256'],
            protocol_sha256=c['protocol_sha256'],run_name=c['run_name'],
            raw_sha256=c['row']['raw_sha256'],wheel_FK_max_error_m=c['fk_error']) for c in cases],artifacts=artifacts)
    (a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(artifacts))


if __name__=='__main__':main()

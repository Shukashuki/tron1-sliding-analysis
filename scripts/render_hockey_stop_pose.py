"""4x1 source/2D/hip-relative3D/quality preview; no retarget or simulation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from capture_hockey_stop_pose import EDGES,LEGS


def run(a):
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    assert not a.output.exists()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=json.loads((a.pose/'report.json').read_text())
    assert sha(a.video)==report['video_sha256'] and sha(a.pose/'pose.npz')==report['pose_sha256']
    with np.load(a.pose/'pose.npz',allow_pickle=False) as z:r={k:z[k] for k in z.files}
    cap=cv2.VideoCapture(str(a.video));assert cap.isOpened();a.output.mkdir(parents=True)
    font=ImageFont.load_default(size=18);small=ImageFont.load_default(size=15)
    visibility=r['visibility'];presence=r['presence'];quality=np.minimum(visibility,presence)
    time=r['time_s'];legq=np.stack([quality[:,ids].min(-1) for ids in LEGS.values()],-1)
    az=np.deg2rad(135);el=np.deg2rad(20)
    view=np.array([[-np.sin(az),np.cos(az),0],[-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)]])
    def color(edge):return '#d62728' if edge[0]%2==1 else '#9467bd'
    def bones(draw,points,offset):
        for edge in EDGES:
            if np.isfinite(points[list(edge)]).all():
                c=color(edge) if np.min(quality[i,list(edge)])>=.5 else '#aaaaaa'
                draw.line([tuple(points[k]+offset) for k in edge],fill=c,width=3)
        for k in (23,24,25,26,27,28,29,30,31,32):
            if np.isfinite(points[k]).all():
                x,y=points[k]+offset;draw.ellipse((x-3,y-3,x+3,y+3),fill='#d62728' if k%2 else '#9467bd')
    writer=imageio_ffmpeg.write_frames(str(a.output/'pose_capture_4x1.mp4'),(2560,640),fps=report['fps'],
        codec='libx264',quality=7,macro_block_size=2,output_params=['-movflags','+faststart']);writer.send(None)
    try:
        for i in range(len(time)):
            ok,frame=cap.read();assert ok
            im=Image.new('RGB',(2560,640),'#fafafa');d=ImageDraw.Draw(im)
            picture=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));picture.thumbnail((640,400))
            im.paste(picture,(0,60));d.text((12,10),'Source video + detected landmarks',font=font,fill='#111111')
            points=r['image_landmarks_normalized'][i,:,:2]*np.array(picture.size)+[0,60]
            bones(d,points,np.array([0,0]))
            d.text((12,485),f"t={time[i]:.3f}s | {a.video.name}",font=small,fill='#333333')
            d.text((12,515),'Red: anatomical left | purple: right | gray: low confidence',font=small,fill='#333333')
            d.text((12,545),'No gap filling or foot-lock correction',font=small,fill='#333333')
            d.text((652,10),'2D keypoints | image coordinates',font=font,fill='#111111');bones(d,points,np.array([640,0]))
            d.text((652,485),'Heel/toe landmarks are not measured blade edges.',font=small,fill='#333333')
            d.text((652,515),'Image directions are not ground-plane sideslip angles.',font=small,fill='#8b2222')
            model=r['hip_relative_world_landmarks_m'][i]
            relative=np.stack((model[:,0],model[:,2],-model[:,1]),-1)
            projected=relative@view.T
            projected=projected*np.array([210,-210])+[1600,290]
            d.text((1292,10),'Hip-relative3D skeleton | fixed projection',font=font,fill='#111111')
            bones(d,projected,np.array([0,0]))
            d.text((1292,485),'Model-predicted relative pose; no global pelvis trajectory.',font=small,fill='#333333')
            d.text((1292,515),'No calibrated stopping distance or speed from this output.',font=small,fill='#8b2222')
            x0=1932;d.text((x0,10),'Bilateral leg quality and missing data',font=font,fill='#111111')
            for j,(side,c) in enumerate(zip(LEGS,('#d62728','#9467bd'))):
                path=[(x0+600*t/max(time[-1],1e-6),270-190*np.clip(legq[k,j],0,1)) for k,t in enumerate(time)]
                d.line(path,fill=c,width=3);d.text((x0,305+35*j),f'{side}: current minimum visibility/presence {legq[i,j]:.3f}',font=small,fill=c)
            x=x0+600*time[i]/max(time[-1],1e-6);d.line((x,65,x,270),fill='#111111',width=2)
            d.line((x0,175,x0+600,175),fill='#888888',width=1)
            lines=[f"Pose detection coverage {100*report['detected_fraction']:.1f}%",
                'Confidence is not a ground-truth accuracy measurement.',
                'Forces and contact state: NOT observed',
                'Global trajectory / GMR / physics: NOT executed']
            for j,line in enumerate(lines):d.text((x0,405+34*j),line,font=small,fill='#333333')
            writer.send(np.asarray(im))
            if i in [int(len(time)*.25),int(len(time)*.5),int(len(time)*.75)]:im.save(a.output/f'frame_{i:03d}.png')
            if i==int(len(time)*.5):im.save(a.output/'comparison.png')
            if i%100==0:print('render',i,flush=True)
    finally:cap.release();writer.close()
    provenance=dict(kind='source_pose_qualification_preview',fps=report['fps'],frames=len(time),
        source_sha256=sha(__file__),source_video_sha256=sha(a.video),pose_sha256=sha(a.pose/'pose.npz'),
        fixed_projection=True,global_translation_inferred=False,pose_corrections=0,physics_steps=0,
        source_video_redistribution_rights_not_inferred_from_metadata_license=True,
        artifacts={p.name:sha(p) for p in a.output.iterdir() if p.is_file()})
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('video','pose','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())

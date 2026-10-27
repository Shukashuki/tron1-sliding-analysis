"""CPU video pose capture for source qualification, NOT metric world mocap.

MediaPipe WorldLandmarks are hip-relative; retain this explicit limitation.
Never infer contact force, friction saturation, global speed or stopping distance
from the normalized image landmarks or the hip-relative3D output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

LEGS={'left':(23,25,27,29,31),'right':(24,26,28,30,32)}
EDGES=((11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),
       (23,25),(25,27),(27,29),(29,31),(31,27),(24,26),(26,28),(28,30),(30,32),(32,28))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def capture(a):
    import cv2
    import mediapipe as mp
    assert not a.output.exists()
    assert sha(a.model)=='64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b'
    a.output.mkdir(parents=True)
    cap=cv2.VideoCapture(str(a.video));assert cap.isOpened()
    fps=float(cap.get(cv2.CAP_PROP_FPS));assert np.isfinite(fps) and fps>0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    options=mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(a.model),delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1,
        min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    images=[];world=[];visibility=[];presence=[];found=[];start=time.monotonic();i=0
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
        while True:
            ok,frame=cap.read()
            if not ok:break
            result=model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),round(1000*i/fps))
            valid=bool(result.pose_landmarks)
            if valid:
                point=result.pose_landmarks[0];w=result.pose_world_landmarks[0]
                images.append([[p.x,p.y,p.z] for p in point]);world.append([[p.x,p.y,p.z] for p in w])
                visibility.append([p.visibility for p in point]);presence.append([p.presence for p in point])
            else:
                images.append(np.full((33,3),np.nan));world.append(np.full((33,3),np.nan))
                visibility.append(np.zeros(33));presence.append(np.zeros(33))
            found.append(valid);i+=1
            if i%100==0:print('pose',i,'wall_s',round(time.monotonic()-start,1),flush=True)
    cap.release();assert i>0
    images=np.asarray(images,dtype=np.float32);world=np.asarray(world,dtype=np.float32)
    visibility=np.asarray(visibility,dtype=np.float32);presence=np.asarray(presence,dtype=np.float32);found=np.asarray(found)
    np.savez_compressed(a.output/'pose.npz',time_s=np.arange(i)/fps,image_landmarks_normalized=images,
        hip_relative_world_landmarks_m=world,visibility=visibility,presence=presence,detected=found,
        fps=fps,image_width=width,image_height=height)
    legs={}
    for side,index in LEGS.items():
        reliable=found&np.all(visibility[:,index]>=.5,axis=-1)&np.all(presence[:,index]>=.5,axis=-1)
        legs[side]=dict(all_hip_knee_ankle_heel_toe_confident_fraction=float(np.mean(reliable)),
            ankle_heel_toe_visibility_median=np.median(visibility[:,index[-3:]],axis=0).tolist())
    report=dict(kind='video_pose_source_qualification_only',status='captured_requires_quality_review',
        video_file=a.video.name,video_sha256=sha(a.video),pose_sha256=sha(a.output/'pose.npz'),
        source_sha256=sha(__file__),model_sha256=sha(a.model),mediapipe_version=mp.__version__,
        model_source_url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
        cpu_delegate=True,frames=i,fps=fps,width=width,height=height,detected_fraction=float(found.mean()),
        bilateral_leg_quality=legs,global_translation_available=False,metric_global_speed_available=False,
        stopping_distance_measurable=False,force_contact_observed=False,friction_saturation_measured=False,
        world_coordinates_semantics='predicted33landmarks in meters relative to hip center; NOT rink-global coordinates',
        no_gap_filling=True,no_foot_lock=True,GMR_retarget_executed=False,physics_executed=False,
        wall_time_s=time.monotonic()-start,task_success_claimed=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('video','model','output'):p.add_argument('--'+k,type=Path,required=True)
    capture(p.parse_args())

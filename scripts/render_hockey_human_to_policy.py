#!/usr/bin/env python3
"""Render one shared clock: source RGB, human skeleton, GMR, recorded PhysX PPO.

The source is held at its original 30 Hz samples. The slow video displays every
400 Hz recorded state. Rendering uses forward kinematics only, never dynamics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

W, H = 640, 720
REPO = Path(__file__).resolve().parents[1]
EDGES = [(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),
         (0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),
         (18,20),(9,14),(14,17),(17,19),(19,21)]
LEFT = {1,4,7,10,13,16,18,20}
RIGHT = {2,5,8,11,14,17,19,21}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def timeline(source_times, physics_dt, horizon):
    """Zero-order hold on a shared elapsed clock; never normalize durations."""
    source_times = np.asarray(source_times, dtype=float)
    require(source_times.ndim == 1 and len(source_times) > 1
            and np.isfinite(source_times).all() and abs(source_times[0]) < 1e-10
            and (np.diff(source_times) > 0).all(), 'Invalid source clock')
    require(np.isfinite(physics_dt) and physics_dt > 0 and np.isfinite(horizon)
            and horizon > 0, 'Invalid physical clock')
    steps = round(horizon / physics_dt)
    require(abs(steps * physics_dt - horizon) < 1e-9, 'Horizon must be a physics sample')
    times = np.arange(steps + 1) * physics_dt
    source_index = np.searchsorted(source_times, times + 1e-10, side='right') - 1
    source_index = np.clip(source_index, 0, len(source_times) - 1)
    return times, source_index, times > source_times[-1] + 1e-10


def text(im, message, x, y, size=.55, color=(35,44,58), thickness=1):
    cv2.putText(im, message, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


def video_writer(path, fps):
    import imageio_ffmpeg
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner',
        '-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{4*W}x{H}','-r',str(fps),'-i','pipe:0','-an',
        '-c:v','libx264','-threads','2','-preset','fast','-crf','19',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)], stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true', help='Only four diagnostic stills')
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists; use a fresh directory')
    cfg = read(args.config)
    paths = {name:(args.root / value).resolve() for name,value in cfg['paths'].items()}
    for name, expected in cfg['sha256'].items():
        require(sha(paths[name]) == expected, f'Pinned input changed: {name}')
    gmr = dict(np.load(paths['gmr'], allow_pickle=False))
    trace = dict(np.load(paths['rollout'], allow_pickle=False))
    raw = dict(np.load(paths['contacts'], allow_pickle=False))
    reference = dict(np.load(paths['reference'], allow_pickle=False))
    gm = read(paths['gmr'].parent / 'report.json')
    rm = read(paths['reference'].parent / 'report.json')
    reconstruction = read(paths['reconstruction_provenance'])
    run = read(paths['rollout'].parent / 'report.json')
    checked = read(paths['qualification'])
    qualification = checked['qualification']
    robustness = read(paths['robustness'])
    require(reconstruction['video_sha256'] == sha(paths['video'])
            and reconstruction['outputs']['world_raw.npz'] == gm['source_sha256'],
            'RGB/reconstruction/GMR source mismatch')
    require(rm['retarget_sha256'] == sha(paths['gmr'])
            and rm['motion_sha256'] == sha(paths['reference'])
            and rm['model_sha256'] == gm['model_sha256'] == sha(paths['model']),
            'GMR/training/model identity mismatch')
    require(run['motion_sha256'] == sha(paths['reference'])
            and checked['run_report_sha256'] == sha(paths['rollout'].parent / 'report.json'),
            'Recorded policy/reference identity mismatch')
    for name in ('rollout','contacts'):
        require(run['artifacts'][paths[name].name] == sha(paths[name])
                == robustness['case_hashes']['nominal'][paths[name].name],
                'Recording differs from qualified nominal')
    require(run['actor_sha256'] == checked['actor_sha256'] == robustness['actor_sha256']
            == cfg['actor_sha256'], 'Actor identity mismatch')
    require(robustness['identities']['screen_qualification_sha256'] == sha(paths['qualification'])
            and robustness['all_33_qualifications_reproduced']
            and robustness['nominal_success'] and robustness['perturbation_successes'] == 32
            and robustness['perturbation_trials'] == 32
            and qualification['task_success_claimed'] and all(qualification['gates'].values()),
            'Selected success result is not fully qualified')
    material = run['material_audit']
    require(np.allclose(material['wheel_mu_static_range'], .9, atol=1e-7)
            and np.allclose(material['wheel_mu_dynamic_range'], .7, atol=1e-7)
            and material['live_physx_materials_verified'], 'Unexpected live friction')
    require(rm['source_world_XY_used'] is False, 'Unexpected task reference construction')

    dt = float(trace['physics_dt_s'])
    horizon = (len(reference['joint_pos'])-1) / float(reference['fps'])
    require(abs(dt-.0025) < 1e-12, 'Expected 400 Hz states')
    times, source_index, held = timeline(gmr['time_s'], dt, horizon)
    require(len(times) < len(trace['root_pos']), 'Need native contacts at the terminal pose')
    require(np.allclose(gmr['time_s'], np.arange(len(gmr['time_s']))/30,
                        atol=1e-9, rtol=0), 'Expected original 30 Hz source clock')
    require(abs(rm['source_pose_timing_unchanged_until_s']-gmr['time_s'][-1]) < 1e-9,
            'Training/source time relationship changed')

    cap = cv2.VideoCapture(str(paths['video']))
    require(abs(cap.get(cv2.CAP_PROP_FPS)-30) < 1e-8, 'Wrong source video frame rate')
    rgb = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    require(len(rgb) == len(gmr['time_s']), 'Source video/GMR frame mismatch')

    from analyze_hockey_ground_contact import contact_kinematics
    from audit_hockey_contact_timing import previous_state_trace
    from render_physx_ppo_fullbody import build_qpos, create_runtime, _pose, mesh_provenance
    runtime = create_runtime(paths['model'], W, 420)
    import mujoco
    camera = runtime.cameras[0]
    camera.distance, camera.azimuth, camera.elevation = 2.0, 135., -18.
    skeleton_option = mujoco.MjvOption()
    skeleton_option.geomgroup[:] = 0
    skeleton_option.geomgroup[2] = 1
    skeleton_option.sitegroup[:] = 0
    contact = contact_kinematics(previous_state_trace(trace), raw, .0001)
    qpos, _ = build_qpos(runtime.model, dict(time_s=np.arange(len(trace['root_pos']))*dt,
        joint_names=trace['joint_names'],position=trace['root_pos'],
        quaternion=trace['root_quat'],joint_pos=trace['joint_pos']))
    body_ids = [list(trace['body_names']).index('wheel_'+side+'_Link') for side in ('L','R')]
    mj_ids = [runtime.model.body('wheel_'+side+'_Link').id for side in ('L','R')]
    fk_error = 0.
    for i, pose in enumerate(qpos):
        _pose(runtime.model,runtime.data,pose)
        fk_error = max(fk_error,float(np.linalg.norm(
            runtime.data.xpos[mj_ids]-trace['body_pos'][i,body_ids],axis=-1).max()))
    require(fk_error < 2e-5, 'Rendered wheel positions differ from PhysX recording')

    def add_sphere(point, radius, color):
        scene = runtime.renderer.scene
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3,radius),point,np.eye(3).ravel(),np.array([*color,1.]))
        scene.ngeom += 1

    def skeleton_image(joints):
        _pose(runtime.model,runtime.data,gmr['qpos'][0])
        camera.lookat[:] = joints[0]+[0.,0.,-.24]
        runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=skeleton_option)
        for a,b in EDGES:
            color = (.86,.29,.21) if a in LEFT or b in LEFT else (
                (.23,.44,.82) if a in RIGHT or b in RIGHT else (.34,.39,.45))
            scene = runtime.renderer.scene
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([*color,1.]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.009,joints[a],joints[b])
            scene.ngeom += 1
        for point in joints[:22]:
            add_sphere(point,.015,(.17,.22,.27))
        return runtime.renderer.render()

    def robot_image(pose, i=None):
        _pose(runtime.model,runtime.data,pose)
        camera.lookat[:] = pose[:3]+[0.,0.,-.24]
        runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
        if i is not None:
            # Native contact row i is located at pre-step recorded pose i.
            for side in (0,1):
                count = int(raw[f'w{side}_normal_buffer_4'][i].item())
                start = int(raw[f'w{side}_normal_buffer_5'][i].item())
                normal = raw[f'w{side}_normal_buffer_2'][i,start:start+count]
                gap = raw[f'w{side}_normal_buffer_3'][i,start:start+count].reshape(-1)
                valid = (np.linalg.norm(normal,axis=-1)>=1e-8)&(gap<=.0001)
                require(valid.sum() == contact['touching_point_count'][i+1,side],
                        'Contact marker/sample mismatch')
                for point in raw[f'w{side}_normal_buffer_1'][i,start:start+count][valid]:
                    add_sphere(point-trace['environment_origin_w'],.008,(.1,.9,.25))
        return runtime.renderer.render()

    # Each source pose is rendered once. Their clock is never sped up to match
    # the shorter source duration to the longer robot task.
    source_panels = []
    for k, frame in enumerate(rgb):
        letterbox = np.full((420,W,3),241,np.uint8)
        h,w = frame.shape[:2]
        ratio = min(W/w,420/h)
        resized = cv2.resize(frame,(round(w*ratio),round(h*ratio)))
        y,x = (420-resized.shape[0])//2,(W-resized.shape[1])//2
        letterbox[y:y+resized.shape[0],x:x+resized.shape[1]] = resized
        source_panels.append((letterbox,skeleton_image(gmr['human_joints_zup_m'][k]),
                              robot_image(gmr['qpos'][k])))

    def compose(i, slow):
        now, k = times[i], int(source_index[i])
        out = np.full((H,4*W,3),249,np.uint8)
        out[:62] = (22,33,48)
        text(out,'HOCKEY STOP  |  HUMAN TO TRON1',20,39,.83,(245,248,255),2)
        text(out,f'SHARED TIME  {now:5.3f} / {horizon:.2f} s',1440,39,.8,(245,248,255),2)
        text(out,'8x SLOW / all 400 Hz states' if slow else 'REAL TIME / 50 fps',2095,39,.62,(116,218,239))
        headers = ['1  Human video','2  Human skeleton','3  GMR retarget','4  Trained PPO / PhysX']
        subtitles = ['Original hockey-stop episode','GVHMR estimate / scaled to robot legs',
                     'Source reference / kinematic playback','mu_s = 0.9  |  mu_d = 0.7']
        for p in range(4):
            x = p*W
            text(out,headers[p],x+18,88,.68,thickness=2)
            text(out,subtitles[p],x+18,111,.49)
            out[120:540,x:x+W] = source_panels[k][p] if p<3 else robot_image(qpos[i],i)
            if p:
                cv2.line(out,(x,63),(x,633),(211,217,224),1)
            if p<3:
                text(out,f'Source frame {k:02d} / 64  |  sample t={gmr["time_s"][k]:.3f} s',x+18,566,.49)
                if held[i]:
                    cv2.rectangle(out,(x+12,504),(x+W-12,537),(255,229,181),-1)
                    text(out,'SOURCE ENDED - LAST POSE HELD',x+24,527,.54,(113,66,11),2)
        text(out,'Original framing / shared playback clock',18,594,.5)
        text(out,'Source samples: 0.000-2.133 s',18,620,.5)
        text(out,'Fixed camera / estimated 3D, uncalibrated',W+18,594,.5)
        text(out,'Floor shown as geometry; no measured forces',W+18,620,.47)
        text(out,'Pinned GMR + its existing height projection',2*W+18,594,.48)
        text(out,'Source XY retained; wheel spin placeholder',2*W+18,620,.48)
        speed = float(np.linalg.norm(trace['root_com_vel'][i,:2]))
        touching = contact['touching'][i+1]
        text(out,f'vXY={speed:.2f} m/s  |  L: {"TOUCH" if touching[0] else "OFF"}  R: {"TOUCH" if touching[1] else "OFF"}',3*W+18,566,.51)
        text(out,'Nominal PASS | 33/33 prescribed checks pass',3*W+18,594,.49,(28,121,66))
        text(out,'GMR posture + designed stop XY | RMSE 2.31 cm',3*W+18,620,.46)
        # The task phase is designed, not a measured event in the source RGB.
        out[638:] = (236,241,246)
        text(out,'DESIGNED TASK',18,670,.51)
        x0,x1 = 235,4*W-30
        def px(t):
            return x0+round(t/horizon*(x1-x0))
        for start,end,label,color in [(0,.4,'Approach',(186,208,226)),
                                      (.4,1.8,'Braking',(247,204,145)),
                                      (1.8,horizon,'Stop / hold',(178,219,194))]:
            cv2.rectangle(out,(px(start),648),(px(end),677),color,-1)
            text(out,label,px(start)+12,669,.51)
        cursor = px(now)
        cv2.line(out,(cursor,643),(cursor,682),(24,36,50),2)
        end = px(float(gmr['time_s'][-1]))
        cv2.line(out,(end,648),(end,704),(142,86,29),1)
        text(out,'Source ends 2.133 s; PPO continues',end-180,704,.43,(120,70,23))
        if now<rm['initial_posture_blend_s']:
            text(out,'PPO reference: initial posture blend (0-0.30 s)',235,704,.46)
        else:
            text(out,'Fixed follow offset and zoom in all 3D panels',235,704,.46)
        return out

    args.output.mkdir(parents=True)
    snapshot_indices = {round(t/dt) for t in (0.,.8,1.2,2.4)}
    outputs = []
    try:
        if args.preview:
            indices = sorted(snapshot_indices)
            writers = []
        else:
            indices = range(len(times))
            writers = [video_writer(args.output/'human_gmr_ppo_realtime_4x1.mp4',50),
                       video_writer(args.output/'human_gmr_ppo_slow8x_4x1.mp4',50)]
        try:
            for i in indices:
                frame = compose(i,True)
                if i in snapshot_indices:
                    cv2.imwrite(str(args.output/f'comparison_{times[i]:.3f}s.png'),
                                cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                if writers:
                    writers[1].stdin.write(frame.tobytes())
                    if i%8 == 0:
                        frame[:62] = compose_header(times[i],horizon)
                        writers[0].stdin.write(frame.tobytes())
                if i%100 == 0:
                    print(f'Rendered physical state {i}/{len(times)-1}',flush=True)
        finally:
            for writer in writers:
                writer.stdin.close()
            for writer in writers:
                require(writer.wait() == 0,'Video encoding failed')
    finally:
        runtime.renderer.close()
    with (args.output/'frame_alignment.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['physical_index','elapsed_s','source_frame','source_sample_s',
                         'source_held_after_end','included_in_realtime_video'])
        for i,t in enumerate(times):
            writer.writerow([i,f'{t:.7f}',int(source_index[i]),
                f'{gmr["time_s"][source_index[i]]:.9f}',int(held[i]),int(i%8==0)])
    if not args.preview:
        for name,expected,slowdown in [('human_gmr_ppo_realtime_4x1.mp4',len(times[::8]),1),
                                      ('human_gmr_ppo_slow8x_4x1.mp4',len(times),8)]:
            path = args.output/name
            cap = cv2.VideoCapture(str(path));decoded=0
            require(cap.get(cv2.CAP_PROP_FPS)==50,'Encoded FPS mismatch')
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                require(frame.shape==(H,4*W,3),'Encoded dimensions mismatch')
                decoded+=1
            cap.release()
            require(decoded==expected,'Encoded frame count mismatch')
            outputs.append(dict(file=name,sha256=sha(path),decoded_frames=decoded,
                fps=50,slowdown=slowdown,encoded_duration_s=decoded/50))
    report = dict(kind='hockey_human_gmr_policy_aligned_render_v1',
        config_sha256=sha(args.config),renderer_sha256=sha(__file__),
        pinned_input_sha256=cfg['sha256'],actor_sha256=run['actor_sha256'],
        selected_run='v9 nominal, not a selected perturbation',
        mu_static=.9,mu_dynamic=.7,nominal_xy_RMSE_m=qualification['xy_RMSE_m'],
        robustness_passes=33,robustness_scope='one nominal + 32 prescribed initial-state perturbations',
        source_frame_count=len(rgb),source_last_sample_s=float(gmr['time_s'][-1]),
        source_encoded_duration_s=len(rgb)/30,shared_horizon_s=horizon,
        alignment='Shared elapsed seconds; source RGB/skeleton/GMR use the same preceding source frame',
        source_max_hold_delay_before_endpoint_s=float((times-gmr['time_s'][source_index])[~held].max()),
        post_source='All three source panels freeze with an explicit banner; policy continues',
        physics_dt_s=dt,slow_video_uses_all_400Hz_states=True,
        realtime_video_physics_stride=8,pose_interpolation=False,physics_steps=0,
        renderer_backend='MuJoCo OSMesa graphics/FK only; final panel is a recorded PhysX trajectory',
        camera=dict(distance_m=2.0,azimuth_deg=135,elevation_deg=-18,
                    lookat='each 3D subject pelvis/root + [0,0,-0.24]',zoom_changes=False),
        human_robot_leg_scale=gm['robot_leg_scale'],human_floor_is_visual_proxy=True,
        training_reference=dict(source_world_XY_used=False,
            initial_posture_blend_s=rm['initial_posture_blend_s'],
            brake_start_s=rm['brake_start_s'],brake_end_s=rm['brake_end_s']),
        wheel_FK_max_error_m=fk_error,native_contact_alignment='pose i / raw row i / reconstructed contact i+1',
        contact_tolerance_m=.0001,contact_force_threshold=None,
        visual_meshes=mesh_provenance(paths['model']),
        collision_proxies_displayed=False,source_RGB_included=True,artifacts=outputs)
    report['generated_file_sha256']={p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(args.output,flush=True)


def compose_header(now,horizon):
    header = np.full((62,4*W,3),(22,33,48),np.uint8)
    text(header,'HOCKEY STOP  |  HUMAN TO TRON1',20,39,.83,(245,248,255),2)
    text(header,f'SHARED TIME  {now:5.3f} / {horizon:.2f} s',1440,39,.8,(245,248,255),2)
    text(header,'REAL TIME / 50 fps',2095,39,.62,(116,218,239))
    return header


if __name__=='__main__':
    main()

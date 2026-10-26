"""PPO feedback + complete-rollout MPPI residual search in native PhysX.

No MuJoCo predictor, hidden reset, pose playback, external tire force, or new
policy training. The final weighted Gaussian mean is independently rolled out.
The saved ref is an actual dynamic trajectory, and is NOT certified feasible
merely because an NPZ was exported. Reference feasibility is reported separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from physx_reference_math import smooth_basis, trajectory_metrics, reduce_contacts
from mppi_shooting import _mppi_weights


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(a):
    cfg=json.loads(a.config.read_text())
    assert cfg['kind']=='physx_policy_assisted_mppi_reference_v1'
    assert not a.output_dir.exists() and 'CUDA_VISIBLE_DEVICES' not in os.environ
    for path, key in [(a.actor,'actor'),(a.motion,'motion'),(a.asset,'asset'),(a.original_motion,'original_car_motion')]:
        assert sha(path)==cfg[key+'_sha256'], f'{key} input changed'
    assert sha(a.tracking_root/'training/tron1_tracking.py')==cfg['tracking_sha256']
    assert sha(a.seed_trace)==json.loads(a.seed_report.read_text())['physics_trajectory_sha256']
    speed_spec=cfg.get('speed_reference')
    speed_scale=1. if speed_spec is None else float(speed_spec['effective_scale'])
    core_interval=(1.44/speed_scale,4./speed_scale)
    assert np.isfinite(speed_scale) and speed_scale>=1.
    sys.path.insert(0,str(a.tracking_root))
    os.environ['TRON1_BEYONDMIMIC_ROOT']=str(a.beyondmimic_root)
    from isaaclab.app import AppLauncher
    ordinal=int(a.device.split(':')[1])
    launch=argparse.Namespace(headless=True,device=a.device,enable_cameras=False,
        kit_args=f'--/renderer/activeGpu={ordinal} --/renderer/multiGpu/enabled=false --/physics/cudaDevice={ordinal} --/physics/sceneMultiGPUMode=0')
    app=AppLauncher(launch).app
    a.output_dir.mkdir()
    start=time.monotonic()
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from training.tron1_tracking import make_env_cfg, ACTION_JOINT_NAMES
        from physx_mjlab_recipe import install_physx_mjlab_recipe, audit_physx_mjlab_recipe
        torch.cuda.set_device(ordinal)
        torch.manual_seed(cfg['seed'])
        with np.load(a.original_motion,allow_pickle=False) as z:
            original={k:z[k] for k in z.files}
        with np.load(a.motion,allow_pickle=False) as z:
            frames=len(z['joint_pos'])
        steps=min(a.steps,frames) if a.steps else frames
        samples=a.samples if a.samples is not None else cfg['samples']
        iterations=a.iterations if a.iterations is not None else cfg['iterations']
        serial=getattr(a,'serial_candidates',False)
        n=1 if a.validate_mean is not None or serial else samples+1
        assert n>=1 and steps>=2 and iterations>=0
        # The old recipe is used only to construct the EXACT actor command/obs
        # and motor ABI. Planning costs below are explicit and independent.
        task=json.loads(a.seed_report.read_text())['evaluation_task_profile']
        envcfg=make_env_cfg(str(a.motion),str(a.asset),n,device=a.device,eval_mode=True,task_profile=task)
        envcfg.sim.dt=.0025;envcfg.decimation=8;envcfg.sim.render_interval=8
        envcfg.scene.contact_forces.update_period=.0025
        install_physx_mjlab_recipe(envcfg,training=False,fixed_mu=cfg['mu'])
        env=ManagerBasedRLEnv(cfg=envcfg)
        env.reset()
        robot=env.scene['robot'];command=env.command_manager.get_term('motion')
        names=list(robot.joint_names);body_names=list(robot.body_names)
        jid=[names.index(k) for k in ACTION_JOINT_NAMES]
        wid=[body_names.index(k) for k in ('wheel_L_Link','wheel_R_Link')]
        sensors=[env.scene.sensors[k] for k in ('wheel_L_ground','wheel_R_ground')]
        material=audit_physx_mjlab_recipe(env,check_usd=True)
        actor=torch.jit.load(str(a.actor),map_location=a.device).eval()
        basis=torch.tensor(smooth_basis(steps,cfg['knots']),dtype=torch.float32,device=a.device)
        std=np.array(cfg['noise_std']);rng=np.random.default_rng(cfg['seed'])
        mean=np.zeros((cfg['knots'],8));history=[]
        originals={k:torch.tensor(original[k][:steps],device=a.device,dtype=torch.float32)
                   for k in ('body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')}
        weights=cfg['planning_weights']
        # Cached PhysX sample contains COM velocity. This legacy setter is COM
        # based (explicitly NOT write_root_link_velocity_to_sim).
        with np.load(a.seed_trace,allow_pickle=False) as z:
            seed={k:torch.tensor(z[k][0,0],device=a.device,dtype=torch.float32) for k in
                  ('root_position_m','root_quaternion_wxyz','root_linear_velocity_w_m_s',
                   'root_angular_velocity_w_rad_s','joint_position_rad','joint_velocity_rad_s')}
            seed_names=z['joint_names'].astype(str).tolist()
        seed_order=[seed_names.index(k) for k in names]
        if speed_spec is not None:
            # Pre-accelerated entry state, explicitly NOT acceleration from rest.
            # Positions and orientation are unchanged. All initial COM angular/
            # linear and joint velocities follow the same physical time scaling.
            for key in ('root_linear_velocity_w_m_s','root_angular_velocity_w_rad_s','joint_velocity_rad_s'):
                seed[key]*=speed_scale

        def heading(q):
            return torch.atan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2].square()+q[...,3].square()))

        def rollout(latent,label,light=False):
            """Every world starts once; all subsequent steps are real dynamics."""
            env.reset()
            root=torch.cat((seed['root_position_m'],seed['root_quaternion_wxyz'])).repeat(n,1)
            root[:,:3]+=env.scene.env_origins
            velocity=torch.cat((seed['root_linear_velocity_w_m_s'],seed['root_angular_velocity_w_rad_s'])).repeat(n,1)
            robot.write_root_pose_to_sim(root)
            robot.write_root_velocity_to_sim(velocity)
            robot.write_joint_state_to_sim(seed['joint_position_rad'][seed_order].repeat(n,1),
                                          seed['joint_velocity_rad_s'][seed_order].repeat(n,1))
            env.scene.write_data_to_sim();env.scene.update(.0025)
            env.action_manager.reset()
            residual=torch.einsum('tk,nka->tna',basis,torch.tensor(latent,device=a.device,dtype=torch.float32))
            costs=torch.zeros(n,device=a.device);fault=torch.zeros(n,device=a.device,dtype=torch.bool)
            records={k:[] for k in ('root_pos','root_quat','root_com_vel','root_omega','joint_pos','joint_vel',
                'body_pos','body_com_pos','body_quat','body_com_vel','body_omega','normal_fz','lateral_slip_rms',
                'utilization_proxy','friction_anchor_power','friction_force_w','normal_points_count','friction_anchors_count')}
            actions=[];observations=[];raw_contacts=[]
            force_readback_error=torch.zeros((),device=a.device)
            previous=torch.zeros((n,8),device=a.device)
            def record(contacts=None):
                state=dict(root_pos=robot.data.root_pos_w-env.scene.env_origins,
                    root_quat=robot.data.root_quat_w,root_com_vel=robot.data.root_com_lin_vel_w,
                    root_omega=robot.data.root_com_ang_vel_w,joint_pos=robot.data.joint_pos[:,jid],
                    joint_vel=robot.data.joint_vel[:,jid],body_pos=robot.data.body_pos_w-env.scene.env_origins[:,None],
                    body_com_pos=robot.data.body_com_pos_w-env.scene.env_origins[:,None],
                    body_quat=robot.data.body_quat_w,body_com_vel=robot.data.body_com_lin_vel_w,
                    body_omega=robot.data.body_com_ang_vel_w)
                if contacts is None:
                    contacts={k:torch.zeros((n,2,3) if k=='friction_force_w' else (n,2),device=a.device)
                              for k in records if k not in state}
                state.update(contacts)
                for k in records:records[k].append(state[k].detach().clone())
            record()
            for frame in range(steps):
                command.time_steps.fill_(frame);command._refresh_relative_targets()
                obs=env.observation_manager.compute()['policy']
                assert obs.shape==(n,51)
                action=torch.clamp(actor(obs)+residual[frame],-1,1)
                env.action_manager.process_action(action)
                if not light:
                    actions.append(action.detach().clone());observations.append(obs.detach().clone())
                costs+=weights['action_rate']*(action-previous).square().sum(-1)/steps
                previous=action
                rq=originals['body_quat_w'][frame,0];rv=originals['body_lin_vel_w'][frame,0]
                refbeta=torch.atan2(rv[1],rv[0])-heading(rq)
                for substep in range(8):
                    env._sim_step_counter+=1
                    env.action_manager.apply_action();env.scene.write_data_to_sim()
                    env.sim.step(render=False);env.scene.update(.0025)
                    readings=[]
                    raw_step={}
                    for side,sensor in enumerate(sensors):
                        if light:
                            force=sensor.contact_physx_view.get_contact_force_matrix(.0025)
                            readings.append({'normal_fz':force.reshape(n,-1,3).sum(1)[:,2]})
                            continue
                        i=wid[side]
                        normal_data=sensor.contact_physx_view.get_contact_data(.0025)
                        friction_data=sensor.contact_physx_view.get_friction_data(.0025)
                        if a.validate_mean is not None:
                            for j,x in enumerate(normal_data):raw_step[f'w{side}_normal_buffer_{j}']=x.detach().clone()
                            for j,x in enumerate(friction_data):raw_step[f'w{side}_friction_buffer_{j}']=x.detach().clone()
                        data=reduce_contacts(normal_data,
                            friction_data,robot.data.body_com_pos_w[:,i],
                            robot.data.body_com_lin_vel_w[:,i],robot.data.body_com_ang_vel_w[:,i],
                            robot.data.body_link_quat_w[:,i],cfg['mu'])
                        fault|=data.pop('fault');readings.append(data)
                        matrix=sensor.contact_physx_view.get_contact_force_matrix(.0025)
                        force_readback_error=torch.maximum(force_readback_error,(matrix.reshape(n,-1,3).sum(1)[:,2]-data['normal_fz']).abs().max())
                    if raw_step:raw_contacts.append(raw_step)
                    contact={k:torch.stack([d[k] for d in readings],1) for k in readings[0]}
                    if not light:record(contact)
                    v=robot.data.root_com_lin_vel_w;q=robot.data.root_quat_w
                    beta=torch.atan2(v[:,1],v[:,0])-heading(q)
                    eb=torch.atan2(torch.sin(beta-refbeta),torch.cos(beta-refbeta))
                    xy=((robot.data.root_pos_w-env.scene.env_origins)[:,:2]-originals['body_pos_w'][frame,0,:2]).square().sum(-1)
                    speed=(torch.linalg.vector_norm(v[:,:2],dim=-1)-torch.linalg.vector_norm(rv[:2])).square()
                    upright=2*(q[:,1].square()+q[:,2].square())
                    gap=(1.-contact['normal_fz'].max(-1).values/5.).clamp(0,1)
                    cost=(weights['xy']*xy+weights['speed']*speed+weights['beta']*eb.square()
                          +weights['upright']*upright+weights['support_gap']*gap)
                    costs+=cost/(steps*8)
                if frame%100==0:print(label,'frame',frame,'wall_s',round(time.monotonic()-start,2),flush=True)
            if bool(fault.any()):raise RuntimeError('Invalid contact buffer, overflow, geometry, or normal force')
            if not light and float(force_readback_error)>1e-3:raise RuntimeError('Lightweight force-matrix readback differs from actual normal-point sum')
            values={k:torch.stack(v).cpu().numpy() for k,v in records.items()}
            cost_array=costs.detach().cpu().numpy()
            if not np.isfinite(cost_array).all():raise RuntimeError('Nonfinite dynamic rollout')
            if not light:
                values['actions']=torch.stack(actions).cpu().numpy()
                values['observations']=torch.stack(observations).cpu().numpy()
            if raw_contacts:
                np.savez_compressed(a.output_dir/'native_contact_buffers.npz',
                    **{k:torch.stack([d[k] for d in raw_contacts]).cpu().numpy() for k in raw_contacts[0]},
                    physical_sample_indices=np.arange(1,steps*8+1))
            return cost_array,values

        with torch.inference_mode():
            if a.validate_mean is not None:
                with np.load(a.validate_mean,allow_pickle=False) as z:mean=z['latent_mean']
                assert mean.shape==(cfg['knots'],8)
            else:
                baseline_cost,base=rollout(np.zeros((n,cfg['knots'],8)),'baseline')
                np.savez_compressed(a.output_dir/'baseline.npz',**{k:v[:,0] for k,v in base.items()})
                clone_error=float(np.max(abs(base['root_pos']-base['root_pos'][:,0:1])))
                history.append(dict(kind='zero_residual_baseline',costs=baseline_cost.tolist(),
                                    maximum_clone_root_position_difference_m=clone_error))
                if serial:
                    repeat,_=rollout(np.zeros((1,cfg['knots'],8)),'same_world_baseline_repeat',light=True)
                    history.append(dict(kind='same_world_baseline_repeat',costs=repeat.tolist(),
                        cost_difference_from_full_baseline=(repeat-baseline_cost).tolist(),
                        lightweight_normal_force_readback_verified_on_full_baseline=True))
                for iteration in range(iterations):
                    noise=rng.normal(size=(samples,cfg['knots'],8))*std
                    candidates=np.concatenate((mean[None],mean[None]+noise),0)
                    if serial:
                        cost=np.array([rollout(z[None],f'search_{iteration}_candidate_{i}',light=True)[0][0]
                                       for i,z in enumerate(candidates)])
                    else:cost,_=rollout(candidates,f'search_{iteration}')
                    probability,stats=_mppi_weights(cost[1:],mean,noise,std,cfg['temperature'])
                    if probability is None:raise RuntimeError('No finite MPPI samples')
                    mean=mean+np.einsum('n,nka->ka',probability,noise)
                    history.append(dict(iteration=iteration,costs=cost.tolist(),importance=stats,
                                        weighted_samples=samples,baseline_excluded_from_gaussian_population=True))
                    np.savez_compressed(a.output_dir/'mean_latest.npz',latent_mean=mean)
                    write(a.output_dir/'search_history.json',history)
            final_cost,final=rollout(np.repeat(mean[None],n,axis=0),'final_mean')
        trace={k:v[:,0] for k,v in final.items()}
        np.savez_compressed(a.output_dir/'final_rollout.npz',**trace,
            joint_names=np.array(ACTION_JOINT_NAMES),body_names=np.array(body_names),
            environment_origin_w=env.scene.env_origins[0].cpu().numpy(),physics_dt_s=.0025,policy_dt_s=.02)
        np.savez_compressed(a.output_dir/'mean.npz',latent_mean=mean)
        # Save all links, including IMU, by native names. Playback does not
        # change simulator states in generation. Sample BEFORE each action.
        index=np.arange(steps)*8
        np.savez_compressed(a.output_dir/'reference_candidate.npz',fps=np.array(50.),
            joint_pos=trace['joint_pos'][index],joint_vel=trace['joint_vel'][index],
            body_pos_w=trace['body_pos'][index],body_quat_w=trace['body_quat'][index],
            body_lin_vel_w=trace['body_com_vel'][index],body_ang_vel_w=trace['body_omega'][index],
            joint_names=np.array(ACTION_JOINT_NAMES),body_names=np.array(body_names))
        report=dict(kind=cfg['kind'],status='completed_candidate_requires_qualification',
            source_sha256=sha(__file__),math_source_sha256=sha(Path(__file__).with_name('physx_reference_math.py')),
            config_sha256=sha(a.config),actor_sha256=sha(a.actor),motion_sha256=sha(a.motion),asset_sha256=sha(a.asset),
            seed_trace_sha256=sha(a.seed_trace),seed_report_sha256=sha(a.seed_report),
            original_motion_sha256=sha(a.original_motion),tracking_source_sha256=cfg['tracking_sha256'],
            search_history=history,final_costs=final_cost.tolist(),
            metrics=trajectory_metrics(trace,original,core_interval_s=core_interval) if steps==frames else {'short_smoke_only':True,'task_success_claimed':False},
            steps=steps,physical_substeps_per_rollout=steps*8,worlds=n,iterations=iterations,
            serial_candidate_search=serial,planning_lightweight_contact_readback=serial,
            lightweight_planning_note='same cost at every400Hz step; direct native normal-force matrix; omit slip/anchor telemetry and state recording during proposal evaluation only; verify force sum on full baseline and fully record final mean',
            residual_interpolation='piecewise_cubic_smoothstep_times_sine_squared_endpoint_envelope',
            seed_initial_velocity_semantics='root_COM_linear_and_angular_via_legacy_COM_writer',
            initial_state_writes='only once per complete rollout; no writes within trajectory',
            speed_reference=speed_spec,initial_velocity_multiplier=speed_scale,
            starts_preaccelerated=speed_spec is not None,core_interval_s=list(core_interval),
            action_interface='clip(PPO(normalized_51D_observation)+smooth_independent_8D_residual,-1,1)',
            controller_label='PPO-assisted reference generation; NOT independent pure-model control baseline',
            material_audit=material,friction_law='native_PhysX_Coulomb_rigid_cylinder',
            native_friction_anchor_power='sum F_anchor dot [v_COM + omega cross (anchor-COM)]',
            wheel_utilization_proxy='norm(sum F_anchor_XY)/(mu*sum normal_Fz), loaded wheels only',
            pointwise_normal_to_friction_anchor_mapping_verified=False,
            pointwise_friction_saturation_certified=False,planning_costs_equal_PPO_reward=False,
            external_forces=0,policy_training=False,new_friction_sweep=False,
            single_world_validation=a.validate_mean is not None,wall_time_s=time.monotonic()-start,
            device=a.device,torch_version=torch.__version__,artifacts={})
        for p in a.output_dir.glob('*.npz'):report['artifacts'][p.name]=sha(p)
        write(a.output_dir/'report.json',report)
        env.close()
    except BaseException as e:
        write(a.output_dir/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
    finally:app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('config','actor','motion','asset','original-motion','seed-report','seed-trace','tracking-root','beyondmimic-root','output-dir'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--steps',type=int,default=0)
    p.add_argument('--samples',type=int)
    p.add_argument('--iterations',type=int)
    p.add_argument('--validate-mean',type=Path)
    p.add_argument('--serial-candidates',action='store_true')
    run(p.parse_args())

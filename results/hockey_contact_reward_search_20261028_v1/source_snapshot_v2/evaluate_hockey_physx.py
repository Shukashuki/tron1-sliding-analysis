#!/usr/bin/env python3
"""Independent local qualification of an archived native PhysX stop rollout.

Normal points and friction anchors stay separate. This verifies wheel-level
corroboration, never a pointwise friction-cone certificate. Instrumentation mode
can check old car traces without presenting them as hockey-stop experiments.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def verify_hockey_search_source(source,run):
    """Accept only the reviewed native-search implementation, not arbitrary costs."""
    source=Path(source).resolve()
    expected={
        'generate_hockey_physx_mppi.py':'260dc6112940382938e9e671ed2d2944a9771c7edcba3d1f4af257ffd0adbee3',
        'hockey_mppi_math.py':'8895c0482289d80e476e7cd8e016fd63a857920150b672c286faeb1b51f4ec4c',
        'dispatch_hockey_physx_mppi.py':'601bfe51d6558cdf7563a796f6aa9217876213a27e38d2a5a6cc20e0bcf6b8ba',
    }
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['kind']!='hockey_native_physx_mppi_snapshot_v1':raise ValueError('Wrong search snapshot')
    for row in manifest['files']:
        p=(source/row['path']).resolve()
        if not p.is_relative_to(source) or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise ValueError('Search snapshot changed')
    for name,digest in expected.items():
        if sha(source/name)!=digest:raise ValueError('Unreviewed native hockey search source')
    fields={'source_sha256':'generate_hockey_physx_mppi.py','task_math_sha256':'hockey_mppi_math.py',
            'config_sha256':'config.json','motion_sha256':'motion.npz',
            'conditioning_table_sha256':'nominal_conditioning.npz','initial_mean_sha256':'initial_mean.npz',
            'math_source_sha256':'physx_reference_math.py'}
    if any(run.get(k)!=sha(source/v) for k,v in fields.items()):raise ValueError('Search report/source mismatch')
    cfg=json.loads((source/'config.json').read_text())
    if (run.get('kind')!='hockey_native_physx_mppi_v1' or run.get('conditioning_mode')!='reference_dq'
            or not run.get('actual_actor_input_archived') or run.get('policy_weights_changed') is not False
            or run.get('policy_training') is not False or run.get('external_forces')!=0
            or not run.get('single_world_validation') or run.get('worlds')!=1 or run.get('iterations')!=0
            or run.get('search_history') or run.get('actor_sha256')!=cfg['actor_sha256']
            or run.get('asset_sha256')!=cfg['asset_sha256'] or run.get('tracking_source_sha256')!=cfg['tracking_sha256']):
        raise ValueError('Not an independent native hockey search-plan validation')
    return dict(source_manifest_sha256=sha(source/'manifest.json'),reviewed_source_sha256=expected,
                planning_costs_used_for_success=False,qualification_thresholds_unchanged=True)


def verify_control_replay(path,run_dir,run,source_audit):
    """Bind a separately executed CPU actor replay to this exact physical trace."""
    path=Path(path);run_dir=Path(run_dir)
    report=json.loads(path.read_text())
    files={'run_report_sha256':'report.json','trace_sha256':'final_rollout.npz','plan_sha256':'mean.npz'}
    if (report.get('kind')!='hockey_recorded_action_replay_v1' or report.get('status')!='passed'
            or report.get('audit_script_sha256')!=sha(Path(__file__).with_name('audit_hockey_control_replay.py'))
            or report.get('actor_sha256')!=run['actor_sha256'] or report.get('motion_sha256')!=run['motion_sha256']
            or any(report.get(k)!=sha(run_dir/v) for k,v in files.items())
            or report.get('source_audit')!=source_audit or report.get('control_steps_checked')!=run['steps']
            or report.get('tolerance')!=5e-5 or not 0<=report.get('maximum_action_error',float('inf'))<=5e-5):
        raise ValueError('Action replay does not certify this native validation trace')
    replay_path=path.parent/'replay.npz'
    if sha(replay_path)!=report['replay_sha256']:raise ValueError('Action replay arrays changed')
    replay=load(replay_path);trace=load(run_dir/'final_rollout.npz')
    expected=np.clip(np.clip(replay['raw_actor'],-1,1)+replay['residual'],-1,1)
    if (expected.shape!=trace['actions'].shape or not np.isfinite(expected).all()
            or np.max(abs(expected-trace['actions']),initial=0)>5e-5
            or not np.allclose(replay['decoded_actions'],expected,atol=1e-7,rtol=0)):
        raise ValueError('Replayed actions disagree with native applied actions')
    return dict(report_sha256=sha(path),maximum_action_error=report['maximum_action_error'],
                control_steps_checked=report['control_steps_checked'],independent_residual_interpolation=True,
                physics_replay_performed=False)


def verify_execution_evidence(root,run_dir):
    """Bind unmodified engine output to the launched wrapper via the collected audit.

    Kit shutdown did not return to the wrapper's post-run annotation code in the
    approved v2 probes. Never manufacture missing fields in the original report.
    """
    root=Path(root).resolve();run_dir=Path(run_dir).resolve()
    manifest=json.loads((root/'collection_manifest_v2.json').read_text())
    if manifest['approved_payload_sha256']!='fc87c10983c54ca74900fab66e86d498eb0e66def04e7b6a4cfffa5b12d3642c':
        raise ValueError('Unexpected approved payload')
    for row in manifest['files']:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Execution collection changed: '+row['path'])
    bundle=root/'source_snapshot_v2'
    frozen={'hockey_physx_probe.py':'dd29fe1ddd736f0faf8fd78e6290020374346e24753bb6259dac6e59cc97bf2e',
        'dispatch_hockey_physx_probe.py':'d0648de412fdfed7efaabc645f0c3609e538a2215f3bd71df1bdff6c300f1de2'}
    for name,digest in frozen.items():
        if sha(bundle/name)!=digest:raise ValueError('Frozen probe source changed')
    payload=json.loads((bundle/'manifest.json').read_text())
    for row in payload['files']:
        if sha(bundle/row['path'])!=row['sha256']:raise ValueError('Frozen payload changed')
    report=json.loads((run_dir/'report.json').read_text())
    records=json.loads((root/'probes_v2/status.json').read_text())
    matches=[r for r in records if (root/'probes_v2'/r['id']).resolve()==run_dir]
    if len(matches)!=1:raise ValueError('Missing unique execution record')
    row=matches[0];argv=row['argv'];get=lambda flag:argv[argv.index(flag)+1]
    remote_root='/data/zengxiong/workspaces/tron1-hockey-stop-20261026-v1'
    if (row['status']!='completed' or row['returncode']!=0 or not row['gpu_before_launch']['idle']
            or row['report_sha256']!=sha(run_dir/'report.json')
            or argv[3]!=remote_root+'/source_snapshot_v2/hockey_physx_probe.py'
            or get('--bundle')!=remote_root+'/source_snapshot_v2'
            or get('--output-dir')!=remote_root+'/probes_v2/'+row['id']
            or report['iterations']!=0 or report['search_history']
            or report['motion_sha256']!=sha(bundle/'motion.npz')
            or report['config_sha256']!=sha(bundle/'config.json')):
        raise ValueError('Probe execution identity mismatch')
    expected_steps={'smoke':16,'baseline':139}
    if report['steps']!=expected_steps.get(row['id']):raise ValueError('Unexpected probe length')
    for name,digest in report['artifacts'].items():
        if sha(run_dir/name)!=digest:raise ValueError('Probe artifact changed')
    return dict(wrapper_sha256=frozen['hockey_physx_probe.py'],
        collection_manifest_sha256=sha(root/'collection_manifest_v2.json'),
        execution_record_sha256=sha(root/'probes_v2/status.json'),
        source='collected frozen payload + dispatcher execution record; original run report unchanged',
        wrapper_post_run_annotations_present='hockey_wrapper_sha256' in report)


def segments(time,mask,*,interval_samples=False):
    """States span first-to-last timestamps; force intervals represent one dt each."""
    t=np.asarray(time,dtype=float);mask=np.asarray(mask,dtype=bool)
    if t.ndim!=1 or mask.shape!=t.shape or len(t)<2 or not np.isfinite(t).all():raise ValueError('Invalid sampled event')
    dt=float(np.median(np.diff(t)))
    if dt<=0 or not np.allclose(np.diff(t),dt,atol=1e-9,rtol=1e-6):raise ValueError('Missing/nonuniform samples cannot form an event')
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return [dict(first_index=int(a),last_index=int(b-1),start_s=float(t[a]-(dt if interval_samples else 0)),end_s=float(t[b-1]),
        duration_s=float(t[b-1]-t[a]+(dt if interval_samples else 0))) for a,b in zip(edges[::2],edges[1::2])]


def q_matrix(q):
    q=np.asarray(q,dtype=float);norm=np.linalg.norm(q,axis=-1)
    if not np.isfinite(q).all() or np.any(abs(norm-1)>1e-3):raise ValueError('Invalid quaternion')
    w,x,y,z=np.moveaxis(q/norm[...,None],-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1).reshape(q.shape[:-1]+(3,3))


def audit_actor_observations(trace,reference,conditioning=None,mode=None):
    """Reconstruct all actor fields from PRE-action physical states and targets."""
    n=len(trace['actions']);index=np.arange(n)*8
    if trace['observations'].shape!=(n,51) or len(reference['joint_pos'])<n:raise ValueError('Invalid actor-input clock')
    if trace['joint_names'].astype(str).tolist()!=reference['joint_names'].astype(str).tolist():raise ValueError('Actor joint order mismatch')
    R=q_matrix(trace['root_quat'][index]);inv=np.swapaxes(R,-1,-2)
    target_R=q_matrix(reference['body_quat_w'][:n,0])
    rotate=lambda v:np.einsum('nij,nj->ni',inv,v)
    phase=np.arange(n)/(len(reference['joint_pos'])-1)*2*np.pi
    expected=np.column_stack((reference['joint_pos'][:n,:6],reference['joint_vel'][:n,:6],np.sin(phase),np.cos(phase),
        rotate(reference['body_pos_w'][:n,0]-trace['root_pos'][index]),
        np.einsum('nij,njk->nik',inv,target_R)[:,:,:2].reshape(n,6),
        rotate(trace['root_com_vel'][index]),rotate(trace['root_omega'][index]),
        trace['joint_pos'][index,:6]-reference['joint_pos'][0,:6],trace['joint_vel'][index],
        np.vstack((np.zeros((1,8)),trace['actions'][:-1]))))
    if mode is not None:
        if mode not in ('reference_dq','nominal_command') or conditioning is None or conditioning.shape!=(n,14):
            raise ValueError('Invalid conditioning intervention')
        sl=slice(6,12) if mode=='reference_dq' else slice(0,14)
        expected[:,sl]=conditioning[:,sl]
    error=abs(expected-trace['observations'])
    if not np.isfinite(error).all() or error.max()>5e-5:raise ValueError('Recorded actor inputs do not match physical states/reference intervention')
    return dict(control_samples_checked=n,maximum_error=float(error.max()),tolerance=5e-5,
        conditioning_mode=mode,all_51_input_columns_reconstructed=True,
        state_anchor_and_previous_action_fields_unmodified=True)


def heading_offsets(velocity,heading_deg,valid_heading=None):
    v=np.asarray(velocity);speed=np.linalg.norm(v[...,:2],axis=-1)
    valid=(speed>=.25)&np.isfinite(heading_deg)
    if valid_heading is not None:valid&=valid_heading
    directed=(np.degrees(np.arctan2(v[...,1],v[...,0]))-heading_deg+180)%360-180
    axial=(directed+90)%180-90
    return np.where(valid,directed,np.nan),np.where(valid,axial,np.nan),valid


def validate_trace(trace):
    dt=float(trace['physics_dt_s']);policy_dt=float(trace['policy_dt_s'])
    if not np.isclose(dt,.0025) or not np.isclose(policy_dt,.02):raise ValueError('This frozen protocol requires 400/50 Hz')
    n=len(trace['root_pos']);b=len(trace['body_names'])
    expected={'root_pos':(n,3),'root_quat':(n,4),'root_com_vel':(n,3),
        'body_com_pos':(n,b,3),'body_com_vel':(n,b,3),'body_omega':(n,b,3),'body_quat':(n,b,4),
        'normal_fz':(n,2),'lateral_slip_rms':(n,2),'utilization_proxy':(n,2),
        'friction_anchor_power':(n,2),'friction_force_w':(n,2,3)}
    for key,shape in expected.items():
        if trace[key].shape!=shape or not np.isfinite(trace[key]).all():raise ValueError('Invalid trace field: '+key)
    t=np.arange(n)*dt
    if 'time_s' in trace and not np.allclose(trace['time_s'],t,atol=1e-9,rtol=0):raise ValueError('Trace clock mismatch')
    q_matrix(trace['root_quat']);q_matrix(trace['body_quat'])
    return t


def reconstruct_contacts(trace,raw,mu,*,readback_mode='float64'):
    """Recompute exact force/velocity pairing at reported anchors; never infer Fn there."""
    t=validate_trace(trace);n=len(t)
    if not np.array_equal(raw['physical_sample_indices'],np.arange(1,n)):raise ValueError('Missing or reordered native contact samples')
    if not np.isfinite(mu) or mu<=0:raise ValueError('Invalid friction coefficient')
    names=trace['body_names'].astype(str).tolist()
    scalar=('normal_fz','normal_point_lateral_rms','anchor_power','anchor_lateral_power',
        'aggregate_utilization','anchor_force_sum_utilization','dissipative_slipping_anchor_count',
        'anchor_force_fraction_dissipative_lateral_slip','loaded_normal_points','friction_anchor_count')
    result={k:np.zeros((n,2)) for k in scalar};result['friction_force']=np.zeros((n,2,3))
    origin=trace['environment_origin_w'].astype(float)
    for side,name in enumerate(('wheel_L_Link','wheel_R_Link')):
        bid=names.index(name);axles=q_matrix(trace['body_quat'][:,bid])[...,:,1]
        for row,i in enumerate(raw['physical_sample_indices']):
            normal=[raw[f'w{side}_normal_buffer_{k}'][row] for k in range(6)]
            friction=[raw[f'w{side}_friction_buffer_{k}'][row] for k in range(4)]
            count,start=int(normal[4].item()),int(normal[5].item())
            ac,ast=int(friction[2].item()),int(friction[3].item())
            if min(count,start,ac,ast)<0 or start+count>len(normal[0]) or ast+ac>len(friction[0]):raise ValueError('Contact buffer range error')
            fn=normal[0][start:start+count].reshape(-1).astype(float)
            points=normal[1][start:start+count].astype(float)-origin
            normals=normal[2][start:start+count].astype(float)
            ft=friction[0][ast:ast+ac].astype(float);anchors=friction[1][ast:ast+ac].astype(float)-origin
            if not all(np.isfinite(x).all() for x in (fn,points,normals,ft,anchors)) or np.any(fn<-.0001):raise ValueError('Invalid native contact values')
            if count:
                length=np.linalg.norm(normals,axis=-1,keepdims=True)
                if np.any(length[fn>5]<1e-8):raise ValueError('Degenerate loaded normal')
                normals/=np.maximum(length,1e-12)
                if np.any(abs(normals[fn>5]-[0,0,1])>1e-4):raise ValueError('This protocol requires a stationary flat floor')
            com=trace['body_com_pos'][i,bid].astype(float);v=trace['body_com_vel'][i,bid].astype(float);w=trace['body_omega'][i,bid].astype(float)
            loaded=fn>5.;fz=float((fn[:,None]*normals).sum(0)[2]);slip=0.
            if loaded.any():
                lateral=axles[i]-(normals[loaded]@axles[i])[:,None]*normals[loaded]
                length=np.linalg.norm(lateral,axis=-1,keepdims=True)
                if np.any(length<1e-8):raise ValueError('Degenerate loaded wheel plane')
                lateral/=length;pv=v+np.cross(w,points[loaded]-com)
                lat=(pv*lateral).sum(-1);slip=float(np.sqrt(np.average(lat*lat,weights=fn[loaded])))
            av=v+np.cross(w,anchors-com);power=(ft*av).sum(-1)
            tangent_norm=np.linalg.norm(ft[:,:2],axis=-1)
            axle_xy=axles[i].copy();axle_xy[2]=0.;length=np.linalg.norm(axle_xy)
            if ac and fz>5 and length<1e-8:raise ValueError('Degenerate friction-anchor wheel plane')
            lateral=axle_xy/max(length,1e-8)
            anchor_lateral=av@lateral;lateral_force=ft@lateral
            same_anchor=(abs(anchor_lateral)>.05)&(power<0)&(lateral_force*anchor_lateral<0)
            total_force=ft.sum(0)
            values=dict(normal_fz=fz,normal_point_lateral_rms=slip,anchor_power=power.sum(),
                anchor_lateral_power=(lateral_force*anchor_lateral).sum(),
                aggregate_utilization=np.linalg.norm(total_force[:2])/(mu*fz) if fz>5 else 0.,
                anchor_force_sum_utilization=tangent_norm.sum()/(mu*fz) if fz>5 else 0.,
                dissipative_slipping_anchor_count=same_anchor.sum(),
                anchor_force_fraction_dissipative_lateral_slip=tangent_norm[same_anchor].sum()/tangent_norm.sum() if tangent_norm.sum()>1e-12 else 0.,
                loaded_normal_points=loaded.sum(),friction_anchor_count=ac)
            for key,value in values.items():result[key][i,side]=value
            result['friction_force'][i,side]=total_force
    if readback_mode not in ('float64','recorded_float32'):
        raise ValueError('Unknown native readback arithmetic')
    readback=result
    if readback_mode=='recorded_float32':
        from audit_physx_contact_float32 import replay_contact_readback
        readback=replay_contact_readback(trace,raw,mu)
    checks={};float64_differences={}
    for computed,recorded,tolerance in [('normal_fz','normal_fz',.001),('normal_point_lateral_rms','lateral_slip_rms',4e-5),
        ('anchor_power','friction_anchor_power',.005),('friction_force','friction_force_w',.001),('aggregate_utilization','utilization_proxy',1e-4)]:
        float64_differences[computed]=float(abs(result[computed]-trace[recorded]).max())
        err=float(abs(readback[computed]-trace[recorded]).max());checks[computed]=err
        if err>tolerance:raise ValueError(f'Native readback mismatch {computed}: {err}')
    result['native_sample_valid']=np.arange(n)>0
    return result,dict(physical_samples_checked=n-1,maximum_readback_errors=checks,
        readback_arithmetic=readback_mode,normalized_float64_differences=float64_differences,
        pointwise_normal_load_at_friction_anchors_available=False,
        force_velocity_timing='completed-substep force/impulse divided by dt, paired with post-step material velocity; power proxy, not exact step work')


def derive_signals(trace,contact):
    t=validate_trace(trace);names=trace['body_names'].astype(str).tolist()
    ids=[names.index(k) for k in ('wheel_L_Link','wheel_R_Link')]
    R=q_matrix(trace['root_quat']);forward=R[..., :,0]
    body_heading=np.degrees(np.arctan2(forward[:,1],forward[:,0]))
    body_beta,_,body_valid=heading_offsets(trace['root_com_vel'],body_heading,np.linalg.norm(forward[:,:2],axis=-1)>1e-6)
    axle=q_matrix(trace['body_quat'][:,ids])[...,:,1]
    heading=np.degrees(np.arctan2(-axle[...,0],axle[...,1]))
    directed,axial,valid=heading_offsets(trace['body_com_vel'][:,ids],heading,np.linalg.norm(axle[...,:2],axis=-1)>1e-6)
    return dict(time_s=t,body_heading_deg=body_heading,body_beta_deg=body_beta,body_beta_valid=body_valid,
        body_speed_m_s=np.linalg.norm(trace['root_com_vel'][:,:2],axis=-1),
        root_tilt_deg=np.degrees(np.arccos(np.clip(R[:,2,2],-1,1))),
        wheel_heading_deg=heading,wheel_directed_beta_deg=directed,wheel_axial_beta_deg=axial,
        wheel_beta_valid=valid,wheel_com_speed_m_s=np.linalg.norm(trace['body_com_vel'][:,ids,:2],axis=-1),**contact)


def imitation_metrics(trace,signals,reference,command):
    """Posture fidelity independent of root height and continuous wheel spin.

    These diagnostics do not relax or retrospectively alter the frozen stop
    gates. A stable car-like stance alone is not evidence of GMR imitation.
    """
    from scipy.spatial.transform import Rotation,Slerp
    t=signals['time_s'];rt=np.arange(len(reference['joint_pos']))/float(reference['fps'])
    inside=(t>=rt[0])&(t<=rt[-1]+1e-9)&signals['native_sample_valid']
    sample_t=np.clip(t,rt[0],rt[-1])
    leg_names=('abad_L_Joint','abad_R_Joint','hip_L_Joint','hip_R_Joint','knee_L_Joint','knee_R_Joint')
    ids=[trace['joint_names'].astype(str).tolist().index(n) for n in leg_names]
    ref_ids=[reference['joint_names'].astype(str).tolist().index(n) for n in leg_names]
    target_joint=np.column_stack([np.interp(sample_t,rt,reference['joint_pos'][:,i]) for i in ref_ids])
    joint_error=trace['joint_pos'][:,ids]-target_joint
    def interpolated_matrix(name):
        i=reference['body_names'].astype(str).tolist().index(name)
        q=reference['body_quat_w'][:,i]
        return Slerp(rt,Rotation.from_quat(q[:,[1,2,3,0]]))(sample_t).as_matrix()
    target_root=interpolated_matrix('base_Link')
    root=q_matrix(trace['root_quat'])
    up_error=np.degrees(np.arccos(np.clip((root[:,:,2]*target_root[:,:,2]).sum(-1),-1,1)))
    # Both robot wheels use the link's local +Y axle; spinning about that axle
    # has no effect on camber. The sign convention is recorded, not inferred.
    wheel_ids=[trace['body_names'].astype(str).tolist().index('wheel_'+s+'_Link') for s in ('L','R')]
    wheel_R=q_matrix(trace['body_quat'][:,wheel_ids])
    camber=np.degrees(np.arcsin(np.clip(wheel_R[...,2,1],-1,1)))
    target_camber=np.column_stack([np.degrees(np.arcsin(np.clip(interpolated_matrix('wheel_'+s+'_Link')[:,2,1],-1,1))) for s in ('L','R')])
    tilt_indices=np.flatnonzero(signals['root_tilt_deg']>=45)
    first=signals['time_s'][tilt_indices[0]] if len(tilt_indices) else np.inf
    windows={'whole_reference':inside,'before_first_45deg_tilt':inside&(t<first),
        'braking':inside&(t>=command['brake_start_s'])&(t<=command['brake_end_s']),
        'braking_before_first_45deg_tilt':inside&(t>=command['brake_start_s'])&(t<=command['brake_end_s'])&(t<first)}
    rows={}
    for name,mask in windows.items():
        row=dict(samples=int(mask.sum()),leg_joint_RMSE_rad=None,leg_joint_max_abs_error_rad=None,body_up_axis_RMSE_deg=None,loaded_wheel_camber_RMSE_deg=[None,None])
        if mask.any():
            row.update(leg_joint_RMSE_rad=float(np.sqrt(np.mean(joint_error[mask]**2))),
                leg_joint_max_abs_error_rad=float(abs(joint_error[mask]).max()),
                per_joint_RMSE_rad=np.sqrt(np.mean(joint_error[mask]**2,axis=0)).tolist(),
                body_up_axis_RMSE_deg=float(np.sqrt(np.mean(up_error[mask]**2))))
        for k in (0,1):
            loaded=mask&(signals['normal_fz'][:,k]>5)
            if loaded.any():row['loaded_wheel_camber_RMSE_deg'][k]=float(np.sqrt(np.mean((camber[loaded,k]-target_camber[loaded,k])**2)))
        rows[name]=row
    return dict(kind='GMR_posture_fidelity_diagnostics',leg_joint_names=list(leg_names),windows=rows,
        camber_definition='asin(world_z component of local +Y wheel axle), degrees',
        reference_orientation_interpolation='quaternion Slerp on original clock',
        reference_joint_interpolation='linear on original clock',
        absolute_height_error_used=False,wheel_spin_error_used=False,prospective_stop_gates_changed=False,
        posture_success_threshold_established=False),dict(time_s=t,leg_joint_error_rad=joint_error,
            body_up_axis_error_deg=up_error,wheel_camber_deg=camber,target_wheel_camber_deg=target_camber,
            native_sample_valid=signals['native_sample_valid'],normal_fz=signals['normal_fz'])


def qualify_stop(trace,signals,reference,command):
    t=signals['time_s'];dt=float(np.median(np.diff(t)))
    rt=np.arange(len(reference['body_pos_w']))/float(reference['fps'])
    end=rt[-1];brake_start=command['brake_start_s'];brake_end=command['brake_end_s']
    if not 0<brake_start<brake_end<end:raise ValueError('Invalid stop phases')
    valid=(t-dt>=.02-1e-9)&(t<=end+1e-9)&signals['native_sample_valid']
    if not valid.any():raise ValueError('No valid physical samples')
    interp=lambda x:np.column_stack([np.interp(t,rt,x[:,k]) for k in range(x.shape[-1])])
    xy=np.linalg.norm(trace['root_pos'][:,:2]-interp(reference['body_pos_w'][:,0,:2]),axis=-1)
    target_speed=np.linalg.norm(interp(reference['body_lin_vel_w'][:,0,:2]),axis=-1)
    ref_R=q_matrix(reference['body_quat_w'])
    ref_yaw=np.arctan2(ref_R[:,0,1,0],ref_R[:,0,0,0])
    target_yaw=np.degrees(np.interp(t,rt,np.unwrap(ref_yaw)))
    heading_error=(signals['body_heading_deg']-target_yaw+180)%360-180
    ref_ids=[reference['body_names'].astype(str).tolist().index(k) for k in ('wheel_L_Link','wheel_R_Link')]
    ref_axle=ref_R[:,ref_ids][...,:,1]
    if ref_axle.shape!=(len(rt),2,3):raise ValueError('Reference wheel-axis shape mismatch')
    ref_heading=np.arctan2(-ref_axle[...,0],ref_axle[...,1])
    target_heading=np.degrees(interp(np.unwrap(2*ref_heading,axis=0))/2)
    wheel_heading_error=(signals['wheel_heading_deg']-target_heading+90)%180-90
    speed=signals['body_speed_m_s'];support=signals['normal_fz']>5.;any_support=support.any(-1)
    approach=valid&(t<brake_start-1e-9);entry=approach&(t>=brake_start-.1-1e-9)
    braking=valid&(t>=brake_start-1e-9)&(t<=brake_end+1e-9)
    terminal=valid&(t>=end-.5-1e-9)
    body_heading_rmse=float(np.sqrt(np.mean(heading_error[braking]**2))) if braking.any() else None
    wheel_heading_mask=braking[:,None]&support&signals['wheel_beta_valid']
    wheel_heading_rmse=float(np.sqrt(np.mean(wheel_heading_error[wheel_heading_mask]**2))) if wheel_heading_mask.any() else None
    entry_speed=float(speed[entry].mean()) if entry.any() else 0.
    ref_mean=float(target_speed[approach].mean()) if approach.any() else 0.
    ratio=float(speed[approach].mean()/ref_mean) if ref_mean>1e-8 else None
    stop=valid&(t>=brake_start)&(speed<=.1)&(signals['root_tilt_deg']<=30)&any_support
    stop_events=[x for x in segments(t,stop) if x['duration_s']>=.5-1e-9]
    terminal_events=segments(t,stop&terminal)
    terminal_duration=max((x['duration_s'] for x in terminal_events),default=0.)
    corroborated=support&(signals['normal_point_lateral_rms']>.05)&(signals['aggregate_utilization']>=.9)&(signals['aggregate_utilization']<=1.05)
    corroborated&=(signals['anchor_power']<-.05)&(signals['dissipative_slipping_anchor_count']>=1)
    corroborated&=signals['wheel_beta_valid']&(abs(signals['wheel_axial_beta_deg'])>=20.)
    full_braking_intervals=braking&(t-dt>=brake_start-1e-9)
    corroborated&=full_braking_intervals[:,None]&(signals['root_tilt_deg'][:,None]<45.)
    events=[segments(t,corroborated[:,k],interval_samples=True) for k in (0,1)]
    longest=[max((x['duration_s'] for x in ev),default=0.) for ev in events]
    gates=dict(full_reference_observed=bool(t[-1]>=end-1e-9),entry_speed_at_least_0p8=entry_speed>=.8,
        approach_speed_retained=ratio is not None and .95<=ratio<=1.1,
        xy_rmse_le_0p15=float(np.sqrt(np.mean(xy[valid]**2)))<=.15,xy_peak_le_0p3=float(xy[valid].max())<=.3,
        anywheel_continuously_supported=bool(any_support[valid].all()),
        upright_supported_terminal_hold_ge_0p5=terminal_duration>=.5-1e-9,
        loaded_lateral_sliding_during_braking_ge_0p05=max(longest)>=.05-1e-9,
        braking_body_heading_RMSE_le_15deg=body_heading_rmse is not None and body_heading_rmse<=15.,
        loaded_braking_wheel_heading_RMSE_le_20deg=wheel_heading_rmse is not None and wheel_heading_rmse<=20.,
        tilt_below_45deg_during_task=bool(np.all(signals['root_tilt_deg'][valid]<45.)))
    result=dict(gates=gates,task_success_claimed=all(gates.values()),failed_gates=[k for k,v in gates.items() if not v],
        entry_mean_speed_last_0p1s_m_s=entry_speed,approach_COM_speed_ratio=ratio,
        braking_body_heading_RMSE_deg=body_heading_rmse,loaded_braking_wheel_heading_RMSE_deg=wheel_heading_rmse,
        braking_COM_speed_RMSE_m_s=float(np.sqrt(np.mean((speed[braking]-target_speed[braking])**2))) if braking.any() else None,
        xy_RMSE_m=float(np.sqrt(np.mean(xy[valid]**2))),xy_peak_m=float(xy[valid].max()),
        support_fraction=float(any_support[valid].mean()),longest_support_gap_s=max((x['duration_s'] for x in segments(t,valid&~any_support,interval_samples=True)),default=0.),
        terminal_stop_hold_s=terminal_duration,stop_events=stop_events,
        bilateral_loaded_slip_events=events,bilateral_loaded_slip_longest_s=longest,
        wheel_beta_gate='axial abs(beta)>=20 degrees and speed>=.25 m/s; prospective hockey distinction from reverse rolling',
        aggregate_utilization_screen_range=[.9,1.05],
        absolute_height_gate=False,pointwise_cone_saturation_certified=False)
    if stop_events:
        i=stop_events[0]['first_index'];start=np.array([np.interp(brake_start,t,trace['root_pos'][:,k]) for k in (0,1)])
        p=trace['root_pos'][i,:2];v=np.array([np.interp(brake_start,t,trace['root_com_vel'][:,k]) for k in (0,1)])
        norm=np.linalg.norm(v)
        path=trace['root_pos'][(t>brake_start)&(t<=t[i]),:2]
        result['stopping']=dict(time_s=float(t[i]),duration_since_brake_s=float(t[i]-brake_start),
            displacement_m=float(np.linalg.norm(p-start)),path_length_m=float(np.linalg.norm(np.diff(np.vstack((start,path)),axis=0),axis=-1).sum()),
            forward_distance_m=float((p-start)@(v/norm)) if norm>1e-8 else None,
            metric_scope='simulator base-link path; not a calibrated human stopping distance')
    return result


def render_signals(signals,path,mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=signals;t=s['time_s'];colors=('#d85b40','#3978bf')
    fig,ax=plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
    loaded=s['normal_fz']>5
    ax[0].plot(t,s['body_beta_deg'],color='black',lw=.9,label='Body COM beta')
    for k,c in enumerate(colors):
        label=('L','R')[k]
        ax[0].plot(t,np.where(loaded[:,k],s['wheel_axial_beta_deg'][:,k],np.nan),color=c,lw=.8,label=label+' loaded wheel axial beta')
        ax[1].plot(t,np.where(loaded[:,k],s['normal_point_lateral_rms'][:,k],np.nan),color=c,lw=.8,label=label)
        ax[2].plot(t,np.where(loaded[:,k],s['aggregate_utilization'][:,k],np.nan),color=c,lw=.8,label=label)
        ax[3].plot(t,s['normal_fz'][:,k],color=c,lw=.8,label=label+' normal load')
    ax[0].set(ylabel='Degrees',title='Body / bilateral velocity-heading offsets');ax[0].legend(fontsize=7)
    ax[1].set(ylabel='m/s',title='Loaded normal-point lateral slip RMS');ax[1].axhline(.05,color='black',ls='--',lw=.8)
    ax[2].set(ylabel='Aggregate Ft / (mu Fz)',title='Wheel utilization proxy; not pointwise cone');ax[2].axhline(.9,color='black',ls='--',lw=.8)
    ax[3].set(ylabel='N',title='Actual wheel normal load',yscale='symlog');ax[3].axhline(5,color='black',ls='--',lw=.8);ax[3].legend(fontsize=7)
    tilted=np.flatnonzero(s['root_tilt_deg']>=45)
    for a in ax:
        a.set(xlabel='Time [s]',xlim=(t[0],t[-1]));a.grid(alpha=.2)
        if len(tilted):a.axvline(t[tilted[0]],color='#777777',ls=':',lw=1)
    fig.suptitle('Historical car instrumentation preflight; NOT a hockey-stop experiment' if mode.startswith('instrumentation') else 'Hockey-stop native PhysX diagnostics; inspect qualification gates separately')
    fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--command-dir',type=Path)
    p.add_argument('--execution-evidence',type=Path,help='Collected v2 launch/payload audit when Kit shutdown skipped post-run annotations')
    p.add_argument('--conditioning-table',type=Path,help='Required for the approved input-intervention probes')
    p.add_argument('--search-source',type=Path,help='Reviewed native hockey MPPI snapshot for single-world plan validation')
    p.add_argument('--action-replay',type=Path,help='CPU control-replay report required for native MPPI qualification')
    p.add_argument('--instrumentation-only',action='store_true')
    a=p.parse_args()
    if (a.command_dir is None)!=a.instrumentation_only:p.error('Choose a hockey command directory or instrumentation-only')
    if a.action_replay is not None and a.search_source is None:p.error('--action-replay requires --search-source')
    if a.output.exists():raise FileExistsError(a.output)
    run=json.loads((a.run_dir/'report.json').read_text());files=('final_rollout.npz','native_contact_buffers.npz')
    for name in files:
        if sha(a.run_dir/name)!=run['artifacts'][name]:raise ValueError('Archived artifact changed: '+name)
    material=run['material_audit']
    if not (material['live_physx_materials_verified'] and material['ground_material_usd_verified']):raise ValueError('Unaudited materials')
    low,high=material['wheel_mu_range']
    if abs(low-high)>1e-7 or any(g['friction']!=1 or g['combine']!='multiply' for g in material['ground_colliders']):raise ValueError('Expected fixed equal-wheel mu and unit multiply ground')
    trace=load(a.run_dir/files[0]);raw=load(a.run_dir/files[1])
    contact,audit=reconstruct_contacts(trace,raw,float(low));signals=derive_signals(trace,contact)
    report=dict(kind='hockey_stop_independent_physical_evaluator_v1',mode='instrumentation_preflight_on_historical_car' if a.instrumentation_only else 'hockey_stop',
        analyzer_sha256=sha(__file__),run_report_sha256=sha(a.run_dir/'report.json'),
        input_sha256={name:sha(a.run_dir/name) for name in files},mu=float(low),native_reconstruction=audit,
        friction_indicator='wheel aggregate force utilization; exact force/velocity pairing at friction anchors; no per-anchor Fn',
        forces_from_native_buffers=True,physics_rerun=False,new_hockey_physics_run=False,
        physical_step_timing_warning='instantaneous post-state force-power proxy; not exact substep work',
        pointwise_cone_saturation_certified=False)
    if a.command_dir:
        reference=load(a.command_dir/'motion.npz');command=json.loads((a.command_dir/'report.json').read_text())
        if sha(a.command_dir/'motion.npz')!=run['motion_sha256'] or sha(a.command_dir/'motion.npz')!=command['motion_sha256']:raise ValueError('Hockey command/run mismatch')
        wrapper_sha=run.get('hockey_wrapper_sha256')
        if a.execution_evidence:
            report['execution_evidence']=verify_execution_evidence(a.execution_evidence,a.run_dir)
            wrapper_sha=report['execution_evidence']['wrapper_sha256']
        base_wrapper='dd29fe1ddd736f0faf8fd78e6290020374346e24753bb6259dac6e59cc97bf2e'
        conditioning_wrapper='51447598d9c840de94df203dfa21679c22f99a8d5d5b2876de7d9c671ab9d266'
        if a.search_source:
            if a.execution_evidence:raise ValueError('Search validation cannot use historical probe identity')
            report['search_source_audit']=verify_hockey_search_source(a.search_source,run)
            if a.action_replay is None:raise ValueError('Native MPPI qualification requires a separate control replay')
            report['control_replay_audit']=verify_control_replay(a.action_replay,a.run_dir,run,report['search_source_audit'])
        elif (run['source_sha256']!='3aacb6b4f3b62d32ce6fa2326f2cb6bfe372b693b53c9830a27a85f26f3f7763'
                or wrapper_sha not in (base_wrapper,conditioning_wrapper)
                or not run['single_world_validation'] or run['worlds']!=1 or run['external_forces']!=0):
            raise ValueError('Hockey probe is not the frozen single-world native engine')
        if len(trace['root_pos'])!=8*len(trace['actions'])+1:raise ValueError('Physical/action clock mismatch')
        mode=None;conditioning=None
        if a.search_source:
            if a.conditioning_table is None or sha(a.conditioning_table)!=run['conditioning_table_sha256']:
                raise ValueError('Unverified search conditioning table')
            mode=run['conditioning_mode'];conditioning=load(a.conditioning_table)['command'][:len(trace['actions'])]
            report['conditioning_table_sha256']=sha(a.conditioning_table)
        elif wrapper_sha==conditioning_wrapper:
            if (a.conditioning_table is None or sha(a.conditioning_table)!=run['conditioning_table_sha256']
                    or run['conditioning_adapter_sha256']!='47bf422eb4acc93cb34cea52297e6648f24c881368f1160f6cbf596733da85df'
                    or not run['actual_actor_input_archived'] or run['policy_weights_changed'] or run['MPPI_search_run']):
                raise ValueError('Unverified reference-conditioning intervention')
            mode=run['conditioning_mode'];conditioning=load(a.conditioning_table)['command']
            report['conditioning_table_sha256']=sha(a.conditioning_table)
        elif a.conditioning_table is not None:raise ValueError('Unexpected conditioning table for unadapted probe')
        report['actor_observation_audit']=audit_actor_observations(trace,reference,conditioning,mode)
        report['new_hockey_physics_run']=True
        report['frozen_run_inline_success_flag_superseded']=True
        report['qualification']=qualify_stop(trace,signals,reference,command)
        report['imitation'],_=imitation_metrics(trace,signals,reference,command)
    else:report['task_success_claimed']=False
    report['loaded_beta_summary']=[]
    tilted=np.flatnonzero(signals['root_tilt_deg']>=45)
    until=int(tilted[0]) if len(tilted) else len(signals['time_s'])
    report['first_45deg_tilt_diagnostic_s']=float(signals['time_s'][until]) if len(tilted) else None
    for k in (0,1):
        mask=(signals['normal_fz'][:,k]>5)&signals['wheel_beta_valid'][:,k]&signals['native_sample_valid']
        before=mask&(np.arange(len(mask))<until)
        corroboration=before&(signals['normal_point_lateral_rms'][:,k]>.05)&(signals['aggregate_utilization'][:,k]>=.9)&(signals['aggregate_utilization'][:,k]<=1.05)
        corroboration&=(signals['anchor_power'][:,k]<-.05)&(signals['dissipative_slipping_anchor_count'][:,k]>=1)
        report['loaded_beta_summary'].append(dict(side=('left','right')[k],samples=int(mask.sum()),
            abs_axial_beta_deg_p50_p95_max=np.quantile(abs(signals['wheel_axial_beta_deg'][mask,k]),[.5,.95,1]).tolist() if mask.any() else None,
            before_first_45deg_tilt_samples=int(before.sum()),
            loaded_aggregate_ratio_above_1p05_samples=int((mask&(signals['aggregate_utilization'][:,k]>1.05)).sum()),
            before_first_45deg_tilt_abs_axial_beta_deg_p50_p95_max=np.quantile(abs(signals['wheel_axial_beta_deg'][before,k]),[.5,.95,1]).tolist() if before.any() else None,
            before_first_45deg_tilt_same_anchor_corroboration_events=segments(signals['time_s'],corroboration,interval_samples=True)))
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'signals.npz',**signals)
    render_signals(signals,a.output/'contact_beta_4x1.png',report['mode'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Audit source/retarget dynamic scaling and tire arithmetic; no physics rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from nonlinear_tire_contact import tangent_wrench

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MOTION=Path('/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    upstream=ROOT/'upstream/tron1-learning-from-human'
    sys.path.insert(0,str(upstream/'scripts'))
    import prepare_drift_drl_turn_reference as decoder
    cfg_path=upstream/'config/drift_turn_source.json';cfg=json.loads(cfg_path.read_text())
    csv_path=ROOT/'upstream/drift_drl/code/ref_trajectory/traj_4.csv'
    assert sha(csv_path)==cfg['sha256']=='eb67a507817605d9b0018bdf2196c04f58f9a9238e7e1906350a7568ab87cf27'
    assert sha(a.motion)=='da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254'
    source=decoder.decode_csv(csv_path.read_bytes())
    ids=source['unique_pose_indices'];ids=ids[(ids>=cfg['drift_row_start_inclusive'])&(ids<=cfg['drift_row_end_inclusive'])]
    t=source['time_s_estimated'][ids];velocity=source['velocity_xy_m_s'][ids]
    speed=np.linalg.norm(velocity,axis=-1);course=np.unwrap(np.arctan2(velocity[:,1],velocity[:,0]))
    lateral=speed*np.gradient(course,t);demand=np.abs(lateral)/9.81
    space,stretch=cfg['spatial_scale'],cfg['time_scale'];acceleration_factor=space/stretch**2
    with np.load(a.motion,allow_pickle=False) as z:
        names=[str(x) for x in z['body_names']];base=names.index('base_Link')
        xy=z['body_pos_w'][:,base,:2].copy();v=z['body_lin_vel_w'][:,base,:2].copy()
        rt=np.arange(len(xy))/np.asarray(z['fps']).item()
    source_ids=source['unique_pose_indices'];source_ids=source_ids[(source_ids>=cfg['source_row_start_inclusive'])&(source_ids<=cfg['source_row_end_inclusive'])]
    origin=source['position_xy_m'][cfg['source_row_start_inclusive']];yaw=source['yaw_rad'][cfg['source_row_start_inclusive']]
    rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    scaled_xy=space*(source['position_xy_m'][source_ids]-origin)@rotation.T
    a.output_dir.mkdir(parents=True)
    analysis={'kind':'source_and_reference_kinematics_plus_contact_law_arithmetic_not_PhysX_task_result',
        'analyzer_source_sha256':sha(__file__),
        'source_csv_sha256':sha(csv_path),'source_config_sha256':sha(cfg_path),'motion_sha256':sha(a.motion),
        'source_decode_code_sha256':sha(decoder.__file__),'source_clock':'reconstructed estimate, not recorded timestamps',
        'source_is_real_car_capture':False,'core_source_rows':[int(ids[0]),int(ids[-1])],
        'estimated_core_duration_s':float(t[-1]-t[0]),'source_core_speed_range_m_s':[float(speed.min()),float(speed.max())],
        'source_core_beta_range_deg':np.rad2deg([source['beta_rad'][ids].min(),source['beta_rad'][ids].max()]).tolist(),
        'scale':{'space':space,'time':stretch,'speed':space/stretch,'acceleration':acceleration_factor},
        'kinematic_flat_floor_lateral_acceleration_over_g_quantiles_50_95_100':np.quantile(demand,[.5,.95,1.]).tolist(),
        'uniformly_scaled_lateral_acceleration_over_g_quantiles_50_95_100':(acceleration_factor*np.quantile(demand,[.5,.95,1.])).tolist(),
        'load_force_or_tire_mu_identified':False,
        'interpretation':'Scaling preserves path shape and sideslip angle but changes acceleration/gravity and friction utilization. Derivatives use an estimated clock; these numbers are kinematic demand diagnostics, not measured tire forces.',
        'source_mu_caution':'The source code default test tire coefficient3.5 is not proof of traj_4 capture friction. Current robot mu0.7 must not be described as equal dynamic conditions.',
        'paper_review':{'source':'https://arxiv.org/html/2607.10815v1','geometry':'9 cylinder slices','platform':'humanoid with passive skate wheels',
        'motion_prior':'GMR retargeting followed by AMP style reward plus task rewards','drift_task_evidence':'No reported high-speed large-beta turn comparison establishes transfer to this TRON1 task.',
        'assessment':'Potentially useful for posture/load-transfer/support timing; cannot infer tire parameters or drift feasibility from motion imitation alone.',
        'necessary_test':'Matched physics/task/budget: task-only PPO versus phase-aligned imitation versus AMP. Preserve speed/nonzero-beta/XY criteria and measure per-wheel contact slip.',
        'code_and_motion_data_release_located':False,'website_videos_redistributed':False},
        'physics_cases_completed_by_this_script':0,'human_prior_benefit_verified':False,'task_success_claimed':False}
    (a.output_dir/'analysis.json').write_text(json.dumps(analysis,indent=2)+chr(10))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4.8),dpi=150)
    ax=axes[0];ax.plot(scaled_xy[:,0],scaled_xy[:,1],label='Source car XY x0.08',color='gray')
    moving=rt<=4.96;ax.plot(xy[moving,0],xy[moving,1],label='TRON1 reference',color='tab:blue',ls='--')
    ax.set(title='Reference paths in task coordinates',xlabel='X (m)',ylabel='Y (m)',aspect='equal');ax.legend(fontsize=8)
    ax=axes[1];ax.plot(t-t[0],speed,label='Original car, estimated clock',color='gray')
    ax.plot(t-t[0],speed*space/stretch,label='Uniformly scaled intent',color='tab:blue')
    ax.set(title='Core speed: native units',xlabel='Original estimated time (s)',ylabel='Speed (m/s)');ax.legend(fontsize=8)
    ax=axes[2];positions=np.arange(3);quantiles=np.quantile(demand,[.5,.95,1.])
    ax.bar(positions-.18,quantiles,width=.36,color='gray',label='Original car')
    ax.bar(positions+.18,quantiles*acceleration_factor,width=.36,color='tab:blue',label='Scaled intent')
    ax.axhline(.7,color='tab:red',ls='--',label='Robot nominal mu=.7')
    ax.set_xticks(positions,['median','p95','max']);ax.set(title='Lateral acceleration / g',ylabel='Kinematic demand (not measured mu)');ax.legend(fontsize=8)
    ax=axes[3];angles=np.linspace(0,60,301);vy=np.tan(np.deg2rad(angles));velocity=np.column_stack((np.ones(301),vy,np.zeros(301)))
    for law,label in [('external_coulomb_regularized','External regularized Coulomb'),('combined_slip_magic_formula','Combined-slip Magic Formula')]:
        value=tangent_wrench(velocity,[0,1/.127,0],[0,0,-.127],[0,1,0],[0,0,1],100.,law=law)
        ax.plot(angles,-value['force_w_n'][:,1]/70.,label=label)
    ax.set(title='Force-law arithmetic, NOT a rollout',xlabel='Wheel hub side angle (deg)',ylabel='Lateral force / (mu Fn)',ylim=(0,1.1));ax.legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('PhysX study preflight: source scaling and tire-law hypotheses | no car/TRON1 physics success claimed')
    fig.tight_layout();fig.savefig(a.output_dir/'scaling_and_tire_laws.png');plt.close(fig)
    np.savez_compressed(a.output_dir/'reference_context.npz',source_core_time_s_estimated=t,
        source_core_velocity_xy_m_s=source['velocity_xy_m_s'][ids],
        source_core_lateral_acceleration_m_s2=lateral,source_scaled_xy_m=scaled_xy,robot_reference_xy_m=xy,robot_reference_time_s=rt)
    print(json.dumps({'output':str(a.output_dir),'source_speed_range_m_s':analysis['source_core_speed_range_m_s'],
        'acceleration_scale':acceleration_factor,'physics_task_result':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--motion',type=Path,default=DEFAULT_MOTION)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())

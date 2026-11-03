"""Equal-forward-distance reference preview; retain archived PhysX evidence.

This produces no policy, physics rollout, inverse kinematics or success claim
for the direct-brake arm. Frozen initial posture plus the same translation is
a proposed command, not a prediction of stability under braking.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from evaluate_hockey_physx import load, q_matrix, sha
from audit_hockey_contact_cone_relation import read_case
from analyze_ppo_contact_timestep import band_spectrum

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2'
SELECTION = ROOT / 'results/hockey_static_dynamic_transition_20261029_v1/fullbody_selection.json'
GEOMETRY = ROOT / 'results/hockey_stop_evaluation_audit_20261026_v1/corrected_command_geometry/report.json'
MOTION_SHA = 'e23fbd82639d243f0c485cf391b1eff0e275d31f6bb6ba940647874a0ac83050'


def direct_reference(motion):
    """Same root XY at EVERY time; rigid initial posture, no wheel-spin target."""
    result = {k: np.array(v, copy=True) for k, v in motion.items()}
    n = len(motion['joint_pos'])
    t = np.arange(n) / float(motion['fps'])
    base = motion['body_names'].astype(str).tolist().index('base_Link')
    root = motion['body_pos_w'][:, base].copy()
    root[:, 2] = root[0, 2]
    result['body_pos_w'] = motion['body_pos_w'][:1] + (root-root[:1])[:, None]
    result['body_pos_w'][:, base] = root
    result['body_quat_w'] = np.repeat(motion['body_quat_w'][:1], n, axis=0)
    result['joint_pos'] = np.repeat(motion['joint_pos'][:1], n, axis=0)
    result['joint_vel'] = np.zeros_like(result['joint_pos'])
    result['body_ang_vel_w'] = np.zeros_like(motion['body_ang_vel_w'])
    result['body_lin_vel_w'] = np.repeat(np.gradient(root, t, axis=0)[:, None], len(motion['body_names']), axis=1)
    return result


def reference_geometry(motion, geometry):
    names = motion['body_names'].astype(str).tolist()
    centers, axles, gaps = [], [], []
    for side in ('L', 'R'):
        row = next(g for g in geometry if g['side'] == side)
        body = names.index('wheel_' + side + '_Link')
        rotation = q_matrix(motion['body_quat_w'][:, body])
        center = motion['body_pos_w'][:, body] + rotation @ np.asarray(row['USD_center_in_link_m'])
        axle = rotation @ np.asarray(row['USD_rotation_cylinder_to_link'])[:, 2]
        az = np.clip(np.abs(axle[:, 2]), 0, 1)
        gap = center[:, 2] - row['USD_radius_m'] * np.sqrt(1 - az**2) - row['USD_width_m'] / 2 * az
        centers.append(center); axles.append(axle); gaps.append(gap)
    return np.stack(centers, 1), np.stack(axles, 1), np.stack(gaps, 1)


def forward_budget(t, position, brake_start, brake_end):
    position = np.asarray(position)
    direction = position[1, :2] - position[0, :2]
    if np.linalg.norm(direction) < 1e-10:
        raise ValueError('Initial forward direction is undefined')
    direction /= np.linalg.norm(direction)
    longitudinal = (position[:, :2] - position[0, :2]) @ direction
    start = float(np.interp(brake_start, t, longitudinal))
    end = float(np.interp(brake_end, t, longitudinal))
    return dict(direction_xy=direction.tolist(), approach_distance_m=start,
                braking_distance_m=end-start, total_distance_m=end)


def persistent_stop(t, speed, eligible, start=.4, threshold=.1, hold=.5):
    """Native samples only: retain rebound, end censoring, and missing support."""
    t, speed, eligible = np.asarray(t), np.asarray(speed), np.asarray(eligible)
    if t.ndim != 1 or speed.shape != t.shape or eligible.shape != t.shape or eligible.dtype.kind != 'b':
        raise ValueError('Expected matching time, speed, and boolean eligibility')
    if len(t)<2 or np.any(np.diff(t)<=0) or not np.isfinite(t).all() or not np.isfinite(speed).all():
        raise ValueError('Finite increasing native time required')
    low = (t >= start-1e-9) & (speed <= threshold) & eligible
    edges = np.flatnonzero(np.diff(np.r_[False, low, False]))
    for first, end in zip(edges[::2], edges[1::2]):
        if t[end-1]-t[first] >= hold-1e-9:
            return dict(index=int(first), time_s=float(t[first]),
                        since_brake_s=float(t[first]-start), observed_hold_s=float(t[end-1]-t[first]))
    return None


def archived_cases(reference, budget):
    selection = json.loads(SELECTION.read_text())
    cases, trajectories, initial_state = [], {}, None
    direction = np.asarray(budget['direction_xy']); normal = np.array([-direction[1], direction[0]])
    for i in (0, 3):
        source = dict(selection[i]); source['label'] = source['caption']
        prior, signal = read_case(source, SELECTION, REFERENCE)
        directory = (SELECTION.parent / source['evidence'] / source['run_name'] / source['case'].replace(':','/')).resolve()
        trace = load(directory / 'final_rollout.npz')
        n = len(signal['time_start_s'])+1; t = np.arange(n)*float(trace['physics_dt_s'])
        p = trace['root_pos'][:n]; v = trace['root_com_vel'][:n]
        speed = np.linalg.norm(v[:, :2], axis=1)
        tilt = np.degrees(np.arccos(np.clip(1-2*np.sum(trace['root_quat'][:n, 1:3]**2, axis=1),-1,1)))
        touching = signal['touching_states'].all(axis=1)
        stop = persistent_stop(t, speed, touching & (tilt < 45))
        x = (p[:, :2]-p[0,:2]) @ direction
        y = (p[:, :2]-p[0,:2]) @ normal
        mask = (t>=.4-1e-9)&(t<=1.8+1e-9)
        fullbase = trace['body_names'].astype(str).tolist().index('base_Link')
        cm = trace['body_com_pos'][:n, fullbase]
        measured = None
        if stop is not None:
            k = stop['index']
            measured = dict(**stop, forward_base_origin_from_start_m=float(x[k]),
                forward_base_origin_since_brake_m=float(x[k]-np.interp(.4,t,x)),
                forward_base_COM_from_start_m=float((cm[k,:2]-cm[0,:2])@direction),
                maximum_lateral_base_origin_excursion_m=float(np.abs(y[:k+1]).max()))
        case = dict(id='v8_equal' if i==0 else 'v9_split', source_directory=str(directory.relative_to(ROOT)),
            material=[prior['mu_static'], prior['mu_dynamic']], actor_sha256=json.loads((directory/'report.json').read_text())['actor_sha256'],
            initial_base_COM_planar_speed_m_s=float(speed[0]), brake_entry_base_COM_planar_speed_m_s=float(np.interp(.4,t,speed)),
            supported_upright_speed_stop=measured, terminal_base_origin_distance_m=float(x[-1]),
            maximum_forward_base_origin_m=float(x.max()), overshoot_above_reference_endpoint_m=float(max(0,x.max()-budget['total_distance_m'])),
            terminal_planar_speed_m_s=float(speed[-1]), max_tilt_deg=float(tilt.max()),
            bilateral_native_contact_fraction=float(touching[:-1].mean()),
            bilateral_interruption_s=prior['bilateral_interruption_ms']/1000,
            core_angular_xy_band=band_spectrum(trace['root_omega'][:n][mask,:2],.0025),
            original_qualification_task_success=prior['task_success'],
            source_artifacts_sha256=prior['raw_sha256'], paired_direct_brake_policy_available=False,
            not_a_matched_distance_new_experiment=True)
        cases.append(case)
        key=case['id']; trajectories[key+'_time_s']=t;trajectories[key+'_forward_m']=x;trajectories[key+'_speed_m_s']=speed
        if initial_state is None:
            fields=('root_pos','root_quat','root_com_vel','root_omega','joint_pos','joint_vel',
                    'body_pos','body_com_pos','body_quat','body_com_vel','body_omega')
            initial_state={name:trace[name][0].copy() for name in fields}
            initial_state.update(joint_names=trace['joint_names'],body_names=trace['body_names'])
    return cases, trajectories, initial_state


def render(output, t, motions, geometries, budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    colors=('#d34c42','#704db7')
    fig,axs=plt.subplots(1,4,figsize=(20,5))
    direction=np.asarray(budget['direction_xy']); lateral=np.array([-direction[1],direction[0]])
    basis=np.stack((direction,lateral),axis=1)
    root0=motions[0]['body_pos_w'][0,0,:2]
    dynamic=[]
    for panel,(motion,(center,axle,gap)) in enumerate(zip(motions,geometries)):
        ax=axs[panel]; p=(motion['body_pos_w'][:,0,:2]-root0)@basis
        ax.plot(p[:,0],p[:,1],'--',color='#999',lw=1)
        ax.axvline(budget['total_distance_m'],color='#333',ls=':',label='Shared stop line')
        ax.set(xlim=(-.25,budget['total_distance_m']+.3),ylim=(-.6,.6),xlabel='Initial forward direction (m)',ylabel='Lateral (m)',title=('Hockey reference','Direct-brake reference')[panel])
        ax.set_aspect('equal'); ax.grid(alpha=.2)
        path,=ax.plot([],[],color='#23845f',lw=2)
        body,=ax.plot([],[],color='#2465a6',lw=3,label='Body heading')
        wheels=[ax.plot([],[],color=col,lw=4,label=side+' wheel plane')[0] for col,side in zip(colors,('L','R'))]
        heads=q_matrix(motion['body_quat_w'][:,0])[...,:2,0]@basis
        roll=np.cross(axle,np.array([0.,0.,1.]))[:,:,:2]@basis
        roll/=np.maximum(np.linalg.norm(roll,axis=2,keepdims=True),1e-12)
        cp=(center[:,:,:2]-root0)@basis
        dynamic.append((p,heads,cp,roll,path,body,wheels))
        for ti in (.4,1.2,1.8):
            k=int(round(ti*50));a=p[k];h=.18*heads[k]
            ax.plot([a[0],a[0]+h[0]],[a[1],a[1]+h[1]],color='#2465a6',lw=1)
            for w,c in enumerate(colors):
                ends=cp[k,w]+np.array([[-1.],[1.]])*.127*roll[k,w]
                ax.plot(ends[:,0],ends[:,1],color=c,lw=1)
    for i,(motion,(_,_,gap)) in enumerate(zip(motions,geometries)):
        speed=np.linalg.norm(np.gradient(motion['body_pos_w'][:,0,:2],t,axis=0),axis=1)
        axs[2].plot(t,speed,ls='-' if i==0 else '--',lw=3 if i==0 else 1.5,label=('Hockey','Direct')[i])
        for side,color in enumerate(colors):
            axs[3].plot(t,gap[:,side]*1000,color=color,ls='-' if i==0 else '--',label=f'{("Hockey","Direct")[i]} {("L","R")[side]}')
    axs[2].set(xlabel='Time (s)',ylabel='Base-origin speed (m/s)',title='Identical translation and timing')
    axs[3].set(xlabel='Time (s)',ylabel='Authored cylinder gap (mm)',title='Reference geometry; not force contact')
    cursors=[]
    for ax in axs[2:]:
        ax.axvspan(.4,1.8,alpha=.07,color='#23845f');ax.grid(alpha=.2);ax.legend(fontsize=8)
        cursors.append(ax.axvline(0,color='#222',lw=.8))
    axs[3].axhline(.1,color='#444',ls=':',lw=.6)
    title=fig.suptitle('Equal forward distance | same root XY and clock | kinematics only; no direct-brake policy yet')
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(output/'reference_comparison_4x1.png',dpi=130)
    def update(k):
        for p,h,cp,roll,path,body,wheels in dynamic:
            path.set_data(p[:k+1,0],p[:k+1,1]);b=p[k]+np.stack((np.zeros(2),.18*h[k]))
            body.set_data(b[:,0],b[:,1])
            for side,line in enumerate(wheels):
                ends=cp[k,side]+np.array([[-1.],[1.]])*.127*roll[k,side]
                line.set_data(ends[:,0],ends[:,1])
        for cursor in cursors:cursor.set_xdata([t[k],t[k]])
        title.set_text(f'Equal-distance reference preview | t={t[k]:.2f} s | kinematics only, no physical stability claim')
        return []
    animation=FuncAnimation(fig,update,frames=len(t),interval=20,blit=False)
    animation.save(output/'reference_comparison_4x1.gif',writer=PillowWriter(fps=50),dpi=85)
    plt.close(fig)


def run(a):
    if a.output.exists():raise FileExistsError('Use a new immutable output directory')
    assert sha(REFERENCE/'motion.npz')==MOTION_SHA
    reference=load(REFERENCE/'motion.npz'); direct=direct_reference(reference)
    meta=json.loads((REFERENCE/'report.json').read_text())
    geometry_report=json.loads(GEOMETRY.read_text());assert geometry_report['command_motion_sha256']==MOTION_SHA
    geometry=geometry_report['geometry']
    t=np.arange(len(reference['joint_pos']))/float(reference['fps'])
    base=reference['body_names'].astype(str).tolist().index('base_Link')
    assert base==0
    np.testing.assert_array_equal(reference['body_pos_w'][:,base,:2],direct['body_pos_w'][:,base,:2])
    np.testing.assert_array_equal(reference['body_pos_w'][0],direct['body_pos_w'][0])
    np.testing.assert_array_equal(reference['body_quat_w'][0],direct['body_quat_w'][0])
    budget=forward_budget(t,reference['body_pos_w'][:,base],.4,1.8)
    cases,series,initial=archived_cases(reference,budget)
    geometries=[reference_geometry(m,geometry) for m in (reference,direct)]
    a.output.mkdir(parents=True)
    for name in ('hockey','direct_brake'):(a.output/name).mkdir()
    shutil.copyfile(REFERENCE/'motion.npz',a.output/'hockey/motion.npz')
    np.savez_compressed(a.output/'direct_brake/motion.npz',**direct)
    np.savez_compressed(a.output/'common_evaluation_initial_state.npz',**initial)
    np.savez_compressed(a.output/'archived_hockey_signals.npz',**series)
    report=dict(kind='equal_forward_distance_hockey_vs_direct_brake_reference_preview',status='reference_pair_prepared_direct_policy_missing',
        primary_user_objective='stable completed stopping with less jitter',budget=budget,
        designed_base_origin_entry_speed_m_s=meta['entry_speed_m_s'],brake_window_s=[.4,1.8],hold_window_s=[1.8,2.76],
        physics_steps=0,new_training_transitions=0,reference_XY_and_clock_exactly_equal=True,
        distance_marker='base_Link origin projected onto initial reference course; not path length',
        direct_brake_definition='same translation; frozen initial base orientation and joint pose; no extra human maneuver',
        same_reference_initial_pose=True,same_reference_initial_velocities=False,
        initial_reference_body_velocity_max_difference_m_s=float(abs(reference['body_lin_vel_w'][0]-direct['body_lin_vel_w'][0]).max()),
        evaluation_reset_rule='One common physical state from common_evaluation_initial_state.npz for both arms; never reset separately from each reference velocity.',
        direct_joint_and_pose_targets_are_soft=True,no_absolute_height_or_wheel_spin_reward_added=True,
        simulated_direct_brake_stability_available=False,admitted_for_training=False,
        missing_runtime_gate='verify common physical reset and frame semantics; freeze direct reward/gates; no reference playback accepted as control',
        archived_hockey_cases=cases,
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in (REFERENCE/'motion.npz',REFERENCE/'report.json',GEOMETRY,SELECTION,Path(__file__))},
        reference_geometry={name:dict(max_wheel_gap_m=float(g[2].max()),bilateral_envelope_within_0p1mm_fraction=float((g[2][:-1]<=.0001).all(1).mean())) for name,g in zip(('hockey','direct_brake'),geometries)},
        limitations=['Equal reference braking distance/time is imposed; it is not measured performance.',
            'The direct pose is not a force-balanced braking solution; the learned controller must be allowed to lean and move its legs.',
            'Wheel angle is a placeholder, not a prescribed zero wheel-speed actuator command.',
            'Existing seed42 hockey policies were adaptively selected and are not paired held-out comparisons.',
            'Archived stop speed uses base COM velocity while primary distance uses base link origin; separate COM displacement is reported.',
            'Same forward distance alone does not enforce a lateral corridor; compare lateral excursion and overshoot too.'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    protocol=dict(status='reference_and_objective_prepared_runtime_and_reward_freeze_pending',
        training_backend='Isaac Lab / PhysX',algorithm='PPO',primary_objective=report['primary_user_objective'],
        shared=dict(distance_budget=budget,base_origin_entry_speed_m_s=meta['entry_speed_m_s'],reference_clock_hz=50,physics_hz=400,
            materials=[[.7,.7],[.9,.7]],seeds=[43,44,45,46,47],reference_XY_equal=True,
            shared_physical_initial_state_sha256=sha(a.output/'common_evaluation_initial_state.npz'),stop_speed_m_s=.1,hold_duration_s=.5),
        success_first=['stopped and held within the forward-distance tolerance','per-wheel native contact and geometry; hockey bilateral criterion preserved',
            'no fall and no non-wheel support','report longitudinal overshoot and lateral excursion'],
        distance_and_lateral_tolerances='Not yet frozen; report raw distances now, calibrate before prospective success scoring.',
        stability_metrics=['10-100 Hz world angular XY RMS at native 400 Hz','roll/pitch extrema and angular acceleration',
            'action-rate RMS and actuator saturation','per-wheel contact interruptions and vertical body motion'],
        reward_design=dict(direct='Remove positive lateral-sliding/friction-boundary incentives and hockey-specific success gates; keep common stopping/contact/smoothness terms.',
            comparison_kind='behavior-recipe comparison; not an isolated causal test of the human posture prior'),
        braking_torque='Actuated wheel policy output, not zeroing wheel joint state or prescribing a discontinuous lock',
        equality_verification=['same asset and actuator limits','same DR distribution and training transitions','same paired reset perturbations','same measured initial speed and raw material audit'],
        no_fixed_time_advantage_claim=True,remote_launch_performed=False)
    (a.output/'prospective_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    render(a.output,t,(reference,direct),geometries,budget)
    print(json.dumps(dict(budget=budget,reference_geometry=report['reference_geometry'],archived=[dict(id=c['id'],stop=c['supported_upright_speed_stop']) for c in cases]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

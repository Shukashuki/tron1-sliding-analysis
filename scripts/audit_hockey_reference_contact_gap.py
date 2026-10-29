"""Explain archived contact interruptions without changing a policy or reference.

The root-roll calculation is a per-frame geometry feasibility check. It is not
an IK retargeter, a new reference, a collision/dynamics rollout, or task success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from evaluate_hockey_contact_priority import bilateral_gate
from evaluate_hockey_physx import load, q_matrix, sha


def rotate_axis(vectors, axis, angle):
    v, u = np.asarray(vectors, float), np.asarray(axis, float)
    if v.shape[-1] != 3 or u.shape != (3,) or not np.isfinite(v).all() or not np.isfinite(u).all():
        raise ValueError('Finite vectors and one rotation axis required')
    if not np.isfinite(angle) or not np.isclose(np.linalg.norm(u), 1., rtol=0, atol=1e-8):
        raise ValueError('Unit rotation axis required')
    return v*np.cos(angle)+np.cross(u,v)*np.sin(angle)+np.sum(v*u,axis=-1,keepdims=True)*u*(1-np.cos(angle))


def cylinder_gap(center, axes, geometry):
    center, axes = np.asarray(center,float), np.asarray(axes,float)
    if center.shape != axes.shape or center.shape[-1] != 3:
        raise ValueError('Matching center/axis arrays required')
    if not np.isfinite(center).all() or not np.isfinite(axes).all():
        raise ValueError('Nonfinite wheel geometry')
    if not np.allclose(np.linalg.norm(axes,axis=-1),1.,rtol=0,atol=1e-8):
        raise ValueError('Unit wheel axes required')
    support = geometry['radius_m']*np.sqrt(np.maximum(0,1-axes[...,2]**2))
    support += geometry['half_width_m']*np.abs(axes[...,2])
    return center[...,2]-support-geometry['floor_z_m']


def wheel_geometry(data, position_key, quaternion_key, geometry):
    names=data['body_names'].astype(str).tolist()
    ids=[names.index('wheel_'+side+'_Link') for side in ('L','R')]
    # Select bodies before the final column; separated NumPy advanced indices
    # can otherwise move the wheel axis ahead of the time axis.
    axes=q_matrix(data[quaternion_key][:,ids])[...,:,1]
    center=data[position_key][:,ids]+axes*np.asarray(geometry['center_y_in_link_m'])[None,:,None]
    return center,axes,cylinder_gap(center,axes,geometry)


def root_roll_feasibility(root, root_forward, center, axes, geometry, max_angle_deg=15.):
    if root.shape != (3,) or center.shape != (2,3) or axes.shape != (2,3):
        raise ValueError('One root pose and two wheel geometries required')
    if not np.isfinite(max_angle_deg) or not 0 < max_angle_deg <= 45:
        raise ValueError('Bounded root-roll search required')
    relative=center-root

    def gaps(angle):
        c=rotate_axis(relative,root_forward,angle)+root
        a=rotate_axis(axes,root_forward,angle)
        return cylinder_gap(c,a,geometry)

    grid=np.radians(np.linspace(-max_angle_deg,max_angle_deg,121))
    difference=np.array([np.diff(gaps(angle)).item() for angle in grid])
    roots=[float(grid[k]) for k in np.flatnonzero(np.abs(difference)<1e-12)]
    for k in np.flatnonzero(difference[:-1]*difference[1:]<0):
        roots.append(brentq(lambda angle:np.diff(gaps(angle)).item(),grid[k],grid[k+1],xtol=1e-13))
    if not roots:
        return dict(geometrically_feasible=False, reason='No root-roll solution in the declared interval')
    angle=min(roots,key=abs); gap=gaps(angle); dz=-float(gap.mean())
    transformed=rotate_axis(relative,root_forward,angle)+root+np.array([0.,0.,dz])
    transformed_axes=rotate_axis(axes,root_forward,angle)
    residual=cylinder_gap(transformed,transformed_axes,geometry)
    if np.max(np.abs(residual))>1e-9:
        raise ValueError('Geometric root correction did not close both gaps')
    return dict(geometrically_feasible=True, root_roll_delta_deg=float(np.degrees(angle)),
        root_z_delta_m=dz, gap_residual_m=residual.tolist(),
        wheel_axis_change_deg=np.degrees(np.arccos(np.clip(np.sum(axes*transformed_axes,axis=-1),-1,1))).tolist())


def plot(output, reference_time, reference_gap, physical_time, physical_gap, root_delta, gate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('#2677ad','#db772a')
    fig, axes=plt.subplots(1,4,figsize=(22,4.4),constrained_layout=True)
    for side,color in enumerate(colors):
        label=('Left','Right')[side]
        axes[0].plot(reference_time,reference_gap[:,side]*1000,color=color,label=label)
        axes[1].plot(physical_time,physical_gap[:,side]*1000,color=color,label=label)
    for ax in axes[:2]:
        ax.axhline(.1,color='black',ls='--',lw=.7,label='0.1 mm contact tolerance')
        ax.set_ylabel('Signed cylinder clearance [mm]');ax.legend(fontsize=8)
    reference_count=int((reference_gap<=.0001).all(-1).sum())
    axes[0].set_title('Original reference geometry\n'+f'{reference_count} / {len(reference_gap)} frames have both gaps <=0.1 mm')
    axes[1].set_title('Recorded parent PPO geometry\nNative bilateral touch: '+f"{100*gate['both_touch_fraction']:.2f}%")
    for event in gate['bilateral_interruption']['events']:
        axes[1].axvspan(event['start_s'],event['end_s'],color='crimson',alpha=.16)
    axes[2].plot(reference_time,[row['root_roll_delta_deg'] for row in root_delta],color='#47724b')
    axes[2].set(title='Per-frame geometric root-roll correction\nJoints and root XY held fixed',ylabel='Roll correction [deg]')
    axes[3].plot(reference_time,[row['root_z_delta_m']*1000 for row in root_delta],color='#804f96')
    axes[3].set(title='Corresponding root vertical correction\nGeometry only; no dynamics validation',ylabel='Root vertical correction [mm]')
    for ax in axes:
        ax.set(xlim=(0,reference_time[-1]),xlabel='Time [s]')
        ax.axvspan(.4,1.8,color='gray',alpha=.07)
        ax.grid(alpha=.2)
    fig.suptitle('Hockey stop: original reference versus archived PhysX PPO | No new training or changed success criterion')
    fig.savefig(output/'reference_contact_gap_4x1.png',dpi=160)
    fig.savefig(output/'reference_contact_gap_4x1.svg')
    svg=output/'reference_contact_gap_4x1.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','reference','geometry-config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh diagnostic output required')
    read=lambda path:json.loads(path.read_text())
    report=read(a.run/'report.json');reference_report=read(a.reference/'report.json')
    for file in ('final_rollout.npz','native_contact_buffers.npz'):
        if sha(a.run/file)!=report['artifacts'][file]:raise ValueError('Archived rollout hash mismatch')
    if sha(a.reference/'motion.npz')!=report['motion_sha256'] or reference_report['motion_sha256']!=report['motion_sha256']:
        raise ValueError('Reference identity mismatch')
    if reference_report['physics_asset_sha256']!=report['asset_sha256']:
        raise ValueError('Reference and rollout use different wheel assets')
    geometry=read(a.geometry_config)['geometry']
    trace=load(a.run/'final_rollout.npz');raw=load(a.run/'native_contact_buffers.npz')
    reference=load(a.reference/'motion.npz')
    center,axis,gap=wheel_geometry(reference,'body_pos_w','body_quat_w',geometry)
    executed_center,executed_axis,executed_gap=wheel_geometry(trace,'body_pos','body_quat',geometry)
    frame_time=np.arange(len(gap))/float(reference['fps']);dt=float(trace['physics_dt_s'])
    te=np.arange(len(executed_gap))*dt;ts=te-dt
    contact=contact_kinematics(previous_state_trace(trace),raw,.0001)
    gate=bilateral_gate(ts,te,contact['touching'],contact['contact_unknown'],(0.,float(frame_time[-1])),dt)
    pre=np.maximum(np.arange(len(executed_gap))-1,0)
    inside=(ts>=-1e-9)&(ts<frame_time[-1]-1e-9)
    error=np.abs(contact['min_separation_m']-executed_gap[pre])
    valid=inside[:,None]&np.isfinite(error)
    if not valid.any() or error[valid].max()>1e-6:
        raise ValueError('Authored geometry does not match archived native separation')
    base=reference['body_names'].astype(str).tolist().index('base_Link')
    root=reference['body_pos_w'][:,base];forward=q_matrix(reference['body_quat_w'][:,base])[...,:,0]
    corrections=[root_roll_feasibility(root[i],forward[i],center[i],axis[i],geometry) for i in range(len(root))]
    if not all(r['geometrically_feasible'] for r in corrections):
        raise ValueError('Cannot summarize a complete geometric correction; inspect per-frame feasibility')
    rows=[]
    for side in (0,1):
        off=inside&~contact['touching'][:,side]&~contact['contact_unknown'][:,side]
        rows.append(dict(side=('left','right')[side],interruption_s=float(off.sum()*dt),
            maximum_gap_during_interruption_mm=float(executed_gap[pre,side][off].max()*1000) if off.any() else None,
            interruption_samples=int(off.sum()),unknown_samples=int((inside&contact['contact_unknown'][:,side]).sum())))
    result=dict(kind='hockey_reference_contact_geometry_audit_v1',
        script_sha256=sha(__file__),input_sha256=dict(run_report=sha(a.run/'report.json'),
            final_rollout=sha(a.run/'final_rollout.npz'),native_contact_buffers=sha(a.run/'native_contact_buffers.npz'),
            reference=sha(a.reference/'motion.npz'),reference_report=sha(a.reference/'report.json'),geometry_config=sha(a.geometry_config)),
        native_geometry_alignment_max_error_m=float(error[valid].max()),
        reference=dict(frames=len(gap),bilateral_geometry_frames=int((gap<=.0001).all(-1).sum()),
            bilateral_geometry_fraction=float((gap<=.0001).all(-1).mean()),maximum_gap_mm_per_wheel=(gap.max(0)*1000).tolist(),
            original_projection_is_one_wheel=reference_report['one_wheel_ground_projection']),
        recorded_parent=dict(primary_gate=gate,wheels=rows),
        geometric_root_correction=dict(frames=len(corrections),all_frames_have_geometry_solution=True,
            max_abs_roll_delta_deg=max(abs(r['root_roll_delta_deg']) for r in corrections),
            root_z_delta_range_mm=[min(r['root_z_delta_m'] for r in corrections)*1000,max(r['root_z_delta_m'] for r in corrections)*1000],
            max_wheel_axis_change_deg=max(max(r['wheel_axis_change_deg']) for r in corrections),
            max_gap_residual_m=max(max(abs(v) for v in r['gap_residual_m']) for r in corrections),
            joint_angles_changed=False,root_XY_changed=False,root_heading_forward_axis_changed=False,
            roll_search_range_deg=[-15.,15.],per_frame=corrections),
        interpretation='The original reference is not bilaterally grounded. Small free-root corrections admit bilateral geometry with the same joints; this does not prove a dynamically feasible stop or identify the cause of the policy interruptions.',
        reference_modified=False,policy_modified=False,physics_steps=0,task_success_claimed=False,
        criterion_relaxed=False,previous_upload_request_payload_modified=False)
    a.output.mkdir(parents=True)
    plot(a.output,frame_time,gap,te,executed_gap,corrections,gate)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(a.output/'geometry_diagnostics.npz',reference_time_s=frame_time,reference_gap_m=gap,
        physical_time_s=te,executed_gap_m=executed_gap,native_gap_pre_m=contact['min_separation_m'],
        native_touching=contact['touching'],root_roll_delta_deg=np.array([r['root_roll_delta_deg'] for r in corrections]),
        root_z_delta_m=np.array([r['root_z_delta_m'] for r in corrections]))
    print(json.dumps(dict(reference=result['reference'],recorded_wheels=rows,
        roll_correction_max_deg=result['geometric_root_correction']['max_abs_roll_delta_deg'],
        native_gap_alignment_max_m=result['native_geometry_alignment_max_error_m'])))


if __name__=='__main__':main()

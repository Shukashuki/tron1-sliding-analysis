"""Read-only contact and heading diagnostics of the first, unqualified actor.

This never admits an export-rejected training attempt to the formal benchmark.
Wheel kinematics are reconstructed from actual poses/velocities using FK only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_contact_v2 as metrics
from controller_metrics import wrap
import run_cem_ground_contact_v2 as assets


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def angles(value,mask):
    x=np.degrees(np.asarray(value)[mask])
    return dict(samples=len(x),mean_signed_deg=float(x.mean()) if len(x) else None,
        mean_abs_deg=float(np.abs(x).mean()) if len(x) else None,
        abs_p95_deg=float(np.percentile(np.abs(x),95)) if len(x) else None)


def run(a):
    assert not a.output.exists()
    folder=a.collection/'posthoc_cpu_nominal_v1'
    meta=json.loads((folder/'posthoc_diagnostic_status.json').read_text())
    assert meta['kind']=='posthoc_recovered_checkpoint_CPU_MuJoCo_diagnostic'
    assert meta['formally_qualified_training_cohort'] is False and meta['qualified_formal_comparison_admission'] is False
    assert sha(folder/'rollout.npz')==meta['recording_proof']['recording_sha256']
    raw=npz(folder/'rollout.npz');reference=metrics.metrics.reference_data(a.reference)
    values,trace=metrics.recorded_metrics(raw,'mujoco',reference)
    original_raw,_=metrics.load_baseline('mujoco');original,_=metrics.recorded_metrics(original_raw,'mujoco',reference)
    protocol=json.loads((Path(__file__).resolve().parents[1]/'config/ppo_ground_contact_v2k_400hz_v1.json').read_text())
    screen=metrics.nominal_screen(values,original,protocol,'mujoco')
    fk,source=metrics.metrics.load_fk(a.model,a.legacy_evaluator)
    reconstructed=fk.reconstruct(trace);signals=reconstructed['signals']
    time=trace['time_s'];core=(time>=1.44-1e-9)&(time<=4.+1e-9)
    loaded=trace['floor_normal_force'][:,:,2]>5
    wheel=[]
    for side in range(2):
        mask=core&loaded[:,side]&signals['wheel_center_course_to_wheel_valid'][:,side]
        point_valid=core&loaded[:,side]&(trace['contact_count'][:,side]>0)
        wheel.append(dict(side=('left','right')[side],
            core_loaded_heading=angles(signals['wheel_center_course_to_wheel_gamma_rad'][:,side],mask),
            measured_point_slip_samples=int(point_valid.sum()),
            measured_force_weighted_lateral_contact_slip_RMS_m_s=float(np.sqrt(np.mean(trace['contact_slip'][point_valid,side]**2))) if point_valid.any() else None))
    _,heading=metrics.metrics.smooth._orientation(reference['quaternion'])
    beta_ref=np.unwrap(np.arctan2(reference['velocity'][:,1],reference['velocity'][:,0])-heading)
    desired=np.interp(time,reference['time_s'],beta_ref)
    beta=dict(reference_core=angles(wrap(desired),core),
              actual_core=angles(signals['root_directed_beta_rad'],core&signals['root_beta_valid']))
    contact=loaded.any(-1);window=(time>=.02-1e-9)&(time<=6.96+1e-9)
    starts=np.flatnonzero(window&~contact&np.r_[False,contact[:-1]])
    phases=(starts-1)%8
    phase_hist=np.bincount(phases,minlength=8)
    result=dict(kind='first_recovered_nonlinear_actor_CPU_MuJoCo_native_diagnostic',
        status='completed_analysis',formally_qualified_comparison=False,task_success_claimed=False,
        actor_sha256=meta['actor_sha256'],recording_sha256=sha(folder/'rollout.npz'),
        analyzer_sha256=sha(__file__),reference_sha256=sha(a.reference),
        FK_source=source,FK_audit=reconstructed['audit'],
        metrics=values,diagnostic_original_nominal_screen=screen,
        core_beta=beta,core_ground_conditioned_wheel_angles_and_patch_slip=wheel,
        support_interruption_update_phase_histogram=phase_hist.tolist(),
        phase0_is_first_native_sample_after_50Hz_action_update=True,
        heading_is_axial_modulo_pi=True,loaded_force_threshold_n_strict=5,
        point_slip_gating='Native active point count>0 as well as summed wheel Fz>5N; zero slip without active points is unavailable',
        causal_controller_or_friction_conclusion_claimed=False)
    a.output.mkdir(parents=True)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(22,4.5),layout='constrained')
    axes[0].plot(reference['position'][:,0],reference['position'][:,1],'--',c='gray',label='reference')
    axes[0].plot(trace['position'][:,0],trace['position'][:,1],label='recovered actor / native MuJoCo')
    axes[0].set_aspect('equal',adjustable='datalim');axes[0].set_title('World XY [m]');axes[0].legend(fontsize=7)
    reference_moving=np.interp(time,reference['time_s'],np.linalg.norm(reference['velocity'][:,:2],axis=-1))>=.25
    axes[1].plot(time,np.where(reference_moving,np.degrees(wrap(desired)),np.nan),'--',c='gray',label='reference body beta')
    axes[1].plot(time,np.where(signals['root_beta_valid'],np.degrees(signals['root_directed_beta_rad']),np.nan),label='actual body beta')
    for side,color in enumerate(('tab:red','tab:purple')):
        mask=loaded[:,side]&signals['wheel_center_course_to_wheel_valid'][:,side]
        axes[1].scatter(time[mask],np.degrees(signals['wheel_center_course_to_wheel_gamma_rad'][mask,side]),s=2,c=color,
                        label=('left','right')[side]+' loaded wheel angle')
    axes[1].set_title('Body beta / loaded wheel angle [deg]');axes[1].legend(fontsize=7)
    for side,color in enumerate(('tab:red','tab:purple')):
        axes[2].plot(time,trace['floor_normal_force'][:,side,2],c=color,label=('left','right')[side])
    axes[2].axhline(5,ls='--',c='gray');axes[2].set_title('Actual wheel floor normal force [N]')
    axes[2].legend(fontsize=7)
    up=metrics.quaternion_rotate(trace['quaternion'],np.array([0.,0.,1.]))
    axes[3].plot(time,np.degrees(np.arccos(np.clip(up[:,2],-1,1))))
    axes[3].set_title('Body tilt [deg], diagnostic')
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle('First checkpoint post-hoc CPU diagnostic only | NOT the qualified formal cohort | mu=0.7')
    fig.savefig(a.output/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(output=str(a.output),core_beta=beta,ground_conditioned_wheels=wheel,phase_histogram=phase_hist.tolist(),
                          formal_comparison_admission=False,task_success_claimed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=assets.paired.DEFAULT_REFERENCE)
    p.add_argument('--model',type=Path,default=assets.paired.DEFAULT_MODEL)
    p.add_argument('--legacy-evaluator',type=Path,default=assets.paired.DEFAULT_LEGACY)
    run(p.parse_args())

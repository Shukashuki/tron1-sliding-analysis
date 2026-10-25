"""Recompute gain-transfer mechanics and original task gates from native records."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import analyze_physx_leg_command_probe as previous
import eval_physx_leg_pd_probe as evaluator
import ppo_backend_400hz_contract as recording

ROOT=Path(__file__).resolve().parents[1]
PROFILES=('baseline_500_10','soft_40_2p5')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
npz=previous.prior.previous.npz
metrics=previous.prior.metrics


def verify_case(folder,profile,protocol):
    meta=read(folder/'leg_pd_status.json');report=read(folder/'report.json')
    assert meta==read(folder/'wheel_contact_model_status.json')==report['wheel_contact_model_transfer']
    assert meta['kind']=='frozen_K_PhysX_explicit_leg_PD_transfer'
    assert meta['normalized_actor_sha256']==protocol['actor_sha256']
    assert meta['geometry']=='authored' and meta['law']=='native_coulomb' and meta['mu']==.7
    assert meta['additional_training_updates']==0
    assert meta['native_states']==2793 and meta['policy_decisions']==349
    assert meta['gain_protocol_sha256']==sha(ROOT/'config/physx_leg_pd_probe_v1.json')
    assert report['status']=='evaluated' and report['terminal_states_captured_before_auto_reset']
    assert not report['domain_randomization_enabled'] and not report['observation_noise_enabled']
    assert report['deterministic_start_frame']==0 and report['runtime_device_audit']['status']=='verified'
    pd=meta['explicit_leg_PD_intervention']
    assert pd==report['explicit_leg_PD_intervention']==report['policy_export']['explicit_leg_PD_intervention']
    assert pd['profile']==profile and pd['physical_motor_computations']==2792
    assert pd['inference_only'] and pd['retrained_for_modified_gains'] is False
    assert pd['training_leg_kp']==500. and pd['training_leg_kd']==10.
    kp,kd=protocol['profiles'][profile]
    assert pd['actual_gain_readback']==dict(kp_range=[kp,kp],kd_range=[kd,kd])
    deployed=report['policy_export']
    assert (deployed['leg_kp'],deployed['leg_kd'])==(kp,kd)
    assert (deployed['policy_training_leg_kp'],deployed['policy_training_leg_kd'])==(500.,10.)
    assert report['contact_material_audit']['leg_pd_gain_readback']==pd['actual_gain_readback']
    assert pd['torque_speed_limits_and_wheel_gains_unchanged']
    for remote,digest in meta['source_sha256'].items():
        path=ROOT/('config' if Path(remote).suffix=='.json' else 'scripts')/Path(remote).name
        assert sha(path)==digest,remote
    for name in ('physics_trajectory','tire_trajectory','leg_pd_trajectory'):
        assert sha(folder/(name+'.npz'))==meta[name+'_sha256']
    recording.verify_recording(folder/'physics_trajectory.npz','PhysX')
    if profile=='baseline_500_10':assert meta['physics_trajectory_sha256']==protocol['baseline_physics_trajectory_sha256']
    return meta


def load_cases(collection,reference):
    collection=Path(collection);protocol=read(ROOT/'config/physx_leg_pd_probe_v1.json')
    evidence=read(collection/'collection_manifest.json')
    assert evidence['kind']=='completed_K_leg_PD_probe_native_evidence'
    run_id=evidence['run_id'];assert run_id.startswith('episodes_') and run_id.replace('_','').isalnum()
    for name,row in evidence['files'].items():
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts
        assert sha(collection/p)==row['sha256']
    state=read(collection/run_id/'status.json')
    assert state['status']=='both_explicit_PD_transfers_completed_require_analysis'
    assert [c['id'] for c in state['cases']]==list(PROFILES)
    assert all(c['status']=='completed_verified' and c['returncode']==0 for c in state['cases'])
    criteria=read(ROOT/'config/ppo_ground_contact_v2k_400hz_v1.json')
    original=read(ROOT/'results/physx_tron1_wheel_models_20261025_v1/analysis.json')['original_baseline_metrics']
    rows=[];traces={};motors={}
    for profile in PROFILES:
        folder=collection/run_id/profile;meta=verify_case(folder,profile,protocol)
        raw=npz(folder/'physics_trajectory.npz');pd=npz(folder/'leg_pd_trajectory.npz')
        evaluator.verify_motor_recording(pd,raw)
        values,trace=metrics.recorded_metrics(raw,'physx',reference)
        trace['contact_force']=trace['floor_normal_force'].copy()
        expected=pd['kp']*(pd['target_pos_rad']-pd['joint_pos_rad'])+pd['kd']*(pd['target_vel_rad_s']-pd['joint_vel_rad_s'])+pd['feedforward_effort_nm']
        np.testing.assert_allclose(expected,pd['computed_effort_nm'],rtol=2e-6,atol=2e-4)
        np.testing.assert_array_equal(pd['kp'],np.full((2792,6),protocol['profiles'][profile][0]))
        np.testing.assert_array_equal(pd['kd'],np.full((2792,6),protocol['profiles'][profile][1]))
        np.testing.assert_allclose(pd['time_before_step_s'],trace['time_s'][:-1],rtol=0,atol=1e-12)
        core=(pd['time_before_step_s']>=1.44-1e-9)&(pd['time_before_step_s']<=4.+1e-9)
        err=pd['target_pos_rad']-pd['joint_pos_rad'];applied=pd['applied_effort_nm'];computed=pd['computed_effort_nm']
        servo=dict(core_position_error_RMS_rad=float(np.sqrt(np.mean(err[core]**2))),
            core_computed_effort_RMS_nm=float(np.sqrt(np.mean(computed[core]**2))),
            core_applied_effort_RMS_nm=float(np.sqrt(np.mean(applied[core]**2))),
            core_applied_effort_absolute_peak_nm=float(np.abs(applied[core]).max()),
            core_clipped_motor_sample_fraction=float(np.mean(np.abs(computed[core]-applied[core])>1e-5)))
        fz=trace['floor_normal_force'][:,:,2];time=trace['time_s'];turn=(time>=1.44-1e-9)&(time<=4.+1e-9)
        rows.append(dict(id=profile,metrics=values,nominal_screen=metrics.nominal_screen(values,original,criteria,'physx'),
            servo_diagnostics=servo,force_diagnostics=dict(initial_readout_peak_n=float(fz[0].max()),
                policy_controlled_peak_n=float(fz[1:].max()),core_peak_n=float(fz[turn].max())),
            phase_statistics=previous.phase_statistics(raw,trace),metadata=meta))
        traces[profile]=trace;motors[profile]=pd
    for field in ('position','quaternion','com_velocity','angular_velocity','joint_pos','joint_vel'):
        np.testing.assert_array_equal(traces[PROFILES[0]][field][0],traces[PROFILES[1]][field][0])
    return rows,traces,motors


def run(args):
    assert not args.output_dir.exists()
    reference=metrics.metrics.reference_data(args.reference)
    rows,traces,motors=load_cases(args.collection,reference)
    args.output_dir.mkdir(parents=True)
    result=dict(kind='frozen_K_explicit_leg_PD_native_transfer_comparison',status='completed_analysis',rows=rows,
        reference_sha256=sha(args.reference),analyzer_sha256=sha(__file__),same_initial_mechanical_state=True,
        baseline_observer_byte_identical=True,inference_only_not_a_retrained_policy_comparison=True,
        actuator_PD_gains_changed=True,original_nominal_acceptance_unchanged=True,
        common_reward_reference_actor_and_contact_model=True,cross_engine_qualification_complete=False,task_success_claimed=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4.5),layout='constrained')
    axes[0].plot(reference['position'][:,0],reference['position'][:,1],'--',c='gray',label='reference')
    for row,color in zip(rows,('#2271b2','#d55e00'),strict=True):
        key=row['id'];t=traces[key];motor=motors[key]
        axes[0].plot(t['position'][:,0],t['position'][:,1],c=color,label=key)
        axes[1].plot(t['time_s'],t['floor_normal_force'][:,:,2].max(-1),c=color,lw=.65,label=key)
        axes[2].plot(t['time_s'],np.linalg.norm(t['angular_velocity'][:,:2],axis=-1),c=color,lw=.65,label=key)
        axes[3].plot(motor['time_before_step_s'],np.sqrt(np.mean(motor['applied_effort_nm']**2,axis=-1)),c=color,lw=.65,label=key)
    axes[0].set(title='Native world XY',xlabel='X(m)',ylabel='Y(m)',aspect='equal')
    axes[1].set(title='Maximum upward wheel force',xlabel='Time(s)',ylabel='Fz(N)',yscale='symlog');axes[1].axhline(5,c='black',ls='--',lw=1)
    axes[2].set(title='Native root angular motion',xlabel='Time(s)',ylabel='Angular XY norm(rad/s)')
    axes[3].set(title='Actual applied leg torque',xlabel='Time(s)',ylabel='Six-joint RMS(Nm)')
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.suptitle('Same frozen K policy | PhysX μ=.7 | explicit leg PD500/10 vs40/2.5; no retraining')
    fig.savefig(args.output_dir/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps([dict(id=r['id'],core_XY=r['metrics']['windows']['core']['tracking']['xy_rmse_m'],
        support=r['metrics']['any_wheel_ground_support']['full_contact']['any_wheel_support_sample_fraction'],
        gap_s=r['metrics']['any_wheel_ground_support']['full_contact']['longest_no_support_observed_s'],
        HF_ratio=r['nominal_screen']['core_HF_ratio_to_original_same_engine_PPO'],failed=r['nominal_screen']['failed_checks']) for r in rows]))


if __name__=='__main__':
    import run_cem_ground_contact_v2 as assets
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=assets.paired.DEFAULT_REFERENCE);run(p.parse_args())

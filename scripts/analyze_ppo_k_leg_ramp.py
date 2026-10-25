"""Independent K versus ramp-trained K nominal tracking/support/motor audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_contact_v2 as metrics
import collect_ppo_k_leg_ramp as collection
from ppo_backend_400hz_contract import verify_recording
from train_ppo_k_leg_ramp import EXECUTION

ROOT=Path(__file__).resolve().parents[1]
ORDER=('K_physx','ramp_physx','K_mujoco','ramp_mujoco')
K_ACTOR='fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912'
K_RECORDING={'physx':'cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a',
    'mujoco':'af790839128d5d02a8233a7a7b2cd8b23eb508e9056cd4d1c619b8d4173d14ef'}
BASELINE=ROOT/'results/native_contact_manifold_20261025_v1/evidence'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def command_audit(command,raw,trace,engine,reference_joint0,reference_names):
    names=command['leg_joint_names'].tolist()
    ids=[trace['joint_names'].index(n) for n in names]
    ref_ids=[reference_names.index(n) for n in names]
    assert len(ids)==len(set(ids))==6
    desired=command['desired'].reshape(349,8,6);start=command['start'].reshape(349,8,6)
    actual=command['actual'].reshape(349,8,6)
    actions=raw['actions_clipped'][1::8,0] if engine=='physx' else raw['actions_clipped']
    assert actions.shape==(349,8)
    np.testing.assert_allclose(desired[:,0],reference_joint0[ref_ids]+actions[:,:6],rtol=0,atol=2e-6)
    np.testing.assert_array_equal(desired,np.broadcast_to(desired[:,:1],desired.shape))
    np.testing.assert_array_equal(start,np.broadcast_to(start[:,:1],start.shape))
    np.testing.assert_allclose(start[0,0],trace['joint_pos'][0,ids],rtol=0,atol=2e-6)
    np.testing.assert_array_equal(start[1:,0],actual[:-1,-1])
    expected=start+(desired-start)*(np.arange(1,9)[None,:,None]/8.)
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)
    np.testing.assert_array_equal(actual[:,-1],desired[:,-1])
    np.testing.assert_allclose(command['wheel_effort'].reshape(349,8,2),
        np.broadcast_to((12*actions[:,6:])[:,None],(349,8,2)),rtol=0,atol=2e-6)
    # Independently reconstruct the actual pre-step PD + electrical envelope.
    # Float32 eager PhysX arithmetic and float64 CPU arithmetic share the
    # retained 0.005 Nm/2e-5 tolerance; these are checks, never task gates.
    q=trace['joint_pos'][:-1,ids];dq=trace['joint_vel'][:-1,ids]
    computed=500*(actual.reshape(2792,6)-q)-10*dq
    speed=np.clip(dq,-30.,30.)
    lower=np.maximum(80*(-1-speed/15),-80)
    upper=np.minimum(80*(1-speed/15),80)
    expected_torque=np.clip(computed,lower,upper)
    raw_names=raw['joint_names'].tolist();raw_ids=[raw_names.index(n) for n in names]
    torque=raw['applied_joint_torque_nm'][1:,0,raw_ids] if engine=='physx' else raw['joint_torque_nm'][1:,raw_ids]
    np.testing.assert_allclose(torque,expected_torque,rtol=2e-5,atol=.005)
    return dict(all_native_command_fractions_and_exact_endpoints_verified=True,
        unchanged_desired_raw_action_mapping_and_wheel_effort_verified=True,
        actual_native_leg_PD_and_motor_envelope_verified=True,
        maximum_leg_torque_reconstruction_error_nm=float(np.abs(torque-expected_torque).max()),
        torque_check_tolerance=dict(rtol=2e-5,atol_nm=.005),native_applications=2792)


def load_cases(folder,reference,reference_file):
    folder=Path(folder);_,status,proof=collection.verify_completed(folder)
    p=read(folder/'source_snapshot_v1/config/ppo_k_leg_ramp_400hz_v1.json')
    old=read(ROOT/'config/ppo_ground_contact_v2k_400hz_v1.json')
    assert p['leg_command_execution']==EXECUTION
    allowed={'cohort_id','experiment_design','leg_command_execution'}
    assert all(p[k]==v for k,v in old.items() if k not in allowed)
    manifest=read(folder/'formal_v1/manifest.json');audit=read(folder/'formal_v1/leg_ramp_training_audit.json')
    assert audit['execution_audit']==manifest['leg_command_training']
    assert audit['initial_policy_state_sha256']=='7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d'
    assert manifest['num_envs']==512 and manifest['iterations']==1000
    assert audit['execution_audit']['all15_rewards_and_original_DR_unchanged']
    weight_check=read(folder/'completion_process_audit.json')['final_neural_files_read_and_hash_verified_on_execution_host_only']
    assert weight_check=={'policy.pt':proof['actor_sha256'],'model_final.pt':proof['checkpoint_sha256']}
    ref=npz(reference_file);reference_names=ref['joint_names'].tolist()
    rows=[];traces={}
    for engine in ['physx','mujoco']:
        original_raw,_=metrics.load_baseline(engine)
        original_metrics,_=metrics.recorded_metrics(original_raw,engine,reference)
        for variant in ['K','ramp']:
            f=BASELINE/engine/(engine+'_v1') if variant=='K' else folder/'nominal_v1'/engine
            name='physics_trajectory.npz' if engine=='physx' else 'rollout.npz'
            verify_recording(f/name,'PhysX' if engine=='physx' else 'native CPU MuJoCo')
            if variant=='K':
                assert sha(f/name)==K_RECORDING[engine]
                assert read(f/'manifold_status.json')['normalized_actor_sha256']==K_ACTOR
            else:
                report=read(f/'report.json')
                if engine=='physx':
                    assert report['status']=='evaluated' and report['deterministic_start_frame']==0
                    assert report['terminal_states_captured_before_auto_reset']
                    assert not report['domain_randomization_enabled'] and not report['observation_noise_enabled']
                else:
                    assert report['status']=='completed' and report['completed_full_reference']
                    assert report['hidden_resets']==0 and report['root_state_writes']==1
            raw=npz(f/name);values,trace=metrics.recorded_metrics(raw,engine,reference)
            trace['contact_force']=trace['floor_normal_force'].copy();key=variant+'_'+engine
            row=dict(id=key,engine=engine,title=variant+' PPO | '+engine,variant=variant,
                actor_sha256=K_ACTOR if variant=='K' else proof['actor_sha256'],recording_sha256=sha(f/name),
                metrics=values,nominal_screen=metrics.nominal_screen(values,original_metrics,p,engine))
            if variant=='ramp':
                row['independent_command_audit']=command_audit(npz(f/'leg_command_trajectory.npz'),raw,trace,engine,
                    ref['joint_pos'][0],reference_names)
                prior=traces['K_'+engine]
                for n in ['position','com_velocity','angular_velocity','joint_pos','joint_vel']:
                    np.testing.assert_allclose(trace[n][0],prior[n][0],rtol=0,atol=2e-6)
                assert abs(np.dot(trace['quaternion'][0],prior['quaternion'][0]))>=1-1e-7
            rows.append(row);traces[key]=trace
    assert [r['id'] for r in rows]==list(ORDER)
    return rows,traces,proof


def run(a):
    assert not a.output_dir.exists()
    reference=metrics.metrics.reference_data(a.reference)
    rows,traces,proof=load_cases(a.collection,reference,a.reference)
    a.output_dir.mkdir(parents=True)
    result=dict(kind='K_vs_ramp_trained_K_dual_engine',status='completed_analysis',rows=rows,
        analyzer_sha256=sha(__file__),reference_sha256=sha(a.reference),completed_training_proof=proof,
        all_original_nominal_gates_retained=True,qualitative_review_complete=False,task_success_claimed=False)
    (a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(22,4.5),layout='constrained')
    axes[0].plot(reference['position'][:,0],reference['position'][:,1],'--',color='gray',label='reference')
    for row in rows:
        t=traces[row['id']];time=t['time_s']
        axes[0].plot(t['position'][:,0],t['position'][:,1],label=row['title'])
        axes[1].plot(time,np.linalg.norm(t['com_velocity'][:,:2],axis=-1))
        axes[2].plot(time,t['floor_normal_force'][:,:,2].max(-1))
        up=metrics.quaternion_rotate(t['quaternion'],np.array([0.,0.,1.]))
        axes[3].plot(time,np.degrees(np.arccos(np.clip(up[:,2],-1,1))))
    axes[0].set_aspect('equal',adjustable='datalim');axes[0].legend(fontsize=6)
    for ax,title in zip(axes,['World XY [m]','COM XY speed [m/s]','Maximum wheel ground Fz [N]','Body tilt [deg]']):
        ax.set_title(title);ax.grid(alpha=.2)
    axes[2].axhline(5,color='gray',ls='--')
    fig.suptitle('Same K reward and 12M budget | trained for leg ramp versus original hold | mu=.7')
    fig.savefig(a.output_dir/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({'rows':len(rows),'task_success_claimed':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ['collection','reference','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

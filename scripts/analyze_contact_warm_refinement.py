"""Independent paired warm K/M nominal audit, retaining original task gates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_contact_v2 as metrics
import collect_contact_warm_refinement as collection
from ppo_backend_400hz_contract import verify_recording

ROOT=Path(__file__).resolve().parents[1]
ORDER=('K_physx','M_physx','K_mujoco','M_mujoco')
BASELINE=ROOT/'results/native_contact_manifold_20261025_v1/evidence'
K_SHA={'physx':'cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a',
       'mujoco':'af790839128d5d02a8233a7a7b2cd8b23eb508e9056cd4d1c619b8d4173d14ef'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def verify_pair(folder):
    driver,status=collection.verify_completed(folder)
    protocol=read(folder/'source_snapshot_v1/config/contact_warm_refinement_400hz_v1.json')
    original=read(ROOT/'config/ppo_ground_contact_v2k_400hz_v1.json')
    assert protocol['nominal_acceptance_before_training']==original['nominal_acceptance_before_training']
    assert protocol['training_randomization']==original['training_randomization']
    manifests={v:read(folder/('formal_'+v)/'manifest.json') for v in ['K','M']}
    k,m=manifests['K'],manifests['M']
    assert k['warm_start']==m['warm_start'] and k['initial_policy_state_sha256']==m['initial_policy_state_sha256']==collection.INITIAL
    assert read(folder/'formal_K/runner_config.json')==read(folder/'formal_M/runner_config.json')
    for v,manifest in manifests.items():
        completion=read(folder/('formal_'+v)/'cohort_completion_audit.json')
        assert completion['cohort']==manifest['cohort']
        assert manifest['num_envs']==512 and manifest['iterations']==200 and manifest['seed']==42
        assert manifest['contact_settings']==protocol['variant_contact_settings'][v]
        assert manifest['cohort']['normalizer_final_count']==14745600
        assert manifest['algorithm_source_sha256']==dict(runner='e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5',
            ppo='deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c')
        proof=driver['proofs']['formal_'+v]
        assert completion['actor_sha256']==proof['actor_sha256'] and not completion['from_scratch']
    allowed={'ground_contact_reward_v2','description'}
    assert {n:v for n,v in k['task_profile'].items() if n not in allowed}=={n:v for n,v in m['task_profile'].items() if n not in allowed}
    assert k['environment']['reward_terms']==m['environment']['reward_terms']
    assert len(k['environment']['reward_terms'])==15
    assert k['cohort']['refinement_optimizer']==m['cohort']['refinement_optimizer']==protocol['refinement_optimizer']
    return driver,status,protocol


def load_cases(folder,reference):
    folder=Path(folder);driver,status,protocol=verify_pair(folder)
    rows=[];traces={};historical={};old_traces={};old_metrics={}
    for engine in ['physx','mujoco']:
        raw,_=metrics.load_baseline(engine);old_metrics[engine],_=metrics.recorded_metrics(raw,engine,reference)
        name='physics_trajectory.npz' if engine=='physx' else 'rollout.npz'
        f=BASELINE/engine/(engine+'_v1')/name;assert sha(f)==K_SHA[engine]
        values,t=metrics.recorded_metrics(npz(f),engine,reference)
        historical[engine]=dict(metrics=values,nominal_screen=metrics.nominal_screen(values,old_metrics[engine],protocol,engine),recording_sha256=sha(f))
        old_traces[engine]=t
    for child in status['cases']:
        key=child['id'];engine=child['backend'];variant=child['variant'];f=folder/'nominal_v1'/key
        name='physics_trajectory.npz' if engine=='physx' else 'rollout.npz'
        verify_recording(f/name,'PhysX' if engine=='physx' else 'native CPU MuJoCo')
        report=read(f/'report.json');metadata=read(f/'warm_contact_evaluation_status.json')
        if engine=='physx':
            assert report['status']=='evaluated' and report['deterministic_start_frame']==0
            assert report['terminal_states_captured_before_auto_reset'] and not report['domain_randomization_enabled']
            assert not report['observation_noise_enabled'] and report['runtime_device_audit']['status']=='verified'
        else:
            assert report['status']=='completed' and report['completed_full_reference']
            assert report['hidden_resets']==0 and report['root_state_writes']==1
            assert report['physics_steps']==2792 and report['policy_steps']==349
        values,trace=metrics.recorded_metrics(npz(f/name),engine,reference)
        trace['contact_force']=trace['floor_normal_force'].copy()
        for n in ['position','com_velocity','angular_velocity','joint_pos','joint_vel']:
            np.testing.assert_allclose(trace[n][0],old_traces[engine][n][0],atol=2e-6,rtol=0)
        assert abs(np.dot(trace['quaternion'][0],old_traces[engine]['quaternion'][0]))>=1-1e-7
        rows.append(dict(id=key,variant=variant,engine=engine,title=variant+' warm refinement | '+engine,
            actor_sha256=metadata['normalized_actor_sha256'],recording_sha256=sha(f/name),metrics=values,
            nominal_screen=metrics.nominal_screen(values,old_metrics[engine],protocol,engine)))
        traces[key]=trace
    rows.sort(key=lambda r:ORDER.index(r['id']))
    assert [r['id'] for r in rows]==list(ORDER)
    return rows,traces,driver['proofs'],historical


def run(a):
    assert not a.output_dir.exists();reference=metrics.metrics.reference_data(a.reference)
    rows,traces,proofs,historical=load_cases(a.collection,reference);a.output_dir.mkdir(parents=True)
    result=dict(kind='paired_warm_K_M_contact_refinement_dual_engine',status='completed_analysis',rows=rows,
        paired_training_proofs=proofs,historical_original_K=historical,analyzer_sha256=sha(__file__),reference_sha256=sha(a.reference),
        stage_transitions_per_arm=2457600,cumulative_transitions_per_resulting_policy=14745600,
        same_K_initialization_and_optimizer_between_arms=True,all_original_gates_retained=True,
        qualitative_review_complete=False,task_success_claimed=False)
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
    for ax,title in zip(axes,['World XY [m]','COM XY speed [m/s]','Max wheel floor Fz [N]','Body tilt [deg]']):ax.set_title(title);ax.grid(alpha=.2)
    axes[2].axhline(5,color='gray',ls='--')
    fig.suptitle('Paired warm K/M | same K checkpoint + same conservative2.46M refinement | mu=.7')
    fig.savefig(a.output_dir/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({'rows':len(rows),'task_success_claimed':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ['collection','reference','output-dir']:p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

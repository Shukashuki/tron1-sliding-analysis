#!/usr/bin/env python3
"""CPU intervention audit of the exact frozen car actor on archived hockey states.

These are action predictions on fixed recorded states, not new closed-loop
trajectories. Normalized magnitudes use the actor's std+epsilon, not z-scores.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=Path(__file__).resolve().parents[1]
    paths=dict(hockey=root/'results/hockey_stop_physx_probe_20261026_v2/evidence/probes_v2/baseline/final_rollout.npz',
        car=root/'results/physx_policy_mppi_reference_20261025_v1/evidence/baseline_validation_v1/final_rollout.npz',
        reference=root/'results/contact_duration_pair_20261025_v1/completed_evidence_v1/source_snapshot_v3/references/fit_differential.npz')
    if sha(a.actor)!='e5b94e87ed070f28c391cc7b0a11b216ce6b6d887141a28e7c83157085a9d8d0':raise ValueError('Wrong frozen actor')
    for name in ('hockey','car'):
        report=json.loads(paths[name].with_name('report.json').read_text())
        if sha(paths[name])!=report['artifacts']['final_rollout.npz'] or report['actor_sha256']!=sha(a.actor):raise ValueError('Archived actor/trace mismatch')
    if sha(paths['reference'])!=report['motion_sha256']:raise ValueError('Car conditioning reference mismatch')
    h=np.load(paths['hockey'],allow_pickle=False);c=np.load(paths['car'],allow_pickle=False);ref=np.load(paths['reference'],allow_pickle=False)
    for key in ('root_pos','root_quat','root_com_vel','root_omega','joint_pos','joint_vel'):
        np.testing.assert_array_equal(h[key][0],c[key][0])
    # Build from the training reference, independently of recorded observations.
    count=len(h['observations']);phase=np.arange(count,dtype=np.float64)/(len(ref['joint_pos'])-1)*2*np.pi
    nominal=np.column_stack((ref['joint_pos'][:count,:6],ref['joint_vel'][:count,:6],np.sin(phase),np.cos(phase))).astype(np.float32)
    np.testing.assert_allclose(nominal,c['observations'][:count,:14],atol=4e-7,rtol=0)
    import torch
    torch.set_num_threads(2)
    actor=torch.jit.load(str(a.actor),map_location='cpu').eval()
    normalizer=dict(actor.actor.named_children())['0']
    buffers=dict(normalizer.named_buffers());eps=float(normalizer.eps)
    observations=torch.from_numpy(h['observations'].copy());carobs=torch.from_numpy(c['observations'].copy())
    slices={'reference_q':slice(0,6),'reference_dq':slice(6,12),'reference_q_dq':slice(0,12),'nominal_command':slice(0,14),'reference_phase':slice(12,14)}
    with torch.inference_mode():
        normalized=normalizer(observations)
        torch.testing.assert_close(normalized,(observations-buffers['_mean'])/(buffers['_std']+eps))
        raw=actor(observations);actual=raw.clamp(-1,1);car_action=actor(carobs).clamp(-1,1)
        replay=max(float(abs(actual-torch.from_numpy(h['actions'])).max()),float(abs(car_action-torch.from_numpy(c['actions'])).max()))
        if replay>1e-5:raise ValueError('CPU replay does not match archived actor actions')
        actions={};rows=[]
        for name,sl in slices.items():
            x=observations.clone();x[:,sl]=torch.from_numpy(nominal[:,sl]);y=actor(x).clamp(-1,1)
            actions[name]=y.numpy()
            rows.append(dict(intervention=name,first_action_error_vs_original_car=float(abs(y[0]-car_action[0]).max()),
                action_delta_vs_hockey_RMS_before_first_tilt=float((y[:14]-actual[:14]).square().mean().sqrt()),
                changed_input_columns=list(range(*sl.indices(51)))))
        carnorm=normalizer(carobs[:count]).numpy();norm=normalized.numpy()
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'interventions.npz',time_s=np.arange(count)*.02,
        actual_hockey_actions=actual.numpy(),unclipped_hockey_actions=raw.numpy(),original_car_actions=car_action[:count].numpy(),
        normalized_hockey_observations=norm,normalized_car_observations=carnorm,nominal_command=nominal,
        **{name+'_actions':v for name,v in actions.items()})
    np.savez_compressed(a.output/'nominal_conditioning.npz',command=nominal,
        original_reference_frames=np.array(len(ref['joint_pos'])),policy_dt_s=np.array(.02))
    records=[]
    for i in (0,1,2,5,10,15,50):
        records.append(dict(time_s=i*.02,ref_q_max_abs_normalized=float(abs(norm[i,:6]).max()),
            ref_dq_max_abs_normalized=float(abs(norm[i,6:12]).max()),unclipped_action_max=float(abs(raw[i].numpy()).max())))
    report=dict(kind='frozen_PPO_command_conditioning_intervention_audit',runner_sha256=sha(__file__),actor_sha256=sha(a.actor),
        input_sha256={k:sha(v) for k,v in paths.items()},physics_run=False,policy_weights_changed=False,
        normalizer_epsilon=eps,normalization_definition='(x-mean)/(std+eps); magnitude is NOT a number of standard deviations',
        original_reference_frames=len(ref['joint_pos']),hockey_reference_frames=count,
        phase_rate_multiplier=(len(ref['joint_pos'])-1)/(count-1),initial_physical_states_bit_identical=True,
        initial_observation_differing_columns=np.flatnonzero(h['observations'][0]!=c['observations'][0]).tolist(),
        maximum_cpu_action_replay_error=replay,cpu_torch_version=torch.__version__,interventions=rows,selected_times=records,
        inference='Reference-velocity input alone explains the first-action difference. Closed-loop stabilization by an adapter remains untested.',
        training_coverage_quantiles_available=False,dynamic_task_success_claimed=False,
        artifacts={k:sha(a.output/k) for k in ('interventions.npz','nominal_conditioning.npz')})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4),layout='constrained')
    t=np.arange(count)*.02
    for sl,title,ax in [(slice(0,6),'Reference position input',axes[0]),(slice(6,12),'Reference velocity input',axes[1])]:
        ax.plot(t,abs(norm[:,sl]).max(1),label='Hockey');ax.plot(t,abs(carnorm[:,sl]).max(1),label='Original car')
        ax.set(title=title,ylabel='Max abs normalized value');ax.legend(fontsize=8)
    axes[2].plot(t,abs(raw.numpy()).max(1),label='Hockey: raw actor output');axes[2].axhline(1,color='gray',ls='--',label='Action clip limit')
    axes[2].set(title='Recorded-state actor replay',ylabel='Max abs action');axes[2].legend(fontsize=8)
    for key in ('reference_dq','nominal_command','reference_phase'):
        axes[3].plot(t,np.sqrt(np.mean((actions[key]-actual.numpy())**2,axis=1)),label=key)
    axes[3].set(title='Fixed-state input interventions',ylabel='Action RMS change vs hockey');axes[3].legend(fontsize=7)
    for ax in axes:
        ax.set(xlim=(0,.3),xlabel='Time [s]');ax.axvline(.275,color='red',ls=':',lw=1);ax.grid(alpha=.2)
    fig.suptitle('CPU action diagnosis only: red line is first 45-degree tilt in the archived PhysX run')
    fig.savefig(a.output/'conditioning_4x1.png',dpi=150);plt.close(fig)
    report['artifacts']['conditioning_4x1.png']=sha(a.output/'conditioning_4x1.png')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

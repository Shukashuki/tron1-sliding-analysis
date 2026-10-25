"""Read-only sampling audit on completed, hash-verified nominal recordings.

Compare the existing bounded 50 Hz COM-vz reward signal with a hypothetical
mean of eight bounded 400 Hz costs. This is not a new reward evaluation,
training run, causal attribution or replacement of the frozen task gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_k_nonlinear_tire as evidence
from height_smoothness import acceleration_cost


def signals(vz, dt=.0025, scale=5.):
    vz=np.asarray(vz,dtype=float)
    if vz.shape!=(2793,) or not np.isfinite(vz).all():
        raise ValueError("Require all consecutive native COM-vz states")
    acceleration=np.diff(vz).reshape(349,8)/dt
    coarse=(vz[8::8]-vz[:-8:8])/(8*dt)
    np.testing.assert_allclose(coarse,acceleration.mean(-1),atol=1e-12,rtol=1e-12)
    coarse_cost=acceleration_cost(vz[8::8],vz[:-8:8],8*dt,scale,np.exp)
    fine_cost=acceleration_cost(vz[1:],vz[:-1],dt,scale,np.exp).reshape(349,8).mean(-1)
    square_mean=(acceleration**2).mean(-1)
    # The exact Jensen identity applies to the unbounded square, not to the
    # bounded exponential cost. Keep both rather than claiming an inequality.
    variance=((acceleration-coarse[:,None])**2).mean(-1)
    np.testing.assert_allclose(square_mean,coarse**2+variance,atol=1e-8,rtol=1e-12)
    return acceleration,coarse,coarse_cost,fine_cost,square_mean,variance


def run(args):
    if args.output_dir.exists():raise ValueError("Preserve previous diagnostic")
    root=args.result.resolve()
    reference=evidence.metrics.metrics.reference_data(root/'reference.npz')
    assert evidence.sha(root/'reference.npz')==evidence.REFERENCE_SHA
    rows,traces,_,proof,_=evidence.load_cases(root/'completed_evidence_v1',reference)
    manifest=evidence.read(root/'completed_evidence_v1/formal_v1/manifest.json')
    setting=manifest['environment']['height_regularization']
    assert setting['weight']==-.1 and setting['acceleration_scale_m_s2']==5.
    assert setting['smoothness_signal']=='base_Link COM world vz difference / control dt'
    assert evidence.sha(Path(__file__).with_name('height_smoothness.py'))==manifest['height_adapter_sha256']
    detail=[];plots=[]
    for row in rows:
        trace=traces[row['id']]
        acc,coarse,c50,c400,square,variance=signals(trace['com_velocity'][:,2])
        start=np.arange(349)*.02;end=start+.02
        valid=(start>=1.44-1e-9)&(end<=4.+1e-9)
        # First reward after reset has invalid history and is unpenalized;
        # core windows are later and contain no reset or randomized pushes.
        assert not valid[0] and valid.sum()==128
        selected_acc=acc[valid]
        ms=float(square[valid].mean());cs=float(np.mean(coarse[valid]**2))
        detail.append(dict(id=row['id'],recording_sha256=row['recording_sha256'],
            core_whole_control_windows=int(valid.sum()),
            core_400hz_acceleration_rms_m_s2=float(np.sqrt(ms)),
            core_50hz_interval_mean_acceleration_rms_m_s2=float(np.sqrt(cs)),
            core_within_interval_acceleration_variance_mean=float(variance[valid].mean()),
            core_acceleration_energy_missed_by_interval_mean_fraction=float(1-cs/ms) if ms else 0.,
            core_physical_abs_acceleration_p95_m_s2=float(np.percentile(abs(selected_acc),95)),
            core_mean_existing_bounded50hz_cost=float(c50[valid].mean()),
            core_mean_hypothetical_bounded400hz_cost=float(c400[valid].mean()),
            core_low_coarse_cost_lt0p1_but_fine_mean_gt0p5_windows=int(((c50<.1)&(c400>.5)&valid).sum()),
            existing_training_reward_or_acceptance_modified=False))
        plots.append((row['title'],start+.01,coarse,c50,c400,np.sqrt(square)))
    args.output_dir.mkdir(parents=True)
    obj=dict(kind='read_only_vertical_smoothing_sample_rate_diagnostic',
        existing_height_regularization=setting,formal_actor_sha256=proof['actor_sha256'],rows=detail,
        source_sha256=evidence.sha(__file__),height_source_sha256=evidence.sha(Path(__file__).with_name('height_smoothness.py')),
        nominal_analysis_sha256=evidence.sha(root/'analysis/analysis.json'),
        interpretation='Within-window acceleration cancellation is quantified. It does not prove reward sampling caused support gaps, or that a new 400Hz reward will improve learning. Bounded cost can saturate and need not obey Jensen.',
        physical_velocity='base_Link COM world vz, identical recorded Isaac getter root_lin_vel_w; MuJoCo COM convention retained by adapter',
        first_post_reset_reward_unpenalized=True,reset_or_push_crossing_in_core=False,
        gate_changes=False,new_training=False,task_success_claimed=False)
    (args.output_dir/'diagnosis.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(15,12),layout='constrained')
    for (name,time,coarse,c50,c400,rms),axis in zip(plots,axes):
        axis[0].plot(time,abs(coarse),label='absolute interval-mean acceleration')
        axis[0].plot(time,rms,label='RMS of eight actual accelerations',alpha=.8)
        axis[0].set_ylabel(name+'\nm/s²');axis[0].legend(fontsize=7)
        axis[1].plot(time,c50,label='existing 50 Hz bounded cost')
        axis[1].plot(time,c400,label='hypothetical mean of eight bounded costs',alpha=.8)
        axis[1].legend(fontsize=7)
        for a in axis:a.axvspan(1.44,4.,color='grey',alpha=.08);a.grid(alpha=.2);a.set_xlim(0,6.98)
    axes[-1,0].set_xlabel('time [s]');axes[-1,1].set_xlabel('time [s]')
    fig.suptitle('Read-only native acceleration sampling audit | original gates and training unchanged')
    fig.savefig(args.output_dir/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(detail,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())

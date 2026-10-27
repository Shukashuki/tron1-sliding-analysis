#!/usr/bin/env python3
"""Report GMR posture fidelity in a recorded PhysX hockey attempt, without rerun."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load,sha,imitation_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','qualification','command-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    qualification=json.loads((a.qualification/'report.json').read_text())
    if sha(a.run_dir/'final_rollout.npz')!=qualification['input_sha256']['final_rollout.npz']:raise ValueError('Trace/qualification mismatch')
    run=json.loads((a.run_dir/'report.json').read_text())
    if sha(a.command_dir/'motion.npz')!=run['motion_sha256']:raise ValueError('Reference mismatch')
    trace=load(a.run_dir/'final_rollout.npz');s=load(a.qualification/'signals.npz')
    command=json.loads((a.command_dir/'report.json').read_text())
    report,signals=imitation_metrics(trace,s,load(a.command_dir/'motion.npz'),command)
    report.update(analyzer_sha256=sha(__file__),metric_source_sha256=sha(Path(__file__).with_name('evaluate_hockey_physx.py')),
        trace_sha256=sha(a.run_dir/'final_rollout.npz'),qualification_sha256=sha(a.qualification/'report.json'),
        physics_rerun=False,new_hockey_physics_run=False)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'signals.npz',**signals)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4),layout='constrained');t=signals['time_s']
    axes[0].plot(t,signals['body_up_axis_error_deg']);axes[0].set(title='Body tilt-axis error',ylabel='degrees')
    axes[1].plot(t,np.sqrt(np.mean(signals['leg_joint_error_rad']**2,axis=-1)));axes[1].set(title='Six-leg-joint pose error',ylabel='RMS radians')
    for k in (0,1):
        mask=(s['normal_fz'][:,k]>5)&s['native_sample_valid'];ax=axes[k+2]
        ax.plot(t,signals['target_wheel_camber_deg'][:,k],color='gray',ls='--',label='GMR command')
        ax.plot(t,np.where(mask,signals['wheel_camber_deg'][:,k],np.nan),color=('#2469ba','#cb6a22')[k],label='PhysX, loaded')
        ax.set(title=('Left','Right')[k]+' wheel camber',ylabel='degrees',ylim=(-90,90));ax.legend(fontsize=8)
    for ax in axes:
        ax.set(xlabel='Time [s]',xlim=(0,2.76));ax.grid(alpha=.2)
        first=qualification['first_45deg_tilt_diagnostic_s']
        if first is not None:ax.axvline(first,color='red',ls=':',lw=1)
        ax.axvspan(command['brake_start_s'],command['brake_end_s'],color='gray',alpha=.06)
    fig.suptitle('Hockey GMR posture fidelity; height and continuous wheel spin are not tracking errors')
    fig.savefig(a.output/'posture_fidelity_4x1.png',dpi=150);plt.close(fig)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['windows'],indent=2))


if __name__=='__main__':main()

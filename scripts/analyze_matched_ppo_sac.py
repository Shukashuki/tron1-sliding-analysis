"""Same offline metrics for matched-task PPO/SAC, with contact kept explicit."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def geometric_contact(trace, shared):
    """Ideal-cylinder envelope only; not force-thresholded contact truth."""
    axle=shared.quaternion_rotate(trace['wheel_quaternion_wxyz'],np.array([0.,1.,0.]))
    az=np.clip(np.abs(axle[:,:,2]),0,1)
    extent=.127*np.sqrt(np.maximum(0,1-az**2))+.025*az
    gap=trace['wheel_pos'][:,:,2]-extent
    t=trace['time_s'];mask=(t>=1.44-1e-9)&(t<=4.+1e-9)
    dt=np.diff(t[mask]);g=gap[mask]
    def duty(v):return float(np.sum(v[:-1]*dt)/dt.sum())
    return dict(method='ideal R=0.127 m width=0.05 m cylinder envelope vs world Z=0; not cooked contact readback',
        native_force_threshold_used=False,core_gap_min_max_m=[float(g.min()),float(g.max())],
        core_any_penetration_over_5mm_fraction=duty((g<-.005).any(axis=1)),
        core_both_within_0p1mm_fraction=duty((g<=.0001).all(axis=1)),
        core_both_within_1mm_fraction=duty((g<=.001).all(axis=1)),
        core_both_within_2mm_fraction=duty((g<=.002).all(axis=1)))


def run(a):
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    sys.path.insert(0,str(a.validation_source/'scripts'))
    import analyze_ppo_backend_400hz_matrix as shared
    import ppo_backend_400hz_contract as contract
    protocol=a.validation_source/'config/ppo_backend_400hz_comparison_v1.json'
    # This validates all original PPO hashes and lifecycle proofs.
    ppo_nominal,reference,ppo=shared.load_matrix(a.ppo_root,a.reference,a.model,
        a.legacy_source/'scripts/eval_tracking_mujoco.py',protocol)
    fk,_=shared.load_fk(a.model,a.legacy_source/'scripts/eval_tracking_mujoco.py')
    cases=read(a.sac_root/'evaluation_cases.json')
    expected={(o,e,mu) for o in ('physx','mjlab') for e in ('physx','mujoco')
              for mu in read(protocol)['validation']['friction_grid']}
    formal=[c for c in cases if c['phase']=='formal']
    assert len(cases)==72 and len(formal)==68
    assert {(c['training_backend'],c['engine_tag'],c['mu']) for c in formal}==expected
    rows=[];nominal={('PPO',*k):v for k,v in ppo_nominal.items()}
    for row in ppo['rows']:
        rows.append(dict(algorithm='PPO',**row))
    for case in formal:
        folder=Path(case['directory'])
        report=read(folder/'report.json');audit=report['matched_sac_backend_evaluation']
        assert audit['status']=='verified' and audit['algorithm']=='SAC' and not audit['smoke_only']
        assert sha(folder/'report.json')==case['report_sha256']
        trace_path=folder/('physics_trajectory.npz' if case['engine_tag']=='physx' else 'rollout.npz')
        proof=contract.verify_recording(trace_path,audit['validation_backend'])
        assert proof['recording_sha256']==case['recording_sha256']==audit['recording_sha256']
        with np.load(trace_path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        trace,metrics=shared.analyze_case(raw,case['engine_tag'],reference,fk)
        rows.append(dict(algorithm='SAC',id=shared.case_key(case),training_backend=case['training_backend'],
            engine_tag=case['engine_tag'],validation_backend=audit['validation_backend'],mu=case['mu'],
            directory=str(folder),actor_sha256=case['actor_sha256'],input_hashes=dict(report=case['report_sha256'],
            recording=case['recording_sha256']),ideal_cylinder_contact=geometric_contact(trace,shared),**metrics))
        if case['mu']==.7:nominal[('SAC',case['training_backend'],case['engine_tag'])]=trace
    # Apply the same geometry diagnostics to PPO, without substituting measured
    # native force-bearing contacts with these analytic-envelope estimates.
    for row in rows:
        if row['algorithm']!='PPO':continue
        path=Path(row['directory'])/('physics_trajectory.npz' if row['engine_tag']=='physx' else 'rollout.npz')
        with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        trace,_=shared.analyze_case(raw,row['engine_tag'],reference,fk)
        row['ideal_cylinder_contact']=geometric_contact(trace,shared)
    table=[]
    for row in rows:
        compact=shared.compact_metrics({'rows':[row]})[0][0]
        table.append(dict(algorithm=row['algorithm'],**compact,
            ideal_bilateral_gap_within_1mm_fraction=row['ideal_cylinder_contact']['core_both_within_1mm_fraction']))
    a.output_dir.mkdir(parents=True)
    with (a.output_dir/'metric_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xy=np.concatenate([reference['position'][:,:2],*[v['position'][:,:2] for v in nominal.values()]])
    low,high=xy.min(axis=0)-.3,xy.max(axis=0)+.3
    for algorithm in ('PPO','SAC'):
        fig,axs=plt.subplots(1,4,figsize=(20,5),sharex=True,sharey=True)
        for ax,(origin,engine) in zip(axs,[(o,e) for o in ('physx','mjlab') for e in ('physx','mujoco')]):
            trace=nominal[(algorithm,origin,engine)];p=trace['position'];q=trace['quaternion']
            ax.plot(reference['position'][:,0],reference['position'][:,1],'--',color='gray',label='reference')
            ax.plot(p[:,0],p[:,1],color='#15956c',label='base path')
            indexes=np.arange(0,len(p),160)
            forward=shared.quaternion_rotate(q,np.array([1.,0,0]))
            axle=shared.quaternion_rotate(trace['wheel_quaternion_wxyz'],np.array([0.,1,0]))
            rolling=np.cross(axle,np.array([0.,0,1.]))
            for positions,vectors,color,label in [(p,forward,'#1464b4','body'),
                (trace['wheel_pos'][:,0],rolling[:,0],'#d62728','left wheel'),
                (trace['wheel_pos'][:,1],rolling[:,1],'#9467bd','right wheel')]:
                norm=np.linalg.norm(vectors[:,:2],axis=1);idx=indexes[norm[indexes]>1e-6]
                v=.16*vectors[idx,:2]/norm[idx,None]
                ax.quiver(positions[idx,0],positions[idx,1],v[:,0],v[:,1],color=color,label=label,
                    angles='xy',scale_units='xy',scale=1,width=.006)
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='X (m)',ylabel='Y (m)',
                title=f'{shared.TRAIN_LABELS[origin]} → {shared.ENGINE_LABELS[engine]}')
            ax.set_aspect('equal');ax.grid(alpha=.2)
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5)
        fig.suptitle(f'{algorithm} | matched task, 400 Hz | mu=0.7 | N=1 per cell')
        fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(a.output_dir/f'{algorithm.lower()}_nominal_4x1.png',dpi=150);plt.close(fig)
    report=dict(status='completed_verified',kind='matched_task_PPO_SAC_2x2x17',formal_episodes=136,
        seed=42,sample_size_per_cell=1,rows=rows,nominal_table=[r for r in table if r['mu']==.7],
        ppo_manifest_sha256=sha(a.ppo_root/'validation_matrix_manifest.json'),sac_cases_sha256=sha(a.sac_root/'evaluation_cases.json'),
        limitations=['Matched environment, reward, DR and transition budget; algorithm normalization/entropy/update rules remain explicitly different.',
            'CPU MuJoCo evaluation must not be called MuJoCo-Warp evaluation.',
            'Thresholded native force counts measure load eligibility, not geometric contact truth.',
            'Ideal-cylinder envelope is a separate geometric diagnostic, not an exact cooked PhysX contact measurement.',
            'Single-seed descriptive results do not establish statistical success probabilities or sim2real accuracy.'])
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sac-root','ppo-root','validation-source','reference','model','legacy-source','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

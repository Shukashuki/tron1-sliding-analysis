"""Read scalar curves from completed contact-search logs without weights.

Episode statistics are the runner's logged values, not independently measured
task success, physical contact fractions, or evidence of convergence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

FIELDS = ['Mean reward', 'Mean episode length', 'Mean action noise std',
          'Episode_Reward/contact_search_contact_fraction',
          'Episode_Reward/contact_search_contact_loss',
          'Episode_Reward/contact_search_clearance',
          'Episode_Reward/contact_search_sliding',
          'Episode_Reward/contact_search_angular_acceleration',
          'Episode_Reward/hockey_stop_hold', 'Episode_Reward/hockey_xy',
          'Episode_Reward/hockey_velocity', 'Episode_Termination/hockey_fall']

REMOTE_READER = r'''
import hashlib,json,re,time
from pathlib import Path
root=Path('/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1')
bundle=root/('source_snapshot_'+VERSION);run=root/('run_'+VERSION)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(bundle/'manifest.json')==MANIFEST_SHA
plan=json.loads((bundle/'plan.json').read_text());rows=[]
for cell in plan['cells']:
    assert re.fullmatch(r'[a-zA-Z0-9_]+',cell['id'])
    report_path=run/cell['id']/'formal/report.json'
    if not report_path.is_file():continue
    report=json.loads(report_path.read_text());cfg=json.loads((bundle/cell['protocol']).read_text())
    assert report['status']=='training_completed_not_task_success'
    assert report['config_sha256']==sha(bundle/cell['protocol'])
    iterations=cfg['formal']['iterations'];per_iteration=cfg['formal']['num_envs']*cfg['ppo']['steps_per_env']
    assert report['transitions']==iterations*per_iteration
    log_path=run/cell['id']/'formal.log';payload=log_path.read_bytes()
    text=re.sub(r'\x1b\[[0-9;]*m','',payload.decode())
    headers=list(re.finditer(r'Learning iteration\s+(\d+)/(\d+)',text));points=[]
    assert [int(h[1]) for h in headers]==list(range(iterations))
    assert all(int(h[2])==iterations for h in headers)
    for i,h in enumerate(headers):
        block=text[h.end():headers[i+1].start() if i+1<len(headers) else len(text)]
        values={}
        for line in block.splitlines():
            label,separator,value=line.strip().partition(':')
            if separator and label in FIELDS:
                assert label not in values
                values[label]=float(value.strip())
        points.append(dict(iteration=i,transitions=(i+1)*per_iteration,values=values))
    rows.append(dict(cell=cell['id'],protocol_sha256=sha(bundle/cell['protocol']),
        training_report_sha256=sha(report_path),log_sha256=hashlib.sha256(payload).hexdigest(),
        log_bytes=len(payload),completed_iterations=iterations,points=points))
print(json.dumps(dict(observed_unix_s=time.time(),source_manifest_sha256=MANIFEST_SHA,cases=rows),allow_nan=False))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True,help='Local completed report/recording subset')
    p.add_argument('--version',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not re.fullmatch(r'v[1-9][0-9]*',a.version):raise ValueError('Invalid study version')
    if a.bundle.name!='source_snapshot_'+a.version:raise ValueError('Snapshot version mismatch')
    if a.output.exists():raise FileExistsError(a.output)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_sha=sha(a.bundle/'manifest.json')
    code='VERSION='+repr(a.version)+'\nMANIFEST_SHA='+repr(manifest_sha)+'\nFIELDS='+repr(FIELDS)+'\n'+REMOTE_READER
    result=subprocess.run(['/home/roy/.codex/skills/remote-compute-ssh/scripts/remote_compute.sh',
        'exec','4090d','--','python3','-'],input=code,text=True,capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError('Remote curve read failed: '+result.stderr[-2000:])
    curves=json.loads(result.stdout)
    if not 1<=len(curves['cases'])<=3:
        raise ValueError('This view requires one fixed reward and up to three completed model arms')
    if any(row['completed_iterations']<100 for row in curves['cases']):
        raise ValueError('Need at least 100 completed iterations for adjacent 50-iteration windows')
    import numpy as np
    summaries=[]
    for row in curves['cases']:
        report=a.evidence/('run_'+a.version)/row['cell']/'formal/report.json'
        if sha(report)!=row['training_report_sha256']:raise ValueError('Local/remote training identity differs')
        metrics={}
        for field in FIELDS:
            values=np.asarray([p['values'].get(field,np.nan) for p in row['points']],dtype=float)
            if not np.isfinite(values[-100:]).all():raise ValueError('Missing late training statistics: '+field)
            metrics[field]=dict(previous_50_iterations_mean=float(values[-100:-50].mean()),
                last_50_iterations_mean=float(values[-50:].mean()),
                last_50_minus_previous_50=float(values[-50:].mean()-values[-100:-50].mean()),
                last_50_iterations_std=float(values[-50:].std()))
        summaries.append(dict(cell=row['cell'],metrics=metrics))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4.5),constrained_layout=True)
    plotted=['Mean reward','Episode_Reward/contact_search_contact_fraction',
             'Episode_Reward/hockey_stop_hold','Episode_Reward/contact_search_angular_acceleration']
    labels=['Logged mean reward','Logged contact-fraction reward','Logged stop reward','Logged angular penalty']
    colors=['#377eb8','#e67923','#36986a']
    for row,color in zip(curves['cases'],colors):
        x=np.asarray([p['transitions'] for p in row['points']])/1e6
        for ax,field,label in zip(axes,plotted,labels):
            y=np.asarray([p['values'].get(field,np.nan) for p in row['points']])
            ax.plot(x,y,color=color,alpha=.2,lw=.65)
            ax.plot(x[24:],np.convolve(y,np.ones(25)/25,mode='valid'),color=color,lw=1.4,
                    label=row['cell'].split('_smooth')[0])
            ax.set(xlabel='New training transitions (millions)',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Completed fixed-budget screens | raw + trailing 25-iteration mean | not task success')
    a.output.mkdir(parents=True)
    fig.savefig(a.output/'training_curves_4x1.png',dpi=140);plt.close(fig)
    (a.output/'curves.json').write_text(json.dumps(curves,indent=2,allow_nan=False)+'\n')
    report=dict(kind='hockey_logged_training_curve_audit_v1',script_sha256=sha(Path(__file__)),
        source_manifest_sha256=manifest_sha,observed_unix_s=curves['observed_unix_s'],
        physics_steps=0,training_changed=False,task_success_claimed=False,convergence_claimed=False,
        note='Runner episode averages mix randomized phases and materials; they are not physical contact percentages or independent success measurements. Changes between final 50-iteration windows are descriptive, not significance tests.',
        cases=summaries,artifacts={name:sha(a.output/name) for name in ('curves.json','training_curves_4x1.png')})
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries))


if __name__=='__main__':main()

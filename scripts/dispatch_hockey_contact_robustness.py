"""Run the frozen 33-case suite only after a verified nominal screen pass.

No training, reward changes, policy selection or retries. Requalify the
original recording on CPU before allocating one available GPU for the suite.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready_candidate(bundle, screen_root, plan):
    read=lambda path:json.loads(path.read_text())
    state=read(screen_root/'status.json')
    if (state['status']!='nominal_pass_requires_robustness_suite'
        or state.get('current') is not None or state['manifest_sha256']!=sha(bundle/'manifest.json')):
        raise ValueError('Require a terminal nominal-pass screening state')
    cells=[cell for cell in plan['cells'] if cell['id']==state.get('candidate')]
    if len(cells)!=1:raise ValueError('Candidate must identify exactly one declared cell')
    cell=cells[0];directory=screen_root/cell['id'];protocol=bundle/cell['protocol']
    cfg=read(protocol);screen=read(directory/'nominal_qualification.json')
    training=read(directory/'formal/report.json');run=read(directory/'nominal/report.json')
    actor=directory/'formal/actor_normalized.pt'
    q=screen['qualification'];budget=cfg['formal']['num_envs']*cfg['formal']['iterations']*cfg['ppo']['steps_per_env']
    if not q['task_success_claimed'] or not all(q['gates'].values()):
        raise ValueError('Candidate has an unpassed physical success gate')
    if training['status']!='training_completed_not_task_success' or training['transitions']!=budget:
        raise ValueError('Candidate training budget is incomplete')
    if training['warm_start']['checkpoint_sha256']!=cfg['warm_start']['checkpoint_sha256']:
        raise ValueError('Candidate used a different parent')
    if not (training['actor_sha256']==screen['actor_sha256']==run['actor_sha256']==sha(actor)):
        raise ValueError('Candidate actor identity differs')
    if (screen['source_manifest_sha256']!=sha(bundle/'manifest.json')
        or screen['protocol_sha256']!=sha(protocol)
        or screen['run_report_sha256']!=sha(directory/'nominal/report.json')):
        raise ValueError('Screen qualification has different provenance')
    for name,digest in run['artifacts'].items():
        if sha(directory/'nominal'/name)!=digest:raise ValueError('Candidate recording changed')
    return cell,cfg,directory,protocol,actor


def configure_gpu_discovery(helper):
    """The shared helper filters query_gpus by its module-level GPU_ORDER."""
    order=(7,0,1,4,6,2,5,3)
    helper.GPU_ORDER=order
    return order


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','screen-root','prior-launch','audit-helper','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:raise ValueError('Require unmasked GPU identities')
    # This launcher is an auxiliary file; never add it to or rewrite the
    # already-running snapshot. Load its scheduling helpers from that snapshot.
    sys.path.insert(0,str(a.bundle.resolve()))
    from dispatch_hockey_contact_search import inspect,write
    from dispatch_hockey_bilateral_capacity import probe_available_device
    for name in ('dispatch_hockey_contact_search','dispatch_hockey_bilateral_capacity'):
        if Path(sys.modules[name].__file__).resolve().parent!=a.bundle.resolve():
            raise ValueError('Scheduling helper escaped frozen snapshot')
    plan,paths,python=inspect(a.bundle,a.prior_launch,a.audit_helper)
    cell,cfg,directory,protocol,actor=ready_candidate(a.bundle,a.screen_root,plan)
    if not (a.bundle/'analyze_hockey_contact_robustness.py').is_file():
        raise ValueError('Frozen bundle must include the independent robustness qualifier')
    if a.output.exists():raise FileExistsError('Never duplicate a robustness run')
    a.output.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('gpu_audit',a.audit_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    order=configure_gpu_discovery(helper)
    state=dict(status='preparing',candidate=cell['id'],completed=[],current=None,idle_wait_s=0,
        source_manifest_sha256=sha(a.bundle/'manifest.json'),dispatcher_sha256=sha(__file__),
        original_nominal_qualification_sha256=sha(directory/'nominal_qualification.json'),
        actor_sha256=sha(actor),started_unix_s=time.time(),training_jobs_started=0,goal_success_claimed=False)
    def save():
        state['updated_unix_s']=time.time();write(a.output/'status.json',state)
    save()

    def run(argv,label,gpu_needed):
        inspect(a.bundle,a.prior_launch,a.audit_helper)
        ready_candidate(a.bundle,a.screen_root,plan)
        gpu=None
        if gpu_needed:
            while gpu is None:
                for index in order:
                    gpu=probe_available_device(helper,index,plan['dependencies'])
                    if gpu is not None:break
                if gpu is None:
                    state.update(status='waiting_for_idle_device',current={'stage':label});save()
                    if state['idle_wait_s']>=plan['idle_wait_limit_s']:raise RuntimeError('Capacity wait exhausted')
                    time.sleep(30);state['idle_wait_s']+=30
            argv+=['--device','cuda:'+str(gpu['physical_index'])]
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2')
        if not gpu_needed:env['CUDA_VISIBLE_DEVICES']=''
        with (a.output/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,env=env,cwd=a.output,stdout=log,stderr=subprocess.STDOUT)
        row=dict(stage=label,pid=child.pid,argv=argv,identity=helper.process_snapshot(child.pid),
            gpu_before_launch=gpu,started_unix_s=time.time())
        state.update(status='running',current=row);save()
        code=child.wait();row.update(returncode=code,finished_unix_s=time.time())
        if code:raise RuntimeError(label+' failed; no automatic retry')
        state['completed'].append(row);state['current']=None;save()

    try:
        reference=Path(plan['reference_directory'])
        recheck=a.output/'nominal_requalification.json'
        run([python,'-B',str(a.bundle/'qualify_hockey_contact_search.py'),
            '--bundle',str(a.bundle),'--protocol',str(protocol),'--run',str(directory/'nominal'),
            '--actor',str(actor),'--reference',str(reference),'--output',str(recheck)],'nominal_requalification',False)
        q=json.loads(recheck.read_text())['qualification']
        if not q['task_success_claimed'] or not all(q['gates'].values()):
            raise ValueError('Independent nominal requalification failed before GPU suite')
        argv=[python,'-B','-u',str(a.bundle/'train_hockey_contact_search.py'),
            '--protocol',str(protocol),'--motion',str(reference/'motion.npz'),'--variant','tracking',
            '--mode','eval-suite','--actor',str(actor),'--output-dir',str(a.output/'eval_suite')]
        for key,value in paths.items():argv+=['--'+key,str(value)]
        run(argv,'eval_suite',True)
        run([python,'-B','-u',str(a.bundle/'analyze_hockey_contact_robustness.py'),
            '--bundle',str(a.bundle),'--protocol',str(protocol),'--training-dir',str(directory/'formal'),
            '--suite',str(a.output/'eval_suite'),'--actor',str(actor),'--reference',str(reference),
            '--screen-qualification',str(directory/'nominal_qualification.json'),
            '--output',str(a.output/'qualification')],'qualification',False)
        report=json.loads((a.output/'qualification/report.json').read_text())
        if report['perturbation_trials']!=32:raise ValueError('Incomplete robustness qualification')
        state.update(status=('completed_all_cases_pass_requires_goal_audit' if report['all_tested_cases_pass']
            else 'completed_robustness_has_failures_requires_followup'),
            qualification_sha256=sha(a.output/'qualification/report.json'),
            nominal_success=report['nominal_success'],perturbation_successes=report['perturbation_successes'])
        save()
    except BaseException as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()

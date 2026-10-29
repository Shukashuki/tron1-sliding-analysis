"""Serial bounded reward screens. No old jobs resumed; never displace a GPU job."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

from evaluate_hockey_physx import sha
from dispatch_hockey_bilateral_capacity import probe_available_device


def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def verify_predecessor(plan):
    required=plan.get('required_completed_predecessor')
    if required is None:return
    status=json.loads(Path(required['status_path']).read_text())
    screens=status.get('screens',[])
    if (status['manifest_sha256']!=required['manifest_sha256']
        or status['status']!='screens_completed_requires_next_reward_or_longer_training'
        or status.get('current') is not None
        or len(screens)!=required['completed_screens']
        or len({row['id'] for row in screens})!=len(screens)
        or any(row['qualification']['task_success_claimed'] for row in screens)):
        raise ValueError('Predecessor sweep must finish with all declared screens unsuccessful before refinement')


def inspect(bundle, prior_launch, audit_helper):
    read=lambda path:json.loads(path.read_text())
    for row in read(bundle/'manifest.json')['files']:
        path=(bundle/row['path']).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path)!=row['sha256']:
            raise ValueError('Snapshot changed')
        if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
    if sha(audit_helper)!='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866':
        raise ValueError('GPU audit helper changed')
    plan=read(bundle/'plan.json')
    verify_predecessor(plan)
    ref=Path(plan['reference_directory'])
    for file,digest in plan['reference_sha256'].items():
        if sha(ref/file)!=digest:raise ValueError('Reference changed')
    for key in ('checkpoint','actor'):
        if sha(plan['warm_start'][key])!=plan['warm_start'][key+'_sha256']:
            raise ValueError('Parent policy changed')
    old=read(prior_launch)['argv'];get=lambda key:Path(old[old.index('--'+key)+1])
    paths={key:get(key) for key in ('asset','tracking-root','beyondmimic-root')}
    cfg=read(bundle/plan['cells'][0]['protocol'])
    if sha(paths['asset'])!=cfg['asset_sha256'] or sha(paths['tracking-root']/'training/tron1_tracking.py')!=cfg['tracking_sha256']:
        raise ValueError('Runtime/asset identity changed')
    for key,digest in read(bundle/'upstream_provenance.json').items():
        if key.endswith('scripts/train_tracking.py'):continue
        prefix='upstream/whole_body_tracking/' if key.startswith('upstream/whole_body_tracking/') else 'upstream/tron1-learning-from-human/'
        root=paths['beyondmimic-root' if 'whole_body_tracking' in prefix else 'tracking-root']
        if sha(root/key[len(prefix):])!=digest:raise ValueError('Upstream dependency changed')
    return plan,paths,old[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','prior-launch','audit-helper','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:raise ValueError('Require unmasked GPU identities')
    plan,paths,python=inspect(a.bundle,a.prior_launch,a.audit_helper)
    if a.preflight_only:
        print(json.dumps(dict(status='sources_assets_parent_verified_no_GPU_job_started',cells=len(plan['cells']))));return
    if a.output.exists():raise FileExistsError('Never duplicate a run')
    a.output.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('gpu_audit',a.audit_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.GPU_ORDER=(7,0,1,4,6,2,5,3)
    state=dict(status='preparing',completed=[],current=None,idle_wait_s=0,
        manifest_sha256=sha(a.bundle/'manifest.json'),started_unix_s=time.time(),old_jobs_resumed=False,
        goal_success_claimed=False)
    def save():
        state['updated_unix_s']=time.time();write(a.output/'status.json',state)
    save()

    def run(argv, label, output, gpu_needed):
        inspect(a.bundle,a.prior_launch,a.audit_helper)
        gpu=None
        if gpu_needed:
            while gpu is None:
                for index in helper.GPU_ORDER:
                    gpu=probe_available_device(helper,index,plan['dependencies'])
                    if gpu is not None:break
                if gpu is None:
                    state.update(status='waiting_for_idle_device',current={'stage':label});save()
                    if state['idle_wait_s']>=plan['idle_wait_limit_s']:raise RuntimeError('Capacity wait exhausted')
                    time.sleep(30);state['idle_wait_s']+=30
            argv+=['--device','cuda:'+str(gpu['physical_index'])]
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2')
        if not gpu_needed:env['CUDA_VISIBLE_DEVICES']=''
        with output.open('xb') as log:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=a.output)
        row=dict(stage=label,pid=process.pid,argv=argv,identity=helper.process_snapshot(process.pid),
            gpu_before_launch=gpu,started_unix_s=time.time())
        state.update(status='running',current=row);save()
        code=process.wait();row.update(returncode=code,finished_unix_s=time.time())
        if code:raise RuntimeError(label+' failed; stop without automatic retries')
        state['completed'].append(row);state['current']=None;save()

    try:
        seen=set()
        for cell in plan['cells']:
            folder=a.output/cell['id'];folder.mkdir()
            protocol=a.bundle/cell['protocol']
            cfg=json.loads(protocol.read_text())
            base=[python,'-B','-u',str(a.bundle/'train_hockey_contact_search.py'),
                '--protocol',str(protocol),'--motion',str(Path(plan['reference_directory'])/'motion.npz'),'--variant','tracking']
            for key,value in paths.items():base+=['--'+key,str(value)]

            def evaluate(label, actor):
                run(base+['--mode','eval','--actor',str(actor),'--output-dir',str(folder/label)],
                    cell['id']+'/'+label,folder/(label+'.log'),True)
                qualification=folder/(label+'_qualification.json')
                run([python,'-B',str(a.bundle/'qualify_hockey_contact_search.py'),'--bundle',str(a.bundle),
                    '--protocol',str(protocol),'--run',str(folder/label),'--actor',str(actor),
                    '--reference',plan['reference_directory'],'--output',str(qualification)],
                    cell['id']+'/'+label+'_qualification',folder/(label+'_qualification.log'),False)
                return json.loads(qualification.read_text())['qualification']

            if cell['model'] not in seen:
                evaluate('parent_transfer',Path(plan['warm_start']['actor']))
                seen.add(cell['model'])
            for mode in ('smoke','formal'):
                run(base+['--mode',mode,'--output-dir',str(folder/mode)],cell['id']+'/'+mode,folder/(mode+'.log'),True)
                report=json.loads((folder/mode/'report.json').read_text())
                expected=cfg[mode]['num_envs']*cfg['ppo']['steps_per_env']*cfg[mode]['iterations']
                if report['status']!='training_completed_not_task_success' or report['transitions']!=expected:
                    raise ValueError('Training did not complete the declared screen')
                if report['bilateral_audit']['fault_seen']:raise ValueError('Native collector fault')
            q=evaluate('nominal',folder/'formal/actor_normalized.pt')
            state.setdefault('screens',[]).append(dict(id=cell['id'],qualification=q));save()
            if q['task_success_claimed'] and plan['stop_after_nominal_pass']:
                state.update(status='nominal_pass_requires_robustness_suite',candidate=cell['id']);save();return
        state['status']='screens_completed_requires_next_reward_or_longer_training';save()
    except BaseException as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()

"""Run the same retraining now on a GPU with no active workload.

Only small inactive contexts from identity-verified existing hockey jobs on
OTHER devices are allowed. No foreign or unrecognized process is accepted.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import hashlib

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def inspect(bundle,prior_launch,audit_helper):
    read=lambda p:json.loads(Path(p).read_text())
    manifest=read(bundle/'manifest.json');plan=read(bundle/'plan.json');cfg=read(bundle/'protocol.json')
    for row in manifest['files']:
        path=(bundle/row['path']).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path)!=row['sha256']:raise ValueError('Snapshot changed')
        if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
    assert (plan['formal_training_runs'],plan['transitions'],plan['evaluation_episodes'])==(1,12288000,33)
    assert cfg['kind']=='hockey_ppo_bilateral_sliding_v1' and cfg['speed_scale']==1. and cfg['seed']==42
    reference=Path(plan['reference_directory'])
    for file,digest in plan['reused_reference_sha256'].items():assert sha(reference/file)==digest
    old=read(prior_launch)['argv'];get=lambda key:old[old.index('--'+key)+1]
    paths={key:Path(get(key)) for key in ('asset','tracking-root','beyondmimic-root')}
    assert sha(paths['asset'])==cfg['asset_sha256']
    assert sha(paths['tracking-root']/'training/tron1_tracking.py')==cfg['tracking_sha256']
    for key,digest in read(bundle/'upstream_provenance.json').items():
        if key.endswith('scripts/train_tracking.py'):continue
        prefix='upstream/whole_body_tracking/' if key.startswith('upstream/whole_body_tracking/') else 'upstream/tron1-learning-from-human/'
        root=paths['beyondmimic-root' if 'whole_body_tracking' in prefix else 'tracking-root']
        assert sha(root/key[len(prefix):])==digest,key
    assert sha(audit_helper)=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
    assert importlib.metadata.version('rsl-rl-lib')=='3.0.1'
    for module in ('torch','scipy','numpy'):importlib.metadata.version(module)
    return plan,cfg,paths,old[0]



def context_eligibility(gpu, trusted_jobs, process_probe, uid):
    """A low utilization percentage alone never licenses a different job."""
    if gpu['utilization_percent'] != 0 or gpu['memory_used_mib'] >= 2048:
        return False, []
    accepted=[]
    for process in gpu['all_reported_compute_and_graphics_processes']:
        pid=int(process['pid'])
        job=trusted_jobs.get(pid)
        if job is None:return False, []
        current=process_probe(pid)
        if current is None or current['uid'] != uid or current['argv'] != job['argv']:
            return False, []
        identity=job.get('identity') or {}
        if identity.get('start_ticks') != current['start_ticks']:return False, []
        argv=current['argv']
        if '--device' not in argv or argv.count('--device') != 1:return False, []
        device=argv[argv.index('--device')+1]
        if device == 'cuda:'+str(gpu['physical_index']):return False, []
        if not device.startswith('cuda:') or int(device.split(':')[1]) not in range(8):return False, []
        script=next((v for v in argv if v.endswith('/train_hockey_ppo.py')),None)
        allowed_prefixes=('/data/zengxiong/workspaces/tron1-hockey-speed-success-20261028-v1/source_snapshot_v1/',
                          '/data/zengxiong/workspaces/tron1-hockey-friction-boundary-20261028-v1/source_snapshot_v1/')
        if script is None or not script.startswith(allowed_prefixes):return False, []
        try:used=int(process['used_memory'].split()[0])
        except (KeyError,ValueError):return False, []
        if used >= 768:return False, []
        accepted.append(dict(pid=pid,main_device=device,context_memory_mib=used,
                             uid=current['uid'],start_ticks=current['start_ticks']))
    return True, accepted


def probe_available_device(helper, index, dependencies):
    samples=[]
    for sample in range(3):
        trusted={}
        for dependency in dependencies:
            path=Path(dependency['path'])
            if not path.exists():continue
            status=json.loads(path.read_text())
            if status.get('manifest_sha256') != dependency['manifest_sha256']:raise ValueError('Unrecognized existing study')
            job=status.get('current')
            if status.get('status')=='running' and job is not None:trusted[job['pid']]=job
        gpu=next(g for g in helper.query_gpus() if g['physical_index']==index)
        ok,contexts=context_eligibility(gpu,trusted,helper.process_snapshot,os.getuid())
        if not ok:return None
        samples.append(dict(gpu=gpu,inactive_owned_contexts=contexts))
        if sample<2:time.sleep(1)
    if len({s['gpu']['uuid'] for s in samples})!=1:raise ValueError('GPU identity changed')
    return dict(samples[-1]['gpu'],capacity_eligible=True,scheduling_samples=samples,
                reason='No active workload; zero utilization, less than 2 GiB used, only verified small contexts of owned hockey jobs on other devices')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','prior-launch','audit-helper','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--physical-gpu',type=int,choices=(0,1,4),default=1)
    a=p.parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:raise ValueError('Require unmasked GPU identities')
    plan,cfg,paths,python=inspect(a.bundle,a.prior_launch,a.audit_helper)
    if a.preflight_only:
        print(json.dumps(dict(status='runtime_and_sources_verified_no_GPU_allocated',manifest_sha256=sha(a.bundle/'manifest.json'),reference_reused=True,python=python)));return
    if a.output.exists():raise FileExistsError('Never duplicate an experiment')
    if shutil.disk_usage(a.output.parent).free<10*1024**3:raise ValueError('Need at least 10 GiB')
    a.output.mkdir(parents=True)
    state=dict(status='preparing_immediate_retraining',started_unix_s=time.time(),completed=[],current=None,
               manifest_sha256=sha(a.bundle/'manifest.json'),dispatcher_sha256=sha(__file__),physical_gpu=a.physical_gpu,gpu_jobs_started=0,idle_wait_s=0.)
    def save():state['updated_unix_s']=time.time();write(a.output/'status.json',state)
    save()
    try:
        # The user requested using currently available 4090D capacity. The old
        # waiting dispatcher is superseded separately after identity checks.
        state['dependency_wait_bypassed_by_user']=True;save()
        module=importlib.util.spec_from_file_location('gpu_audit',a.audit_helper)
        helper=importlib.util.module_from_spec(module);module.loader.exec_module(helper)
        helper.GPU_ORDER=(a.physical_gpu,)
        reference=Path(plan['reference_directory'])
        for stage in ('smoke','formal','eval-suite','analyze'):
            inspect(a.bundle,a.prior_launch,a.audit_helper)
            gpu=None
            if stage!='analyze':
                while True:
                    gpu=probe_available_device(helper,a.physical_gpu,plan['dependencies'])
                    if gpu is not None:break
                    state.update(status='waiting_for_device_without_active_workload',stage=stage);save()
                    if state['idle_wait_s']>=plan['idle_wait_limit_s']:raise RuntimeError('Six-hour capacity wait exhausted')
                    time.sleep(30);state['idle_wait_s']+=30
                argv=[python,'-B','-u',str(a.bundle/'train_hockey_bilateral.py'),'--protocol',str(a.bundle/'protocol.json'),
                      '--motion',str(reference/'motion.npz'),'--mode',stage,'--variant','tracking','--device','cuda:'+str(gpu['physical_index']),
                      '--output-dir',str(a.output/stage)]
                for key,value in paths.items():argv.extend(['--'+key,str(value)])
                if stage=='eval-suite':argv.extend(['--actor',str(a.output/'formal/actor_normalized.pt')])
                env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID')
                state['gpu_jobs_started']+=1
            else:
                argv=[python,'-B','-u',str(a.bundle/'analyze_hockey_bilateral.py'),'--bundle',str(a.bundle),
                      '--protocol',str(a.bundle/'protocol.json'),'--command-dir',str(reference),
                      '--training-dir',str(a.output/'formal'),'--suite',str(a.output/'eval-suite'),
                      '--actor',str(a.output/'formal/actor_normalized.pt'),'--output',str(a.output/'analyze')]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
            with (a.output/(stage+'.log')).open('xb') as log:
                child=subprocess.Popen(argv,cwd=a.output,env=env,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=stage,pid=child.pid,identity=helper.process_snapshot(child.pid),argv=argv,gpu_before_launch=gpu,started_unix_s=time.time())
            state.update(status='running',current=row);save()
            code=child.wait();row.update(returncode=code,finished_unix_s=time.time())
            if code:raise RuntimeError(stage+' failed; no automatic retry or further GPU stages')
            report=json.loads((a.output/stage/'report.json').read_text())
            if stage in ('smoke','formal'):
                assert report['transitions']==(1536 if stage=='smoke' else 12288000)
                audit=report['bilateral_audit'];assert audit['geometry_samples']>0 and not audit['fault_seen']
                assert audit['native_vs_pre_geometry_max_error_m']<=cfg['bilateral_sliding']['geometry_alignment_max_error_m']
            if stage=='eval-suite':assert len(report['cases'])==33
            if stage=='analyze':assert report['perturbation_trials']==32
            row['report_sha256']=sha(a.output/stage/'report.json');state['completed'].append(row);state['current']=None;save()
        state['status']='completed_training_and_qualification';save()
    except BaseException as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()

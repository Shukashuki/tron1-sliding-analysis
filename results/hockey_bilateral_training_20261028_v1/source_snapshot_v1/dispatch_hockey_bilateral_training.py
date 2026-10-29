"""Serial, bounded cold-start retraining after the already authorized studies."""
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

from evaluate_hockey_physx import sha


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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','prior-launch','audit-helper','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:raise ValueError('Require unmasked GPU identities')
    plan,cfg,paths,python=inspect(a.bundle,a.prior_launch,a.audit_helper)
    if a.preflight_only:
        print(json.dumps(dict(status='runtime_and_sources_verified_no_GPU_allocated',manifest_sha256=sha(a.bundle/'manifest.json'),reference_reused=True,python=python)));return
    if a.output.exists():raise FileExistsError('Never duplicate an experiment')
    if shutil.disk_usage(a.output.parent).free<10*1024**3:raise ValueError('Need at least 10 GiB')
    a.output.mkdir(parents=True)
    state=dict(status='waiting_for_existing_studies',started_unix_s=time.time(),completed=[],current=None,
               manifest_sha256=sha(a.bundle/'manifest.json'),gpu_jobs_started=0,idle_wait_s=0.)
    def save():state['updated_unix_s']=time.time();write(a.output/'status.json',state)
    save()
    try:
        for dependency in plan['dependencies']:
            path=Path(dependency['path'])
            while True:
                wrapper=Path(plan['boundary_wrapper_status'])
                if wrapper.exists() and json.loads(wrapper.read_text())['status']=='stopped':raise RuntimeError('Existing boundary queue stopped; new experiment not started')
                parent=json.loads(path.read_text()) if path.exists() else None
                if parent is not None:
                    assert parent['manifest_sha256']==dependency['manifest_sha256']
                    if parent['status']=='stopped':raise RuntimeError('Existing dependency stopped; no retry')
                    if parent['status']==dependency['expected_status']:
                        assert len(parent['completed'])==dependency['expected_stages'] and parent.get('current') is None
                        break
                state.update(status='waiting_for_existing_studies',dependency=str(path),dependency_stage=parent.get('current') if parent else None);save()
                if time.time()-state['started_unix_s']>plan['dependency_wait_limit_s']:raise RuntimeError('24-hour dependency wait expired')
                time.sleep(30)
        module=importlib.util.spec_from_file_location('gpu_audit',a.audit_helper)
        helper=importlib.util.module_from_spec(module);module.loader.exec_module(helper)
        helper.GPU_ORDER=(3,5,4,0,1,2,6,7)
        reference=Path(plan['reference_directory'])
        for stage in ('smoke','formal','eval-suite','analyze'):
            inspect(a.bundle,a.prior_launch,a.audit_helper)
            gpu=None
            if stage!='analyze':
                while True:
                    gpu=next((g for g in helper.query_gpus() if g['idle']),None)
                    if gpu is not None:break
                    state.update(status='waiting_for_completely_idle_RTX',stage=stage);save()
                    if state['idle_wait_s']>=plan['idle_wait_limit_s']:raise RuntimeError('Six-hour idle wait exhausted')
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

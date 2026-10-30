"""Serial fixed-policy material probes, then the matched split-friction continuation.

No material is changed midway through a rollout. Static/dynamic switching is
left to native PhysX. Each GPU allocation uses the existing identity checks.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe-bundle','continuation-bundle','prior-launch','audit-helper','root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    a=parser.parse_args()
    sys.path.insert(0,str(a.probe_bundle.resolve()))
    from dispatch_hockey_contact_search import inspect,write
    from dispatch_hockey_bilateral_capacity import probe_available_device
    from evaluate_hockey_physx import sha
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Unmasked physical GPU identities required')
    plan,paths,python=inspect(a.probe_bundle,a.prior_launch,a.audit_helper)
    inspect(a.continuation_bundle,a.prior_launch,a.audit_helper)
    study=json.loads((a.probe_bundle/'material_study.json').read_text())
    actor=Path(study['actor'])
    if sha(actor)!=study['actor_sha256']:
        raise ValueError('Frozen successful policy changed')
    expected=[(.7,.7),(.9,.7),(.9,.9)]
    if len(plan['cells'])!=3 or study['continuation_manifest_sha256']!=sha(a.continuation_bundle/'manifest.json'):
        raise ValueError('Study identity changed')
    for cell,(ms,md) in zip(plan['cells'],expected):
        c=json.loads((a.probe_bundle/cell['protocol']).read_text())
        if abs(c['evaluation_mu']-md)>1e-12 or abs(c['evaluation_mu']*c['static_dynamic_friction_ratio']-ms)>1e-12:
            raise ValueError('Material pair changed')
    if a.preflight_only:
        print(json.dumps(dict(status='sources_assets_actor_and_continuation_verified',gpu_jobs_started=0)))
        return
    out=a.root/'static_dynamic_pipeline_v1';probes=a.root/'run_v10'
    for path in (out,probes,a.root/'run_v9',a.root/'robustness_v9_v1'):
        if path.exists():raise FileExistsError(path)
    out.mkdir();probes.mkdir()
    spec=importlib.util.spec_from_file_location('gpu_audit',a.audit_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.GPU_ORDER=(7,0,1,4,6,2,5,3)
    state=dict(status='preparing',completed=[],current=None,idle_wait_s=0,
        manifest_sha256=sha(a.probe_bundle/'manifest.json'),actor_sha256=sha(actor),
        started_unix_s=time.time(),screens=[],training_jobs_started_by_probe=0)
    def save():
        state['updated_unix_s']=time.time();write(out/'status.json',state)
        write(probes/'status.json',state)
    save()
    def run(argv,label,gpu_needed=False):
        inspect(a.probe_bundle,a.prior_launch,a.audit_helper)
        inspect(a.continuation_bundle,a.prior_launch,a.audit_helper)
        if sha(actor)!=study['actor_sha256']:raise ValueError('Actor changed')
        gpu=None
        if gpu_needed:
            while gpu is None:
                for index in helper.GPU_ORDER:
                    gpu=probe_available_device(helper,index,plan['dependencies'])
                    if gpu is not None:break
                if gpu is None:
                    state.update(status='waiting_for_available_device',current=dict(stage=label));save()
                    if state['idle_wait_s']>=plan['idle_wait_limit_s']:raise RuntimeError('Capacity wait exhausted')
                    time.sleep(30);state['idle_wait_s']+=30
            argv+=['--device','cuda:'+str(gpu['physical_index'])]
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2')
        # Nested dispatchers must see physical devices but allocate only in their own checked stages.
        with (out/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT)
        row=dict(stage=label,pid=child.pid,argv=argv,identity=helper.process_snapshot(child.pid),
                 gpu_before_launch=gpu,started_unix_s=time.time())
        state.update(status='running',current=row);save()
        code=child.wait();row.update(returncode=code,finished_unix_s=time.time())
        if code:raise RuntimeError(label+' failed; no automatic retry')
        state['completed'].append(row);state['current']=None;save()
    try:
        for cell in plan['cells']:
            folder=probes/cell['id'];folder.mkdir();protocol=a.probe_bundle/cell['protocol']
            argv=[python,'-B','-u',str(a.probe_bundle/'train_hockey_contact_search.py'),
                '--protocol',str(protocol),'--motion',str(Path(plan['reference_directory'])/'motion.npz'),
                '--variant','tracking','--mode','eval','--actor',str(actor),'--output-dir',str(folder/'nominal')]
            for key,path in paths.items():argv+=['--'+key,str(path)]
            run(argv,cell['id']+'_eval',True)
            result=folder/'nominal_qualification.json'
            run([python,'-B',str(a.probe_bundle/'qualify_hockey_contact_search.py'),'--bundle',str(a.probe_bundle),
                 '--protocol',str(protocol),'--run',str(folder/'nominal'),'--actor',str(actor),
                 '--reference',plan['reference_directory'],'--output',str(result)],cell['id']+'_qualify')
            q=json.loads(result.read_text())['qualification']
            state['screens'].append(dict(id=cell['id'],qualification=q));save()
        # Recheck actual state arrays: a material comparison must start in the same state.
        import numpy as np
        anchor=None;errors={}
        for cell in plan['cells']:
            with np.load(probes/cell['id']/'nominal/final_rollout.npz',allow_pickle=False) as trace:
                initial={k:trace[k][0] for k in ('root_pos','root_quat','root_com_vel','root_omega','joint_pos','joint_vel')}
            if anchor is None:anchor=initial
            error=max(float(np.max(abs(initial[k]-anchor[k]))) for k in initial)
            if error>5e-5:raise ValueError('Initial state differs across material probes')
            errors[cell['id']]=error
        write(probes/'initial_state_audit.json',dict(maximum_state_errors=errors,all_same_within_5e_5=True))
        # Verify the repeated 0.7 control against its already-qualified nominal recording.
        previous=np.load(study['baseline_recording'],allow_pickle=False)
        current=np.load(probes/plan['cells'][0]['id']/'nominal/final_rollout.npz',allow_pickle=False)
        control_error={k:float(np.max(abs(previous[k]-current[k]))) for k in
                       ('root_pos','root_quat','root_com_vel','joint_pos','actions')}
        previous.close();current.close()
        write(probes/'baseline_reproduction.json',dict(maximum_errors=control_error,
            identical_within_5e_5=all(x<=5e-5 for x in control_error.values())))
        if not state['screens'][0]['qualification']['task_success_claimed'] or any(x>5e-5 for x in control_error.values()):
            raise ValueError('Repeated successful baseline differs; diagnose before training')
        state['material_probes_completed']=True;save()
        argv=[python,'-B','-u',str(a.continuation_bundle/'dispatch_hockey_contact_search.py'),
            '--bundle',str(a.continuation_bundle),'--prior-launch',str(a.prior_launch),
            '--audit-helper',str(a.audit_helper),'--output',str(a.root/'run_v9')]
        run(argv,'matched_split_continuation')
        completed=json.loads((a.root/'run_v9/status.json').read_text())
        if completed['status']=='nominal_pass_requires_robustness_suite':
            run([python,'-B','-u',str(a.probe_bundle/'dispatch_hockey_contact_robustness.py'),
                 '--bundle',str(a.continuation_bundle),'--screen-root',str(a.root/'run_v9'),
                 '--prior-launch',str(a.prior_launch),'--audit-helper',str(a.audit_helper),
                 '--output',str(a.root/'robustness_v9_v1')],'split_robustness')
        state.update(status='completed_requires_comparison_analysis',current=None,
            continuation_status=completed['status']);save()
    except BaseException as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()

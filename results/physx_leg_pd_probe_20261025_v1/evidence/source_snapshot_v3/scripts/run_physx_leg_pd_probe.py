"""Two declared K gain transfers, serial single worlds on fully idle GPUs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def run(args):
    root=args.workspace.resolve();snapshot=args.snapshot.resolve()
    assert args.run_id.startswith('episodes_') and args.run_id.replace('_','').isalnum()
    out=root/args.run_id;out.mkdir(exist_ok=False)
    state=dict(kind='frozen_K_explicit_leg_PD_transfers',status='preflight',cases=[],additional_training_updates=0,task_success_claimed=False)
    write(out/'status.json',state)
    try:
        assert sha(args.idle_audit_source)=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
        spec=importlib.util.spec_from_file_location('_PD_idle_audit',args.idle_audit_source)
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        for pid in (528887,528956,529379,530031,547996,550510,587951,588047,594180,594183):
            p=audit.process_snapshot(pid);assert p is None or p['state']=='Z',('Prior owned job still live',p)
        manifest=json.loads((root/(snapshot.name+'.manifest.json')).read_text())
        ledger={str(snapshot/r['path']):r['sha256'] for r in manifest['files']}
        assert all(sha(p)==s for p,s in ledger.items())
        protocol=snapshot/'config/physx_leg_pd_probe_v1.json'
        settings=json.loads(protocol.read_text());assert list(settings['profiles'])==['baseline_500_10','soft_40_2p5']
        template=json.loads((snapshot/'config/physx_leg_pd_probe_launch_template.json').read_text())['argv']
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
            TRON1_BEYONDMIMIC_ROOT='/data/zengxiong/workspaces/tron1-beyondmimic-20261002/third_party/whole_body_tracking')
        for profile in settings['profiles']:
            state['status']='waiting_for_fully_idle_GPU_'+profile;write(out/'status.json',state)
            while True:
                gpus=audit.query_gpus();selected=next((g for g in gpus if g['idle']),None)
                if selected:break
                time.sleep(15.)
            assert all(sha(p)==s for p,s in ledger.items())
            command=template[:];command[0]=str(args.python);command[3]=str(snapshot/'scripts/eval_physx_leg_pd_probe.py')
            for flag,value in {'--gain-profile':profile,'--gain-protocol':str(protocol),
                    '--output-dir':str(out/profile),'--physical-gpu-index':str(selected['physical_index'])}.items():
                assert command.count(flag)==1;command[command.index(flag)+1]=value
            with (out/(profile+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT)
                row=dict(id=profile,pid=child.pid,argv=command,status='running',gpu_audit=gpus,selected_gpu=selected,
                    started_unix_time=time.time(),process_at_launch=audit.process_snapshot(child.pid))
                state['cases'].append(row);state['status']='running_'+profile;write(out/'status.json',state)
                while child.poll() is None:time.sleep(2.)
            row['returncode']=child.returncode
            assert child.returncode==0,('Gain transfer failed, no retry',profile,child.returncode)
            proof=json.loads((out/profile/'leg_pd_status.json').read_text())
            assert proof['kind']=='frozen_K_PhysX_explicit_leg_PD_transfer'
            assert proof['native_states']==2793 and proof['policy_decisions']==349
            pd=proof['explicit_leg_PD_intervention'];assert pd['physical_motor_computations']==2792
            assert pd['actual_gain_readback']==dict(kp_range=[settings['profiles'][profile][0]]*2,kd_range=[settings['profiles'][profile][1]]*2)
            for name in ('physics_trajectory','tire_trajectory','leg_pd_trajectory'):
                assert sha(out/profile/(name+'.npz'))==proof[name+'_sha256']
            row.update(status='completed_verified',proof_sha256=sha(out/profile/'leg_pd_status.json'))
            write(out/'status.json',state)
        state['status']='both_explicit_PD_transfers_completed_require_analysis';write(out/'status.json',state)
    except BaseException as error:
        state.update(status='failed_no_retry_or_process_signal',error=repr(error));write(out/'status.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('workspace','snapshot','idle-audit-source','python'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',default='episodes_v1')
    run(p.parse_args())

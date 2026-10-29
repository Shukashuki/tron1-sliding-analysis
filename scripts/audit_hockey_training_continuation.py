"""Read-only CPU checkpoint restoration check on the approved experiment host.

No simulator, environment step, optimizer update, model export or GPU job.
Only scalar diagnostics and hashes return; weights remain on the remote host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


REMOTE = r'''
import contextlib,copy,hashlib,importlib.metadata,importlib.util,io,json,time,types
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
root=Path('/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1')
bundle=root/('source_snapshot_'+VERSION);run=root/('run_'+VERSION)/CASE
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(bundle/'manifest.json')==EXPECTED_MANIFEST
for row in json.loads((bundle/'manifest.json').read_text())['files']:
    p=(bundle/row['path']).resolve()
    assert p.is_relative_to(bundle.resolve()) and sha(p)==row['sha256']
assert sha(run/'formal/report.json')==EXPECTED_REPORT
report=json.loads((run/'formal/report.json').read_text())
plan=json.loads((bundle/'plan.json').read_text())
cell=next(c for c in plan['cells'] if c['id']==CASE)
protocol_path=bundle/cell['protocol'];source=json.loads(protocol_path.read_text())
assert sha(protocol_path)==report['config_sha256']
assert sha(run/'formal/actor_normalized.pt')==report['actor_sha256']
runner_path=Path(importlib.util.find_spec('rsl_rl.runners.on_policy_runner').origin)
assert sha(runner_path)=='e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5'
assert importlib.metadata.version('rsl-rl-lib')=='3.0.1'
spec=importlib.util.spec_from_file_location('frozen_trainer',bundle/'train_hockey_contact_search.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
module=types.ModuleType('continuation_under_audit')
exec(compile(HELPER_SOURCE,'hockey_training_continuation.py','exec'),module.__dict__)
candidate=copy.deepcopy(source)
candidate['formal']['iterations']=1  # Schema probe only; learn() is never called.
candidate['initialization']='optimizer_continuation'
candidate['warm_start']=dict(checkpoint=str(run/'formal/model_final.pt'),
    checkpoint_sha256=report['checkpoint_sha256'],actor=str(run/'formal/actor_normalized.pt'),
    actor_sha256=report['actor_sha256'])
candidate['training_continuation']=dict(kind='hockey_optimizer_continuation_v1',
    source_report=str(run/'formal/report.json'),source_report_sha256=EXPECTED_REPORT,
    source_protocol=str(protocol_path),source_protocol_sha256=sha(protocol_path))
class ShapeOnlyEnvironment:
    num_envs=2
    num_actions=8
    device='cpu'
    def get_observations(self):
        return TensorDict(dict(policy=torch.zeros(2,source['actor_dim']),
            critic=torch.zeros(2,source['critic_dim'])),batch_size=[2])
    def step(self,*args,**kwargs):
        raise AssertionError('Audit must not step an environment')
with contextlib.redirect_stdout(io.StringIO()):
    runner=OnPolicyRunner(ShapeOnlyEnvironment(),trainer.runner_config(source),device='cpu')
    assert not runner.is_distributed
    restored=module.initialize_continuation(runner,candidate)
runner.alg.policy.eval()
actor=torch.jit.load(str(run/'formal/actor_normalized.pt'),map_location='cpu').eval()
probe=torch.linspace(-3,3,7*source['actor_dim']).reshape(7,source['actor_dim'])
with torch.inference_mode():
    actual=runner.alg.policy.act_inference({'policy':probe})
    expected=actor(probe)
assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
error=float((actual-expected).abs().max());assert error<=1e-6
assert not torch.cuda.is_initialized()
print(json.dumps(dict(kind='hockey_optimizer_continuation_cpu_audit_v1',observed_unix_s=time.time(),
    source_case=CASE,source_manifest_sha256=EXPECTED_MANIFEST,source_report_sha256=EXPECTED_REPORT,
    helper_sha256=hashlib.sha256(HELPER_SOURCE.encode()).hexdigest(),
    frozen_trainer_sha256=sha(bundle/'train_hockey_contact_search.py'),
    installed_runner_sha256=sha(runner_path),rsl_rl_version=importlib.metadata.version('rsl-rl-lib'),
    torch_version=str(torch.__version__),restoration=restored,exported_actor_sha256=report['actor_sha256'],
    maximum_action_difference=error,probe_observations=7,optimizer_parameter_states=len(runner.alg.optimizer.state),
    physics_steps=0,training_iterations=0,optimizer_updates=0,cuda_initialized=False,
    weights_transferred=False,remote_files_written=False,task_success_claimed=False),allow_nan=False))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',required=True)
    parser.add_argument('--case',required=True)
    for key in ('bundle','evidence','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'v[1-9][0-9]*',args.version) or not re.fullmatch(r'[a-zA-Z0-9_]+',args.case):
        raise ValueError('Invalid source identifiers')
    if args.bundle.name!='source_snapshot_'+args.version:
        raise ValueError('Snapshot version mismatch')
    if args.output.exists():raise FileExistsError(args.output)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    helper=Path(__file__).with_name('hockey_training_continuation.py')
    expected_report=sha(args.evidence/('run_'+args.version)/args.case/'formal/report.json')
    values=dict(VERSION=args.version,CASE=args.case,EXPECTED_MANIFEST=sha(args.bundle/'manifest.json'),
                EXPECTED_REPORT=expected_report,HELPER_SOURCE=helper.read_text())
    program='\n'.join(key+'='+repr(value) for key,value in values.items())+'\n'+REMOTE
    result=subprocess.run(['/home/roy/.codex/skills/remote-compute-ssh/scripts/remote_compute.sh',
        'exec','4090d','--','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2',
        '/data/zengxiong/.conda/envs/regrind/bin/python','-B','-'],
        input=program,text=True,capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError('Remote CPU audit failed: '+result.stderr[-3000:])
    report=json.loads(result.stdout)
    if report['helper_sha256']!=sha(helper):raise ValueError('Executed helper differs')
    report['audit_script_sha256']=sha(Path(__file__))
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()

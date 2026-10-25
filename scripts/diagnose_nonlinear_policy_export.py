"""Inspect our completed checkpoint/export on CPU and one freshly idle GPU.

No training, simulator, parameter update, package install or process signal.
The owned checkpoint is read with weights_only=True; weights stay on host.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') is None
    assert sha(a.idle_audit_source)=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
    spec=importlib.util.spec_from_file_location('_export_diagnostic_GPU_audit',a.idle_audit_source)
    idle=importlib.util.module_from_spec(spec);spec.loader.exec_module(idle)
    for pid in (625241,625713,638442):
        p=idle.process_snapshot(pid);assert p is None or p['state']=='Z'
    gpus=idle.query_gpus();selected=next((g for g in gpus if g['idle']),None)
    assert selected is not None,'No fully idle approved GPU; do not share or retry automatically'
    import torch
    from rsl_rl.modules import ActorCritic
    torch.cuda.set_device(selected['physical_index'])
    root=a.training_dir;cfg=json.loads((root/'runner_config.json').read_text())['policy'];cfg.pop('class_name')
    checkpoint=torch.load(root/'model_final.pt',map_location='cpu',weights_only=True)
    model=ActorCritic({'policy':torch.zeros(2,51),'critic':torch.zeros(2,120)},
                      {'policy':['policy'],'critic':['critic']},8,**cfg)
    model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    actor=torch.jit.load(str(root/'policy.pt'),map_location='cpu').eval()
    state=actor.state_dict()
    for name,value in checkpoint['model_state_dict'].items():
        if name.startswith('actor_obs_normalizer.'):
            assert torch.equal(state['0.'+name.split('actor_obs_normalizer.',1)[1]],value)
        if name.startswith('actor.'):
            assert torch.equal(state['1.'+name.split('actor.',1)[1]],value)
    probe=torch.linspace(-2.,2.,51).repeat(2,1)
    with torch.inference_mode():
        cpu=model.act_inference({'policy':probe});export=actor(probe)
    assert torch.equal(cpu,export) and torch.isfinite(cpu).all()
    device='cuda:'+str(selected['physical_index']);gpu=copy.deepcopy(model).to(device)
    original=dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32,precision=torch.get_float32_matmul_precision())
    tests=[]
    try:
        for tf32 in (True,False):
            torch.backends.cuda.matmul.allow_tf32=tf32
            with torch.inference_mode():value=gpu.act_inference({'policy':probe.to(device)}).cpu()
            delta=(value-cpu).abs()
            tests.append(dict(allow_tf32=tf32,max_abs_error=float(delta.max()),
                old_allclose_rtol1e5_atol1e6=bool(torch.allclose(value,cpu,rtol=1e-5,atol=1e-6)),
                predeclared_max_abs_error1e4_pass=bool(delta.max()<=1e-4),
                max_clipped_action_error=float((value.clamp(-1,1)-cpu.clamp(-1,1)).abs().max()),
                gpu_outputs=value.tolist()))
    finally:
        torch.set_float32_matmul_precision(original['precision'])
        torch.backends.cuda.matmul.allow_tf32=original['allow_tf32']
    result=dict(kind='owned_nonlinear_actor_export_precision_diagnostic',checkpoint_iteration=checkpoint['iter'],
        checkpoint_sha256=sha(root/'model_final.pt'),actor_sha256=sha(root/'policy.pt'),
        CPU_module_and_TorchScript_bit_identical=True,export_all_weights_and_normalizer_equal_checkpoint=True,
        original_precision=original,gpu_tests=tests,selected_gpu=selected,full_GPU_audit=gpus,
        CPU_outputs=cpu.tolist(),normalizer_samples=int(model.actor_obs_normalizer.count),
        checkpoint_all_tensors_finite=all(bool(torch.isfinite(v).all()) for v in checkpoint['model_state_dict'].values()),
        source_sha256=sha(__file__),training_or_simulation_executed=False,task_success_claimed=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('training-dir','idle-audit-source','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

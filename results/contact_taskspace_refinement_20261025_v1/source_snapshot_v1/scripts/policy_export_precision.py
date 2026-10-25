"""Independent CPU export identity plus bounded, explicit full-FP32 GPU check.

CPU TorchScript is compared with separate CPU copies of the source modules,
never with itself. The existing absolute 1e-4 contract cap also applies to a
GPU reference with TF32 disabled. All original precision flags are restored.
Training and physics precision settings are not changed.
"""
from __future__ import annotations

import copy


def verify_export(policy, exported, probe, *, absolute_gpu_tolerance=1e-4):
    import torch
    assert absolute_gpu_tolerance==1e-4, "Use the existing declared export absolute cap"
    normalizer=copy.deepcopy(policy.actor_obs_normalizer).cpu().eval()
    actor=copy.deepcopy(policy.actor).cpu().eval()
    saved=exported.state_dict()
    expected={**{'0.'+k:v for k,v in normalizer.state_dict().items()},
              **{'1.'+k:v for k,v in actor.state_dict().items()}}
    assert set(saved)==set(expected)
    assert all(torch.equal(saved[k].cpu(),v) for k,v in expected.items()),"Exported weights or normalizer changed"
    assert all(bool(torch.isfinite(v).all()) for v in expected.values())
    device=next(policy.actor.parameters()).device
    original_tf32=torch.backends.cuda.matmul.allow_tf32
    original_precision=torch.get_float32_matmul_precision()
    try:
        with torch.inference_mode():
            reference=actor(normalizer(probe.cpu()))
            observed=exported(probe.cpu())
            assert torch.equal(reference,observed),"CPU source-module/export mismatch"
            torch.backends.cuda.matmul.allow_tf32=False
            gpu_reference=policy.act_inference({'policy':probe.to(device)}).cpu()
            delta=(gpu_reference-reference).abs()
            assert bool(torch.isfinite(gpu_reference).all())
            assert float(delta.max())<=absolute_gpu_tolerance,"Full-FP32 GPU/export difference exceeds declared cap"
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32=original_tf32
    result=dict(kind='independent_CPU_export_and_full_FP32_GPU_verification',
        exported_weights_and_normalizer_bit_identical=True,CPU_source_module_export_bit_identical=True,
        CPU_max_abs_error=0.,GPU_reference_device=str(device),GPU_reference_TF32=False,
        GPU_FP32_max_abs_error=float(delta.max()),GPU_FP32_absolute_tolerance=absolute_gpu_tolerance,
        GPU_FP32_max_clipped_action_difference=float((gpu_reference.clamp(-1,1)-reference.clamp(-1,1)).abs().max()),
        original_precision=dict(allow_tf32=original_tf32,matmul_precision=original_precision),
        original_precision_restored=(torch.backends.cuda.matmul.allow_tf32==original_tf32
                                     and torch.get_float32_matmul_precision()==original_precision),
        training_precision_or_policy_parameters_changed=False,
        parent_verifier_reference='independent_CPU_source_modules_not_export_itself')
    assert result['original_precision_restored']
    return reference,result

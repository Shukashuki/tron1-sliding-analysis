"""CPU fixtures for independent export identity and precision restoration."""
from pathlib import Path
import copy
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from policy_export_precision import verify_export


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_obs_normalizer=torch.nn.LayerNorm(51)
        self.actor=torch.nn.Sequential(torch.nn.Linear(51,16),torch.nn.ELU(),torch.nn.Linear(16,8))

    def act_inference(self,obs):return self.actor(self.actor_obs_normalizer(obs['policy']))


def setup():
    torch.manual_seed(13);policy=Policy().eval()
    export=torch.jit.script(torch.nn.Sequential(copy.deepcopy(policy.actor_obs_normalizer),copy.deepcopy(policy.actor)).eval())
    return policy,export,torch.linspace(-2,2,51).repeat(2,1)


@pytest.mark.parametrize('tf32',[False,True])
def test_independent_CPU_reference_exact_identity_and_precision_restored(tf32):
    original=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=tf32
        policy,export,probe=setup()
        before={k:v.clone() for k,v in policy.state_dict().items()}
        reference,audit=verify_export(policy,export,probe)
        assert audit['CPU_source_module_export_bit_identical'] and audit['CPU_max_abs_error']==0
        assert audit['GPU_FP32_absolute_tolerance']==1e-4 and audit['original_precision_restored']
        assert torch.backends.cuda.matmul.allow_tf32==tf32
        assert all(torch.equal(policy.state_dict()[k],v) for k,v in before.items())
        assert torch.equal(export(probe),reference)
    finally:torch.backends.cuda.matmul.allow_tf32=original


def test_modified_exported_parameter_is_rejected():
    policy,export,probe=setup()
    with torch.no_grad():next(export.parameters()).add_(1.)
    with pytest.raises(AssertionError,match='weights or normalizer'):verify_export(policy,export,probe)


def test_bad_device_reference_is_rejected_and_flags_restored():
    policy,export,probe=setup()
    policy.act_inference=lambda obs:torch.full((2,8),float('nan'))
    original=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        with pytest.raises(AssertionError):verify_export(policy,export,probe)
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:torch.backends.cuda.matmul.allow_tf32=original

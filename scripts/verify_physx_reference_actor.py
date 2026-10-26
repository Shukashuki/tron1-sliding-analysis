"""Verify a cached export against every recorded PPO observation/action on CPU."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def run(a):
    import numpy as np
    import torch
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert not a.output.exists()
    cfg=json.loads(a.config.read_text());report=json.loads(a.report.read_text())
    assert sha(a.actor)==cfg['actor_sha256']
    assert sha(a.trace)==report['physics_trajectory_sha256']
    assert report['policy_export']['exported_policy_sha256']==cfg['training_export_actor_sha256']
    actor=torch.jit.load(str(a.actor),map_location='cpu').eval()
    with np.load(a.trace,allow_pickle=False) as z:
        observations=z['action_observation'][1::8,0]
        actions=z['actions_raw'][1::8,0]
    assert observations.shape==(349,51) and actions.shape==(349,8)
    with torch.inference_mode():predicted=actor(torch.tensor(observations,dtype=torch.float32)).numpy()
    error=float(np.max(abs(predicted-actions)))
    assert np.isfinite(error) and error<1e-5
    result=dict(kind='cached_PPO_export_recorded_observation_check',status='passed',
        actor_sha256=sha(a.actor),original_training_export_sha256=cfg['training_export_actor_sha256'],
        source_trace_sha256=sha(a.trace),source_report_sha256=sha(a.report),source_sha256=sha(__file__),
        observations=349,observation_dim=51,actions=8,maximum_raw_action_error=error,tolerance=1e-5,
        normalization_applied_by='cached actor exactly once',device='cpu',
        equivalence_scope='349 recorded observations only; file-byte identity NOT asserted')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('actor','trace','report','config','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())

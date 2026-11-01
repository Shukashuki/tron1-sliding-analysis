"""Independently qualify a nominal screen using unchanged full-horizon gates."""
import argparse
import json
from pathlib import Path
import numpy as np

from evaluate_hockey_physx import load, sha
from analyze_hockey_ppo import audit_observations
from analyze_hockey_bilateral import qualify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','protocol','run','actor','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda path:json.loads(path.read_text())
    for row in read(a.bundle/'manifest.json')['files']:
        path=(a.bundle/row['path']).resolve()
        if not path.is_relative_to(a.bundle.resolve()) or sha(path)!=row['sha256']:
            raise ValueError('Source snapshot changed')
    cfg=read(a.protocol); run=read(a.run/'report.json')
    for key,path in [('source_sha256',a.bundle/'train_hockey_contact_search.py'),
        ('task_source_sha256',a.bundle/'hockey_ppo_task.py'),('bilateral_source_sha256',a.bundle/'hockey_bilateral_task.py'),
        ('reward_search_source_sha256',a.bundle/'hockey_contact_search.py'),
        ('config_sha256',a.protocol),('motion_sha256',a.reference/'motion.npz'),('actor_sha256',a.actor)]:
        if run[key]!=sha(path):raise ValueError('Run/source identity mismatch: '+key)
    if (run['worlds'],run['steps'],run['external_forces'],run['mode'])!=(1,139,0,'eval'):
        raise ValueError('Require full single-world native evaluation')
    for name,digest in run['artifacts'].items():
        if sha(a.run/name)!=digest:raise ValueError('Rollout artifact changed')
    mu=cfg['evaluation_mu']; mus=mu*cfg['static_dynamic_friction_ratio']
    material=run['material_audit']
    if not material['live_physx_materials_verified'] or not material['ground_material_usd_verified']:
        raise ValueError('Materials unverified')
    np.testing.assert_allclose(material['wheel_mu_range'],[mu,mu],rtol=0,atol=1e-6)
    np.testing.assert_allclose(material.get('wheel_mu_static_range',material['wheel_mu_range']),[mus,mus],rtol=0,atol=1e-6)
    if not all(g['friction']==1 and g['combine']=='multiply' for g in material['ground_colliders']):
        raise ValueError('Unexpected ground material')
    trace=load(a.run/'final_rollout.npz'); raw=load(a.run/'native_contact_buffers.npz')
    if trace['root_pos'].shape!=(1113,3) or trace['actions'].shape!=(139,8):
        raise ValueError('Incomplete horizon')
    reference=load(a.reference/'motion.npz'); command=read(a.reference/'report.json')
    obs=audit_observations(trace,reference,cfg)
    import torch
    torch.set_num_threads(2)
    actor=torch.jit.load(str(a.actor),map_location='cpu').eval()
    with torch.inference_mode():
        predicted=actor(torch.as_tensor(trace['observations'],dtype=torch.float32)).clamp(-1,1).numpy()
    error=float(np.max(np.abs(predicted-trace['actions'])))
    if not np.isfinite(error) or error>5e-5:raise ValueError('Actor action replay differs')
    result=dict(qualification=qualify(trace,raw,reference,command,mu=mu,mu_static=mus),
        observation_audit=obs,action_replay_max_error=error,actor_sha256=sha(a.actor),
        protocol_sha256=sha(a.protocol),run_report_sha256=sha(a.run/'report.json'),
        source_manifest_sha256=sha(a.bundle/'manifest.json'),
        criterion='Unchanged bilateral contact first, then full hockey task and force-corroborated sliding',
        screen_only=True,robustness_cases_not_yet_run=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    q=result['qualification']
    print(json.dumps({k:q[k] for k in ('task_success_claimed','failed_gates','xy_RMSE_m','first_45deg_tilt_s')}))


if __name__=='__main__':main()

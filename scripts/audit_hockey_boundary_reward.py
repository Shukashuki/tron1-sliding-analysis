"""Replay the candidate objective on recorded PhysX data; this is not training."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch

from evaluate_hockey_physx import sha, load
from analyze_hockey_prefall_friction import read_case
from hockey_friction_boundary import boundary_sample, boundary_objective


def replay(cell, case, reference, mg, cfg):
    audited = read_case(cell, case)
    trace=load(cell/'eval-suite'/case/'final_rollout.npz')
    raw=load(cell/'eval-suite'/case/'native_contact_buffers.npz')
    count=len(raw['physical_sample_indices'])
    if count!=1112 or not np.array_equal(raw['physical_sample_indices'],np.arange(1,1113)):
        raise ValueError('Require the complete recorded 400 Hz horizon')
    tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float64)
    sides=[]
    for side,name in enumerate(('wheel_L_Link','wheel_R_Link')):
        b=trace['body_names'].astype(str).tolist().index(name)
        def buffers(kind,size):
            out=[]
            for i in range(size):
                v=raw[f'w{side}_{kind}_buffer_{i}']
                if i>=size-2:
                    v=v.copy()
                    if i==size-1:
                        v += (np.arange(count)*128).reshape(count,1,1)
                    out.append(torch.as_tensor(v.reshape(count,1)))
                else:
                    out.append(tensor(v.reshape(-1,v.shape[-1])))
            return out
        out=boundary_sample(buffers('normal',6),buffers('friction',4),
            tensor(trace['body_com_pos'][1:,b]+trace['environment_origin_w']),
            tensor(trace['body_com_vel'][1:,b]),tensor(trace['body_omega'][1:,b]),
            tensor(trace['body_quat'][1:,b]),torch.full((count,),.7,dtype=torch.float64))
        if bool(out.pop('fault').any()):
            raise ValueError('Invalid replay contact sample')
        sides.append(out)
    v={k:torch.stack([s[k] for s in sides],1) for k in sides[0]}
    mapping={'normal_fz':'normal_fz','lateral_slip_rms':'normal_point_lateral_rms',
        'utilization_proxy':'aggregate_utilization','friction_anchor_power':'anchor_power',
        'friction_force_w':'friction_force','opposing_sliding_anchors':'dissipative_slipping_anchor_count'}
    errors={}
    for k, archived in mapping.items():
        expected=audited['signals'][archived][1:]
        error=float(np.max(abs(v[k].numpy()-expected)))
        errors[k]=error
        tolerance={'lateral_slip_rms':4e-5,'friction_anchor_power':.005,
                   'friction_force_w':.001,'normal_fz':.001,'utilization_proxy':1e-4}.get(k,0.)
        # Independent diagnostic marks unloaded rho as NaN; native reducer uses zero.
        if k=='utilization_proxy':
            expected=np.where(v['normal_fz'].numpy()>5,expected,0.)
        if not np.allclose(v[k].numpy(),expected,atol=tolerance,rtol=1e-6,equal_nan=False):
            raise ValueError(f'Boundary sampler differs from independent reconstruction: {k}, max error {error}')
    t=np.arange(1,count+1)*.0025
    rt=np.arange(len(reference['joint_pos']))*.02
    interp=lambda field:np.column_stack([np.interp(t,rt,field[:,j]) for j in range(2)])
    target_p=interp(reference['body_pos_w'][:,0,:2]);target_v=interp(reference['body_lin_vel_w'][:,0,:2])
    v.update(root_velocity_xy=tensor(trace['root_com_vel'][1:,:2]),
        xy_error_sq=tensor(np.sum((trace['root_pos'][1:,:2]-target_p)**2,-1)),
        velocity_error_sq=tensor(np.sum((trace['root_com_vel'][1:,:2]-target_v)**2,-1)),
        up=tensor(1-2*(trace['root_quat'][1:,1]**2+trace['root_quat'][1:,2]**2)),
        braking_phase=tensor(((t-.00125)>=.4)&((t-.00125)<1.8)))
    s={k:x.reshape(139,8,*x.shape[1:]) for k,x in v.items()}
    weights=torch.full((139,),mg,dtype=torch.float64)
    reward=boundary_objective(s,weights,cfg)['reward'].numpy()
    counterfactual={}
    for field,value in [('normal_fz',0.),('lateral_slip_rms',0.),('friction_anchor_power',1.),
                         ('opposing_sliding_anchors',0.),('braking_phase',0.),('up',.7)]:
        altered=dict(s);altered[field]=torch.full_like(s[field],value)
        maximum=float(boundary_objective(altered,weights,cfg)['reward'].max())
        if maximum!=0:
            raise ValueError('Reward exploits counterfactual '+field)
        counterfactual[field]=maximum
    return dict(cell=cell.name,case=case,provenance=audited['provenance'],
        native_reconstruction_max_errors=errors,counterfactual_max_bonus=counterfactual,
        mean_unweighted_bonus=float(reward.mean()),max_unweighted_bonus=float(reward.max()),
        positive_control_intervals=int(np.count_nonzero(reward>1e-8)),
        nominal_task_success=audited['qualification']['qualification']['task_success_claimed']),reward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.asset)!='327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e':
        raise ValueError('Different asset mass source')
    mass=sum(float(x) for x in re.findall(r'float physics:mass = ([0-9.eE+-]+)',a.asset.read_text()))
    cfg=json.loads(a.config.read_text())['friction_boundary'];ref=load(a.reference)
    results=[];curves={}
    for method in ('beyondmimic','hockey'):
        for case in ('nominal','trial_022'):
            r,curve=replay(a.runs/f'{method}_s100_seed42',case,ref,mass*9.81,cfg)
            results.append(r);curves[f'{method}_{case}']=curve
    a.output.mkdir(parents=True)
    report=dict(status='offline_reward_audit_passed_not_trained',mass_kg=mass,
        asset_sha256=sha(a.asset),reference_sha256=sha(a.reference),config_sha256=sha(a.config),
        reward_source_sha256=sha(Path(__file__).with_name('hockey_friction_boundary.py')),
        cases=results,physics_or_policy_changed=False,pointwise_cone_certified=False,
        interpretation='Objective sanity checks on historical original-speed recordings; not evidence of improved learning or a validation success score.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(a.output/'replayed_bonus.npz',**curves)
    print(json.dumps({'status':report['status'],'cases':len(results),'mass_kg':mass,
                      'mean_bonuses':[r['mean_unweighted_bonus'] for r in results]}))


if __name__=='__main__':
    main()

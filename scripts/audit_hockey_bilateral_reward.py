"""Replay the new GPU-compatible sampler against independent aligned evidence."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch

from evaluate_hockey_physx import load,sha
from hockey_bilateral_task import geometric_sample,cylinder_gap,objective
from hockey_friction_boundary import boundary_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('runs','audit','references','asset','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('config/hockey_bilateral_training_v1.json'))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2)
    cfg=json.loads(a.config.read_text());source=json.loads((a.audit/'report.json').read_text())
    assert sha(a.asset)==source['asset_sha256']
    assert sha(a.audit/'aligned_signals.npz')==source['aligned_signals_sha256']
    aligned=load(a.audit/'aligned_signals.npz')
    mg=9.81*sum(float(v) for v in re.findall(r'float physics:mass = ([0-9.eE+-]+)',a.asset.read_text()))
    tensor=lambda v:torch.as_tensor(np.asarray(v),dtype=torch.float64)
    rows=[];curves={}
    for cell in source['cells']:
        name=cell['cell'];d=a.runs/name/'eval-suite/nominal'
        for file in ('final_rollout.npz','native_contact_buffers.npz'):
            assert sha(d/file)==cell['provenance']['recorded_artifacts'][file]
        tr=load(d/'final_rollout.npz');raw=load(d/'native_contact_buffers.npz');count=len(raw['physical_sample_indices'])
        assert count==1112 and np.array_equal(raw['physical_sample_indices'],np.arange(1,1113))
        independent={k.split('__',1)[1]:v[1:] for k,v in aligned.items() if k.startswith(name+'__')}
        sides=[];errors=[]
        for side,wheel in enumerate(('wheel_L_Link','wheel_R_Link')):
            b=tr['body_names'].astype(str).tolist().index(wheel)
            def buffers(kind,size):
                out=[]
                for i in range(size):
                    v=raw[f'w{side}_{kind}_buffer_{i}'].copy()
                    if i>=size-2:
                        if i==size-1:v+=(np.arange(count)*128).reshape(count,1,1)
                        out.append(torch.as_tensor(v.reshape(count,1)))
                    else:out.append(tensor(v.reshape(-1,v.shape[-1])))
                return out
            normal,friction=buffers('normal',6),buffers('friction',4)
            com=tensor(tr['body_com_pos'][:-1,b]+tr['environment_origin_w']);vel=tensor(tr['body_com_vel'][:-1,b])
            omega=tensor(tr['body_omega'][:-1,b]);q=tensor(tr['body_quat'][:-1,b])
            geo=geometric_sample(normal,com,vel,omega,q,cfg['contact_tolerance_m'])
            f=boundary_sample(normal,friction,com,vel,omega,q,torch.full((count,),.7,dtype=torch.float64))
            assert not bool((geo['fault']|f['fault']).any())
            assert np.array_equal(geo['touching'].numpy(),independent['touching'][:,side])
            mask=geo['touching'].numpy()
            e=float(np.max(abs(geo['lateral_slip_rms'].numpy()[mask]-independent['lateral_rms_m_s'][mask,side])))
            power_error=float(np.max(abs(f['friction_anchor_power'].numpy()-independent['anchor_power_pre'][:,side])))
            assert e<1e-6 and power_error<1e-5
            pre_gap=cylinder_gap(tensor(tr['body_pos'][:-1,b]),q,cfg['geometry'],side)
            measured=torch.isfinite(geo['min_gap']);gap_error=float(abs(geo['min_gap'][measured]-pre_gap[measured]).max())
            assert gap_error<cfg['geometry_alignment_max_error_m']
            errors.append(dict(side=side,lateral_velocity_max_error_m_s=e,anchor_power_max_error_W=power_error,
                               authored_native_gap_max_error_m=gap_error))
            f['utilization_proxy']=torch.where(f['normal_fz']>1e-6,f['friction_force_w'][...,:2].norm(dim=-1)/(.7*f['normal_fz'].clamp_min(1e-6)),0.)
            sides.append(dict(touching=geo['touching'].double(),post_gap=cylinder_gap(tensor(tr['body_pos'][1:,b]),tensor(tr['body_quat'][1:,b]),cfg['geometry'],side),
                lateral_slip_rms=geo['lateral_slip_rms'],**{k:f[k] for k in ('normal_fz','utilization_proxy','friction_anchor_power','opposing_sliding_anchors','friction_force_w')}))
        v={k:torch.stack([s[k] for s in sides],1) for k in sides[0]}
        reference=load(a.references/name.split('_')[1]/'motion.npz');rt=np.arange(139)*.02;t=np.arange(count)*.0025
        interp=lambda x:np.column_stack([np.interp(t,rt,x[:,j]) for j in (0,1)])
        v.update(root_velocity_xy=tensor(tr['root_com_vel'][:-1,:2]),
            xy_error_sq=tensor(np.sum((tr['root_pos'][:-1,:2]-interp(reference['body_pos_w'][:,0,:2]))**2,-1)),
            velocity_error_sq=tensor(np.sum((tr['root_com_vel'][:-1,:2]-interp(reference['body_lin_vel_w'][:,0,:2]))**2,-1)),
            up=tensor(1-2*(tr['root_quat'][:-1,1]**2+tr['root_quat'][:-1,2]**2)),
            braking_phase=tensor((t+.00125>=.4)&(t+.00125<1.8)),
            angular_acceleration_sq=tensor(np.sum((np.diff(tr['root_omega'],axis=0)/.0025)**2,-1)))
        samples={k:x.reshape(139,8,*x.shape[1:]) for k,x in v.items()}
        out=objective(samples,torch.full((139,),mg,dtype=torch.float64),cfg)
        missing=~samples['touching'].bool().all(-1).all(-1)
        assert not bool((out['sliding'][missing]>0).any())
        rows.append(dict(cell=name,sampler_errors=errors,mean_components={k:float(x.mean()) for k,x in out.items()},
                         contact_failed_windows_with_slide_bonus=0))
        curves.update({name+'__'+k:x.numpy() for k,x in out.items()})
    a.output.mkdir(parents=True)
    report=dict(status='offline_sampler_and_reward_checks_passed_not_trained',cells=rows,
        wheel_physics_samples=4*1112*2,script_sha256=sha(__file__),reward_sha256=sha(Path(__file__).with_name('hockey_bilateral_task.py')),
        config_sha256=sha(a.config),aligned_audit_sha256=sha(a.audit/'report.json'),asset_sha256=sha(a.asset),
        training_or_physics_changed=False,full_task_success_claimed=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(a.output/'replayed_components.npz',**curves)
    print(json.dumps(dict(status=report['status'],cases=len(rows),samples=report['wheel_physics_samples'])))


if __name__=='__main__':main()

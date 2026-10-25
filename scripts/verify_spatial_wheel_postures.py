"""Recompute all retained spatial snapshot states and independent base wrenches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import generate_camber_reference as base
from fit_spatial_wheel_postures import Placement,measure


def run(a):
    import mujoco
    from PIL import Image
    root=a.result;target=root/'verification_v1.json'
    if target.exists():raise FileExistsError(target)
    stages={name:json.loads((root/path/'analysis.json').read_text()) for name,path in
        [('atlas',Path('.')),('wrench',Path('wrench')),('Powell',Path('placement_fit')),('TRF',Path('placement_TRF')),('SQP',Path('placement_SQP'))]}
    scripts=Path(__file__).resolve().parent
    for name,source in [('atlas','audit_spatial_camber_postures.py'),('wrench','audit_spatial_camber_wrenches.py'),
            ('Powell','fit_spatial_wheel_postures.py'),('TRF','refine_spatial_wheel_postures.py'),('SQP','refine_spatial_wheel_postures_sqp.py')]:
        d=stages[name]
        assert d['source_sha256']==base.sha(scripts/source)
        assert d['geometry_source_sha256']==base.sha(scripts/'spatial_wheel_posture.py')
        assert d['model_sha256']==base.sha(a.model)==base.MODEL_SHA
        assert d['reference_sha256']==base.sha(a.reference)==base.SOURCE_SHA
        assert not d['task_success_claimed']
    with np.load(a.reference,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    g=Placement(mujoco.MjModel.from_xml_path(str(a.model)),source);weight=float(g.model.body_mass.sum())*9.81
    retained=[]
    for frame,items in stages['wrench']['details'].items():
        for row in items['best_wrench_postures']+[items['previous_XY_only_choice']]:retained.append(('wrench',int(frame),row,1e-4))
    for stage in ['Powell','TRF','SQP']:
        for row in stages[stage]['rows']:retained.append((stage,row['frame'],row['best'],stages[stage]['config']['min_material_sliding_speed_m_s']))
    maximum=dict(required_force=0.,contact_force_moment=0.,mass_matrix_plus_bias=0.,geometry_gap=0.)
    for stage,frame,saved,margin in retained:
        params=saved.get('posture_parameters',saved['posture_deg_and_height_m']+[0.,0.])
        actual=measure(g,frame,params,.7,weight,margin)
        for field in ['qpos','qvel','qacc','required_generalized_force','conditional_contact_generalized_force','unactuated_base_residual']:
            np.testing.assert_allclose(actual[field],saved[field],atol=1e-10,rtol=0)
        maximum['required_force']=max(maximum['required_force'],float(abs(np.asarray(actual['required_generalized_force'])-saved['required_generalized_force']).max()))
        assert abs(actual['weighted_base_wrench_residual']-saved['weighted_base_wrench_residual'])<1e-12
        q,v,acc=(np.asarray(saved[key]) for key in ['qpos','qvel','qacc'])
        independent=mujoco.MjData(g.model);independent.qpos[:]=q;independent.qvel[:]=v
        mujoco.mj_forward(g.model,independent)
        force=np.empty(g.model.nv);mujoco.mj_mulM(g.model,independent,force,acc);force+=independent.qfrc_bias
        error=float(abs(force-np.asarray(saved['required_generalized_force'])).max())
        maximum['mass_matrix_plus_bias']=max(maximum['mass_matrix_plus_bias'],error)
        assert error<1e-10 and independent.time==0.
        g.pose(q);points,axes,_=g.wheel_geometry();rotation=g.data.xmat[g.model.body('base_Link').id].reshape(3,3)
        normal=np.asarray(saved['conditional_normal_load_L_R_n'])
        assert (normal>=0).all() and abs(normal.sum()-force[2])<1e-10
        slips=[]
        for point,body in zip(points,g.wheels):
            mujoco.mj_jac(g.model,g.data,g.jp,g.jr,point,body);slips.append((g.jp@v)[:2])
        slips=np.asarray(slips);directions=np.c_[-.7*slips/np.linalg.norm(slips,axis=-1)[:,None],np.ones(2)]
        forces=normal[:,None]*directions
        world_moment=np.cross(points-q[:3],forces).sum(0)
        direct=np.r_[forces.sum(0),rotation.T@world_moment]
        error=float(abs(direct-np.asarray(saved['conditional_contact_generalized_force'])[:6]).max())
        maximum['contact_force_moment']=max(maximum['contact_force_moment'],error);assert error<1e-10
        maximum['geometry_gap']=max(maximum['geometry_gap'],float(abs(points[:,2]).max()))
        np.testing.assert_array_equal(q[:2],source['qpos_mujoco'][frame,:2])
        yaw=2*np.arctan2(source['qpos_mujoco'][frame,6],source['qpos_mujoco'][frame,3])
        assert abs(np.arctan2(rotation[1,0],rotation[0,0])-yaw)<1e-12
    render=json.loads((root/'render/provenance.json').read_text())
    assert render['source_sha256']==base.sha(scripts/'render_spatial_wheel_postures.py')
    assert render['fit_sha256']==base.sha(root/'placement_SQP/analysis.json')
    assert render['image_sha256']==base.sha(root/'render/snapshots_4x1.png')
    with Image.open(root/'render/snapshots_4x1.png') as image:assert image.size==(2560,520);image.verify()
    assert all(row['distance_m']==1.9 and row['lookat_offset_m']==[0.,0.,-.25] and row['azimuth_deg']==135. and row['elevation_deg']==-20. for row in render['camera'])
    result=dict(status='retained_snapshot_numerics_and_artifacts_verified',retained_snapshots_recomputed=len(retained),maximum_errors=maximum,
        independent_required_force_method='mass matrix multiply plus bias;forward refresh without stepping',
        independent_contact_wrench_method='world force sum and root-local cross-product moment',
        original_root_XY_and_projected_body_heading_verified=True,source_sha256=base.sha(__file__),
        render_image_sha256=base.sha(root/'render/snapshots_4x1.png'),manual_visual_review_separate=True,
        full_trajectory_or_native_support_verified=False,physics_integration_steps=0,policy_updates=0,task_success_claimed=False)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('result','reference','model'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())

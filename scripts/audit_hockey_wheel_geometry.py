#!/usr/bin/env python3
"""Read-only comparison of MJCF retarget geometry and the locked PhysX USD.

Runs FK and authored cylinder support calculations, never a physics step or
asset edit. Authored USD support is not PhysX cooked-manifold separation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics

from wheel_ground_geometry import cylinder_support


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('command','model','asset','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.asset)!='327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e':raise ValueError('PhysX asset differs from frozen probe')
    command=json.loads((a.command/'report.json').read_text())
    if sha(a.model)!=command['model_sha256']:raise ValueError('Retarget model changed')
    z=np.load(a.command/'command_diagnostics.npz',allow_pickle=False)
    stage=Usd.Stage.Open(str(a.asset));cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    model=mujoco.MjModel.from_xml_path(str(a.model));data=mujoco.MjData(model)
    geometry=[];usd_gap=np.empty((len(z['qpos']),2));mj_gap=np.empty_like(usd_gap)
    for k,side in enumerate(('L','R')):
        name=f'wheel_{side}_Link';body=stage.GetPrimAtPath('/wheel_biped/'+name)
        collider=stage.GetPrimAtPath(str(body.GetPath())+'/collisions/mesh_0/cylinder')
        if not body.HasAPI(UsdPhysics.RigidBodyAPI) or not collider.IsA(UsdGeom.Cylinder):raise ValueError('Unexpected wheel asset structure')
        cylinder=UsdGeom.Cylinder(collider);matrix,_=cache.ComputeRelativeTransform(collider,body);matrix=np.asarray(matrix)
        np.testing.assert_allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6)
        local_axis=matrix[{'X':0,'Y':1,'Z':2}[str(cylinder.GetAxisAttr().Get())],:3]
        center=matrix[3,:3];radius=float(cylinder.GetRadiusAttr().Get());width=float(cylinder.GetHeightAttr().Get())
        bid=model.body(name).id;gid=model.geom(f'wheel_{side}_collision').id
        geometry.append(dict(side=side,USD_radius_m=radius,USD_width_m=width,USD_center_in_link_m=center.tolist(),
            USD_rotation_cylinder_to_link=matrix[:3,:3].T.tolist(),
            MJCF_radius_m=float(model.geom_size[gid,0]),MJCF_width_m=float(2*model.geom_size[gid,1]),
            MJCF_center_in_link_m=model.geom_pos[gid].tolist()))
        for i,q in enumerate(z['qpos']):
            data.qpos[:]=q;mujoco.mj_forward(model,data);R=data.xmat[bid].reshape(3,3)
            usd_gap[i,k]=cylinder_support(data.xpos[bid]+R@center,R@local_axis,radius,width/2)['authored_signed_gap_m']
            mj_gap[i,k]=cylinder_support(data.geom_xpos[gid],data.geom_xmat[gid].reshape(3,3)[:,2],*model.geom_size[gid,:2])['authored_signed_gap_m']
    corrected=command.get('support_geometry_source')=='PhysX authored USD wheel cylinders'
    np.testing.assert_allclose(usd_gap if corrected else mj_gap,z['wheel_bottom_proxy_m'],atol=1e-9)
    difference=usd_gap-mj_gap
    report=dict(kind='hockey_command_authored_geometry_crosscheck',script_sha256=sha(__file__),
        command_motion_sha256=sha(a.command/'motion.npz'),command_diagnostics_sha256=sha(a.command/'command_diagnostics.npz'),
        model_sha256=sha(a.model),asset_sha256=sha(a.asset),geometry=geometry,
        USD_minus_MJCF_bottom_min_m=difference.min(0).tolist(),USD_minus_MJCF_bottom_max_m=difference.max(0).tolist(),
        minimum_USD_authored_bottom_m=usd_gap.min(0).tolist(),USD_authored_penetration_beyond_1mm_samples=(usd_gap<-.001).sum(0).tolist(),
        authored_USD_geometry_matches_retarget_MJCF=bool(np.allclose(usd_gap,mj_gap,atol=1e-8,rtol=0)),
        command_support_geometry_matches_USD=corrected,
        original_assets_mutated=False,pending_upload_payload_changed=False,physics_run=False,
        cooked_geometry_or_contact_force_inferred=False,
        implication=('Command clearance uses the frozen PhysX authored wheel geometry. Native contact/forces remain unverified.' if corrected else
            'Pending v1 remains a diagnostic with a documented wheel-geometry mismatch, not a matched-contact-geometry reference. Ground contact must be measured from native buffers.'))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'geometry_gaps.npz',time_s=z['time_s'],MJCF_bottom_m=mj_gap,USD_authored_bottom_m=usd_gap,difference_m=difference)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

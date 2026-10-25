"""CPU USD verification against the existing asset; no Isaac/GPU initialization."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from tron1_wheel_geometry import WHEELS,apply_wheel_geometry,mass_properties


def run(asset):
    from pxr import Gf,Usd,UsdGeom
    rows=[]
    for level in ("authored","slices9"):
        stage=Usd.Stage.Open(str(asset));stage.SetEditTarget(stage.GetSessionLayer())
        root=str(stage.GetDefaultPrim().GetPath())
        before={name:mass_properties(stage.GetPrimAtPath(root+"/"+name)) for name in WHEELS}
        audits=apply_wheel_geometry(stage,root,level=level)
        for name,audit in zip(WHEELS,audits,strict=True):
            body=stage.GetPrimAtPath(root+"/"+name)
            assert mass_properties(body)==before[name]
            assert len(audit["active_collider_paths"])==(1 if level=="authored" else 9)
            # Compute all slice envelopes in the original cylinder frame.
            original=stage.GetPrimAtPath(audit["original_cylinder"])
            original_to_body,_=UsdGeom.XformCache().ComputeRelativeTransform(original,body)
            summed_volume=0.;edges=[]
            for path in audit["active_collider_paths"]:
                shape=UsdGeom.Cylinder(stage.GetPrimAtPath(path));r=float(shape.GetRadiusAttr().Get());h=float(shape.GetHeightAttr().Get())
                matrix,_=UsdGeom.XformCache().ComputeRelativeTransform(shape.GetPrim(),body)
                center=original_to_body.GetInverse().Transform(matrix.Transform(Gf.Vec3d(0.)))
                axis={"X":0,"Y":1,"Z":2}[str(shape.GetAxisAttr().Get())]
                assert all(abs(center[j])<1e-10 for j in range(3) if j!=axis)
                edges.append((center[axis]-h/2,center[axis]+h/2))
                summed_volume+=math.pi*r*r*h
            edges.sort();assert abs(edges[0][0]+audit["total_width_m"]/2)<1e-10
            assert abs(edges[-1][1]-audit["total_width_m"]/2)<1e-10
            assert all(abs(a[1]-b[0])<1e-10 for a,b in zip(edges[:-1],edges[1:]))
            assert abs(summed_volume-math.pi*audit["radius_m"]**2*audit["total_width_m"])<1e-12
            rows.append(dict(wheel=name,level=level,mass_COM_inertia_exact=True,transform_envelope_volume_verified=True))
        assert not stage.GetRootLayer().dirty,"Only session layer may be edited"
    return dict(kind="CPU_USD_wheel_geometry_audit",rows=rows,physics_run=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument("--asset",type=Path,required=True)
    print(json.dumps(run(p.parse_args().asset)))

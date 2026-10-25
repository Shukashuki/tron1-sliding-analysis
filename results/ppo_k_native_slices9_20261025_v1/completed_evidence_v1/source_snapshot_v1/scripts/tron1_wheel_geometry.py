"""Replace authored TRON1 wheel colliders before physics, retaining mass frames.

Unlike the ideal-wheel bench helper, this never substitutes idealized COM or
inertia. Original cylinder-to-body transforms and MassAPI values are retained.
"""
from __future__ import annotations

import math

WHEELS=("wheel_L_Link","wheel_R_Link")


def mass_properties(body):
    from pxr import UsdPhysics
    assert body.HasAPI(UsdPhysics.RigidBodyAPI) and body.HasAPI(UsdPhysics.MassAPI)
    api=UsdPhysics.MassAPI(body)
    mass,com,inertia,axes=(getattr(api,"Get"+n+"Attr")().Get()
                          for n in ("Mass","CenterOfMass","DiagonalInertia","PrincipalAxes"))
    q=[float(axes.GetReal()),*[float(v) for v in axes.GetImaginary()]]
    values=[float(mass),*[float(v) for v in com],*[float(v) for v in inertia],*q]
    assert all(math.isfinite(v) for v in values) and mass>0 and all(v>0 for v in inertia)
    assert abs(sum(v*v for v in q)-1)<1e-5
    return dict(mass_kg=float(mass),COM_m=list(com),principal_inertia_kg_m2=list(inertia),principal_axes_wxyz=q)


def collision_enabled(prim,body):
    from pxr import UsdPhysics
    current=prim
    while current:
        if current.HasAPI(UsdPhysics.CollisionAPI):
            return UsdPhysics.CollisionAPI(current).GetCollisionEnabledAttr().Get() is not False
        if current==body:break
        current=current.GetParent()
    return False


def apply_wheel_geometry(stage,robot_path,*,level,contact_offset_m=.001,rest_offset_m=0.):
    from pxr import Gf,PhysxSchema,Usd,UsdGeom,UsdPhysics,UsdShade
    assert level in ("authored","slices9")
    assert all(math.isfinite(v) for v in (contact_offset_m,rest_offset_m))
    assert 0<contact_offset_m<=.05 and contact_offset_m>rest_offset_m
    audits=[]
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    # Validate both wheels before authoring either one.
    inputs=[]
    for name in WHEELS:
        body=stage.GetPrimAtPath(robot_path+"/"+name)
        mass=mass_properties(body)
        shapes=[p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Gprim) and collision_enabled(p,body)]
        assert len(shapes)==1 and shapes[0].IsA(UsdGeom.Cylinder), "Require exactly one authored cylinder per wheel"
        old=shapes[0];geometry=UsdGeom.Cylinder(old)
        radius,width=float(geometry.GetRadiusAttr().Get()),float(geometry.GetHeightAttr().Get())
        assert abs(radius-.127)<1e-6 and abs(width-.05)<1e-6
        axis=str(geometry.GetAxisAttr().Get());assert axis in ("X","Y","Z")
        transform,_=cache.ComputeRelativeTransform(old,body)
        # Do not support hidden non-unit scaling in this controlled cohort.
        for row in range(3):assert abs(sum(transform[row][j]**2 for j in range(3))-1)<1e-6
        assert not stage.GetPrimAtPath(str(body.GetPath())+"/compound_wheel_colliders")
        inputs.append((body,mass,old,radius,width,axis,transform))
    for body,mass,old,radius,width,axis,transform in inputs:
        old_collision_scopes=[p for p in Usd.PrimRange(body) if p.HasAPI(UsdPhysics.CollisionAPI)]
        material,_=UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial(materialPurpose="physics")
        paths=[]
        if level=="authored":
            paths=[str(old.GetPath())]
        else:
            parent=str(body.GetPath())+"/compound_wheel_colliders"
            UsdGeom.Xform.Define(stage,parent).AddTransformOp().Set(transform)
            index={"X":0,"Y":1,"Z":2}[axis]
            for i in range(9):
                shape=UsdGeom.Cylinder.Define(stage,parent+f"/slice_{i:02d}")
                height=width/9
                shape.CreateAxisAttr(axis);shape.CreateRadiusAttr(radius);shape.CreateHeightAttr(height)
                shift=[0.,0.,0.];shift[index]=-width/2+(i+.5)*height
                shape.AddTranslateOp().Set(Gf.Vec3d(*shift));shape.CreatePurposeAttr("guide")
                low=[-radius]*3;high=[radius]*3;low[index]=-height/2;high[index]=height/2
                shape.CreateExtentAttr([Gf.Vec3f(*low),Gf.Vec3f(*high)])
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
                if material:UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material,materialPurpose="physics")
                paths.append(str(shape.GetPath()))
            for p in old_collision_scopes:
                p.RemoveAPI(UsdPhysics.CollisionAPI);p.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
        for path in paths:
            api=PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(path))
            api.CreateContactOffsetAttr(contact_offset_m);api.CreateRestOffsetAttr(rest_offset_m)
        assert mass_properties(body)==mass,"Mass, COM or principal inertia changed"
        active=[p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Gprim) and collision_enabled(p,body)]
        assert {str(p.GetPath()) for p in active}==set(paths)
        audits.append(dict(body=str(body.GetPath()),level=level,original_cylinder=str(old.GetPath()),
            radius_m=radius,total_width_m=width,original_axis=axis,
            original_cylinder_to_body_matrix=[[float(transform[i][j]) for j in range(4)] for i in range(4)],
            authored_mass_properties_preserved=mass,active_collider_paths=paths,
            contact_offset_m=contact_offset_m,rest_offset_m=rest_offset_m,
            material_path=str(material.GetPath()) if material else None,
            unbound_material_uses_existing_scene_default=not bool(material),
            new_bodies_or_joints=0,original_asset_file_mutated=False))
    return audits

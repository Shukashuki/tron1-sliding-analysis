"""Uniform axial compound cylinders; geometry changes, not extra wheel DOFs.

USD authoring is deliberately separate from the numeric description. Call only
before physics initialization. An authored cylinder is not proof of a cooked
analytic collider: the caller must also record the installed PhysX setting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class UniformWheel:
    radius_m: float = .127
    width_m: float = .05
    mass_kg: float = 1.08
    slices: int = 1

    def __post_init__(self):
        if (type(self.slices) is not int or self.slices not in (1, 3, 5, 9)
                or not all(math.isfinite(x) and x > 0
                           for x in (self.radius_m, self.width_m, self.mass_kg))):
            raise ValueError("Positive finite dimensions/mass; slice count 1,3,5,9")

    @property
    def diagonal_inertia_kg_m2(self):
        r, w, m = self.radius_m, self.width_m, self.mass_kg
        transverse = m * (3*r*r + w*w) / 12
        return (transverse, .5*m*r*r, transverse)  # body-local Y axle

    def segments(self):
        edges = [-self.width_m/2 + self.width_m*i/self.slices
                 for i in range(self.slices+1)]
        return [{"center_y_m": (a+b)/2, "height_m": b-a,
                 "radius_m": self.radius_m, "mass_fraction": 1/self.slices}
                for a, b in zip(edges[:-1], edges[1:])]

    def describe(self):
        return {"radius_m": self.radius_m, "width_m": self.width_m,
                "mass_kg": self.mass_kg, "slices": self.slices,
                "body_local_axle": "Y", "COM_m": [0., 0., 0.],
                "diagonal_inertia_kg_m2": list(self.diagonal_inertia_kg_m2),
                "segments": self.segments(), "rigid_bodies": 1,
                "additional_DOF": 0, "profile_changes_outer_envelope": False}


def author_compound(stage, body_path, old_collider_path, wheel: UniformWheel):
    """Replace one enabled collider, preserving its physics material binding.

    Explicit total mass/COM/principal inertia prevents collision-volume based
    inertia recomputation. This bench helper requires an unscaled rigid frame
    and authors Y-axis cylinders directly in that frame. The original visual
    remains, but its collision is disabled. Never invoke on a running stage.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    body, old = stage.GetPrimAtPath(body_path), stage.GetPrimAtPath(old_collider_path)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Existing rigid wheel body required")
    if not old or not old.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("Existing collider required")
    enabled = [p for p in Usd.PrimRange(body)
               if p.HasAPI(UsdPhysics.CollisionAPI)
               and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    if [p.GetPath() for p in enabled] != [old.GetPath()]:
        raise ValueError("Exactly the declared original collider must be enabled")
    material, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial(materialPurpose="physics")
    if not material:
        raise ValueError("Original physics material binding not resolved")
    destination = body_path+"/contact_slices"
    if stage.GetPrimAtPath(destination):
        raise ValueError("Refuse to overwrite existing geometry")
    # Validate before changing the stage.
    if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(body) if p != body):
        raise ValueError("Nested rigid body is not a compound wheel")
    old_physics = PhysxSchema.PhysxCollisionAPI(old)
    offsets = {}
    for name in ("ContactOffset", "RestOffset"):
        attr = getattr(old_physics, "Get"+name+"Attr")()
        # Schema fallback -inf means automatic offset, not a measured length.
        # Do not turn an unauthored fallback into an explicit nonfinite value.
        value = attr.Get()
        offsets[name] = (float(value) if attr.HasAuthoredValueOpinion()
                         and value is not None and math.isfinite(value) else None)
    UsdGeom.Xform.Define(stage, destination)
    paths = []
    for i, segment in enumerate(wheel.segments()):
        cylinder = UsdGeom.Cylinder.Define(stage, destination+f"/slice_{i:02d}")
        cylinder.CreateAxisAttr("Y")
        cylinder.CreateRadiusAttr(segment["radius_m"])
        cylinder.CreateHeightAttr(segment["height_m"])
        cylinder.AddTranslateOp().Set(Gf.Vec3d(0., segment["center_y_m"], 0.))
        cylinder.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim()).CreateCollisionEnabledAttr(True)
        physics = PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
        for name, value in offsets.items():
            if value is not None:
                getattr(physics, "Create"+name+"Attr")(value)
        UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(material, materialPurpose="physics")
        paths.append(str(cylinder.GetPath()))
    UsdPhysics.CollisionAPI(old).CreateCollisionEnabledAttr(False)
    # A disabled collider still occupies a native shape slot in this installed
    # PhysX build. Remove the old collision APIs so the compound has exactly N
    # shapes, while retaining the original rendering geometry.
    old.RemoveAPI(UsdPhysics.CollisionAPI)
    old.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(wheel.mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*wheel.diagonal_inertia_kg_m2))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
    return {**wheel.describe(), "new_collider_paths": paths,
            "disabled_original_collider": old_collider_path,
            "original_collision_API_removed": True,
            "physics_material_path": str(material.GetPath()),
            "copied_collision_offsets_m": offsets,
            "explicit_mass_COM_principal_inertia_authored": True,
            "cooked_collision_geometry_readback_verified": False}

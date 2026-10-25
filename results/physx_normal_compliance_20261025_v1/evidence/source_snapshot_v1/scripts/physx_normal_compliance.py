"""Predeclared native force-spring contact profiles, not identified tire data."""
from __future__ import annotations
import math

PROFILES = ("rigid", "compliant_1mm_d1", "compliant_3mm_d0", "compliant_3mm_d1")
REFERENCE_REPORT_SHA256 = "29499d653e272b74aadb491be2f41c940b5232674ffb7e5f1c3a4dbc333d9130"
DOC = "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html"


def profile(name, supported_mass_kg):
    assert name in PROFILES and math.isfinite(supported_mass_kg) and supported_mass_kg > 0
    deflection = 0. if name == "rigid" else (.001 if "1mm" in name else .003)
    ratio = 0. if name in ("rigid", "compliant_3mm_d0") else 1.
    stiffness = supported_mass_kg*9.81/deflection if deflection else 0.
    damping = 2*ratio*math.sqrt(stiffness*supported_mass_kg)
    return dict(id=name, reference_supported_mass_kg=supported_mass_kg,
        reference_static_load_n=supported_mass_kg*9.81, nominal_single_point_deflection_m=deflection,
        nominal_single_point_damping_ratio=ratio, stiffness_n_m=stiffness, damping_n_s_m=damping,
        acceleration_spring=False, physical_tire_parameters_identified=False,
        per_contact_stiffness_not_whole_manifold_stiffness=True)


def author(material_prim, settings):
    from pxr import PhysxSchema
    api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    api.CreateCompliantContactAccelerationSpringAttr(False)
    api.CreateCompliantContactStiffnessAttr(settings["stiffness_n_m"])
    api.CreateCompliantContactDampingAttr(settings["damping_n_s_m"])
    api.CreateRestitutionCombineModeAttr("max")
    api.CreateDampingCombineModeAttr("max")
    return read_material(material_prim)


def read_material(prim):
    from pxr import PhysxSchema
    api = PhysxSchema.PhysxMaterialAPI(prim)
    return dict(path=str(prim.GetPath()), stiffness_n_m=api.GetCompliantContactStiffnessAttr().Get(),
        damping_n_s_m=api.GetCompliantContactDampingAttr().Get(), acceleration_spring=api.GetCompliantContactAccelerationSpringAttr().Get(),
        restitution_combine=str(api.GetRestitutionCombineModeAttr().Get()), damping_combine=str(api.GetDampingCombineModeAttr().Get()))

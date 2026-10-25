"""Frozen K actor under four native normal-contact models, unchanged commands.

Each wheel receives a separate material before physics initialization. The
startup recipe remains intact; after its audit, only the two wheel materials
are switched live to their prevalidated force-spring profile. No extra physics
steps, pose changes, actuator filtering, reward edits or PPO updates are added.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import physx_normal_compliance as normal

PARENT_SHA="32b690dac706929b4df582a1b758fb0cbebe242f9ca8c5646f6c064d2a2eec62"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    pairs=[('                return prim\n',
        '                _normal_prepare(prim.GetStage(),str(prim.GetPath()))\n                return prim\n'),
        ('        original=physx.audit_materials(env,current)\n',
         '        original=physx.audit_materials(env,current)\n        _normal_install(env,original)\n'),
        ('        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata\n',
         '        _normal_finish(metadata,report,contract,current)\n'
         '        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata\n')]
    for old,new in pairs:
        assert source.count(old)==1,"Missing/ambiguous native-normal model hook"
        source=source.replace(old,new,1)
    compile(source,"<native_normal_model_parent>","exec")
    return source


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument("--contact-model-evaluator",type=Path,required=True)
    parser.add_argument("--normal-profile",choices=normal.PROFILES,required=True)
    parser.add_argument("--normal-protocol",type=Path,required=True)
    parser.add_argument("--normal-reference-report",type=Path,required=True)
    parser.add_argument("--bench-dir",type=Path,required=True)
    own,remaining=parser.parse_known_args(argv)
    assert sha(own.contact_model_evaluator)==PARENT_SHA and sha(own.normal_reference_report)==normal.REFERENCE_REPORT_SHA256
    for flag,value in (("--geometry","authored"),("--law","native_coulomb"),("--contact-offset-m",".001"),("--rest-offset-m","0")):
        assert remaining.count(flag)==1 and remaining[remaining.index(flag)+1]==value
    protocol=json.loads(own.normal_protocol.read_text());assert protocol["profiles"]==list(normal.PROFILES)
    reference=json.loads(own.normal_reference_report.read_text())["contact_material_audit"]["native_plant_readback"]
    mass=float(np.asarray(reference["readbacks"]["get_masses"]["values"]).sum())
    settings=normal.profile(own.normal_profile,mass/2)
    bench=json.loads((own.bench_dir/"status.json").read_text())
    assert bench["status"]=="four_native_bench_cases_completed_require_analysis"
    assert [c["id"] for c in bench["cases"]]==list(normal.PROFILES)
    bench_proof={str((own.bench_dir/"status.json").resolve()):sha(own.bench_dir/"status.json")}
    for case in bench["cases"]:
        folder=own.bench_dir/case["id"];proof=json.loads((folder/"status.json").read_text())
        assert case["status"]=="completed_verified" and sha(folder/"status.json")==case["proof_sha256"]
        assert proof["peak_impulse_balance_error_n"]<.1 and proof["profile"]==normal.profile(case["id"],mass/2)
        assert sha(folder/"native_trace.npz")==proof["native_trace_sha256"]==case["native_trace_sha256"]
        bench_proof[str((folder/"status.json").resolve())]=sha(folder/"status.json")
    rigid=json.loads((own.bench_dir/"rigid/status.json").read_text())
    assert rigid["live_switch"]["USD_update_without_physics_consumed"]
    assert sha(own.bench_dir/"rigid/live_switch_trace.npz")==rigid["live_switch"]["trace_sha256"]
    with np.load(own.bench_dir/"rigid/live_switch_trace.npz",allow_pickle=False) as live, np.load(own.bench_dir/"compliant_3mm_d1/native_trace.npz",allow_pickle=False) as static:
        assert abs(float(live["root_state_w"][-1,2])-float(static["root_state_w"][-1,2]))<1e-5
    source=materialize(own.contact_model_evaluator.read_text())
    sources={str(Path(p).resolve()):sha(p) for p in (__file__,normal.__file__,own.contact_model_evaluator,own.normal_protocol,own.normal_reference_report)}
    prepared=[];runtime=[]

    def prepare(stage,robot_path):
        from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema
        import tron1_wheel_geometry as geometry
        assert not prepared
        for name in geometry.WHEELS:
            body=stage.GetPrimAtPath(robot_path+"/"+name)
            shapes=[p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Gprim) and geometry.collision_enabled(p,body)]
            assert len(shapes)==1
            path=robot_path+"/normal_material_"+name
            assert not stage.GetPrimAtPath(path)
            material=UsdShade.Material.Define(stage,path)
            api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(1.);api.CreateDynamicFrictionAttr(1.);api.CreateRestitutionAttr(0.)
            PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("multiply")
            normal.author(material.GetPrim(),normal.profile("rigid",mass/2))
            UsdShade.MaterialBindingAPI.Apply(shapes[0]).Bind(material,materialPurpose="physics")
            prepared.append(dict(wheel=name,collider=str(shapes[0].GetPath()),material=path))

    def install(env,original_audit):
        import torch,omni.usd
        from pxr import UsdShade,UsdPhysics
        from physx_mjlab_recipe import _shape_map
        assert len(prepared)==2 and not runtime and env.num_envs==1
        robot=env.scene["robot"];view=robot.root_physx_view
        mapping=_shape_map(robot);slots=[slot for item in prepared for slot in mapping[item["wheel"]]]
        assert len(slots)==2 and view.max_shapes==9
        before=view.get_material_properties().clone();stage=omni.usd.get_context().get_stage()
        stamp=lambda:(int(env._sim_step_counter),int(env.sim.current_time_step_index),float(env.sim.current_time))
        clock=stamp();material_rows=[]
        for item in prepared:
            collider=stage.GetPrimAtPath(item["collider"])
            material,_=UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
            assert str(material.GetPath())==item["material"]
            prior=normal.read_material(material.GetPrim())
            if own.normal_profile!="rigid":
                api=UsdPhysics.MaterialAPI(material.GetPrim())
                api.GetStaticFrictionAttr().Set(.7);api.GetDynamicFrictionAttr().Set(.7)
                normal.author(material.GetPrim(),settings)
            material_rows.append(dict(**item,before_USD=prior,after_USD=normal.read_material(material.GetPrim())))
        after=view.get_material_properties();assert stamp()==clock
        expected=before.clone();expected[:,slots,2]=-settings["stiffness_n_m"]
        assert torch.allclose(after,expected,rtol=1e-6,atol=1e-6),"Native normal profile did not update exactly the wheel shape slots"
        observed=dict(profile=settings,robot_native_mass_kg=mass,preparation=material_rows,
            native_shape_slots=slots,native_material_before=before.cpu().numpy().tolist(),native_material_after=after.cpu().numpy().tolist(),
            only_wheel_normal_material_changed=True,friction_and_nonwheel_materials_unchanged=True,
            no_additional_physical_steps=True,clock_at_intervention=list(clock),
            damping_direct_native_tensor_readback_available=False,bench_source_sha256=bench_proof)
        original_audit["normal_contact_model_intervention"]=observed
        original_audit["audit_phase_note"]="Original recipe verified before live normal-material update; current wheel coefficients are recorded in native_material_after"
        runtime.append(observed)

    def finish(metadata,report,contract,plan):
        assert len(runtime)==1 and all(sha(p)==digest for p,digest in sources.items())
        assert all(sha(p)==digest for p,digest in bench_proof.items())
        with np.load(plan["output"]/"physics_trajectory.npz",allow_pickle=False) as raw:
            force=raw["wheel_ground_contact_force_w_n"]
            assert force.shape==(2793,1,2,3) and np.isfinite(force).all() and force[...,2].min()>=-1e-4
        metadata["kind"]="frozen_K_PhysX_native_normal_contact_model_transfer"
        metadata["normal_contact_model_intervention"]=runtime[0]
        metadata["normal_protocol_sha256"]=sha(own.normal_protocol)
        metadata["source_sha256"].update(sources)
        metadata["actor_and_command_execution_changed"]=False
        metadata["native_normal_force_min_Z_n"]=float(force[...,2].min())
        metadata["additional_training_updates"]=0
        report["native_normal_compliance_transfer"]=metadata
        contract["normal_contact_model_intervention"]=runtime[0]
        with (plan["output"]/"normal_compliance_status.json").open("x") as f:json.dump(metadata,f,indent=2,allow_nan=False)

    namespace=dict(__name__="_frozen_K_normal_model_transfer",__file__=str(own.contact_model_evaluator.resolve()),
        _normal_prepare=prepare,_normal_install=install,_normal_finish=finish)
    previous=sys.argv
    try:
        sys.path.append(str(own.contact_model_evaluator.resolve().parent))
        sys.argv=[str(own.contact_model_evaluator),*remaining]
        exec(compile(source,"<native_normal_model_parent>","exec"),namespace)
        namespace["main"]()
    finally:sys.argv=previous


if __name__=="__main__":main(sys.argv[1:])

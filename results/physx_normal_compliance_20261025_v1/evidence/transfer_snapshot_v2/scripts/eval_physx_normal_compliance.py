"""Frozen K actor under four native normal-contact models, unchanged commands.

Each wheel receives a full separate material before physics initialization.
Fixed wheel/nonwheel friction is authored too; startup metadata and audits
make no three-property material write that could change compliance. No extra physics
steps, pose changes, actuator filtering, reward edits or PPO updates are added.
"""
from __future__ import annotations
import argparse
import copy
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
        ('        physx.configure(cfg,profile,current)\n',
         '        physx.configure(cfg,profile,current)\n        _normal_configure(cfg)\n'),
        ('        original=physx.audit_materials(env,current)\n',
         '        original=_normal_audit_materials(env,current)\n        _normal_install(env,original)\n'),
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
    parser.add_argument("--material-roundtrip-dir",type=Path,required=True)
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
    roundtrip=json.loads((own.material_roundtrip_dir/"status.json").read_text())
    assert roundtrip["status"]=="completed_requires_analysis" and roundtrip["material_startup_roundtrip"]["no_extra_physics_steps"]
    assert roundtrip["profile"]==normal.profile("compliant_3mm_d1",mass/2)
    assert sha(own.material_roundtrip_dir/"native_trace.npz")==roundtrip["native_trace_sha256"]
    with np.load(own.material_roundtrip_dir/"native_trace.npz",allow_pickle=False) as restored,np.load(own.bench_dir/"compliant_3mm_d1/native_trace.npz",allow_pickle=False) as static:
        roundtrip_errors={key:float(np.max(np.abs(restored[key]-static[key]))) for key in ("root_state_w","ground_normal_force_w_n","native_tangent_force_w_n")}
    bench_proof[str((own.material_roundtrip_dir/"status.json").resolve())]=sha(own.material_roundtrip_dir/"status.json")
    source=materialize(own.contact_model_evaluator.read_text())
    sources={str(Path(p).resolve()):sha(p) for p in (__file__,normal.__file__,own.contact_model_evaluator,own.normal_protocol,own.normal_reference_report)}
    prepared=[];runtime=[]
    recipe_path=Path(remaining[remaining.index("--base-snapshot")+1])/"scripts/physx_mjlab_recipe.py"
    assert sha(recipe_path)=="1365f4eecc6129bef4738e05a491763a2884407d8cf00a3a7d0faffbc24fa699"
    recipe_source=recipe_path.read_text()
    anchor="or torch.count_nonzero(values[..., 2])):"
    assert recipe_source.count(anchor)==1
    recipe_source=recipe_source.replace(anchor,"or not torch.allclose(values[..., 2], _expected_normal_slot(values, wheel_slots), rtol=1.e-6, atol=1.e-6)):",1)
    def expected_normal_slot(values,slots):
        import torch
        expected=torch.zeros_like(values[...,2]);expected[:,slots]=-settings["stiffness_n_m"]
        return expected
    recipe_namespace=dict(__name__="_normal_model_readonly_recipe_audit",__file__=str(recipe_path),_expected_normal_slot=expected_normal_slot)
    exec(compile(recipe_source,"<normal_model_readonly_recipe_audit>","exec"),recipe_namespace)
    sources[str(recipe_path.resolve())]=sha(recipe_path)

    def fixed_material_metadata(env,env_ids,training,fixed_mu):
        import torch
        assert not training and fixed_mu==.7 and env.num_envs==1
        from physx_mjlab_recipe import _shape_map,recipe_manifest
        # All actual coefficients/profile fields were authored before parsing.
        # The standard3-column tensor setter changes compliant response in the
        # verified roundtrip diagnostic, so fixed evaluation uses no such write.
        env.mjlab_recipe_mu=torch.full((env.num_envs,),.7)
        env.mjlab_recipe_shape_slots=_shape_map(env.scene["robot"])
        env.mjlab_recipe_manifest=recipe_manifest(training=False,fixed_mu=.7)
        env._normal_material_startup_readonly=True

    def configure(cfg):
        event=cfg.events.mjlab_contact_material
        assert event.func.__name__=="initialize_contact_materials" and event.params==dict(training=False,fixed_mu=.7)
        event.func=fixed_material_metadata

    def audit_materials(env,current):
        import torch
        import eval_ppo_backend_400hz_isaac as physx
        result=recipe_namespace["audit_physx_mjlab_recipe"](env,check_usd=True)
        assert result["recipe"]["training"] is False and result["encoder_bias_range_rad"]==[0.,0.]
        legs=env.scene["robot"].actuators["legs"]
        assert torch.isfinite(legs.stiffness).all() and torch.all(legs.stiffness==500.)
        assert torch.isfinite(legs.damping).all() and torch.all(legs.damping==10.)
        result["leg_pd_gain_readback"]={"kp_range":[500.,500.],"kd_range":[10.,10.]}
        result["effective_pair_mapping"]=copy.deepcopy(current["contract"]["evaluation_contact_friction_override"])
        result["native_plant_readback"]=physx.plant_readback(env)
        return result

    def prepare(stage,robot_path):
        from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema
        import tron1_wheel_geometry as geometry
        assert not prepared
        nonwheel=UsdShade.Material.Define(stage,robot_path+"/normal_nonwheel_material")
        rigid=UsdPhysics.MaterialAPI.Apply(nonwheel.GetPrim())
        rigid.CreateStaticFrictionAttr(.7);rigid.CreateDynamicFrictionAttr(.7);rigid.CreateRestitutionAttr(0.)
        rigid_physx=PhysxSchema.PhysxMaterialAPI.Apply(nonwheel.GetPrim())
        rigid_physx.CreateFrictionCombineModeAttr("multiply");rigid_physx.CreateRestitutionCombineModeAttr("multiply")
        for body_name in reference["body_names"]:
            if body_name in geometry.WHEELS:continue
            body=stage.GetPrimAtPath(robot_path+"/"+body_name)
            for prim in Usd.PrimRange(body):
                if prim.IsA(UsdGeom.Gprim) and geometry.collision_enabled(prim,body):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(nonwheel,materialPurpose="physics")
        for name in geometry.WHEELS:
            body=stage.GetPrimAtPath(robot_path+"/"+name)
            shapes=[p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Gprim) and geometry.collision_enabled(p,body)]
            assert len(shapes)==1
            path=robot_path+"/normal_material_"+name
            assert not stage.GetPrimAtPath(path)
            material=UsdShade.Material.Define(stage,path)
            api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(.7);api.CreateDynamicFrictionAttr(.7);api.CreateRestitutionAttr(0.)
            PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("multiply")
            normal.author(material.GetPrim(),settings)
            UsdShade.MaterialBindingAPI.Apply(shapes[0]).Bind(material,materialPurpose="physics")
            prepared.append(dict(wheel=name,collider=str(shapes[0].GetPath()),material=path))

    def install(env,original_audit):
        import torch,omni.usd
        from pxr import UsdShade
        from physx_mjlab_recipe import _shape_map
        assert len(prepared)==2 and not runtime and env.num_envs==1
        assert env._normal_material_startup_readonly
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
            material_rows.append(dict(**item,before_USD=prior,after_USD=normal.read_material(material.GetPrim())))
        expected=before.clone();expected[:,slots,2]=-settings["stiffness_n_m"]
        after=view.get_material_properties();assert stamp()==clock
        audit=dict(profile=settings,native_before=before.cpu().numpy().tolist(),expected=expected.cpu().numpy().tolist(),native_after=after.cpu().numpy().tolist(),materials=material_rows)
        output=Path(remaining[remaining.index("--output-dir")+1])
        (output/"native_normal_material_update_audit.json").write_text(json.dumps(audit,indent=2))
        assert torch.allclose(after,expected,rtol=1e-6,atol=1e-6),"Native normal profile did not update exactly the wheel shape slots"
        observed=dict(profile=settings,robot_native_mass_kg=mass,preparation=material_rows,
            native_shape_slots=slots,native_material_before=before.cpu().numpy().tolist(),native_material_after=after.cpu().numpy().tolist(),
            only_wheel_normal_material_changed=True,friction_and_nonwheel_materials_unchanged=True,
            no_additional_physical_steps=True,clock_at_intervention=list(clock),
            installation="full profile and fixed wheel friction authored before physics; startup metadata and audits are read-only",
            native_shape_material_tensor_setter_used=False,
            negative_roundtrip_response_difference=roundtrip_errors,
            materialized_readonly_recipe_audit_sha256=hashlib.sha256(recipe_source.encode()).hexdigest(),
            damping_direct_native_tensor_readback_available=False,bench_source_sha256=bench_proof)
        original_audit["normal_contact_model_intervention"]=observed
        original_audit["audit_phase_note"]="Fixed friction, original encoder/PD checks and native signed-stiffness slot verified read-only; no runtime material overwrite"
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
        _normal_prepare=prepare,_normal_configure=configure,_normal_audit_materials=audit_materials,_normal_install=install,_normal_finish=finish)
    previous=sys.argv
    try:
        sys.path.append(str(own.contact_model_evaluator.resolve().parent))
        sys.argv=[str(own.contact_model_evaluator),*remaining]
        exec(compile(source,"<native_normal_model_parent>","exec"),namespace)
        namespace["main"]()
    finally:sys.argv=previous


if __name__=="__main__":main(sys.argv[1:])

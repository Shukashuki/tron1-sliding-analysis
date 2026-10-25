"""Frozen K actor under audited wheel geometry/force models; no PPO updates.

The authored_auto/native arm must reproduce the original native recording
byte for byte. Modified model arms retain all masses, COM/inertias, motors,
reference, control clock, observations, and full-horizon evaluation lifecycle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--base-snapshot",type=Path,required=True)
    p.add_argument("--baseline-dir",type=Path,required=True)
    p.add_argument("--geometry",choices=("authored_auto","authored","slices9"),required=True)
    p.add_argument("--law",choices=("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula"),required=True)
    p.add_argument("--contact-offset-m",type=float)
    p.add_argument("--rest-offset-m",type=float,default=0.)
    a,remaining=p.parse_known_args()
    scripts=a.base_snapshot.resolve()/"scripts"
    assert sha(scripts/"eval_ppo_ground_contact_v2k.py")=="cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c"
    if a.geometry=="authored_auto":
        assert a.law=="native_coulomb" and a.contact_offset_m is None
    else:
        assert a.contact_offset_m is not None
    sys.path.append(str(scripts))
    import numpy as np
    import eval_ppo_ground_contact_v2k as base
    import eval_ppo_backend_400hz_isaac as physx
    import physx_tire_runtime as tire
    import tron1_wheel_geometry as geometry
    assert Path(base.__file__).resolve()==scripts/"eval_ppo_ground_contact_v2k.py"
    assert sha(tire.__file__)=="9b7dd75cbb937496807d74ebda24318aa7016a975c0723233bbec4797b8889fa"
    plan=base.prepare_plan(remaining)
    assert plan["args"].backend=="physx" and plan["mu"]==.7
    assert plan["cohort"]["input_sha256"]["policy.pt"]=="fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912"
    baseline=a.baseline_dir.resolve()
    proof=base.validate_output(baseline,plan["args"].training_dir,plan["args"].protocol,backend="physx")
    baseline_report=json.loads((baseline/"report.json").read_text())
    baseline_plant=baseline_report["contact_material_audit"]["native_plant_readback"]
    generated=physx.materialize_source(plan["evaluator"].read_text(),plan["legacy_root"])
    plan["generated_sha256"]=hashlib.sha256(generated.encode()).hexdigest()
    geometry_audits=[];runtime=[];native_audits=[]

    def configure(cfg,profile,current):
        physx.configure(cfg,profile,current)
        tire.configure_sensors(cfg.scene)
        import carb
        from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
        setting=carb.settings.get_settings()
        before=setting.get(SETTING_COLLISION_APPROXIMATE_CYLINDERS)
        assert before is False,"Do not change the native baseline's cylinder setting"
        if a.geometry!="authored_auto":
            original=cfg.scene.robot.spawn.func
            def spawn(prim_path,spawn_cfg,*args,**kwargs):
                prim=original(prim_path,spawn_cfg,*args,**kwargs)
                assert not geometry_audits,"Single-environment spawn only"
                geometry_audits.extend(geometry.apply_wheel_geometry(prim.GetStage(),str(prim.GetPath()),
                    level=a.geometry,contact_offset_m=a.contact_offset_m,rest_offset_m=a.rest_offset_m))
                return prim
            cfg.scene.robot.spawn.func=spawn

    def audit_materials(env,current):
        import torch
        from physx_mjlab_recipe import _shape_map
        torch.cuda.set_device(current["args"].physical_gpu_index)
        original=physx.audit_materials(env,current)
        observed=original["native_plant_readback"]
        assert observed["body_names"]==baseline_plant["body_names"] and observed["joint_names"]==baseline_plant["joint_names"]
        for name in ("get_masses","get_inertias","get_coms","get_dof_armatures","get_dof_dampings"):
            np.testing.assert_allclose(observed["readbacks"][name]["values"],baseline_plant["readbacks"][name]["values"],rtol=0,atol=1e-7)
        view=env.scene["robot"].root_physx_view
        mapping=_shape_map(env.scene["robot"])
        contact=tire.numpy(view.get_contact_offsets());rest=tire.numpy(view.get_rest_offsets())
        assert contact.shape==rest.shape==(1,view.max_shapes)
        assert np.isfinite(contact).all() and np.isfinite(rest).all()
        wheels={}
        for name in geometry.WHEELS:
            slots=mapping[name];assert len(slots)==(9 if a.geometry=="slices9" else 1)
            wheels[name]=dict(slots=slots,contact_offsets_m=contact[:,slots].tolist(),rest_offsets_m=rest[:,slots].tolist())
            if a.geometry!="authored_auto":
                np.testing.assert_allclose(contact[:,slots],a.contact_offset_m,rtol=0,atol=1e-8)
                np.testing.assert_allclose(rest[:,slots],a.rest_offset_m,rtol=0,atol=1e-8)
        native_audits.append(dict(wheels=wheels,masses_COM_inertias_armature_damping_match_baseline=True,
            shape_mapping=mapping,total_native_shapes=view.max_shapes,contact_offset_readbacks_m=contact.tolist(),
            rest_offset_readbacks_m=rest.tolist()))
        assert not runtime
        runtime.append(tire.TireRuntime(env,law=a.law,mu=.7))
        original["wheel_contact_model_before_force_substitution"]=native_audits[0]
        return original

    def runner(cohort,device):
        result=base.common.StrictPPOExportRunner(cohort,device)
        result.inference_provenance=base.provenance(cohort)
        return result

    def finish(report,contract,current,raw):
        output=current["output"]
        native_proof=base.common.verify_recording(output/"physics_trajectory.npz","PhysX")
        assert report["status"]=="evaluated" and report["deterministic_start_frame"]==0
        assert report["runtime_device_audit"]["status"]=="verified"
        observed=sha(output/"physics_trajectory.npz")
        if a.geometry=="authored_auto":
            assert observed==proof["recording_sha256"],"Read-only native offset audit changed physical trajectory"
        coupling=runtime[0].save(output/"tire_trajectory.npz")
        with np.load(output/"tire_trajectory.npz",allow_pickle=False) as values:
            assert np.max(values["native_patch_count"])<128,"Contact patch buffer may have saturated"
        assert sha(baseline/"physics_trajectory.npz")==proof["recording_sha256"]
        metadata=dict(kind="frozen_K_PhysX_wheel_contact_model_transfer",status="completed_requires_analysis",
            geometry=a.geometry,law=a.law,mu=.7,normalized_actor_sha256=current["cohort"]["input_sha256"]["policy.pt"],
            physics_dt_s=.0025,policy_dt_s=.02,native_states=2793,policy_decisions=349,
            authored_geometry=geometry_audits,native_audit=native_audits[0],runtime=coupling,
            checkpoint_and_actor_provenance=base.provenance(current["cohort"]),
            baseline_recording_sha256=proof["recording_sha256"],baseline_native_proof=proof,
            native_recording_proof=native_proof,physics_trajectory_sha256=observed,
            tire_trajectory_sha256=sha(output/"tire_trajectory.npz"),
            read_only_baseline_byte_identical=observed==proof["recording_sha256"],
            retraining_or_reward_action_reference_change=False,free_absolute_height=True,
            all_physical_samples_retained=True,source_sha256={},task_success_claimed=False)
        for path in (Path(__file__),Path(geometry.__file__),Path(tire.__file__),Path(tire.__file__).with_name("nonlinear_tire_contact.py")):
            metadata["source_sha256"][str(path.resolve())]=sha(path)
        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata
        with (output/"wheel_contact_model_status.json").open("x") as stream:json.dump(metadata,stream,indent=2,allow_nan=False)

    namespace=dict(__name__="_frozen_K_wheel_contact_models",__file__=str(Path(__file__).resolve()),
        _new_plan=plan,_new_configure=configure,_new_audit_materials=audit_materials,
        _new_runner=runner,_new_finish_report=finish)
    os.environ.setdefault("CUDA_DEVICE_ORDER","PCI_BUS_ID")
    previous=sys.argv
    try:
        sys.argv=[str(Path(__file__).resolve()),*plan["inherited_argv"]]
        exec(compile(generated,"<frozen_K_wheel_contact_models>","exec"),namespace)
        namespace["main"]()
    finally:sys.argv=previous


if __name__=="__main__":main()

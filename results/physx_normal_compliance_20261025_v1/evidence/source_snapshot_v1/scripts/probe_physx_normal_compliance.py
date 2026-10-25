"""One loaded-wheel vertical fixture; native rigid/compliant contact telemetry.

All main episodes have the same fixed half-robot fixture mass, geometry and
initial state. XY translation and all rotation are locked only for this
normal-force identification fixture. Z is never prescribed after initialization.
One3s trace includes a1s additional downward load, followed by release. The
rigid arm additionally checks a live USD material switch for0.5s after its
completed main trace, preserving separate records and continuous timestamps.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import physx_normal_compliance as normal
from wheel_contact_geometry import UniformWheel, author_compound


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    assert not args.output_dir.exists() and args.device.startswith("cuda:")
    protocol=json.loads(args.protocol.read_text())
    assert protocol["kind"]=="predeclared_native_normal_compliance_model_sensitivity"
    assert protocol["profiles"]==list(normal.PROFILES)
    assert protocol["bench"]["physics_dt_s"]==.0025 and protocol["bench"]["main_steps"]==1200
    assert protocol["bench"]["vertical_impulse_balance_abs_tolerance_n"]==.1
    assert protocol["bench"]["minimum_native_normal_force_n"]==-1e-4
    assert sha(args.reference_report) == normal.REFERENCE_REPORT_SHA256
    report=json.loads(args.reference_report.read_text())
    robot_mass=float(np.asarray(report["contact_material_audit"]["native_plant_readback"]["readbacks"]["get_masses"]["values"]).sum())
    mass=robot_mass/2; settings=normal.profile(args.profile,mass);descriptor=UniformWheel(mass_kg=mass)
    dt=.0025;ordinal=int(args.device.split(":")[1])
    from isaaclab.app import AppLauncher
    app=AppLauncher(argparse.Namespace(headless=True,device=args.device,enable_cameras=False,
        kit_args=f"--/renderer/activeGpu={ordinal} --/renderer/multiGpu/enabled=false --/physics/cudaDevice={ordinal} --/physics/sceneMultiGPUMode=0")).app
    try:
        import torch
        torch.cuda.set_device(ordinal)
        import carb
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject,RigidObjectCfg
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
        from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
        from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
        carb.settings.get_settings().set_bool(SETTING_COLLISION_APPROXIMATE_CYLINDERS,False)
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=args.device));stage=sim.stage
        args.output_dir.mkdir()
        ground=UsdGeom.Mesh.Define(stage,"/World/Ground")
        ground.CreatePointsAttr([(-10,-10,0),(10,-10,0),(10,10,0),(-10,10,0)])
        ground.CreateFaceVertexCountsAttr([3,3]);ground.CreateFaceVertexIndicesAttr([0,1,2,0,2,3]);ground.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr("none")
        gm=UsdShade.Material.Define(stage,"/World/GroundMaterial")
        api=UsdPhysics.MaterialAPI.Apply(gm.GetPrim());api.CreateStaticFrictionAttr(1.);api.CreateDynamicFrictionAttr(1.);api.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(gm.GetPrim()).CreateFrictionCombineModeAttr("multiply")
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(gm,materialPurpose="physics")
        cfg=sim_utils.CylinderCfg(radius=.127,height=.05,axis="Z",activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.,angular_damping=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.7,restitution=0.,friction_combine_mode="multiply"))
        wheel=RigidObject(RigidObjectCfg(prim_path="/World/Wheel",spawn=cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,.1271))))
        shape=UsdGeom.Xformable(stage.GetPrimAtPath("/World/Wheel/geometry/mesh"))
        op=next(o for o in shape.GetOrderedXformOps() if o.GetOpType()==UsdGeom.XformOp.TypeOrient)
        op.Set(Gf.Quatd(np.sqrt(.5),Gf.Vec3d(np.sqrt(.5),0,0)) if op.GetPrecision()==UsdGeom.XformOp.PrecisionDouble
               else Gf.Quatf(np.sqrt(.5),Gf.Vec3f(np.sqrt(.5),0,0)))
        geometry=author_compound(stage,"/World/Wheel","/World/Wheel/geometry/mesh",descriptor)
        material=stage.GetPrimAtPath(geometry["physics_material_path"])
        authored=normal.author(material,settings)
        fixture=PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Wheel"))
        fixture.CreateLockedPosAxisAttr(3);fixture.CreateLockedRotAxisAttr(7)
        sensor=ContactSensor(ContactSensorCfg(prim_path="/World/Wheel",update_period=0.,history_length=0,
            filter_prim_paths_expr=["/World/Ground"],track_contact_points=True,max_contact_data_count_per_prim=128))
        sim.reset();wheel.reset();sensor.reset()
        plain=lambda value:value.detach().cpu().numpy().copy()
        view=wheel.root_physx_view
        np.testing.assert_allclose(plain(view.get_masses()),mass,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(plain(view.get_inertias()).reshape(3,3),np.diag(descriptor.diagonal_inertia_kg_m2),rtol=1e-5,atol=1e-7)
        np.testing.assert_allclose(plain(view.get_coms())[...,:3],0.,rtol=0,atol=1e-8)
        np.testing.assert_allclose(plain(view.get_contact_offsets()),.001,rtol=0,atol=1e-8)
        np.testing.assert_allclose(plain(view.get_rest_offsets()),0.,rtol=0,atol=1e-8)
        native_material=plain(view.get_material_properties())
        assert native_material.shape==(1,1,3) and view.max_shapes==1
        np.testing.assert_allclose(native_material[...,:2],.7,rtol=0,atol=1e-6)
        np.testing.assert_allclose(native_material[...,2],-settings["stiffness_n_m"],rtol=1e-6,atol=1e-5)
        pose=torch.tensor([[0.,0.,.1271,1.,0.,0.,0.]],device=args.device)
        velocity=torch.tensor([[0.,0.,-.3,0.,0.,0.]],device=args.device)
        wheel.write_root_pose_to_sim(pose);wheel.write_root_velocity_to_sim(velocity)
        sources={str(Path(p).resolve()):sha(p) for p in (__file__,normal.__file__,Path(__file__).with_name("wheel_contact_geometry.py"),args.reference_report,args.protocol)}
        def observe():
            state=plain(wheel.data.root_state_w)[0];force=plain(sensor.data.force_matrix_w)[0,0,0]
            assert state.shape==(13,) and np.isfinite(state).all() and np.isfinite(force).all()
            assert force[2]>=-1e-4,"Native normal material generated tensile force"
            assert np.max(np.abs(state[[0,1,7,8,10,11,12]]))<1e-4,"Vertical fixture constraints failed"
            return state,force
        state,force=observe();states=[state];forces=[force];loads=[];counters=[sim.current_time_step_index]
        def step(load):
            nonlocal state,force
            wheel.set_external_force_and_torque(torch.tensor([[[0.,0.,-load]]],device=args.device),
                torch.zeros((1,1,3),device=args.device),positions=torch.as_tensor(state[None,None,:3],device=args.device,dtype=torch.float32),is_global=True)
            wheel.write_data_to_sim();sim.step(render=False);wheel.update(dt);sensor.update(dt)
            state,force=observe()
        for index in range(1200):
            load=mass*9.81 if 400<=index<800 else 0.
            step(load);states.append(state);forces.append(force);loads.append(load);counters.append(sim.current_time_step_index)
        arrays=dict(time_s=np.arange(1201)*dt,root_state_w=np.asarray(states),ground_normal_force_w_n=np.asarray(forces),
            applied_extra_downward_load_n=np.asarray(loads),native_step_counter=np.asarray(counters))
        assert np.array_equal(np.diff(arrays["native_step_counter"]),np.ones(1200))
        residual=mass*np.diff(arrays["root_state_w"][:,9])/dt-arrays["ground_normal_force_w_n"][1:,2]+mass*9.81+arrays["applied_extra_downward_load_n"]
        assert np.max(np.abs(residual))<.1,"Native vertical impulse balance failed"
        np.savez_compressed(args.output_dir/"native_trace.npz",**arrays)
        live=None
        if args.profile=="rigid":
            update=normal.profile("compliant_3mm_d1",mass)
            before_counter=sim.current_time_step_index
            live_authored=normal.author(material,update)
            live_native=plain(view.get_material_properties())
            assert sim.current_time_step_index==before_counter
            consumed=bool(np.allclose(live_native[...,2],-update["stiffness_n_m"],rtol=1e-6,atol=1e-5))
            live=dict(USD_update_without_physics_consumed=consumed,authored=live_authored,native_material_readback=live_native.tolist())
            if consumed:
                live_states=[state];live_forces=[force]
                for _ in range(200):
                    step(0.);live_states.append(state);live_forces.append(force)
                np.savez_compressed(args.output_dir/"live_switch_trace.npz",time_s=3.+np.arange(201)*dt,
                    root_state_w=np.asarray(live_states),ground_normal_force_w_n=np.asarray(live_forces))
                live.update(physical_steps=200,trace_sha256=sha(args.output_dir/"live_switch_trace.npz"))
        status=dict(kind="PhysX_loaded_wheel_native_normal_compliance_bench",status="completed_requires_analysis",
            profile=settings,robot_reference_mass_kg=robot_mass,fixture_mass_kg=mass,
            fixture_is_half_robot_equivalent_vertical_mass_not_original_wheel_mass=True,
            fixture_XY_translation_and_all_rotation_locked=True,Z_position_prescribed_during_episode=False,
            geometry=geometry,authored_material=authored,native_material_readback=native_material.tolist(),
            negative_native_restitution_slot_represents_spring_stiffness=True,
            damping_direct_native_tensor_readback_available=False,physics_dt_s=dt,
            main_physical_steps=1200,main_native_states=1201,live_switch=live,
            peak_impulse_balance_error_n=float(np.abs(residual).max()),native_trace_sha256=sha(args.output_dir/"native_trace.npz"),
            source_sha256=sources,official_model_documentation=normal.DOC,policy_or_robot_turning_task_result=False,task_success_claimed=False,
            protocol_sha256=sha(args.protocol),
            torch_current_device=torch.cuda.current_device(),native_state_tensor_device=str(wheel.data.root_state_w.device))
        with (args.output_dir/"status.json").open("x") as stream:json.dump(status,stream,indent=2,allow_nan=False)
        print(json.dumps(status))
    except BaseException as error:
        import traceback
        if args.output_dir.is_dir():
            (args.output_dir/"failure.json").write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),task_success_claimed=False),indent=2))
        raise
    finally:
        app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--profile",choices=normal.PROFILES,required=True)
    p.add_argument("--reference-report",type=Path,required=True)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--device",required=True);p.add_argument("--output-dir",type=Path,required=True)
    run(p.parse_args())

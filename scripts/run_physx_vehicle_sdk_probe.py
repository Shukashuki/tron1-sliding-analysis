"""Installed PhysX Vehicle SDK bring-up, separate from the matched tire-law arms.

The native Vehicle tire/suspension update replaces rigid wheel-ground contact.
It is not a Pacejka implementation and not an isolated tire-curve ablation.
The default installed NVIDIA sample is used through its API, then explicitly
resized to the declared source-car surrogate. No original CARLA plant claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def native_json(value):
    if all(hasattr(value,k) for k in ("x","y","z")):
        return [float(value.x),float(value.y),float(value.z)]+([float(value.w)] if hasattr(value,"w") else [])
    raise TypeError(type(value).__name__)


def run(a):
    assert not a.output_dir.exists() and a.mu in (.7,3.5)
    assert a.render_gpu in (3,5,4)
    with np.load(a.reference,allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
    assert hashlib.sha256(a.reference.read_bytes()).hexdigest()=="1b4852178af7e1cd4d04aee4d42d86a7d12307f5ceb004c1b099b073683f298c"
    from isaaclab.app import AppLauncher
    app=AppLauncher(argparse.Namespace(headless=True,device="cuda:"+str(a.render_gpu),enable_cameras=False,
        kit_args=f"--/renderer/activeGpu={a.render_gpu} --/renderer/multiGpu/enabled=false --/physics/sceneMultiGPUMode=0")).app
    try:
        import torch
        torch.cuda.set_device(a.render_gpu)
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.physx.vehicle")
        import omni.physx
        from omni.physx.bindings import _physx
        from omni.physxvehicle.scripts.samples import BasicSetup
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject,RigidObjectCfg
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
        a.output_dir.mkdir()
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.0025,device="cpu",
            physics_prim_path="/World/PhysicsScene",gravity=(0.,-9.81,0.),enable_scene_query_support=True))
        stage=sim.stage;world=UsdGeom.Xform.Define(stage,"/World");stage.SetDefaultPrim(world.GetPrim())
        paths=BasicSetup.create(stage,False)
        scene=UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene"));scene.GetGravityMagnitudeAttr().Set(9.81)
        assert str(UsdGeom.GetStageUpAxis(stage))=="Y"
        vehicle=stage.GetPrimAtPath(paths.vehiclePath)
        geom=UsdGeom.Xformable(vehicle)
        next(op for op in geom.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3f(0.,3.,0.))
        mass=UsdPhysics.MassAPI(vehicle);mass.GetCenterOfMassAttr().Set(Gf.Vec3f(0.,0.,0.))
        mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1800/12*(.4**2+3.6**2),1800/12*(1.6**2+3.6**2),1800/12*(1.6**2+.4**2)))
        collision=UsdGeom.Xformable(stage.GetPrimAtPath(paths.vehiclePath+"/ChassisCollision"))
        next(op for op in collision.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(.8,.2,1.8))
        PhysxSchema.PhysxVehicleDriveBasicAPI(vehicle).GetPeakTorqueAttr().Set(3000.)
        PhysxSchema.PhysxVehicleSteeringAPI(vehicle).GetMaxSteerAngleAttr().Set(.6)
        for token in ("brakes0","brakes1"):PhysxSchema.PhysxVehicleBrakesAPI(vehicle,token).GetMaxBrakeTorqueAttr().Set(2000.)
        for p in stage.Traverse():
            if p.IsA(PhysxSchema.PhysxVehicleTireFrictionTable):PhysxSchema.PhysxVehicleTireFrictionTable(p).GetFrictionValuesAttr().Set([a.mu,a.mu])
        wheel_positions=[]
        for k,path in enumerate(paths.wheelAttachmentPaths):
            p=stage.GetPrimAtPath(path);position=Gf.Vec3f(.7 if k%2==0 else -.7,-.25,1.2 if k<2 else -1.2)
            wheel_positions.append(list(position))
            xform=UsdGeom.Xformable(p)
            next(op for op in xform.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate).Set(position)
            attach=PhysxSchema.PhysxVehicleWheelAttachmentAPI(p)
            attach.GetSuspensionFramePositionAttr().Set(Gf.Vec3f(position[0],position[1]+450*9.81/45000,position[2]))
            suspension=PhysxSchema.PhysxVehicleSuspensionAPI(p)
            suspension.GetTravelDistanceAttr().Set(2*450*9.81/45000)
            wheel=PhysxSchema.PhysxVehicleWheelAPI(p);wheel.GetRadiusAttr().Set(.3);wheel.GetWidthAttr().Set(.18);wheel.GetMoiAttr().Set(.9)
            for suffix in ("Collision","Render"):
                shape=UsdGeom.Cylinder(stage.GetPrimAtPath(path+"/"+suffix))
                shape.GetRadiusAttr().Set(.3);shape.GetHeightAttr().Set(.18)
                shape.GetExtentAttr().Set(UsdGeom.Cylinder.ComputeExtentFromPlugins(shape,0))
            tire=PhysxSchema.PhysxVehicleTireAPI(p)
            tire.GetLateralStiffnessGraphAttr().Set(Gf.Vec2f(2.,(17. if k<2 else 25.)*450*9.81))
        obj=RigidObject(RigidObjectCfg(prim_path=paths.vehiclePath,spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,3.,0.))))
        controller=PhysxSchema.PhysxVehicleControllerAPI(vehicle)
        interface=omni.physx.get_physx_interface()
        sim.reset();obj.reset()
        constants={n:getattr(_physx,n) for n in dir(_physx) if n.startswith("VEHICLE_WHEEL_STATE_")}
        methods={n:hasattr(interface,n) for n in ("get_wheel_state","set_wheel_rotation_speed","set_wheel_rotation_angle")}
        assert all(methods.values())
        settling=[]
        for k in range(800):
            sim.step(render=False);obj.update(.0025)
            if k in (0,1,10,100,200,400,799):
                settling.append({"step":k+1,"root_state":obj.data.root_state_w[0].detach().cpu().numpy().tolist(),
                    "wheel_states":[interface.get_wheel_state(path) for path in paths.wheelAttachmentPaths]})
        wheel_state=[interface.get_wheel_state(path) for path in paths.wheelAttachmentPaths]
        assert all(wheel_state),"Native Vehicle SDK did not initialize wheel states"
        ground=[bool(s[constants["VEHICLE_WHEEL_STATE_IS_ON_GROUND"]]) for s in wheel_state]
        bearing=[float(s[constants["VEHICLE_WHEEL_STATE_SUSPENSION_FORCE"]][1])>5. for s in wheel_state]
        settled=obj.data.root_state_w[0].detach().cpu().numpy().copy()
        preflight={"physics_device":"cpu","renderer_physical_gpu":a.render_gpu,"wheel_state_methods":methods,
            "scene_queries_enabled":sim.cfg.enable_scene_query_support,
            "wheel_state_keys":list(wheel_state[0]),"wheel_state_constants":constants,"settled_root_state_native_Yup":settled.tolist(),
            "settling_readbacks":settling,"stage_gravity_direction":list(scene.GetGravityDirectionAttr().Get()),
            "vehicle_vertical_axis":str(PhysxSchema.PhysxVehicleContextAPI(scene.GetPrim()).GetVerticalAxisAttr().Get()),
            "stationary_initial_spawn_height_native_Yup_m":3.,
            "settled_wheels_on_ground":ground,"native_sample_source":str(Path(BasicSetup.__file__)),
            "settled_wheels_suspension_upward_force_gt5N":bearing,
            "is_on_ground_caution":"Official PhysX API reports false for sleeping vehicles; use suspension-force readback and settled geometry for this stationary initialization check.",
            "native_sample_sha256":hashlib.sha256(Path(BasicSetup.__file__).read_bytes()).hexdigest(),
            "native_vehicle_suspension_and_tire_forces":True,"external_load_delay":False,
            "same_plant_as_rigid_axle_surrogate":False,"tire_curve_only_ablation":False,
            "COM_height_assumption_m":.55,"suspension_spring_N_m":45000.,"suspension_damper_N_s_m":4500.,
            "wheel_positions_relative_to_chassis_native_Yup":wheel_positions,"exact_original_CARLA_car":False}
        (a.output_dir/"preflight.json").write_text(json.dumps(preflight,indent=2,default=native_json)+chr(10))
        assert sum(bearing)>=3 and .4<settled[1]<.7,"SDK failed stationary load-bearing validation"
        v0=ref["velocity_xy_m_s"][0];r0=float(np.gradient(ref["yaw_rad"],ref["time_s_estimated"])[0])
        # Initialize once. The native tire SDK integrates all subsequent states.
        obj.write_root_velocity_to_sim(torch.tensor([[v0[1],0.,v0[0],0.,r0,0.]],dtype=torch.float32))
        pose=obj.data.root_pose_w.clone();pose[:,:]=torch.tensor([[0.,float(settled[1]),0.,1.,0.,0.,0.]])
        obj.write_root_pose_to_sim(pose)
        for k,path in enumerate(paths.wheelAttachmentPaths):
            interface.set_wheel_rotation_speed(path,float((v0[0]-r0*(.7 if k%2==0 else -.7))/.3))
            interface.set_wheel_rotation_angle(path,0.)
        fields=("ROTATION_SPEED","STEER_ANGLE","GROUND_HIT_POSITION","SUSPENSION_JOUNCE","SUSPENSION_FORCE","TIRE_FRICTION",
            "TIRE_LONGITUDINAL_DIRECTION","TIRE_LATERAL_DIRECTION","TIRE_LONGITUDINAL_SLIP","TIRE_LATERAL_SLIP","TIRE_FORCE","IS_ON_GROUND")
        records=[];controls=[]
        def read():
            state=obj.data.root_state_w[0].detach().cpu().numpy().copy()
            # Fixed right-handed basis mapping: sourceXYZ = nativeZXY.
            converted=np.r_[state[:3][[2,0,1]],state[3:7][[0,3,1,2]],state[7:10][[2,0,1]],state[10:13][[2,0,1]]]
            value={"root_state_source_Zup":converted}
            states=[interface.get_wheel_state(path) for path in paths.wheelAttachmentPaths]
            for key in fields:
                values=[s[constants["VEHICLE_WHEEL_STATE_"+key]] for s in states]
                value[key.lower()]=np.asarray([native_json(v) if hasattr(v,"x") else v for v in values])
            assert np.isfinite(converted).all()
            return value
        records.append(read());count=40 if a.smoke else 992
        for k in range(count):
            t=k*.0025
            if k%8==0:
                idx=min(np.searchsorted(ref["time_s_estimated"],t,side="right")-1,len(ref["time_s_estimated"])-1)
                command=ref["normalized_steer_throttle_brake_handbrake"][idx]
                controller.GetSteerAttr().Set(float(command[0]));controller.GetAcceleratorAttr().Set(float(command[1]))
                controller.GetBrake0Attr().Set(float(command[2]));controller.GetBrake1Attr().Set(float(command[3]))
            controls.append(command.copy());sim.step(render=False);obj.update(.0025);records.append(read())
        arrays={key:np.asarray([r[key] for r in records]) for key in records[0]}
        arrays.update(time_s=np.arange(count+1)*.0025,source_controls=np.asarray(controls))
        np.savez_compressed(a.output_dir/"native_trace.npz",**arrays)
        status={"status":"completed_requires_analysis","kind":"installed_PhysX_Vehicle_SDK_suspension_tire_model",
            "smoke_only":a.smoke,"physics_dt_s":.0025,"native_states":count+1,"mu":a.mu,
            "same_plant_as_rigid_axle_surrogate":False,"tire_curve_only_ablation":False,
            "exact_source_vehicle_reproduced":False,"native_default_tire_model_is_Pacejka":False,
            "preflight":preflight,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_trace_sha256":hashlib.sha256((a.output_dir/"native_trace.npz").read_bytes()).hexdigest(),"task_success_claimed":False}
        (a.output_dir/"status.json").write_text(json.dumps(status,indent=2,default=native_json)+chr(10));print(json.dumps(status,default=native_json),flush=True)
    except BaseException as e:
        import traceback
        v={"status":"failed_no_task_result","error":str(e),"traceback":traceback.format_exc()}
        if a.output_dir.is_dir():(a.output_dir/"failure.json").write_text(json.dumps(v,indent=2)+chr(10))
        print(json.dumps(v),file=sys.stderr,flush=True);raise
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--mu",type=float,required=True)
    p.add_argument("--render-gpu",type=int,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    run(p.parse_args())

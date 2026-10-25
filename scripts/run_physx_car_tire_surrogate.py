"""Unscaled traj_4 open-loop PhysX FWD-car sensitivity, with explicit assumptions.

This is not the unavailable exact CARLA0.9.5 vehicle. Native rigid joints and
normal contacts are used for every law. Source steering/throttle are converted
through the declared surrogate maps. No per-step root or wheel pose playback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from nonlinear_tire_contact import TireParameters, tangent_wrench


def run(a):
    assert not a.output_dir.exists() and a.mu in (.7,3.5) and a.device in ("cuda:3","cuda:5","cuda:4")
    cfg=json.loads(a.config.read_text())
    assert cfg["kind"]=="explicitly_assumed_Audi_A2_size_FWD_PhysX_surrogate_not_original_CARLA_plant"
    dt=cfg["physics_dt_s"];assert dt==.0025
    with np.load(a.reference,allow_pickle=False) as z:reference={k:z[k].copy() for k in z.files}
    assert hashlib.sha256(a.reference.read_bytes()).hexdigest()=="1b4852178af7e1cd4d04aee4d42d86a7d12307f5ceb004c1b099b073683f298c"
    count=int(np.floor(reference["time_s_estimated"][-1]/dt))
    if a.smoke:count=40
    from isaaclab.app import AppLauncher
    ordinal=int(a.device.split(":")[1])
    launcher=AppLauncher(argparse.Namespace(headless=True,device=a.device,enable_cameras=False,
        kit_args=f"--/renderer/activeGpu={ordinal} --/renderer/multiGpu/enabled=false --/physics/cudaDevice={ordinal} --/physics/sceneMultiGPUMode=0"))
    app=launcher.app
    try:
        import torch
        torch.cuda.set_device(ordinal)
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation,ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,UsdShade
        a.output_dir.mkdir()
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=a.device))
        stage=sim.stage
        floor=UsdGeom.Mesh.Define(stage,"/World/Ground")
        floor.CreatePointsAttr([(-200.,-200.,0.),(200.,-200.,0.),(200.,200.,0.),(-200.,200.,0.)])
        floor.CreateFaceVertexCountsAttr([3,3]);floor.CreateFaceVertexIndicesAttr([0,1,2,0,2,3]);floor.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(floor.GetPrim()).CreateApproximationAttr("none")
        material=UsdShade.Material.Define(stage,"/World/GroundMaterial")
        api=UsdPhysics.MaterialAPI.Apply(material.GetPrim());api.CreateStaticFrictionAttr(1.);api.CreateDynamicFrictionAttr(1.);api.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("multiply")
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(material,materialPurpose="physics")
        UsdGeom.Xform.Define(stage,"/World/Car")
        body_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.,angular_damping=0.,
            solver_position_iteration_count=32,solver_velocity_iteration_count=4)
        def mat(mu):return sim_utils.RigidBodyMaterialCfg(static_friction=mu,dynamic_friction=mu,restitution=0.,friction_combine_mode="multiply")
        chassis=sim_utils.CuboidCfg(size=tuple(cfg["body_dimensions_m"]),rigid_props=body_props,
            mass_props=sim_utils.MassPropertiesCfg(mass=cfg["chassis_mass_kg"]),collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=mat(a.mu))
        chassis.func("/World/Car/chassis",chassis,translation=(0.,0.,cfg["chassis_initial_center_z_m"]))
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/Car/chassis"))
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath("/World/Car/chassis")).CreateEnabledSelfCollisionsAttr(False)
        wheels=("FL","FR","RL","RR")
        def joint(name,parent,child,axis,anchor):
            j=UsdPhysics.RevoluteJoint.Define(stage,"/World/Car/"+name)
            j.CreateBody0Rel().SetTargets([parent]);j.CreateBody1Rel().SetTargets([child]);j.CreateAxisAttr(axis)
            j.CreateLocalPos0Attr(Gf.Vec3f(*anchor));j.CreateLocalPos1Attr(Gf.Vec3f(0.,0.,0.))
            j.CreateCollisionEnabledAttr(False)
            return j
        for label in wheels:
            x=(1 if label[0]=="F" else -1)*cfg["wheelbase_m"]/2
            y=(1 if label[1]=="L" else -1)*cfg["track_m"]/2
            z=cfg["wheel_radius_m"];path="/World/Car/wheel_"+label
            cylinder=sim_utils.CylinderCfg(radius=z,height=cfg["wheel_width_m"],axis="Z",
                activate_contact_sensors=True,rigid_props=body_props,mass_props=sim_utils.MassPropertiesCfg(mass=cfg["wheel_mass_kg"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=mat(.7))
            cylinder.func(path,cylinder,translation=(x,y,z))
            geometry=UsdGeom.Xformable(stage.GetPrimAtPath(path+"/geometry/mesh"))
            orient=next(op for op in geometry.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeOrient)
            if orient.GetPrecision()==UsdGeom.XformOp.PrecisionDouble:orient.Set(Gf.Quatd(np.sqrt(.5),Gf.Vec3d(np.sqrt(.5),0.,0.)))
            else:orient.Set(Gf.Quatf(np.sqrt(.5),Gf.Vec3f(np.sqrt(.5),0.,0.)))
            parent="/World/Car/chassis";anchor=(x,y,z-cfg["chassis_initial_center_z_m"])
            if label[0]=="F":
                parent="/World/Car/knuckle_"+label
                knuckle=sim_utils.SphereCfg(radius=.025,rigid_props=body_props,mass_props=sim_utils.MassPropertiesCfg(mass=cfg["front_knuckle_mass_kg"]))
                knuckle.func(parent,knuckle,translation=(x,y,z))
                steer=joint("steer_"+label,"/World/Car/chassis",parent,"Z",anchor)
                steer.CreateLowerLimitAttr(-35.);steer.CreateUpperLimitAttr(35.);anchor=(0.,0.,0.)
            joint("spin_"+label,parent,path,"Y",anchor)
        car=Articulation(ArticulationCfg(prim_path="/World/Car",spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,cfg["chassis_initial_center_z_m"])),
            actuators={"spin":ImplicitActuatorCfg(joint_names_expr=["spin_.*"],stiffness=0.,damping=0.,effort_limit_sim=5000.,velocity_limit_sim=1000.),
                "steer":ImplicitActuatorCfg(joint_names_expr=["steer_.*"],stiffness=cfg["steering_PD_stiffness"],damping=cfg["steering_PD_damping"],effort_limit_sim=100000.,velocity_limit_sim=20.)}))
        sensors=[ContactSensor(ContactSensorCfg(prim_path="/World/Car/wheel_"+n,update_period=0.,history_length=0,
            filter_prim_paths_expr=["/World/Ground"],track_contact_points=True,max_contact_data_count_per_prim=128)) for n in wheels]
        sim.reset();car.reset()
        for sensor in sensors:sensor.reset()
        wheel_ids=[car.body_names.index("wheel_"+n) for n in wheels]
        spin_ids=[car.joint_names.index("spin_"+n) for n in wheels]
        steer_ids=[car.joint_names.index("steer_"+n) for n in wheels[:2]]
        def update():
            car.update(dt)
            for sensor in sensors:sensor.update(dt)
        for _ in range(400):car.write_data_to_sim();sim.step(render=False);update()
        mass=float(car.root_physx_view.get_masses().sum().cpu())
        assert np.isclose(mass,1800.,rtol=0,atol=.01)
        settled=car.data.root_state_w[0].detach().cpu().numpy().copy()
        (a.output_dir/"settled_preflight.json").write_text(json.dumps({"root_state_w":settled.tolist(),"mass_kg":mass,
            "joint_names":list(car.joint_names),"body_names":list(car.body_names)},indent=2)+chr(10))
        assert .5<settled[2]<.6,"Car did not establish nominal upright support"
        v0=reference["velocity_xy_m_s"][0];r0=float(np.gradient(reference["yaw_rad"],reference["time_s_estimated"])[0])
        car.write_root_pose_to_sim(torch.tensor([[0.,0.,cfg["chassis_initial_center_z_m"],1.,0.,0.,0.]],device=a.device))
        car.write_root_velocity_to_sim(torch.tensor([[v0[0],v0[1],0.,0.,0.,r0]],device=a.device,dtype=torch.float32))
        jp=torch.zeros((1,car.num_joints),device=a.device);jv=torch.zeros_like(jp)
        delta=float(reference["normalized_steer_throttle_brake_handbrake"][0,0])*cfg["normalized_steering_scale_rad"]
        jp[:,steer_ids]=delta
        for k,idx in enumerate(spin_ids):
            offset_y=(1 if k%2==0 else -1)*cfg["track_m"]/2
            longitudinal=v0[0]-r0*offset_y
            if k<2:longitudinal=longitudinal*np.cos(delta)+(v0[1]+r0*cfg["wheelbase_m"]/2)*np.sin(delta)
            jv[:,idx]=longitudinal/cfg["wheel_radius_m"]
        car.write_joint_state_to_sim(jp,jv)
        sim.forward();update()
        # All law/mu arms establish the same initial contact state at mu=.7.
        # Change only the wheel material after that common initialization.
        before=car.root_physx_view.get_material_properties().clone();after=before.clone()
        shape_counts=[car._physics_sim_view.create_rigid_body_view(p).max_shapes for p in car.root_physx_view.link_paths[0]]
        assert sum(shape_counts)==car.root_physx_view.max_shapes
        slots=[];start=0
        for i,n in enumerate(shape_counts):
            if i in wheel_ids:slots.extend(range(start,start+n))
            start+=n
        assert len(slots)==4
        after[:,slots,:2]=a.mu if a.law=="native_coulomb" else 0.
        car.root_physx_view.set_material_properties(after,torch.tensor([0],dtype=torch.int32))
        assert torch.allclose(car.root_physx_view.get_material_properties(),after,atol=1e-6,rtol=0)
        parameters=TireParameters(mu=a.mu)
        rows=[];actions=[];applied=[];moments=[]
        def npy(t):return t.detach().cpu().numpy().copy()
        def read():
            forces=[];points=[];friction=[]
            for sensor in sensors:
                forces.append(npy(sensor.data.force_matrix_w)[0,0,0]);points.append(npy(sensor.data.contact_pos_w)[0,0,0])
                f,_,c,s=sensor.contact_physx_view.get_friction_data(dt);c,s=int(c[0,0]),int(s[0,0])
                friction.append(npy(f[s:s+c]).sum(axis=0))
            state={"root_state_w":npy(car.data.root_state_w)[0],"joint_pos_rad":npy(car.data.joint_pos)[0],
                "joint_vel_rad_s":npy(car.data.joint_vel)[0],"wheel_COM_position_w_m":npy(car.data.body_com_pos_w)[0,wheel_ids],
                "wheel_COM_velocity_w_m_s":npy(car.data.body_com_lin_vel_w)[0,wheel_ids],
                "wheel_angular_velocity_w_rad_s":npy(car.data.body_com_ang_vel_w)[0,wheel_ids],
                "wheel_quaternion_wxyz":npy(car.data.body_link_quat_w)[0,wheel_ids],
                "ground_normal_force_w_n":np.asarray(forces),"contact_position_w_m":np.asarray(points),"native_friction_force_w_n":np.asarray(friction)}
            assert np.isfinite(state["root_state_w"]).all()
            return state
        state=read();rows.append(state)
        for step in range(count):
            t=step*dt
            if step%8==0:
                idx=min(int(np.searchsorted(reference["time_s_estimated"],t,side="right")-1),len(reference["time_s_estimated"])-1)
                command=reference["normalized_steer_throttle_brake_handbrake"][idx].copy()
                car.set_joint_position_target(torch.full((1,2),float(command[0]*cfg["normalized_steering_scale_rad"]),device=a.device),joint_ids=steer_ids)
            effort=np.zeros(4);effort[:2]=command[1]*cfg["max_front_wheel_drive_torque_each_nm"]
            brake=np.full(4,command[2]);brake[2:]=np.maximum(brake[2:],command[3])
            effort-=brake*cfg["max_all_wheel_brake_torque_each_nm"]*np.sign(state["joint_vel_rad_s"][spin_ids])
            car.set_joint_effort_target(torch.as_tensor(effort[None],device=a.device,dtype=torch.float32),joint_ids=spin_ids)
            F,T=np.zeros((4,3)),np.zeros((4,3))
            for k in range(4):
                Fn=max(float(state["ground_normal_force_w_n"][k,2]),0.)
                if Fn<=0 or a.law=="native_coulomb":continue
                point=state["contact_position_w_m"][k];assert np.isfinite(point).all()
                w,x,y,z=state["wheel_quaternion_wxyz"][k];axle=[2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)]
                value=tangent_wrench(state["wheel_COM_velocity_w_m_s"][k],state["wheel_angular_velocity_w_rad_s"][k],
                    point-state["wheel_COM_position_w_m"][k],axle,[0,0,1],Fn,law=a.law,parameters=parameters)
                F[k],T[k]=value["force_w_n"],value["torque_about_COM_w_nm"]
            if a.law!="native_coulomb":
                if step>0:assert np.linalg.norm(state["native_friction_force_w_n"])<1e-3
                car.set_external_force_and_torque(torch.as_tensor(F[None],device=a.device,dtype=torch.float32),
                    torch.as_tensor(T[None],device=a.device,dtype=torch.float32),body_ids=wheel_ids,
                    positions=torch.as_tensor(state["wheel_COM_position_w_m"][None],device=a.device,dtype=torch.float32),is_global=True)
            actions.append(np.r_[command,effort]);applied.append(F);moments.append(T)
            car.write_data_to_sim();sim.step(render=False);update();state=read();rows.append(state)
        trace={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
        trace.update(time_s=np.arange(count+1)*dt,source_controls_and_wheel_torques=np.asarray(actions),
            applied_tangent_force_w_n=np.asarray(applied),applied_tangent_moment_COM_w_nm=np.asarray(moments))
        np.savez_compressed(a.output_dir/"native_trace.npz",**trace)
        status={"status":"completed_requires_analysis","kind":cfg["kind"],"law":a.law,"mu":a.mu,
            "physics_steps":count,"smoke_only":a.smoke,"native_states":count+1,"mass_kg":mass,
            "root_pose_writes_at_initialization_only":True,"per_step_root_pose_prescription":False,
            "exact_original_CARLA_plant_reproduced":False,"control_mapping_identified":False,
            "native_trace_sha256":hashlib.sha256((a.output_dir/"native_trace.npz").read_bytes()).hexdigest(),
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name("nonlinear_tire_contact.py"),a.config,a.reference]},
            "config":cfg,"initial_mechanical_state":{k:v.tolist() for k,v in rows[0].items()},"task_success_claimed":False}
        status["material_substitution"]={"wheel_shape_slots":slots,"before":npy(before).tolist(),"after":npy(after).tolist(),
            "common_initialization_wheel_mu":.7,"per_step_material_updates":False,"nonwheel_materials_changed_by_substitution":False}
        (a.output_dir/"status.json").write_text(json.dumps(status,indent=2)+chr(10));print(json.dumps(status),flush=True)
    except BaseException as e:
        import traceback
        value={"status":"failed_no_task_result","error":str(e),"traceback":traceback.format_exc()}
        if a.output_dir.is_dir():(a.output_dir/"failure.json").write_text(json.dumps(value,indent=2)+chr(10))
        print(json.dumps(value),file=sys.stderr,flush=True)
        raise
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--law",choices=("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula"),required=True)
    p.add_argument("--mu",type=float,required=True);p.add_argument("--device",required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--smoke",action="store_true")
    run(p.parse_args())

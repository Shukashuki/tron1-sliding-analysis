"""Single-wheel PhysX contact-law bench, BEFORE car/TRON1 task comparisons.

One free wheel, one flat floor, one fresh process/condition. Native normal
contact is retained. External-law arms have zero native tangential friction,
use the preceding physical sample's measured load/point, and apply a tangential
force plus its COM moment. This explicit one-step coupling is not a built-in
PhysX Vehicle/Pacejka implementation. No policy or task success is claimed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from nonlinear_tire_contact import TireParameters, tangent_wrench


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--law", choices=("native_coulomb", "external_coulomb_regularized", "combined_slip_magic_formula"), required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--dt", type=float, default=.0025)
    p.add_argument("--duration", type=float, default=2.)
    p.add_argument("--speed", type=float, default=1.)
    p.add_argument("--side-angle-deg", type=float, default=20.)
    p.add_argument("--mu", type=float, default=.7)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def run(a):
    if a.output_dir.exists():
        raise FileExistsError("Fresh output required")
    if (a.dt not in (.0025, .00125) or not 0 < a.duration <= 3
            or not np.isfinite([a.speed, a.side_angle_deg, a.mu]).all()
            or a.speed <= 0 or abs(a.side_angle_deg) > 60
            or not a.device.startswith("cuda:")):
        raise ValueError("Use an explicit CUDA device and frozen short-bench clocks/ranges")
    steps = round(a.duration / a.dt)
    if not np.isclose(steps * a.dt, a.duration, rtol=0, atol=1e-12):
        raise ValueError("Duration must contain whole physical steps")
    parameters = TireParameters(mu=a.mu)
    from isaaclab.app import AppLauncher
    ordinal = int(a.device.split(":")[1])
    launch = argparse.Namespace(headless=True, device=a.device, enable_cameras=False,
        kit_args=f"--/renderer/activeGpu={ordinal} --/renderer/multiGpu/enabled=false --/physics/cudaDevice={ordinal} --/physics/sceneMultiGPUMode=0")
    app = AppLauncher(launch).app
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import UsdPhysics, UsdGeom, UsdShade, PhysxSchema, Gf
        a.output_dir.mkdir()
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=a.dt, device=a.device))
        stage = sim.stage
        # Procedural static floor avoids network/Nucleus asset dependencies.
        floor = "/World/ground"
        ground = UsdGeom.Cube.Define(stage, floor)
        ground.AddTranslateOp().Set(Gf.Vec3f(0.,0.,-.05))
        ground.AddScaleOp().Set(Gf.Vec3f(100.,100.,.05))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        material = UsdShade.Material.Define(stage,"/World/GroundMaterial")
        physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physical.CreateStaticFrictionAttr(1.);physical.CreateDynamicFrictionAttr(1.)
        physical.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("multiply")
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material,materialPurpose="physics")
        assert stage.GetPrimAtPath(floor).HasAPI(UsdPhysics.CollisionAPI)
        native_mu = a.mu if a.law == "native_coulomb" else 0.
        shape = sim_utils.CylinderCfg(radius=.127, height=.05, axis="Y",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0., angular_damping=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=native_mu,
                dynamic_friction=native_mu, restitution=0., friction_combine_mode="multiply"))
        wheel = RigidObject(RigidObjectCfg(prim_path="/World/Wheel", spawn=shape,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., .129))))
        sensor = ContactSensor(ContactSensorCfg(prim_path="/World/Wheel", update_period=0.,
            history_length=0, filter_prim_paths_expr=[floor], track_contact_points=True,
            max_contact_data_count_per_prim=64))
        sim.reset()
        wheel.reset(); sensor.reset()
        for _ in range(round(.5/a.dt)):
            wheel.write_data_to_sim(); sim.step(); wheel.update(a.dt); sensor.update(a.dt)
        alpha = np.deg2rad(a.side_angle_deg)
        initial = [a.speed*np.cos(alpha), a.speed*np.sin(alpha), 0.,
                   0., a.speed*np.cos(alpha)/.127, 0.]
        wheel.write_root_velocity_to_sim(torch.tensor([initial], device=a.device, dtype=torch.float32))
        mass = float(wheel.root_physx_view.get_masses().cpu().numpy().reshape(-1)[0])
        assert np.isclose(mass, 1.08, atol=1e-5)
        actual_normal = sensor.data.force_matrix_w
        assert tuple(actual_normal.shape) == (1, 1, 1, 3)
        assert tuple(sensor.data.contact_pos_w.shape) == (1, 1, 1, 3)
        sources = [Path(__file__), Path(__file__).with_name("nonlinear_tire_contact.py")]
        status = {"kind":"PhysX_single_wheel_contact_law_bench", "status":"running",
            "law":a.law,"parameters":asdict(parameters),"physics_dt_s":a.dt,
            "native_tangential_friction_coefficient":native_mu,"intended_tire_mu":a.mu,
            "external_normal_load_delay_physical_steps":0 if a.law=="native_coulomb" else 1,
            "mass_kg_readback":mass,"radius_m":.127,"width_m":.05,
            "normal_force_source":"ground-filtered PhysX contact impulse divided by physical dt",
            "native_contact_tangent_force_source":"m*delta_COM_velocity_xy/dt; single unactuated body, zero damping, gravity only",
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "native_frictions_and_material_combine_authored":True,
            "cooked_collision_geometry_runtime_readback_verified":False,
            "static_adhesion_modeled_by_external_law":False,
            "car_or_TRON1_task_result":False,"policy_training":False,"task_success_claimed":False}
        (a.output_dir/"status.json").write_text(json.dumps(status,indent=2)+chr(10))
        states, normal_forces, points, applied_forces, applied_torques, source_loads = [],[],[],[],[],[]
        def read():
            state = wheel.data.root_state_w[0].detach().cpu().numpy().copy()
            force = sensor.data.force_matrix_w[0,0,0].detach().cpu().numpy().copy()
            point = sensor.data.contact_pos_w[0,0,0].detach().cpu().numpy().copy()
            assert state.shape==(13,) and np.isfinite(state).all() and np.isfinite(force).all()
            assert force[2]>=-1e-4
            return state,force,point
        state, force, point = read()
        states.append(state);normal_forces.append(force);points.append(point)
        for k in range(steps):
            F, T = np.zeros(3), np.zeros(3)
            Fn = max(float(force[2]), 0.)
            if a.law != "native_coulomb" and Fn > 0:
                assert np.isfinite(point).all(),"Loaded contact lacks measured contact position"
                w,x,y,z = state[3:7]
                axle = np.array([2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)])
                value = tangent_wrench(state[7:10], state[10:13], point-state[:3], axle,
                    [0,0,1], Fn, law=a.law, parameters=parameters)
                F,T = value["force_w_n"],value["torque_about_COM_w_nm"]
            wheel.set_external_force_and_torque(torch.as_tensor(F[None,None],device=a.device,dtype=torch.float32),
                torch.as_tensor(T[None,None],device=a.device,dtype=torch.float32),is_global=True)
            applied_forces.append(F);applied_torques.append(T);source_loads.append(Fn)
            wheel.write_data_to_sim();sim.step();wheel.update(a.dt);sensor.update(a.dt)
            state,force,point=read();states.append(state);normal_forces.append(force);points.append(point)
            if k%200==0:print(f"{a.law} physical_step={k}/{steps}",flush=True)
        array = np.asarray(states)
        np.savez_compressed(a.output_dir/"native_trace.npz",time_s=np.arange(steps+1)*a.dt,
            root_state_w=array,ground_normal_force_w_n=np.asarray(normal_forces),
            contact_position_w_m=np.asarray(points),applied_tangent_force_w_n=np.asarray(applied_forces),
            applied_tangent_moment_w_nm=np.asarray(applied_torques),preceding_sample_load_n=np.asarray(source_loads),
            inferred_total_tangent_force_xy_n=mass*np.diff(array[:,7:9],axis=0)/a.dt)
        status.update(status="completed",physical_steps=steps,native_states=steps+1,
            native_trace_sha256=hashlib.sha256((a.output_dir/"native_trace.npz").read_bytes()).hexdigest(),
            root_pose_prescribed_during_episode=False,automatic_retry=False)
        (a.output_dir/"status.json").write_text(json.dumps(status,indent=2)+chr(10))
        print(json.dumps(status),flush=True)
    finally:
        app.close()


if __name__=="__main__":
    run(parser().parse_args())

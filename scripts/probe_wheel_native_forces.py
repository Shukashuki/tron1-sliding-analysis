"""Short policy-free PhysX wheel probe with native normal/friction readback.

Cold initialization has no prior ground contact. Legacy initialization settles,
resets pose, advances one physical step, then writes the initial velocity.
Each condition runs in a fresh process. No policy, external force or installation.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback
import numpy as np


def selected(values, count_index, start_index):
    """Copy only active native entries; never include stale buffer slots."""
    count = np.asarray(values[count_index]); start = np.asarray(values[start_index])
    if count.size != 1 or start.size != 1:
        raise ValueError('Single body/filter pair required')
    n, begin = int(count.ravel()[0]), int(start.ravel()[0])
    capacity = len(values[0])
    if n < 0 or n >= capacity or (n and (begin < 0 or begin+n > capacity)):
        raise ValueError('Invalid or saturated native buffer')
    return [np.asarray(v)[begin:begin+n].copy() if n else np.asarray(v)[:0].copy()
            for v in values[:count_index]]


def reduce_native(normal, friction):
    f, p, n, gap = selected(normal, 4, 5)
    ft, fp = selected(friction, 2, 3)
    if not all(np.isfinite(v).all() for v in (f, p, n, gap, ft, fp)):
        raise ValueError('Nonfinite active contact entry')
    return dict(normal_force_w_n=(f.reshape(-1, 1)*n).sum(axis=0),
                friction_force_w_n=ft.sum(axis=0), normal_count=len(f), friction_count=len(ft),
                normal_points=p, normal_scalars=f.reshape(-1), normal_normals=n, normal_gaps=gap.reshape(-1),
                friction_points=fp, friction_vectors=ft)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def run(a):
    if a.output_dir.exists(): raise FileExistsError('Fresh output required')
    if a.dt not in (.005, .0025, .00125) or a.duration not in (.04, 2.):
        raise ValueError('Frozen diagnostic clocks/horizons required')
    ordinal = int(a.device.removeprefix('cuda:'))
    assert a.device == f'cuda:{ordinal}' and 0 <= ordinal < 8
    from isaaclab.app import AppLauncher
    launch = argparse.Namespace(headless=True, device=a.device, enable_cameras=False,
        kit_args=f'--/renderer/activeGpu={ordinal} --/renderer/multiGpu/enabled=false --/physics/cudaDevice={ordinal} --/physics/sceneMultiGPUMode=0')
    app = AppLauncher(launch).app
    a.output_dir.mkdir(parents=True)
    try:
        import torch
        torch.cuda.set_device(ordinal)
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import UsdPhysics, UsdGeom, UsdShade, PhysxSchema, Gf
        def plain(v): return v.detach().cpu().numpy().copy()
        sim_cfg = sim_utils.SimulationCfg(dt=a.dt, device=a.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        stage = sim.stage
        floor = '/World/ground'
        ground = UsdGeom.Mesh.Define(stage, floor)
        ground.CreatePointsAttr([(x,y+a.floor_y_offset_m,z) for x,y,z in
            [(-10.,-10.,0.),(10.,-10.,0.),(10.,10.,0.),(-10.,10.,0.)]])
        ground.CreateFaceVertexCountsAttr([3,3]); ground.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
        ground.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr('none')
        material = UsdShade.Material.Define(stage, '/World/GroundMaterial')
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(1.); physics_material.CreateDynamicFrictionAttr(1.)
        physics_material.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('multiply')
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material, materialPurpose='physics')
        shape = sim_utils.CylinderCfg(radius=.127, height=.05, axis='Z', activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0., angular_damping=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.08), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.7, dynamic_friction=.7,
                restitution=0., friction_combine_mode='multiply'))
        wheel = RigidObject(RigidObjectCfg(prim_path='/World/Wheel', spawn=shape,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,1. if a.initialization=='cold' else .129))))
        geom = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Wheel/geometry/mesh'))
        orient = next(op for op in geom.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeOrient)
        if orient.GetPrecision()==UsdGeom.XformOp.PrecisionDouble:
            orient.Set(Gf.Quatd(np.sqrt(.5), Gf.Vec3d(np.sqrt(.5),0.,0.)))
        else: orient.Set(Gf.Quatf(np.sqrt(.5), Gf.Vec3f(np.sqrt(.5),0.,0.)))
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath('/World/Wheel'))
        api.CreateLockedRotAxisAttr(5)
        assert api.GetLockedPosAxisAttr().Get()==0
        sensor = ContactSensor(ContactSensorCfg(prim_path='/World/Wheel', update_period=0., history_length=0,
            filter_prim_paths_expr=[floor], track_contact_points=True, max_contact_data_count_per_prim=128))
        sim.reset(); wheel.reset(); sensor.reset()
        view = sensor.contact_physx_view
        def native():
            torch.cuda.synchronize(ordinal)
            normal = tuple(plain(v) for v in view.get_contact_data(a.dt))
            torch.cuda.synchronize(ordinal)
            friction = tuple(plain(v) for v in view.get_friction_data(a.dt))
            torch.cuda.synchronize(ordinal)
            return normal, friction
        def step():
            wheel.write_data_to_sim(); sim.step(render=False); wheel.update(a.dt); sensor.update(a.dt)
        # Cold starts away from ground while simulator initializes its views.
        # The legacy branch intentionally retains the historical settling/reset.
        if a.initialization=='legacy':
            for _ in range(round(1/a.dt)): step()
        before = reduce_native(*native())
        if a.initialization=='cold':
            assert before['normal_count']==before['friction_count']==0
        alpha = np.deg2rad(20.)
        initial_velocity = np.array([np.cos(alpha), np.sin(alpha), 0., 0., np.cos(alpha)/.127, 0.])
        initial_pose = np.array([0.,0.,.12701,1.,0.,0.,0.])
        replay = None
        if a.initial_state_from is not None:
            assert a.initialization=='cold'
            previous=json.loads(a.initial_state_from.read_text())
            assert previous['status']=='completed_requires_analysis' and previous['initialization']=='legacy'
            assert previous['physics_dt_s']==a.dt
            initial=np.asarray(previous['initial_state'],dtype=float)
            assert initial.shape==(13,) and np.isfinite(initial).all()
            initial_pose,initial_velocity=initial[:7],initial[7:]
            replay=dict(path=str(a.initial_state_from),sha256=hashlib.sha256(a.initial_state_from.read_bytes()).hexdigest())
        pose = torch.as_tensor(initial_pose[None], dtype=torch.float32, device=a.device)
        wheel.write_root_pose_to_sim(pose)
        if a.initialization=='legacy':
            wheel.write_root_velocity_to_sim(torch.zeros((1,6), device=a.device)); step()
        wheel.write_root_velocity_to_sim(torch.as_tensor(initial_velocity[None], dtype=torch.float32, device=a.device))
        q0 = plain(wheel.data.root_state_w[0])
        if a.initialization=='cold': np.testing.assert_allclose(q0[:7], pose.cpu().numpy()[0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(q0[7:], initial_velocity, atol=1e-6, rtol=0)
        native_view = wheel.root_physx_view
        mass = float(plain(native_view.get_masses()).ravel()[0])
        materials = plain(native_view.get_material_properties())
        np.testing.assert_allclose(materials[..., :2], .7, atol=1e-6, rtol=0)
        solver = {}
        for key in ('solverPositionIterationCount','solverVelocityIterationCount','sleepThreshold','stabilizationThreshold'):
            attribute = stage.GetPrimAtPath('/World/Wheel').GetAttribute('physxRigidBody:'+key)
            solver[key] = dict(value=attribute.Get(), authored=attribute.HasAuthoredValueOpinion())
        source_sdk = Path(inspect.getfile(type(view)))
        meta = dict(status='running', initialization=a.initialization, initial_state_replay=replay, physics_dt_s=a.dt,
            floor_y_offset_m=a.floor_y_offset_m, floor_triangles=[0,1,2,0,2,3],
            duration_s=a.duration, initial_state=q0.tolist(), pre_initialization_contact_count=before['normal_count'],
            mass_kg=mass, inertia=plain(native_view.get_inertias()).tolist(),
            com_frame=plain(native_view.get_coms()).tolist(), materials=materials.tolist(),
            contact_offsets_m=plain(native_view.get_contact_offsets()).tolist(),
            rest_offsets_m=plain(native_view.get_rest_offsets()).tolist(),
            rigid_solver_attributes=solver, scene_physx_cfg=sim_cfg.physx.to_dict(),
            device=a.device, torch_device=torch.cuda.current_device(), state_tensor_device=str(wheel.data.root_state_w.device),
            tensor_api_sha256=hashlib.sha256(source_sdk.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            force_timing='Read completed interval k immediately after step; root_state[k] and root_state[k+1] bracket it.',
            normal_friction_point_pairing_assumed=False, external_forces=False, training=False,
            cooked_geometry_verified=False, solver_iteration_values_are_USD_not_native_readback=True)
        write(a.output_dir/'status.json', meta)
        states=[q0]; raw=[]; matrices=[]; normal_sums=[]; friction_sums=[]; counters=[]
        for k in range(round(a.duration/a.dt)):
            step()
            states.append(plain(wheel.data.root_state_w[0]))
            normal, friction = native()
            reduced = reduce_native(normal, friction)
            raw.append({**{f'normal_{j}':v for j,v in enumerate(normal)},
                        **{f'friction_{j}':v for j,v in enumerate(friction)}})
            direct = plain(view.get_contact_force_matrix(a.dt)).reshape(-1,3)
            assert direct.shape == (1,3)
            matrices.append(direct[0]); normal_sums.append(reduced['normal_force_w_n']); friction_sums.append(reduced['friction_force_w_n'])
            counters.append(sim.current_time_step_index)
        q=np.asarray(states); fn=np.asarray(normal_sums); ft=np.asarray(friction_sums)
        arrays=dict(time_s=np.arange(len(q))*a.dt, root_state_w=q, normal_force_w_n=fn,
            friction_force_w_n=ft, matrix_force_w_n=np.asarray(matrices), native_step_counters=np.asarray(counters),
            **{key:np.stack([r[key] for r in raw]) for key in raw[0]})
        target=a.output_dir/'native_trace.npz'; np.savez_compressed(target, **arrays)
        # Keep all data even if the diagnostic discovers failed consistency.
        observed = mass*np.diff(q[:,7:10].astype(float),axis=0)/a.dt
        observed[:,2] += mass*9.81
        residual = observed-fn-ft
        meta.update(status='completed_requires_analysis', samples=len(q),
            native_trace_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            maximum_normal_sum_matrix_difference_n=float(np.abs(fn-arrays['matrix_force_w_n']).max()),
            momentum_residual_rms_by_axis_n=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
            momentum_residual_peak_by_axis_n=np.abs(residual).max(axis=0).tolist(),
            no_pose_or_velocity_writes_during_recorded_episode=True)
        write(a.output_dir/'status.json', meta)
        print(json.dumps({k:meta[k] for k in ('status','samples','momentum_residual_rms_by_axis_n','momentum_residual_peak_by_axis_n')}), flush=True)
    except BaseException as error:
        write(a.output_dir/'failure.json',dict(error=str(error), traceback=traceback.format_exc()))
        raise
    finally: app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--initialization',choices=('cold','legacy'),required=True)
    p.add_argument('--dt',type=float,default=.0025)
    p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--device',required=True)
    p.add_argument('--initial-state-from',type=Path)
    p.add_argument('--floor-y-offset-m',type=float,choices=(0.,-5.),default=0.)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())

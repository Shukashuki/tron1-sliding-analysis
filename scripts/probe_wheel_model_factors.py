"""Cold, seam-controlled PhysX geometry/law/numerical factorial.

Native normal contact is retained. External laws use each native normal point's
previous-step load, sum their wrenches, and disable native tangential friction.
The contact model is a sensitivity model, not an identified real TRON1 tire.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback
import numpy as np
from nonlinear_tire_contact import TireParameters, tangent_wrench

MASS = 1.0800000429153442
INERTIA = (.004579830449074507, .008709659799933434, .004579830449074507)


def external_wrench(state, reduced, law, mu):
    """One-step delayed, pointwise normal loads; no patch pairing assumption."""
    force, torque = np.zeros(3), np.zeros(3)
    power = 0.
    if law == 'native_coulomb':
        return force, torque, power
    w, x, y, z = state[3:7]
    axle = np.array([2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)])
    for point, normal, load in zip(reduced['normal_points'], reduced['normal_normals'], reduced['normal_scalars']):
        if load <= 0:
            continue
        value = tangent_wrench(state[7:10], state[10:13], point-state[:3], axle,
                               normal, load, law=law, parameters=TireParameters(mu=mu))
        force += value['force_w_n']
        torque += value['torque_about_COM_w_nm']
        power += float(value['contact_power_w'])
    return force, torque, power


def author_geometry(stage, slices, contact_offset):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    body = stage.GetPrimAtPath('/World/Wheel')
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*INERTIA))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    original = stage.GetPrimAtPath('/World/Wheel/geometry/mesh')
    material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial(materialPurpose='physics')
    assert material, 'Native material binding must be explicit'
    if slices == 1:
        colliders = [original]
    else:
        old_scopes = [p for p in Usd.PrimRange(body) if p.HasAPI(UsdPhysics.CollisionAPI)]
        colliders = []
        for i in range(slices):
            geom = UsdGeom.Cylinder.Define(stage, f'/World/Wheel/slice_{i:02d}')
            geom.CreateRadiusAttr(.127); geom.CreateHeightAttr(.05/slices); geom.CreateAxisAttr('Y')
            geom.AddTranslateOp().Set(Gf.Vec3d(0, -.025+(i+.5)*.05/slices, 0))
            geom.CreatePurposeAttr('guide')
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material, materialPurpose='physics')
            colliders.append(geom.GetPrim())
        for prim in old_scopes:
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
    for prim in colliders:
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(contact_offset); api.CreateRestOffsetAttr(0.)
    return [str(p.GetPath()) for p in colliders]


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
        native_mu = a.mu if a.law == 'native_coulomb' else 0.
        shape = sim_utils.CylinderCfg(radius=.127, height=.05, axis='Z', activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0., angular_damping=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.08), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=native_mu, dynamic_friction=native_mu,
                restitution=0., friction_combine_mode='multiply'))
        wheel = RigidObject(RigidObjectCfg(prim_path='/World/Wheel', spawn=shape,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,1. if a.initialization=='cold' else .129))))
        geom = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Wheel/geometry/mesh'))
        orient = next(op for op in geom.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeOrient)
        if orient.GetPrecision()==UsdGeom.XformOp.PrecisionDouble:
            orient.Set(Gf.Quatd(np.sqrt(.5), Gf.Vec3d(np.sqrt(.5),0.,0.)))
        else: orient.Set(Gf.Quatf(np.sqrt(.5), Gf.Vec3f(np.sqrt(.5),0.,0.)))
        collider_paths = author_geometry(stage, a.slices, .001)
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath('/World/Wheel'))
        api.CreateLockedRotAxisAttr(5)
        assert api.GetLockedPosAxisAttr().Get()==0
        sensor = ContactSensor(ContactSensorCfg(prim_path='/World/Wheel', update_period=0., history_length=0,
            filter_prim_paths_expr=[floor], track_contact_points=True, max_contact_data_count_per_prim=512))
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
        np.testing.assert_allclose(materials[..., :2], native_mu, atol=1e-6, rtol=0)
        np.testing.assert_allclose(plain(native_view.get_inertias()).reshape(3,3), np.diag(INERTIA), atol=1e-8, rtol=0)
        np.testing.assert_allclose(plain(native_view.get_coms()).reshape(-1)[:3], 0., atol=1e-8, rtol=0)
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
            normal_friction_point_pairing_assumed=False, external_forces=a.law!='native_coulomb', training=False,
            geometry_slices=a.slices, collider_paths=collider_paths, law=a.law, mu=a.mu,
            native_mu=native_mu, external_normal_load_delay_physical_steps=0 if a.law=='native_coulomb' else 1,
            external_point_source='preceding native normal points with their individual scalar loads and normals',
            external_static_adhesion_modeled=False, real_tire_parameters_identified=False,
            tire_helper_sha256=hashlib.sha256(Path(__file__).with_name('nonlinear_tire_contact.py').read_bytes()).hexdigest(),
            cooked_geometry_verified=False, solver_iteration_values_are_USD_not_native_readback=True)
        write(a.output_dir/'status.json', meta)
        states=[q0]; raw=[]; matrices=[]; normal_sums=[]; friction_sums=[]; counters=[]
        applied=[]; torques=[]; source_loads=[]; external_powers=[]
        previous = reduce_native(*native())
        for k in range(round(a.duration/a.dt)):
            F,T,P = external_wrench(states[-1], previous, a.law, a.mu)
            if a.law != 'native_coulomb':
                wheel.set_external_force_and_torque(torch.as_tensor(F[None,None],device=a.device,dtype=torch.float32),
                    torch.as_tensor(T[None,None],device=a.device,dtype=torch.float32),
                    positions=torch.as_tensor(states[-1][None,None,:3],device=a.device,dtype=torch.float32),is_global=True)
            applied.append(F); torques.append(T); external_powers.append(P)
            source_loads.append(float(previous['normal_scalars'].sum()))
            step()
            states.append(plain(wheel.data.root_state_w[0]))
            normal, friction = native()
            reduced = reduce_native(normal, friction)
            if a.law != 'native_coulomb' and np.linalg.norm(reduced['friction_force_w_n']) > 1e-4:
                raise RuntimeError('Native tangential force remained after material substitution')
            previous = reduced
            raw.append({**{f'normal_{j}':v for j,v in enumerate(normal)},
                        **{f'friction_{j}':v for j,v in enumerate(friction)}})
            direct = plain(view.get_contact_force_matrix(a.dt)).reshape(-1,3)
            assert direct.shape == (1,3)
            matrices.append(direct[0]); normal_sums.append(reduced['normal_force_w_n']); friction_sums.append(reduced['friction_force_w_n'])
            counters.append(sim.current_time_step_index)
        q=np.asarray(states); fn=np.asarray(normal_sums); ft=np.asarray(friction_sums)
        arrays=dict(time_s=np.arange(len(q))*a.dt, root_state_w=q, normal_force_w_n=fn,
            applied_tangent_force_w_n=np.asarray(applied), applied_tangent_moment_w_nm=np.asarray(torques),
            preceding_scalar_normal_load_n=np.asarray(source_loads), external_contact_power_w=np.asarray(external_powers),
            friction_force_w_n=ft, matrix_force_w_n=np.asarray(matrices), native_step_counters=np.asarray(counters),
            **{key:np.stack([r[key] for r in raw]) for key in raw[0]})
        target=a.output_dir/'native_trace.npz'; np.savez_compressed(target, **arrays)
        # Keep all data even if the diagnostic discovers failed consistency.
        observed = mass*np.diff(q[:,7:10].astype(float),axis=0)/a.dt
        observed[:,2] += mass*9.81
        residual = observed-fn-ft-np.asarray(applied)
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
    p.add_argument('--initialization',choices=('cold',),default='cold')
    p.add_argument('--slices',type=int,choices=(1,9),required=True)
    p.add_argument('--law',choices=('native_coulomb','external_coulomb_regularized','combined_slip_magic_formula'),required=True)
    p.add_argument('--mu',type=float,choices=(.7,),default=.7)
    p.add_argument('--dt',type=float,default=.0025)
    p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--device',required=True)
    p.add_argument('--initial-state-from',type=Path,default=None)
    p.add_argument('--floor-y-offset-m',type=float,choices=(-5.,),default=-5.)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())

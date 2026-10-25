"""Batched read-only authored wheel gap, with native all-environment audits."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
import wheel_ground_geometry as reference

WHEELS = ('wheel_L_Link', 'wheel_R_Link')


def rotate(q, vector):
    xyz = q[..., 1:]
    vector = vector.expand_as(xyz)
    uv = torch.cross(xyz, vector, dim=-1)
    return vector+2*(q[..., :1]*uv+torch.cross(xyz, uv, dim=-1))


def authored_gaps(position, quaternion, local_centers, local_axes, radii, half_widths):
    if position.shape[-2:] != (2, 3) or quaternion.shape != (*position.shape[:-1], 4):
        raise ValueError('Require link position[...,2,3] and wxyz[...,2,4]')
    centers = position+rotate(quaternion, local_centers)
    axes = rotate(quaternion, local_axes)
    cosine = axes[..., 2]
    extent = radii*(1.-cosine.square()).clamp_min(0.).sqrt()+half_widths*cosine.abs()
    return centers[..., 2]-extent


class WheelGroundGap:
    def __init__(self, env, *, allow_cpu_fixture=False):
        self.env, self.robot = env, env.scene['robot']
        self.device = torch.device(env.device); self.ids = [self.robot.body_names.index(n) for n in WHEELS]
        self.fixture = self.device.type == 'cpu'
        if self.fixture and not allow_cpu_fixture: raise ValueError('Production wheel gap requires CUDA')
        self.specs = []
        if self.fixture:
            self.specs = [dict(radius_m=.127, half_width_m=.025, local_center_m=[0., sign*.00707, 0.], local_axis=[0., -1., 0.]) for sign in (1., -1.)]
            env_count, paths = env.num_envs, []
            native_offsets = None
        else:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            from tron1_wheel_geometry import collision_enabled
            stage = omni.usd.get_context().get_stage(); cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            paths = self.robot.root_physx_view.link_paths
            if len(paths) != env.num_envs: raise ValueError('Native link path environment count differs')
            for world, links in enumerate(paths):
                if len(links) != len(self.robot.body_names): raise ValueError('Native body link order differs')
                specs = []
                for name, index in zip(WHEELS, self.ids, strict=True):
                    body = stage.GetPrimAtPath(links[index]); assert body and str(body.GetName()) == name
                    shapes = [p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Gprim) and collision_enabled(p, body)]
                    assert len(shapes) == 1 and shapes[0].IsA(UsdGeom.Cylinder), 'Require original one-cylinder wheels'
                    cylinder = UsdGeom.Cylinder(shapes[0]); matrix, _ = cache.ComputeRelativeTransform(shapes[0], body)
                    matrix = np.array(matrix)
                    np.testing.assert_allclose(matrix[:3,:3]@matrix[:3,:3].T, np.eye(3), rtol=0, atol=1e-6)
                    spec = dict(radius_m=float(cylinder.GetRadiusAttr().Get()), half_width_m=float(cylinder.GetHeightAttr().Get())/2.,
                        local_center_m=matrix[3,:3].tolist(), local_axis=matrix[{'X':0,'Y':1,'Z':2}[str(cylinder.GetAxisAttr().Get())],:3].tolist())
                    assert abs(spec['radius_m']-.127) < 1e-6 and abs(spec['half_width_m']-.025) < 1e-6
                    specs.append(spec)
                if world == 0: self.specs = specs
                else:
                    for actual, expected in zip(specs, self.specs, strict=True):
                        for key in ('radius_m','half_width_m','local_center_m','local_axis'):
                            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-9,
                                err_msg='Authored wheel frames differ across worlds')
            slots = env.mjlab_recipe_shape_slots
            wheel_slots = [i for name in WHEELS for i in slots[name]]; assert len(wheel_slots) == 2
            view = self.robot.root_physx_view
            contacts, rests = view.get_contact_offsets().detach().cpu().numpy(), view.get_rest_offsets().detach().cpu().numpy()
            np.testing.assert_allclose(contacts[:, wheel_slots], .001, rtol=0, atol=1e-8)
            np.testing.assert_allclose(rests[:, wheel_slots], 0., rtol=0, atol=1e-8)
            native_offsets = dict(contact_m=[float(contacts[:, wheel_slots].min()), float(contacts[:, wheel_slots].max())], rest_m=[float(rests[:, wheel_slots].min()), float(rests[:, wheel_slots].max())])
            assert env.cfg.scene.terrain.terrain_type == 'plane' and tuple(env.cfg.sim.gravity[:2]) == (0.,0.)
            env_count = len(paths)
        tensor = lambda value: torch.tensor(value, device=self.device, dtype=torch.float32)
        self.local_centers = tensor([s['local_center_m'] for s in self.specs])
        self.local_axes = tensor([s['local_axis'] for s in self.specs])
        self.radii = tensor([s['radius_m'] for s in self.specs]); self.half_widths = tensor([s['half_width_m'] for s in self.specs])
        self.audit_result = dict(kind='readonly_authored_cylinder_gap', verified_environment_count=env_count,
            actual_wheels_per_environment=2, cpu_fixture_only=self.fixture, original_one_cylinder_per_wheel=True,
            specs=self.specs, native_wheel_offsets=native_offsets, no_geometry_or_physics_writes=True,
            distance_is_authored_geometry_not_native_manifold=True, all_environment_frames_checked=not self.fixture,
            source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__, reference.__file__)})
        # Independent CPU geometry comparison at actual startup poses only.
        p, q = self.pose(); actual = self.gaps().detach().cpu().numpy()
        p, q = p.detach().cpu().numpy(), q.detach().cpu().numpy()
        expected = np.zeros_like(actual)
        for side, spec in enumerate(self.specs):
            center = p[:,side]+reference.rotate(q[:,side], spec['local_center_m'])
            axis = reference.rotate(q[:,side], spec['local_axis'])
            expected[:,side] = reference.cylinder_support(center, axis, spec['radius_m'], spec['half_width_m'])['authored_signed_gap_m']
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
        self.audit_result['startup_CPU_Torch_max_gap_difference_m'] = float(np.abs(actual-expected).max())

    def pose(self):
        p = self.robot.data.body_link_pos_w[:, self.ids]
        q = self.robot.data.body_link_quat_w[:, self.ids]
        if p.shape != (self.env.num_envs,2,3) or q.shape != (self.env.num_envs,2,4):
            raise ValueError('Native body geometry tensors differ')
        return p, q

    def gaps(self):
        p, q = self.pose()
        return authored_gaps(p, q, self.local_centers, self.local_axes, self.radii, self.half_widths)

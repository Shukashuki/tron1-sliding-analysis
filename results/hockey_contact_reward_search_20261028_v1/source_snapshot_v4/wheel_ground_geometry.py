"""Authored finite-cylinder support geometry; never infer native force/contact.

Uses link/collider frames and actual COM velocities. The lowest authored point
is a geometry diagnostic, distinct from a simulator's persistent manifold.
"""
from __future__ import annotations
import numpy as np


def rotate(quaternion_wxyz, vector):
    q = np.asarray(quaternion_wxyz, dtype=float)
    v = np.asarray(vector, dtype=float)
    if q.shape[-1:] != (4,) or v.shape[-1:] != (3,) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('Require finite quaternion[...,4] and vector[...,3]')
    if not np.allclose(np.linalg.norm(q, axis=-1), 1., rtol=0, atol=1e-5):
        raise ValueError('Normalized wxyz quaternion required')
    xyz = q[..., 1:]
    return v+2*np.cross(xyz, np.cross(xyz, v)+q[..., :1]*v)


def cylinder_support(center_w, axle_w, radius_m, half_width_m, normal_w=(0., 0., 1.), plane_offset_m=0.):
    center, axle, normal = (np.asarray(v, dtype=float) for v in (center_w, axle_w, normal_w))
    if any(v.shape[-1:] != (3,) or not np.isfinite(v).all() for v in (center, axle, normal)):
        raise ValueError('Require finite vector[...,3]')
    if not np.isfinite([radius_m, half_width_m, plane_offset_m]).all() or radius_m <= 0 or half_width_m <= 0:
        raise ValueError('Positive cylinder radius/half width required')
    if not np.allclose(np.linalg.norm(axle, axis=-1), 1., rtol=0, atol=1e-5) or not np.allclose(np.linalg.norm(normal, axis=-1), 1., rtol=0, atol=1e-8):
        raise ValueError('Unit axis and unit ground normal required')
    axial = np.sum(axle*normal, axis=-1, keepdims=True)
    radial = normal-axial*axle
    magnitude = np.linalg.norm(radial, axis=-1, keepdims=True)
    direction = np.divide(radial, magnitude, out=np.zeros_like(radial), where=magnitude > 1e-10)
    offset = -half_width_m*np.sign(axial)*axle-radius_m*direction
    point = center+offset
    return dict(authored_lowest_point_w_m=point,
        authored_signed_gap_m=np.sum(point*normal, axis=-1)-plane_offset_m,
        axle_dot_ground_normal=np.squeeze(axial, axis=-1))


def point_velocity(point_w, com_position_w, com_velocity_w, angular_velocity_w):
    values = [np.asarray(v, dtype=float) for v in (point_w, com_position_w, com_velocity_w, angular_velocity_w)]
    if any(v.shape[-1:] != (3,) or not np.isfinite(v).all() for v in values):
        raise ValueError('Require finite point/COM/velocity/omega vectors')
    point, com, velocity, omega = values
    return velocity+np.cross(omega, point-com)

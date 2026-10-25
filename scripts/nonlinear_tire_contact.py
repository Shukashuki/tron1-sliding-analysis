"""Contact-law diagnostics for a new PhysX cohort; not an existing PPO reward.

Keep native normal contact and apply ONLY a tangential wrench to each wheel.
Callers must disable native wheel/ground tangential friction before applying
this law, and record the physical-step delay of the measured normal load.
The combined-slip Magic Formula shape follows the open xcar-rlgpu submodule;
it is a sensitivity model, not an identified TRON1 or CARLA tire.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TireParameters:
    mu: float = .7
    B: float = 4.5
    C: float = 1.8
    E: float = 1.
    speed_regularization_m_s: float = .2
    coulomb_regularization_m_s: float = .02

    def __post_init__(self):
        values = np.asarray(list(vars(self).values()), dtype=float)
        if (not np.isfinite(values).all() or self.mu <= 0 or self.B <= 0
                or not 0 < self.C <= 2 or not 0 <= self.E <= 1
                or min(self.speed_regularization_m_s, self.coulomb_regularization_m_s) <= 0):
            raise ValueError("Require a finite dissipative tire profile and positive regularization")


def _vectors(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim < 1 or value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError(name + " must contain finite world vectors[...,3]")
    return value


def wheel_frame(axle_w, normal_w):
    """Rolling +X is axle cross normal; lateral +Y completes a right-handed frame."""
    axle, normal = np.broadcast_arrays(_vectors(axle_w, "axle"), _vectors(normal_w, "normal"))
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(norm < 1e-9):
        raise ValueError("Zero contact normal")
    normal = normal / norm
    rolling = np.cross(axle, normal)
    length = np.linalg.norm(rolling, axis=-1, keepdims=True)
    if np.any(length < 1e-9):
        raise ValueError("Wheel axle is parallel to contact normal")
    rolling = rolling / length
    return rolling, np.cross(normal, rolling), normal


def tangent_wrench(hub_velocity_w, angular_velocity_w, contact_offset_w,
                   axle_w, normal_w, normal_load_n, *, law, parameters=None):
    """Return force and COM moment from physical contact velocity v + omega cross r.

    hub_velocity_w is the wheel COM velocity; contact_offset_w is measured from
    that COM to the ground contact point. angular_velocity_w includes wheel spin.
    Native solver supplies normal force separately. No tire force is generated
    for zero normal load. No spring, suspension, rolling resistance, camber thrust,
    tire transient, static adhesion, or mass/inertia modification is introduced.
    """
    p = parameters or TireParameters()
    if law not in ("external_coulomb_regularized", "combined_slip_magic_formula"):
        raise ValueError("Unknown tangential law")
    velocity, omega, offset, axle, normal = np.broadcast_arrays(
        _vectors(hub_velocity_w, "COM velocity"), _vectors(angular_velocity_w, "angular velocity"),
        _vectors(contact_offset_w, "contact offset"), _vectors(axle_w, "axle"), _vectors(normal_w, "normal"))
    load = np.broadcast_to(np.asarray(normal_load_n, dtype=float), velocity.shape[:-1])
    if not np.isfinite(load).all() or np.any(load < 0):
        raise ValueError("Normal load must be finite and nonnegative")
    forward, lateral, n = wheel_frame(axle, normal)
    contact_velocity = velocity + np.cross(omega, offset)
    longitudinal = np.sum(contact_velocity * forward, axis=-1)
    sideways = np.sum(contact_velocity * lateral, axis=-1)
    hub_longitudinal = np.sum(velocity * forward, axis=-1)
    circumferential = hub_longitudinal - longitudinal
    denominator = np.maximum.reduce((np.abs(hub_longitudinal), np.abs(circumferential),
                                    np.full_like(load, p.speed_regularization_m_s)))
    slip = np.stack((longitudinal, sideways), axis=-1) / denominator[..., None]
    magnitude = np.linalg.norm(slip, axis=-1)
    direction = np.divide(slip, magnitude[..., None], out=np.zeros_like(slip), where=magnitude[..., None] > 0)
    if law == "combined_slip_magic_formula":
        Bs = p.B * magnitude
        utilization = np.sin(p.C * np.arctan(Bs - p.E * (Bs - np.arctan(Bs))))
    else:
        utilization = np.tanh(np.hypot(longitudinal, sideways) / p.coulomb_regularization_m_s)
    force_local = -direction * (p.mu * load * utilization)[..., None]
    force = force_local[..., :1] * forward + force_local[..., 1:] * lateral
    torque = np.cross(offset, force)
    hub_lateral = np.sum(velocity * lateral, axis=-1)
    moving = np.hypot(hub_longitudinal, hub_lateral) >= p.speed_regularization_m_s
    alpha = np.where(moving, np.arctan2(hub_lateral, np.abs(hub_longitudinal)), np.nan)
    return {"force_w_n": force, "torque_about_COM_w_nm": torque,
            "contact_velocity_w_m_s": contact_velocity,
            "longitudinal_contact_slip_m_s": longitudinal, "lateral_contact_slip_m_s": sideways,
            "normalized_contact_slip": slip, "hub_side_angle_rad": alpha,
            "friction_utilization": np.where(load > 0, utilization, np.nan),
            "contact_power_w": np.sum(force * contact_velocity, axis=-1),
            "normal_component_of_applied_tire_force_n": np.sum(force * n, axis=-1)}

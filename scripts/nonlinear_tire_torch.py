"""Batched device counterpart of nonlinear_tire_contact.tangent_wrench.

This is a force-law kernel, not a simulator or a calibrated tire. All inputs
are tensors on one device/dtype. mu broadcasts over wheel batches (e.g.
[N,1] for two wheels). Validation of configuration happens outside the hot
path; bad loaded measurements are returned as device flags, never hidden.
"""
from __future__ import annotations

import torch


def tangent_wrench_torch(velocity, omega, offset, axle, normal, load, mu, *,
                         nonlinear=True, B=4.5, C=1.8, E=1.0,
                         speed_regularization=.2, coulomb_regularization=.02):
    """No host transfers, random draws, world loops or simulator writes.

    No-load measurements may contain NaN contact positions; these yield zero
    force. Loaded invalid measurements are flagged and yield zero force so a
    subsequent audit can reject the run without poisoning the solver.
    """
    finite = (torch.isfinite(velocity).all(-1) & torch.isfinite(omega).all(-1)
              & torch.isfinite(offset).all(-1) & torch.isfinite(axle).all(-1)
              & torch.isfinite(normal).all(-1) & torch.isfinite(load)
              & torch.isfinite(mu) & (mu >= 0) & (load >= -1e-4))
    n_length = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    n = normal / n_length.clamp_min(1e-9)
    forward = torch.linalg.cross(axle, n, dim=-1)
    length = torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
    finite = finite & (n_length[..., 0] > 1e-9) & (length[..., 0] > 1e-9)
    loaded = load > 0
    valid = finite & loaded
    forward = torch.nan_to_num(forward / length.clamp_min(1e-9))
    lateral = torch.linalg.cross(torch.nan_to_num(n), forward, dim=-1)
    safe_v = torch.where(valid[..., None], velocity, 0.)
    safe_w = torch.where(valid[..., None], omega, 0.)
    safe_r = torch.where(valid[..., None], offset, 0.)
    contact_v = safe_v + torch.linalg.cross(safe_w, safe_r, dim=-1)
    longitudinal = (contact_v * forward).sum(-1)
    sideways = (contact_v * lateral).sum(-1)
    hub_longitudinal = (safe_v * forward).sum(-1)
    circumferential = hub_longitudinal - longitudinal
    denominator = torch.maximum(hub_longitudinal.abs(), circumferential.abs()).clamp_min(speed_regularization)
    slip = torch.stack((longitudinal, sideways), dim=-1) / denominator[..., None]
    magnitude = torch.linalg.vector_norm(slip, dim=-1)
    direction = slip / magnitude.clamp_min(1e-30)[..., None]
    if nonlinear:
        bs = B * magnitude
        utilization = torch.sin(C * torch.atan(bs - E * (bs - torch.atan(bs))))
    else:
        utilization = torch.tanh(torch.hypot(longitudinal, sideways) / coulomb_regularization)
    capacity = torch.where(valid, mu * load, 0.)
    local = -direction * (capacity * utilization)[..., None]
    force = local[..., :1] * forward + local[..., 1:] * lateral
    torque = torch.linalg.cross(safe_r, force, dim=-1)
    hub_lateral = (safe_v * lateral).sum(-1)
    moving = torch.hypot(hub_longitudinal, hub_lateral) >= speed_regularization
    angle = torch.where(valid & moving,
                        torch.atan2(hub_lateral, hub_longitudinal.abs()), float("nan"))
    # An invalid mu/load must also be rejected for a nominally unloaded wheel.
    fault = ((loaded & ~finite) | ~torch.isfinite(load) | (load < -1e-4)
             | ~torch.isfinite(mu) | (mu < 0))
    return dict(force_w_n=force, torque_about_COM_w_nm=torque,
                contact_velocity_w_m_s=contact_v,
                longitudinal_contact_slip_m_s=longitudinal,
                lateral_contact_slip_m_s=sideways, normalized_contact_slip=slip,
                hub_side_angle_rad=angle,
                friction_utilization=torch.where(valid, utilization, float("nan")),
                contact_power_w=(force * contact_v).sum(-1),
                normal_component_of_applied_tire_force_n=(force * torch.nan_to_num(n)).sum(-1),
                invalid_measurement=fault)

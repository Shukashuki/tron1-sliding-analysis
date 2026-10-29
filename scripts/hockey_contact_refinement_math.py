"""Offline candidate reward math; not installed in the frozen v2 training.

Preserves dense task feedback and adds no body-height or equal-load target.
These functions are proposed refinements, not evidence of learned success.
"""
from __future__ import annotations
import math


def bilateral_substep_stop(touching, post_gap, velocity_pre, velocity_post,
                          up_pre, up_post, phase_hold, *, contact_tolerance_m=.0001,
                          speed_scale_m_s=.1, tilt_scale_rad=.35, velocity_dimensions=2,
                          speed_form='exponential'):
    """Half mean + half worst of eight geometric-contact stop scores.

    Both ends of each physical interval contribute to speed/tilt scoring.
    Native pre-step touching and analytic post-step gap are used exactly as
    in the existing contact-shaping term. No force is used to define support.
    Phase is supplied per control window to preserve the existing phase clock.
    Three-dimensional stopping is explicit; the default retains the planar rule.
    The optional separate pseudo-Huber speed cost applies even without contact,
    so leaving the ground cannot disable that cost. It has no height target.
    """
    import torch
    if not (math.isfinite(contact_tolerance_m) and contact_tolerance_m>=0
            and math.isfinite(speed_scale_m_s) and speed_scale_m_s>0
            and math.isfinite(tilt_scale_rad) and tilt_scale_rad>0):
        raise ValueError('Invalid stop-reward scales')
    if touching.ndim!=3 or touching.shape[1:]!=(8,2):
        raise ValueError('Require eight physical substeps and two wheels')
    if velocity_dimensions not in (2,3):
        raise ValueError('Stop velocity must be explicitly planar or three-dimensional')
    if speed_form not in ('exponential','separate_pseudo_huber'):
        raise ValueError('Unknown stop-speed score')
    if speed_form=='separate_pseudo_huber' and velocity_dimensions!=3:
        raise ValueError('Dense stop-speed cost requires three-dimensional velocity')
    n=len(touching)
    if (post_gap.shape!=touching.shape or velocity_pre.shape!=(n,8,velocity_dimensions)
        or velocity_post.shape!=(n,8,velocity_dimensions) or up_pre.shape!=(n,8)
        or up_post.shape!=(n,8) or phase_hold.shape!=(n,)):
        raise ValueError('Inconsistent physical-window shapes')
    both=touching.bool().all(-1)&(post_gap<=contact_tolerance_m).all(-1)
    speed_sq=torch.maximum(velocity_pre.square().sum(-1),velocity_post.square().sum(-1))
    tilt_sq=torch.maximum(torch.acos(up_pre.clamp(-1,1)).square(),torch.acos(up_post.clamp(-1,1)).square())
    if speed_form=='separate_pseudo_huber':
        supported_tilt=both*torch.exp(-tilt_sq/tilt_scale_rad**2)
        x=speed_sq/speed_scale_m_s**2
        speed_cost=2*x/(torch.sqrt(1+x)+1)
        return phase_hold*(.5*(supported_tilt.mean(-1)+supported_tilt.amin(-1))
                           -.5*(speed_cost.mean(-1)+speed_cost.amax(-1)))
    score=both*torch.exp(-speed_sq/speed_scale_m_s**2-tilt_sq/tilt_scale_rad**2)
    return phase_hold*(.5*score.mean(-1)+.5*score.amin(-1))


def angular_pseudo_huber(angular_acceleration_sq, scale_rad_s2=50.):
    """Non-saturating penalty with the old quadratic scale near zero.

    2(sqrt(1+x)-1), x=|alpha|^2/scale^2. The equivalent rational expression
    avoids cancellation for small x. Large accelerations retain linear growth.
    This does not add a differentiable dynamics model or change action limits.
    """
    import torch
    if not math.isfinite(scale_rad_s2) or scale_rad_s2<=0:
        raise ValueError('Positive angular-acceleration scale required')
    if angular_acceleration_sq.ndim!=2 or angular_acceleration_sq.shape[1]!=8:
        raise ValueError('Require eight physical substeps')
    x=angular_acceleration_sq.clamp_min(0)/scale_rad_s2**2
    return (2*x/(torch.sqrt(1+x)+1)).mean(-1)

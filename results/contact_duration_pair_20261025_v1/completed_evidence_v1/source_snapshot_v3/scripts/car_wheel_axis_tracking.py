"""Reference 3D wheel-plane tracking, invariant to spin and axle sign.

Reuse the immutable task-space target adapter, replacing only its orientation
quality. No height target,new contact law,action filter or acceptance gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PARENT_SHA = "e431bb7eaafa0175e521796b22a14ffa796f3c90953855d2f5d30e7983e172a4"


def _quality(reference_quat, actual_quat, std=.4):
    import torch
    if std != .4:
        raise ValueError("Reference wheel-axis angular scale is frozen at0.4rad")
    axes, valid = [], []
    for q in (reference_quat, actual_quat):
        length = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        w, x, y, z = (q/length.clamp_min(1e-12)).unbind(-1)
        axis = torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), dim=-1)
        axis_length = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        axes.append(axis/axis_length.clamp_min(1e-12))
        valid.append((length[...,0] > 1e-6) & (axis_length[...,0] > 1e-6))
    a, b = axes
    cross = torch.linalg.vector_norm(torch.linalg.cross(a, b, dim=-1), dim=-1)
    dot = (a*b).sum(-1).abs()
    error = torch.atan2(cross, dot)
    quality = torch.exp(-error.square().mean(-1)/(std*std))
    # Finite zero quaternions get zero quality. NaN/Inf physical states remain
    # nonfinite even when validity is false; health checks must reject them.
    return quality*(valid[0] & valid[1]).all(-1)


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    start = source.index("def heading_quality(")
    end = source.index("def wheel_plane_heading_tracking(", start)
    source = source[:start]+"def heading_quality(reference_quat, actual_quat, std=.4):\n    return _axis_quality(reference_quat, actual_quat, std)\n\n\n"+source[end:]
    for old, new in [
        ("base_wheel_targets_free_leg_links_v1", "base_wheel_targets_3D_axles_free_leg_links_v1"),
        ("world_axial_wheel_plane_heading", "world_axial_3D_wheel_axis"),
        ("wheel_heading_std_rad", "wheel_axis_std_rad"),
        ("wheel_heading_min_projection", "wheel_axis_min_norm"),
        ("wheel_plane_heading_tracking", "wheel_plane_axis_tracking"),
        ("heading_quality", "axis_quality"),
        ("live_heading_reward_", "live_axis_reward_"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, "<3D_wheel_axis_target_adapter>", "exec")
    return source


_source = materialize((Path(__file__).parent/"car_taskspace_tracking.py").read_text())
_namespace = dict(__name__=__name__, __file__=__file__, _axis_quality=_quality)
exec(compile(_source, "<3D_wheel_axis_target_adapter>", "exec"), _namespace)
for _name in ("SETTINGS", "CHANGED", "settings", "axis_quality", "wheel_plane_axis_tracking", "configure", "runtime_audit"):
    globals()[_name] = _namespace[_name]

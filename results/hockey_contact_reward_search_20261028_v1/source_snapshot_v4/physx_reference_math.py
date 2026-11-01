"""Small simulator-independent parts of policy-assisted PhysX trajectory search."""
from __future__ import annotations
import numpy as np


def smooth_basis(frames, knots):
    """Cubic smoothstep interpolation; zero residual and slope at both endpoints."""
    if frames < 2 or knots < 2:
        raise ValueError('Need at least two frames and two knots')
    x = np.linspace(0., knots - 1., frames)
    lo = np.minimum(x.astype(int), knots - 2)
    u = x - lo
    s = u*u*(3. - 2.*u)
    basis = np.zeros((frames, knots))
    basis[np.arange(frames), lo] = 1. - s
    basis[np.arange(frames), lo + 1] = s
    envelope = np.sin(np.linspace(0., np.pi, frames))**2
    return basis * envelope[:, None]


def wrap_angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


def trajectory_metrics(trace, reference, dt=.0025, *, core_interval_s=(1.44,4.), support_interval_s=None):
    """One continuous rollout, no interpolation across missing contact samples."""
    t = np.arange(len(trace['root_pos'])) * dt
    ref_time=np.arange(len(reference['body_pos_w']))*.02
    core = (t >= core_interval_s[0]-1e-9) & (t <= core_interval_s[1]+1e-9)
    support_interval_s=(.02,float(ref_time[-1])) if support_interval_s is None else support_interval_s
    window = (t >= support_interval_s[0]-1e-9) & (t <= support_interval_s[1]+1e-9)
    if not np.any(core) or not np.any(window):raise ValueError('Requested analysis window has no physical samples')
    interp=lambda x:np.column_stack([np.interp(t,ref_time,x[:,j]) for j in range(x.shape[-1])])
    pos, vel = trace['root_pos'].astype(float), trace['root_com_vel'].astype(float)
    q = trace['root_quat'].astype(float)
    yaw = np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2]**2+q[:,3]**2))
    rq = reference['body_quat_w'][:, 0].astype(float)
    ryaw = np.interp(t,ref_time,np.unwrap(np.arctan2(2*(rq[:,0]*rq[:,3]+rq[:,1]*rq[:,2]), 1-2*(rq[:,2]**2+rq[:,3]**2))))
    speed = np.linalg.norm(vel[:, :2], axis=-1)
    rv = interp(reference['body_lin_vel_w'][:, 0])
    rs = np.linalg.norm(rv[:, :2], axis=-1)
    beta = wrap_angle(np.arctan2(vel[:,1], vel[:,0])-yaw)
    rb = wrap_angle(np.arctan2(rv[:,1], rv[:,0])-ryaw)
    error = np.linalg.norm(pos[:, :2]-interp(reference['body_pos_w'][:,0,:2]), axis=-1)
    supported = np.max(trace['normal_fz'], axis=-1) > 5.
    gaps = np.flatnonzero(np.diff(np.r_[False, ~supported[window], False]))
    longest = int(np.max(gaps[1::2]-gaps[::2], initial=0))*dt
    tilt = np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2), -1, 1)))
    def tracking(mask):
        return dict(xy_rmse_m=float(np.sqrt(np.mean(error[mask]**2))),
                    xy_peak_m=float(np.max(error[mask])),
                    speed_retention=float(np.mean(speed[mask])/np.mean(rs[mask])),
                    beta_rmse_deg=float(np.degrees(np.sqrt(np.mean(wrap_angle(beta-rb)[mask]**2)))))
    return dict(core=tracking(core), whole=tracking(t<=ref_time[-1]+1e-9),
                support_fraction=float(np.mean(supported[window])), longest_support_gap_s=longest,
                maximum_root_tilt_deg=float(tilt.max()),
                wheel_aggregate_utilization_proxy_p95=float(np.quantile(trace['utilization_proxy'][window], .95)),
                pointwise_friction_cone_saturation_certified=False,
                task_success_claimed=False)


def reduce_contacts(normal_data, friction_data, com, velocity, angular, quaternion, mu, cap=128):
    """GPU segmented gather. Normal points and friction anchors stay separate.

    This function does NOT infer per-anchor normal load or pointwise cone
    utilization. Anchor power uses the force and material velocity at the SAME
    anchor. Slip RMS uses actual loaded normal points and the local+Y axle.
    Returns tensors without a CPU synchronization in the physics loop.
    """
    import torch
    fn, points, normals, _, counts, starts = normal_data
    ft, anchors, acounts, astarts = friction_data
    n = len(com)
    def gather(values, count, start):
        if count.numel() != n or start.numel() != n:
            raise ValueError('Expected exactly one floor collider filter per body')
        j = torch.arange(cap, device=com.device)[None]
        c, st = count.reshape(n, 1), start.reshape(n, 1)
        index = (st+j).long()
        mask = j < c
        bad = ((c > cap) | (c < 0)).reshape(n) | ((mask & ((index < 0) | (index >= len(values)))).any(-1))
        if len(values) == 0:
            shape = (n, cap, *values.shape[1:])
            return torch.zeros(shape, device=com.device, dtype=com.dtype), mask & False, bad
        return values[index.clamp(0, len(values)-1)], mask, bad
    force, mask, fault = gather(fn.reshape(-1), counts, starts)
    p, _, bad = gather(points, counts, starts); fault |= bad
    normal, _, bad = gather(normals, counts, starts); fault |= bad
    norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    normal = normal/norm.clamp_min(1e-12)
    force = torch.where(mask, force, 0.)
    loaded = mask & (force > 5.)
    w,x,y,z = quaternion.unbind(-1)
    axle = torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)
    lateral = axle[:,None]-(axle[:,None]*normal).sum(-1,keepdim=True)*normal
    length = torch.linalg.vector_norm(lateral,dim=-1,keepdim=True)
    fault |= (loaded & ((norm[...,0] < 1e-8) | (length[...,0] < 1e-8))).any(-1)
    lateral = lateral/length.clamp_min(1e-8)
    pv = velocity[:,None]+torch.cross(angular[:,None].expand_as(p),p-com[:,None],dim=-1)
    lat = (pv*lateral).sum(-1)
    weight = torch.where(loaded, force, 0.)
    slip = torch.sqrt((weight*lat.square()).sum(-1)/weight.sum(-1).clamp_min(1e-12))
    nf = (force[...,None]*normal).sum(1)
    af, amask, bad = gather(ft, acounts, astarts); fault |= bad
    ap, _, bad = gather(anchors, acounts, astarts); fault |= bad
    af = torch.where(amask[...,None], af, 0.)
    av = velocity[:,None]+torch.cross(angular[:,None].expand_as(ap),ap-com[:,None],dim=-1)
    power = (af*av).sum((-1,-2))
    aggregate = af.sum(1)
    proxy = torch.linalg.vector_norm(aggregate[:,:2],dim=-1)/(mu*nf[:,2].clamp_min(1e-8))
    proxy = torch.where(nf[:,2]>5., proxy, 0.)
    fault |= (~torch.isfinite(nf).all(-1) | ~torch.isfinite(slip) | ~torch.isfinite(power)
              | (force < -1e-4).any(-1))
    return dict(normal_fz=nf[:,2], lateral_slip_rms=slip, utilization_proxy=proxy,
                friction_anchor_power=power, friction_force_w=aggregate,
                normal_points_count=loaded.sum(-1), friction_anchors_count=acounts.reshape(n), fault=fault)

"""Task-specific cost and action decoding for native-PhysX hockey search.

Planning uses wheel-aggregate contact proxies. Independent qualification still
requires raw contact buffers and same-anchor force/velocity corroboration.
"""
import numpy as np


def stop_basis(frames, knots, dt=.02, ramp_s=.10):
    """Smooth initial ramp, with a free final knot to maintain a stopped stance."""
    if frames < 2 or not 2 <= knots <= frames or dt <= 0 or ramp_s <= 0:
        raise ValueError('Invalid residual clock')
    x = np.linspace(0., knots - 1., frames)
    lo = np.minimum(x.astype(int), knots - 2)
    u = x - lo
    s = u*u*(3-2*u)
    b = np.zeros((frames, knots))
    b[np.arange(frames), lo] = 1-s
    b[np.arange(frames), lo+1] = s
    u = np.clip(np.arange(frames)*dt/ramp_s, 0, 1)
    return b*(u*u*(3-2*u))[:, None]


def decode_action(raw_actor, residual):
    # Bound the prior before adding the optimizer's correction. Zero residual
    # exactly reproduces the previous controller's clipped output.
    return (raw_actor.clamp(-1, 1)+residual).clamp(-1, 1)


def dense_reference(reference, dt=.0025):
    from scipy.spatial.transform import Rotation, Slerp
    t = np.arange(len(reference['joint_pos']))/float(reference['fps'])
    query = np.minimum(np.arange(len(t)*8+1)*dt, t[-1])
    names = reference['body_names'].astype(str).tolist()
    r = {}
    for key in ('body_pos_w', 'body_lin_vel_w', 'joint_pos'):
        x = reference[key];flat = x.reshape(len(t), -1)
        r[key] = np.stack([np.interp(query, t, col) for col in flat.T], -1).reshape((len(query), *x.shape[1:]))
    def matrices(name):
        q = reference['body_quat_w'][:, names.index(name)]
        return Slerp(t, Rotation.from_quat(q[:, [1, 2, 3, 0]]))(query).as_matrix()
    body = matrices('base_Link')
    axle = np.stack([matrices('wheel_'+s+'_Link')[:, :, 1] for s in ('L', 'R')], 1)
    ids = [names.index('wheel_'+s+'_Link') for s in ('L', 'R')]
    return dict(time_s=query, xy=r['body_pos_w'][:, names.index('base_Link'), :2],
                velocity=r['body_lin_vel_w'][:, names.index('base_Link'), :2],
                leg_q=r['joint_pos'][:, :6], up=body[:, :, 2],
                yaw=np.arctan2(body[:, 1, 0], body[:, 0, 0]),
                wheel_heading=np.arctan2(-axle[:, :, 0], axle[:, :, 1]),
                wheel_camber=np.arcsin(np.clip(axle[:, :, 2], -1, 1)),
                wheel_velocity=r['body_lin_vel_w'][:, ids, :2])


def axes(q):
    import torch
    w, x, y, z = q.unbind(-1)
    up = torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), -1)
    axle = torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return up, axle, yaw


def wrap(x):
    import torch
    return torch.atan2(torch.sin(x), torch.cos(x))


def cost_terms(state, target, contact, time_s):
    """Unweighted dimensionless terms, shape [...], with no absolute Z term."""
    import torch
    up, _, yaw = axes(state['root_quat'])
    _, axle, _ = axes(state['wheel_quat'])
    heading = torch.atan2(-axle[..., 0], axle[..., 1])
    wheel_valid = torch.linalg.vector_norm(axle[..., :2], dim=-1) > 1e-6
    support = contact['normal_fz'] > 5
    loaded = support & wheel_valid
    load_count = loaded.sum(-1).clamp_min(1)
    heading_error = .5*wrap(2*(heading-target['wheel_heading']))
    camber = torch.asin(axle[..., 2].clamp(-1, 1))
    wv = state['wheel_velocity'][..., :2]
    ws = torch.linalg.vector_norm(wv, dim=-1)
    beta = .5*wrap(2*(torch.atan2(wv[..., 1], wv[..., 0])-heading))
    rv = target['wheel_velocity'];rs = torch.linalg.vector_norm(rv, dim=-1)
    ref_beta = .5*wrap(2*(torch.atan2(rv[..., 1], rv[..., 0])-target['wheel_heading']))
    beta_valid = loaded & (ws >= .25) & (rs >= .25)
    beta_error = .5*wrap(2*(beta-ref_beta))
    braking = .4 <= time_s <= 1.8
    slide = (support & wheel_valid & (ws >= .25) & (beta.abs() >= np.deg2rad(20))
             & (contact['lateral_slip_rms'] > .05) & (contact['utilization_proxy'] >= .9)
             & (contact['utilization_proxy'] <= 1.05) & (contact['friction_anchor_power'] < -.05)
             & (up[..., 2, None] > np.cos(np.pi/4)) & braking)
    terms = dict(
        xy=((state['root_pos'][..., :2]-target['xy'])/.15).square().sum(-1),
        velocity=((state['root_velocity'][..., :2]-target['velocity'])/.35).square().sum(-1),
        body_heading=(wrap(yaw-target['yaw'])/np.deg2rad(15)).square(),
        wheel_heading=((heading_error/np.deg2rad(20)).square()*loaded).sum(-1)/load_count,
        leg_pose=((state['leg_q']-target['leg_q'])/.25).square().mean(-1),
        body_up=(1-(up*target['up']).sum(-1)).clamp_min(0)/(1-np.cos(np.deg2rad(15))),
        camber=(((camber-target['wheel_camber'])/np.deg2rad(20)).square()*loaded).sum(-1)/load_count,
        beta=(beta_error.square()*beta_valid).sum(-1)/beta_valid.sum(-1).clamp_min(1)/np.deg2rad(20)**2,
        support_gap=(1-contact['normal_fz'].max(-1).values/5).clamp(0, 1) if time_s > .02+1e-9 else torch.zeros_like(yaw),
        tilt_excess=((np.cos(np.pi/4)-up[..., 2]).clamp_min(0)/.25).square(),
        terminal_speed=(state['root_velocity'][..., :2]/.1).square().sum(-1) if time_s >= 2.26-1e-9 else torch.zeros_like(yaw),
    )
    return terms, slide

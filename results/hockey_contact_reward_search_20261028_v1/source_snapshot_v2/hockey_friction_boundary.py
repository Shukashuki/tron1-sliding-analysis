"""Bounded, phase-gated friction-boundary sliding objective for flat PhysX ground.

Forces are wheel aggregates, not a per-anchor normal-load/cone certificate.
Force/velocity products pair completed-step forces with post-step velocities.
No simulator imports at module load: the reward is replayable on CPU.
"""
from __future__ import annotations

import math

from hockey_ppo_task import SupportWindow, SENSORS, WHEELS
from physx_reference_math import reduce_contacts


def boundary_sample(normal, friction, com, velocity, angular, quaternion, mu, cap=128):
    """Keep same-anchor lateral opposition separate from aggregate saturation."""
    import torch
    out = reduce_contacts(normal, friction, com, velocity, angular, quaternion, mu, cap)
    ft, points, counts, starts = friction
    n = len(com)
    j = torch.arange(cap, device=com.device)[None]
    ix = (starts.reshape(n, 1)+j).long()
    mask = j < counts.reshape(n, 1)
    bad = (mask & ((ix < 0) | (ix >= len(ft)))).any(-1)
    if len(ft):
        f = ft[ix.clamp(0, len(ft)-1)]
        p = points[ix.clamp(0, len(points)-1)]
    else:
        f = torch.zeros((n, cap, 3), device=com.device, dtype=com.dtype)
        p = com[:, None].expand_as(f)
        mask = mask & False
    v = velocity[:, None]+torch.cross(angular[:, None].expand_as(p), p-com[:, None], dim=-1)
    f = torch.where(mask[..., None], f, 0.)
    v = torch.where(mask[..., None], v, 0.)
    w, x, y, z = quaternion.unbind(-1)
    lateral = torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), torch.zeros_like(w)), -1)
    length = lateral.norm(dim=-1)
    lateral = lateral/length.clamp_min(1e-8)[:, None]
    vl = (v*lateral[:, None]).sum(-1)
    fl = (f*lateral[:, None]).sum(-1)
    power = (v*f).sum(-1)
    opposing = mask & (vl.abs() > .05) & (vl*fl < 0) & (power < 0)
    out['opposing_sliding_anchors'] = opposing.sum(-1)
    out['fault'] |= bad | ((length < 1e-8) & (out['normal_fz'] > 5))
    out['fault'] |= ~torch.isfinite(f).all((-1, -2)) | ~torch.isfinite(v).all((-1, -2))
    return out


def boundary_objective(samples, mg, cfg):
    """Inputs [world, substep, wheel], vectors have one additional final axis.

    At least one wheel may slide; there is no requirement for bilateral load.
    Every substep must retain load on at least one wheel to earn this bonus.
    No reward is given merely for rho=1, nor for larger unbounded slip/power.
    """
    import torch
    fn, rho = samples['normal_fz'], samples['utilization_proxy']
    loaded = fn > cfg['load_threshold_N']
    load = (fn/(mg[:, None, None]*cfg['load_full_fraction_mg'])).clamp(0, 1)
    near = torch.exp(-((rho-1)/cfg['rho_width']).square())*(rho <= cfg['rho_upper_screen'])
    slip = ((samples['lateral_slip_rms']-cfg['slip_threshold_m_s'])/cfg['slip_ramp_m_s']).clamp(0, 1)
    dissipation = ((-samples['friction_anchor_power']-cfg['power_threshold_W'])/cfg['power_ramp_W']).clamp(0, 1)
    braking_power = -(samples['friction_force_w'][..., :2]*samples['root_velocity_xy'][:, :, None]).sum(-1)
    brake = (braking_power/cfg['braking_power_ramp_W']).clamp(0, 1)
    local_evidence = samples['opposing_sliding_anchors'] > 0
    per_wheel = loaded*load*near*slip*dissipation*brake*local_evidence
    # Task and upright gates are measured at the same completed substep.
    task = torch.exp(-samples['xy_error_sq']/cfg['xy_scale_m']**2
                     -samples['velocity_error_sq']/cfg['velocity_scale_m_s']**2)
    upright = ((samples['up']-math.cos(math.radians(cfg['zero_at_tilt_deg'])))/
               (1-math.cos(math.radians(cfg['zero_at_tilt_deg'])))).clamp(0, 1)
    continuous = loaded.any(-1).all(-1)
    per_step = per_wheel.amax(-1)*task*upright*samples['braking_phase']
    reward = continuous*per_step.mean(-1)
    return dict(reward=reward, continuous_support=continuous,
                wheel_quality=per_wheel, task_gate=task, per_step=per_step)


class FrictionBoundaryWindow(SupportWindow):
    """Eight GPU contact samples per action; no CPU contact copies per substep."""
    def __init__(self, env):
        import torch
        super().__init__(env)
        self.cfg = env.cfg.hockey_protocol['friction_boundary']
        self.robot = env.scene['robot']
        self.wheel_ids = [self.robot.body_names.index(n) for n in WHEELS]
        # The existing DR draws once at startup, equally for both wheels. Read
        # actual per-world mu rather than assuming the validation coefficient.
        self.mu = env.mjlab_recipe_mu.to(device=env.device).clone()
        if not bool(torch.isfinite(self.mu).all() & (self.mu > 0).all()):
            raise ValueError('Invalid startup friction coefficients')
        shapes = {k:(env.num_envs, 8, 2) for k in ('normal_fz','lateral_slip_rms',
            'utilization_proxy','friction_anchor_power','opposing_sliding_anchors')}
        shapes.update(friction_force_w=(env.num_envs, 8, 2, 3), root_velocity_xy=(env.num_envs, 8, 2))
        shapes.update({k:(env.num_envs, 8) for k in ('xy_error_sq','velocity_error_sq','up','braking_phase')})
        self.values = {k:torch.zeros(s, device=env.device) for k,s in shapes.items()}
        self.fault = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.sample_count = 0
        self.reward_total = torch.zeros((), device=env.device)
        self.reward_count = 0
        self.reward_stamp = None

    def step(self, *args, **kwargs):
        self.fault.zero_()
        for value in self.values.values():
            value.zero_()
        self.reference_start = self.env.command_manager.get_term('motion').time_steps.clone()
        result = super().step(*args, **kwargs)
        # One fail-fast read per control step, never per physical substep.
        if bool(self.fault.any()):
            raise RuntimeError('Invalid native friction-boundary contact sample')
        # Log the same diagnostic in the zero-weight arm, where Isaac omits
        # the reward term. This does not alter the reward returned to PPO.
        reward_term(self.env)
        return result

    def update(self, *args, **kwargs):
        import torch
        before = self.samples
        result = super().update(*args, **kwargs)
        if self.samples == before:
            return result
        k = self.samples-1
        d = self.robot.data
        c = self.env.command_manager.get_term('motion')
        for side, (sensor, b) in enumerate(zip(self.sensors, self.wheel_ids)):
            view = sensor.contact_physx_view
            v = boundary_sample(view.get_contact_data(.0025), view.get_friction_data(.0025),
                d.body_com_pos_w[:, b], d.body_com_lin_vel_w[:, b], d.body_com_ang_vel_w[:, b],
                d.body_quat_w[:, b], self.mu)
            self.fault.logical_or_(v['fault'])
            for key in ('normal_fz','lateral_slip_rms','utilization_proxy','friction_anchor_power',
                        'opposing_sliding_anchors','friction_force_w'):
                self.values[key][:, k, side] = v[key]
        f0 = self.reference_start.clamp(max=c.motion.time_step_total-1)
        f1 = (f0+1).clamp(max=c.motion.time_step_total-1)
        alpha = (k+1)/8
        ref_p = torch.lerp(c.motion.body_pos_w[f0, 0, :2], c.motion.body_pos_w[f1, 0, :2], alpha)
        ref_v = torch.lerp(c.motion.body_lin_vel_w[f0, 0, :2], c.motion.body_lin_vel_w[f1, 0, :2], alpha)
        vxy = d.root_com_lin_vel_w[:, :2]
        self.values['root_velocity_xy'][:, k] = vxy
        self.values['xy_error_sq'][:, k] = (d.root_pos_w[:, :2]-self.env.scene.env_origins[:, :2]-ref_p).square().sum(-1)
        self.values['velocity_error_sq'][:, k] = (vxy-ref_v).square().sum(-1)
        self.values['up'][:, k] = 1-2*(d.root_quat_w[:, 1].square()+d.root_quat_w[:, 2].square())
        # Interval midpoint assigns boundary intervals without float32 0.4/1.8 ambiguity.
        phase = self.reference_start+(k+.5)/8
        protocol = self.env.cfg.hockey_protocol
        self.values['braking_phase'][:, k] = ((phase >= round(protocol['brake_start_s']/.02))
                                            & (phase < round(protocol['brake_end_s']/.02)))
        for value in self.values.values():
            self.fault |= ~torch.isfinite(value[:, k]).reshape(self.env.num_envs, -1).all(-1)
        self.sample_count += 1
        return result

    def audit(self):
        return dict(native_substeps_collected=self.sample_count, fault_seen=bool(self.fault.any()),
            unweighted_reward_mean=float(self.reward_total/max(1, self.reward_count)),
            reward_world_steps=self.reward_count, mu_min=float(self.mu.min()), mu_max=float(self.mu.max()),
            pointwise_cone_certified=False, physics_dt_s=.0025,
            cpu_contact_copy_per_physics_step=False, fault_check_per_control_step=True)


def reward_term(env):
    import torch
    window = getattr(env, 'hockey_support_window', None)
    if not isinstance(window, FrictionBoundaryWindow):
        return torch.zeros(env.num_envs, device=env.device)
    if window.active and window.samples != 8:
        raise RuntimeError('Boundary reward requires eight completed physical samples')
    values = boundary_objective(window.values, window.mg, window.cfg)['reward']
    stamp = (env.common_step_counter, env._sim_step_counter)
    if window.reward_stamp != stamp:
        window.reward_total += values.sum().detach()
        window.reward_count += env.num_envs
        window.reward_stamp = stamp
    return values


def configure(cfg, protocol):
    from isaaclab.managers import RewardTermCfg
    if protocol.get('method') != 'hockey' or protocol['friction_boundary']['weight'] < 0:
        raise ValueError('Boundary objective requires the declared hockey task and nonnegative weight')
    for name in SENSORS:
        getattr(cfg.scene, name).track_contact_points = True
    cfg.rewards.hockey_friction_boundary = RewardTermCfg(
        func=reward_term, weight=protocol['friction_boundary']['weight'])

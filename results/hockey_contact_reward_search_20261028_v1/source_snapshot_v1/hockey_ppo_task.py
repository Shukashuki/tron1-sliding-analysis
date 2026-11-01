"""Direct hockey PPO task. Import Isaac only after AppLauncher.

The two arms differ ONLY in the weight of a smooth any-wheel support deficit.
No car-command adapter, frozen stabilizer, MPPI residual, wheel-XY/spin target,
absolute height target, or positive reward for friction saturation is used.
"""
from __future__ import annotations

import math
from types import MethodType

GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
WHEELS = ("wheel_L_Link", "wheel_R_Link")
SENSORS = ("hockey_left_ground", "hockey_right_ground")


def yaw(q):
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def axle(q):
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)


def wrap(angle, *, axial=False):
    import torch
    factor = 2 if axial else 1
    return torch.atan2(torch.sin(factor*angle), torch.cos(factor*angle))/factor


def desired_command(env):
    """Append five fields; existing 51/120 observations retain their order."""
    import torch
    c = env.command_manager.get_term("motion")
    cfg = env.cfg.hockey_protocol
    velocity = c.anchor_lin_vel_w[:, :2]
    angle = yaw(c.robot_anchor_quat_w)
    x, y = velocity.unbind(-1)
    t = c.time_steps * env.step_dt
    if cfg.get("phase_clock") == "integer_reference_frames":
        brake_start = round(cfg["brake_start_s"]/env.step_dt)
        brake_end = round(cfg["brake_end_s"]/env.step_dt)
        braking = (c.time_steps >= brake_start) & (c.time_steps < brake_end)
        holding = c.time_steps >= brake_end
    else:
        braking = (t >= cfg["brake_start_s"]) & (t < cfg["brake_end_s"])
        holding = t >= cfg["brake_end_s"]
    return torch.stack((torch.cos(angle)*x+torch.sin(angle)*y,
                        -torch.sin(angle)*x+torch.cos(angle)*y,
                        c.anchor_ang_vel_w[:, 2],
                        braking.float(), holding.float()), -1)


def support_deficit(forces, mg, fraction):
    """N x 8 x 2 floor-normal forces. One wheel may carry the entire load.

    Bounded [0,1]: half worst-substep deficit plus half mean deficit. Forces
    above fraction*mg earn no additional benefit; no bilateral-load target.
    """
    gap = (1-forces.clamp_min(0).amax(-1)/(mg[:, None]*fraction)).clamp(0, 1)
    return .5*(gap.amax(-1)+gap.mean(-1))


def fallen(env):
    q = env.scene["robot"].data.root_quat_w
    return 1-2*(q[:, 1].square()+q[:, 2].square()) < math.cos(math.pi/4)


class SupportWindow:
    """Read all eight completed physical substeps before reward/reset.

    The original Isaac environment owns integration and reset. A counter check
    prevents reset-only scene refreshes from counting as physical samples.
    Storage is cleared every control step, so RSI resets cannot leak forces.
    """
    def __init__(self, env):
        import torch
        self.env = env
        self.force = torch.zeros((env.num_envs, 8, 2), device=env.device)
        self.mg = env.scene["robot"].root_physx_view.get_masses().sum(-1).to(env.device)*9.81
        self.sensors = [env.scene.sensors[n] for n in SENSORS]
        self.samples = 0
        self.active = False
        self.original_step, self.original_update = env.step, env.scene.update
        self.last_counter = env._sim_step_counter
        env.step = MethodType(lambda owner, *a, **k: self.step(*a, **k), env)
        env.scene.update = MethodType(lambda owner, *a, **k: self.update(*a, **k), env.scene)
        env.hockey_support_window = self

    def step(self, *args, **kwargs):
        if self.active:
            raise RuntimeError("Reentrant hockey step")
        self.force.zero_()
        self.samples = 0
        self.last_counter = self.env._sim_step_counter
        self.active = True
        try:
            result = self.original_step(*args, **kwargs)
            if self.samples != 8:
                raise RuntimeError("Hockey support requires eight actual samples")
            return result
        finally:
            self.active = False

    def update(self, *args, **kwargs):
        result = self.original_update(*args, **kwargs)
        counter = self.env._sim_step_counter
        if not self.active or counter == self.last_counter:
            return result
        if counter != self.last_counter+1 or self.samples >= 8:
            raise RuntimeError("Missed or duplicate hockey physical sample")
        for i, sensor in enumerate(self.sensors):
            value = sensor.data.force_matrix_w
            if value.shape != (self.env.num_envs, 1, 1, 3):
                raise RuntimeError("Expected one wheel/ground filtered normal-force vector")
            self.force[:, self.samples, i] = value[:, 0, 0, 2]
        self.last_counter = counter
        self.samples += 1
        return result


def rewards(env):
    import torch
    stamp = (env.common_step_counter, env._sim_step_counter)
    cached = getattr(env, "hockey_reward_cache", None)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    c = env.command_manager.get_term("motion")
    cfg, robot = env.cfg.hockey_protocol, env.scene["robot"]
    s = cfg["reward_scales"]
    # Isaac computes reward before advancing the command. Compare the completed
    # control interval to its endpoint reference, not the pre-action frame.
    index = (c.time_steps+1).clamp(max=c.motion.time_step_total-1)
    p = c.motion.body_pos_w[index, 0] + env.scene.env_origins
    q = c.motion.body_quat_w[index, 0]
    v = c.motion.body_lin_vel_w[index, 0]
    omega = c.motion.body_ang_vel_w[index, 0]
    actual_q = robot.data.root_quat_w
    actual_v = robot.data.root_com_lin_vel_w
    up = (1-2*(actual_q[:, 1].square()+actual_q[:, 2].square())).clamp(-1, 1)
    tilt = torch.acos(up)
    wheel_ids = [c.cfg.body_names.index(n) for n in WHEELS]
    aq = c.robot_body_quat_w[:, wheel_ids]
    rq = c.motion.body_quat_w[index][:, wheel_ids]
    aa, ra = axle(aq), axle(rq)
    angle = torch.acos((aa*ra).sum(-1).abs().clamp(0, 1))
    av = c.robot_body_lin_vel_w[:, wheel_ids, :2]
    rv = c.motion.body_lin_vel_w[index][:, wheel_ids, :2]
    ah = torch.atan2(-aa[..., 0], aa[..., 1])
    rh = torch.atan2(-ra[..., 0], ra[..., 1])
    beta_error = wrap(torch.atan2(av[..., 1], av[..., 0])-ah
                      -torch.atan2(rv[..., 1], rv[..., 0])+rh, axial=True)
    valid = ((av.norm(dim=-1) >= .25) & (rv.norm(dim=-1) >= .25)
             & (aa[..., :2].norm(dim=-1) > 1e-6) & (ra[..., :2].norm(dim=-1) > 1e-6))
    collector = getattr(env, "hockey_support_window", None)
    if collector is None:
        force = torch.zeros((env.num_envs, 8, 2), device=env.device)
        deficit = torch.ones(env.num_envs, device=env.device)
    else:
        if collector.active and collector.samples != 8:
            raise RuntimeError("Reward consumed before the physical window completed")
        force = collector.force
        deficit = support_deficit(force, collector.mg, cfg["support_load_fraction"])
    loaded_fraction = (force > 5).float().mean(1)
    supported = (force.amax(-1) > 5).float().mean(-1)
    exp = lambda error, scale: torch.exp(-error/scale**2)
    phase_hold = index*env.step_dt >= cfg["brake_end_s"]
    if cfg.get("phase_clock") == "integer_reference_frames":
        phase_hold = index >= round(cfg["brake_end_s"]/env.step_dt)
    values = {
        "xy": exp((robot.data.root_pos_w[:, :2]-p[:, :2]).square().sum(-1), s["xy_m"]),
        "velocity": exp((actual_v[:, :2]-v[:, :2]).square().sum(-1), s["velocity_m_s"]),
        "heading": exp(wrap(yaw(actual_q)-yaw(q)).square(), s["heading_rad"]),
        "yaw_rate": exp((robot.data.root_com_ang_vel_w[:, 2]-omega[:, 2]).square(), s["yaw_rate_rad_s"]),
        "leg_pose": exp((robot.data.joint_pos[:, c.leg_joint_indices]-c.motion.joint_pos[index][:, c.leg_joint_indices]).square().mean(-1), s["leg_pose_rad"]),
        "wheel_axis": exp(angle.square().mean(-1), s["wheel_axis_rad"]),
        "beta": (exp(beta_error.square(), s["beta_rad"])*valid*loaded_fraction).mean(-1),
        "stop_hold": phase_hold*exp(actual_v[:, :2].square().sum(-1), s["stop_speed_m_s"])*exp(tilt.square(), s["stop_tilt_rad"])*supported,
        "tilt": 1-up, "fall": (tilt > math.pi/4).float(), "support": -deficit,
    }
    env.hockey_reward_cache = (stamp, values)
    return values


def reward_term(env, key):
    return rewards(env)[key]


def configure(cfg, protocol, variant, *, training):
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
    from isaaclab.sensors import ContactSensorCfg
    from physx_mjlab_recipe import install_physx_mjlab_recipe
    from height_smoothness import vertical_acceleration_term
    cfg.hockey_protocol = protocol
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .0025, 8, 8
    cfg.scene.env_spacing = 8.
    cfg.scene.contact_forces.update_period = .0025
    cfg.commands.motion.start_fraction = protocol["rsi_full_clip_start_fraction"]
    if protocol.get("adaptive_sampling") == "fall_only":
        from hockey_ppo_sampling import motion_command_class
        cfg.commands.motion.class_type = motion_command_class()
    install_physx_mjlab_recipe(cfg, training=training, fixed_mu=protocol["evaluation_mu"],
        static_dynamic_ratio=protocol.get('static_dynamic_friction_ratio', 1.))
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.hockey_target = ObservationTermCfg(func=desired_command)
    for name in SENSORS:
        if getattr(cfg.scene, name, None) is not None:
            raise ValueError("Sensor already exists")
    for name, body in zip(SENSORS, WHEELS):
        setattr(cfg.scene, name, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/"+body, update_period=0., history_length=0,
            filter_prim_paths_expr=[GROUND], track_contact_points=not training,
            max_contact_data_count_per_prim=128, track_air_time=False, debug_vis=False))
    # Remove every inherited imitation term, including relative wheel positions.
    for name in tuple(vars(cfg.rewards)):
        if isinstance(getattr(cfg.rewards, name), RewardTermCfg) and name not in (
                "action_rate_l2", "joint_limit", "undesired_contacts"):
            setattr(cfg.rewards, name, None)
    weights = protocol["reward_weights"]
    for name in ("xy", "velocity", "heading", "yaw_rate", "leg_pose", "wheel_axis", "beta", "stop_hold", "tilt", "fall"):
        setattr(cfg.rewards, "hockey_"+name, RewardTermCfg(func=reward_term, weight=weights[name], params={"key": name}))
    # Include a zero-weight term in the baseline declaration; manager may omit it.
    cfg.rewards.hockey_support = RewardTermCfg(func=reward_term, weight=protocol["variants"][variant], params={"key": "support"})
    cfg.rewards.action_rate_l2.weight = weights["action_rate"]
    cfg.rewards.joint_limit.weight = weights["joint_limit"]
    cfg.rewards.undesired_contacts.weight = weights["undesired_contacts"]
    cfg.rewards.hockey_vertical_acceleration = RewardTermCfg(func=vertical_acceleration_term(),
        weight=weights["vertical_acceleration"], params={"acceleration_scale_m_s2": 5.})
    cfg.terminations.anchor_pos = cfg.terminations.ee_body_pos = None
    cfg.terminations.anchor_ori = None
    cfg.terminations.hockey_fall = TerminationTermCfg(func=fallen)
    # This is a finite-horizon stop task, not a timeout into a repeated motion.
    cfg.terminations.motion_end.time_out = False
    return cfg

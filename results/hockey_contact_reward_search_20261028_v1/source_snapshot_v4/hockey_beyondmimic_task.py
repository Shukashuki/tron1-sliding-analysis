"""Historical TRON1 BeyondMimic-style tracking, under shared physics settings.

Preserve its 51/120 observations, six exponential tracking rewards, adaptive
PPO schedule, orientation-error termination and motion-end timeout semantics.
Apply the already requested free-height / 400 Hz / action-rate -0.5 / shared DR
adaptations. No hockey phase fields, beta reward or stop/support reward.
"""
from __future__ import annotations


def configure(cfg, protocol, *, training):
    from isaaclab.sensors import ContactSensorCfg
    from height_smoothness import install_height_smoothness
    from physx_mjlab_recipe import install_physx_mjlab_recipe
    from hockey_ppo_task import GROUND, SENSORS, WHEELS

    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .0025, 8, 8
    cfg.scene.env_spacing = 8.
    cfg.scene.contact_forces.update_period = .0025
    cfg.commands.motion.start_fraction = protocol["rsi_full_clip_start_fraction"]
    install_height_smoothness(cfg, dict(mode="vertical_acceleration_only", weight=-.1,
                                      acceleration_scale_m_s2=5.))
    cfg.rewards.action_rate_l2.weight = -.5
    install_physx_mjlab_recipe(cfg, training=training, fixed_mu=protocol["evaluation_mu"])
    for sensor, body in zip(SENSORS, WHEELS):
        if getattr(cfg.scene, sensor, None) is not None:
            raise ValueError("Contact sensor already installed")
        setattr(cfg.scene, sensor, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/"+body, update_period=0., history_length=0,
            filter_prim_paths_expr=[GROUND], track_contact_points=not training,
            max_contact_data_count_per_prim=128, track_air_time=False, debug_vis=False))
    # Deliberately retain the historical anchor_ori criterion and motion-end
    # timeout. The independent task-success evaluator is identical for methods.
    return cfg

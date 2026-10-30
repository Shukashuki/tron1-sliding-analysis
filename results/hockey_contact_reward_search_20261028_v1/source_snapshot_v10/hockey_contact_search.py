"""Dense contact shaping for warm-started hockey PPO; success stays external.

Missing a contact does not erase XY/velocity feedback. Every contact metric
still uses all eight native substeps, with no load threshold or equal-load goal.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from hockey_bilateral_task import BilateralWindow, objective as historical_objective
from hockey_ppo_task import SENSORS, rewards as task_rewards


def objective(samples, mg, cfg):
    old = historical_objective(samples, mg, cfg)
    both = samples['touching'].bool().all(-1) & (samples['post_gap'] <= cfg['contact_tolerance_m']).all(-1)
    gap = ((samples['post_gap']-cfg['contact_tolerance_m']).clamp_min(0)/cfg['gap_scale_m']).clamp(0, 1)
    # Gap is a wheel-ground distance, not an absolute body-height target.
    gap = gap.amax(-1)
    return dict(contact_fraction=both.float().mean(-1),
                contact_loss=(~both).float().mean(-1),
                clearance=.5*(gap.mean(-1)+gap.amax(-1)),
                sliding=old['sliding'], angular_acceleration=old['angular_acceleration'])


class DenseContactWindow(BilateralWindow):
    def components(self):
        if self.active and self.samples != 8:
            raise RuntimeError('Contact reward consumed before all physical substeps')
        if self.cache is None:
            self.cache = objective(self.values, self.mg, self.cfg)
        return self.cache


def reward_term(env, key):
    import torch
    window = getattr(env, 'hockey_support_window', None)
    if not isinstance(window, DenseContactWindow):
        return torch.zeros(env.num_envs, device=env.device)
    return window.components()[key]


def configure(cfg, protocol):
    from isaaclab.managers import RewardTermCfg
    if protocol['bilateral_sliding']['gate_positive_tracking_on_bilateral_window']:
        raise ValueError('Dense search must preserve the original tracking feedback')
    for name in SENSORS:
        getattr(cfg.scene, name).track_contact_points = True
    for key, weight in protocol['bilateral_sliding']['weights'].items():
        setattr(cfg.rewards, 'contact_search_'+key,
                RewardTermCfg(func=reward_term, weight=weight, params={'key': key}))
    # Fall reward is multiplied by control dt by Isaac Lab's reward manager.
    expected = protocol['contact_search']['fall_terminal_cost']
    if not math.isclose(cfg.rewards.hockey_fall.weight*protocol['policy_dt_s'], -expected):
        raise ValueError('Fall cost does not match its declared per-event magnitude')


def initialize_policy(runner, checkpoint, expected_sha256, noise_std):
    """Copy actor, critic and normalizers; start a fresh optimizer and budget.

    The source checkpoint is trusted local experiment data, not an uploaded
    model. This is explicitly PPO-assisted fine tuning, never cold-start PPO.
    """
    import torch
    path = Path(checkpoint)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Warm-start checkpoint hash changed')
    if not 0 < noise_std <= .5:
        raise ValueError('Exploration standard deviation outside declared bounds')
    if runner.tot_timesteps != 0 or runner.current_learning_iteration != 0:
        raise ValueError('Fine-tuning budget must start at zero')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    runner.alg.policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    with torch.no_grad():
        runner.alg.policy.log_std.fill_(math.log(noise_std))
    return dict(checkpoint_sha256=digest, parent_iteration=checkpoint['iter'],
                actor_critic_and_normalizers_loaded=True, optimizer_loaded=False,
                exploration_std=noise_std, fresh_transition_budget=True)


def configure_geometry(cfg, model):
    """Change the single prototype before cloning; same offsets in all arms."""
    from isaaclab.sim.utils import clone
    import carb
    from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
    from tron1_wheel_geometry import apply_wheel_geometry
    if model not in ('authored', 'slices9'):
        raise ValueError('Unsupported native geometry')
    original = cfg.scene.robot.spawn.func
    if not hasattr(original, '__wrapped__') or original.__wrapped__.__name__ != 'spawn_from_usd':
        raise ValueError('Unexpected USD cloning implementation')
    if carb.settings.get_settings().get(SETTING_COLLISION_APPROXIMATE_CYLINDERS) is not False:
        raise ValueError('Require native cylinders, not implicit convex approximation')
    context = dict(model=model, spawn_calls=0)

    @clone
    def spawn(path, spawn_cfg, *args, **kwargs):
        if context['spawn_calls']:
            raise ValueError('Expected one prototype spawn')
        prim = original.__wrapped__(path, spawn_cfg, *args, **kwargs)
        context['prototype'] = apply_wheel_geometry(prim.GetStage(), str(prim.GetPath()),
            level=model, contact_offset_m=.001, rest_offset_m=0.)
        context['spawn_calls'] += 1
        return prim
    cfg.scene.robot.spawn.func = spawn
    return context


def audit_geometry(env, context):
    import torch
    from physx_mjlab_recipe import _shape_map
    from hockey_ppo_task import WHEELS
    robot = env.scene['robot']
    view = robot.root_physx_view
    shapes = _shape_map(robot)
    count = 9 if context['model'] == 'slices9' else 1
    slots = [slot for name in WHEELS for slot in shapes[name]]
    if context['spawn_calls'] != 1 or any(len(shapes[name]) != count for name in WHEELS):
        raise ValueError('Unexpected native wheel collider count')
    # Check every cloned physical body, not just the USD prototype.
    for links in view.link_paths:
        for name in WHEELS:
            link = links[robot.body_names.index(name)]
            single = robot._physics_sim_view.create_rigid_body_view(link)
            if single.count != 1 or single.max_shapes != count:
                raise ValueError('Cloned native wheel geometry mismatch')
    for getter, expected in (('get_contact_offsets', .001), ('get_rest_offsets', 0.)):
        values = getattr(view, getter)()[:, slots]
        torch.testing.assert_close(values, torch.full_like(values, expected), rtol=0, atol=1e-8)
    return dict(context, native_bodies_checked=env.num_envs*2,
                per_wheel_shapes=count, external_forces=0, mass_inertia_changed=False,
                elastic_tread=False, static_adhesion_model='native_PhysX')

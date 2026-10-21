"""Opt-in PhysX counterpart of the pinned mjlab tracking DR recipe.

Importing this module does not import Isaac, Torch, or start a simulator.  Call
``install_physx_mjlab_recipe`` after the existing TRON1 task/height adapters and
before constructing ManagerBasedRLEnv.  Historical experiments are untouched.

The match is in sampled scalar contact coefficients and disturbance semantics;
it is NOT a claim that PhysX and MuJoCo implement identical contact laws.
"""
from __future__ import annotations

import copy
import math


MJLAB_COMMIT = "4360096b49c1bcc478668c11ce7d458db747e49d"
LEG_JOINT_NAMES = (
    "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
    "knee_L_Joint", "knee_R_Joint",
)
WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
FRICTION_RANGE = (0.3, 1.2)
VELOCITY_RANGE = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
}


def recipe_manifest(*, training=True, fixed_mu=0.7):
    if isinstance(fixed_mu, bool) or not isinstance(fixed_mu, (int, float)):
        raise ValueError("fixed_mu must be a finite nonnegative scalar")
    if not math.isfinite(fixed_mu) or fixed_mu < 0.0:
        raise ValueError("fixed_mu must be a finite nonnegative scalar")
    return {
        "name": "tron1_mjlab_tracking_dr_v1",
        "source_repository": "https://github.com/mujocolab/mjlab",
        "source_commit": MJLAB_COMMIT,
        "source_configuration": "src/mjlab/tasks/tracking/tracking_env_cfg.py",
        "training": bool(training),
        "wheel_contact_friction": {
            "distribution": "uniform_continuous" if training else "fixed",
            "range": list(FRICTION_RANGE) if training else [fixed_mu, fixed_mu],
            "sampling": "startup_once_per_environment_shared_between_both_wheels",
            "physx_static_equals_dynamic": True,
            "physx_ground_friction": 1.0,
            "physx_ground_friction_combine_mode": "multiply",
            "mujoco_wheel_geom_priority": 1,
            "mujoco_tangential_axis": 0,
            "buckets": None,
        },
        # Unmodified nonwheel MuJoCo geoms have mu=.6 and priority=0, the
        # plane has mu=1 and priority=0: max-combination yields contact mu=1.
        # PhysX multiply with unit ground therefore needs nonwheel mu=1 too.
        # This adapts raw material values to match plane-contact coefficients;
        # it is not an assertion that the two contact solvers are equivalent.
        "nonwheel_friction": 1.0 if training else float(fixed_mu),
        "nonwheel_plane_contact_mapping": {
            "mujoco_raw_nonwheel_friction": 0.6 if training else float(fixed_mu),
            "mujoco_raw_ground_friction": 1.0 if training else float(fixed_mu),
            "mujoco_nonwheel_and_ground_priority": 0,
            "mujoco_combine_rule": "max_for_equal_priority",
            "physx_combine_rule": "multiply_by_unit_ground",
            "effective_plane_friction": 1.0 if training else float(fixed_mu),
            "raw_material_values_identical": False,
            "purpose": "match_nonwheel_plane_contact_coefficient_not_solver_dynamics",
        },
        "restitution_randomization": False,
        "physx_robot_and_ground_restitution": 0.0,
        "mujoco_restitution_equivalence_claimed": False,
        "encoder_bias_range_rad": [-0.01, 0.01] if training else [0.0, 0.0],
        "encoder_bias_semantics": {
            "applies_to": "six_position_controlled_leg_joints",
            "actor_joint_position": "q + b - nominal_q0",
            "critic_joint_position": "q - nominal_q0",
            "position_control_target": "nominal_q0 + clipped_action - b",
            "default_joint_positions_mutated": False,
            "wheel_velocity_noise": False,
        },
        "base_com_offset_range_m": {
            "x": [-0.025, 0.025], "y": [-0.05, 0.05], "z": [-0.05, 0.05],
        } if training else {key: [0.0, 0.0] for key in "xyz"},
        "base_com_body": "base_Link",
        "push_interval_s": [1.0, 3.0] if training else None,
        "push_velocity_range": copy.deepcopy(VELOCITY_RANGE) if training else {},
        "push_semantics": "add_world_root_LINK_velocity_delta_not_COM_velocity",
        "push_timer_reset": "resample_per_episode_matching_intended_recipe_and_Isaac",
        "mjlab_v1_4_function_interval_reset_bug_reproduced": False,
        "task_action_reward_termination_changed": False,
        "additional_observation_corruption": False,
        "physics_equivalence_claimed": False,
    }


def _device_ids(env, env_ids, *, device):
    import torch
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=device)


def _shape_map(asset):
    """Map articulation body names to contiguous PhysX shape slots, with readback."""
    paths = asset.root_physx_view.link_paths[0]
    if len(paths) != len(asset.body_names):
        raise RuntimeError("PhysX body path order does not match articulation body names")
    counts = [asset._physics_sim_view.create_rigid_body_view(path).max_shapes
              for path in paths]
    if sum(counts) != asset.root_physx_view.max_shapes:
        raise RuntimeError("PhysX per-body shape counts do not cover the material table")
    result = {}
    start = 0
    for name, count in zip(asset.body_names, counts):
        result[name] = list(range(start, start + count))
        start += count
    for name in WHEEL_BODY_NAMES:
        if not result.get(name):
            raise RuntimeError(f"Missing collidable wheel body {name}")
    return result


def initialize_contact_materials(env, env_ids, training=True, fixed_mu=0.7):
    """Continuous per-world draw, not Isaac's independent material buckets."""
    import torch
    manifest = recipe_manifest(training=training, fixed_mu=fixed_mu)
    asset = env.scene["robot"]
    ids = _device_ids(env, env_ids, device="cpu")
    mapping = _shape_map(asset)
    materials = asset.root_physx_view.get_material_properties().clone()
    if materials.shape != (env.num_envs, asset.root_physx_view.max_shapes, 3):
        raise RuntimeError("Unexpected live PhysX material-table dimensions")
    nonwheel_mu = manifest["nonwheel_friction"]
    materials[ids, :, :2] = nonwheel_mu
    materials[ids, :, 2] = 0.0
    if training:
        values = torch.rand(len(ids), device=env.device).cpu()
        values = values * (FRICTION_RANGE[1] - FRICTION_RANGE[0]) + FRICTION_RANGE[0]
    else:
        values = torch.full((len(ids),), fixed_mu, dtype=materials.dtype)
    wheel_slots = [slot for name in WHEEL_BODY_NAMES for slot in mapping[name]]
    slot_ids = torch.tensor(wheel_slots, dtype=torch.long)
    materials[ids[:, None], slot_ids[None, :], :2] = values[:, None, None]
    asset.root_physx_view.set_material_properties(materials, ids.to(dtype=torch.int32))
    if not hasattr(env, "mjlab_recipe_mu"):
        env.mjlab_recipe_mu = torch.full((env.num_envs,), float("nan"))
    env.mjlab_recipe_mu[ids] = values
    env.mjlab_recipe_shape_slots = mapping
    env.mjlab_recipe_manifest = manifest


def initialize_encoder_bias(env, env_ids, training=True):
    """Apply the upstream sensor/target signs without changing the nominal q0."""
    import torch
    asset = env.scene["robot"]
    ids = _device_ids(env, env_ids, device=env.device)
    command = env.command_manager.get_term("motion")
    joint_ids = command.leg_joint_indices
    nominal = asset.data.default_joint_pos[:, joint_ids]
    action = env.action_manager.get_term("joint_pos")
    if tuple(action._offset.shape) != (env.num_envs, 6):
        raise RuntimeError("Expected six ordered leg position-action offsets")
    if not hasattr(env, "mjlab_recipe_encoder_bias"):
        env.mjlab_recipe_encoder_bias = torch.zeros_like(nominal)
    bias = (torch.rand((len(ids), 6), device=env.device) * 0.02 - 0.01
            if training else torch.zeros((len(ids), 6), device=env.device))
    env.mjlab_recipe_encoder_bias[ids] = bias
    action._offset[ids] = nominal[ids] - bias


def biased_leg_joint_positions(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    asset = command.robot
    ids = command.leg_joint_indices
    value = (asset.data.joint_pos - asset.data.default_joint_pos)[:, ids]
    # ObservationManager probes shapes before startup events have run.
    bias = getattr(env, "mjlab_recipe_encoder_bias", None)
    return value if bias is None else value + bias


def unbiased_leg_joint_positions(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    asset = command.robot
    ids = command.leg_joint_indices
    return (asset.data.joint_pos - asset.data.default_joint_pos)[:, ids]


def initialize_base_com(env, env_ids):
    import torch
    asset = env.scene["robot"]
    ids = _device_ids(env, env_ids, device="cpu")
    body_id = asset.body_names.index("base_Link")
    coms = asset.root_physx_view.get_coms().clone()
    offsets = (torch.rand((len(ids), 3), device=env.device).cpu() * 2.0 - 1.0)
    offsets *= torch.tensor([0.025, 0.05, 0.05])
    coms[ids, body_id, :3] += offsets
    asset.root_physx_view.set_coms(coms, ids.to(dtype=torch.int32))
    if not hasattr(env, "mjlab_recipe_com_offsets"):
        env.mjlab_recipe_com_offsets = torch.zeros((env.num_envs, 3))
    env.mjlab_recipe_com_offsets[ids] = offsets


def push_root_link_velocity(env, env_ids, velocity_range):
    """Use explicit link APIs; PhysX's legacy root velocity alias is COM-based."""
    import torch
    asset = env.scene["robot"]
    ids = _device_ids(env, env_ids, device=env.device)
    velocity = asset.data.root_link_vel_w[ids].clone()
    bounds = torch.tensor([velocity_range.get(key, (0.0, 0.0))
                           for key in ("x", "y", "z", "roll", "pitch", "yaw")],
                          dtype=velocity.dtype, device=env.device)
    delta = torch.rand(velocity.shape, dtype=velocity.dtype, device=env.device)
    delta = delta * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    asset.write_root_link_velocity_to_sim(velocity + delta, env_ids=ids)
    env.mjlab_recipe_push_count = getattr(env, "mjlab_recipe_push_count", 0) + len(ids)


def install_physx_mjlab_recipe(cfg, *, training=True, fixed_mu=0.7):
    """Install before environment creation; leave task/reward/height gates alone."""
    from isaaclab.managers import EventTermCfg

    manifest = recipe_manifest(training=training, fixed_mu=fixed_mu)
    material = cfg.scene.terrain.physics_material
    material.static_friction = material.dynamic_friction = 1.0
    material.restitution = 0.0
    material.friction_combine_mode = "multiply"
    material.restitution_combine_mode = "multiply"
    cfg.sim.physics_material = material
    # Remove original PhysX-specific DR instead of stacking two recipes.
    for name in ("physics_material", "add_joint_default_pos", "base_com",
                 "push_robot", "domain_randomization"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.events.mjlab_contact_material = EventTermCfg(
        func=initialize_contact_materials, mode="startup",
        params={"training": bool(training), "fixed_mu": float(fixed_mu)})
    cfg.events.mjlab_encoder_bias = EventTermCfg(
        func=initialize_encoder_bias, mode="startup", params={"training": bool(training)})
    if training:
        cfg.events.mjlab_base_com = EventTermCfg(func=initialize_base_com, mode="startup")
        cfg.events.mjlab_push_robot = EventTermCfg(
            func=push_root_link_velocity, mode="interval", interval_range_s=(1.0, 3.0),
            params={"velocity_range": copy.deepcopy(VELOCITY_RANGE)})
    else:
        cfg.events.mjlab_base_com = cfg.events.mjlab_push_robot = None
    cfg.observations.policy.joint_pos.func = biased_leg_joint_positions
    cfg.observations.critic.joint_pos.func = unbiased_leg_joint_positions
    return manifest


def audit_physx_mjlab_recipe(env, *, check_usd=True):
    """Fail closed on material/bias/offset readback; optionally inspect bound USD."""
    import torch
    if not hasattr(env, "mjlab_recipe_manifest"):
        raise RuntimeError("mjlab recipe startup material event has not run")
    manifest = env.mjlab_recipe_manifest
    asset = env.scene["robot"]
    values = asset.root_physx_view.get_material_properties().cpu()
    slots = env.mjlab_recipe_shape_slots
    wheel_slots = [slot for name in WHEEL_BODY_NAMES for slot in slots[name]]
    nonwheel_slots = [slot for name, shape_slots in slots.items()
                      if name not in WHEEL_BODY_NAMES for slot in shape_slots]
    mu = env.mjlab_recipe_mu
    expected = mu[:, None, None].expand(-1, len(wheel_slots), 2)
    if (not torch.isfinite(values).all()
            or not torch.allclose(values[:, wheel_slots, :2], expected, atol=1e-6, rtol=0.0)
            or torch.count_nonzero(values[..., 2])):
        raise RuntimeError("Live wheel material coefficient/restitution readback differs from the draw")
    if nonwheel_slots and not torch.allclose(values[:, nonwheel_slots, :2],
            torch.full_like(values[:, nonwheel_slots, :2], manifest["nonwheel_friction"]),
            atol=1e-6, rtol=0.0):
        raise RuntimeError("Nonwheel materials differ from the declared fixed coefficients")
    command = env.command_manager.get_term("motion")
    nominal = asset.data.default_joint_pos[:, command.leg_joint_indices]
    actual_offset = env.action_manager.get_term("joint_pos")._offset
    bias = env.mjlab_recipe_encoder_bias
    if not torch.allclose(actual_offset, nominal - bias, atol=1e-7, rtol=0.0):
        raise RuntimeError("Encoder bias target sign or nominal action offset is incorrect")
    lower, upper = manifest["wheel_contact_friction"]["range"]
    if torch.any((mu < lower - 1e-6) | (mu > upper + 1e-6)):
        raise RuntimeError("Friction draw lies outside the pinned recipe range")
    result = {"recipe": copy.deepcopy(manifest),
              "wheel_mu_range": [float(mu.min()), float(mu.max())],
              "wheel_shape_slots": wheel_slots,
              "all_wheel_shapes_share_one_draw_per_environment": True,
              "encoder_bias_range_rad": [float(bias.min()), float(bias.max())],
              "position_target_bias_sign_verified": True,
              "live_physx_materials_verified": True,
              "ground_material_usd_verified": False}
    if check_usd:
        result.update(_audit_bound_materials(env))
        result["ground_material_usd_verified"] = True
    return result


def _collision_enabled_in_hierarchy(prim, UsdPhysics):
    """Isaac's imported TRON1 applies CollisionAPI to a parent Xform.

    The actual Cube/Cylinder descendants need their material audited, whereas
    that non-geometric Xform is not a physical shape. A collider can therefore
    be a Gprim with CollisionAPI on itself OR on an enclosing collision scope.
    """
    current = prim
    while current:
        if current.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(current).GetCollisionEnabledAttr().Get()
            return enabled is not False
        current = current.GetParent()
    return False


def _audit_bound_materials(env):
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    ground = []
    robot = []
    ignored_containers = []
    verified_default = None

    def default_scene_material():
        nonlocal verified_default
        if verified_default is not None:
            return verified_default
        # Installed Isaac Lab SimulationContext._set_additional_physx_params
        # creates cfg.physics_prim_path/defaultMaterial and binds it to the
        # physics scene; this is PhysX's fallback for unbound shape geometry.
        # Verify that binding, never infer the mode from missing shape bindings.
        path = env.cfg.sim.physics_prim_path
        scene = stage.GetPrimAtPath(path)
        if not scene or not scene.IsA(UsdPhysics.Scene):
            raise RuntimeError(f"No auditable physics scene for unbound robot geometry: {path}")
        material, _ = UsdShade.MaterialBindingAPI(scene).ComputeBoundMaterial("physics")
        if not material:
            raise RuntimeError(f"Physics scene lacks a bound default material: {path}")
        api = UsdPhysics.MaterialAPI(material.GetPrim())
        values = [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(),
                  api.GetRestitutionAttr().Get()]
        mode = PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetFrictionCombineModeAttr().Get()
        if mode != "multiply" or values != [1.0, 1.0, 0.0]:
            raise RuntimeError("Physics scene default is not verified unit-friction multiply")
        verified_default = material
        return material

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        is_ground = path.startswith("/World/ground/")
        is_first_robot = "/env_0/Robot/" in path
        if not (is_ground or is_first_robot):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                ignored_containers.append(path)
            continue
        if not _collision_enabled_in_hierarchy(prim, UsdPhysics):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        binding_source = "shape_or_inherited_physics_material"
        if not material and is_first_robot:
            material = default_scene_material()
            binding_source = "verified_physics_scene_default_material"
        if not material:
            raise RuntimeError(f"Collider lacks an auditable bound material: {path}")
        physx = PhysxSchema.PhysxMaterialAPI(material.GetPrim())
        mode = physx.GetFrictionCombineModeAttr().Get()
        # Average/min/multiply lose to ground multiply. Max would win and floor
        # coefficient 1 would silently eliminate the low-friction training draws.
        if is_first_robot and mode == "max":
            raise RuntimeError(f"Robot material max-combine overrides wheel DR: {path}")
        if is_first_robot:
            robot.append({"collider": path, "material": str(material.GetPath()),
                          "binding_source": binding_source,
                          "friction_combine_mode": mode or "average_schema_default"})
        if is_ground:
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            values = [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(),
                      api.GetRestitutionAttr().Get()]
            if mode != "multiply" or values != [1.0, 1.0, 0.0]:
                raise RuntimeError(f"Ground material is not unit-friction multiply: {path}")
            ground.append({"collider": path, "material": str(material.GetPath()),
                           "friction": 1.0, "restitution": 0.0, "combine": mode})
    if not ground:
        raise RuntimeError("No bound ground collision material was verified")
    if not robot:
        raise RuntimeError("No actual robot collision geometry material was verified")
    return {"ground_colliders": ground, "robot_collision_geometry_materials": robot,
            "ignored_non_geometry_collision_containers": ignored_containers}


# Public names used by the paired-backend training harness.
apply_mjlab_recipe = install_physx_mjlab_recipe
recipe_audit = audit_physx_mjlab_recipe

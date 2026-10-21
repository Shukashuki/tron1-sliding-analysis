"""Asset-free recipe tests; runtime Torch checks run when Torch is available."""
import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import physx_mjlab_recipe as recipe


def fake_cfg():
    material = SimpleNamespace(static_friction=.7, dynamic_friction=.7,
                               restitution=.5, friction_combine_mode="average")
    return SimpleNamespace(
        scene=SimpleNamespace(terrain=SimpleNamespace(physics_material=material)),
        sim=SimpleNamespace(physics_material=object()),
        events=SimpleNamespace(physics_material=object(), add_joint_default_pos=object(),
                               base_com=object(), push_robot=object(), domain_randomization=object(),
                               legacy_joint_friction=object()),
        observations=SimpleNamespace(
            policy=SimpleNamespace(joint_pos=SimpleNamespace(func=object())),
            critic=SimpleNamespace(joint_pos=SimpleNamespace(func=object()))),
        rewards=SimpleNamespace(sideslip=object(), grounded=object()),
        terminations=SimpleNamespace(anchor_pos=None, ee_body_pos=None, anchor_ori=object()))


def test_manifest_pins_exact_shared_tangent_recipe_not_legacy_buckets():
    manifest = recipe.recipe_manifest()
    assert manifest["source_commit"] == "4360096b49c1bcc478668c11ce7d458db747e49d"
    assert manifest["wheel_contact_friction"]["range"] == [.3, 1.2]
    assert manifest["wheel_contact_friction"]["buckets"] is None
    assert manifest["wheel_contact_friction"]["mujoco_wheel_geom_priority"] == 1
    assert manifest["physx_robot_and_ground_restitution"] == 0.
    assert manifest["restitution_randomization"] is False
    assert manifest["physics_equivalence_claimed"] is False
    assert manifest["nonwheel_friction"] == 1.
    mapping = manifest["nonwheel_plane_contact_mapping"]
    assert mapping["mujoco_raw_nonwheel_friction"] == .6
    assert mapping["effective_plane_friction"] == 1.
    assert mapping["raw_material_values_identical"] is False
    assert manifest["encoder_bias_semantics"]["critic_joint_position"] == "q - nominal_q0"
    assert manifest["mjlab_v1_4_function_interval_reset_bug_reproduced"] is False


@pytest.mark.parametrize("mu", [float("nan"), float("inf"), -.01, True, "0.7"])
def test_manifest_rejects_invalid_fixed_coefficient(mu):
    with pytest.raises(ValueError):
        recipe.recipe_manifest(fixed_mu=mu)


def test_installer_replaces_dr_without_changing_task_or_reintroducing_height(monkeypatch):
    module = ModuleType("isaaclab.managers")
    module.EventTermCfg = SimpleNamespace
    monkeypatch.setitem(sys.modules, "isaaclab.managers", module)
    cfg = fake_cfg()
    old_reward, old_termination, old_clear = cfg.rewards, cfg.terminations, cfg.events.legacy_joint_friction
    manifest = recipe.apply_mjlab_recipe(cfg)
    assert cfg.rewards is old_reward and cfg.terminations is old_termination
    assert cfg.events.legacy_joint_friction is old_clear
    assert cfg.terminations.anchor_pos is None and cfg.terminations.ee_body_pos is None
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot", "domain_randomization"):
        assert getattr(cfg.events, name) is None
    assert cfg.scene.terrain.physics_material.friction_combine_mode == "multiply"
    assert cfg.scene.terrain.physics_material.static_friction == 1.
    assert cfg.scene.terrain.physics_material.dynamic_friction == 1.
    assert cfg.sim.physics_material is cfg.scene.terrain.physics_material
    assert cfg.observations.policy.joint_pos.func is recipe.biased_leg_joint_positions
    assert cfg.observations.critic.joint_pos.func is recipe.unbiased_leg_joint_positions
    assert cfg.events.mjlab_push_robot.interval_range_s == (1., 3.)
    assert manifest["task_action_reward_termination_changed"] is False
    eval_cfg = fake_cfg()
    recipe.install_physx_mjlab_recipe(eval_cfg, training=False, fixed_mu=.1)
    assert eval_cfg.events.mjlab_base_com is None and eval_cfg.events.mjlab_push_robot is None
    assert eval_cfg.events.mjlab_contact_material.params == {"training": False, "fixed_mu": .1}


def fake_env(num_envs=64):
    torch = pytest.importorskip("torch")
    names = ["base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link",
             "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link", "imu_Link"]
    counts = [1] * 9 + [0]

    class View:
        link_paths = [[f"/robot/{name}" for name in names]]
        max_shapes = 9
        materials = torch.zeros((num_envs, 9, 3))
        coms = torch.zeros((num_envs, 10, 7))

        def get_material_properties(self):
            return self.materials

        def set_material_properties(self, values, ids):
            self.materials[ids] = values[ids]

        def get_coms(self):
            return self.coms

        def set_coms(self, values, ids):
            self.coms[ids] = values[ids]

    shapes = dict(zip(View.link_paths[0], counts))
    asset = SimpleNamespace(
        root_physx_view=View(), body_names=names,
        _physics_sim_view=SimpleNamespace(
            create_rigid_body_view=lambda path: SimpleNamespace(max_shapes=shapes[path])),
        data=SimpleNamespace(default_joint_pos=torch.arange(8).float().repeat(num_envs, 1),
                             joint_pos=torch.arange(8).float().repeat(num_envs, 1) + .1,
                             root_link_vel_w=torch.ones((num_envs, 6))))
    action = SimpleNamespace(_offset=asset.data.default_joint_pos[:, :6].clone())
    command = SimpleNamespace(robot=asset, leg_joint_indices=list(range(6)))
    env = SimpleNamespace(
        num_envs=num_envs, device="cpu", scene={"robot": asset},
        command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(get_term=lambda name: action))
    asset.write_root_link_velocity_to_sim = lambda values, env_ids: setattr(env, "written_link_velocity", values)
    return env


def test_runtime_continuous_shared_wheel_draw_preserves_nonwheel_and_has_no_buckets():
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    env = fake_env(128)
    recipe.initialize_contact_materials(env, None)
    recipe.initialize_encoder_bias(env, None)
    audit = recipe.recipe_audit(env, check_usd=False)
    assert audit["all_wheel_shapes_share_one_draw_per_environment"] is True
    values = env.scene["robot"].root_physx_view.materials
    torch.testing.assert_close(values[:, 7, :2], values[:, 8, :2])
    assert len(torch.unique(env.mjlab_recipe_mu)) == 128
    assert env.mjlab_recipe_mu.min() < .6 and env.mjlab_recipe_mu.max() > 1.
    torch.testing.assert_close(values[:, :7, :2], torch.ones_like(values[:, :7, :2]))
    assert not torch.count_nonzero(values[..., 2])
    values[0, 7, 0] += .1
    with pytest.raises(RuntimeError, match="readback"):
        recipe.recipe_audit(env, check_usd=False)


def test_runtime_encoder_actor_plus_target_minus_b_critic_unbiased_and_default_unchanged():
    torch = pytest.importorskip("torch")
    env = fake_env()
    q0 = env.scene["robot"].data.default_joint_pos.clone()
    raw = recipe.biased_leg_joint_positions(env)
    torch.testing.assert_close(raw, recipe.unbiased_leg_joint_positions(env))
    recipe.initialize_encoder_bias(env, None)
    bias = env.mjlab_recipe_encoder_bias
    torch.testing.assert_close(env.scene["robot"].data.default_joint_pos, q0)
    torch.testing.assert_close(recipe.biased_leg_joint_positions(env), raw + bias)
    torch.testing.assert_close(recipe.unbiased_leg_joint_positions(env), raw)
    torch.testing.assert_close(env.action_manager.get_term("joint_pos")._offset, q0[:, :6] - bias)
    assert torch.max(torch.abs(bias)) <= .01


def test_runtime_com_one_body_only_and_root_LINK_push_is_added_not_overwritten():
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    env = fake_env()
    recipe.initialize_base_com(env, None)
    coms = env.scene["robot"].root_physx_view.coms
    torch.testing.assert_close(coms[:, 0, :3], env.mjlab_recipe_com_offsets)
    assert not torch.count_nonzero(coms[:, 1:])
    assert torch.max(torch.abs(coms[:, 0, 0])) <= .025
    assert torch.max(torch.abs(coms[:, 0, 1:3])) <= .05
    recipe.push_root_link_velocity(env, torch.tensor([2, 5]), recipe.VELOCITY_RANGE)
    assert env.mjlab_recipe_push_count == 2
    delta = env.written_link_velocity - 1.
    bounds = torch.tensor([.5, .5, .2, .52, .52, .78])
    assert torch.all(torch.abs(delta) <= bounds + 1.e-7)
    assert torch.all(env.written_link_velocity[:, :2] >= .5)


def test_fixed_validation_disables_noise_and_all_robot_pair_mu_is_the_declared_value():
    torch = pytest.importorskip("torch")
    env = fake_env()
    recipe.initialize_contact_materials(env, None, training=False, fixed_mu=.03)
    recipe.initialize_encoder_bias(env, None, training=False)
    audit = recipe.recipe_audit(env, check_usd=False)
    assert audit["wheel_mu_range"] == pytest.approx([.03, .03])
    values = env.scene["robot"].root_physx_view.materials
    torch.testing.assert_close(values[..., :2], torch.full_like(values[..., :2], .03))
    assert not torch.count_nonzero(env.mjlab_recipe_encoder_bias)


def test_shape_map_fails_closed_when_material_shapes_are_not_covered():
    env = fake_env()
    env.scene["robot"].root_physx_view.max_shapes = 10
    with pytest.raises(RuntimeError, match="shape counts"):
        recipe.initialize_contact_materials(env, None)


def fake_usd_stage(monkeypatch, *, robot_material="missing", default_material=True):
    """Exercise the actual audit control flow without Isaac or USD binaries."""
    gprim_type, scene_type = object(), object()

    class Attr:
        def __init__(self, value):
            self.value = value

        def Get(self):
            return self.value

    class CollisionAPI:
        def __init__(self, prim):
            self.prim = prim

        def GetCollisionEnabledAttr(self):
            return Attr(True)

    class Material:
        def __init__(self, path, mode="multiply"):
            self.path, self.mode = path, mode

        def GetPath(self):
            return self.path

        def GetPrim(self):
            return self

        def GetStaticFrictionAttr(self):
            return Attr(1.)

        def GetDynamicFrictionAttr(self):
            return Attr(1.)

        def GetRestitutionAttr(self):
            return Attr(0.)

        def GetFrictionCombineModeAttr(self):
            return Attr(self.mode)

    class Prim:
        def __init__(self, path, *, kind=None, collision=False, parent=None, material=None):
            self.path, self.kind, self.collision = path, kind, collision
            self.parent, self.material = parent, material

        def GetPath(self):
            return self.path

        def GetParent(self):
            return self.parent

        def HasAPI(self, schema):
            return schema is CollisionAPI and self.collision

        def IsA(self, schema):
            return self.kind is schema

    material = Material("/physicsScene/defaultMaterial")
    scene = Prim("/physicsScene", kind=scene_type,
                 material=material if default_material else None)
    container = Prim("/World/envs/env_0/Robot/base_Link/collisions", collision=True)
    bound_robot = None if robot_material == "missing" else Material("/robotMaterial", robot_material)
    robot = Prim(container.path + "/mesh_0/box", kind=gprim_type,
                 parent=container, material=bound_robot)
    ground = Prim("/World/ground/plane", kind=gprim_type, collision=True,
                  material=material)
    visual = Prim("/World/envs/env_0/Robot/base_Link/visual/mesh", kind=gprim_type)
    stage = SimpleNamespace(Traverse=lambda: [scene, container, robot, ground, visual],
                            GetPrimAtPath=lambda path: scene if path == "/physicsScene" else None)
    pxr = ModuleType("pxr")
    pxr.UsdGeom = SimpleNamespace(Gprim=gprim_type)
    pxr.UsdPhysics = SimpleNamespace(CollisionAPI=CollisionAPI, Scene=scene_type,
                                    MaterialAPI=lambda material: material)
    pxr.UsdShade = SimpleNamespace(MaterialBindingAPI=lambda prim: SimpleNamespace(
        ComputeBoundMaterial=lambda purpose: (prim.material, None)))
    pxr.PhysxSchema = SimpleNamespace(PhysxMaterialAPI=lambda material: material)
    omni, usd = ModuleType("omni"), ModuleType("omni.usd")
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni.usd = usd
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    env = SimpleNamespace(cfg=SimpleNamespace(sim=SimpleNamespace(physics_prim_path="/physicsScene")))
    return env, robot, ground, material


def test_usd_audit_checks_geometry_under_collision_xform_and_verified_default_fallback(monkeypatch):
    env, robot, ground, _ = fake_usd_stage(monkeypatch)
    audit = recipe._audit_bound_materials(env)
    shapes = audit["robot_collision_geometry_materials"]
    assert len(shapes) == 1 and shapes[0]["collider"] == robot.path
    assert shapes[0]["binding_source"] == "verified_physics_scene_default_material"
    assert audit["ground_colliders"][0]["collider"] == ground.path
    assert audit["ignored_non_geometry_collision_containers"] == [robot.parent.path]


def test_usd_audit_does_not_silently_skip_actual_unbound_robot_shape(monkeypatch):
    env, _, _, _ = fake_usd_stage(monkeypatch, default_material=False)
    with pytest.raises(RuntimeError, match="bound default material"):
        recipe._audit_bound_materials(env)


def test_usd_audit_rejects_shape_max_mode_and_bad_scene_default(monkeypatch):
    env, _, _, _ = fake_usd_stage(monkeypatch, robot_material="max")
    with pytest.raises(RuntimeError, match="max-combine"):
        recipe._audit_bound_materials(env)
    env, _, _, material = fake_usd_stage(monkeypatch)
    material.mode = "average"
    with pytest.raises(RuntimeError, match="default is not verified"):
        recipe._audit_bound_materials(env)


def test_usd_audit_requires_real_robot_geometry_and_bound_ground(monkeypatch):
    env, robot, ground, _ = fake_usd_stage(monkeypatch)
    robot.kind = None
    with pytest.raises(RuntimeError, match="No actual robot collision geometry"):
        recipe._audit_bound_materials(env)
    env, _, ground, _ = fake_usd_stage(monkeypatch)
    ground.material = None
    with pytest.raises(RuntimeError, match="Collider lacks an auditable bound material"):
        recipe._audit_bound_materials(env)

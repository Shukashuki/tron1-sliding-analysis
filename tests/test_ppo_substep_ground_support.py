"""CPU Torch lifecycle/math fixtures only; no Isaac/GPU physics is started."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ppo_substep_ground_support as support


class Scene:
    env_regex_ns = "/World/envs/env_.*"

    def __init__(self, robot, sensors):
        self.robot, self.sensors, self.calls = robot, sensors, 0

    def __getitem__(self, key):
        assert key == "robot"
        return self.robot

    def update(self, dt):
        self.calls += 1


def sensor(body, values):
    cfg = SimpleNamespace(update_period=0., history_length=0, track_contact_points=False,
                          track_air_time=False, track_pose=False,
                          filter_prim_paths_expr=[support.GROUND_COLLIDER],
                          prim_path="{ENV_REGEX_NS}/Robot/"+body,
                          max_contact_data_count_per_prim=32)
    return SimpleNamespace(cfg=cfg, body_names=[body], num_bodies=1,
                           contact_physx_view=SimpleNamespace(filter_count=1, sensor_count=2,
                                                               max_contact_data_count=64),
                           _sim_physics_dt=.005,
                           data=SimpleNamespace(force_matrix_w=values))


class Env:
    def __init__(self):
        self.num_envs, self.device = 2, "cpu"
        self.cfg = SimpleNamespace(decimation=4, sim=SimpleNamespace(dt=.005, gravity=(0., 0., -10.)))
        self.physics_dt, self.step_dt = .005, .02
        self._sim_step_counter = 0
        self.sim = SimpleNamespace(current_time_step_index=0, current_time=0., get_physics_dt=lambda: .005)
        self.mass_reads = 0
        self.masses = torch.tensor([[2., 3.], [4., 6.]], dtype=torch.float32)

        def get_masses():
            self.mass_reads += 1
            return self.masses

        self.robot = SimpleNamespace(body_names=["base_Link", "wheel_Link"],
                                     root_physx_view=SimpleNamespace(get_masses=get_masses),
                                     actuators={"legs": SimpleNamespace(stiffness=torch.full((2, 6), 500.),
                                                                        damping=torch.full((2, 6), 10.))})
        sensors = {name: sensor(body, torch.zeros((2, 1, 1, 3)))
                   for name, body in zip(support.SENSOR_NAMES, support.WHEEL_NAMES)}
        self.scene = Scene(self.robot, sensors)
        self.force_sequences = [torch.tensor([[25., 25.], [50., 50.]])]*4
        self.extra_refresh = self.read_twice = self.reset_after_reward = self.reset_before_reward = False
        self.skip_reward = False
        self.action_received = None

    def step(self, actions=None):
        self.action_received = actions
        for value in self.force_sequences:
            for index, name in enumerate(support.SENSOR_NAMES):
                matrix = self.scene.sensors[name].data.force_matrix_w
                matrix.zero_(); matrix[:, 0, 0, 2] = value[:, index]
            self._sim_step_counter += 1; self.sim.current_time_step_index += 1; self.sim.current_time += .005
            self.scene.update(.005)
            if self.extra_refresh:
                self.scene.update(.005)
        handler = getattr(self, support.ENV_MARKER)
        if self.reset_before_reward:
            handler.reset([0])
        result = None if self.skip_reward else handler.reward().clone()
        if self.read_twice:
            assert torch.equal(result, handler.reward())
        if self.reset_after_reward:
            handler.reset([0])
            for name in support.SENSOR_NAMES:
                self.scene.sensors[name].data.force_matrix_w.zero_()
            self.scene.update(.005)
        return result


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(support, "_event_audit", lambda env: [{"known_historical_recipe": True}])
    monkeypatch.setattr(support, "_ground_audit", lambda: {"path": support.GROUND_COLLIDER, "usd_type": "Plane"})
    monkeypatch.setattr(support, "_native_gravity", lambda: (0., 0., -10.))
    return Env()


def install(env):
    return support.GroundSupportAccumulator(env, _allow_cpu_fixture=True)


def test_balanced_support_matches_exact_mean_cost_and_mass_is_read_once(env):
    handler = install(env)
    result = env.step()
    torch.testing.assert_close(result, torch.full((2,), 1.-torch.exp(torch.tensor(-2.5))))
    env.step()
    assert env.mass_reads == 1
    assert handler.total_samples == 8 and handler.steps_completed == 2
    assert handler.unique_cost_windows == 2
    audit = handler.audit(check_device_health=True)
    assert audit["device_health_finite"]
    assert audit["mass_kg_range"] == [5., 10.]
    assert audit["mg_N_range"] == [50., 100.]
    assert audit["positive_normal_force_samples_left_right"] == [16, 16]
    assert audit["negative_normal_force_samples"] == 0
    assert audit["observed_ground_fn_z_min_N"] == [25., 25.]
    assert audit["observed_ground_fn_z_max_N"] == [50., 50.]


def test_no_support_is_zero_and_one_wheel_has_lower_reward_than_two(env):
    install(env)
    env.force_sequences = [torch.zeros((2, 2))]*4
    assert torch.equal(env.step(), torch.zeros(2))
    env.force_sequences = [torch.tensor([[50., 0.], [100., 0.]])]*4
    one = env.step()
    expected = 1.-(1.+torch.exp(torch.tensor(-5.)))/2.
    torch.testing.assert_close(one, torch.full((2,), expected))
    assert bool((one < 1.-torch.exp(torch.tensor(-2.5))).all())


def test_mean_of_costs_not_cost_of_mean_force(env):
    install(env)
    env.force_sequences = [torch.tensor([[100., 100.], [200., 200.]])]+[torch.zeros((2, 2))]*3
    measured = env.step()
    mean_cost_result = .25*(1.-torch.exp(torch.tensor(-10.)))
    cost_of_mean_result = 1.-torch.exp(torch.tensor(-2.5))
    torch.testing.assert_close(measured, torch.full((2,), mean_cost_result))
    assert float(measured[0]) < float(cost_of_mean_result)-.5


def test_same_counter_refreshes_and_reset_updates_do_not_duplicate_samples(env):
    env.extra_refresh = env.reset_after_reward = True
    handler = install(env)
    value = env.step()
    assert handler.samples == handler.total_samples == 4
    assert env.scene.calls == 9
    assert handler.reset_calls == 1
    env.reset_after_reward = False
    env.force_sequences = [torch.zeros((2, 2))]*4
    assert torch.equal(env.step(), torch.zeros(2))
    assert bool((value > 0.).all())


def test_no_history_splices_partial_reset_or_force_discontinuity(env):
    handler = install(env)
    env.step(); handler.reset([0])
    env.force_sequences = [torch.zeros((2, 2))]*4
    assert torch.equal(env.step(), torch.zeros(2))
    assert handler.audit()["history_carried_across_windows"] is False


def test_duplicate_reward_reads_are_idempotent(env):
    env.read_twice = True
    handler = install(env); env.step()
    assert handler.cost_reads == 2 and handler.unique_cost_windows == 1


def test_reset_before_reward_is_rejected(env):
    env.reset_before_reward = True
    handler = install(env)
    with pytest.raises(ValueError, match="reset preceded ground reward"):
        env.step()
    assert handler.steps_failed == 1


def test_reward_not_consumed_is_rejected(env):
    env.skip_reward = True
    handler = install(env)
    with pytest.raises(ValueError, match="not consumed"):
        env.step()
    assert handler.steps_failed == 1


def test_outside_step_forward_has_no_samples_and_reward_unavailable(env):
    handler = install(env)
    env.scene.update(.005)
    assert handler.total_samples == 0
    with pytest.raises(ValueError, match="all four"):
        handler.reward()


@pytest.mark.parametrize("count", [3, 5])
def test_missing_or_extra_native_substep_fails(env, count):
    env.force_sequences = env.force_sequences[:1]*count
    handler = install(env)
    with pytest.raises(ValueError):
        env.step()
    assert handler.steps_failed == 1


def test_force_read_happens_after_original_scene_update(env):
    original = env.scene.update
    env.force_sequences = [torch.zeros((2, 2))]*4
    def update(dt):
        original(dt)
        for name in support.SENSOR_NAMES:
            env.scene.sensors[name].data.force_matrix_w[:, 0, 0, 2] += torch.tensor([25., 50.])
    env.scene.update = update
    install(env)
    torch.testing.assert_close(env.step(), torch.full((2,), 1.-torch.exp(torch.tensor(-2.5))))


def test_early_nonfinite_force_latches_even_if_later_samples_are_finite(env):
    handler = install(env)
    env.force_sequences = [value.clone() for value in env.force_sequences]
    env.force_sequences[0][0, 0] = float("nan")
    value = env.step()
    assert torch.isnan(value[0]) and torch.isfinite(value[1])
    handler.reset()
    env.force_sequences = [torch.tensor([[25., 25.], [50., 50.]])]*4
    assert torch.isnan(env.step()[0])
    assert not handler.audit(check_device_health=True)["device_health_finite"]


def test_downward_force_latches_fault_instead_of_clamping_it_into_valid_support(env):
    env.force_sequences = [torch.tensor([[-1., 25.], [50., 50.]])]*4
    handler = install(env)
    assert torch.isnan(env.step()[0])
    assert handler.audit(check_device_health=True)["negative_normal_force_samples"] == 4


def test_force_shape_sensor_identity_and_impulse_dt_are_strict(env):
    env.scene.sensors[support.SENSOR_NAMES[0]]._sim_physics_dt = .02
    with pytest.raises(ValueError, match="impulses"):
        install(env)


def test_sensor_replacement_after_install_is_rejected(env):
    install(env)
    env.scene.sensors[support.SENSOR_NAMES[0]] = sensor(support.WHEEL_NAMES[0], torch.zeros((2, 1, 1, 3)))
    with pytest.raises(ValueError, match="replaced"):
        env.step()


@pytest.mark.parametrize("marker", support.CONFLICT_MARKERS)
def test_existing_strict_owner_or_old_cache_is_rejected(env, marker):
    setattr(env.scene, marker, object())
    with pytest.raises(ValueError, match="Conflicting"):
        install(env)


def test_bad_native_clock_and_nonfinite_mass_rejected(env):
    env.masses[0, 0] = float("nan")
    with pytest.raises(ValueError, match="body masses"):
        install(env)


def test_zero_total_mass_rejected(env):
    env.masses.zero_()
    with pytest.raises(ValueError, match="positive total"):
        install(env)


def test_wrong_pd_gain_or_gravity_rejected(env):
    env.robot.actuators["legs"].damping.fill_(5.)
    with pytest.raises(ValueError, match="Kp500/Kd10"):
        install(env)


def test_production_installation_has_no_cpu_fallback(env):
    with pytest.raises(ValueError, match="CUDA"):
        support.install_ground_support_accumulator(env)


def test_close_restores_methods_and_returns_idempotent_audit(env):
    before_step, before_update = env.step, env.scene.update
    handler = install(env); env.step()
    assert handler.close()["closed"]
    assert env.step == before_step and env.scene.update == before_update
    assert not hasattr(env, support.ENV_MARKER)
    assert handler.close()["closed"]


def test_owner_replacement_is_not_silently_unwrapped(env):
    handler = install(env)
    env.scene.update = lambda dt: None
    with pytest.raises(ValueError, match="owner changed"):
        handler.close()


def test_source_forces_and_actions_are_not_modified(env):
    values = [item.clone() for item in env.force_sequences]
    action = torch.ones((2, 8)); copy = action.clone()
    install(env); env.step(action)
    assert torch.equal(action, copy)
    assert all(torch.equal(left, right) for left, right in zip(values, env.force_sequences))


def test_hot_path_has_no_explicit_device_to_host_copies():
    tree = ast.parse(Path(support.__file__).read_text())
    hot = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
           and node.name in ("_step", "_update", "reward", "reset", "_matrix")]
    methods = {node.func.attr for fn in hot for node in ast.walk(fn)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not methods.intersection({"cpu", "numpy", "tolist", "item"})


def test_fresh_sensor_configuration_is_single_wheel_ground_filtered(monkeypatch):
    module = SimpleNamespace(ContactSensorCfg=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", module)
    scene = SimpleNamespace()
    support.configure_ground_support_sensors(scene)
    for name, body in zip(support.SENSOR_NAMES, support.WHEEL_NAMES):
        cfg = getattr(scene, name)
        assert cfg.prim_path == "{ENV_REGEX_NS}/Robot/"+body
        assert cfg.filter_prim_paths_expr == [support.GROUND_COLLIDER]
        assert cfg.history_length == 0 and cfg.update_period == 0.
        assert not cfg.track_contact_points and not cfg.track_air_time and not cfg.track_pose
    with pytest.raises(ValueError, match="overwrite"):
        support.configure_ground_support_sensors(scene)


def test_reward_term_wraps_idempotent_raw_reward_and_reset_guard(env, monkeypatch):
    class ManagerTermBase:
        def __init__(self, cfg, env):
            pass
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(
        ManagerTermBase=ManagerTermBase, RewardTermCfg=lambda **kwargs: SimpleNamespace(**kwargs)))
    cfg = support.make_reward_term()
    assert cfg.weight == .4 and cfg.params == {"scale_fraction": .20}
    term = cfg.func(cfg, env)
    term.reset([0])  # Legal initial environment reset before explicit install.
    handler = install(env)
    env.step()
    assert cfg.func.__name__ == "GroundSupportReward"
    term.reset([0])
    assert handler.reset_calls == 1
    with pytest.raises(ValueError, match="weight"):
        support.make_reward_term(weight=1.)
    with pytest.raises(ValueError, match="scale_fraction"):
        support.make_reward_term(scale_fraction=.1)


def test_real_event_audit_accepts_exact_recipe_and_rejects_mass_dr(monkeypatch):
    # Test the original audit without the CPU fixture's event-audit replacement.
    import importlib.util
    spec = importlib.util.spec_from_file_location("isolated_support_recipe_audit", support.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def make_function(name):
        def function():
            pass
        function.__name__ = name
        return function
    funcs = {name: make_function(name) for name in (
        "randomize_rigid_body_material", "randomize_joint_default_pos", "randomize_rigid_body_com",
        "push_by_setting_velocity", "clear_legacy_joint_friction")}
    tracking = SimpleNamespace(mdp=SimpleNamespace(**funcs), clear_legacy_joint_friction=funcs["clear_legacy_joint_friction"])
    monkeypatch.setitem(sys.modules, "training.tron1_tracking", tracking)
    expected = {"physics_material": ("startup", funcs["randomize_rigid_body_material"]),
                "legacy_joint_friction": ("startup", funcs["clear_legacy_joint_friction"]),
                "add_joint_default_pos": ("startup", funcs["randomize_joint_default_pos"]),
                "base_com": ("startup", funcs["randomize_rigid_body_com"]),
                "push_robot": ("interval", funcs["push_by_setting_velocity"])}
    manager = SimpleNamespace(active_terms={"startup": list(expected)[:-1], "interval": ["push_robot"]},
                              get_term_cfg=lambda name: SimpleNamespace(mode=expected[name][0], func=expected[name][1]))
    fixture = SimpleNamespace(event_manager=manager)
    assert len(module._event_audit(fixture)) == 5
    manager.active_terms["startup"].append("mass_randomization")
    with pytest.raises(ValueError, match="Uninspected"):
        module._event_audit(fixture)

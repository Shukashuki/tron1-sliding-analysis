"""CPU fixtures: no Isaac, stage, native API, GPU or policy calls occur."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import copy
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import eval_ppo_contact_numerics as m

LEGACY = ROOT/"outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_friction_isaac.py"
BASE_REPORT = ROOT/"outputs/contact_damping_20261021_v2/kd10/report.json"


def fake_base_plan(argv):
    assert argv[:2] == ["--leg-kd", "10"]
    return {"leg_kd": 10., "inherited_argv": argv[2:], "audit": json.loads(BASE_REPORT.read_text())["contact_stability_diagnostic"]}


def make_plan(iterations, monkeypatch):
    monkeypatch.setattr(m.base, "prepare_plan", fake_base_plan)
    return m.prepare_plan(["--solver-velocity-iterations", str(iterations)])


def fixture_env(iterations=4):
    props = NS(solver_position_iteration_count=8, solver_velocity_iteration_count=4, enabled_self_collisions=True)
    cfg = NS(scene=NS(robot=NS(spawn=NS(articulation_props=props), actuators={"legs": NS(stiffness=500., damping=10.)})),
        sim=NS(dt=.005, physx=NS(min_position_iteration_count=1, max_position_iteration_count=255,
            min_velocity_iteration_count=0, max_velocity_iteration_count=255)), decimation=4)
    m.apply_config(cfg, {"contact_friction": .7}, {"solver_velocity_iterations": iterations})
    class View:
        count, max_shapes = 1, 3
        prim_paths = ["/World/envs/env_0/Robot"]
        link_paths = [["/World/envs/env_0/Robot/base_Link", "/World/envs/env_0/Robot/wheel_L_Link", "/World/envs/env_0/Robot/wheel_R_Link"]]
        def __init__(self):
            self.calls = {"contact": 0, "rest": 0, "material": 0}
            self.contact = np.full((1, 3), .02, dtype=np.float32)
            self.rest = np.full((1, 3), -.001, dtype=np.float32)
            self.material = np.tile(np.array([.7, .7, 0.], dtype=np.float32), (1, 3, 1))
        def get_contact_offsets(self): self.calls["contact"] += 1; return self.contact
        def get_rest_offsets(self): self.calls["rest"] += 1; return self.rest
        def get_material_properties(self): self.calls["material"] += 1; return self.material
    view = View()
    robot = NS(root_physx_view=view, actuators={"legs": NS(stiffness=np.full((1, 6), 500.), damping=np.full((1, 6), 10.))})
    env = NS(_sim_step_counter=0, sim=NS(current_time_step_index=0, current_time=0.), scene={"robot": robot})
    return cfg, env, view


def fake_usd(iterations):
    return lambda stage, paths: {"source": "CPU fixture only", "native_backend_readback": False,
        "roots": [{"native_prim_path": paths[0], "articulation_root_prim_path": paths[0]+"/base_Link",
            "position": 8, "velocity": iterations, "position_has_authored_value": True,
            "velocity_has_authored_value": True, "enabled_self_collisions_usd": True}]}


def readback(cfg, env, plan, *, reader=None):
    return m.runtime_readback(env, cfg, plan, stage=object(),
        source_inspector=lambda getter: {"path": "/CPU-fixture/api.py", "sha256": m.TENSOR_API_SHA256},
        usd_reader=reader or fake_usd(plan["solver_velocity_iterations"]))


@pytest.mark.parametrize("iterations", [4, 8])
def test_only_cfg_velocity_changes_and_honest_audit(iterations, monkeypatch):
    plan = make_plan(iterations, monkeypatch)
    cfg, env, view = fixture_env(iterations)
    profile = {"contact_friction": .7}; original = copy.deepcopy(profile)
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 8
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True
    audit = plan["audit"]
    assert audit["kind"] == m.KIND and audit["only_changed_control_field"] is None
    assert audit["native_solver_iteration_readback"]["available"] is False
    assert "deployment_leg_kd" not in audit
    runtime = readback(cfg, env, plan)
    assert view.calls == {"contact": 1, "rest": 1, "material": 1}
    assert profile == original
    assert runtime["native_shape_readback"]["rest_offsets_m"][0][0] < 0
    assert runtime["clock_before"] == runtime["clock_after"] == [0, 0, 0.]
    assert runtime["usd_authored_solver"]["native_backend_readback"] is False
    assert runtime["native_shape_readback"]["shape_columns_are_body_ids"] is False
    contract = {"leg_kp": 500., "leg_kd": 10.}; m.decorate_contract(contract, plan)
    assert contract["contact_numerics_diagnostic"] == audit
    assert "contact_stability_diagnostic" not in contract
    assert contract["solver_counts_are_native_readback"] is False
    with pytest.raises(ValueError, match="exactly once"):
        readback(cfg, env, plan)


@pytest.mark.parametrize("bad", ["position", "pd", "dt"])
def test_original_cfg_required(bad):
    cfg, _, _ = fixture_env()
    if bad == "position": cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 4
    if bad == "pd": cfg.scene.robot.actuators["legs"].damping = 5.
    if bad == "dt": cfg.sim.dt = .01
    with pytest.raises(ValueError): m.apply_config(cfg, {"contact_friction": .7}, {"solver_velocity_iterations": 8})


@pytest.mark.parametrize("bad", ["nan", "dtype", "shape", "friction", "rest_above_contact", "gains", "usd_native_claim", "usd_wrong_root_value", "step_advanced"])
def test_invalid_readback_rejected(bad, monkeypatch):
    plan = make_plan(8, monkeypatch); cfg, env, view = fixture_env(8)
    reader = None
    if bad == "nan": view.contact[0, 0] = np.nan
    if bad == "dtype": view.contact = view.contact.astype(float)
    if bad == "shape": view.rest = view.rest[:, :2]
    if bad == "friction": view.material[..., 0] = .9
    if bad == "rest_above_contact": view.rest[:] = .1
    if bad == "gains": env.scene["robot"].actuators["legs"].stiffness[:] = 250.
    if bad == "usd_native_claim":
        def reader(stage, paths):
            value = fake_usd(8)(stage, paths); value["native_backend_readback"] = True; return value
    if bad == "usd_wrong_root_value": reader = fake_usd(4)
    if bad == "step_advanced":
        def reader(stage, paths):
            env.sim.current_time_step_index += 1; return fake_usd(8)(stage, paths)
    with pytest.raises(ValueError): readback(cfg, env, plan, reader=reader)


@pytest.mark.parametrize("flag", ["--leg-kp=250", "--leg-kd", "--solver-position-iterations=16", "--contact-offset=.01"])
def test_combined_interventions_blocked(flag):
    with pytest.raises(ValueError, match="blocked"):
        m.prepare_plan(["--solver-velocity-iterations", "8", flag])


def test_exact_legacy_source_hooks_before_actor_load_and_no_added_step_calls():
    source = m.materialize_source(LEGACY.read_text(), LEGACY.parent.parent)
    ast.parse(source)
    assert source.count("_contact_readback(env, cfg, _contact_plan)") == 1
    assert source.index("_contact_readback(env, cfg, _contact_plan)") < source.index("runner.load(")
    assert '"contact_numerics_diagnostic"' in source and '"contact_stability_diagnostic"' not in source
    tree = ast.parse(Path(m.__file__).read_text())
    names = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "" for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not set(names).intersection({"step", "reset", "forward", "mj_step", "acquire_dynamic_control_interface"})


def test_tensor_source_and_provenance_mutations_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="source differs"):
        m.tensor_api_source(fixture_env)
    path = tmp_path/"adapter.py"; path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.load_base_adapter(path)

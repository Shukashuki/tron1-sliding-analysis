from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ppo_backend_400hz_contract as shared
import eval_ppo_backend_400hz_isaac as physx
import eval_ppo_backend_400hz_mujoco as native
from test_ppo_backend_400hz_contract import make_cohort

LEGACY = ROOT / "outputs/controller_evaluation_source_20261015_v2"


@pytest.mark.parametrize("module,name", [(physx, "eval_tracking_friction_isaac.py"), (native, "eval_tracking_mujoco.py")])
def test_exact_counted_real_immutable_sources_compile(module, name):
    path = LEGACY / "scripts" / name
    if not path.is_file():
        pytest.skip("Historical evaluator snapshot is not published in this checkout")
    before = shared.sha256(path)
    source = path.read_text()
    generated = module.materialize_source(source, LEGACY)
    ast.parse(generated)
    assert shared.sha256(path) == before == module.BASE_SHA256
    assert "_new_finish_report(report," in generated
    if module is physx:
        assert "runner = _new_runner(" in generated
        assert "from training.tron1_scripted_actor_runner" not in generated
        assert "_new_configure(cfg, evaluation_profile," in generated
    with pytest.raises(ValueError, match="source changed|content differs"):
        module.materialize_source(source + "\n", LEGACY)


def test_physx_only_400hz_clock_and_declared_material_recipe(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "physx_mjlab_recipe", NS(install_physx_mjlab_recipe=lambda cfg, **kw: calls.append(kw)))
    cfg = NS(sim=NS(dt=.005, render_interval=4), decimation=4,
        scene=NS(robot=NS(actuators={"legs": NS(stiffness=500., damping=10.)}),
            contact_forces=NS(update_period=.005), wheel_L_ground=NS(update_period=0.),
            wheel_R_ground=NS(update_period=0.), nonwheel_ground_base_Link=NS(update_period=0.)))
    profile = {"contact_friction": 1.}
    physx.configure(cfg, profile, {"mu": .03})
    assert (cfg.sim.dt, cfg.decimation, cfg.sim.render_interval) == (.0025, 8, 8)
    assert cfg.scene.contact_forces.update_period == cfg.scene.wheel_L_ground.update_period == .0025
    assert cfg.scene.wheel_R_ground.update_period == cfg.scene.nonwheel_ground_base_Link.update_period == .0025
    assert calls == [{"training": False, "fixed_mu": .03}]
    assert profile["contact_friction"] == .03
    assert cfg.scene.robot.actuators["legs"].stiffness == 500.
    assert cfg.scene.robot.actuators["legs"].damping == 10.


def model_fixture(mu=.7):
    return NS(npair=0, nbody=4, njnt=1, body_mass=np.array([0., 1., 1., 1.]),
        body_inertia=np.ones((4, 3)), body_ipos=np.zeros((4, 3)), body_iquat=np.tile([1., 0., 0., 0.], (4, 1)),
        dof_armature=np.zeros(6), dof_damping=np.zeros(6), dof_frictionloss=np.zeros(6),
        geom_margin=np.zeros(4), geom_gap=np.zeros(4), joint=lambda i: NS(name="floating"),
        geom_contype=np.ones(4, int), geom_conaffinity=np.ones(4, int),
        geom_bodyid=np.array([0, 1, 2, 3]), geom_priority=np.zeros(4, int),
        geom_friction=np.tile([mu, .005, .0001], (4, 1)), geom_condim=np.full(4, 3),
        geom_solref=np.tile([.02, 1.], (4, 1)), geom_solimp=np.tile([.9, .95, .001, .5, 2.], (4, 1)),
        opt=NS(timestep=.0025, iterations=100, ls_iterations=50, tolerance=1e-8, ls_tolerance=.01,
            solver=2, integrator=0, cone=0, impratio=1.),
        body=lambda i: NS(name=["world", "base_Link", "wheel_L_Link", "wheel_R_Link"][i]),
        geom=lambda i: NS(name=["ground", "base", "left", "right"][i]))


def test_native_pair_mapping_priorities_and_no_mu_squared():
    model = model_fixture(.03)
    friction = model.geom_friction.copy()
    result = native.audit_materials(model, {}, {"mu": .03})
    assert model.geom_priority.tolist() == [0, 0, 1, 1]
    assert result["wheel_geom_ids"] == [2, 3]
    assert result["pair_coefficient"] == .03
    assert np.array_equal(model.geom_friction, friction)
    assert result["explicit_contact_pair_count"] == 0
    assert result["native_mapping_verified"] is True
    assert result["native_plant_readback"]["readbacks"]["body_mass"]["values"] == [0., 1., 1., 1.]
    model.npair = 1
    with pytest.raises(ValueError, match="Explicit"):
        native.audit_materials(model, {}, {"mu": .03})


def test_native_rejects_material_or_native_clock_disagreement():
    model = model_fixture()
    model.geom_friction[0, 0] = 1.
    with pytest.raises(ValueError, match="friction differs"):
        native.audit_materials(model, {}, {"mu": .7})
    model = model_fixture()
    model.opt.timestep = .005
    with pytest.raises(ValueError, match="timestep"):
        native.audit_materials(model, {}, {"mu": .7})


def recording(backend):
    t = np.arange(2793) * .0025
    actions = np.linspace(-2., 2., 349 * 8).reshape(349, 8)
    if backend == "PhysX":
        repeated = np.concatenate([np.zeros((1, 8)), np.repeat(actions, 8, axis=0)])[:, None]
        return {"time_s": t, "physics_dt_s": np.array(.0025), "policy_dt_s": np.array(.02),
            "valid_mask": np.ones((2793, 1), bool), "simulation_step_counter": np.arange(17, 2810),
            "action_was_applied": np.r_[False, np.ones(2792, bool)][:, None],
            "action_observation": np.zeros((2793, 1, 51)), "actions_raw": repeated,
            "actions_clipped": np.clip(repeated, -1, 1), "root_position_m": np.zeros((2793, 1, 3)),
            "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (2793, 1, 1)),
            "joint_position_rad": np.zeros((2793, 1, 8)), "joint_velocity_rad_s": np.zeros((2793, 1, 8)),
            "applied_joint_torque_nm": np.zeros((2793, 1, 8))}
    clipped = np.clip(actions, -1, 1)
    return {"time_s": t, "physics_dt_s": np.array(.0025), "policy_dt_s": np.array(.02),
        "policy_time_s": np.arange(349) * .02, "action_observation": np.zeros((349, 51)),
        "actions_raw": actions, "actions_clipped": clipped,
        "previous_clipped_actions": np.concatenate([np.zeros((1, 8)), clipped[:-1]]),
        "root_pos": np.zeros((2793, 3)), "root_quat_wxyz": np.tile([1., 0., 0., 0.], (2793, 1)),
        "joint_pos": np.zeros((2793, 8)), "joint_vel": np.zeros((2793, 8)), "joint_torque_nm": np.zeros((2793, 8))}


@pytest.mark.parametrize("backend", ["PhysX", "native CPU MuJoCo"])
def test_native_recording_400hz_349_decisions_no_resampling(tmp_path, backend):
    path = tmp_path / "native.npz"
    np.savez_compressed(path, **recording(backend))
    audit = shared.verify_recording(path, backend)
    assert audit["native_states"] == 2793
    assert audit["policy_decisions"] == 349
    assert audit["recording_sha256"] == shared.sha256(path)


@pytest.mark.parametrize("defect", ["counter", "held_context", "nonfinite", "clock", "invalid"])
def test_physx_recording_rejects_timing_context_and_state_defects(tmp_path, defect):
    data = recording("PhysX")
    if defect == "counter": data["simulation_step_counter"][10] += 1
    if defect == "held_context": data["actions_raw"][2, 0, 0] += .1
    if defect == "nonfinite": data["joint_position_rad"][5, 0, 0] = np.nan
    if defect == "clock": data["time_s"][5] += .0025
    if defect == "invalid": data["valid_mask"][5, 0] = False
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        shared.verify_recording(path, "PhysX")


def test_cpu_previous_action_and_clipping_guard(tmp_path):
    data = recording("native CPU MuJoCo")
    data["previous_clipped_actions"][1, 0] += .1
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="Previous-action"):
        shared.verify_recording(path, "native CPU MuJoCo")


def test_physx_native_plant_getters_not_authored_usd_parameters():
    class View:
        def get_masses(self): return np.ones((1, 9))
        def get_inertias(self): return np.tile(np.eye(3).reshape(9), (1, 9, 1))
        def get_coms(self): return np.zeros((1, 9, 7))
    env = NS(scene={"robot": NS(root_physx_view=View(), body_names=[str(i) for i in range(9)], joint_names=[str(i) for i in range(8)])})
    result = physx.plant_readback(env)
    assert result["readbacks"]["get_masses"]["shape"] == [1, 9]
    assert result["readbacks"]["get_inertias"]["shape"] == [1, 9, 9]
    assert result["readbacks"]["get_masses"]["getter_source_sha256"] == shared.sha256(__file__)
    assert result["no_physics_or_state_mutation"] is True
    assert "get_dof_armatures" in result["unavailable"]


@pytest.mark.parametrize("backend", ["physx", "mjlab"])
def test_physx_plan_uses_new_actor_no_old_strict_adapter(tmp_path, monkeypatch, backend):
    if not (LEGACY / "scripts/eval_tracking_friction_isaac.py").is_file():
        pytest.skip("Historical evaluator snapshot unavailable")
    cohort = make_cohort(tmp_path, backend=backend)
    saved = shared.verify_cohort(cohort)
    motion, asset = tmp_path / "motion.npz", tmp_path / "model.usd"
    motion.write_bytes(b"reference")
    asset.write_bytes(b"asset")
    monkeypatch.setitem(shared.INPUT_PINS, "motion", shared.sha256(motion))
    monkeypatch.setitem(shared.INPUT_PINS, "asset", shared.sha256(asset))
    monkeypatch.setattr(physx, "verify_cohort", lambda *a, **k: saved)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    args = ["--cohort-dir", str(cohort), "--legacy-evaluator", str(LEGACY / "scripts/eval_tracking_friction_isaac.py"),
        "--motion-file", str(motion), "--asset-path", str(asset), "--output-dir", str(tmp_path / "result"),
        "--evaluation-contact-friction", ".7", "--physical-gpu-index", "3"]
    plan = physx.prepare_plan(args)
    argv = plan["inherited_argv"]
    assert str(saved["checkpoint"]) in argv
    assert "cuda:3" in argv
    assert "--physics-dt-s" not in argv
    assert "--deployment-leg-kp" not in argv
    assert plan["contract"]["physics_dt"] == .0025
    assert "activeGpu=3" in argv[-1]
    with pytest.raises(ValueError, match="Preserve"):
        physx.prepare_plan(args)


def published_native_fixture(tmp_path):
    cohort_dir = make_cohort(tmp_path, backend="mjlab")
    cohort = shared.verify_cohort(cohort_dir)
    output = tmp_path / "output"
    output.mkdir()
    np.savez_compressed(output / "rollout.npz", **recording("native CPU MuJoCo"))
    contract = shared.evaluation_contract(cohort, .7, "native CPU MuJoCo")
    audit = shared.provenance(cohort)
    audit.update(status="verified", validation_backend="native CPU MuJoCo",
        wrapper_sha256=shared.sha256(native.__file__), recording_sha256=shared.sha256(output / "rollout.npz"))
    report = {"backend_400hz_cohort_evaluation": audit, "status": "completed", "completed_full_reference": True,
        "hidden_resets": 0, "root_state_writes": 1, "start_reference_frame": 0,
        "domain_parameters_resampled_during_episode": False, "contract": contract,
        "policy_sha256": cohort["input_sha256"]["policy.pt"],
        "contact_material_audit": native.audit_materials(model_fixture(), {}, {"mu": .7})}
    (output / "report.json").write_text(json.dumps(report))
    return cohort_dir, output, report


def test_parent_verifier_accepts_complete_native_case(tmp_path):
    cohort, output, report = published_native_fixture(tmp_path)
    value = shared.validate_output(output, cohort, validation_backend="native CPU MuJoCo")
    assert value["status"] == "verified"
    assert value["native_states"] == 2793
    assert value["training_backend"] == "mjlab"
    assert value["smoke_only"] is False


@pytest.mark.parametrize("defect", ["actor", "wrapper", "material", "priority", "height", "clock", "reset"])
def test_parent_verifier_rejects_mixed_source_or_physics_claims(tmp_path, defect):
    cohort, output, report = published_native_fixture(tmp_path)
    if defect == "actor": report["policy_sha256"] = "0" * 64
    if defect == "wrapper": report["backend_400hz_cohort_evaluation"]["wrapper_sha256"] = "0" * 64
    if defect == "material": report["contact_material_audit"]["geom_friction"][0][0] = .49
    if defect == "priority": report["contact_material_audit"]["geom_priority"][2] = 0
    if defect == "height": report["contract"]["full_horizon_diagnostic"] = False
    if defect == "clock": report["contract"]["physics_dt"] = .005
    if defect == "reset": report["hidden_resets"] = 1
    (output / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        shared.validate_output(output, cohort, validation_backend="native CPU MuJoCo")

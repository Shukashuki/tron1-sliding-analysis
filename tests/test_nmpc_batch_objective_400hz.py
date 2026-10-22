"""Eight-substep mapping and exact scalar-reward parity, no GPU required."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from nmpc_batch_objective_400hz import (
    BatchHeightSmoothTrackingObjective400Hz, NativeBatchTrackingPredictor400Hz,
    PHYSICS_DT, SUBSTEPS,
)
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, HeightSmoothTrackingObjective


def profile(beta=2.):
    result = json.loads((ROOT/"config/car_drift_height_smooth_task.json").read_text())
    result["action_rate_weight"] = -.5
    result["reward_weights"]["sideslip"] = beta
    return result


def fixture(batch=3, horizon=4):
    rng = np.random.default_rng(471)
    shape = (batch, horizon)
    native = {}
    for name, width in (("body_pos_w", 3), ("body_quat_w", 4),
                        ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
        pre = rng.normal(scale=.3, size=(*shape, 9, width))
        post = rng.normal(scale=.4, size=(*shape, SUBSTEPS, 9, width))
        if width == 4:
            pre /= np.linalg.norm(pre, axis=-1, keepdims=True)
            post /= np.linalg.norm(post, axis=-1, keepdims=True)
        native["pre_"+name], native[name] = pre, post
    native["qpos"] = rng.normal(size=(*shape, SUBSTEPS, 15))
    native["body_net_forces_w"] = rng.normal(scale=3., size=(*shape, SUBSTEPS, 9, 3))
    native["contact_force_history_w"] = native["body_net_forces_w"][..., -3:, :, :].copy()
    native["interval_time_s"] = np.broadcast_to(
        np.arange(1, SUBSTEPS+1)*PHYSICS_DT, (*shape, SUBSTEPS)).copy()
    reference = {"body_names": BODY_NAMES,
                 **{name: native["pre_"+name].copy() for name in
                    ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}}
    actions = rng.uniform(-.9, .9, (*shape, 8))
    previous = rng.uniform(-.6, .6, (batch, 8))
    vz = rng.normal(size=batch)
    qids = np.array([7, 11, 8, 12, 9, 13, 10, 14])
    limits = np.array([[-1.3, 1.1]]*8)
    return native, reference, actions, previous, vz, qids, limits


def score(objective, parts, **kwargs):
    native, reference, actions, previous, vz, qids, limits = parts
    return objective.evaluate_native(native, reference, actions, previous, vz,
        joint_qpos_indices=qids, soft_joint_pos_limits=limits,
        soft_joint_limits_source="isaac_runtime", **kwargs)


@pytest.mark.parametrize("beta", [0., 2.])
@pytest.mark.parametrize("discount", [1., .99])
def test_every_stage_matches_original_scalar_reward_and_sequential_memory(beta, discount):
    parts = fixture()
    native, reference, actions, previous, vz, qids, limits = parts
    objective = BatchHeightSmoothTrackingObjective400Hz(profile(beta), discount_factor=discount)
    reset = np.array([True, False, True])
    actual = score(objective, parts, initial_reset_samples=reset)
    scalar = HeightSmoothTrackingObjective(profile(beta), soft_joint_limits=limits)
    for b in range(3):
        old_action, old_vz = previous[b], vz[b]
        for h in range(4):
            pre = {name: native["pre_"+name][b, h] for name in reference if name != "body_names"}
            target = {name: reference[name][b, h] for name in pre}
            post = {name: native[name][b, h, -1] for name in pre}
            post.update(joint_pos=native["qpos"][b, h, -1, qids],
                        body_net_forces_w=native["body_net_forces_w"][b, h, -1],
                        contact_force_history_w=native["contact_force_history_w"][b, h])
            expected = scalar.evaluate(post, scalar.hold_reference(target, pre), actions[b, h],
                old_action, old_vz, reset_sample=bool(reset[b] and h == 0))
            for group in ("raw_terms", "weighted_terms", "diagnostics"):
                for name, value in expected[group].items():
                    assert actual[group][name][b, h] == pytest.approx(value, abs=3e-13, rel=3e-13)
            assert actual["stage_cost"][b, h] == pytest.approx(expected["stage_cost"], abs=3e-13)
            old_action, old_vz = actions[b, h], expected["base_com_vz"]
    np.testing.assert_allclose(actual["candidate_costs"],
        (actual["stage_cost"]*discount**np.arange(4)).sum(axis=1), atol=3e-13)
    np.testing.assert_allclose(actual["undiscounted_candidate_costs"], actual["stage_cost"].sum(axis=1))
    assert actual["raw_terms"]["height_vertical_acceleration"][0, 0] == 0.
    assert actual["raw_terms"]["height_vertical_acceleration"][0, 1] > 0.
    assert objective.weights["action_rate_l2"] == -.5
    assert objective.contract()["stage_reward_formulas_changed"] is False


def test_eighth_not_fourth_substep_is_scored_without_input_mutation():
    parts = fixture()
    before = copy.deepcopy(parts)
    objective = BatchHeightSmoothTrackingObjective400Hz(profile())
    expected = score(objective, parts)
    for key in parts[0]:
        np.testing.assert_array_equal(parts[0][key], before[0][key])
    parts[0]["body_pos_w"][..., 3, :, :] += 100.
    parts[0]["body_lin_vel_w"][..., 3, :, :] += 100.
    actual = score(objective, parts)
    np.testing.assert_array_equal(actual["stage_cost"], expected["stage_cost"])
    parts[0]["body_pos_w"][..., 7, :, :2] += 1.
    assert np.any(score(objective, parts)["stage_cost"] != expected["stage_cost"])


@pytest.mark.parametrize("bad", ["four_substeps", "wrong_clock", "history", "nonfinite"])
def test_rejects_relabelled_or_inconsistent_native_records(bad):
    parts = fixture()
    if bad == "four_substeps":
        parts[0]["body_pos_w"] = parts[0]["body_pos_w"][..., :4, :, :]
    elif bad == "wrong_clock":
        parts[0]["interval_time_s"] *= 2.
    elif bad == "history":
        parts[0]["contact_force_history_w"][0, 0, 0, 0, 0] += 1.
    else:
        parts[0]["body_lin_vel_w"][0, 0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        score(BatchHeightSmoothTrackingObjective400Hz(profile()), parts)


def test_all_three_contact_samples_are_used_but_older_substeps_are_not():
    parts = fixture()
    parts[0]["body_net_forces_w"][:] = 0.
    parts[0]["body_net_forces_w"][..., 4, 0, 0] = 20.  # Outside three-sample history.
    parts[0]["contact_force_history_w"] = parts[0]["body_net_forces_w"][..., -3:, :, :].copy()
    objective = BatchHeightSmoothTrackingObjective400Hz(profile())
    np.testing.assert_array_equal(score(objective, parts)["raw_terms"]["undesired_contacts"], 0.)
    parts[0]["body_net_forces_w"][..., 5, 0, 0] = 1.0001
    parts[0]["contact_force_history_w"] = parts[0]["body_net_forces_w"][..., -3:, :, :].copy()
    np.testing.assert_array_equal(score(objective, parts)["raw_terms"]["undesired_contacts"], 1.)


@pytest.mark.parametrize("discount", [False, 0., -1., 1.01, np.inf, np.nan, "0.99"])
def test_bad_discount_rejected(discount):
    with pytest.raises(ValueError, match="discount_factor"):
        BatchHeightSmoothTrackingObjective400Hz(profile(), discount_factor=discount)


def test_other_reward_clock_rejected():
    with pytest.raises(ValueError, match="50 Hz"):
        BatchHeightSmoothTrackingObjective400Hz(profile(), dt=.04)


def test_adapter_uses_new_worker_batched_population_and_clamps_reference(monkeypatch):
    import nmpc_native_batch_400hz as native_module
    parts = fixture(batch=2, horizon=4)
    native, reference, actions, previous, vz, qids, limits = parts
    calls = []

    class Worker:
        metadata = {"physics_dt_s": .0025, "abi_version": 2}

        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))

        def rollout(self, qpos, qvel, plans, *, initial_history):
            calls.append(("rollout", qpos.copy(), qvel.copy(), plans.copy(), initial_history.copy()))
            return copy.deepcopy(native)

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(native_module, "NativeBatchTransition", Worker)
    packed = (np.zeros(15), np.zeros(14), np.zeros((3, 9, 3)), True)
    model = SimpleNamespace(joint=lambda name: SimpleNamespace(qposadr=[qids[JOINT_NAMES.index(name)]]))
    frames = [{name: value[0, i] for name, value in reference.items() if name != "body_names"}
              for i in range(4)]
    base = SimpleNamespace(layout=SimpleNamespace(unpack=lambda state: packed),
        objective=HeightSmoothTrackingObjective(profile()), mu_scenarios=(.7,),
        contract={"physics_dt": .0025, "decimation": 8}, full_references=frames,
        models={.7: model}, provider=SimpleNamespace(q0=np.zeros(8)),
        extractors={.7: SimpleNamespace(soft_limits=limits, soft_limits_source="isaac_runtime")})
    with NativeBatchTrackingPredictor400Hz(base, "unused.so", discount_factor=.99) as adapter:
        result = adapter.evaluate_batch("packed", actions, 2, mu=.7,
            previous_action=previous[0], previous_vz=vz[0])
        np.testing.assert_array_equal(result["reference_indices"], [2, 3, 3, 3])
        assert result["candidate_costs"].shape == (2,)
        assert adapter.metadata["horizon_discount_factor"] == .99
        assert len([call for call in calls if call[0] == "rollout"]) == 1
        population = next(call for call in calls if call[0] == "rollout")
        np.testing.assert_array_equal(population[3], actions)
        assert population[1].shape == (2, 15)
        assert population[4].shape == (2, 3, 9, 3)
        with pytest.raises(ValueError, match="declared"):
            adapter.rollout_batch("packed", actions, 0, mu=.1,
                previous_action=previous[0], previous_vz=vz[0])
    assert calls[-1][0] == "close"


def test_real_native_worker_matches_scalar_eight_substeps_and_every_reward_term(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    if mujoco.__version__ != "3.8.1" or sys.platform != "linux":
        pytest.skip("Optional native integration requires Linux MuJoCo3.8.1")
    raw = ROOT/"outputs/car_failure_sideslip_20261015/raw"
    model, contract_path = raw/"mujoco_model.xml", raw/"mu0p180/isaac/policy_contract.json"
    motion = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
    if not all(path.is_file() for path in (model, contract_path, motion)):
        pytest.skip("Licensed TRON1 model and locked reference are unavailable")
    from nmpc_native_batch_400hz import build_native_library
    from run_nmpc_tracking import MujocoTrackingPredictor, build_provider
    settings = json.loads((ROOT/"config/nmpc_height_smooth.json").read_text())
    settings["physics_dt_s"] = .0025
    contract = json.loads(contract_path.read_text())
    contract.update(physics_dt=.0025, decimation=8)
    contract.pop("physics_clock_refinement", None)
    provider = build_provider(model, motion, contract, settings, (.16, .7))
    predictor = MujocoTrackingPredictor(provider, model, profile(), (.16, .7))
    library = build_native_library(tmp_path/"native")
    rng = np.random.default_rng(62)
    plans = rng.uniform(-.015, .015, (2, 3, 8))
    previous = np.linspace(-.12, .12, 8)
    history = rng.normal(scale=.5, size=(3, 9, 3))
    initial = predictor.layout.pack(provider.targets[0, :15], provider.targets[0, 15:], history, True)
    initial_vz = predictor.vertical_velocity(initial)
    with NativeBatchTrackingPredictor400Hz(predictor, library, discount_factor=.99) as adapter:
        for mu in (.16, .7):
            actual = adapter.evaluate_batch(initial, plans, 0, mu=mu,
                previous_action=previous, previous_vz=initial_vz)
            for candidate, plan in enumerate(plans):
                state, old_action, old_vz, total = initial.copy(), previous.copy(), initial_vz, 0.
                for h, action in enumerate(plan):
                    before = state.copy()
                    state = predictor.transition(before, action, mu=mu)
                    expected = predictor.objective.evaluate(predictor.measure(state),
                        predictor.objective.hold_reference(predictor.reference_frame(h), predictor.measure(before)),
                        action, old_action, old_vz, reset_sample=predictor.layout.unpack(before)[-1])
                    for group in ("raw_terms", "weighted_terms", "diagnostics"):
                        for name, value in expected[group].items():
                            assert actual[group][name][candidate, h] == pytest.approx(value, abs=2e-9, rel=2e-9)
                    total += .99**h*expected["stage_cost"]
                    old_action, old_vz = action.copy(), predictor.vertical_velocity(state)
                assert actual["candidate_costs"][candidate] == pytest.approx(total, abs=2e-9, rel=2e-9)

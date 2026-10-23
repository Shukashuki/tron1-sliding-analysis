"""Arithmetic fixtures only: no simulator, contact correctness or success claim."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ground_contact_reward_v2 as reward


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_frozen_formula_and_broadcast(dtype):
    force = np.full((2, 3, 8, 2), 5., dtype=dtype)
    mg = np.array([[100.], [200.]], dtype=dtype)
    original = force.copy()
    result = reward.reward_numpy(force, mg, .8, np.array([.25, .5, 1.], dtype=dtype))
    quality = force.sum(axis=-1) / (.5 * mg[..., None])
    expected = quality.mean(axis=-1) * .8 * np.array([.25, .5, 1.])
    np.testing.assert_allclose(result["raw_reward"], expected, rtol=2e-7)
    assert result["raw_reward"].shape == (2, 3)
    assert result["raw_reward"].dtype == dtype
    assert result["per_step_support"].shape == (2, 3, 8)
    np.testing.assert_array_equal(force, original)


def test_at_least_one_support_allows_single_wheel_and_alternation_but_not_gaps():
    full = np.full((8, 2), 50.)
    assert reward.reward_numpy(full, 100., 1., 1.)["raw_reward"] == 1.
    empty = reward.reward_numpy(full * 0., 100., 1., 1.)
    assert empty["raw_reward"] == 0.
    single = np.tile([100., 0.], (8, 1))
    assert reward.reward_numpy(single, 100., 1., 1.)["raw_reward"] == 1.
    alternate = single.copy(); alternate[1::2] = [0., 100.]
    assert reward.reward_numpy(alternate, 100., 1., 1.)["raw_reward"] == 1.
    gap = full.copy(); gap[3, :] = 0.
    result = reward.reward_numpy(gap, 100., 1., 1.)
    assert result["support_mean"] == 7 / 8
    assert result["support_min"] == 0.
    assert result["support"] == .75 * 7 / 8
    # A single huge pulse cannot replace seven unsupported substeps.
    pulse = np.zeros((8, 2)); pulse[0] = 1.e6
    assert reward.reward_numpy(pulse, 100., 1., 1.)["raw_reward"] == 0.


def test_saturation_tracking_gate_and_impact_gate():
    low, saturated, excessive = [np.full((8, 2), x) for x in (5., 50., 600.)]
    assert reward.reward_numpy(low, 100., 1., 1.)["raw_reward"] == .2
    assert reward.reward_numpy(saturated, 100., .8, .25)["raw_reward"] == pytest.approx(.2)
    assert reward.reward_numpy(saturated, 100., 0., 1.)["raw_reward"] == 0.
    assert reward.reward_numpy(saturated, 100., 1., 0.)["raw_reward"] == 0.
    impact = reward.reward_numpy(excessive, 100., 1., 1.)
    np.testing.assert_array_equal(impact["impact_excess"], np.ones((8, 2)))
    assert impact["impact_gate"] == pytest.approx(np.exp(-1.))
    assert impact["raw_reward"] == pytest.approx(np.exp(-1.))


@pytest.mark.parametrize("shape", [(4, 2), (7, 2), (9, 2), (8, 3), (8,), (0, 8, 2)])
def test_reject_wrong_or_empty_native_window(shape):
    with pytest.raises(ValueError):
        reward.reward_numpy(np.zeros(shape), 100., 1., 1.)


@pytest.mark.parametrize("field,value", [("force", np.nan), ("force", np.inf), ("force", -.001),
    ("mg", 0.), ("mg", -1.), ("mg", np.nan), ("mg", np.inf),
    ("anchor", -.01), ("anchor", 1.01), ("anchor", np.nan),
    ("speed", -1.), ("speed", 2.), ("speed", np.inf)])
def test_reject_invalid_values(field, value):
    force, mg, anchor, speed = np.ones((8, 2)), 100., 1., 1.
    if field == "force": force[0, 0] = value
    if field == "mg": mg = value
    if field == "anchor": anchor = value
    if field == "speed": speed = value
    with pytest.raises(ValueError):
        reward.reward_numpy(force, mg, anchor, speed)


def test_tolerated_negative_is_zero_not_fake_support():
    force = np.full((8, 2), -1.e-4)
    result = reward.reward_numpy(force, 100., 1., 1.)
    assert result["raw_reward"] == 0.
    np.testing.assert_array_equal(result["per_step_support"], np.zeros(8))
    force[0, 0] = -1.0001e-4
    with pytest.raises(ValueError): reward.reward_numpy(force, 100., 1., 1.)


def test_batch_dimensions_do_not_broadcast_into_force_axes():
    force = np.ones((2, 3, 8, 2))
    with pytest.raises(ValueError): reward.reward_numpy(force, np.ones(2), 1., 1.)
    with pytest.raises(ValueError): reward.reward_numpy(force, 100., np.ones((8, 2)), 1.)


@pytest.mark.parametrize("field,value", [("support_load_fraction", .2), ("physics_substeps", 4),
    ("mean_weight", True), ("impact_multiple", np.nan), ("schema_version", True)])
def test_settings_reject_retuning(field, value):
    cfg = copy.deepcopy(reward.DEFAULT_SETTINGS); cfg[field] = value
    with pytest.raises(ValueError): reward.validate_settings(cfg)


def test_settings_json_contract_no_hidden_height_slip_streak():
    import json
    cfg = reward.validate_settings(json.loads(json.dumps(reward.DEFAULT_SETTINGS)))
    cfg["support_load_fraction"] = 1.
    assert reward.DEFAULT_SETTINGS["support_load_fraction"] == .5
    for mutate in (lambda x: x.pop("candidate"), lambda x: x.update(height=0.75)):
        cfg = reward.validate_settings(); mutate(cfg)
        with pytest.raises(ValueError): reward.validate_settings(cfg)
    meta = reward.formula_metadata()
    assert meta["height_or_zero_sideslip_target"] is False
    assert meta["cross_window_streak_memory"] is False
    assert meta["continuous_contact_or_task_success_certified"] is False
    assert meta["bilateral_support_required"] is False
    assert meta["settings"]["external_reward_weight"] == .4
    assert meta["settings"]["reward_dt_s"] == .02


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_optional_torch_numpy_parity_all_components(dtype_name):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    force = rng.uniform(0., 800., (2, 3, 8, 2)).astype(dtype_name)
    force[0, 0, 0, 0] = -5.e-5
    mg = np.array([[100.], [150.]], dtype=dtype_name)
    anchor = rng.uniform(0, 1, (2, 3)).astype(dtype_name)
    speed = rng.uniform(0, 1, (2, 3)).astype(dtype_name)
    expected = reward.reward_numpy(force, mg, anchor, speed)
    actual = reward.reward_torch(torch.as_tensor(force), torch.as_tensor(mg),
                                 torch.as_tensor(anchor), torch.as_tensor(speed))
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name].numpy(), expected[name],
            rtol=4e-6 if dtype_name == "float32" else 2e-14,
            atol=2e-7 if dtype_name == "float32" else 2e-15)
        assert actual[name].dtype == getattr(torch, dtype_name)


def test_optional_torch_hotpath_poison_invalid_window_without_touching_valid():
    torch = pytest.importorskip("torch")
    force = torch.full((4, 8, 2), 50., dtype=torch.float64)
    force[1, 0, 0] = -.001; force[2, 0, 0] = float("nan")
    anchor = torch.tensor([1., 1., 1., 2.], dtype=torch.float64)
    with pytest.raises(ValueError): reward.reward_torch(force, 100., anchor, 1.)
    result = reward.reward_torch(force, 100., anchor, 1., validate=False)
    for value in result.values():
        assert torch.isfinite(value[0]).all()
        assert torch.isnan(value[1:]).all()
    assert result["raw_reward"][0] == 1.


def test_optional_torch_keeps_gradient_graph():
    torch = pytest.importorskip("torch")
    force = torch.full((8, 2), 5., dtype=torch.float64, requires_grad=True)
    anchor = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    result = reward.reward_torch(force, 100., anchor, .5)
    result["raw_reward"].backward()
    assert torch.isfinite(force.grad).all() and force.grad.abs().sum() > 0
    assert anchor.grad == pytest.approx(.1)


def test_optional_torch_python_scalar_float64_precision():
    torch = pytest.importorskip("torch")
    force = np.full((8, 2), 50., dtype=np.float64)
    expected = reward.reward_numpy(force, 100., .8, .3)
    actual = reward.reward_torch(torch.as_tensor(force), 100., .8, .3)
    assert actual["raw_reward"].item() == expected["raw_reward"]

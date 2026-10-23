"""C arithmetic only: immutable B bonus plus fixed ungated dense deficit."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2c as reward


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_all_B_components_bitwise_preserved_except_raw(dtype):
    rng = np.random.default_rng(42)
    force = rng.uniform(0., 60., (2, 3, 8, 2)).astype(dtype); force[0, 0, 3] = 0.
    mg = np.array([[100.], [200.]], dtype=dtype)
    anchor = np.array([.2, .8, 1.], dtype=dtype); speed = np.array([[.5], [.9]], dtype=dtype)
    original = force.copy(); b = reward.base.reward_numpy(force, mg, anchor, speed)
    c = reward.reward_numpy(force, mg, anchor, speed)
    for name in b.keys()-{"raw_reward"}: np.testing.assert_array_equal(c[name], b[name])
    np.testing.assert_array_equal(c["gated_support_bonus"], b["raw_reward"])
    np.testing.assert_array_equal(c["support_deficit"], (1.-b["per_step_support"]).mean(-1))
    np.testing.assert_array_equal(c["raw_reward"], b["raw_reward"]-c["support_deficit"])
    assert c["raw_reward"].dtype == dtype and c["raw_reward"].shape == (2, 3)
    np.testing.assert_array_equal(force, original)


def test_single_and_alternating_support_no_bilateral_constraint():
    single = np.tile([100., 0.], (8, 1)); alternate = single.copy(); alternate[1::2] = [0., 100.]
    for force in (np.full((8, 2), 50.), single, alternate):
        value = reward.reward_numpy(force, 100., 1., 1.)
        assert value["raw_reward"] == 1. and value["support_deficit"] == 0.


def test_dense_deficit_is_ungated_and_preserves_full_B_bonus():
    empty = np.zeros((8, 2))
    for anchor, speed in ((0., 0.), (1., 1.), (0., 1.), (1., 0.)):
        value = reward.reward_numpy(empty, 100., anchor, speed)
        assert value["gated_support_bonus"] == 0. and value["raw_reward"] == -1.
    force = np.full((8, 2), 50.); force[3] = 0.
    value = reward.reward_numpy(force, 100., 1., 1.)
    assert value["support"] == 0. and value["support_deficit"] == 1/8 and value["raw_reward"] == -1/8
    low = reward.reward_numpy(np.full((8, 2), 5.), 100., 1., 1.)
    assert low["gated_support_bonus"] == .2 and low["raw_reward"] == pytest.approx(-.6)
    assert reward.reward_numpy(np.full((8, 2), 5.), 100., 0., 1.)["raw_reward"] == pytest.approx(-.8)


def test_impact_gate_unchanged_and_not_applied_to_deficit():
    force = np.full((8, 2), 600.); force[3] = 0.
    b = reward.base.reward_numpy(force, 100., .8, .3)
    c = reward.reward_numpy(force, 100., .8, .3)
    assert c["impact_gate"] == b["impact_gate"] < 1.
    assert c["support_deficit_penalty"] == 1/8 and c["raw_reward"] == -1/8
    force[3] = 600.
    b = reward.base.reward_numpy(force, 100., .8, .3); c = reward.reward_numpy(force, 100., .8, .3)
    assert c["raw_reward"] == b["raw_reward"] and c["support_deficit"] == 0.


def test_raw_bounds_and_tolerated_negative():
    rng = np.random.default_rng(123)
    force = rng.uniform(0., 1.e5, (1000, 8, 2)); force[:5] = 0.
    value = reward.reward_numpy(force, 200., rng.random(1000), rng.random(1000))
    assert (value["raw_reward"] >= -1.).all() and (value["raw_reward"] <= 1.).all()
    assert reward.reward_numpy(np.full((8, 2), -1.e-4), 100., 1., 1.)["raw_reward"] == -1.


@pytest.mark.parametrize("shape", ((4, 2), (7, 2), (9, 2), (8, 3), (8,), (0, 8, 2)))
def test_wrong_window_rejected(shape):
    with pytest.raises(ValueError): reward.reward_numpy(np.zeros(shape), 100., 1., 1.)


@pytest.mark.parametrize("field,value", (("force", np.nan), ("force", np.inf), ("force", -.001),
    ("mg", 0.), ("mg", -1.), ("mg", np.nan), ("mg", np.inf), ("anchor", -.01), ("anchor", 1.01),
    ("anchor", np.nan), ("speed", -1.), ("speed", 2.), ("speed", np.inf)))
def test_all_B_input_checks_retained(field, value):
    force, mg, anchor, speed = np.ones((8, 2)), 100., 1., 1.
    if field == "force": force[0, 0] = value
    if field == "mg": mg = value
    if field == "anchor": anchor = value
    if field == "speed": speed = value
    with pytest.raises(ValueError): reward.reward_numpy(force, mg, anchor, speed)


@pytest.mark.parametrize("field,value", (("support_deficit_weight", 0.), ("support_deficit_weight", True),
    ("support_deficit_weight", np.bool_(True)), ("candidate", "ground_contact_v2b"), ("mean_weight", .75),
    ("minimum_weight", .25), ("support_load_fraction", 1.), ("external_reward_weight", .8)))
def test_no_retuning_or_B_settings_allowed(field, value):
    cfg = reward.validate_settings(); cfg[field] = value
    with pytest.raises(ValueError): reward.validate_settings(cfg)


def test_metadata_frozen_A_B_and_no_hotpath_IO():
    meta = reward.formula_metadata()
    assert meta["raw_reward_bounds"] == [-1., 1.]
    assert meta["immutable_v2a_definition_sha256"] == reward.BASE_DEFINITION_SHA256
    assert meta["immutable_v2b_definition_sha256"] == reward.BASE_B_DEFINITION_SHA256
    assert not meta["support_deficit_tracking_gated"] and not meta["support_deficit_impact_gated"]
    with patch.object(Path, "read_bytes", side_effect=AssertionError("No reward filesystem I/O")):
        assert reward.reward_numpy(np.zeros((8, 2)), 100., 0., 0.)["raw_reward"] == -1.
    with patch.object(reward, "BASE_B_DEFINITION_SHA256", "0"*64):
        with pytest.raises(ValueError, match="V2b arithmetic"): reward.verify_base()


@pytest.mark.parametrize("dtype_name", ("float32", "float64"))
def test_optional_torch_all_component_parity(dtype_name):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(42); force = rng.uniform(0., 150., (2, 3, 8, 2)).astype(dtype_name)
    force[0, 0, 2] = 0.; mg = np.array([[100.], [200.]], dtype=dtype_name)
    expected = reward.reward_numpy(force, mg, .8, .3)
    actual = reward.reward_torch(torch.as_tensor(force), torch.as_tensor(mg), .8, .3)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name].numpy(), expected[name],
            rtol=4e-6 if dtype_name == "float32" else 2e-14, atol=2e-7 if dtype_name == "float32" else 2e-15)


def test_optional_torch_hotpath_NaN_poison_and_gradient():
    torch = pytest.importorskip("torch")
    force = torch.full((4, 8, 2), 5., dtype=torch.float64)
    force[1, 0, 0] = -.001; force[2, 0, 0] = float("nan")
    result = reward.reward_torch(force, 100., torch.tensor([1., 1., 1., 2.]), 1., validate=False)
    for value in result.values():
        assert torch.isfinite(value[0]).all() and torch.isnan(value[1:]).all()
    force = torch.full((8, 2), 5., dtype=torch.float64, requires_grad=True)
    anchor = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    value = reward.reward_torch(force, 100., anchor, .5); value["raw_reward"].backward()
    assert torch.isfinite(force.grad).all() and force.grad.abs().sum() > 0.
    assert anchor.grad == pytest.approx(.1)

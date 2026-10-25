"""Compare the batched law with the independently retained NumPy reference."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nonlinear_tire_contact import TireParameters, tangent_wrench
from nonlinear_tire_torch import tangent_wrench_torch


@pytest.mark.parametrize("nonlinear", [True, False])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 2e-10), (torch.float32, 5e-4)])
def test_batched_different_world_mu_matches_reference(nonlinear, dtype, tolerance):
    rng = np.random.default_rng(7)
    shape = (37, 2, 3)
    v, w, r, axle, normal = [rng.normal(size=shape) for _ in range(5)]
    load = rng.uniform(1, 900, (37, 2))
    mu = rng.uniform(.3, 1.2, (37, 1))
    args = [torch.tensor(a, dtype=dtype) for a in (v, w, r, axle, normal, load, mu)]
    result = tangent_wrench_torch(*args, nonlinear=nonlinear)
    law = "combined_slip_magic_formula" if nonlinear else "external_coulomb_regularized"
    for world in range(37):
        ref = tangent_wrench(*(a[world].numpy().astype(float) for a in args[:5]),
                             args[5][world].numpy().astype(float), law=law,
                             parameters=TireParameters(mu=float(args[6][world, 0])))
        for key in ref:
            np.testing.assert_allclose(result[key][world].numpy(), ref[key],
                                       atol=tolerance, rtol=tolerance, equal_nan=True)
    assert not result["invalid_measurement"].any()
    capacity = args[6] * args[5]
    assert torch.all(torch.linalg.vector_norm(result["force_w_n"], dim=-1) <= capacity + tolerance)
    assert torch.all(result["contact_power_w"] <= tolerance)


def inputs():
    v = torch.tensor([[[1., 0., 0.], [1., .5, 0.]]], dtype=torch.float64)
    w = torch.tensor([[[0., 10., 0.], [0., 20., 0.]]], dtype=torch.float64)
    r = torch.tensor([[[0., 0., -.1]]], dtype=torch.float64).expand_as(v).clone()
    axle = torch.tensor([[[0., 1., 0.]]], dtype=torch.float64).expand_as(v)
    normal = torch.tensor([[[0., 0., 1.]]], dtype=torch.float64).expand_as(v)
    return [v, w, r, axle, normal, torch.full((1, 2), 100., dtype=torch.float64),
            torch.full((1, 1), .7, dtype=torch.float64)]


def test_pure_rolling_driven_slip_and_zero_load_nan_point():
    args = inputs()
    result = tangent_wrench_torch(*args)
    assert torch.equal(result["force_w_n"][0, 0], torch.zeros(3, dtype=torch.float64))
    assert result["force_w_n"][0, 1, 0] > 0
    assert result["force_w_n"][0, 1, 1] < 0
    args[2][0, 1] = float("nan")
    args[5][0, 1] = 0
    result = tangent_wrench_torch(*args)
    assert not result["invalid_measurement"].any()
    assert torch.equal(result["force_w_n"], torch.zeros_like(args[0]))
    assert torch.isfinite(result["torque_about_COM_w_nm"]).all()


@pytest.mark.parametrize("bad", ["point", "axis", "mu", "load"])
def test_bad_loaded_measurement_is_flagged_without_solver_nan(bad):
    args = inputs()
    if bad == "point": args[2][0, 1] = float("nan")
    if bad == "axis": args[3] = args[4].clone()
    if bad == "mu": args[6][0, 0] = float("nan")
    if bad == "load": args[5][0, 1] = -1
    result = tangent_wrench_torch(*args)
    assert result["invalid_measurement"].any()
    assert torch.isfinite(result["force_w_n"]).all()
    assert torch.isfinite(result["torque_about_COM_w_nm"]).all()


def test_no_cross_world_or_cross_wheel_force_leakage():
    args = [a.expand((23,) + a.shape[1:]).clone() for a in inputs()]
    first = tangent_wrench_torch(*args)
    args[0][11, 1] *= -7
    second = tangent_wrench_torch(*args)
    mask = torch.ones((23, 2), dtype=torch.bool); mask[11, 1] = False
    assert torch.equal(first["force_w_n"][mask], second["force_w_n"][mask])

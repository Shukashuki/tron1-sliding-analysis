"""Pure finite-horizon algebra; no simulator, model, controller runtime, or asset."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tvlqr_core import generalized_affine_riccati


def problem(seed=1, T=4, n=3, m=2):
    rng = np.random.default_rng(seed)
    A = .6 * np.eye(n)[None] + rng.normal(scale=.15, size=(T, n, n))
    B = rng.normal(scale=.3, size=(T, n, m))
    d = rng.normal(scale=.3, size=(T, n))
    C = rng.normal(size=(T, n + m, n + m))
    cost = np.swapaxes(C, 1, 2) @ C + .4 * np.eye(n + m)[None]
    terminal = rng.normal(size=(n, n))
    return {"A": A, "B": B, "d": d, "Q": cost[:, :n, :n],
            "R": cost[:, n:, n:], "N": cost[:, :n, n:],
            "q": rng.normal(size=(T, n)), "r": rng.normal(size=(T, m)),
            "Qf": terminal.T @ terminal + .2 * np.eye(n),
            "qf": rng.normal(size=n)}


def dense_condensed(p, initial):
    """Independently assemble J(U)=.5 U'HU+g'U+c with all controls stacked."""
    T, n, m = p["B"].shape
    width = T * m
    H = np.zeros((width, width)); g = np.zeros(width); constant = 0.
    M = np.zeros((n, width)); c = initial.copy()
    for k in range(T):
        E = np.zeros((m, width)); E[:, k*m:(k+1)*m] = np.eye(m)
        Q, R, N, q, r = (p[name][k] for name in ("Q", "R", "N", "q", "r"))
        H += M.T @ Q @ M + E.T @ R @ E + M.T @ N @ E + E.T @ N.T @ M
        g += M.T @ (Q @ c + q) + E.T @ (N.T @ c + r)
        constant += .5 * c @ Q @ c + q @ c
        M, c = p["A"][k] @ M + p["B"][k] @ E, p["A"][k] @ c + p["d"][k]
    H += M.T @ p["Qf"] @ M
    g += M.T @ (p["Qf"] @ c + p["qf"])
    constant += .5 * c @ p["Qf"] @ c + p["qf"] @ c
    return H, g, constant


def closed_loop(p, initial, K, feed):
    z = initial.copy(); controls = []; cost = 0.
    for k in range(len(K)):
        v = K[k] @ z + feed[k]
        cost += (.5 * z @ p["Q"][k] @ z + .5 * v @ p["R"][k] @ v
                 + z @ p["N"][k] @ v + p["q"][k] @ z + p["r"][k] @ v)
        controls.append(v)
        z = p["A"][k] @ z + p["B"][k] @ v + p["d"][k]
    cost += .5 * z @ p["Qf"] @ z + p["qf"] @ z
    return np.asarray(controls).reshape(-1), float(cost)


def test_scalar_analytical_solution_has_signed_gain_cross_term_and_half_cost_convention():
    p = {"A": np.array([[[1.2]]]), "B": np.array([[[.7]]]), "d": np.array([[.3]]),
         "Q": np.array([[[2.]]]), "R": np.array([[[3.]]]), "N": np.array([[[.4]]]),
         "q": np.array([[.6]]), "r": np.array([[-.2]]), "Qf": np.array([[4.]]), "qf": np.array([-.5])}
    K, feed, P, gradient = generalized_affine_riccati(**p)
    H = 3. + .7**2 * 4.
    G = .4 + .7 * 4. * 1.2
    g = -.2 + .7 * (4. * .3 - .5)
    np.testing.assert_allclose(K, [[[-G/H]]], atol=1e-15)
    np.testing.assert_allclose(feed, [[-g/H]], atol=1e-15)
    np.testing.assert_allclose(P[:, 0, 0], [2. + 1.2**2 * 4. - G**2/H, 4.])
    np.testing.assert_allclose(gradient[:, 0], [.6 + 1.2*(4.*.3-.5) - G*g/H, -.5])


@pytest.mark.parametrize("seed,T,n,m", [(1, 1, 1, 1), (2, 2, 2, 1), (3, 4, 3, 2),
    (4, 5, 2, 3), (5, 7, 4, 2), (9, 3, 1, 2)])
def test_generalized_solution_equals_independent_dense_qp_and_closed_loop_value(seed, T, n, m):
    p = problem(seed, T, n, m)
    K, feed, P, gradient = generalized_affine_riccati(**p)
    assert K.shape == (T, m, n) and feed.shape == (T, m)
    assert P.shape == (T+1, n, n) and gradient.shape == (T+1, n)
    zero_cost = closed_loop(p, np.zeros(n), K, feed)[1]
    for initial in (np.zeros(n), np.arange(n) + .3, -np.arange(n) - .8):
        H, g, constant = dense_condensed(p, initial)
        np.linalg.cholesky(H)
        optimum = -np.linalg.solve(H, g)
        controls, actual_cost = closed_loop(p, initial, K, feed)
        np.testing.assert_allclose(controls, optimum, atol=2e-11, rtol=2e-11)
        expected_cost = .5 * optimum @ H @ optimum + g @ optimum + constant
        np.testing.assert_allclose(actual_cost, expected_cost, atol=3e-11, rtol=3e-11)
        expected_difference = .5 * initial @ P[0] @ initial + gradient[0] @ initial
        np.testing.assert_allclose(actual_cost-zero_cost, expected_difference, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(P, np.swapaxes(P, 1, 2), atol=0., rtol=0.)


def test_shifting_state_and_action_nominals_preserves_physical_optimal_policy():
    p = problem(12)
    T, n, m = p["B"].shape
    rng = np.random.default_rng(50)
    state_shift = rng.normal(size=(T+1, n)); action_shift = rng.normal(size=(T, m))
    shifted = copy.deepcopy(p)
    for k in range(T):
        shifted["d"][k] = p["A"][k] @ state_shift[k] + p["B"][k] @ action_shift[k] + p["d"][k] - state_shift[k+1]
        shifted["q"][k] = p["q"][k] + p["Q"][k] @ state_shift[k] + p["N"][k] @ action_shift[k]
        shifted["r"][k] = p["r"][k] + p["R"][k] @ action_shift[k] + p["N"][k].T @ state_shift[k]
    shifted["qf"] = p["qf"] + p["Qf"] @ state_shift[-1]
    K, feed, P, gradient = generalized_affine_riccati(**p)
    other_K, other_feed, other_P, other_gradient = generalized_affine_riccati(**shifted)
    np.testing.assert_allclose(other_K, K, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(other_P, P, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(other_feed, feed + np.einsum("tmn,tn->tm", K, state_shift[:-1]) - action_shift)
    np.testing.assert_allclose(other_gradient, gradient + np.einsum("tnm,tm->tn", P, state_shift))


def test_uniform_cost_scaling_changes_values_not_actions_and_zero_affine_terms_have_zero_feed():
    p = problem(21)
    K, feed, P, gradient = generalized_affine_riccati(**p)
    scaled = copy.deepcopy(p)
    for name in ("Q", "R", "N", "q", "r", "Qf", "qf"):
        scaled[name] *= 7.
    K2, feed2, P2, gradient2 = generalized_affine_riccati(**scaled)
    for actual, expected in ((K2, K), (feed2, feed), (P2, 7.*P), (gradient2, 7.*gradient)):
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    for name in ("d", "q", "r", "qf"):
        p[name][:] = 0.
    _, feed0, _, gradient0 = generalized_affine_riccati(**p)
    np.testing.assert_array_equal(feed0, np.zeros_like(feed0))
    np.testing.assert_array_equal(gradient0, np.zeros_like(gradient0))


def test_terminal_only_horizon_returns_exact_terminal_cost_and_empty_policy():
    p = problem(T=0)
    K, feed, P, gradient = generalized_affine_riccati(**p)
    assert K.shape == (0, 2, 3) and feed.shape == (0, 2)
    np.testing.assert_array_equal(P[0], p["Qf"])
    np.testing.assert_array_equal(gradient[0], p["qf"])


@pytest.mark.parametrize("field", ["A", "B", "d", "Q", "R", "N", "q", "r", "Qf", "qf"])
def test_invalid_dimensions_are_rejected_without_broadcasting(field):
    p = problem()
    p[field] = np.zeros(1)
    with pytest.raises(ValueError, match=field):
        generalized_affine_riccati(**p)


@pytest.mark.parametrize("field", ["A", "B", "d", "Q", "R", "N", "q", "r", "Qf", "qf"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coefficients_rejected(field, value):
    p = problem()
    p[field] = p[field].copy()
    p[field].flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        generalized_affine_riccati(**p)


@pytest.mark.parametrize("value", [np.array([1.+2.j]), np.array(["1"]), np.array([True])])
def test_nonreal_or_nonnumeric_coefficients_are_not_silently_coerced(value):
    p = problem(); p["d"] = value
    with pytest.raises(ValueError, match="real numeric"):
        generalized_affine_riccati(**p)


@pytest.mark.parametrize("field", ["Q", "R", "Qf"])
def test_asymmetric_cost_matrices_rejected(field):
    p = problem(); p[field] = p[field].copy()
    p[field][(..., 0, 1)] += .01
    with pytest.raises(ValueError, match="symmetric"):
        generalized_affine_riccati(**p)


@pytest.mark.parametrize("hessian", [0., -1.])
def test_nonpositive_control_hessian_fails_without_regularization(hessian):
    p = problem(T=1, n=1, m=1)
    p["Qf"][:] = 0.; p["R"][:] = hessian
    with pytest.raises(np.linalg.LinAlgError, match="stage 0.*positive definite"):
        generalized_affine_riccati(**p)


def test_stage_R_need_not_be_positive_when_full_control_hessian_is_positive():
    p = problem(T=1, n=1, m=1)
    p["R"][:] = -1.; p["B"][:] = 1.; p["Qf"][:] = 2.
    K, feed, P, gradient = generalized_affine_riccati(**p)
    assert all(np.isfinite(value).all() for value in (K, feed, P, gradient))


def test_inputs_unchanged_and_outputs_independent_of_input_storage():
    p = problem(); original = copy.deepcopy(p)
    result = generalized_affine_riccati(**p)
    for name in p:
        np.testing.assert_array_equal(p[name], original[name])
    for value in result:
        value[:] = -999.
    for name in p:
        np.testing.assert_array_equal(p[name], original[name])


def test_overflow_is_reported_instead_of_returning_nonfinite_gain_or_retrying():
    p = problem(T=1, n=1, m=1)
    p["B"][:] = 1.e308
    with pytest.raises(FloatingPointError):
        generalized_affine_riccati(**p)

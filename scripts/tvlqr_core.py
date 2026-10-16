"""Finite-horizon affine LQR with cross and linear cost terms; NumPy only."""
from __future__ import annotations

import numpy as np


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric coefficients")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite coefficients")
    return result


def _symmetric(value, name):
    if not np.allclose(value, np.swapaxes(value, -1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    # Only remove floating-point antisymmetry; never clip eigenvalues or
    # regularize a Hessian. Half first also avoids gratuitous sum overflow.
    return .5 * value + .5 * np.swapaxes(value, -1, -2)


def generalized_affine_riccati(A, B, d, Q, R, N, q, r, Qf, qf):
    """Solve an unconstrained, time-varying affine quadratic control problem.

    Dynamics: ``z[k+1] = A[k] @ z[k] + B[k] @ v[k] + d[k]``.
    Stage cost: ``.5*z'Q*z + .5*v'R*v + z'N*v + q'z + r'v``.
    Terminal cost: ``.5*z'Qf*z + qf'z``.

    Inputs have shapes A[T,n,n], B[T,n,m], d[T,n], Q[T,n,n],
    R[T,m,m], N[T,n,m], q[T,n], r[T,m], Qf[n,n], qf[n].
    No time-invariant broadcasting is performed; T=0 is terminal-only.

    Returns ``K, feed, P, p`` with shapes [T,m,n], [T,m],
    [T+1,n,n], [T+1,n]. The feedback is **v = K @ z + feed**:
    K already includes its minus sign. The value function is
    ``.5*z'P[k]*z + p[k]'z + constant[k]``; constants are not returned.

    Cost matrices must be symmetric within atol=1e-12, rtol=1e-10;
    accepted roundoff-level antisymmetry is replaced by the symmetric part.
    Every control Hessian must pass Cholesky. Failure raises, without
    regularization, eigenvalue clipping, retry, or a replacement controller.
    This algebraic solver does not impose actuator or state constraints.
    """
    arrays = {name: _real_array(value, name) for name, value in (
        ("A", A), ("B", B), ("d", d), ("Q", Q), ("R", R),
        ("N", N), ("q", q), ("r", r), ("Qf", Qf), ("qf", qf))}
    A, B = arrays["A"], arrays["B"]
    if A.ndim != 3 or A.shape[1] < 1 or A.shape[1] != A.shape[2]:
        raise ValueError("A must have shape (T,n,n) with n >= 1")
    T, n, _ = A.shape
    if B.ndim != 3 or B.shape[:2] != (T, n) or B.shape[2] < 1:
        raise ValueError("B must have shape (T,n,m) with m >= 1")
    m = B.shape[2]
    shapes = {"d": (T, n), "Q": (T, n, n), "R": (T, m, m),
              "N": (T, n, m), "q": (T, n), "r": (T, m),
              "Qf": (n, n), "qf": (n,)}
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arrays[name].shape}")
    d, N, q, r, qf = (arrays[name] for name in ("d", "N", "q", "r", "qf"))
    Q, R, Qf = (_symmetric(arrays[name], name) for name in ("Q", "R", "Qf"))
    K, feed = np.zeros((T, m, n)), np.zeros((T, m))
    P, p = np.zeros((T + 1, n, n)), np.zeros((T + 1, n))
    P[-1], p[-1] = Qf, qf
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for k in range(T - 1, -1, -1):
            next_P, next_p = P[k + 1], p[k + 1]
            H = R[k] + B[k].T @ next_P @ B[k]
            H = .5 * H + .5 * H.T
            G = N[k].T + B[k].T @ next_P @ A[k]
            affine = next_P @ d[k] + next_p
            g = r[k] + B[k].T @ affine
            if not all(np.isfinite(value).all() for value in (H, G, g, affine)):
                raise FloatingPointError(f"Nonfinite Riccati arithmetic at stage {k}")
            try:
                L = np.linalg.cholesky(H)
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    f"Control Hessian H at stage {k} is not positive definite") from error
            solution = -np.linalg.solve(L.T, np.linalg.solve(L, np.column_stack((G, g))))
            K[k], feed[k] = solution[:, :n], solution[:, n]
            value = Q[k] + A[k].T @ next_P @ A[k] + G.T @ K[k]
            P[k] = .5 * value + .5 * value.T
            p[k] = q[k] + A[k].T @ affine + G.T @ feed[k]
            if not all(np.isfinite(value).all() for value in (K[k], feed[k], P[k], p[k])):
                raise FloatingPointError(f"Nonfinite Riccati solution at stage {k}")
    return K, feed, P, p

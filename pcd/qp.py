"""The PCD quadratic program, solved exactly from the K x K Gram matrix of the gradients.

    d* = argmin_d 1/2 ||d - g_1||^2   s.t.   g_j^T d >= tau ||g_j||^2   for j = 2, ..., K      (Def. 4.1)

Everything the QP needs is the Gram matrix G[i, j] = <g_i, g_j>, so this module is plain NumPy and
independent of the deep-learning framework. By the KKT conditions (Thm. 4.2) the solution has the form

    d* = g_1 + sum_j mu_j g_j,   mu_j >= 0,

so it is returned as K weights w = (1, mu_2, ..., mu_K) with d* = sum_i w_i g_i. Indices are 0-based:
row/column 0 of the Gram matrix is the primary objective.
"""
from __future__ import annotations

from itertools import combinations
from typing import NamedTuple, Sequence, Tuple, Union

import numpy as np


class QPResult(NamedTuple):
    weights: np.ndarray       # shape (K,), weights[0] == 1; the direction is sum_i weights[i] * g_i
    active: Tuple[int, ...]   # indices j >= 1 whose constraint binds at the solution
    feasible: bool            # False when no direction satisfies every constraint (weights is then e_0)


def solve_qp(gram, tau: Union[float, Sequence[float]], tol: float = 1e-9) -> QPResult:
    """Solve the PCD QP by KKT enumeration over working sets (App. B.2, Prop. B.4).

    Working sets of linearly independent constraints are visited in order of increasing size; the first
    one whose multipliers are non-negative and whose direction satisfies every other constraint is the
    optimum. That is at most 2^(K-1) solves of size <= K-1, which is instant for the handful of
    objectives PCD is meant for. For K = 2 this is exactly the closed form of Cor. 4.6; for K = 3 the
    two-constraint system is the Cramer's-rule solution of Cor. B.5.

    Args:
        gram: (K, K) Gram matrix of the (normalized) gradients, primary first.
        tau:  tolerance in [0, 1]; a scalar as in the paper, or one value per secondary objective.
        tol:  numerical tolerance for the linear-independence and KKT tests.

    If the constraints are mutually incompatible (only possible for K >= 3, e.g. two exactly
    anti-parallel secondaries, Cor. B.3), the result has ``feasible=False`` and weights e_0: the
    constraints are dropped for that step, which is the deployment rule of Algorithm 1.
    """
    G = np.asarray(gram, dtype=np.float64)
    K = G.shape[0]
    if G.shape != (K, K):
        raise ValueError(f"gram must be square, got shape {G.shape}")
    weights = np.zeros(K)
    weights[0] = 1.0
    if K == 1:
        return QPResult(weights, (), True)

    taus = np.broadcast_to(np.asarray(tau, dtype=np.float64), (K - 1,))
    Gs = G[1:, 1:]                              # Gram matrix of the secondaries
    b = taus * np.diag(Gs) - G[1:, 0]           # b_j > 0  <=>  g_1 violates constraint j
    if np.all(b <= 0):                          # Cor. 4.3: g_1 is already feasible, d* = g_1
        return QPResult(weights, (), True)

    atol = tol * max(float(np.max(np.diag(G))), np.finfo(np.float64).tiny)
    candidates = [j for j in range(K - 1) if Gs[j, j] > 0]   # a zero secondary's constraint is 0 >= 0
    for size in range(1, len(candidates) + 1):
        for subset in combinations(candidates, size):
            A = list(subset)
            GA = Gs[np.ix_(A, A)]
            if not _independent(GA, tol):
                continue
            mu = np.linalg.solve(GA, b[A])
            if np.any(mu < -tol):
                continue
            mu = np.maximum(mu, 0.0)
            slack = Gs[:, A] @ mu - b           # g_j^T d_A - tau ||g_j||^2 for every secondary j
            if np.all(slack >= -atol):
                weights[1 + np.asarray(A)] = mu
                return QPResult(weights, tuple(1 + j for j in A), True)
    return QPResult(weights, (), False)         # F_tau is empty (Thm. B.2): fall back to g_1


def _independent(GA: np.ndarray, tol: float) -> bool:
    """Whether the gradients behind a Gram block are linearly independent (scale-free test)."""
    if GA.shape[0] == 1:
        return True
    d = np.sqrt(np.diag(GA))
    return bool(np.linalg.eigvalsh(GA / np.outer(d, d))[0] > tol)

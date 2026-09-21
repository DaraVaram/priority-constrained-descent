"""The QP solver against the paper's closed forms, the KKT conditions and an independent projection."""
import numpy as np
import pytest

from pcd import solve_qp


def direction(V, tau):
    """Solve the QP for the gradients in the rows of V (row 0 = primary)."""
    res = solve_qp(V @ V.T, tau)
    return res, res.weights @ V


def dykstra(V, tau, sweeps=20000):
    """Euclidean projection of V[0] onto {d : V[j] d >= tau ||V[j]||^2, j >= 1} (Dykstra's algorithm)."""
    x = V[0].copy()
    incr = np.zeros_like(V[1:])
    for _ in range(sweeps):
        x_start = x.copy()
        for j, a in enumerate(V[1:]):
            y = x + incr[j]
            viol = tau * a @ a - a @ y
            x = y + (viol / (a @ a)) * a if viol > 0 and a @ a > 0 else y
            incr[j] = y - x
        if np.max(np.abs(x - x_start)) < 1e-15:          # a full sweep that moves nothing is a fixed point
            break
    return x


def assert_kkt(V, tau, res, atol=1e-8):
    """Stationarity holds by construction (d = sum w_i g_i); check the rest of the KKT system."""
    d = res.weights @ V
    mu = res.weights[1:]
    slack = V[1:] @ d - tau * np.einsum("ij,ij->i", V[1:], V[1:])
    assert np.all(mu >= 0)
    assert np.all(slack >= -atol)                        # primal feasibility
    assert np.allclose(mu * slack, 0.0, atol=atol)       # complementary slackness


@pytest.mark.parametrize("seed", range(20))
def test_k2_matches_closed_form(seed):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((2, 30))
    tau = rng.uniform(0, 1)
    res, d = direction(V, tau)
    g1, g2 = V
    if g2 @ g1 >= tau * g2 @ g2:                          # Cor. 4.6, inactive
        expected = g1
    else:                                                 # Cor. 4.6, active
        expected = g1 + (tau - g2 @ g1 / (g2 @ g2)) * g2
    assert np.allclose(d, expected, atol=1e-12)
    assert_kkt(V, tau, res)


def test_k3_both_active_matches_cramer():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(50):
        g1 = rng.standard_normal(40)
        g2 = -0.5 * g1 + rng.standard_normal(40)          # both secondaries oppose the primary
        g3 = -0.5 * g1 + rng.standard_normal(40)
        V, tau = np.stack([g1, g2, g3]), 0.3
        res = solve_qp(V @ V.T, tau)
        if res.active != (1, 2):
            continue
        n2, n3, c23 = g2 @ g2, g3 @ g3, g2 @ g3             # Cor. B.5
        b2, b3 = tau * n2 - g2 @ g1, tau * n3 - g3 @ g1
        delta = n2 * n3 - c23 ** 2
        assert np.allclose(res.weights[1:], [(n3 * b2 - c23 * b3) / delta, (n2 * b3 - c23 * b2) / delta])
        checked += 1
    assert checked >= 10


@pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", range(10))
def test_general_k_is_the_projection(K, seed):
    rng = np.random.default_rng(100 * K + seed)
    V = rng.standard_normal((K, 50))
    V[1:] -= rng.uniform(0, 1.5, size=(K - 1, 1)) * V[0]   # make some secondaries conflict with the primary
    tau = rng.uniform(0.01, 1)
    res, d = direction(V, tau)
    assert res.feasible
    assert_kkt(V, tau, res)
    assert np.allclose(d, dykstra(V, tau), atol=1e-6)


def test_inactive_regime_returns_primary():
    V = np.array([[1.0, 0.0], [1.0, 0.2]])                # g2 already gets > tau of its progress from g1
    res = solve_qp(V @ V.T, 0.1)
    assert res.active == () and np.array_equal(res.weights, [1.0, 0.0])


def test_infeasible_falls_back_to_primary():
    g1 = np.array([1.0, 0.0, 0.0])
    g2 = np.array([0.0, 1.0, 0.0])
    V = np.stack([g1, g2, -2.0 * g2])                     # exactly anti-parallel secondaries (Cor. B.3)
    res = solve_qp(V @ V.T, 0.2)
    assert not res.feasible and np.array_equal(res.weights, [1.0, 0.0, 0.0])
    assert solve_qp(V @ V.T, 0.0).feasible                # at tau = 0, d = 0 is always feasible


def test_parallel_secondaries_resolved_by_the_more_demanding_one():
    g1 = np.array([1.0, 0.0])
    g2 = np.array([-1.0, 1.0])
    for c, expect in [(3.0, (2,)), (0.5, (1,))]:
        V = np.stack([g1, g2, c * g2])
        res = solve_qp(V @ V.T, 0.4)
        assert res.feasible and res.active == expect
        assert_kkt(V, 0.4, res)


def test_zero_secondary_is_ignored():
    rng = np.random.default_rng(1)
    g1, g2 = rng.standard_normal(10), rng.standard_normal(10)
    V2 = np.stack([g1, g2])
    V3 = np.stack([g1, g2, np.zeros(10)])
    r2, r3 = solve_qp(V2 @ V2.T, 0.5), solve_qp(V3 @ V3.T, 0.5)
    assert np.allclose(r3.weights[:2], r2.weights) and r3.weights[2] == 0


def test_per_objective_tolerance():
    rng = np.random.default_rng(2)
    V = rng.standard_normal((3, 20))
    V[1:] -= V[0]
    res = solve_qp(V @ V.T, [0.1, 0.6])
    d = res.weights @ V
    assert V[1] @ d >= 0.1 * V[1] @ V[1] - 1e-9
    assert V[2] @ d >= 0.6 * V[2] @ V[2] - 1e-9
    assert np.allclose(solve_qp(V @ V.T, [0.3, 0.3]).weights, solve_qp(V @ V.T, 0.3).weights)


def test_single_objective_is_plain_gradient():
    res = solve_qp(np.array([[4.0]]), 0.5)
    assert res.feasible and np.array_equal(res.weights, [1.0])

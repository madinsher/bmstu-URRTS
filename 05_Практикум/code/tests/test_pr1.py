"""Приёмочные тесты ПР1. Запуск: python -m unittest tests.test_pr1 -v"""
import unittest

import numpy as np

from urrts import graph, sim
from urrts import pr1_consensus as pr1


class Step1Laplacian(unittest.TestCase):
    def test_path_graph_laplacian(self):
        L = pr1.laplacian(graph.path_graph(3))
        np.testing.assert_allclose(L, [[1, -1, 0], [-1, 2, -1], [0, -1, 1]])

    def test_rows_sum_to_zero(self):
        rng = np.random.default_rng(1)
        _, A = graph.random_geometric_graph(12, 0.45, rng)
        np.testing.assert_allclose(pr1.laplacian(A).sum(axis=1), 0, atol=1e-12)

    def test_lambda2_known_values(self):
        # λ₂ полного графа Kₙ равна n; кольца — 2 − 2cos(2π/n)
        self.assertAlmostEqual(pr1.algebraic_connectivity(graph.complete_graph(6)), 6.0, places=9)
        self.assertAlmostEqual(pr1.algebraic_connectivity(graph.ring_graph(8)),
                               2 - 2 * np.cos(2 * np.pi / 8), places=9)

    def test_lambda2_zero_iff_disconnected(self):
        A = np.zeros((4, 4))
        A[0, 1] = A[1, 0] = A[2, 3] = A[3, 2] = 1
        self.assertAlmostEqual(pr1.algebraic_connectivity(A), 0.0, places=9)
        A[1, 2] = A[2, 1] = 1
        self.assertGreater(pr1.algebraic_connectivity(A), 1e-6)


class Step2Consensus(unittest.TestCase):
    def test_max_step(self):
        self.assertAlmostEqual(pr1.max_consensus_step(graph.star_graph(5)), 1 / 4)

    def test_converges_to_average(self):
        rng = np.random.default_rng(2)
        _, A = graph.random_geometric_graph(10, 0.5, rng)
        x0 = rng.normal(size=10)
        eps = 0.9 * pr1.max_consensus_step(A)
        xs = pr1.run_consensus(x0, A, eps, 400)
        np.testing.assert_allclose(xs[-1], x0.mean(), atol=1e-6)
        # среднее сохраняется на каждом шаге (граф неориентированный)
        np.testing.assert_allclose(xs.mean(axis=1), x0.mean(), atol=1e-10)

    def test_vector_state(self):
        A = graph.ring_graph(5)
        x0 = np.arange(10, dtype=float).reshape(5, 2)
        x = pr1.consensus_step(x0, A, 0.3)
        self.assertEqual(x.shape, (5, 2))
        np.testing.assert_allclose(x.mean(axis=0), x0.mean(axis=0))

    def test_rate_bounded_by_lambda2(self):
        A = graph.path_graph(6)
        eps = 0.4
        lam2 = pr1.algebraic_connectivity(A)
        x0 = np.array([1.0, -1, 2, 0.5, -2, 3])
        xs = pr1.run_consensus(x0, A, eps, 60)
        bound = (1 - eps * lam2) ** 60 * pr1.disagreement(x0)
        self.assertLessEqual(pr1.disagreement(xs[-1]), bound * (1 + 1e-9))


class Step3Formation(unittest.TestCase):
    def test_formation_assembles(self):
        rng = np.random.default_rng(3)
        n = 6
        ang = 2 * np.pi * np.arange(n) / n
        offsets = 0.3 * np.c_[np.cos(ang), np.sin(ang)]
        p0 = rng.uniform(0, 1, (n, 2))
        A = graph.ring_graph(n)
        hist, _ = sim.simulate(p0, lambda t, p, _A, s: pr1.formation_velocity(p, A, offsets, 1.0),
                               steps=600, dt=0.05)
        self.assertLess(pr1.formation_error(hist.p[-1], offsets), 1e-3)
        # центр формации = среднее начальных ξ (инвариант)
        c0 = (p0 - offsets).mean(axis=0)
        np.testing.assert_allclose((hist.p[-1] - offsets).mean(axis=0), c0, atol=1e-6)


class Step4Connectivity(unittest.TestCase):
    def test_weight(self):
        self.assertEqual(pr1.connectivity_weight(1.0, 1.0), 0.0)
        self.assertAlmostEqual(pr1.connectivity_weight(0.0, 1.0), 2.0)
        self.assertGreater(pr1.connectivity_weight(0.99, 1.0), 1e3)

    def test_rendezvous_keeps_initial_edges(self):
        # цепочка на грани дальности: обычный консенсус её рвёт, взвешенный — нет
        R = 1.0
        p0 = np.array([[0, 0], [0.95, 0], [1.9, 0], [2.85, 0], [2.85, 0.95]], dtype=float)
        A0 = graph.disk_graph(p0, R)
        self.assertTrue(graph.is_connected(A0))
        hist, _ = sim.simulate(p0, lambda t, p, _A, s: pr1.rendezvous_velocity(p, A0, R, 0.5),
                               steps=5000, dt=0.002, vmax=0.5)
        for P in hist.p:
            for i, j in zip(*np.nonzero(np.triu(A0, 1))):
                self.assertLess(np.linalg.norm(P[i] - P[j]), R)
        spread = np.linalg.norm(hist.p[-1] - hist.p[-1].mean(axis=0), axis=1).max()
        self.assertLess(spread, 0.1)


class Step5WMSR(unittest.TestCase):
    def test_single_malicious_agent_is_filtered(self):
        # полный граф из 7 вершин 3-робастен → при F = 1 нормальные агенты
        # сходятся к значению внутри диапазона своих начальных значений
        n, F = 7, 1
        A = graph.complete_graph(n)
        x = np.array([0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.0])
        bad = [6]
        normal = [i for i in range(n) if i not in bad]
        lo, hi = x[normal].min(), x[normal].max()
        for k in range(300):
            x[6] = 100.0 if k % 2 else -100.0  # злонамеренный агент «качает» сеть
            x = pr1.wmsr_step(x, A, F, 0.1, malicious=bad)
        self.assertLess(np.ptp(x[normal]), 1e-3)
        self.assertTrue(lo - 1e-9 <= x[normal].mean() <= hi + 1e-9)

    def test_plain_consensus_is_hijacked(self):
        A = graph.complete_graph(7)
        x = np.array([0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.0])
        for _ in range(300):
            x[6] = 100.0
            x = pr1.consensus_step(x, A, 0.1)
        self.assertGreater(x[:6].mean(), 50.0)


if __name__ == "__main__":
    unittest.main()

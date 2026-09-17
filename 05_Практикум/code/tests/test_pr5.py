"""Приёмочные тесты ПР5. Запуск: python -m unittest tests.test_pr5 -v"""
import unittest

import numpy as np

from urrts import graph
from urrts import pr5_coverage as pr5


class Step1Voronoi(unittest.TestCase):
    def test_two_robots_split_square(self):
        q, dA = pr5.grid_points(0, 1, 0, 1, 0.1)
        p = np.array([[0.25, 0.5], [0.75, 0.5]])
        lab = pr5.voronoi_labels(p, q)
        self.assertTrue(np.all(lab[q[:, 0] < 0.5] == 0))
        self.assertTrue(np.all(lab[q[:, 0] > 0.5] == 1))
        M, C = pr5.mass_centroids(p, q, np.ones(len(q)), dA, lab)
        np.testing.assert_allclose(M, [0.5, 0.5])
        np.testing.assert_allclose(C, [[0.25, 0.5], [0.75, 0.5]], atol=1e-9)

    def test_empty_cell(self):
        q = np.array([[0.0, 0.0], [0.1, 0.0]])
        p = np.array([[0.0, 0.0], [5.0, 5.0]])
        M, C = pr5.mass_centroids(p, q, np.ones(2), 1.0, pr5.voronoi_labels(p, q))
        self.assertEqual(M[1], 0.0)
        np.testing.assert_allclose(C[1], [5.0, 5.0])


class Step2Lloyd(unittest.TestCase):
    def test_cost_value(self):
        q = np.array([[1.0, 0.0], [0.0, 2.0]])
        self.assertAlmostEqual(pr5.coverage_cost([[0.0, 0.0]], q, np.array([1.0, 0.5]), 0.1), 0.1 * (1 + 0.5 * 4))

    def test_monotone_and_converges(self):
        rng = np.random.default_rng(1)
        q, dA = pr5.grid_points(0, 4, 0, 4, 0.1)
        phi = pr5.gaussian_density(q, [3.0, 1.0], 0.8)
        p = rng.uniform(0, 0.8, (6, 2))
        costs = [pr5.coverage_cost(p, q, phi, dA)]
        for _ in range(60):
            p = pr5.lloyd_step(p, q, phi, dA)
            costs.append(pr5.coverage_cost(p, q, phi, dA))
        self.assertTrue(np.all(np.diff(costs) <= 1e-9))
        self.assertLess(costs[-1], 0.3 * costs[0])
        # роботы сгустились у очага
        self.assertLess(np.linalg.norm(p.mean(axis=0) - [3.0, 1.0]), 1.0)


class Step3Limited(unittest.TestCase):
    def test_far_robot_waits(self):
        q, dA = pr5.grid_points(0, 1, 0, 1, 0.05)
        p = np.array([[5.0, 5.0], [0.5, 0.5]])
        p2 = pr5.limited_lloyd_step(p, q, np.ones(len(q)), dA, r_sense=0.3)
        np.testing.assert_allclose(p2[0], [5.0, 5.0])

    def test_limited_decreases_limited_cost(self):
        rng = np.random.default_rng(2)
        q, dA = pr5.grid_points(0, 5, 0, 5, 0.1)
        phi = np.ones(len(q))
        p = rng.uniform(2, 3, (8, 2))
        c0 = pr5.limited_cost(p, q, phi, dA, 0.8)
        for _ in range(80):
            p = pr5.limited_lloyd_step(p, q, phi, dA, 0.8)
        self.assertLess(pr5.limited_cost(p, q, phi, dA, 0.8), c0)
        d = np.linalg.norm(p[:, None] - p[None], axis=2) + 10 * np.eye(8)
        self.assertGreater(d.min(), 0.5)       # роботы разошлись, а не собрались в кучу


class Step4Estimation(unittest.TestCase):
    def test_converges_to_centralized(self):
        rng = np.random.default_rng(3)
        n = 8
        _, A = graph.random_geometric_graph(n, 0.55, rng)
        W = graph.metropolis_weights(A)
        x_true = np.array([2.0, -1.0])
        R = [np.diag(rng.uniform(0.01, 1.0, 2)) for _ in range(n)]
        z = [x_true + rng.multivariate_normal([0, 0], Ri) for Ri in R]
        est = pr5.information_consensus(z, R, W, 300)
        xc, _ = pr5.centralized_estimate(z, R)
        np.testing.assert_allclose(est[-1], np.tile(xc, (n, 1)), atol=1e-6)
        self.assertEqual(est.shape, (301, n, 2))
        np.testing.assert_allclose(est[0], np.array(z))


class Step5CI(unittest.TestCase):
    def test_ci_consistent(self):
        A = np.diag([1.0, 4.0])
        B = np.diag([4.0, 1.0])
        x, P, w = pr5.covariance_intersection([0, 0], A, [1, 1], B)
        self.assertAlmostEqual(w, 0.5)
        # P не меньше, чем «наивное» слияние как независимых: CI не переоценивает точность
        naive = np.linalg.inv(np.linalg.inv(A) + np.linalg.inv(B))
        self.assertTrue(np.all(np.linalg.eigvalsh(P - naive) >= -1e-12))
        np.testing.assert_allclose(x, [0.2, 0.8])      # по x точнее первая оценка, по y — вторая

    def test_ci_picks_better(self):
        x, P, w = pr5.covariance_intersection([0, 0], np.eye(2) * 0.1, [5, 5], np.eye(2) * 10)
        self.assertEqual(w, 1.0)
        np.testing.assert_allclose(x, [0, 0])


if __name__ == "__main__":
    unittest.main()

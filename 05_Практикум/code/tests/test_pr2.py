"""Приёмочные тесты ПР2. Запуск: python -m unittest tests.test_pr2 -v"""
import unittest

import numpy as np

from urrts import pr2_swarm as pr2


class Step1Boids(unittest.TestCase):
    def test_isolated_robot(self):
        p = np.array([[0, 0], [5, 5]], dtype=float)
        v = np.array([[1, 0], [0, 1]], dtype=float)
        np.testing.assert_allclose(pr2.boids_acceleration(p, v, 0, 0.2, 1.0), [0, 0])

    def test_components(self):
        p = np.array([[0, 0], [0.1, 0]], dtype=float)
        v = np.array([[0, 0], [1, 0]], dtype=float)
        a_sep = pr2.boids_acceleration(p, v, 0, 0.5, 1.0, w_sep=1, w_ali=0, w_coh=0)
        np.testing.assert_allclose(a_sep, [-10.0, 0.0])          # (−0.1)/0.01
        a_ali = pr2.boids_acceleration(p, v, 0, 0.5, 1.0, w_sep=0, w_ali=1, w_coh=0)
        np.testing.assert_allclose(a_ali, [1.0, 0.0])
        a_coh = pr2.boids_acceleration(p, v, 0, 0.5, 1.0, w_sep=0, w_ali=0, w_coh=1)
        np.testing.assert_allclose(a_coh, [0.1, 0.0])

    def test_flock_aligns_without_collisions(self):
        rng = np.random.default_rng(4)
        n = 20
        p = rng.uniform(0, 1.0, (n, 2))
        v = rng.normal(0, 0.1, (n, 2))
        for _ in range(800):
            p, v = pr2.boids_step(p, v, 0.05, 0.2, 0.3, r_sep=0.15, r_view=0.6)
        self.assertGreater(pr2.polarization(v), 0.9)
        d = np.linalg.norm(p[:, None] - p[None], axis=2) + np.eye(n)
        self.assertGreater(d.min(), 0.1)


class Step2Vicsek(unittest.TestCase):
    def test_polarization(self):
        self.assertAlmostEqual(pr2.polarization([[1, 0], [2, 0]]), 1.0)
        self.assertAlmostEqual(pr2.polarization([[1, 0], [-1, 0]]), 0.0)
        self.assertEqual(pr2.polarization([[0, 0], [0, 0]]), 0.0)

    def test_torus_neighbour(self):
        rng = np.random.default_rng(0)
        p = np.array([[0.05, 0.5], [0.95, 0.5]])     # соседи через границу тора
        th = np.array([0.0, np.pi / 2])
        _, th2 = pr2.vicsek_step(p, th, 0.0, 0.2, 0.0, 1.0, rng)
        np.testing.assert_allclose(th2, [np.pi / 4, np.pi / 4], atol=1e-12)

    def test_phase_transition(self):
        rng = np.random.default_rng(5)
        low = pr2.vicsek_order(100, 5.0, 1.0, 0.3, 300, rng)
        high = pr2.vicsek_order(100, 5.0, 1.0, 6.0, 300, rng)
        self.assertGreater(low, 0.8)
        self.assertLess(high, 0.3)


class Step3PSO(unittest.TestCase):
    def test_velocity_formula(self):
        rng = np.random.default_rng(7)
        x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        v = np.array([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
        pbest = np.array([[0.5, 0.0], [1.0, 1.0], [0.0, 1.0]])
        pbest_val = np.array([1.0, 3.0, 2.0])
        neigh = pr2.ring_neighborhood(3, 1)
        got = pr2.pso_velocity(x, v, pbest, pbest_val, neigh, rng, w=0.5, c1=1.0, c2=1.0)
        r = np.random.default_rng(7)
        r1, r2 = r.random((3, 2)), r.random((3, 2))
        lbest = np.array([[1.0, 1.0]] * 3)          # у всех в окрестности лучший — робот 1
        exp = 0.5 * v + r1 * (pbest - x) + r2 * (lbest - x)
        np.testing.assert_allclose(got, exp)

    def test_vmax(self):
        rng = np.random.default_rng(0)
        x = np.zeros((4, 2))
        pbest = np.full((4, 2), 10.0)
        v = pr2.pso_velocity(x, np.zeros((4, 2)), pbest, np.ones(4), pr2.ring_neighborhood(4), rng, vmax=0.2)
        self.assertLessEqual(np.linalg.norm(v, axis=1).max(), 0.2 + 1e-12)

    def test_finds_maximum(self):
        rng = np.random.default_rng(8)
        f = pr2.gaussian_source([0.7, -0.4], sigma=0.5)
        best, val, hist = pr2.pso_optimize(f, rng.uniform(-2, 2, (12, 2)), 150, rng)
        self.assertLess(np.linalg.norm(best - [0.7, -0.4]), 0.02)
        self.assertTrue(np.all(np.diff(hist) >= 0))


class Step4Robots(unittest.TestCase):
    def test_repulsion(self):
        x = np.array([[0.0, 0.0], [0.05, 0.0], [5.0, 5.0]])
        r = pr2.repulsion(x, 0.1)
        np.testing.assert_allclose(r[0], [-0.05, 0.0])
        np.testing.assert_allclose(r[1], [0.05, 0.0])
        np.testing.assert_allclose(r[2], [0.0, 0.0])

    def test_robot_search_reaches_source(self):
        rng = np.random.default_rng(9)
        src = np.array([2.0, 1.5])
        f = pr2.gaussian_source(src, sigma=0.8)
        x0 = rng.uniform(0, 0.5, (8, 2))
        traj, best = pr2.robot_pso_search(f, x0, 400, rng, vmax=0.03, d_min=0.1, noise_std=0.005)
        steps = np.linalg.norm(np.diff(traj, axis=0), axis=2)
        self.assertLessEqual(steps.max(), 0.03 + 1e-9)
        self.assertLess(np.linalg.norm(traj[-1].mean(axis=0) - src), 0.3)


if __name__ == "__main__":
    unittest.main()

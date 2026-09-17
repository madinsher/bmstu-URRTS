"""Приёмочные тесты ПР6. Запуск: python -m unittest tests.test_pr6 -v"""
import unittest

import numpy as np

from urrts import pr6_safety as pr6

DT = 0.02


def run_group(p, goals, steps, d_safe=0.2, unstuck=False, gamma=2.0, vmax=0.3):
    p = np.array(p, dtype=float)
    u_prev = np.zeros_like(p) + 1.0     # «уже движется»: правило не срабатывает на первом шаге
    hold = np.zeros(len(p), dtype=int)
    dmin = np.inf
    for _ in range(steps):
        if unstuck:
            u_nom, hold = pr6.unstuck_nominal(p, goals, u_prev, hold, 1.0, vmax)
            u = np.zeros_like(p)
            poly = pr6.speed_polygon(vmax)
            for i in range(len(p)):
                cons = list(poly) + [pr6.pair_constraint(p[i], p[j], d_safe, gamma)
                                     for j in range(len(p)) if j != i and np.linalg.norm(p[i] - p[j]) < 1.0]
                u[i] = pr6.safe_velocity(u_nom[i], cons)
        else:
            u, _ = pr6.cbf_controller(p, goals, d_safe, 1.0, gamma=gamma, vmax=vmax)
        p = p + DT * u
        u_prev = u
        dmin = min(dmin, pr6.min_pair_distance(p))
    return p, dmin


class Step1Constraint(unittest.TestCase):
    def test_values(self):
        a, b = pr6.pair_constraint([1.0, 0.0], [0.0, 0.0], 0.5, 2.0)
        np.testing.assert_allclose(a, [2.0, 0.0])
        self.assertAlmostEqual(b, -0.5 * 2.0 * (1.0 - 0.25))


class Step2Filter(unittest.TestCase):
    def test_inactive(self):
        u = pr6.safe_velocity([0.1, 0.0], [(np.array([1.0, 0.0]), -1.0)])
        np.testing.assert_allclose(u, [0.1, 0.0])

    def test_single_active(self):
        u = pr6.safe_velocity([-1.0, 1.0], [(np.array([1.0, 0.0]), 0.0)])
        np.testing.assert_allclose(u, [0.0, 1.0])

    def test_corner(self):
        cons = [(np.array([1.0, 0.0]), 0.5), (np.array([0.0, 1.0]), 0.5)]
        np.testing.assert_allclose(pr6.safe_velocity([0.0, 0.0], cons), [0.5, 0.5])

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            cons = [(rng.normal(size=2), rng.uniform(-1, 0)) for _ in range(4)] + pr6.speed_polygon(1.0, 8)
            u_nom = rng.normal(size=2) * 2
            u = pr6.safe_velocity(u_nom, cons)
            g = np.linspace(-1.2, 1.2, 241)
            G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
            ok = np.all([G @ a >= b for a, b in cons], axis=0)
            best = G[ok][np.argmin(np.sum((G[ok] - u_nom) ** 2, axis=1))]
            self.assertLessEqual(np.sum((u - u_nom) ** 2), np.sum((best - u_nom) ** 2) + 1e-9)
            self.assertTrue(all(a @ u >= b - 1e-7 for a, b in cons))


class Step3Group(unittest.TestCase):
    def test_crossing_is_safe(self):
        p0 = np.array([[0.0, 0.0], [1.0, 0.05]])
        goals = np.array([[1.0, 0.0], [0.0, 0.05]])
        p, dmin = run_group(p0, goals, 1500)
        self.assertGreaterEqual(dmin, 0.2 - 1e-3)
        self.assertLess(np.linalg.norm(p - goals, axis=1).max(), 0.05)

    def test_nominal_would_collide(self):
        p = np.array([[0.0, 0.0], [1.0, 0.0]])
        goals = np.array([[1.0, 0.0], [0.0, 0.0]])
        dmin = np.inf
        for _ in range(600):
            p = p + DT * pr6.go_to_goal(p, goals, 1.0, 0.3)
            dmin = min(dmin, pr6.min_pair_distance(p))
        self.assertLess(dmin, 0.05)


class Step4Deadlock(unittest.TestCase):
    def test_hysteresis(self):
        p = np.array([[0.0, 0.0]])
        goals = np.array([[1.0, 0.0]])
        u, hold = pr6.unstuck_nominal(p, goals, np.zeros((1, 2)), np.zeros(1, dtype=int), 1.0, 0.3, hold_steps=3)
        self.assertEqual(hold[0], 2)
        np.testing.assert_allclose(u[0], pr6.rotate(np.array([0.3, 0.0]), -np.pi / 4))
        u, hold = pr6.unstuck_nominal(p, goals, np.array([[0.3, 0.0]]), hold, 1.0, 0.3, hold_steps=3)
        self.assertEqual(hold[0], 1)                     # продвигается, но поворот ещё держится
        np.testing.assert_allclose(u[0], pr6.rotate(np.array([0.3, 0.0]), -np.pi / 4))

    def test_antipodal_swap(self):
        p0, goals = pr6.antipodal(8, 1.0)
        p_plain, d_plain = run_group(p0, goals, 2500)
        p, dmin = run_group(p0, goals, 2500, unstuck=True)
        self.assertGreaterEqual(dmin, 0.2 - 2e-3)
        self.assertGreater(np.linalg.norm(p_plain - goals, axis=1).max(), 0.5)   # без правила — тупик
        self.assertLess(np.linalg.norm(p - goals, axis=1).max(), 0.1)


class Step5Stale(unittest.TestCase):
    def test_margin_and_worst_case(self):
        a, b = pr6.stale_constraint([1.0, 0.0], [0.0, 0.0], 1.0, 0.1, 0.5, 2.0)
        np.testing.assert_allclose(a, [2.0, 0.0])
        self.assertAlmostEqual(b, -2.0 * (1.0 - 0.36) + 2.0 * 0.1)

    def test_evades_uncooperative_neighbour(self):
        # сосед j потерял связь и едет прямо на i; i знает его позицию с опозданием 0,5 с
        pi = np.array([0.0, 0.0])
        pj = np.array([1.2, 0.0])
        vj = np.array([-0.1, 0.0])
        hist_j = [pj.copy()]
        d_safe, dmin = 0.3, np.inf
        for k in range(1500):
            lag = 25                                  # 25 шагов = 0,5 с
            known = hist_j[max(0, len(hist_j) - 1 - lag)]
            age = min(lag, len(hist_j) - 1) * DT
            cons = pr6.speed_polygon(0.3) + [pr6.stale_constraint(pi, known, age, 0.1, d_safe, 2.0)]
            ui = pr6.safe_velocity(np.zeros(2), cons)
            pi = pi + DT * ui
            pj = pj + DT * vj
            hist_j.append(pj.copy())
            dmin = min(dmin, np.linalg.norm(pi - pj))
        self.assertGreaterEqual(dmin, d_safe - 1e-3)


if __name__ == "__main__":
    unittest.main()

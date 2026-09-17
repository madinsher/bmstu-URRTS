"""Тесты общих модулей (должны проходить сразу, до выполнения шагов)."""
import os
import tempfile
import unittest

import numpy as np

from urrts import graph, sim


class Graph(unittest.TestCase):
    def test_disk_graph_and_failures(self):
        p = np.array([[0, 0], [0.5, 0], [2, 0]], dtype=float)
        A = graph.disk_graph(p, 1.0)
        np.testing.assert_array_equal(A, [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        A2 = graph.disk_graph(p, 1.0, alive=[True, False, True])
        self.assertEqual(A2.sum(), 0)
        self.assertEqual(graph.components(A), [[0, 1], [2]])
        self.assertFalse(graph.is_connected(A))

    def test_metropolis_doubly_stochastic(self):
        rng = np.random.default_rng(0)
        _, A = graph.random_geometric_graph(9, 0.5, rng)
        W = graph.metropolis_weights(A)
        np.testing.assert_allclose(W.sum(axis=0), 1)
        np.testing.assert_allclose(W.sum(axis=1), 1)
        self.assertTrue(np.all(W >= 0))


class Sim(unittest.TestCase):
    def test_saturation_failure_and_csv(self):
        p0 = np.zeros((3, 2))
        ctrl = lambda t, p, A, s: np.tile([1.0, 0.0], (3, 1))
        hist, alive = sim.simulate(p0, ctrl, 10, dt=0.1, vmax=0.5, failures={5: [2]})
        np.testing.assert_allclose(hist.p[-1][:, 0], [0.5, 0.5, 0.25])
        self.assertFalse(alive[2])
        path = os.path.join(tempfile.mkdtemp(), "log.csv")
        hist.to_csv(path)
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(len(fh.readlines()), 1 + 3 * 11)

    def test_comm_drop(self):
        comm = sim.CommModel(radius=10, drop_prob=1.0)
        self.assertEqual(comm.adjacency(np.zeros((4, 2))).sum(), 0)

    def test_unicycle(self):
        v, w = sim.unicycle_from_velocity(0.0, [0.1, 0.0])
        self.assertAlmostEqual(v, 0.1)
        self.assertAlmostEqual(w, 0.0)
        left, right = sim.wheel_speeds(0.0, 100.0)
        self.assertAlmostEqual(abs(left), 6.28)
        self.assertAlmostEqual(right, -left)


if __name__ == "__main__":
    unittest.main()

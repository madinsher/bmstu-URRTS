"""Приёмочные тесты ПР3. Запуск: python -m unittest tests.test_pr3 -v"""
import itertools
import unittest

import numpy as np

from urrts import graph
from urrts import pr3_allocation as pr3


class Step1Assignment(unittest.TestCase):
    def test_greedy_example(self):
        C = np.array([[1.0, 2.0], [1.1, 10.0]])
        a, cost = pr3.greedy_assignment(C)
        self.assertEqual(a, {0: 0, 1: 1})
        self.assertAlmostEqual(cost, 11.0)
        _, opt = pr3.optimal_assignment(C)
        self.assertAlmostEqual(opt, 3.1)          # жадность проигрывает в 3,5 раза

    def test_rectangular_and_valid(self):
        rng = np.random.default_rng(1)
        C = rng.uniform(0, 1, (4, 7))
        a, cost = pr3.greedy_assignment(C)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(set(a.values())), 4)
        self.assertAlmostEqual(cost, sum(C[i, j] for i, j in a.items()))
        self.assertGreaterEqual(cost, pr3.optimal_assignment(C)[1] - 1e-12)


class Step2SSI(unittest.TestCase):
    def test_best_insertion(self):
        tasks = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
        pos, inc = pr3.best_insertion([0, 0], [0, 1], tasks, 2)
        self.assertEqual(pos, 1)
        self.assertAlmostEqual(inc, 0.0)

    def test_every_task_once(self):
        rng = np.random.default_rng(2)
        robots = rng.uniform(0, 10, (3, 2))
        tasks = rng.uniform(0, 10, (9, 2))
        routes, rounds, bids = pr3.ssi_auction(robots, tasks)
        self.assertEqual(sorted(sum(routes, [])), list(range(9)))
        self.assertEqual(rounds, 9)
        self.assertEqual(bids, 3 * sum(range(1, 10)))

    def test_ssi_close_to_optimum_small(self):
        # полный перебор для 2 роботов и 5 задач
        rng = np.random.default_rng(3)
        robots = rng.uniform(0, 10, (2, 2))
        tasks = rng.uniform(0, 10, (5, 2))
        best = np.inf
        for split in itertools.product([0, 1], repeat=5):
            r = [[j for j in range(5) if split[j] == k] for k in range(2)]
            tot = 0.0
            for k in range(2):
                tot += min(pr3.route_length(robots[k], list(o), tasks)
                           for o in itertools.permutations(r[k])) if r[k] else 0.0
            best = min(best, tot)
        routes, _, _ = pr3.ssi_auction(robots, tasks)
        self.assertLessEqual(pr3.total_length(robots, routes, tasks), 2.0 * best)


class Step3Bundle(unittest.TestCase):
    def test_single_agent_takes_best_tasks(self):
        tasks = np.array([[1.0, 0.0], [2.0, 0.0], [50.0, 0.0]])
        rewards = np.array([10.0, 10.0, 10.0])
        ag = pr3.CBBAAgent(0, [0, 0], 3, capacity=2)
        pr3.cbba_build_bundle(ag, tasks, rewards)
        self.assertEqual(ag.bundle, [0, 1])
        self.assertEqual(ag.path, [0, 1])
        self.assertAlmostEqual(ag.y[0], 10 * 0.95 ** 1)
        self.assertEqual(list(ag.z), [0, 0, -1])

    def test_does_not_outbid_better_known_bid(self):
        tasks = np.array([[1.0, 0.0]])
        ag = pr3.CBBAAgent(0, [0, 0], 1, capacity=1)
        ag.y[0], ag.z[0] = 100.0, 3
        pr3.cbba_build_bundle(ag, tasks, np.array([10.0]))
        self.assertEqual(ag.bundle, [])


class Step4Consensus(unittest.TestCase):
    def _run(self, seed, n=4, m=10, line=False):
        rng = np.random.default_rng(seed)
        starts = rng.uniform(0, 10, (n, 2))
        tasks = rng.uniform(0, 10, (m, 2))
        rewards = rng.uniform(5, 15, m)
        A = graph.path_graph(n) if line else graph.complete_graph(n)
        return pr3.run_cbba(starts, tasks, rewards, A, capacity=3), m

    def test_conflict_free_complete_graph(self):
        for seed in range(5):
            (agents, it, _), m = self._run(seed)
            self.assertTrue(pr3.is_conflict_free(agents), "seed %d" % seed)
            self.assertEqual(sum(len(a.path) for a in agents), m)

    def test_conflict_free_on_line_graph(self):
        for seed in range(5):
            (agents, it, _), m = self._run(seed, line=True)
            self.assertTrue(pr3.is_conflict_free(agents), "seed %d" % seed)
            self.assertLess(it, 200)

    def test_release_resets_later_tasks(self):
        tasks = np.array([[1.0, 0.0], [2.0, 0.0]])
        rewards = np.array([10.0, 10.0])
        a0 = pr3.CBBAAgent(0, [0, 0], 2, 2)
        a1 = pr3.CBBAAgent(1, [0, 0], 2, 2)
        pr3.cbba_build_bundle(a0, tasks, rewards)      # a0: пакет [0, 1]
        a1.y[0], a1.z[0] = 99.0, 1                     # a1 сильно перебивает задачу 0
        a1.bundle, a1.path = [0], [0]
        pr3.cbba_consensus([a0, a1], graph.complete_graph(2))
        self.assertEqual(a0.bundle, [])
        self.assertEqual(a0.z[1], -1)
        self.assertEqual(a0.z[0], 1)


if __name__ == "__main__":
    unittest.main()

"""Приёмочные тесты ПР4. Запуск: python -m unittest tests.test_pr4 -v"""
import itertools
import unittest

import numpy as np

from urrts import pr4_mapf as pr4

CORRIDOR = pr4.load_map("""
#.#
...
#.#
""")


def valid_moves(grid, path):
    for t in range(1, len(path)):
        if path[t] not in pr4.successors(grid, path[t - 1]):
            return False
    return True


def random_instance(rng, grid, n):
    free = [tuple(c) for c in np.argwhere(~grid)]
    idx = rng.choice(len(free), size=2 * n, replace=False)
    return [free[i] for i in idx[:n]], [free[i] for i in idx[n:]]


class Step1STAStar(unittest.TestCase):
    def test_plain_shortest(self):
        g = pr4.load_map(pr4.WAREHOUSE)
        p = pr4.space_time_astar(g, (0, 0), (6, 9))
        self.assertEqual(len(p) - 1, 15)
        self.assertTrue(valid_moves(g, p))

    def test_vertex_constraint_forces_wait(self):
        g = pr4.load_map("...")
        p = pr4.space_time_astar(g, (0, 0), (0, 2), frozenset({("v", (0, 1), 1)}))
        self.assertEqual(p, [(0, 0), (0, 0), (0, 1), (0, 2)])

    def test_edge_constraint(self):
        g = pr4.load_map("..")
        p = pr4.space_time_astar(g, (0, 0), (0, 1), frozenset({("e", (0, 0), (0, 1), 1)}))
        self.assertEqual(p, [(0, 0), (0, 0), (0, 1)])

    def test_goal_blocked_later(self):
        g = pr4.load_map("...")
        p = pr4.space_time_astar(g, (0, 0), (0, 1), frozenset({("v", (0, 1), 4)}))
        self.assertEqual(len(p) - 1, 5)          # на цели можно остаться только после момента 4

    def test_unreachable(self):
        g = pr4.load_map(".#.")
        self.assertIsNone(pr4.space_time_astar(g, (0, 0), (0, 2)))


class Step2Prioritized(unittest.TestCase):
    def test_corridor_cross(self):
        starts, goals = [(1, 0), (0, 1)], [(1, 2), (2, 1)]
        paths = pr4.prioritized_planning(CORRIDOR, starts, goals)
        self.assertIsNone(pr4.first_conflict(paths))
        self.assertEqual(pr4.sum_of_costs(paths), 5)

    def test_incompleteness(self):
        # агент 0 садится на цель в узком месте и запирает агента 1
        g = pr4.load_map("....")
        starts, goals = [(0, 1), (0, 0)], [(0, 2), (0, 3)]
        self.assertIsNone(pr4.prioritized_planning(g, starts, goals, order=[0, 1]))


class Step3Conflicts(unittest.TestCase):
    def test_vertex(self):
        c = pr4.first_conflict([[(0, 0), (0, 1)], [(0, 2), (0, 1)]])
        self.assertEqual(c, ("v", 0, 1, (0, 1), 1))

    def test_swap(self):
        c = pr4.first_conflict([[(0, 0), (0, 1)], [(0, 1), (0, 0)]])
        self.assertEqual(c, ("e", 0, 1, (0, 0), (0, 1), 1))

    def test_parked_agent(self):
        c = pr4.first_conflict([[(0, 0)], [(0, 2), (0, 1), (0, 0)]])
        self.assertEqual(c, ("v", 0, 1, (0, 0), 2))

    def test_following_is_allowed(self):
        self.assertIsNone(pr4.first_conflict([[(0, 1), (0, 2)], [(0, 0), (0, 1)]]))


class Step4CBS(unittest.TestCase):
    def test_solves_incomplete_case(self):
        g = pr4.load_map("....\n.#..")
        starts, goals = [(0, 1), (0, 0)], [(0, 2), (0, 3)]
        paths, nodes = pr4.cbs(g, starts, goals)
        self.assertIsNotNone(paths)
        self.assertIsNone(pr4.first_conflict(paths))

    def test_optimal_vs_bruteforce_order(self):
        rng = np.random.default_rng(11)
        g = pr4.load_map(pr4.WAREHOUSE)
        for _ in range(4):
            starts, goals = random_instance(rng, g, 4)
            paths, _ = pr4.cbs(g, starts, goals)
            self.assertIsNone(pr4.first_conflict(paths))
            for i, p in enumerate(paths):
                self.assertEqual(p[0], starts[i])
                self.assertEqual(p[-1], goals[i])
                self.assertTrue(valid_moves(g, p))
            best_pp = min((pr4.sum_of_costs(pp) for o in itertools.permutations(range(4))
                           for pp in [pr4.prioritized_planning(g, starts, goals, o)] if pp is not None),
                          default=np.inf)
            self.assertLessEqual(pr4.sum_of_costs(paths), best_pp)


class Step5Execution(unittest.TestCase):
    def test_dependencies_simple_follow(self):
        paths = [[(0, 1), (0, 2)], [(0, 0), (0, 1)]]
        actions, deps = pr4.action_dependencies(paths)
        self.assertEqual(deps[(1, 0)], {(0, 0)})
        self.assertEqual(deps[(0, 0)], set())

    def test_delays_never_collide(self):
        rng = np.random.default_rng(12)
        g = pr4.load_map(pr4.WAREHOUSE)
        for _ in range(5):
            starts, goals = random_instance(rng, g, 6)
            paths, _ = pr4.cbs(g, starts, goals)
            ticks, log = pr4.execute(paths, 0.3, rng)
            self.assertIsNotNone(ticks)
            for pos in log:
                self.assertEqual(len(pos), len(set(pos)))
            self.assertEqual(log[-1], [p[-1] for p in paths])


if __name__ == "__main__":
    unittest.main()

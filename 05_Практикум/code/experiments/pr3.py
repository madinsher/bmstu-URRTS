"""ПР3. Эксперименты: жадное и оптимальное назначение, SSI против оптимума,
CBBA на разных топологиях связи, перераспределение после отказа робота.

    python experiments/pr3.py [--seed N] [--out каталог] [--quick]
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from _common import parse, write_table, done
from urrts import graph, plot
from urrts import pr3_allocation as pr3


def e1_greedy(args, rng):
    """Э1. Отношение стоимости жадного назначения к оптимальной (N роботов = N задач)."""
    trials = 50 if args.quick else 300
    rows, ratios = [], {}
    for n in (5, 10, 20, 40):
        r = []
        t_g = t_o = 0.0
        for _ in range(trials):
            C = pr3.cost_matrix(rng.uniform(0, 10, (n, 2)), rng.uniform(0, 10, (n, 2)))
            t0 = time.perf_counter()
            _, g = pr3.greedy_assignment(C)
            t1 = time.perf_counter()
            _, o = pr3.optimal_assignment(C)
            t2 = time.perf_counter()
            t_g += t1 - t0
            t_o += t2 - t1
            r.append(g / o)
        ratios[n] = r
        rows.append([n, round(float(np.mean(r)), 3), round(float(np.max(r)), 3),
                     round(1e3 * t_g / trials, 3), round(1e3 * t_o / trials, 3)])
    write_table(os.path.join(args.out, "e1_greedy.csv"),
                ["N", "среднее_жадн/опт", "худшее_жадн/опт", "мс_жадный", "мс_венгерский"], rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([ratios[n] for n in ratios])
    ax.set_xticks(range(1, len(ratios) + 1))
    ax.set_xticklabels([str(n) for n in ratios])
    ax.set_xlabel("число роботов и задач N")
    ax.set_ylabel("стоимость жадного / оптимальная")
    ax.grid(alpha=0.3)
    path = os.path.join(args.out, "e1_greedy.png")
    plot.save(path, fig)
    done(path)


def e2_ssi(args, rng):
    """Э2. SSI (MiniSum): суммарная длина, наибольший маршрут и дисбаланс нагрузки, число ставок."""
    trials = 20 if args.quick else 100
    rows = []
    for n, m in ((3, 9), (5, 20), (8, 40)):
        tot, mk, bids = [], [], []
        for _ in range(trials):
            robots, tasks = rng.uniform(0, 10, (n, 2)), rng.uniform(0, 10, (m, 2))
            routes, _, b = pr3.ssi_auction(robots, tasks)
            tot.append(pr3.total_length(robots, routes, tasks))
            mk.append(pr3.makespan(robots, routes, tasks))
            bids.append(b)
        rows.append([n, m, round(float(np.mean(tot)), 2), round(float(np.mean(mk)), 2),
                     round(float(np.mean(mk)) / (float(np.mean(tot)) / n), 2), int(np.mean(bids))])
    write_table(os.path.join(args.out, "e2_ssi.csv"),
                ["роботов", "задач", "сумма_длин_м", "наибольший_маршрут_м",
                 "дисбаланс_макс/средн", "ставок"], rows)
    robots, tasks = rng.uniform(0, 10, (4, 2)), rng.uniform(0, 10, (16, 2))
    routes, _, _ = pr3.ssi_auction(robots, tasks)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for i, r in enumerate(routes):
        pts = np.vstack([robots[i]] + [tasks[j] for j in r])
        ax.plot(pts[:, 0], pts[:, 1], "-o", ms=4, label="робот %d: %d задач, %.1f м"
                % (i, len(r), pr3.route_length(robots[i], r, tasks)))
        ax.plot(*robots[i], "ks", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.legend(fontsize=8)
    ax.set_title("SSI, критерий MiniSum")
    path = os.path.join(args.out, "e2_ssi_routes.png")
    plot.save(path, fig)
    done(path)


def e3_cbba(args, rng):
    """Э3. CBBA: итерации и сообщения в зависимости от диаметра графа связи."""
    trials = 10 if args.quick else 40
    n, m = 8, 20
    tops = {"полный": graph.complete_graph(n), "кольцо": graph.ring_graph(n),
            "звезда": graph.star_graph(n), "путь": graph.path_graph(n)}
    rows = []
    for name, A in tops.items():
        its, msgs, scores = [], [], []
        diam = diameter(A)
        for _ in range(trials):
            starts, tasks, rew = rng.uniform(0, 10, (n, 2)), rng.uniform(0, 10, (m, 2)), rng.uniform(5, 15, m)
            agents, it, msg = pr3.run_cbba(starts, tasks, rew, A, capacity=4)
            assert pr3.is_conflict_free(agents)
            its.append(it)
            msgs.append(msg)
            scores.append(pr3.total_score(agents, tasks, rew))
        rows.append([name, diam, round(float(np.mean(its)), 1), int(np.max(its)),
                     int(np.mean(msgs)), round(float(np.mean(scores)), 2)])
    write_table(os.path.join(args.out, "e3_cbba.csv"),
                ["граф", "диаметр", "итераций_среднее", "итераций_макс", "сообщений", "суммарная_оценка"], rows)


def e4_failure(args, rng):
    """Э4. Отказ робота: перераспределение его задач повторным запуском CBBA из текущего состояния."""
    n, m = 6, 18
    starts, tasks, rew = rng.uniform(0, 10, (n, 2)), rng.uniform(0, 10, (m, 2)), rng.uniform(5, 15, m)
    A = graph.complete_graph(n)
    agents, it, _ = pr3.run_cbba(starts, tasks, rew, A, capacity=4)
    before = ["до отказа", n, sum(len(a.path) for a in agents),
              round(pr3.total_score(agents, tasks, rew), 2), it]
    failed = int(np.argmax([len(a.path) for a in agents]))
    lost = list(agents[failed].path)
    # выжившие сохраняют свои пакеты; задачи отказавшего освобождаются у всех
    alive = [a for a in agents if a.idx != failed]
    for new_idx, a in enumerate(alive):
        a.idx = new_idx
        for j in range(m):
            if a.z[j] == failed or j in lost:
                a.y[j], a.z[j] = 0.0, -1
            elif a.z[j] > failed:
                a.z[j] -= 1
    A2 = graph.complete_graph(n - 1)
    for k in range(1, 100):
        for a in alive:
            pr3.cbba_build_bundle(a, tasks, rew)
        if not pr3.cbba_consensus(alive, A2):
            break
    fresh, it_fresh, _ = pr3.run_cbba(np.delete(starts, failed, axis=0), tasks, rew, A2, capacity=4)
    assert pr3.is_conflict_free(alive)
    rows = [before,
            ["перераспределение", n - 1, sum(len(a.path) for a in alive), round(pr3.total_score(alive, tasks, rew), 2), k],
            ["с нуля без отказавшего", n - 1, sum(len(a.path) for a in fresh),
             round(pr3.total_score(fresh, tasks, rew), 2), it_fresh]]
    write_table(os.path.join(args.out, "e4_failure.csv"),
                ["сценарий", "роботов", "назначено_задач", "суммарная_оценка", "итераций"], rows)
    print("  отказал робот %d, у него было задач: %d" % (failed, len(lost)))


def diameter(A):
    n = len(A)
    D = np.full((n, n), np.inf)
    D[A > 0] = 1
    np.fill_diagonal(D, 0)
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return int(D.max())


if __name__ == "__main__":
    args = parse("ПР3")
    rng = np.random.default_rng(args.seed)
    for f in (e1_greedy, e2_ssi, e3_cbba, e4_failure):
        print(f.__doc__.strip().splitlines()[0])
        f(args, rng)

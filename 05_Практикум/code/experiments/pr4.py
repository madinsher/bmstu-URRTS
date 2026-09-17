"""ПР4. Эксперименты: приоритетное планирование против CBS на складе,
исполнение плана с задержками — по графу зависимостей и «по часам».

    python experiments/pr4.py [--seed N] [--out каталог] [--quick]
"""
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from _common import parse, write_table, done
from urrts import plot
from urrts import pr4_mapf as pr4


def instance(rng, grid, n):
    free = [tuple(c) for c in np.argwhere(~grid)]
    idx = rng.choice(len(free), size=2 * n, replace=False)
    return [free[i] for i in idx[:n]], [free[i] for i in idx[n:]]


def e1_pp_vs_cbs(args, rng):
    """Э1. Доля решённых задач, сумма стоимостей и время: приоритетное планирование и CBS."""
    grid = pr4.load_map(pr4.WAREHOUSE)
    trials = 8 if args.quick else 30
    counts = (2, 4, 6, 8) if args.quick else (2, 4, 6, 8, 10, 12)
    rows = []
    rate = {"PP": [], "CBS": []}
    for n in counts:
        ok_pp = ok_cbs = 0
        soc_pp, soc_cbs, t_pp, t_cbs, nodes = [], [], [], [], []
        for _ in range(trials):
            s, g = instance(rng, grid, n)
            t0 = time.perf_counter()
            pp = pr4.prioritized_planning(grid, s, g)
            t1 = time.perf_counter()
            cb, nd = pr4.cbs(grid, s, g, max_nodes=2000)
            t2 = time.perf_counter()
            t_pp.append(t1 - t0)
            t_cbs.append(t2 - t1)
            nodes.append(nd)
            if pp is not None:
                ok_pp += 1
            if cb is not None:
                ok_cbs += 1
            if pp is not None and cb is not None:
                soc_pp.append(pr4.sum_of_costs(pp))
                soc_cbs.append(pr4.sum_of_costs(cb))
        rate["PP"].append(ok_pp / trials)
        rate["CBS"].append(ok_cbs / trials)
        gap = (np.mean(soc_pp) / np.mean(soc_cbs) - 1) * 100 if soc_cbs else float("nan")
        rows.append([n, "%d/%d" % (ok_pp, trials), "%d/%d" % (ok_cbs, trials), round(float(gap), 2),
                     round(1e3 * float(np.median(t_pp)), 2), round(1e3 * float(np.median(t_cbs)), 1),
                     int(np.max(nodes))])
    write_table(os.path.join(args.out, "e1_pp_vs_cbs.csv"),
                ["агентов", "решено_PP", "решено_CBS", "PP_хуже_CBS_%", "мс_PP_медиана",
                 "мс_CBS_медиана", "узлов_CBS_макс"], rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in rate.items():
        ax.plot(counts, v, "o-", label=k)
    ax.set_xlabel("число роботов")
    ax.set_ylabel("доля решённых задач")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Склад 7 × 10, лимит CBS 2000 узлов")
    path = os.path.join(args.out, "e1_success.png")
    plot.save(path, fig)
    done(path)


def e2_execution(args, rng):
    """Э2. Исполнение с задержками: столкновения «по часам» и удлинение по графу зависимостей."""
    grid = pr4.load_map(pr4.WAREHOUSE)
    trials = 10 if args.quick else 40
    rows = []
    plans = []
    while len(plans) < trials:
        s, g = instance(rng, grid, 6)
        paths, _ = pr4.cbs(grid, s, g, max_nodes=2000)
        if paths is not None:
            plans.append(paths)
    for p_delay in (0.0, 0.1, 0.3, 0.5):
        coll, stretch = [], []
        for paths in plans:
            coll.append(pr4.execute_naive(paths, p_delay, rng))
            ticks, log = pr4.execute(paths, p_delay, rng)
            assert all(len(x) == len(set(x)) for x in log)
            makespan = max(len(p) - 1 for p in paths)
            stretch.append(ticks / max(makespan, 1))
        rows.append([p_delay, round(float(np.mean(coll)), 2), "%d/%d" % (sum(c > 0 for c in coll), trials),
                     0, round(float(np.mean(stretch)), 2)])
    write_table(os.path.join(args.out, "e2_execution.csv"),
                ["вероятность_задержки", "столкновений_по_часам_среднее", "планов_со_столкновением",
                 "столкновений_по_графу", "длительность/план"], rows)
    # иллюстрация одного плана
    paths = plans[0]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.imshow(grid, cmap="Greys", origin="upper")
    for i, p in enumerate(paths):
        P = np.array(p) + 0.08 * (i - len(paths) / 2) / len(paths)
        ax.plot(P[:, 1], P[:, 0], "-", lw=2)
        ax.plot(P[0, 1], P[0, 0], "o", ms=8)
        ax.plot(P[-1, 1], P[-1, 0], "s", ms=8)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xlabel("столбец")
    ax.set_ylabel("строка")
    ax.set_title("Решение CBS: 6 роботов, стеллажи — тёмные клетки")
    path = os.path.join(args.out, "e2_plan.png")
    plot.save(path, fig)
    done(path)


if __name__ == "__main__":
    args = parse("ПР4")
    rng = np.random.default_rng(args.seed)
    for f in (e1_pp_vs_cbs, e2_execution):
        print(f.__doc__.strip().splitlines()[0])
        f(args, rng)

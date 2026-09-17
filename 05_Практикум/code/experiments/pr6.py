"""ПР6. Эксперименты: обмен местами группы с барьерными функциями, влияние γ,
правило выхода из тупика, устаревшие данные о соседе.

    python experiments/pr6.py [--seed N] [--out каталог] [--quick]
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from _common import parse, write_table, done
from urrts import plot, sim
from urrts import pr6_safety as pr6

DT = 0.02
D_SAFE = 0.2
VMAX = 0.3


def run(p0, goals, steps, gamma=2.0, unstuck=True, record=False):
    p = np.array(p0, dtype=float)
    n = len(p)
    u_prev = np.ones_like(p)
    hold = np.zeros(n, dtype=int)
    poly = pr6.speed_polygon(VMAX)
    dmin, arrive, P = np.inf, None, [p.copy()]
    interventions = 0
    for k in range(steps):
        if unstuck:
            u_nom, hold = pr6.unstuck_nominal(p, goals, u_prev, hold, 1.0, VMAX)
        else:
            u_nom = pr6.go_to_goal(p, goals, 1.0, VMAX)
        u = np.zeros_like(p)
        for i in range(n):
            cons = list(poly) + [pr6.pair_constraint(p[i], p[j], D_SAFE, gamma)
                                 for j in range(n) if j != i and np.linalg.norm(p[i] - p[j]) < 1.0]
            u[i] = pr6.safe_velocity(u_nom[i], cons)
            interventions += int(np.linalg.norm(u[i] - u_nom[i]) > 1e-6)
        p = p + DT * u
        u_prev = u
        dmin = min(dmin, pr6.min_pair_distance(p))
        if record:
            P.append(p.copy())
        if arrive is None and np.linalg.norm(p - goals, axis=1).max() < 0.05:
            arrive = (k + 1) * DT
    return p, dmin, arrive, np.array(P), interventions / (steps * n)


def e1_swap(args, rng):
    """Э1. Обмен местами N роботов: безопасность, прибытие, доля шагов с вмешательством фильтра."""
    steps = 1500 if args.quick else 3500
    rows = []
    for n in (2, 4, 8, 12):
        for unstuck in (False, True):
            p0, goals = pr6.antipodal(n, 1.0)
            p0 = p0 + 0.002 * rng.normal(size=p0.shape)
            _, dmin, arrive, _, frac = run(p0, goals, steps, unstuck=unstuck)
            rows.append([n, "да" if unstuck else "нет", round(dmin, 4),
                         "не прибыли" if arrive is None else round(arrive, 2), round(frac, 3)])
    write_table(os.path.join(args.out, "e1_swap.csv"),
                ["роботов", "правило_тупика", "мин_дистанция_м", "время_прибытия_с", "доля_вмешательств"], rows)
    p0, goals = pr6.antipodal(8, 1.0)
    _, _, _, P, _ = run(p0, goals, steps, unstuck=True, record=True)
    hist = sim.History()
    for k in range(0, len(P), 5):
        hist.add(k * DT, P[k], np.zeros_like(P[k]), np.zeros((8, 8)))
    plot.trajectories(hist, "Обмен местами 8 роботов, d = %.1f м" % D_SAFE, goals=goals)
    path = os.path.join(args.out, "e1_swap_traj.png")
    plot.save(path)
    done(path)


def e2_gamma(args, rng):
    """Э2. Коэффициент γ: насколько близко роботы подходят к границе и как быстро прибывают."""
    steps = 1500 if args.quick else 3500
    rows = []
    for gamma in (0.2, 1.0, 5.0, 20.0):
        p0, goals = pr6.antipodal(6, 1.0)
        _, dmin, arrive, _, frac = run(p0, goals, steps, gamma=gamma)
        rows.append([gamma, round(dmin, 4), "не прибыли" if arrive is None else round(arrive, 2), round(frac, 3)])
    write_table(os.path.join(args.out, "e2_gamma.csv"),
                ["gamma", "мин_дистанция_м", "время_прибытия_с", "доля_вмешательств"], rows)


def e3_stale(args, rng):
    """Э3. Сосед без связи едет на робота: минимальная дистанция при разной задержке данных."""
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for lag_s in (0.0, 0.2, 0.5, 1.0):
        for aware in (False, True):
            lag = int(round(lag_s / DT))
            pi, pj, vj = np.array([0.0, 0.0]), np.array([1.2, 0.02]), np.array([-0.15, 0.0])
            hist_j, dist = [pj.copy()], []
            for _ in range(1200):
                known = hist_j[max(0, len(hist_j) - 1 - lag)]
                age = min(lag, len(hist_j) - 1) * DT
                if aware:
                    c = pr6.stale_constraint(pi, known, age, 0.15, D_SAFE, 2.0)
                else:
                    c = pr6.pair_constraint(pi, known, D_SAFE, 2.0)      # как будто сосед свежий и соблюдает правило
                ui = pr6.safe_velocity(np.zeros(2), pr6.speed_polygon(VMAX) + [c])
                pi = pi + DT * ui
                pj = pj + DT * vj
                hist_j.append(pj.copy())
                dist.append(np.linalg.norm(pi - pj))
            rows.append([lag_s, "да" if aware else "нет", round(min(dist), 4),
                         "нарушение" if min(dist) < D_SAFE - 1e-3 else "безопасно"])
            if lag_s == 0.5:
                ax.plot(DT * np.arange(len(dist)), dist, label="учёт устаревания: %s" % ("да" if aware else "нет"))
    ax.axhline(D_SAFE, color="r", ls="--", label="d безопасная")
    ax.set_xlabel("t, с")
    ax.set_ylabel("дистанция до соседа, м")
    ax.set_title("Задержка данных 0,5 с, сосед не соблюдает ограничение")
    ax.grid(alpha=0.3)
    ax.legend()
    path = os.path.join(args.out, "e3_stale.png")
    plot.save(path, fig)
    done(path)
    write_table(os.path.join(args.out, "e3_stale.csv"),
                ["задержка_с", "учёт_устаревания", "мин_дистанция_м", "итог"], rows)


if __name__ == "__main__":
    args = parse("ПР6")
    rng = np.random.default_rng(args.seed)
    for f in (e1_swap, e2_gamma, e3_stale):
        print(f.__doc__.strip().splitlines()[0])
        f(args, rng)

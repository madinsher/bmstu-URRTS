"""ПР5. Эксперименты: покрытие по Ллойду (равномерная и сосредоточенная важность),
ограниченная дальность восприятия, распределённая оценка против наивного усреднения.

    python experiments/pr5.py [--seed N] [--out каталог] [--quick]
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from _common import parse, write_table, done
from urrts import graph, plot
from urrts import pr5_coverage as pr5


def draw_cells(ax, p, q, lab, phi, title):
    ax.scatter(q[:, 0], q[:, 1], c=lab, s=6, cmap="tab10", alpha=0.35 + 0.0 * phi)
    ax.plot(p[:, 0], p[:, 1], "k^", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(title)


def e1_lloyd(args, rng):
    """Э1. Ллойд: функционал покрытия по итерациям для равномерной и сосредоточенной важности."""
    q, dA = pr5.grid_points(0, 4, 0, 4, 0.05 if not args.quick else 0.1)
    p0 = rng.uniform(0, 1.0, (8, 2))
    dens = {"равномерная": np.ones(len(q)), "очаг в (3; 3)": pr5.gaussian_density(q, [3, 3], 0.6)}
    iters = 30 if args.quick else 80
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (name, phi) in enumerate(dens.items()):
        p = p0.copy()
        H = [pr5.coverage_cost(p, q, phi, dA)]
        traj = [p.copy()]
        for _ in range(iters):
            p = pr5.lloyd_step(p, q, phi, dA)
            H.append(pr5.coverage_cost(p, q, phi, dA))
            traj.append(p.copy())
        axes[0].semilogy(np.array(H) / H[0], label=name)
        k90 = int(np.argmax(np.array(H) <= H[-1] + 0.1 * (H[0] - H[-1])))
        rows.append([name, round(H[0], 3), round(H[-1], 4), round(H[-1] / H[0], 4), k90,
                     "да" if np.all(np.diff(H) <= 1e-9) else "нет"])
        T = np.array(traj)
        draw_cells(axes[k + 1], p, q, pr5.voronoi_labels(p, q), phi, "Итог: важность %s" % name)
        for i in range(T.shape[1]):
            axes[k + 1].plot(T[:, i, 0], T[:, i, 1], "k-", lw=0.6)
    axes[0].set_xlabel("итерация")
    axes[0].set_ylabel("H / H₀")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    path = os.path.join(args.out, "e1_lloyd.png")
    plot.save(path, fig)
    done(path)
    write_table(os.path.join(args.out, "e1_lloyd.csv"),
                ["важность", "H0", "H_итог", "H_итог/H0", "итераций_до_90%", "монотонно"], rows)


def e2_limited(args, rng):
    """Э2. Ограниченная дальность: итоговое покрытие и разброс роботов в зависимости от r."""
    q, dA = pr5.grid_points(0, 5, 0, 5, 0.1)
    phi = np.ones(len(q))
    p0 = rng.uniform(2.0, 3.0, (10, 2))
    iters = 40 if args.quick else 150
    _, _ = None, None
    full = p0.copy()
    for _ in range(iters):
        full = pr5.lloyd_step(full, q, phi, dA)
    H_full = pr5.coverage_cost(full, q, phi, dA)
    rows = [["∞ (полный Ллойд)", round(H_full, 3), 1.0]]
    for r in (0.3, 0.6, 1.0, 1.5):
        p = p0.copy()
        for _ in range(iters):
            p = pr5.limited_lloyd_step(p, q, phi, dA, r)
        H = pr5.coverage_cost(p, q, phi, dA)
        rows.append([r, round(H, 3), round(H / H_full, 3)])
    write_table(os.path.join(args.out, "e2_limited.csv"), ["дальность_r_м", "H_итог", "H/H_полного"], rows)


def e3_estimation(args, rng):
    """Э3. Оценка положения цели: консенсус в информационной форме против наивного усреднения."""
    n = 12
    trials = 20 if args.quick else 100
    iters = 60
    err_info = np.zeros(iters + 1)
    err_naive = np.zeros(iters + 1)
    err_central = 0.0
    x_true = np.array([2.0, 1.0])
    for _ in range(trials):
        _, A = graph.random_geometric_graph(n, 0.45, rng)
        W = graph.metropolis_weights(A)
        R = []
        for i in range(n):
            s = 0.02 if i < 3 else 1.0                  # три робота с точным дальномером
            R.append(np.diag([s, s]))
        z = [x_true + rng.multivariate_normal([0, 0], Ri) for Ri in R]
        info = pr5.information_consensus(z, R, W, iters)
        naive = pr5.naive_average_consensus(z, W, iters)
        xc, _ = pr5.centralized_estimate(z, R)
        err_info += np.linalg.norm(info - x_true, axis=2).mean(axis=1)
        err_naive += np.linalg.norm(naive - x_true, axis=2).mean(axis=1)
        err_central += np.linalg.norm(xc - x_true)
    err_info /= trials
    err_naive /= trials
    err_central /= trials
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(err_info, label="информационная форма")
    ax.plot(err_naive, label="наивное усреднение")
    ax.axhline(err_central, color="k", ls="--", label="централизованная оценка")
    ax.set_xlabel("итерация консенсуса")
    ax.set_ylabel("средняя ошибка оценки, м")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("N = %d: 3 точных и 9 грубых датчиков" % n)
    path = os.path.join(args.out, "e3_estimation.png")
    plot.save(path, fig)
    done(path)
    write_table(os.path.join(args.out, "e3_estimation.csv"),
                ["метод", "ошибка_итер_0_м", "ошибка_итер_10_м", "ошибка_итог_м"],
                [["информационная форма", round(err_info[0], 3), round(err_info[10], 3), round(err_info[-1], 3)],
                 ["наивное усреднение", round(err_naive[0], 3), round(err_naive[10], 3), round(err_naive[-1], 3)],
                 ["централизованная", "—", "—", round(err_central, 3)]])


if __name__ == "__main__":
    args = parse("ПР5")
    rng = np.random.default_rng(args.seed)
    for f in (e1_lloyd, e2_limited, e3_estimation):
        print(f.__doc__.strip().splitlines()[0])
        f(args, rng)

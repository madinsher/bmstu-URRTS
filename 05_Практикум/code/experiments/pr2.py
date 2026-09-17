"""ПР2. Эксперименты: стая Рейнольдса и баланс весов, фазовый переход Вичека,
роевой поиск источника (топология, шум, ложный максимум).

    python experiments/pr2.py [--seed N] [--out каталог] [--quick]
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from _common import parse, write_table, done
from urrts import plot
from urrts import pr2_swarm as pr2


def e1_boids(args, rng):
    """Э1. Стая Рейнольдса: параметр порядка и минимальная дистанция при разных весах правил."""
    n, steps = 30, (300 if args.quick else 1000)
    configs = [("сбалансированные", dict(w_sep=0.05, w_ali=5.0, w_coh=0.1)),
               ("без выравнивания", dict(w_sep=0.05, w_ali=0.0, w_coh=0.1)),
               ("без разделения", dict(w_sep=0.0, w_ali=5.0, w_coh=0.1)),
               ("сильное сплочение", dict(w_sep=0.05, w_ali=1.0, w_coh=3.0))]
    p0 = rng.uniform(0, 2.0, (n, 2))
    v0 = rng.normal(0, 0.1, (n, 2))
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, w in configs:
        p, v = p0.copy(), v0.copy()
        phi, dmin = [], np.inf
        traj = [p.copy()]
        for _ in range(steps):
            p, v = pr2.boids_step(p, v, 0.05, 0.2, 0.3, r_sep=0.15, r_view=0.6, **w)
            phi.append(pr2.polarization(v))
            d = np.linalg.norm(p[:, None] - p[None], axis=2) + 1e9 * np.eye(n)
            dmin = min(dmin, d.min())
            traj.append(p.copy())
        ax.plot(0.05 * np.arange(1, steps + 1), phi, label=name)
        rows.append([name, round(float(np.mean(phi[-100:])), 3), round(float(dmin), 3)])
        if name == "сбалансированные":
            T = np.array(traj)
            f2, a2 = plt.subplots(figsize=(6.4, 6))
            for i in range(n):
                a2.plot(T[:, i, 0], T[:, i, 1], lw=0.6)
            a2.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], color="k", scale=5)
            a2.set_aspect("equal")
            a2.set_xlabel("x, м")
            a2.set_ylabel("y, м")
            a2.set_title("Стая Рейнольдса, N = %d" % n)
            path = os.path.join(args.out, "e1_boids_traj.png")
            plot.save(path, f2)
            done(path)
    ax.set_xlabel("t, с")
    ax.set_ylabel("параметр порядка φ")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    path = os.path.join(args.out, "e1_boids_order.png")
    plot.save(path, fig)
    done(path)
    write_table(os.path.join(args.out, "e1_boids.csv"), ["веса", "phi_в_конце", "мин_дистанция_м"], rows)


def e2_vicsek(args, rng):
    """Э2. Фазовый переход в модели Вичека: φ(η) для разной плотности."""
    etas = np.linspace(0.0, 2 * np.pi, 7 if args.quick else 14)
    steps = 150 if args.quick else 400
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, box in ((50, 7.0), (100, 7.0), (200, 7.0)):
        phis = [pr2.vicsek_order(n, box, 1.0, eta, steps, rng) for eta in etas]
        ax.plot(etas, phis, "o-", label="N = %d, ρ = %.1f м⁻²" % (n, n / box ** 2))
        rows += [[n, round(float(e), 3), round(p, 3)] for e, p in zip(etas, phis)]
    ax.set_xlabel("амплитуда шума η, рад")
    ax.set_ylabel("параметр порядка φ")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Модель Вичека: от стаи к беспорядку")
    path = os.path.join(args.out, "e2_vicsek.png")
    plot.save(path, fig)
    done(path)
    write_table(os.path.join(args.out, "e2_vicsek.csv"), ["N", "eta", "phi"], rows)


def e3_source(args, rng):
    """Э3. Поиск источника: доля успешных прогонов для топологий lbest и шума измерений."""
    src = np.array([3.0, 2.5])
    f = pr2.gaussian_source(src, sigma=0.9, distractors=[([0.8, 3.2], 0.35, 0.6)])
    runs = 5 if args.quick else 20
    steps = 250 if args.quick else 600
    rows = []
    for k, noise, forget in [(1, 0.0, 0.0), (2, 0.0, 0.0), (4, 0.0, 0.0),
                             (2, 0.001, 0.0), (2, 0.01, 0.0), (2, 0.01, 0.02), (2, 0.05, 0.02)]:
        ok, times = 0, []
        for r in range(runs):
            rr = np.random.default_rng(1000 * k + r)
            x0 = rr.uniform(0, 0.6, (8, 2))
            traj, _ = pr2.robot_pso_search(f, x0, steps, rr, vmax=0.03, d_min=0.12, k=k,
                                           noise_std=noise, forget=forget)
            dist = np.linalg.norm(traj.mean(axis=1) - src, axis=1)
            if dist[-1] < 0.3:
                ok += 1
                times.append(int(np.argmax(dist < 0.3)))
        rows.append([k, noise, forget, "%d/%d" % (ok, runs),
                     round(float(np.mean(times)), 1) if times else "—"])
    write_table(os.path.join(args.out, "e3_source.csv"),
                ["k_окрестности", "СКО_шума", "забывание", "успешных", "шагов_до_цели"], rows)
    x0 = rng.uniform(0, 0.6, (8, 2))
    traj, best = pr2.robot_pso_search(f, x0, steps, rng, vmax=0.03, d_min=0.12, k=2, noise_std=0.001)
    g = np.linspace(-0.5, 4.5, 120)
    X, Y = np.meshgrid(g, g)
    Z = np.vectorize(lambda a, b: f([a, b]))(X, Y)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.contourf(X, Y, Z, 20, cmap="Greys")
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], lw=1)
    ax.plot(*src, "r*", ms=14)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title("Роевой поиск источника (k = 2, шум 0,001)")
    path = os.path.join(args.out, "e3_source_traj.png")
    plot.save(path, fig)
    done(path)


if __name__ == "__main__":
    args = parse("ПР2")
    rng = np.random.default_rng(args.seed)
    for f in (e1_boids, e2_vicsek, e3_source):
        print(f.__doc__.strip().splitlines()[0])
        f(args, rng)

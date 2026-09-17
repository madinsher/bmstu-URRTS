"""ПР1. Эксперименты: топология и скорость консенсуса, формация при потерях связи,
сохранение связности, устойчивость к злонамеренному агенту.

    python experiments/pr1.py [--seed N] [--out каталог] [--quick]
"""
import os

import numpy as np

from _common import parse, write_table, done
from urrts import graph, plot, sim
from urrts import pr1_consensus as pr1
import matplotlib.pyplot as plt


def e1_topology(args, rng):
    """Э1. Число итераций до рассогласования 10⁻³ от начального и λ₂ для разных графов."""
    n = 12
    _, rgg = graph.random_geometric_graph(n, 0.45, rng)
    graphs = {"путь": graph.path_graph(n), "кольцо": graph.ring_graph(n), "звезда": graph.star_graph(n),
              "случайный геометрический": rgg, "полный": graph.complete_graph(n)}
    x0 = rng.normal(size=n)
    rows, curves = [], {}
    for name, A in graphs.items():
        eps = 0.5 * pr1.max_consensus_step(A)
        xs = pr1.run_consensus(x0, A, eps, 3000)
        dis = np.array([pr1.disagreement(x) for x in xs]) / pr1.disagreement(x0)
        k = int(np.argmax(dis < 1e-3)) if np.any(dis < 1e-3) else -1
        lam2 = pr1.algebraic_connectivity(A)
        L = pr1.laplacian(A)
        ev = np.linalg.eigvalsh(np.eye(n) - eps * L)
        rho = max(abs(ev[0]), abs(ev[-2]))           # второе по модулю собственное число P
        predicted = int(np.ceil(np.log(1e-3) / np.log(rho))) if rho < 1 else -1
        rows.append([name, round(lam2, 4), round(eps, 4), round(rho, 4), k, predicted])
        curves[name] = dis[:400]
    write_table(os.path.join(args.out, "e1_topology.csv"),
                ["граф", "lambda2", "eps", "rho(P)", "итераций_до_1e-3", "оценка_по_rho"], rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in curves.items():
        ax.semilogy(c, label=name)
    ax.set_xlabel("итерация k")
    ax.set_ylabel("‖x − x̄‖ / ‖x₀ − x̄‖")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Скорость консенсуса, N = %d, ε = 0,5 / Δmax" % n)
    path = os.path.join(args.out, "e1_topology.png")
    plot.save(path, fig)
    done(path)


def e2_formation(args, rng):
    """Э2. Формация «шестиугольник» по радиоканалу с ограниченной дальностью и потерями."""
    n = 6
    ang = 2 * np.pi * np.arange(n) / n
    offsets = 0.4 * np.c_[np.cos(ang), np.sin(ang)]
    p0 = rng.uniform(0, 1.5, (n, 2))
    steps = 300 if args.quick else 1200
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for drop in (0.0, 0.3, 0.6, 0.9):
        comm = sim.CommModel(radius=1.2, drop_prob=drop, rng=np.random.default_rng(args.seed))
        hist, _ = sim.simulate(p0, lambda t, p, A, s: pr1.formation_velocity(p, A, offsets, 1.0),
                               steps, dt=0.05, comm=comm, vmax=0.3)
        err = [pr1.formation_error(P, offsets) for P in hist.p]
        k = next((i for i, e in enumerate(err) if e < 0.02), None)
        rows.append([drop, round(err[-1], 4), "—" if k is None else round(hist.t[k], 2)])
        ax.semilogy(hist.t, err, label="потери %.0f %%" % (100 * drop))
        if drop == 0.3:
            plot.trajectories(hist, "Формация при потерях 30 %%, R = 1,2 м")
            path = os.path.join(args.out, "e2_formation_traj.png")
            plot.save(path)
            done(path)
    ax.set_xlabel("t, с")
    ax.set_ylabel("ошибка формы, м")
    ax.grid(alpha=0.3)
    ax.legend()
    path = os.path.join(args.out, "e2_formation_error.png")
    plot.save(path, fig)
    done(path)
    write_table(os.path.join(args.out, "e2_formation.csv"),
                ["вероятность_потери", "ошибка_в_конце_м", "время_до_2см_с"], rows)


def e3_connectivity(args, rng):
    """Э3. Сближение по текущему графу против сближения с сохранением связности."""
    R = 1.0
    # плотная группа из четырёх роботов, «мост» и «хвост» на границе дальности связи
    p0 = np.array([[0, 0], [0.15, 0.1], [0.1, -0.15], [-0.1, 0.05], [0.9, 0], [1.85, 0.05]], dtype=float)
    p0[:4] += 0.03 * rng.normal(size=(4, 2))
    A0 = graph.disk_graph(p0, R)
    steps = 1000 if args.quick else 3000
    runs = {
        "консенсус по текущему графу": lambda t, p, A, s: -(pr1.laplacian(A) @ p),
        "с сохранением связности": lambda t, p, A, s: pr1.rendezvous_velocity(p, A0, R, 0.5),
    }
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, ctrl) in zip(axes, runs.items()):
        hist, _ = sim.simulate(p0, ctrl, steps, dt=0.002, comm=sim.CommModel(radius=R), vmax=2.0)
        comps = [len(graph.components(graph.disk_graph(P, R))) for P in hist.p]
        spread = np.linalg.norm(hist.p[-1] - hist.p[-1].mean(axis=0), axis=1).max()
        rows.append([name, max(comps), comps[-1], round(float(spread), 3)])
        plot.trajectories(hist, name, A_final=graph.disk_graph(hist.p[-1], R), ax=ax)
        ax.set_aspect("auto")
        ax.set_ylim(-0.5, 0.5)
    path = os.path.join(args.out, "e3_connectivity.png")
    plot.save(path, fig)
    done(path)
    write_table(os.path.join(args.out, "e3_connectivity.csv"),
                ["закон", "наибольшее_число_компонент", "компонент_в_конце", "разброс_в_конце_м"], rows)


def e4_wmsr(args, rng):
    """Э4 (★). Злонамеренный агент против обычного консенсуса и W-MSR."""
    n, F = 10, 1
    A = graph.complete_graph(n)
    for i in range(n):                      # прорежаем полный граф, сохраняя 3-робастность
        A[i, (i + 5) % n] = A[(i + 5) % n, i] = 0
    x0 = rng.uniform(0, 1, n)
    bad = [n - 1]
    steps = 200
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    rows = []
    for ax, name in zip(axes, ("обычный консенсус", "W-MSR, F = 1")):
        x = x0.copy()
        traj = [x.copy()]
        for k in range(steps):
            x[bad] = 0.5 + 2.0 * np.sin(0.2 * k)
            x = pr1.consensus_step(x, A, 0.08) if name.startswith("обычный") else pr1.wmsr_step(x, A, F, 0.08, bad)
            traj.append(x.copy())
        traj = np.array(traj)
        normal = [i for i in range(n) if i not in bad]
        ax.plot(traj[:, normal], color="tab:blue", lw=0.8)
        ax.plot(traj[:, bad], color="tab:red", lw=1.5)
        ax.set_title(name)
        ax.set_xlabel("итерация")
        ax.grid(alpha=0.3)
        inside = x0[normal].min() - 1e-9 <= traj[-1, normal].mean() <= x0[normal].max() + 1e-9
        rows.append([name, round(float(np.ptp(traj[-1, normal])), 4), "да" if inside else "нет"])
    axes[0].set_ylabel("состояние агента")
    path = os.path.join(args.out, "e4_wmsr.png")
    plot.save(path, fig)
    done(path)
    write_table(os.path.join(args.out, "e4_wmsr.csv"),
                ["алгоритм", "разброс_нормальных", "итог_в_диапазоне_начальных"], rows)


if __name__ == "__main__":
    args = parse("ПР1")
    rng = np.random.default_rng(args.seed)
    for f in (e1_topology, e2_formation, e3_connectivity, e4_wmsr):
        print(f.__doc__.strip().splitlines()[0])
        f(args, rng)

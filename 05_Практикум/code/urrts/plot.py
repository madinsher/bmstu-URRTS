"""Графики для отчётов практикума (оси подписаны, единицы указаны)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def trajectories(hist, title="", A_final=None, ax=None, goals=None):
    """Траектории роботов: старт — кружок, финиш — квадрат."""
    P = hist.positions
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
    n = P.shape[1]
    colors = plt.cm.tab10(np.arange(n) % 10)
    for i in range(n):
        ax.plot(P[:, i, 0], P[:, i, 1], "-", color=colors[i], lw=1.2)
        ax.plot(P[0, i, 0], P[0, i, 1], "o", color=colors[i], ms=6)
        ax.plot(P[-1, i, 0], P[-1, i, 1], "s", color=colors[i], ms=6)
    if A_final is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if A_final[i, j] > 0:
                    ax.plot(P[-1, [i, j], 0], P[-1, [i, j], 1], ":", color="0.5", lw=0.8)
    if goals is not None:
        g = np.asarray(goals)
        ax.plot(g[:, 0], g[:, 1], "x", color="k", ms=8)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    return ax


def series(t, ys, labels, ylabel, title="", logy=False, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for y, lab in zip(ys, labels):
        ax.plot(t, y, label=lab)
    ax.set_xlabel("t, с")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    if labels:
        ax.legend()
    if title:
        ax.set_title(title)
    return ax


def save(path, fig=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    (fig or plt.gcf()).tight_layout()
    (fig or plt.gcf()).savefig(path, dpi=150)
    plt.close(fig or plt.gcf())

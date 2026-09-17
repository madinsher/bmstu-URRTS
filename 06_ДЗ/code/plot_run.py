# -*- coding: utf-8 -*-
"""Рисунки по журналам прогона склада.

    python3 plot_run.py <каталог прогона> [<каталог прогона> …] --out <каталог рисунков>

Строит: траектории роботов (positions.csv), диаграмму жизненного пути заказов
(orders.csv) и, если каталогов несколько, сравнение режимов по метрикам.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_positions(path):
    by = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by[r["robot"]].append((float(r["t"]), float(r["x"]), float(r["y"])))
    return by


def plot_trajectories(run, out):
    src = os.path.join(run, "positions.csv")
    if not os.path.exists(src):
        return None
    by = read_positions(src)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, pts in sorted(by.items()):
        ax.plot([p[1] for p in pts], [p[2] for p in pts], lw=1.0, label=name)
        ax.plot(pts[0][1], pts[0][2], "o", ms=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    ax.set_title("Траектории роботов, %s" % os.path.basename(run))
    path = os.path.join(out, "traj_%s.png" % os.path.basename(run))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_orders(run, out):
    src = os.path.join(run, "orders.csv")
    if not os.path.exists(src):
        return None
    rows = []
    with open(src, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["created"]:
                rows.append(r)
    rows.sort(key=lambda r: float(r["created"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    t0 = float(rows[0]["created"])
    for i, r in enumerate(rows):
        c = float(r["created"]) - t0
        k = float(r["committed"]) - t0 if r["committed"] else None
        d = float(r["delivered"]) - t0 if r["delivered"] else None
        ax.plot([c, k if k is not None else c], [i, i], color="0.7", lw=2)
        if k is not None and d is not None:
            ax.plot([k, d], [i, i], color="tab:blue", lw=2)
        ax.plot(c, i, "k.", ms=4)
        if d is not None:
            ax.plot(d, i, "o", color="tab:green", ms=4)
    ax.set_xlabel("время, с")
    ax.set_ylabel("заказ")
    ax.grid(alpha=0.3)
    ax.set_title("Жизненный путь заказов: ожидание (серое), выполнение (синее)")
    path = os.path.join(out, "orders_%s.png" % os.path.basename(run))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_compare(runs, out):
    data = []
    for run in runs:
        p = os.path.join(run, "result.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                m = json.load(fh)
            m["name"] = os.path.basename(run)
            data.append(m)
    if len(data) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = [d["name"] for d in data]
    for ax, key, label in zip(axes, ["throughput_per_min", "lead_time_mean", "path_conflicts"],
                              ["заказов в минуту", "время доставки, с", "конфликтов путей"]):
        ax.bar(range(len(data)), [d.get(key) or 0 for d in data], color="tab:blue")
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    path = os.path.join(out, "compare.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for run in args.runs:
        for f in (plot_trajectories, plot_orders):
            p = f(run, args.out)
            if p:
                print("рисунок:", p)
    p = plot_compare(args.runs, args.out)
    if p:
        print("рисунок:", p)


if __name__ == "__main__":
    main()

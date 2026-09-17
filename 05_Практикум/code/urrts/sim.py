"""Симулятор группы роботов на плоскости.

Модель робота — кинематический интегратор ṗᵢ = uᵢ с ограничением скорости
|uᵢ| ≤ v_max. Для дифференциального привода (e-puck в Webots) скорость
переводится в линейную и угловую через точку, вынесенную на расстояние l
вперёд от оси колёс (`unicycle_from_velocity`), — так один и тот же закон
управления работает и здесь, и на «настоящем» роботе в Webots.

Связь моделируется графом по дальности с независимой потерей каждого ребра
на каждом шаге (вероятность drop_prob) — это грубая, но честная модель
ненадёжного радиоканала. Отказ робота — он перестаёт двигаться и выпадает
из графа связи.

Закон управления передаётся функцией controller(t, p, A, state) → u, где
p — позиции (N, 2), A — текущая матрица смежности, state — словарь, который
закон может использовать для хранения своего состояния между шагами.
"""
import csv
import os

import numpy as np

from . import graph


class CommModel:
    """Радиоканал: дальность связи и вероятность потери ребра на шаге."""

    def __init__(self, radius=np.inf, drop_prob=0.0, rng=None):
        self.radius = float(radius)
        self.drop_prob = float(drop_prob)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def adjacency(self, p, alive=None):
        A = graph.disk_graph(p, self.radius, alive)
        if self.drop_prob > 0.0:
            n = A.shape[0]
            keep = self.rng.random((n, n)) >= self.drop_prob
            keep = np.triu(keep, 1)
            keep = keep | keep.T
            A = A * keep
        return A


class History:
    """Журнал прогона: время, позиции, скорости, число рёбер графа связи."""

    def __init__(self):
        self.t, self.p, self.u, self.edges = [], [], [], []

    def add(self, t, p, u, A):
        self.t.append(float(t))
        self.p.append(np.array(p, dtype=float))
        self.u.append(np.array(u, dtype=float))
        self.edges.append(int(np.count_nonzero(np.triu(A, 1))))

    @property
    def positions(self):
        return np.array(self.p)

    @property
    def times(self):
        return np.array(self.t)

    def to_csv(self, path):
        """Сохранить журнал в CSV: t, i, x, y, ux, uy."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "robot", "x", "y", "ux", "uy"])
            for t, p, u in zip(self.t, self.p, self.u):
                for i in range(p.shape[0]):
                    w.writerow(["%.3f" % t, i, "%.5f" % p[i, 0], "%.5f" % p[i, 1],
                                "%.5f" % u[i, 0], "%.5f" % u[i, 1]])


def saturate(u, vmax):
    """Ограничить модуль скорости каждого робота значением vmax."""
    u = np.array(u, dtype=float)
    if vmax is None or not np.isfinite(vmax):
        return u
    n = np.linalg.norm(u, axis=1, keepdims=True)
    scale = np.minimum(1.0, vmax / np.maximum(n, 1e-12))
    return u * scale


def simulate(p0, controller, steps, dt=0.05, comm=None, vmax=None,
             failures=None, noise_std=0.0, rng=None, record_every=1, state=None):
    """Прогнать закон управления.

    p0        — начальные позиции (N, 2);
    steps     — число шагов, dt — шаг интегрирования, с;
    comm      — CommModel (по умолчанию связь «все со всеми»);
    failures  — словарь {номер шага: [номера роботов]} — отказы роботов;
    noise_std — СКО аддитивного шума исполнения скорости, м/с.

    Возвращает (History, итоговая маска работоспособных роботов).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    comm = comm if comm is not None else CommModel()
    p = np.array(p0, dtype=float)
    n = p.shape[0]
    alive = np.ones(n, dtype=bool)
    state = {} if state is None else state
    state.setdefault("alive", alive)
    hist = History()
    for k in range(steps):
        if failures and k in failures:
            alive[list(failures[k])] = False
        A = comm.adjacency(p, alive)
        u = np.asarray(controller(k * dt, p.copy(), A, state), dtype=float)
        u = saturate(u, vmax)
        u[~alive] = 0.0
        if noise_std > 0.0:
            u[alive] += rng.normal(0.0, noise_std, size=(int(alive.sum()), 2))
        if k % record_every == 0:
            hist.add(k * dt, p, u, A)
        p = p + dt * u
    hist.add(steps * dt, p, np.zeros_like(p), comm.adjacency(p, alive))
    return hist, alive


def unicycle_from_velocity(theta, u, l=0.03):
    """Перевести желаемую скорость точки (ux, uy) в (v, ω) дифференциального
    привода через точку, вынесенную на l вперёд (почти тождественное
    преобразование, пособие §4.2.1)."""
    c, s = np.cos(theta), np.sin(theta)
    v = c * u[0] + s * u[1]
    w = (-s * u[0] + c * u[1]) / l
    return v, w


def wheel_speeds(v, w, wheel_radius=0.0205, axle=0.052, max_speed=6.28):
    """(v, ω) → угловые скорости левого и правого колёс e-puck, рад/с,
    с пропорциональным сжатием при насыщении."""
    left = (v - w * axle / 2.0) / wheel_radius
    right = (v + w * axle / 2.0) / wheel_radius
    m = max(abs(left), abs(right), 1e-12)
    if m > max_speed:
        left, right = left * max_speed / m, right * max_speed / m
    return left, right

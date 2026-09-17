# -*- coding: utf-8 -*-
"""Контроллер e-puck для практикума УРРТС.

Робот знает только себя и то, что услышал по радио от соседей: позицию и
метку времени (и прогресс плана в режиме mapf). Закон управления берётся из
библиотеки urrts — той же, что проходит модульные тесты. Позиция робота —
от GPS и инерциального блока (идеальная локализация — сознательное
упрощение практикума; ошибки локализации — тема ★-заданий).

Аргументы controllerArgs (вида ключ=значение):
    id=<номер>            номер робота (обязательно)
    mode=formation|coverage|swap|mapf
    radio=<м>             дальность радио (задаётся и в мире — поле range)
    vmax=<м/с>            ограничение скорости (e-puck: не больше 0,12)
    stale=<с>             возраст сообщения, после которого сосед считается «потерянным»
    formation:  shape=hexagon|line|wedge, size=<м>, n=<число роботов>, gain=<1/с>
    coverage:   arena=xmin,xmax,ymin,ymax, rsense=<м>, focus=x,y,sigma (необязательно)
    swap:       goal=x,y, dsafe=<м>, gamma=<1/с>
    mapf:       plan=<путь к JSON плана>
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # каталог code/

import numpy as np  # noqa: E402
from controller import Robot  # noqa: E402

from urrts import sim  # noqa: E402
from urrts import pr1_consensus as pr1  # noqa: E402
from urrts import pr4_mapf as pr4  # noqa: E402
from urrts import pr5_coverage as pr5  # noqa: E402
from urrts import pr6_safety as pr6  # noqa: E402


def parse_args(argv):
    out = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def formation_offsets(shape, n, size):
    if shape == "line":
        return np.c_[(np.arange(n) - (n - 1) / 2) * size, np.zeros(n)]
    if shape == "wedge":
        pts = [(0.0, 0.0)]
        k = 1
        while len(pts) < n:
            pts.append((-k * size, k * size))
            if len(pts) < n:
                pts.append((-k * size, -k * size))
            k += 1
        o = np.array(pts)
        return o - o.mean(axis=0)
    ang = 2 * np.pi * np.arange(n) / n
    return size * np.c_[np.cos(ang), np.sin(ang)]


class Agent:
    def __init__(self):
        self.robot = Robot()
        self.dt_ms = int(self.robot.getBasicTimeStep()) * 2
        self.args = parse_args(sys.argv[1:])
        self.id = int(self.args["id"])
        self.mode = self.args.get("mode", "formation")
        self.vmax = float(self.args.get("vmax", 0.1))
        self.stale = float(self.args.get("stale", 1.0))
        self.gps = self.robot.getDevice("gps")
        self.imu = self.robot.getDevice("imu")
        self.tx = self.robot.getDevice("radio_tx")
        self.rx = self.robot.getDevice("radio_rx")
        self.cmd = self.robot.getDevice("cmd_rx")
        for d in (self.gps, self.imu, self.rx, self.cmd):
            d.enable(self.dt_ms)
        self.left = self.robot.getDevice("left wheel motor")
        self.right = self.robot.getDevice("right wheel motor")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.neigh = {}             # id -> dict(x, y, t, prog)
        self.alive = True
        self.silent = False
        self.hold = np.zeros(1, dtype=int)
        self.u_prev = np.ones((1, 2))
        self.prog = 0
        self.setup_mode()

    # ------------------------------------------------------------------
    def setup_mode(self):
        a = self.args
        if self.mode == "formation":
            n = int(a.get("n", 6))
            self.offsets = formation_offsets(a.get("shape", "hexagon"), n, float(a.get("size", 0.25)))
            self.gain = float(a.get("gain", 1.0))
        elif self.mode == "coverage":
            xmin, xmax, ymin, ymax = [float(v) for v in a.get("arena", "-1,1,-1,1").split(",")]
            self.q, self.dA = pr5.grid_points(xmin, xmax, ymin, ymax, 0.05)
            if "focus" in a:
                fx, fy, fs = [float(v) for v in a["focus"].split(",")]
                self.phi = pr5.gaussian_density(self.q, [fx, fy], fs)
            else:
                self.phi = np.ones(len(self.q))
            self.rsense = float(a.get("rsense", 0.5))
            self.gain = float(a.get("gain", 1.0))
        elif self.mode == "swap":
            self.goal = np.array([float(v) for v in a["goal"].split(",")])
            self.dsafe = float(a.get("dsafe", 0.12))
            self.gamma = float(a.get("gamma", 2.0))
        elif self.mode == "mapf":
            path = a["plan"]
            if not os.path.isabs(path):
                path = os.path.join(HERE, path)
            with open(path, encoding="utf-8") as fh:
                plan = json.load(fh)
            self.cell = float(plan["cell"])
            self.origin = np.array(plan["origin"], dtype=float)
            paths = [[tuple(c) for c in p] for p in plan["paths"]]
            self.actions, self.deps = pr4.action_dependencies(paths)
            self.my_actions = self.actions[self.id]

    # ------------------------------------------------------------------
    def pose(self):
        x, y, _ = self.gps.getValues()
        yaw = self.imu.getRollPitchYaw()[2]
        return np.array([x, y]), yaw

    def listen(self, now):
        while self.cmd.getQueueLength() > 0:
            msg = json.loads(self.cmd.getString())
            self.cmd.nextPacket()
            if msg.get("id") in (self.id, "all"):
                if msg.get("cmd") == "fail":
                    self.alive = False
                elif msg.get("cmd") == "mute":
                    self.silent = True
                elif msg.get("cmd") == "unmute":
                    self.silent = False
        while self.rx.getQueueLength() > 0:
            msg = json.loads(self.rx.getString())
            self.rx.nextPacket()
            if msg["id"] != self.id:
                self.neigh[msg["id"]] = msg

    def broadcast(self, p, now):
        if self.silent:
            return
        self.tx.send(json.dumps({"id": self.id, "x": float(p[0]), "y": float(p[1]),
                                 "t": now, "prog": self.prog}))

    def neighbours(self, now, max_age=None):
        max_age = self.stale if max_age is None else max_age
        return [m for m in self.neigh.values() if now - m["t"] <= max_age]

    # ------------------------------------------------------------------
    def velocity(self, p, now):
        nb = self.neighbours(now)
        P = np.array([p] + [[m["x"], m["y"]] for m in nb])
        if self.mode == "formation":
            ids = [self.id] + [m["id"] for m in nb]
            A = np.zeros((len(ids), len(ids)))
            A[0, 1:] = A[1:, 0] = 1.0
            return self.gain * pr1.formation_velocity(P, A, self.offsets[ids], 1.0)[0]
        if self.mode == "coverage":
            new = pr5.limited_lloyd_step(P, self.q, self.phi, self.dA, self.rsense)
            return self.gain * (new[0] - p)
        if self.mode == "swap":
            u_nom, self.hold = pr6.unstuck_nominal(p[None, :], self.goal[None, :], self.u_prev,
                                                   self.hold, 1.0, self.vmax, goal_tol=0.03,
                                                   stuck_speed=0.01)
            cons = pr6.speed_polygon(self.vmax)
            for m in self.neigh.values():
                age = now - m["t"]
                q = np.array([m["x"], m["y"]])
                if np.linalg.norm(p - q) > 0.6:
                    continue
                if age <= 0.2:
                    cons.append(pr6.pair_constraint(p, q, self.dsafe, self.gamma))
                elif age <= 10.0:
                    cons.append(pr6.stale_constraint(p, q, age, self.vmax, self.dsafe, self.gamma))
            return pr6.safe_velocity(u_nom[0], cons)
        if self.mode == "mapf":
            if self.prog >= len(self.my_actions):
                target = self.cell_xy(self.my_actions[-1][1]) if self.my_actions else p
            else:
                done = set()
                for m in self.neigh.values():
                    done |= {(m["id"], k) for k in range(m.get("prog", 0))}
                done |= {(self.id, k) for k in range(self.prog)}
                _, w, _ = self.my_actions[self.prog]
                target = self.cell_xy(w)
                if not self.deps[(self.id, self.prog)] <= done:
                    frm = self.my_actions[self.prog][0]
                    target = self.cell_xy(frm)          # ждать в текущей клетке
                elif np.linalg.norm(target - p) < 0.02:
                    self.prog += 1
            e = target - p
            return 1.5 * e
        raise ValueError("неизвестный режим %s" % self.mode)

    def cell_xy(self, cell):
        r, c = cell
        return self.origin + np.array([c * self.cell, -r * self.cell])

    # ------------------------------------------------------------------
    def run(self):
        while self.robot.step(self.dt_ms) != -1:
            now = self.robot.getTime()
            p, yaw = self.pose()
            self.listen(now)
            if not self.alive:
                self.left.setVelocity(0.0)
                self.right.setVelocity(0.0)
                continue
            u = np.asarray(self.velocity(p, now), dtype=float)
            n = np.linalg.norm(u)
            if n > self.vmax:
                u = u * self.vmax / n
            self.u_prev = u[None, :]
            v, w = sim.unicycle_from_velocity(yaw, u, l=0.03)
            left, right = sim.wheel_speeds(v, w)
            self.left.setVelocity(left)
            self.right.setVelocity(right)
            self.broadcast(p, now)


if __name__ == "__main__":
    Agent().run()

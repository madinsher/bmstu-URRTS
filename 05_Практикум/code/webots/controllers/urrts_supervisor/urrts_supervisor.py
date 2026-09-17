# -*- coding: utf-8 -*-
"""Супервизор сцены: журнал позиций, метрики прогона, внесение отказов.

Роботы — узлы с DEF R0, R1, … Аргументы controllerArgs:
    n=<число роботов>          обязательно
    name=<имя прогона>         имя файлов журнала в каталоге logs/
    duration=<с>               длительность; по окончании — итог и выход из Webots
    fail=<t>:<id>[;<t>:<id>]   отказ робота в момент t (остановка и молчание)
    mute=<t0>-<t1>:<id>        робот молчит в эфире с t0 до t1 (потеря связи)
    dsafe=<м>                  порог «сближения» для подсчёта нарушений
    goals=x,y;x,y;…            цели роботов (для времени прибытия)
    shape=…, size=…            формация (для ошибки формы) — как у роботов
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, os.path.join(HERE, "..", "urrts_epuck"))

import numpy as np  # noqa: E402
from controller import Supervisor  # noqa: E402

from urrts import pr1_consensus as pr1  # noqa: E402


def parse_args(argv):
    out = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    sup = Supervisor()
    dt = int(sup.getBasicTimeStep())
    args = parse_args(sys.argv[1:])
    n = int(args["n"])
    name = args.get("name", "run")
    duration = float(args.get("duration", 60))
    dsafe = float(args.get("dsafe", 0.074))
    logs = os.path.abspath(os.path.join(HERE, "..", "..", "logs"))
    os.makedirs(logs, exist_ok=True)
    nodes = [sup.getFromDef("R%d" % i) for i in range(n)]
    tx = sup.getDevice("cmd_tx")

    events = []
    for part in filter(None, args.get("fail", "").split(";")):
        t, i = part.split(":")
        events.append((float(t), {"cmd": "fail", "id": int(i)}))
    for part in filter(None, args.get("mute", "").split(";")):
        span, i = part.split(":")
        t0, t1 = [float(v) for v in span.split("-")]
        events += [(t0, {"cmd": "mute", "id": int(i)}), (t1, {"cmd": "unmute", "id": int(i)})]
    events.sort(key=lambda e: e[0])

    goals = None
    if "goals" in args:
        goals = np.array([[float(v) for v in g.split(",")] for g in args["goals"].split(";")])
    offsets = None
    if "shape" in args:
        from urrts_epuck import formation_offsets
        offsets = formation_offsets(args["shape"], n, float(args.get("size", 0.25)))

    fh = open(os.path.join(logs, name + ".csv"), "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(["t", "robot", "x", "y"])
    dmin, violations, arrive = math.inf, 0, None
    last_log = -1.0
    close_prev = set()
    ferr = []
    while sup.step(dt) != -1:
        t = sup.getTime()
        while events and events[0][0] <= t:
            _, msg = events.pop(0)
            tx.send(json.dumps(msg))
            print("[%.2f] внесено: %s" % (t, msg))
        P = np.array([nd.getPosition()[:2] for nd in nodes])
        D = np.linalg.norm(P[:, None] - P[None], axis=2) + 1e9 * np.eye(n)
        dmin = min(dmin, float(D.min()))
        close = {(i, j) for i in range(n) for j in range(i + 1, n) if D[i, j] < dsafe}
        violations += len(close - close_prev)
        close_prev = close
        if goals is not None and arrive is None and np.linalg.norm(P - goals, axis=1).max() < 0.03:
            arrive = t
        if t - last_log >= 0.1 - 1e-9:
            last_log = t
            for i in range(n):
                w.writerow(["%.2f" % t, i, "%.4f" % P[i, 0], "%.4f" % P[i, 1]])
            if offsets is not None:
                ferr.append((t, pr1.formation_error(P, offsets)))
        if t >= duration:
            break
    fh.close()
    summary = {"name": name, "n": n, "duration": duration, "min_distance": dmin,
               "close_approaches": violations, "arrival_time": arrive}
    if ferr:
        summary["formation_error_final"] = ferr[-1][1]
        below = [tt for tt, e in ferr if e < 0.02]
        summary["formation_time_2cm"] = below[0] if below else None
    with open(os.path.join(logs, name + ".json"), "w", encoding="utf-8") as js:
        json.dump(summary, js, ensure_ascii=False, indent=2)
    print("ИТОГ", json.dumps(summary, ensure_ascii=False))
    if args.get("quit", "1") == "1":
        sup.simulationQuit(0)


if __name__ == "__main__":
    main()

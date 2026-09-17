# -*- coding: utf-8 -*-
"""Сцены Webots практикума: генерация миров и пакетные прогоны.

    python webots/run.py list                        список сценариев
    python webots/run.py world <сценарий> [--seed N] записать мир (открыть в Webots вручную)
    python webots/run.py run <сценарий> [--seed N]   прогнать без отрисовки, журнал — в reports/webots
    python webots/run.py show <сценарий>             открыть мир в Webots с отрисовкой

Webots R2025a; путь к нему — переменная WEBOTS_HOME или C:\\Program Files\\Webots.
Webots не запускает контроллеры из путей с кириллицей и пробелами, поэтому
каталог code копируется во временную папку (URRTS_WORK или %TEMP%/urrts_webots),
прогон идёт там, журналы копируются обратно.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import numpy as np  # noqa: E402

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")

HEADER = """#VRML_SIM R2025a utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/floors/protos/Floor.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/gctronic/e-puck/protos/E-puck.proto"

WorldInfo {{
  info [ "Практикум УРРТС: {title}" ]
  title "{name}"
  basicTimeStep 32
}}
Viewpoint {{
  orientation 0 1 0 1.5708
  position {cx} {cy} {height}
}}
TexturedBackground {{
}}
TexturedBackgroundLight {{
}}
Floor {{
  size {fx} {fy}
  tileSize 0.25 0.25
}}
"""

ROBOT = """DEF R{i} E-puck {{
  translation {x:.4f} {y:.4f} 0
  rotation 0 0 1 {yaw:.4f}
  name "e-puck {i}"
  controller "urrts_epuck"
  controllerArgs [ {args} ]
  turretSlot [
    GPS {{ name "gps" }}
    InertialUnit {{ name "imu" }}
    Emitter {{ name "radio_tx" type "radio" range {radio} channel 5 }}
    Receiver {{ name "radio_rx" type "radio" channel 5 }}
    Receiver {{ name "cmd_rx" type "radio" channel 6 }}
  ]
}}
"""

SUPERVISOR = """Robot {{
  name "supervisor"
  controller "urrts_supervisor"
  controllerArgs [ {args} ]
  supervisor TRUE
  children [
    Emitter {{ name "cmd_tx" type "radio" range -1 channel 6 }}
  ]
}}
"""

SHELF = """Solid {{
  translation {x:.4f} {y:.4f} 0.05
  children [ Shape {{ appearance PBRAppearance {{ baseColor 0.55 0.4 0.25 roughness 0.9 metalness 0 }} geometry Box {{ size {s:.3f} {s:.3f} 0.1 }} }} ]
  name "shelf {k}"
  boundingObject Box {{ size {s:.3f} {s:.3f} 0.1 }}
}}
"""


def q(*items):
    return " ".join('"%s"' % s for s in items)


# ---------------------------------------------------------------------------
# Сценарии
# ---------------------------------------------------------------------------

def sc_pr1_formation(rng, fail=False):
    n = 6
    from urrts import graph

    def bad(p):   # слишком близко или начальный граф связи (с запасом 0,1 м) несвязен
        return (min(np.linalg.norm(p[i] - p[j]) for i in range(n) for j in range(i)) < 0.15
                or not graph.is_connected(graph.disk_graph(p, 0.6)))
    p = rng.uniform(-0.6, 0.6, (n, 2))
    while bad(p):
        p = rng.uniform(-0.6, 0.6, (n, 2))
    robots = [(p[i], rng.uniform(-np.pi, np.pi),
               q("id=%d" % i, "mode=formation", "n=%d" % n, "shape=hexagon", "size=0.2",
                 "radio=0.7", "vmax=0.1", "stale=0.5")) for i in range(n)]
    sargs = ["n=%d" % n, "name=pr1_formation%s" % ("_fail" if fail else ""), "duration=60",
             "shape=hexagon", "size=0.2"]
    if fail:
        sargs.append("fail=20:2")
    return dict(title="формация «шестиугольник» по радио", robots=robots, radio=0.7,
                sup=q(*sargs), floor=(2.5, 2.5), center=(0, 0))


def sc_pr5_coverage(rng):
    n = 8
    p = np.c_[rng.uniform(-0.95, -0.55, n), rng.uniform(-0.95, -0.55, n)]
    while min(np.linalg.norm(p[i] - p[j]) for i in range(n) for j in range(i)) < 0.09:
        p = np.c_[rng.uniform(-0.95, -0.55, n), rng.uniform(-0.95, -0.55, n)]
    robots = [(p[i], rng.uniform(-np.pi, np.pi),
               q("id=%d" % i, "mode=coverage", "arena=-1,1,-1,1", "rsense=0.5", "focus=0.4,0.4,0.35",
                 "radio=1.0", "vmax=0.1", "stale=0.5", "gain=1.0")) for i in range(n)]
    return dict(title="покрытие области по Ллойду с ограниченной дальностью", robots=robots, radio=1.0,
                sup=q("n=%d" % n, "name=pr5_coverage", "duration=90"), floor=(2.5, 2.5), center=(0, 0))


def sc_pr6_swap(rng, mute=False):
    n = 8
    from urrts import pr6_safety as pr6
    p, goals = pr6.antipodal(n, 0.6)
    p = p + rng.uniform(-0.02, 0.02, p.shape)          # зерно бригады слегка сдвигает старты
    robots = [(p[i], float(np.arctan2(goals[i, 1] - p[i, 1], goals[i, 0] - p[i, 0])),
               q("id=%d" % i, "mode=swap", "goal=%.4f,%.4f" % tuple(goals[i]), "dsafe=0.12", "gamma=2.0",
                 "radio=1.0", "vmax=0.1")) for i in range(n)]
    sargs = ["n=%d" % n, "name=pr6_swap%s" % ("_mute" if mute else ""), "duration=90", "dsafe=0.074",
             "goals=" + ";".join("%.4f,%.4f" % tuple(g) for g in goals)]
    if mute:
        sargs.append("mute=3-40:0")
    return dict(title="обмен местами с барьерными функциями", robots=robots, radio=1.0,
                sup=q(*sargs), floor=(2.0, 2.0), center=(0, 0))


def sc_pr4_warehouse(rng):
    from urrts import pr4_mapf as pr4
    grid = pr4.load_map(pr4.WAREHOUSE)
    cell = 0.25
    rows, cols = grid.shape
    origin = np.array([-(cols - 1) * cell / 2, (rows - 1) * cell / 2])
    free = [tuple(int(v) for v in c) for c in np.argwhere(~grid)]
    while True:
        idx = rng.choice(len(free), size=12, replace=False)
        starts, goals = [free[i] for i in idx[:6]], [free[i] for i in idx[6:]]
        paths, _ = pr4.cbs(grid, starts, goals, max_nodes=3000)
        if paths is not None:
            break
    os.makedirs(os.path.join(HERE, "plans"), exist_ok=True)
    with open(os.path.join(HERE, "plans", "pr4_warehouse.json"), "w", encoding="utf-8") as fh:
        json.dump({"cell": cell, "origin": origin.tolist(), "paths": [[list(c) for c in p] for p in paths]}, fh)

    def xy(c):
        return origin + np.array([c[1] * cell, -c[0] * cell])

    robots = [(xy(starts[i]), 0.0, q("id=%d" % i, "mode=mapf", "plan=../../plans/pr4_warehouse.json",
                                    "radio=5.0", "vmax=0.1")) for i in range(6)]
    shelves = [xy(tuple(c)) for c in np.argwhere(grid)]
    goals_xy = [xy(g) for g in goals]
    return dict(title="склад: исполнение плана CBS по графу зависимостей", robots=robots, radio=5.0,
                sup=q("n=6", "name=pr4_warehouse", "duration=150", "dsafe=0.074",
                      "goals=" + ";".join("%.4f,%.4f" % tuple(g) for g in goals_xy)),
                floor=(3.0, 2.25), center=(0, 0), shelves=shelves, cell=cell)


SCENARIOS = {
    "pr1_formation": lambda rng: sc_pr1_formation(rng),
    "pr1_formation_fail": lambda rng: sc_pr1_formation(rng, fail=True),
    "pr4_warehouse": sc_pr4_warehouse,
    "pr5_coverage": sc_pr5_coverage,
    "pr6_swap": lambda rng: sc_pr6_swap(rng),
    "pr6_swap_mute": lambda rng: sc_pr6_swap(rng, mute=True),
}


def write_world(name, seed):
    sc = SCENARIOS[name](np.random.default_rng(seed))
    fx, fy = sc["floor"]
    text = HEADER.format(title=sc["title"], name=name, cx=sc["center"][0], cy=sc["center"][1],
                         height=1.25 * max(fx, fy), fx=fx, fy=fy)
    for k, s in enumerate(sc.get("shelves", [])):
        text += SHELF.format(x=s[0], y=s[1], s=0.8 * sc["cell"], k=k)
    for i, (p, yaw, args) in enumerate(sc["robots"]):
        text += ROBOT.format(i=i, x=p[0], y=p[1], yaw=yaw, args=args, radio=sc["radio"])
    text += SUPERVISOR.format(args=sc["sup"])
    os.makedirs(os.path.join(HERE, "worlds"), exist_ok=True)
    path = os.path.join(HERE, "worlds", name + ".wbt")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def work_copy():
    """Копия каталога code в путь без кириллицы и пробелов."""
    dst = os.environ.get("URRTS_WORK", os.path.join(tempfile.gettempdir(), "urrts_webots"))
    try:
        dst.encode("ascii")
    except UnicodeEncodeError:
        dst = r"C:\urrts_webots"
    if " " in dst:
        dst = r"C:\urrts_webots"
    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(CODE, dst, ignore=shutil.ignore_patterns("reports", "__pycache__", "*.pyc", ".git"))
    for ctrl in ("urrts_epuck", "urrts_supervisor"):
        with open(os.path.join(dst, "webots", "controllers", ctrl, "runtime.ini"), "w", encoding="utf-8") as fh:
            fh.write("[python]\nCOMMAND = %s\n" % sys.executable)
    return dst


def webots_exe():
    for cand in (os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin", "webots.exe"),
                 os.path.join(WEBOTS_HOME, "webots"), "webots"):
        if os.path.exists(cand) or cand == "webots":
            return cand


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "world", "run", "show"])
    ap.add_argument("scenario", nargs="?")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "list":
        for k in SCENARIOS:
            print(k)
        return 0
    if args.scenario not in SCENARIOS:
        print("неизвестный сценарий; см. python webots/run.py list")
        return 2
    if args.cmd == "world":
        print("мир записан:", write_world(args.scenario, args.seed))
        return 0
    write_world(args.scenario, args.seed)
    work = work_copy()
    world = os.path.join(work, "webots", "worlds", args.scenario + ".wbt")
    exe = webots_exe()
    if args.cmd == "show":
        subprocess.Popen([exe, world])
        print("Webots запущен:", world)
        return 0
    cmd = [exe, "--batch", "--mode=fast", "--no-rendering", "--minimize", "--stdout", "--stderr", world]
    print("прогон:", args.scenario, "(без отрисовки)")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    out = (proc.stdout or "") + (proc.stderr or "")
    reports = os.path.join(CODE, "reports", "webots")
    os.makedirs(reports, exist_ok=True)
    for ext in (".csv", ".json"):
        src = os.path.join(work, "webots", "logs", args.scenario + ext)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(reports, args.scenario + ext))
    with open(os.path.join(reports, args.scenario + ".log"), "w", encoding="utf-8") as fh:
        fh.write(out)
    summary = os.path.join(reports, args.scenario + ".json")
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as fh:
            print(json.dumps(json.load(fh), ensure_ascii=False, indent=2))
        return 0
    print("итог не получен; вывод Webots — в", os.path.join(reports, args.scenario + ".log"))
    print(out[-3000:])
    return 1


if __name__ == "__main__":
    sys.exit(main())

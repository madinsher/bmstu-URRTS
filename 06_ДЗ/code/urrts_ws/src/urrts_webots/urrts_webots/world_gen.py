"""Мир Webots, карта Nav2 и URDF роботов — из того же config/warehouse.yaml, что и симулятор.

    python3 -m urrts_webots.world_gen <warehouse.yaml> <каталог вывода>

Координаты: центр клетки (строка r, столбец c) — точка (c·cell, −r·cell),
как в urrts_sim.warehouse. Стеллаж — ящик в клетку высотой 0,4 м.
"""
import math
import os
import sys

import yaml

PROTO = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects"
RES = 0.05          # разрешение карты Nav2, м/пиксель


def load(cfg_path):
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    rows = cfg["map"].strip("\n").split("\n")
    return cfg, [[ch == "#" for ch in r] for r in rows]


def world(cfg, grid):
    cell = float(cfg["cell"])
    R, C = len(grid), len(grid[0])
    cx, cy = (C - 1) * cell / 2, -(R - 1) * cell / 2
    out = ['#VRML_SIM R2025a utf8', '',
           'EXTERNPROTO "%s/objects/backgrounds/protos/TexturedBackground.proto"' % PROTO,
           'EXTERNPROTO "%s/objects/backgrounds/protos/TexturedBackgroundLight.proto"' % PROTO,
           'EXTERNPROTO "%s/objects/floors/protos/Floor.proto"' % PROTO,
           'EXTERNPROTO "%s/robots/robotis/turtlebot/protos/TurtleBot3Burger.proto"' % PROTO,
           '',
           'WorldInfo {', '  info [ "УРРТС, ДЗ: склад" ]', '  basicTimeStep 32', '}',
           'Viewpoint {', '  orientation 0 1 0 1.5708',
           '  position %.3f %.3f %.3f' % (cx, cy, 1.1 * max(R, C) * cell), '}',
           'TexturedBackground {', '}', 'TexturedBackgroundLight {', '}',
           'Floor {', '  translation %.3f %.3f 0' % (cx, cy),
           '  size %.3f %.3f' % (C * cell + 1.0, R * cell + 1.0), '  tileSize 0.5 0.5', '}']
    k = 0
    # стеллажи и стены по периметру
    boxes = [(r, c) for r in range(R) for c in range(C) if grid[r][c]]
    boxes += [(-1, c) for c in range(-1, C + 1)] + [(R, c) for c in range(-1, C + 1)]
    boxes += [(r, -1) for r in range(R)] + [(r, C) for r in range(R)]
    for r, c in boxes:
        x, y = c * cell, -r * cell
        wall = r < 0 or c < 0 or r >= R or c >= C
        color = "0.6 0.6 0.6" if wall else "0.55 0.40 0.25"
        h = 0.25 if wall else 0.4
        out += ['Solid {', '  translation %.3f %.3f %.3f' % (x, y, h / 2),
                '  children [ Shape { appearance PBRAppearance { baseColor %s roughness 0.9 metalness 0 } '
                'geometry Box { size %.3f %.3f %.3f } } ]' % (color, cell, cell, h),
                '  name "block %d"' % k,
                '  boundingObject Box { size %.3f %.3f %.3f }' % (cell, cell, h), '}']
        k += 1
    rc = cfg["robots"]
    for name, home in zip(rc["names"], rc["homes"]):
        r, c = cfg["stations"][home]
        out += ['TurtleBot3Burger {', '  translation %.3f %.3f 0' % (c * cell, -r * cell),
                '  rotation 0 0 1 %.4f' % (math.pi / 2), '  name "%s"' % name,
                '  controller "<extern>"', '  supervisor TRUE', '}']
    return "\n".join(out) + "\n"


def nav_map(cfg, grid):
    """PGM и YAML для nav2_map_server: 0 — занято (чёрное), 254 — свободно."""
    cell = float(cfg["cell"])
    R, C = len(grid), len(grid[0])
    px = int(round(cell / RES))
    W, H = C * px, R * px
    data = bytearray()
    for i in range(H):
        r = i // px
        for j in range(W):
            c = j // px
            data.append(0 if grid[r][c] else 254)
    pgm = b"P5\n%d %d\n255\n" % (W, H) + bytes(data)
    origin = [-cell / 2, -(R - 1) * cell - cell / 2, 0.0]
    meta = {"image": "warehouse_map.pgm", "mode": "trinary", "resolution": RES, "origin": origin,
            "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.25}
    return pgm, meta


def urdf(name):
    return """<?xml version="1.0"?>
<robot name="%s">
  <webots>
    <device reference="LDS-01" type="Lidar">
      <ros>
        <enabled>true</enabled>
        <updateRate>5</updateRate>
        <topicName>scan</topicName>
        <alwaysOn>true</alwaysOn>
        <frameName>LDS-01</frameName>
      </ros>
    </device>
    <plugin type="urrts_webots.tb3_driver.Tb3Driver"/>
  </webots>
</robot>
""" % name


def generate(cfg_path, out_dir):
    cfg, grid = load(cfg_path)
    os.makedirs(out_dir, exist_ok=True)
    paths = {"world": os.path.join(out_dir, "warehouse.wbt"), "map": os.path.join(out_dir, "warehouse_map.yaml")}
    with open(paths["world"], "w", encoding="utf-8") as fh:
        fh.write(world(cfg, grid))
    pgm, meta = nav_map(cfg, grid)
    with open(os.path.join(out_dir, "warehouse_map.pgm"), "wb") as fh:
        fh.write(pgm)
    with open(paths["map"], "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh)
    for name in cfg["robots"]["names"]:
        p = os.path.join(out_dir, "%s.urdf" % name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(urdf(name))
        paths[name] = p
    return cfg, paths


if __name__ == "__main__":
    print(generate(sys.argv[1], sys.argv[2])[1])

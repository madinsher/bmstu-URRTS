"""Модель склада без ROS: карта, станции, поиск пути по клеткам, время проезда.

Используется симулятором (urrts_sim), агентами CBBA (оценка времени маршрута)
и проверками. Соглашения о координатах — в config/warehouse.yaml.
"""
import heapq
import math

import yaml

MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Warehouse:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cell = float(cfg["cell"])
        rows = [r for r in cfg["map"].strip("\n").split("\n")]
        self.grid = [[ch == "#" for ch in r] for r in rows]
        self.rows, self.cols = len(self.grid), len(self.grid[0])
        self.stations = {k: tuple(v) for k, v in cfg["stations"].items()}
        for name, (r, c) in self.stations.items():
            if self.grid[r][c]:
                raise ValueError("станция %s стоит на стеллаже" % name)
        self._dist_cache = {}

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    # --- координаты -------------------------------------------------------
    def cell_xy(self, cell):
        r, c = cell
        return (c * self.cell, -r * self.cell)

    def xy_cell(self, x, y):
        c = int(round(x / self.cell))
        r = int(round(-y / self.cell))
        return (min(max(r, 0), self.rows - 1), min(max(c, 0), self.cols - 1))

    def station_xy(self, name):
        return self.cell_xy(self.stations[name])

    def free(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols and not self.grid[r][c]

    # --- поиск пути ---------------------------------------------------------
    def path(self, start, goal):
        """Кратчайший путь по клеткам (A*, 4-связность) или None."""
        start, goal = tuple(start), tuple(goal)
        h = lambda a: abs(a[0] - goal[0]) + abs(a[1] - goal[1])
        openq = [(h(start), 0, start)]
        parent = {start: None}
        g = {start: 0}
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur == goal:
                out = []
                while cur is not None:
                    out.append(cur)
                    cur = parent[cur]
                return out[::-1]
            if gc > g[cur]:
                continue
            for dr, dc in MOVES:
                nxt = (cur[0] + dr, cur[1] + dc)
                if self.free(nxt) and gc + 1 < g.get(nxt, math.inf):
                    g[nxt] = gc + 1
                    parent[nxt] = cur
                    heapq.heappush(openq, (gc + 1 + h(nxt), gc + 1, nxt))
        return None

    def distance(self, a, b):
        """Длина кратчайшего пути между клетками, м (кэшируется)."""
        key = (tuple(a), tuple(b))
        if key not in self._dist_cache:
            p = self.path(a, b)
            self._dist_cache[key] = math.inf if p is None else (len(p) - 1) * self.cell
        return self._dist_cache[key]

    def station_distance(self, a, b):
        return self.distance(self.stations[a], self.stations[b])

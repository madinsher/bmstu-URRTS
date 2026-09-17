"""Тесты резервирования клеток (без ROS): python3 -m pytest src/urrts_fleet/test -q"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "urrts_sim"))

import pytest  # noqa: E402

from urrts_fleet.traffic_core import Traffic  # noqa: E402
from urrts_sim.warehouse import Warehouse  # noqa: E402

CONFIG = os.path.join(HERE, "..", "..", "urrts_bringup", "config", "warehouse.yaml")


@pytest.fixture()
def wh():
    return Warehouse.from_file(CONFIG)


def st(stamp, holds, want=None, since=None):
    return {"stamp": stamp, "holds": [list(h) for h in holds], "want": list(want) if want else None,
            "want_since": since}


# --- этап 3, шаг 1 -------------------------------------------------------------

def test_needs_own_want_and_settle(wh):
    t = Traffic("r1", wh, settle=0.3)
    assert not t.may_enter((0, 1), {}, 1.0)
    t.want, t.want_since = (0, 1), 1.0
    assert not t.may_enter((0, 1), {}, 1.2)
    assert t.may_enter((0, 1), {}, 1.31)
    assert not t.may_enter((0, 2), {}, 1.31)


def test_held_by_neighbour(wh):
    t = Traffic("r1", wh, settle=0.3)
    t.want, t.want_since = (0, 1), 0.0
    assert not t.may_enter((0, 1), {"r2": st(1.0, [(0, 1)])}, 1.0)


def test_priority_by_time_then_name(wh):
    t = Traffic("r2", wh, settle=0.3)
    t.want, t.want_since = (0, 1), 1.0
    assert not t.may_enter((0, 1), {"r3": st(1.5, [(0, 2)], (0, 1), 0.9)}, 1.5)
    assert t.may_enter((0, 1), {"r3": st(1.5, [(0, 2)], (0, 1), 1.1)}, 1.5)
    assert not t.may_enter((0, 1), {"r1": st(1.5, [(0, 0)], (0, 1), 1.0)}, 1.5)
    assert t.may_enter((0, 1), {"r4": st(1.5, [(0, 2)], (0, 1), 1.0)}, 1.5)


def test_two_robots_never_reserve_same_cell(wh):
    a, b = Traffic("r1", wh, settle=0.2), Traffic("r2", wh, settle=0.2)
    assert a.set_goal((0, 0), (0, 4)) and b.set_goal((0, 4), (0, 0))
    now = 0.0
    for _ in range(400):
        now += 0.1
        sa, sb = a.export_state(now - 0.05), b.export_state(now - 0.05)   # задержка радио
        a.step({"r2": sb}, now)
        b.step({"r1": sa}, now)
        assert not set(a.holds()) & set(b.holds())
        for tr in (a, b):
            if tr.next_target() is not None:
                tr.arrived()
        if a.done() and b.done():
            break
    assert a.done() and b.done()


# --- этап 3, шаг 2 -------------------------------------------------------------

def test_replan_around_blocker(wh):
    t = Traffic("r1", wh, settle=0.1, wait_limit=1.0)
    assert t.set_goal((0, 0), (0, 6))
    blocker = {"r2": st(0.0, [(0, 3)])}
    changed = t.replan_around(t.fresh(blocker, 0.0), 0.0)
    assert changed and (0, 3) not in t.route and t.route[-1] == (0, 6)
    assert wh.grid[0][3] is False                   # сетка восстановлена


def test_no_detour_keeps_route(wh):
    t = Traffic("r1", wh)
    assert t.set_goal((3, 0), (4, 0))
    blocker = {"r2": st(5.0, [(3, 1), (2, 0)])}     # единственный другой выход занят, но цель рядом
    before = list(t.route)
    t.replan_around(t.fresh(blocker, 5.0), 5.0)
    assert t.route == before or t.route[-1] == (4, 0)

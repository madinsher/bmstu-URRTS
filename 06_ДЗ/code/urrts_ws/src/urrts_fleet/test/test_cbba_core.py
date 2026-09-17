"""Тесты ядра CBBA (без ROS): python3 -m pytest src/urrts_fleet/test -q"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "urrts_sim"))

import pytest  # noqa: E402

from urrts_fleet.cbba_core import CbbaAgent, Order  # noqa: E402
from urrts_sim.warehouse import Warehouse  # noqa: E402

CONFIG = os.path.join(HERE, "..", "..", "urrts_bringup", "config", "warehouse.yaml")


@pytest.fixture(scope="module")
def wh():
    return Warehouse.from_file(CONFIG)


def agent(wh, name, station, cap=3):
    a = CbbaAgent(name, wh, capacity=cap, discount=0.98, speed=0.3, handle_time=3.0)
    a.start_station = station
    return a


def orders():
    return [Order("o1", "P1", "D1", 10), Order("o2", "P3", "D3", 10), Order("o3", "P2", "D2", 10)]


# --- этап 2, шаг 1 -------------------------------------------------------------

def test_bundle_respects_capacity_and_bids(wh):
    a = agent(wh, "r1", "H1", cap=2)
    for o in orders():
        a.add_order(o)
    added = a.build_bundle()
    assert len(added) == 2 and a.bundle == added
    for oid in a.bundle:
        assert a.z[oid] == "r1" and a.y[oid] > 0
    assert sorted(a.path) == sorted(a.bundle)


def test_bundle_skips_committed_and_outbid(wh):
    a = agent(wh, "r1", "H1")
    for o in orders():
        a.add_order(o)
    a.committed.add("o1")
    a.y["o2"], a.z["o2"] = 1e6, "r9"
    a.build_bundle()
    assert "o1" not in a.bundle and "o2" not in a.bundle


def test_marginal_gain_equals_score_difference(wh):
    a = agent(wh, "r1", "P1", cap=1)
    o = Order("o1", "P1", "D1", 10)
    a.add_order(o)
    a.build_bundle()
    t = a.order_time("P1", o)
    assert a.y["o1"] == pytest.approx(10 * 0.98 ** t)


# --- этап 2, шаг 2 -------------------------------------------------------------

def run_pair(wh, rounds=20):
    a, b = agent(wh, "r1", "H1"), agent(wh, "r2", "H4")
    for ag in (a, b):
        for o in orders():
            ag.add_order(o)
    t = 0.0
    for _ in range(rounds):
        t += 0.5
        for ag in (a, b):
            ag.build_bundle()
        sa, sb = a.export_state(t), b.export_state(t)
        ca = a.consensus({"r2": sb}, t)
        cb = b.consensus({"r1": sa}, t)
        if not ca and not cb:
            break
    return a, b


def test_pair_converges_conflict_free(wh):
    a, b = run_pair(wh)
    assert not set(a.bundle) & set(b.bundle)
    for oid in set(a.bundle) | set(b.bundle):
        assert a.z[oid] == b.z[oid]
    assert len(a.bundle) + len(b.bundle) == 3


def test_stale_neighbour_ignored(wh):
    a = agent(wh, "r1", "H1")
    for o in orders():
        a.add_order(o)
    a.build_bundle()
    before = list(a.bundle)
    fake = {"stamp": 0.0, "y": {o: 1e6 for o in before}, "z": {o: "r0" for o in before}, "committed": []}
    a.consensus({"r0": fake}, now=10.0, stale_after=3.0)
    assert a.bundle == before
    a.consensus({"r0": fake}, now=1.0, stale_after=3.0)
    assert a.bundle == []


def test_committed_by_neighbour_removed(wh):
    a = agent(wh, "r1", "H1")
    for o in orders():
        a.add_order(o)
    a.build_bundle()
    victim = a.bundle[0]
    st = {"stamp": 1.0, "y": {}, "z": {}, "committed": [victim]}
    a.consensus({"r2": st}, now=1.0)
    assert victim not in a.orders and victim not in a.bundle and victim in a.committed


# --- этап 4, шаг 1 -------------------------------------------------------------

def test_forget_silent_winner(wh):
    a = agent(wh, "r1", "H1")
    for o in orders():
        a.add_order(o)
    a.y["o1"], a.z["o1"] = 50.0, "r2"
    assert a.forget_silent_winners(0.0, lost_after=6.0) == []
    a.last_heard["r2"] = 1.0
    assert a.forget_silent_winners(5.0, lost_after=6.0) == []
    assert a.forget_silent_winners(8.0, lost_after=6.0) == ["o1"]
    assert a.z["o1"] == "" and a.y["o1"] == 0.0

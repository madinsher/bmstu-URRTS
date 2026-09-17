"""Тесты автомата исполнителя (без ROS): python3 -m pytest src/urrts_fleet/test -q"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from urrts_fleet.executor_core import Executor  # noqa: E402


def new():
    return Executor("H1", max_retries=1)


def test_nominal_cycle():
    e = new()
    assert e.assign("o1", "P1", "D1") == [("goto", "P1")] and e.state == "TO_PICKUP"
    assert e.step(("arrived", None)) == [("handle", "pick")] and e.state == "LOADING"
    assert e.step(("handled", None)) == [("goto", "D1")] and e.state == "TO_DROP"
    assert e.step(("arrived", None)) == [("handle", "drop")] and e.state == "UNLOADING"
    assert e.step(("handled", None)) == [("delivered", "o1")] and e.state == "IDLE"
    assert e.order is None


def test_busy_rejects_new_order():
    e = new()
    e.assign("o1", "P1", "D1")
    assert e.assign("o2", "P2", "D2") == []
    assert e.order == "o1" and e.pickup == "P1" and e.drop == "D1"


def test_nav_failed_to_pickup_retry_then_release():
    e = new()
    e.assign("o1", "P1", "D1")
    assert e.step(("nav_failed", None)) == [("goto", "P1")]
    assert e.step(("nav_failed", None)) == [("release", "o1")]
    assert e.state == "IDLE" and e.order is None


def test_nav_failed_with_load_retries_forever():
    e = new()
    e.assign("o1", "P1", "D1")
    e.step(("arrived", None))
    e.step(("handled", None))
    for _ in range(5):
        assert e.step(("nav_failed", None)) == [("goto", "D1")]
    assert e.state == "TO_DROP"


def test_stop_releases_unloaded_order():
    e = new()
    e.assign("o1", "P1", "D1")
    cmds = e.step(("stop", None))
    assert ("cancel", None) in cmds and ("release", "o1") in cmds
    assert e.state == "FAILED"
    assert e.step(("arrived", None)) == [] and e.assign("o2", "P2", "D2") == []


def test_stop_with_load_keeps_order_and_reset_resumes():
    e = new()
    e.assign("o1", "P1", "D1")
    e.step(("arrived", None))
    e.step(("handled", None))
    cmds = e.step(("stop", None))
    assert cmds == [("cancel", None)] and e.order == "o1"
    assert e.step(("reset", None)) == [("goto", "D1")] and e.state == "TO_DROP"


def test_go_home_and_interrupt():
    e = new()
    e.assign("o1", "P1", "D1")
    for ev in ("arrived", "handled", "arrived", "handled"):
        e.step((ev, None))
    assert e.step(("no_orders", None)) == [("goto", "H1")] and e.state == "TO_HOME"
    assert e.assign("o2", "P2", "D2") == [("cancel", None), ("goto", "P2")]
    assert e.state == "TO_PICKUP"


def test_no_orders_at_home_does_nothing():
    e = new()
    assert e.step(("no_orders", None)) == [] and e.state == "IDLE"

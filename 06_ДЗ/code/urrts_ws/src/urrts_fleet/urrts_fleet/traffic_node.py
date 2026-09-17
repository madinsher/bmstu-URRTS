"""Узел резервирования клеток на роботе (тактический уровень, этап 3).

Предоставляет исполнителю действие /<робот>/goto (как симулятор на этапе 2),
а сам ведёт робота по клеткам через низкоуровневое действие /<робот>/drive:
перед въездом в каждую клетку — заявка и резервирование по радио (Traffic).
"""
import json
import threading
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from urrts_interfaces.action import GoTo
from urrts_interfaces.msg import RadioPacket
from urrts_sim.warehouse import Warehouse

from .traffic_core import Traffic


class TrafficNode(Node):
    def __init__(self):
        super().__init__("traffic")
        self.declare_parameter("config", "")
        self.declare_parameter("robot", "r1")
        self.wh = Warehouse.from_file(self.get_parameter("config").value)
        self.name = self.get_parameter("robot").value
        tc = self.wh.cfg.get("traffic", {})
        self.core = Traffic(self.name, self.wh, settle=float(tc.get("settle", 0.3)),
                            stale_after=float(tc.get("stale_after", 2.0)),
                            wait_limit=float(tc.get("wait_limit", 6.0)),
                            lookahead=int(tc.get("lookahead", 2)))
        self.lock = threading.Lock()
        self.cb = ReentrantCallbackGroup()
        self.neigh = {}
        self.pose = None
        self.driving = False
        self.drive_result = None
        self.drive_gh = None
        self.goal_active = False
        self.radio = self.create_publisher(RadioPacket, "radio_out", 50)
        self.create_subscription(RadioPacket, "radio_in", self.on_radio, 100, callback_group=self.cb)
        self.create_subscription(PoseStamped, "pose", self.on_pose, 10, callback_group=self.cb)
        self.drive = ActionClient(self, GoTo, "drive", callback_group=self.cb)
        ActionServer(self, GoTo, "goto", execute_callback=self.execute,
                     goal_callback=lambda g: GoalResponse.ACCEPT,
                     cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=self.cb)
        self.create_timer(0.1, self.tick, callback_group=self.cb)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_radio(self, pkt):
        if pkt.topic == "traffic":
            with self.lock:
                self.neigh[pkt.sender] = json.loads(pkt.payload)

    def on_pose(self, msg):
        with self.lock:
            self.pose = (msg.pose.position.x, msg.pose.position.y)
            if self.core.cell is None:
                self.core.cell = self.wh.xy_cell(*self.pose)

    def tick(self):
        now = self.now()
        if now <= 0.0 or self.core.cell is None:
            return
        with self.lock:
            if self.goal_active:
                self.core.step(dict(self.neigh), now)
                if not self.driving and self.core.next_target() is not None:
                    self.start_drive(self.core.next_target())
            st = self.core.export_state(now)
        if rclpy.ok():
            self.radio.publish(RadioPacket(sender=self.name, topic="traffic", payload=json.dumps(st)))

    def start_drive(self, cell):
        x, y = self.wh.cell_xy(cell)
        self.driving, self.drive_result = True, None
        fut = self.drive.send_goal_async(GoTo.Goal(station="", x=x, y=y))
        fut.add_done_callback(self.on_drive_goal)

    def on_drive_goal(self, fut):
        gh = fut.result()
        if not gh.accepted:
            with self.lock:
                self.driving, self.drive_result = False, False
            return
        self.drive_gh = gh
        gh.get_result_async().add_done_callback(self.on_drive_result)

    def on_drive_result(self, fut):
        ok = fut.result().result.success
        with self.lock:
            self.driving, self.drive_result = False, ok
            if ok:
                self.core.arrived()
                if self.core.next_target() is not None:       # следующая клетка уже наша — не останавливаться
                    self.start_drive(self.core.next_target())

    def execute(self, gh):
        req = gh.request
        goal = self.wh.stations.get(req.station) if req.station else self.wh.xy_cell(req.x, req.y)
        t0 = self.now()
        res = GoTo.Result()
        while self.core.cell is None and rclpy.ok():
            time.sleep(0.02)
        with self.lock:
            ok = goal is not None and self.core.set_goal(self.core.cell, goal)
            self.goal_active = ok
            self.drive_result = None
        if not ok:
            gh.abort()
            res.success, res.message = False, "нет пути"
            return res
        fb = GoTo.Feedback()
        try:
            while rclpy.ok():
                with self.lock:
                    arrived = self.core.done() and not self.driving
                    failed = self.drive_result is False
                    remaining = (len(self.core.route) + len(self.core.reserved)) * self.wh.cell
                if arrived:
                    break
                if failed:
                    with self.lock:
                        self.goal_active, self.drive_result = False, None
                        self.core.clear()
                        self.core.reserved = []
                    gh.abort()
                    res.success, res.message = False, "остановлен"
                    return res
                if gh.is_cancel_requested:
                    with self.lock:
                        self.core.clear()             # зарезервированные клетки доехать и встать
                        self.goal_active = bool(self.core.reserved)
                    gh.canceled()
                    res.success, res.message = False, "отменено"
                    return res
                fb.distance_remaining = remaining
                gh.publish_feedback(fb)
                time.sleep(0.05)
        except Exception:
            if rclpy.ok():
                raise
            return res
        with self.lock:
            self.goal_active = False
        try:
            gh.succeed()
        except Exception:
            if rclpy.ok():
                raise
        res.success, res.message, res.travel_time = True, "приехал", self.now() - t0
        return res


def main():
    rclpy.init()
    node = TrafficNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

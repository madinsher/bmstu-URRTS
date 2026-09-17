"""Кинематический симулятор склада для этапов 1–2 ДЗ.

Что моделируется:
  * роботы едут по клеткам карты с постоянной скоростью по кратчайшему пути
    (каждый сам по себе: координации движения здесь НЕТ — это этап 3);
  * действия /<робот>/goto (GoTo) и /<робот>/handle (Handle); при drive_action = drive
    действие движения называется /<робот>/drive, а goto предоставляет узел резервирования;
  * радиоканал: пакет из /<робот>/radio_out доставляется в /<сосед>/radio_in,
    если сосед ближе radio_range, пакет не потерян (radio_drop), с задержкой radio_latency;
  * отказы: сервис /sim/inject (stop — робот встал и замолчал; mute/unmute — только радио;
    slow — скорость умножается на value);
  * часы: публикуется /clock с ускорением time_scale — все узлы запускаются с use_sim_time.

Публикует /<робот>/pose (PoseStamped), /sim/conflicts (Int32, накопленное число
конфликтов «два робота в одной клетке» или «обмен клетками»), журнал позиций CSV.
"""
import csv
import math
import os
import random
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Int32

from urrts_interfaces.action import GoTo, Handle
from urrts_interfaces.msg import RadioPacket
from urrts_interfaces.srv import InjectFault

from .warehouse import Warehouse

DT = 0.05          # шаг моделирования, с модельного времени


class SimRobot:
    def __init__(self, name, cell, wh, speed):
        self.name = name
        self.wh = wh
        self.x, self.y = wh.cell_xy(cell)
        self.cell = cell
        self.prev_cell = cell
        self.route = []            # клетки впереди
        self.speed = speed
        self.slow = 1.0
        self.stopped = False
        self.muted = False
        self.nav_active = False
        self.nav_result = None     # None — едет; True — приехал; False — отказ

    def start_route(self, goal_cell):
        path = self.wh.path(self.cell, goal_cell)
        if path is None:
            self.nav_result = False
            self.nav_active = False
            return False
        self.route = path[1:]
        self.nav_active = True
        self.nav_result = None
        return True

    def remaining(self):
        d = 0.0
        px, py = self.x, self.y
        for c in self.route:
            cx, cy = self.wh.cell_xy(c)
            d += math.hypot(cx - px, cy - py)
            px, py = cx, cy
        return d

    def advance(self, dt):
        self.prev_cell = self.cell
        if self.stopped:
            if self.nav_active:
                self.nav_active, self.nav_result = False, False
            return
        if not self.nav_active:
            return
        step = self.speed * self.slow * dt
        while step > 1e-12 and self.route:
            tx, ty = self.wh.cell_xy(self.route[0])
            d = math.hypot(tx - self.x, ty - self.y)
            if d <= step:
                self.x, self.y = tx, ty
                self.cell = self.route.pop(0)
                step -= d
            else:
                self.x += (tx - self.x) * step / d
                self.y += (ty - self.y) * step / d
                step = 0.0
        if not self.route:
            self.nav_active, self.nav_result = False, True


class Sim2D(Node):
    def __init__(self):
        super().__init__("sim2d", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, False)])
        self.declare_parameter("config", "")
        self.declare_parameter("time_scale", 1.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("log_dir", "")
        self.declare_parameter("drive_action", "goto")   # на этапе 3 — "drive", goto предоставляет узел traffic
        cfg_path = self.get_parameter("config").value
        self.wh = Warehouse.from_file(cfg_path)
        rc = self.wh.cfg["robots"]
        self.time_scale = float(self.get_parameter("time_scale").value)
        self.rng = random.Random(int(self.get_parameter("seed").value))
        self.radio_range = float(rc["radio_range"])
        self.radio_drop = float(rc["radio_drop"])
        self.radio_latency = float(rc["radio_latency"])
        self.handle_time = float(rc["handle_time"])
        self.lock = threading.Lock()
        self.t = 0.0
        self.robots = {}
        for name, home in zip(rc["names"], rc["homes"]):
            self.robots[name] = SimRobot(name, self.wh.stations[home], self.wh, float(rc["speed"]))
        self.cb = ReentrantCallbackGroup()
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.conf_pub = self.create_publisher(Int32, "/sim/conflicts", 10)
        self.conflicts = 0
        self.pose_pubs, self.radio_pubs = {}, {}
        self.queue = []            # (время доставки, получатель, пакет)
        for name in self.robots:
            self.pose_pubs[name] = self.create_publisher(PoseStamped, "/%s/pose" % name, 10)
            self.radio_pubs[name] = self.create_publisher(RadioPacket, "/%s/radio_in" % name, 50)
            self.create_subscription(RadioPacket, "/%s/radio_out" % name,
                                     lambda msg, n=name: self.on_radio(n, msg), 50, callback_group=self.cb)
            ActionServer(self, GoTo, "/%s/%s" % (name, self.get_parameter("drive_action").value),
                         execute_callback=lambda gh, n=name: self.exec_goto(n, gh),
                         goal_callback=lambda g: GoalResponse.ACCEPT,
                         cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=self.cb)
            ActionServer(self, Handle, "/%s/handle" % name,
                         execute_callback=lambda gh, n=name: self.exec_handle(n, gh),
                         goal_callback=lambda g: GoalResponse.ACCEPT,
                         cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=self.cb)
        self.create_service(InjectFault, "/sim/inject", self.on_inject, callback_group=self.cb)
        log_dir = self.get_parameter("log_dir").value
        self.log = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_fh = open(os.path.join(log_dir, "positions.csv"), "w", newline="", encoding="utf-8")
            self.log = csv.writer(self.log_fh)
            self.log.writerow(["t", "robot", "x", "y", "stopped"])
        self.last_log = -1.0
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.get_logger().info("склад %dx%d, роботов %d, ускорение %.1f" %
                               (self.wh.rows, self.wh.cols, len(self.robots), self.time_scale))

    # --- время и шаг ------------------------------------------------------------
    def stamp(self):
        sec = int(self.t)
        return Time(sec=sec, nanosec=int((self.t - sec) * 1e9))

    def loop(self):
        period = DT / max(self.time_scale, 1e-6)
        nxt = time.monotonic()
        while rclpy.ok():
            with self.lock:
                self.t += DT
                for r in self.robots.values():
                    r.advance(DT)
                self.count_conflicts()
                self.deliver_radio()
            self.clock_pub.publish(Clock(clock=self.stamp()))
            if int(self.t / DT) % 2 == 0:
                self.publish_poses()
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))

    def count_conflicts(self):
        rs = list(self.robots.values())
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                same = a.cell == b.cell and (a.prev_cell != a.cell or b.prev_cell != b.cell)
                swap = a.cell == b.prev_cell and b.cell == a.prev_cell and a.cell != a.prev_cell
                if same or swap:
                    self.conflicts += 1
        self.conf_pub.publish(Int32(data=self.conflicts))

    def publish_poses(self):
        for name, r in self.robots.items():
            msg = PoseStamped()
            msg.header.stamp = self.stamp()
            msg.header.frame_id = "map"
            msg.pose.position.x, msg.pose.position.y = r.x, r.y
            self.pose_pubs[name].publish(msg)
            if self.log and self.t - self.last_log >= 0.5 - 1e-9:
                self.log.writerow(["%.2f" % self.t, name, "%.3f" % r.x, "%.3f" % r.y, int(r.stopped)])
        if self.log and self.t - self.last_log >= 0.5 - 1e-9:
            self.last_log = self.t
            self.log_fh.flush()

    # --- радио ------------------------------------------------------------------
    def on_radio(self, sender, msg):
        with self.lock:
            src = self.robots[sender]
            if src.stopped or src.muted:
                return
            for name, r in self.robots.items():
                if name == sender or r.stopped:
                    continue
                if math.hypot(r.x - src.x, r.y - src.y) > self.radio_range:
                    continue
                if self.rng.random() < self.radio_drop:
                    continue
                self.queue.append((self.t + self.radio_latency, name, msg))

    def deliver_radio(self):
        due = [q for q in self.queue if q[0] <= self.t]
        self.queue = [q for q in self.queue if q[0] > self.t]
        for _, name, msg in due:
            self.radio_pubs[name].publish(msg)

    # --- действия ---------------------------------------------------------------
    def wait_model(self, seconds_model, cond=None, goal_handle=None):
        """Ждать seconds_model модельного времени или выполнения cond()."""
        t_end = self.t + seconds_model
        while rclpy.ok():
            with self.lock:
                if cond is not None and cond():
                    return True
                if self.t >= t_end:
                    return cond is None
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return None
            time.sleep(0.005)
        return False

    def exec_goto(self, name, gh):
        try:
            return self._exec_goto(name, gh)
        except Exception:
            if rclpy.ok():
                raise
            return GoTo.Result(success=False, message="остановка узла")

    def _exec_goto(self, name, gh):
        r = self.robots[name]
        req = gh.request
        with self.lock:
            goal = self.wh.stations.get(req.station) if req.station else self.wh.xy_cell(req.x, req.y)
            t0 = self.t
            ok = goal is not None and not r.stopped and r.start_route(goal)
        res = GoTo.Result()
        if not ok:
            gh.abort()
            res.success, res.message = False, "нет пути или робот остановлен"
            return res
        fb = GoTo.Feedback()
        while rclpy.ok():
            with self.lock:
                done, result, remaining = not r.nav_active, r.nav_result, r.remaining()
            if gh.is_cancel_requested:
                with self.lock:
                    r.route, r.nav_active = [], False
                gh.canceled()
                res.success, res.message = False, "отменено"
                return res
            if done:
                break
            fb.distance_remaining = remaining
            try:
                gh.publish_feedback(fb)
            except Exception:           # узел останавливается
                return res
            time.sleep(0.02)
        with self.lock:
            res.travel_time = self.t - t0
        if result:
            gh.succeed()
            res.success, res.message = True, "приехал"
        else:
            gh.abort()
            res.success, res.message = False, "остановлен"
        return res

    def exec_handle(self, name, gh):
        try:
            return self._exec_handle(name, gh)
        except Exception:
            if rclpy.ok():
                raise
            return Handle.Result(success=False, message="остановка узла")

    def _exec_handle(self, name, gh):
        r = self.robots[name]
        res = Handle.Result()
        with self.lock:
            at = self.wh.stations.get(gh.request.station)
            ok = at is not None and r.cell == at and not r.stopped and not r.nav_active
        if not ok:
            gh.abort()
            res.success, res.message = False, "робот не на станции"
            return res
        out = self.wait_model(self.handle_time, cond=lambda: r.stopped, goal_handle=gh)
        if out is None:
            gh.canceled()
            res.success, res.message = False, "отменено"
            return res
        if r.stopped:
            gh.abort()
            res.success, res.message = False, "робот остановлен"
            return res
        gh.succeed()
        res.success, res.message = True, "готово"
        return res

    # --- отказы -----------------------------------------------------------------
    def on_inject(self, req, resp):
        with self.lock:
            r = self.robots.get(req.robot)
            if r is None:
                resp.ok, resp.message = False, "нет робота %s" % req.robot
                return resp
            if req.fault == "stop":
                r.stopped = True
            elif req.fault == "mute":
                r.muted = True
            elif req.fault == "unmute":
                r.muted = False
            elif req.fault == "slow":
                r.slow = float(req.value)
            else:
                resp.ok, resp.message = False, "неизвестный отказ"
                return resp
        self.get_logger().warn("[t=%.1f] отказ %s у %s" % (self.t, req.fault, req.robot))
        resp.ok, resp.message = True, "ok"
        return resp


def main():
    rclpy.init()
    node = Sim2D()
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.log:
            node.log_fh.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

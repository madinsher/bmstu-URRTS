"""Инфраструктура сцены Webots: радиоканал, отказы, счётчик опасных сближений.

В Webots физику считает симулятор, поэтому здесь остаётся то, чего в нём нет:
  * радиоканал с ограниченной дальностью, потерями и задержкой (как в urrts_sim);
  * сервис /sim/inject: stop — робот замирает (halt) и замолкает, mute/unmute — только радио;
  * /sim/conflicts — число событий «центры роботов ближе danger метров».
"""
import math
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32

from urrts_interfaces.msg import RadioPacket
from urrts_interfaces.srv import InjectFault
from urrts_sim.warehouse import Warehouse


class World(Node):
    def __init__(self):
        super().__init__("world")
        self.declare_parameter("config", "")
        self.declare_parameter("seed", 0)
        self.declare_parameter("danger", 0.25)
        self.wh = Warehouse.from_file(self.get_parameter("config").value)
        rc = self.wh.cfg["robots"]
        self.names = list(rc["names"])
        self.range = float(rc["radio_range"])
        self.drop = float(rc["radio_drop"])
        self.latency = float(rc["radio_latency"])
        self.danger = float(self.get_parameter("danger").value)
        self.rng = random.Random(int(self.get_parameter("seed").value))
        self.pose, self.muted, self.stopped = {}, set(), set()
        self.queue = []
        self.conflicts = 0
        self.close_prev = set()
        self.radio_in, self.halt_pub = {}, {}
        for n in self.names:
            self.radio_in[n] = self.create_publisher(RadioPacket, "/%s/radio_in" % n, 50)
            self.halt_pub[n] = self.create_publisher(Bool, "/%s/halt" % n, 10)
            self.create_subscription(PoseStamped, "/%s/pose" % n,
                                     lambda m, k=n: self.on_pose(k, m), 10)
            self.create_subscription(RadioPacket, "/%s/radio_out" % n,
                                     lambda m, k=n: self.on_radio(k, m), 50)
        self.conf_pub = self.create_publisher(Int32, "/sim/conflicts", 10)
        self.create_service(InjectFault, "/sim/inject", self.on_inject)
        self.create_timer(0.05, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pose(self, name, msg):
        self.pose[name] = (msg.pose.position.x, msg.pose.position.y)

    def on_radio(self, sender, msg):
        if sender in self.muted or sender in self.stopped or sender not in self.pose:
            return
        sx, sy = self.pose[sender]
        for n in self.names:
            if n == sender or n in self.stopped or n not in self.pose:
                continue
            x, y = self.pose[n]
            if math.hypot(x - sx, y - sy) > self.range or self.rng.random() < self.drop:
                continue
            self.queue.append((self.now() + self.latency, n, msg))

    def tick(self):
        t = self.now()
        due = [q for q in self.queue if q[0] <= t]
        self.queue = [q for q in self.queue if q[0] > t]
        for _, name, msg in due:
            self.radio_in[name].publish(msg)
        close = set()
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                if a in self.pose and b in self.pose:
                    if math.dist(self.pose[a], self.pose[b]) < self.danger:
                        close.add((a, b))
        self.conflicts += len(close - self.close_prev)
        self.close_prev = close
        self.conf_pub.publish(Int32(data=self.conflicts))

    def on_inject(self, req, resp):
        if req.robot not in self.names:
            resp.ok, resp.message = False, "нет робота %s" % req.robot
            return resp
        if req.fault == "stop":
            self.stopped.add(req.robot)
            self.halt_pub[req.robot].publish(Bool(data=True))
        elif req.fault == "mute":
            self.muted.add(req.robot)
        elif req.fault == "unmute":
            self.muted.discard(req.robot)
        else:
            resp.ok, resp.message = False, "в сцене Webots поддержаны stop, mute, unmute"
            return resp
        self.get_logger().warn("[t=%.1f] %s: %s" % (self.now(), req.robot, req.fault))
        resp.ok, resp.message = True, "ok"
        return resp


def main():
    rclpy.init()
    node = World()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

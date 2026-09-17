"""Прогон с расписанием отказов и итоговой записью метрик.

Ждёт duration модельных секунд, вносит отказы из параметра faults
(«t:робот:отказ[:значение];…»: отказ симулятора stop, mute, unmute, slow;
при stop исполнителю робота дополнительно отправляется аварийный останов),
по окончании пишет log_dir/result.json и печатает итог.
"""
import json
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from urrts_interfaces.srv import InjectFault


class RunCheck(Node):
    def __init__(self):
        super().__init__("run_check")
        self.declare_parameter("duration", 600.0)
        self.declare_parameter("faults", "")
        self.declare_parameter("log_dir", "")
        self.declare_parameter("seed", 0)
        self.duration = float(self.get_parameter("duration").value)
        self.log_dir = self.get_parameter("log_dir").value
        self.faults = []
        for part in filter(None, self.get_parameter("faults").value.split(";")):
            f = part.split(":")
            self.faults.append((float(f[0]), f[1], f[2], float(f[3]) if len(f) > 3 else 0.0))
        self.faults.sort()
        self.metrics = {}
        self.t0 = None
        self.done = False
        self.inject = self.create_client(InjectFault, "/sim/inject")
        self.stop_pubs = {}
        self.create_subscription(String, "/fleet/metrics",
                                 lambda m: setattr(self, "metrics", json.loads(m.data)), 10)
        self.create_timer(0.2, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        t = self.now()
        if t <= 0.0:
            return
        if self.t0 is None:
            self.t0 = t
            self.get_logger().info("прогон %.0f с модельного времени, отказов: %d" % (self.duration, len(self.faults)))
        el = t - self.t0
        while self.faults and self.faults[0][0] <= el:
            _, robot, fault, value = self.faults.pop(0)
            self.inject.call_async(InjectFault.Request(robot=robot, fault=fault, value=value))
            if fault == "stop":
                if robot not in self.stop_pubs:
                    self.stop_pubs[robot] = self.create_publisher(String, "/%s/stop" % robot, 10)
                self.stop_pubs[robot].publish(String(data="stop"))
            self.get_logger().warn("t=%.1f: %s у %s" % (el, fault, robot))
        if el >= self.duration and not self.done:
            self.done = True
            self.finish()

    def finish(self):
        m = dict(self.metrics)
        m["seed"] = int(self.get_parameter("seed").value)
        m["faults"] = self.get_parameter("faults").value
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "result.json"), "w", encoding="utf-8") as fh:
                json.dump(m, fh, ensure_ascii=False, indent=2)
        self.get_logger().info("ИТОГ " + json.dumps(m, ensure_ascii=False))
        raise SystemExit(0)


def main():
    rclpy.init()
    node = RunCheck()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

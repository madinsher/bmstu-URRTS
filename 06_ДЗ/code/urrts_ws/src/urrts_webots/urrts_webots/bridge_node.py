"""Мост между флотом и навигационным стеком (этап 4).

Предоставляет роботу те же действия, что кинематический симулятор на этапах 1–3:
  drive  (GoTo)  — довести робота до точки; внутри — цель NavigateToPose для Nav2;
  handle (Handle) — погрузка или разгрузка: проверка, что робот у станции, и выдержка.

Благодаря этому узлы CBBA, исполнителя и резервирования клеток переносятся
на Webots без единой правки: меняется только то, кто исполняет движение.
"""
import math
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from urrts_interfaces.action import GoTo, Handle
from urrts_sim.warehouse import Warehouse


class Bridge(Node):
    def __init__(self):
        super().__init__("bridge")
        self.declare_parameter("config", "")
        self.declare_parameter("robot", "r1")
        self.wh = Warehouse.from_file(self.get_parameter("config").value)
        self.name = self.get_parameter("robot").value
        self.handle_time = float(self.wh.cfg["robots"]["handle_time"])
        self.cb = ReentrantCallbackGroup()
        self.pose = None
        self.create_subscription(PoseStamped, "pose", self.on_pose, 10, callback_group=self.cb)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose", callback_group=self.cb)
        ActionServer(self, GoTo, "drive", execute_callback=self.exec_drive,
                     goal_callback=lambda g: GoalResponse.ACCEPT,
                     cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=self.cb)
        ActionServer(self, Handle, "handle", execute_callback=self.exec_handle,
                     goal_callback=lambda g: GoalResponse.ACCEPT,
                     cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=self.cb)
        self.get_logger().info("мост %s: drive → navigate_to_pose" % self.name)

    def on_pose(self, msg):
        self.pose = (msg.pose.position.x, msg.pose.position.y)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def exec_drive(self, gh):
        req = gh.request
        res = GoTo.Result()
        if req.station:
            x, y = self.wh.station_xy(req.station)
        else:
            x, y = req.x, req.y
        if not self.nav.wait_for_server(timeout_sec=10.0):
            gh.abort()
            res.success, res.message = False, "Nav2 не отвечает"
            return res
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.w = 1.0
        t0 = self.now()
        send = self.nav.send_goal_async(goal)
        while not send.done() and rclpy.ok():
            time.sleep(0.02)
        handle = send.result()
        if handle is None or not handle.accepted:
            gh.abort()
            res.success, res.message = False, "цель не принята"
            return res
        fut = handle.get_result_async()
        fb = GoTo.Feedback()
        while not fut.done() and rclpy.ok():
            if gh.is_cancel_requested:
                handle.cancel_goal_async()
                gh.canceled()
                res.success, res.message = False, "отменено"
                return res
            if self.pose is not None:
                fb.distance_remaining = math.hypot(x - self.pose[0], y - self.pose[1])
                gh.publish_feedback(fb)
            time.sleep(0.05)
        status = fut.result().status if fut.result() is not None else 0
        ok = status == 4                      # STATUS_SUCCEEDED
        res.travel_time = self.now() - t0
        if ok:
            gh.succeed()
            res.success, res.message = True, "приехал"
        else:
            gh.abort()
            res.success, res.message = False, "навигация не довела (статус %d)" % status
        return res

    def exec_handle(self, gh):
        res = Handle.Result()
        sx, sy = self.wh.station_xy(gh.request.station)
        if self.pose is None or math.hypot(sx - self.pose[0], sy - self.pose[1]) > 0.3:
            gh.abort()
            res.success, res.message = False, "робот не у станции"
            return res
        t_end = self.now() + self.handle_time
        fb = Handle.Feedback()
        while self.now() < t_end and rclpy.ok():
            if gh.is_cancel_requested:
                gh.canceled()
                res.success, res.message = False, "отменено"
                return res
            fb.progress = max(0.0, 1.0 - (t_end - self.now()) / self.handle_time)
            gh.publish_feedback(fb)
            time.sleep(0.05)
        gh.succeed()
        res.success, res.message = True, "готово"
        return res


def main():
    rclpy.init()
    node = Bridge()
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

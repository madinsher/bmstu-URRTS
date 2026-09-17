"""Исполнитель заказов на роботе (стратегический уровень).

Получает закреплённый заказ из /<робот>/assign, ведёт автомат Executor,
вызывает действия goto и handle, сообщает о состоянии в /<робот>/executor
(для оценки маршрута агентом CBBA) и в /fleet/status (для наблюдения),
о доставке и возврате заказа — в /fleet/events.
"""
import json

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from urrts_interfaces.action import GoTo, Handle
from urrts_interfaces.msg import RobotStatus
from urrts_sim.warehouse import Warehouse
from geometry_msgs.msg import PoseStamped

from .executor_core import Executor



class ExecutorNode(Node):
    def __init__(self):
        super().__init__("executor")
        self.declare_parameter("config", "")
        self.declare_parameter("robot", "r1")
        self.declare_parameter("idle_before_home", 2.0)   # с простоя до возврата на стоянку
        self.idle_before_home = float(self.get_parameter("idle_before_home").value)
        self.wh = Warehouse.from_file(self.get_parameter("config").value)
        self.name = self.get_parameter("robot").value
        rc = self.wh.cfg["robots"]
        self.home = rc["homes"][rc["names"].index(self.name)]
        self.speed, self.handle_time = float(rc["speed"]), float(rc["handle_time"])
        self.fsm = Executor(self.home)
        self.station = self.home          # последняя достигнутая станция
        self.target = None
        self.goal_handle = None
        self.idle_since = 0.0
        self.x = self.y = 0.0
        self.goto = ActionClient(self, GoTo, "goto")
        self.handle_client = ActionClient(self, Handle, "handle")
        self.events = self.create_publisher(String, "/fleet/events", 50)
        self.status_pub = self.create_publisher(RobotStatus, "/fleet/status", 10)
        self.state_pub = self.create_publisher(String, "executor", 10)
        self.create_subscription(String, "assign", self.on_assign, 10)
        self.create_subscription(PoseStamped, "pose", self.on_pose, 10)
        self.create_subscription(String, "stop", lambda m: self.dispatch(("stop", None)), 10)
        self.create_subscription(String, "reset", lambda m: self.dispatch(("reset", None)), 10)
        self.create_timer(0.5, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pose(self, msg):
        self.x, self.y = msg.pose.position.x, msg.pose.position.y

    def on_assign(self, msg):
        o = json.loads(msg.data)
        cmds = self.fsm.assign(o["id"], o["pickup"], o["drop"])
        if not cmds and self.fsm.order != o["id"]:
            # автомат не принял заказ (занят или отказ) — вернуть в аукцион
            self.publish_event("released", o["id"])
            return
        self.run(cmds)

    def dispatch(self, event):
        self.run(self.fsm.step(event))

    def run(self, cmds):
        for cmd, arg in cmds:
            if cmd == "goto":
                self.target = arg
                self.send_goto(arg)
            elif cmd == "handle":
                self.send_handle(arg == "pick")
            elif cmd == "cancel":
                if self.goal_handle is not None:
                    self.goal_handle.cancel_goal_async()
                    self.goal_handle = None
            elif cmd == "release":
                self.publish_event("released", arg)
            elif cmd == "delivered":
                self.publish_event("delivered", arg)
        self.publish_state()

    # --- действия -----------------------------------------------------------
    def send_goto(self, station):
        if not self.goto.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("нет сервера goto")
            self.dispatch(("nav_failed", None))
            return
        fut = self.goto.send_goal_async(GoTo.Goal(station=station))
        fut.add_done_callback(lambda f: self.on_goal(f, "goto", station))

    def send_handle(self, pick):
        station = self.fsm.pickup if pick else self.fsm.drop
        fut = self.handle_client.send_goal_async(Handle.Goal(pick=pick, station=station))
        fut.add_done_callback(lambda f: self.on_goal(f, "handle", station))

    def on_goal(self, fut, kind, station):
        gh = fut.result()
        if not gh.accepted:
            self.dispatch(("nav_failed", None) if kind == "goto" else ("handle_failed", None))
            return
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(lambda f: self.on_result(f, gh, kind, station))

    def on_result(self, fut, gh, kind, station):
        if gh is not self.goal_handle:
            return                      # результат отменённой цели
        self.goal_handle = None
        res = fut.result().result
        if kind == "goto":
            if res.success:
                self.station = station
                self.dispatch(("arrived", None))
            elif res.message != "отменено":
                self.dispatch(("nav_failed", None))
        else:
            self.dispatch(("handled", None) if res.success else ("handle_failed", None))

    # --- состояние ----------------------------------------------------------
    def free_in(self):
        """Грубая оценка, через сколько секунд робот освободится."""
        s = self.fsm.state
        if s in ("IDLE", "TO_HOME", "FAILED") or not self.fsm.order:
            return 0.0
        v = self.speed
        t = 0.0
        if s in ("TO_PICKUP",):
            t += self.wh.station_distance(self.station, self.fsm.pickup) / v + self.handle_time
        if s in ("TO_PICKUP", "LOADING"):
            t += self.wh.station_distance(self.fsm.pickup, self.fsm.drop) / v + self.handle_time
        if s == "LOADING":
            t += self.handle_time / 2
        if s == "TO_DROP":
            t += self.wh.station_distance(self.station, self.fsm.drop) / v + self.handle_time
        if s == "UNLOADING":
            t += self.handle_time / 2
        return t

    def publish_state(self):
        st = {"state": self.fsm.state, "station": self.station, "order": self.fsm.order or "",
              "drop": self.fsm.drop if self.fsm.order else "", "free_in": self.free_in()}
        self.state_pub.publish(String(data=json.dumps(st)))
        msg = RobotStatus(robot=self.name, state=self.fsm.state, order_id=self.fsm.order or "",
                          x=self.x, y=self.y, battery=1.0)
        msg.stamp = self.get_clock().now().to_msg()
        self.status_pub.publish(msg)

    def publish_event(self, kind, oid):
        self.events.publish(String(data=json.dumps({"robot": self.name, "event": kind, "order": oid})))

    def tick(self):
        now = self.now()
        if self.fsm.state != "IDLE":
            self.idle_since = now
        elif now - self.idle_since > self.idle_before_home:
            self.idle_since = now
            self.dispatch(("no_orders", None))
        self.publish_state()


def main():
    rclpy.init()
    node = ExecutorNode()
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

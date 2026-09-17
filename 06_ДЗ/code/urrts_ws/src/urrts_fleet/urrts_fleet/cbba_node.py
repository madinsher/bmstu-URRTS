"""Агент CBBA на роботе (логический уровень).

Каждые period секунд: взять новые заказы из /fleet/orders, построить пакет,
слить с состояниями соседей, услышанными по радио, забыть молчащих победителей,
разослать своё состояние в /<робот>/radio_out. Если исполнитель свободен,
а первый заказ маршрута выигран и не оспаривался commit_after секунд, —
закрепить заказ: отдать исполнителю в /<робот>/assign и сообщить в /fleet/events.
"""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from urrts_interfaces.msg import OrderArray, RadioPacket
from urrts_sim.warehouse import Warehouse

from .cbba_core import CbbaAgent, Order


class CbbaNode(Node):
    def __init__(self):
        super().__init__("cbba")
        self.declare_parameter("config", "")
        self.declare_parameter("robot", "r1")
        self.wh = Warehouse.from_file(self.get_parameter("config").value)
        self.name = self.get_parameter("robot").value
        cc, rc = self.wh.cfg["cbba"], self.wh.cfg["robots"]
        self.agent = CbbaAgent(self.name, self.wh, capacity=int(cc["capacity"]), discount=float(cc["discount"]),
                               speed=float(rc["speed"]), handle_time=float(rc["handle_time"]))
        self.stale_after = float(cc["stale_after"])
        self.commit_after = float(cc["commit_after"])
        self.lost_after = float(cc["lost_after"])
        home = rc["homes"][rc["names"].index(self.name)]
        self.agent.start_station = home
        self.exec_state = {"state": "IDLE", "station": home, "free_in": 0.0}
        self.neigh = {}
        self.head_since = (None, 0.0)
        self.pending = None       # заказ отдан исполнителю, ждём смены его состояния
        self.t_start = None       # момент первого такта: до t_start + stale_after заказы не закрепляются
        self.events = self.create_publisher(String, "/fleet/events", 50)
        self.assign_pub = self.create_publisher(String, "assign", 10)
        self.radio = self.create_publisher(RadioPacket, "radio_out", 50)
        self.create_subscription(OrderArray, "/fleet/orders", self.on_orders, 10)
        self.create_subscription(RadioPacket, "radio_in", self.on_radio, 100)
        self.create_subscription(String, "executor", self.on_executor, 10)
        self.create_timer(float(cc["period"]), self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_orders(self, msg):
        for oid in msg.done:
            if oid in self.agent.orders or oid in self.agent.bundle:
                self.agent.remove_order(oid, done=True)
        for o in msg.open:
            self.agent.add_order(Order(o.id, o.pickup, o.drop, o.reward))

    def on_radio(self, pkt):
        if pkt.topic != "cbba":
            return
        st = json.loads(pkt.payload)
        self.neigh[pkt.sender] = st

    def on_executor(self, msg):
        self.exec_state = json.loads(msg.data)
        if self.pending and self.exec_state["state"] != "IDLE" and self.exec_state["state"] != "TO_HOME":
            self.pending = None

    def tick(self):
        now = self.now()
        if now <= 0.0:
            return
        if self.t_start is None:
            self.t_start = now
        ag = self.agent
        busy = self.exec_state["state"] not in ("IDLE", "TO_HOME") or self.pending is not None
        if busy:
            ag.start_station = self.exec_state.get("drop") or self.exec_state["station"]
            ag.start_delay = float(self.exec_state.get("free_in", 0.0))
        else:
            ag.start_station, ag.start_delay = self.exec_state["station"], 0.0
        ag.build_bundle()
        ag.consensus(self.neigh, now, self.stale_after)
        ag.forget_silent_winners(now, self.lost_after)
        ag.build_bundle()
        self.maybe_commit(now, busy)
        st = ag.export_state(now)
        if rclpy.ok():
            self.radio.publish(RadioPacket(sender=self.name, topic="cbba", payload=json.dumps(st)))

    def maybe_commit(self, now, busy):
        """Закрепить первый заказ маршрута за роботом, если исполнитель свободен,
        выигрыш стабилен commit_after секунд и все свежие соседи это подтвердили:
        их последние состояния отправлены уже после того, как заказ стал нашим.
        Без подтверждения два робота под нагрузкой (или на границе дальности связи)
        закрепляют один заказ: каждый считает себя победителем по устаревшим данным."""
        ag = self.agent
        head = ag.head()
        if head is None or ag.z.get(head) != self.name:
            self.head_since = (None, now)
            return
        if self.head_since[0] != head:
            self.head_since = (head, now)
            return
        if busy or now - self.head_since[1] < self.commit_after:
            return
        if now - self.t_start < self.stale_after:
            return                # прогрев: соседи ещё не услышаны
        for st in self.neigh.values():
            if now - st["stamp"] <= self.stale_after and st["stamp"] <= self.head_since[1]:
                return            # сосед ещё не мог узнать о нашем выигрыше
        o = ag.orders[head]
        ag.remove_order(head)
        self.pending = head
        self.head_since = (None, now)
        self.assign_pub.publish(String(data=json.dumps({"id": o.id, "pickup": o.pickup, "drop": o.drop})))
        self.events.publish(String(data=json.dumps({"robot": self.name, "event": "committed", "order": o.id})))


def main():
    rclpy.init()
    node = CbbaNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RuntimeError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

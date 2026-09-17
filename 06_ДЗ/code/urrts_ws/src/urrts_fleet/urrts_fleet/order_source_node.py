"""Источник заказов (складская система управления) и сборщик метрик.

Заказы появляются пуассоновским потоком (зерно — параметр seed), публикуются
в /fleet/orders (открытые и доставленные). Роботы сообщают о событиях в
/fleet/events (JSON: robot, event = committed | released | delivered, order).
Возвращённый в аукцион заказ переиздаётся с новым идентификатором «id~k».

Метрики публикуются в /fleet/metrics (JSON) раз в секунду и пишутся в log_dir.
Источник заказов НЕ распределяет заказы: он только знает, какие есть и какие сделаны.
"""
import csv
import json
import os
import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String

from urrts_interfaces.msg import Order, OrderArray
from urrts_sim.warehouse import Warehouse


def base_id(oid):
    return oid.split("~")[0]


class OrderSource(Node):
    def __init__(self):
        super().__init__("order_source")
        self.declare_parameter("config", "")
        self.declare_parameter("seed", 0)
        self.declare_parameter("log_dir", "")
        self.declare_parameter("stop_orders_after", 1e9)
        self.wh = Warehouse.from_file(self.get_parameter("config").value)
        oc = self.wh.cfg["orders"]
        self.rate = float(oc["rate"])
        self.reward = float(oc["reward"])
        self.pickups, self.drops = list(oc["stations_pickup"]), list(oc["stations_drop"])
        self.rng = random.Random(int(self.get_parameter("seed").value))
        self.log_dir = self.get_parameter("log_dir").value
        self.stop_after = float(self.get_parameter("stop_orders_after").value)
        self.open = {}          # id -> Order
        self.done = []
        self.info = {}          # базовый id -> словарь метрик заказа
        self.commits = {}       # текущий id -> множество роботов, закрепивших
        self.duplicates = 0
        self.releases = 0
        self.conflicts = 0
        self.counter = 0
        self.t0 = None
        self.next_arrival = None
        self.pub = self.create_publisher(OrderArray, "/fleet/orders", 10)
        self.mpub = self.create_publisher(String, "/fleet/metrics", 10)
        self.create_subscription(String, "/fleet/events", self.on_event, 100)
        self.create_subscription(Int32, "/sim/conflicts", lambda m: setattr(self, "conflicts", m.data), 10)
        self.create_timer(0.5, self.tick)
        self.create_timer(1.0, self.publish_metrics)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def new_order(self, t):
        self.counter += 1
        oid = "o%03d" % self.counter
        o = Order(id=oid, pickup=self.rng.choice(self.pickups), drop=self.rng.choice(self.drops),
                  reward=self.reward)
        o.created = self.get_clock().now().to_msg()
        self.open[oid] = o
        self.info[oid] = {"id": oid, "pickup": o.pickup, "drop": o.drop, "created": t,
                          "committed": None, "delivered": None, "robot": "", "reissues": 0}

    def tick(self):
        t = self.now()
        if t <= 0.0:
            return                      # часы симулятора ещё не пошли
        if self.t0 is None:
            self.t0 = t
            for _ in range(int(self.wh.cfg["orders"]["first_batch"])):
                self.new_order(t)
            self.next_arrival = t + self.rng.expovariate(self.rate)
        while self.next_arrival <= t and t - self.t0 < self.stop_after:
            self.new_order(self.next_arrival)
            self.next_arrival += self.rng.expovariate(self.rate)
        msg = OrderArray(open=list(self.open.values()), done=list(self.done))
        self.pub.publish(msg)

    def on_event(self, msg):
        ev = json.loads(msg.data)
        oid, rob, kind, t = ev["order"], ev["robot"], ev["event"], self.now()
        b = base_id(oid)
        if b not in self.info:
            return
        if kind == "committed":
            owners = self.commits.setdefault(oid, set())
            if owners and rob not in owners:
                self.duplicates += 1
                self.get_logger().warn("заказ %s закреплён дважды: %s и %s" % (oid, sorted(owners), rob))
            owners.add(rob)
            self.open.pop(oid, None)
            if self.info[b]["committed"] is None:
                self.info[b]["committed"] = t
                self.info[b]["robot"] = rob
        elif kind == "released":
            self.releases += 1
            self.open.pop(oid, None)
            k = self.info[b]["reissues"] + 1
            self.info[b]["reissues"] = k
            self.info[b]["committed"] = None
            nid = "%s~%d" % (b, k)
            o = Order(id=nid, pickup=self.info[b]["pickup"], drop=self.info[b]["drop"], reward=self.reward)
            o.created = self.get_clock().now().to_msg()
            self.open[nid] = o
        elif kind == "delivered":
            if self.info[b]["delivered"] is None:
                self.info[b]["delivered"] = t
                self.done.append(oid)

    def metrics(self):
        t = self.now()
        elapsed = 0.0 if self.t0 is None else t - self.t0
        delivered = [v for v in self.info.values() if v["delivered"] is not None]
        lead = [v["delivered"] - v["created"] for v in delivered]
        wait = [v["committed"] - v["created"] for v in self.info.values() if v["committed"] is not None]
        return {
            "time": round(elapsed, 2),
            "orders_created": len(self.info),
            "orders_open": len(self.open),
            "orders_delivered": len(delivered),
            "throughput_per_min": round(60.0 * len(delivered) / elapsed, 3) if elapsed > 0 else 0.0,
            "lead_time_mean": round(sum(lead) / len(lead), 2) if lead else None,
            "lead_time_max": round(max(lead), 2) if lead else None,
            "wait_commit_mean": round(sum(wait) / len(wait), 2) if wait else None,
            "duplicates": self.duplicates,
            "releases": self.releases,
            "path_conflicts": self.conflicts,
        }

    def publish_metrics(self):
        m = self.metrics()
        self.mpub.publish(String(data=json.dumps(m, ensure_ascii=False)))
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "metrics.json"), "w", encoding="utf-8") as fh:
                json.dump(m, fh, ensure_ascii=False, indent=2)
            with open(os.path.join(self.log_dir, "orders.csv"), "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["id", "pickup", "drop", "created", "committed",
                                                   "delivered", "robot", "reissues"])
                w.writeheader()
                for v in self.info.values():
                    w.writerow(v)


def main():
    rclpy.init()
    node = OrderSource()
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

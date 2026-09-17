"""Плагин webots_ros2_driver для TurtleBot3 Burger в группе роботов.

Исполнительный уровень: cmd_vel (geometry_msgs/Twist) → скорости колёс;
одометрия и поза — от Supervisor (идеальная локализация, сознательное упрощение:
ошибки локализации — отдельная тема). Все топики относительные, в пространстве
имён робота: cmd_vel, odom, pose, tf, halt (std_msgs/Bool — аварийный останов).
"""
import math

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

WHEEL_R = 0.033
AXLE = 0.160
MAX_W = 6.67
CMD_TIMEOUT = 0.5


class Tb3Driver:
    def init(self, webots_node, properties):
        self.robot = webots_node.robot
        self.name = self.robot.getName()
        self.left = self.robot.getDevice("left wheel motor")
        self.right = self.robot.getDevice("right wheel motor")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.me = self.robot.getSelf()
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("tb3_driver", namespace=self.name)
        self.cmd = (0.0, 0.0)
        self.cmd_time = -1e9
        self.halted = False
        self.node.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.node.create_subscription(Bool, "halt", lambda m: setattr(self, "halted", bool(m.data)), 10)
        self.odom_pub = self.node.create_publisher(Odometry, "odom", 10)
        self.pose_pub = self.node.create_publisher(PoseStamped, "pose", 10)
        self.tf_pub = self.node.create_publisher(TFMessage, "tf", 50)
        self.last_pub = -1.0

    def on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)
        self.cmd_time = self.robot.getTime()

    def stamp(self):
        t = self.robot.getTime()
        return Time(sec=int(t), nanosec=int((t - int(t)) * 1e9))

    def step(self):
        rclpy.spin_once(self.node, timeout_sec=0)
        t = self.robot.getTime()
        v, w = self.cmd
        if self.halted or t - self.cmd_time > CMD_TIMEOUT:
            v, w = 0.0, 0.0
        wl = (v - w * AXLE / 2) / WHEEL_R
        wr = (v + w * AXLE / 2) / WHEEL_R
        m = max(abs(wl), abs(wr), 1e-9)
        if m > MAX_W:
            wl, wr = wl * MAX_W / m, wr * MAX_W / m
        self.left.setVelocity(wl)
        self.right.setVelocity(wr)
        if t - self.last_pub < 0.05 - 1e-9:
            return
        self.last_pub = t
        x, y, _ = self.me.getPosition()
        R = self.me.getOrientation()
        yaw = math.atan2(R[3], R[0])
        qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
        vel = self.me.getVelocity()
        st = self.stamp()
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = st, "odom", "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.twist.twist.linear.x = math.cos(yaw) * vel[0] + math.sin(yaw) * vel[1]
        odom.twist.twist.angular.z = vel[5]
        self.odom_pub.publish(odom)
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = st, "odom", "base_link"
        tf.transform.translation.x, tf.transform.translation.y = x, y
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_pub.publish(TFMessage(transforms=[tf]))
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = st, "map"
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)

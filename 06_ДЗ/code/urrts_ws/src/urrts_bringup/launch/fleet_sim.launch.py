"""Склад на кинематическом симуляторе: этапы 1–3 ДЗ.

    ros2 launch urrts_bringup fleet_sim.launch.py seed:=3 duration:=600 time_scale:=5 \
        log_dir:=~/urrts_logs/run1 faults:="200:r2:stop"

duration — модельных секунд до окончания прогона (0 — бесконечно, без проверки);
faults   — расписание отказов «t:робот:отказ[:значение];…» (stop, mute, unmute, slow);
traffic  — true: движение через узлы резервирования клеток (этап 3).
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def build(context):
    share = get_package_share_directory("urrts_bringup")
    cfg_arg = LaunchConfiguration("config").perform(context)
    config = cfg_arg or os.path.join(share, "config", "warehouse.yaml")
    seed = int(LaunchConfiguration("seed").perform(context))
    duration = float(LaunchConfiguration("duration").perform(context))
    scale = float(LaunchConfiguration("time_scale").perform(context))
    log_dir = os.path.expanduser(LaunchConfiguration("log_dir").perform(context))
    faults = LaunchConfiguration("faults").perform(context)
    traffic = LaunchConfiguration("traffic").perform(context).lower() in ("1", "true", "yes")
    with open(config, encoding="utf-8") as fh:
        robots = yaml.safe_load(fh)["robots"]["names"]
    sim_time = {"use_sim_time": True}
    nodes = [
        Node(package="urrts_sim", executable="sim2d", name="sim2d", output="screen",
             parameters=[{"config": config, "time_scale": scale, "seed": seed, "log_dir": log_dir,
                          "drive_action": "drive" if traffic else "goto"}]),
        Node(package="urrts_fleet", executable="order_source", name="order_source", output="screen",
             parameters=[sim_time, {"config": config, "seed": seed, "log_dir": log_dir}]),
    ]
    for r in robots:
        nodes.append(Node(package="urrts_fleet", executable="cbba", name="cbba", namespace=r, output="screen",
                          parameters=[sim_time, {"config": config, "robot": r}]))
        nodes.append(Node(package="urrts_fleet", executable="executor", name="executor", namespace=r,
                          output="screen", parameters=[sim_time, {"config": config, "robot": r}]))
        if traffic:
            nodes.append(Node(package="urrts_fleet", executable="traffic", name="traffic", namespace=r,
                              output="screen", parameters=[sim_time, {"config": config, "robot": r}]))
    if duration > 0:
        check = Node(package="urrts_bringup", executable="run_check", name="run_check", output="screen",
                     parameters=[sim_time, {"duration": duration, "faults": faults, "log_dir": log_dir,
                                            "seed": seed}])
        nodes.append(check)
        nodes.append(RegisterEventHandler(OnProcessExit(
            target_action=check, on_exit=[EmitEvent(event=Shutdown(reason="прогон окончен"))])))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=""),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("duration", default_value="600"),
        DeclareLaunchArgument("time_scale", default_value="5.0"),
        DeclareLaunchArgument("log_dir", default_value="~/urrts_logs/latest"),
        DeclareLaunchArgument("faults", default_value=""),
        DeclareLaunchArgument("traffic", default_value="false"),
        OpaqueFunction(function=build),
    ])

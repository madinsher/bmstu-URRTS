"""Склад в Webots с Nav2 на каждом роботе: этап 4 ДЗ.

    ros2 launch urrts_webots fleet_webots.launch.py seed:=3 duration:=600 gui:=true \
        log_dir:=~/urrts_logs/webots1 faults:="200:r2:stop"

Мир, карта и URDF роботов генерируются из config/warehouse_webots.yaml
(urrts_webots.world_gen) в каталог сборки. Флот (CBBA, исполнитель, резервирование,
источник заказов) — те же узлы, что на этапах 1–3; движение исполняет Nav2 через мост.
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml

from urrts_webots.world_gen import generate


def use_local_webots():
    """webots_ros2 в WSL по умолчанию запускает WINDOWS-версию Webots (webots.exe)
    и соединяется с ней по сети. Если в самой WSL установлен Linux-Webots
    (каталог из WEBOTS_HOME, по умолчанию /usr/local/webots), запускаем его:
    это проще, не зависит от режима сети WSL и работает без графики.
    Чтобы вернуть поведение по умолчанию, задайте URRTS_WEBOTS_WINDOWS=1."""
    if os.environ.get("URRTS_WEBOTS_WINDOWS") == "1":
        return
    home = os.environ.get("WEBOTS_HOME", "/usr/local/webots")
    if not os.path.exists(os.path.join(home, "webots")):
        return
    os.environ["WEBOTS_HOME"] = home
    try:
        from webots_ros2_driver import utils as wr_utils
        from webots_ros2_driver import webots_launcher, webots_controller
    except ImportError:
        return
    wr_utils.is_wsl = lambda: False
    webots_launcher.is_wsl = lambda: False
    for mod in (webots_launcher, webots_controller):
        if hasattr(mod, "has_shared_folder"):
            mod.has_shared_folder = lambda: False
    wr_utils.has_shared_folder = lambda: False


def nav2_for(robot, params_file, map_yaml, use_composition=True):
    """Минимальный Nav2 для одного робота: планировщик, регулятор, поведения, навигатор."""
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static"), ("/map", "/map")]
    params = ParameterFile(RewrittenYaml(source_file=params_file, root_key=robot,
                                         param_rewrites={}, convert_types=True), allow_substs=True)
    common = dict(namespace=robot, parameters=[params], remappings=remappings, output="screen")
    nodes = [
        Node(package="nav2_planner", executable="planner_server", name="planner_server", **common),
        Node(package="nav2_controller", executable="controller_server", name="controller_server",
             **common),
        Node(package="nav2_behaviors", executable="behavior_server", name="behavior_server", **common),
        Node(package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator", **common),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_navigation", namespace=robot, output="screen",
             parameters=[{"autostart": True, "use_sim_time": True,
                          "node_names": ["planner_server", "controller_server",
                                         "behavior_server", "bt_navigator"]}]),
    ]
    return nodes


def build(context):
    use_local_webots()
    from webots_ros2_driver.webots_controller import WebotsController
    from webots_ros2_driver.webots_launcher import WebotsLauncher

    share = get_package_share_directory("urrts_webots")
    cfg_arg = LaunchConfiguration("config").perform(context)
    config = cfg_arg or os.path.join(share, "config", "warehouse_webots.yaml")
    seed = int(LaunchConfiguration("seed").perform(context))
    duration = float(LaunchConfiguration("duration").perform(context))
    log_dir = os.path.expanduser(LaunchConfiguration("log_dir").perform(context))
    faults = LaunchConfiguration("faults").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower() in ("1", "true", "yes")
    gen_dir = os.path.join(tempfile.gettempdir(), "urrts_webots_gen")
    cfg, paths = generate(config, gen_dir)
    robots = cfg["robots"]["names"]
    params_file = os.path.join(share, "config", "nav2_robot.yaml")
    sim_time = {"use_sim_time": True}

    webots = WebotsLauncher(world=paths["world"], mode="realtime", ros2_supervisor=True,
                            gui=gui)
    actions = [webots, webots._supervisor]
    actions += [
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[sim_time, {"yaml_filename": paths["map"], "topic_name": "/map",
                                    "frame_id": "map"}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_map",
             output="screen",
             parameters=[sim_time, {"autostart": True, "node_names": ["map_server"]}]),
        Node(package="urrts_webots", executable="world", name="world", output="screen",
             parameters=[sim_time, {"config": config, "seed": seed}]),
        Node(package="urrts_fleet", executable="order_source", name="order_source", output="screen",
             parameters=[sim_time, {"config": config, "seed": seed, "log_dir": log_dir}]),
    ]
    for r in robots:
        actions.append(WebotsController(robot_name=r, namespace=r,
                                        parameters=[{"robot_description": paths[r], "use_sim_time": True}],
                                        respawn=True))
        actions.append(Node(package="tf2_ros", executable="static_transform_publisher",
                            name="tf_map_odom", namespace=r, output="log",
                            remappings=[("/tf_static", "tf_static")],
                            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"]))
        actions.append(Node(package="tf2_ros", executable="static_transform_publisher",
                            name="tf_lidar", namespace=r, output="log",
                            remappings=[("/tf_static", "tf_static")],
                            arguments=["-0.032", "0", "0.17", "0", "0", "0", "base_link", "LDS-01"]))
        actions += nav2_for(r, params_file, paths["map"])
        actions.append(Node(package="urrts_webots", executable="bridge", name="bridge", namespace=r,
                            output="screen", parameters=[sim_time, {"config": config, "robot": r}]))
        actions.append(Node(package="urrts_fleet", executable="traffic", name="traffic", namespace=r,
                            output="screen", parameters=[sim_time, {"config": config, "robot": r}]))
        actions.append(Node(package="urrts_fleet", executable="cbba", name="cbba", namespace=r,
                            output="screen", parameters=[sim_time, {"config": config, "robot": r}]))
        actions.append(Node(package="urrts_fleet", executable="executor", name="executor", namespace=r,
                            output="screen", parameters=[sim_time, {"config": config, "robot": r}]))
    if duration > 0:
        check = Node(package="urrts_bringup", executable="run_check", name="run_check", output="screen",
                     parameters=[sim_time, {"duration": duration, "faults": faults,
                                            "log_dir": log_dir, "seed": seed}])
        actions.append(check)
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=check, on_exit=[EmitEvent(event=Shutdown(reason="прогон окончен"))])))
    actions.append(RegisterEventHandler(OnProcessExit(
        target_action=webots, on_exit=[EmitEvent(event=Shutdown(reason="Webots закрыт"))])))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=""),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("duration", default_value="0"),
        DeclareLaunchArgument("log_dir", default_value="~/urrts_logs/webots"),
        DeclareLaunchArgument("faults", default_value=""),
        DeclareLaunchArgument("gui", default_value="true"),
        OpaqueFunction(function=build),
    ])

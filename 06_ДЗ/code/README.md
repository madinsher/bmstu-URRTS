# Код домашнего задания УРРТС

Рабочее пространство ROS 2 для группового проекта «Многороботный склад».

## Состав

| Пакет | Назначение |
|---|---|
| `urrts_interfaces` | сообщения, действия и сервисы флота (`Order`, `RobotStatus`, `RadioPacket`, `GoTo`, `Handle`, `InjectFault`) |
| `urrts_sim` | кинематический симулятор склада: движение по клеткам, радиоканал с дальностью и потерями, отказы, ускоренные часы; модель склада `warehouse.py` |
| `urrts_fleet` | распределённая логика: `cbba_core` и `cbba_node` (кто какой заказ берёт), `executor_core` и `executor_node` (поведение робота), `traffic_core` и `traffic_node` (резервирование клеток), `order_source_node` (заказы и метрики) |
| `urrts_bringup` | конфигурация склада, launch-файл этапов 1–3, узел прогона с отказами |
| `urrts_webots` | этап 4: генератор мира и карты, плагин привода TurtleBot3, мост `drive` → Nav2, инфраструктура сцены |

**Ядра без ROS** (`cbba_core`, `executor_core`, `traffic_core`, `warehouse`) содержат всю логику, которую пишет бригада, и покрыты тестами: их можно отлаживать без запуска системы.

## Требования

- Ubuntu 24.04 и **ROS 2 Jazzy** (`ros-jazzy-desktop`), `python3-colcon-common-extensions`, `python3-pytest`;
- для этапа 4 дополнительно: `ros-jazzy-navigation2`, `ros-jazzy-nav2-bringup`, `ros-jazzy-webots-ros2` и **Webots R2025a**.

Работает и в WSL 2. Если Webots установлен внутри WSL, launch-файл этапа 4 запускает именно его (иначе `webots_ros2` ищет Windows-версию); поведение по умолчанию возвращает переменная `URRTS_WEBOTS_WINDOWS=1`.

## Сборка и проверка

```bash
cd urrts_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
python3 -m pytest src/urrts_fleet/test -q        # тесты ядер: до выполнения шагов часть падает
```

В каждом новом терминале: `source /opt/ros/jazzy/setup.bash && source install/setup.bash`. В WSL полезно `export FASTDDS_BUILTIN_TRANSPORTS=SHM` и `ros2 daemon stop`, иначе узлы могут не увидеть друг друга. Бригадам, работающим в одной сети, задать разный `ROS_DOMAIN_ID`.

## Запуск

```bash
# этап 2: без координации движения
ros2 launch urrts_bringup fleet_sim.launch.py seed:=3 duration:=600 time_scale:=8 log_dir:=~/urrts_logs/e2

# этап 3: с резервированием клеток и отказом робота на 300-й секунде
ros2 launch urrts_bringup fleet_sim.launch.py seed:=3 duration:=600 time_scale:=8 traffic:=true \
    log_dir:=~/urrts_logs/e3 faults:="300:r2:stop"

# серия прогонов и сводная таблица
./run_series.sh stage3 600 1 2 3

# этап 4: Webots и Nav2 (медленно, реальное время)
ros2 launch urrts_webots fleet_webots.launch.py seed:=3 duration:=600 gui:=true log_dir:=~/urrts_logs/web
```

`time_scale` ускоряет модельное время (8 — 600 с модели примерно за 75 с); в Webots время реальное. Журналы прогона: `result.json` (итоговые метрики), `metrics.json`, `orders.csv` (жизненный путь каждого заказа), `positions.csv` (траектории, только симулятор).

## Места для бригады

| Блок | Файл | Этап |
|---|---|---|
| `ДЗ-э2-шаг1` | `urrts_fleet/cbba_core.py`, `build_bundle` | 2 |
| `ДЗ-э2-шаг2` | `urrts_fleet/cbba_core.py`, `consensus` | 2 |
| `ДЗ-э2-шаг3` | `urrts_fleet/executor_core.py`, `Executor.step` | 2 |
| `ДЗ-э3-шаг1` | `urrts_fleet/traffic_core.py`, `may_enter` | 3 |
| `ДЗ-э3-шаг2` | `urrts_fleet/traffic_core.py`, `replan_around` | 3 |
| `ДЗ-э4-шаг1` | `urrts_fleet/cbba_core.py`, `forget_silent_winners` | 3 (отказоустойчивость) |

Всё остальное — инфраструктура: её менять не нужно, кроме заданий со звёздочкой. Параметры склада, радио, CBBA и резервирования — в `urrts_bringup/config/warehouse.yaml` (этап 4 — `urrts_webots/config/warehouse_webots.yaml`).

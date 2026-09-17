#!/bin/bash
# Серия прогонов склада и сводная таблица.
#
#   ./run_series.sh <режим> <длительность> <зёрна…>
#     режим: stage2 (без резервирования) | stage3 (с резервированием) | fail (резервирование + отказ робота)
#
# Пример: ./run_series.sh stage3 600 1 2 3
# Результаты: ~/urrts_logs/<режим>_<зерно>/result.json, сводка — ~/urrts_logs/<режим>_summary.csv
set -e
MODE=${1:-stage3}
DUR=${2:-600}
shift 2 || true
SEEDS=${@:-1 2 3}
WS=${WS:-~/urrts_hw_ws}
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export FASTDDS_BUILTIN_TRANSPORTS=SHM

TRAFFIC=true
FAULTS=""
case "$MODE" in
  stage2) TRAFFIC=false ;;
  stage3) TRAFFIC=true ;;
  fail)   TRAFFIC=true; FAULTS="$((DUR/2)):r2:stop" ;;
  *) echo "неизвестный режим: $MODE"; exit 2 ;;
esac

SUM=$HOME/urrts_logs/${MODE}_summary.csv
echo "seed,orders_created,orders_delivered,throughput_per_min,lead_time_mean,lead_time_max,wait_commit_mean,duplicates,releases,path_conflicts" > "$SUM"
for S in $SEEDS; do
  LOG=$HOME/urrts_logs/${MODE}_$S
  rm -rf "$LOG"
  echo "=== $MODE, зерно $S"
  ARGS=(seed:=$S duration:=$DUR time_scale:=8 log_dir:=$LOG traffic:=$TRAFFIC)
  [ -n "$FAULTS" ] && ARGS+=(faults:="$FAULTS")
  ros2 launch urrts_bringup fleet_sim.launch.py "${ARGS[@]}" > "$LOG.out" 2>&1 || true
  python3 - "$LOG/result.json" "$SUM" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    print("нет результата:", sys.argv[1]); raise SystemExit
row = [m.get("seed"), m.get("orders_created"), m.get("orders_delivered"), m.get("throughput_per_min"),
       m.get("lead_time_mean"), m.get("lead_time_max"), m.get("wait_commit_mean"),
       m.get("duplicates"), m.get("releases"), m.get("path_conflicts")]
open(sys.argv[2], "a").write(",".join(str(x) for x in row) + "\n")
print("  доставлено %s из %s, конфликтов %s, дублей %s" %
      (m.get("orders_delivered"), m.get("orders_created"), m.get("path_conflicts"), m.get("duplicates")))
PY
done
echo "сводка: $SUM"
column -s, -t "$SUM"

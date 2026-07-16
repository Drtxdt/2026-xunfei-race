#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_WS="${ROBOT_WS:-/home/ucar/ucar_ws}"
ROS_SETUP="/opt/ros/noetic/setup.bash"
CURRENT_SETUP="$ROOT/devel/setup.bash"
SPEECH_BIN="$ROBOT_WS/devel/lib/speech_command/speech_command_node"
TTS_BIN="$ROBOT_WS/devel/lib/speech_command/voice_speak_node"
SPEECH_PACKAGE="$ROBOT_WS/src/speech_command"
IAT_CREDENTIALS="${IAT_CREDENTIALS_FILE:-/home/ucar/.config/ucar_2026/iat_credentials.json}"
DEBUG=false

usage() {
  echo "Usage: bash run_competition.sh [--debug]"
}

for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage; exit 2 ;;
  esac
done

[[ -f "$ROS_SETUP" ]] || { echo "Missing $ROS_SETUP" >&2; exit 1; }
[[ -f "$CURRENT_SETUP" ]] || {
  echo "Missing $CURRENT_SETUP; run catkin_make in $ROOT first." >&2
  exit 1
}
[[ -f "$ROBOT_WS/devel/setup.bash" ]] || { echo "Missing speech workspace setup." >&2; exit 1; }
[[ -x "$SPEECH_BIN" ]] || { echo "Missing executable: $SPEECH_BIN" >&2; exit 1; }
[[ -x "$TTS_BIN" ]] || { echo "Missing executable: $TTS_BIN" >&2; exit 1; }
[[ -f "$SPEECH_PACKAGE/package.xml" ]] || { echo "Missing ROS package: $SPEECH_PACKAGE" >&2; exit 1; }
[[ -r "$IAT_CREDENTIALS" ]] || {
  echo "Missing IAT credentials: $IAT_CREDENTIALS" >&2
  echo "Run: bash $ROOT/debug/repair_speech_command_asr.sh" >&2
  exit 1
}
python3 -c 'import websocket' 2>/dev/null || {
  echo "Missing Python websocket-client module required by fixed command IAT." >&2
  exit 1
}
[[ -n "${XF_SPARK_API_PASSWORD:-}" ]] || { echo "XF_SPARK_API_PASSWORD is not set." >&2; exit 1; }
[[ -n "${SIM_BRIDGE_HOST:-}" ]] || { echo "SIM_BRIDGE_HOST is not set." >&2; exit 1; }

source "$ROS_SETUP"
source "$CURRENT_SETUP"

# The legacy voice binaries resolve aiui.cfg and SDK resources with
# ros::package::getPath("speech_command").  Expose only that package; sourcing
# the complete legacy workspace would reintroduce its duplicate ROS packages.
VOICE_ROS_PACKAGE_PATH="$SPEECH_PACKAGE${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"

LOG_DIR="$ROOT/logs/competition_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
PIDS=()

cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

roscore >"$LOG_DIR/roscore.log" 2>&1 &
PIDS+=("$!")
for _ in $(seq 1 50); do
  rosparam list >/dev/null 2>&1 && break
  sleep 0.1
done
rosparam list >/dev/null 2>&1 || { echo "ROS master did not become ready." >&2; exit 1; }

ROS_PACKAGE_PATH="$VOICE_ROS_PACKAGE_PATH" "$SPEECH_BIN" >"$LOG_DIR/speech_command.log" 2>&1 &
PIDS+=("$!")
ROS_PACKAGE_PATH="$VOICE_ROS_PACKAGE_PATH" "$TTS_BIN" >"$LOG_DIR/voice_speak.log" 2>&1 &
PIDS+=("$!")
rosrun ucar_2026_competition fixed_command_iat.py >"$LOG_DIR/fixed_command_iat.log" 2>&1 &
PIDS+=("$!")

TRAFFIC_CONFIGURED=false
TRAFFIC_X_VALUE=0.0
TRAFFIC_Y_VALUE=0.0
TRAFFIC_YAW_VALUE=0.0
if [[ -n "${TRAFFIC_X:-}" && -n "${TRAFFIC_Y:-}" && -n "${TRAFFIC_YAW:-}" ]]; then
  TRAFFIC_CONFIGURED=true
  TRAFFIC_X_VALUE="$TRAFFIC_X"
  TRAFFIC_Y_VALUE="$TRAFFIC_Y"
  TRAFFIC_YAW_VALUE="$TRAFFIC_YAW"
else
  echo "Traffic pose is unset. The flow will pause safely after task 3."
fi

echo "Competition logs: $LOG_DIR"
echo "Simulation bridge: $SIM_BRIDGE_HOST:${SIM_BRIDGE_PORT:-26003}"
roslaunch ucar_2026_competition full_competition.launch \
  debug:="$DEBUG" \
  sim_bridge_host:="$SIM_BRIDGE_HOST" \
  sim_bridge_port:="${SIM_BRIDGE_PORT:-26003}" \
  traffic_pose_configured:="$TRAFFIC_CONFIGURED" \
  traffic_x:="$TRAFFIC_X_VALUE" \
  traffic_y:="$TRAFFIC_Y_VALUE" \
  traffic_yaw:="$TRAFFIC_YAW_VALUE" \
  2>&1 | tee "$LOG_DIR/full_competition.log"

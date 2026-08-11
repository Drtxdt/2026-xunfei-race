#!/usr/bin/env bash
set -Eeuo pipefail

# Debug convenience wrapper. The official competition entrypoint remains:
#   roslaunch ucar_2026_competition full_competition.launch

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/noetic/setup.bash"
COMPETITION_WS="${UCAR_COMPETITION_WS:-$ROOT}"
SIM_WS="${UCAR_SIM_WS:-}"
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
[[ -n "$SIM_WS" ]] || { echo "UCAR_SIM_WS is not set." >&2; exit 1; }
[[ -n "${UCAR_ROBOT_HOST:-}" ]] || { echo "UCAR_ROBOT_HOST is not set." >&2; exit 1; }
[[ -f "$SIM_WS/devel/setup.bash" ]] || {
  echo "Invalid UCAR_SIM_WS: missing $SIM_WS/devel/setup.bash" >&2
  exit 1
}
[[ -f "$COMPETITION_WS/devel/setup.bash" ]] || {
  echo "Competition workspace is not built: $COMPETITION_WS" >&2
  exit 1
}

unset ROS_MASTER_URI ROS_IP ROS_HOSTNAME
source "$ROS_SETUP"
source "$SIM_WS/devel/setup.bash"
source "$COMPETITION_WS/devel/setup.bash"

exec roslaunch ucar_2026_competition full_competition.launch debug:="$DEBUG"

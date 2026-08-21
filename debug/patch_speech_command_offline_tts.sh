#!/usr/bin/env bash
set -euo pipefail

APPID="${1:-97d4e3db}"
RESOURCE_DIR="${2:-}"
TTS_SPEED="${TTS_SPEED:-60}"
SPEECH_COMMAND_DIR="${SPEECH_COMMAND_DIR:-/home/ucar/ucar_ws/src/speech_command}"
UCAR_WS="${UCAR_WS:-/home/ucar/ucar_ws}"
VOICE_NODE="${SPEECH_COMMAND_DIR}/src/voice_speak_node.cpp"
TTS_RESOURCE_DIR="${SPEECH_COMMAND_DIR}/bin/msc/res/tts"

if [ ! -f "${VOICE_NODE}" ]; then
  echo "Missing voice_speak_node.cpp: ${VOICE_NODE}" >&2
  exit 1
fi

echo "Patching speech_command offline TTS appid to ${APPID}"
cp -n "${VOICE_NODE}" "${VOICE_NODE}.bak"
sed -i -E "s/appid = [0-9a-fA-F]+/appid = ${APPID}/g" "${VOICE_NODE}"
grep -n "appid =" "${VOICE_NODE}"

echo "Patching speech_command offline TTS speed to ${TTS_SPEED}"
sed -i -E "s/speed = [0-9]+/speed = ${TTS_SPEED}/g" "${VOICE_NODE}"
grep -n "speed =" "${VOICE_NODE}"

if [ -n "${RESOURCE_DIR}" ]; then
  if [ ! -d "${RESOURCE_DIR}" ]; then
    echo "Resource directory does not exist: ${RESOURCE_DIR}" >&2
    exit 1
  fi
  for file in common.jet xiaoyan.jet xiaofeng.jet; do
    if [ ! -f "${RESOURCE_DIR}/${file}" ]; then
      echo "Missing resource file: ${RESOURCE_DIR}/${file}" >&2
      exit 1
    fi
  done

  echo "Replacing MSC TTS resources from ${RESOURCE_DIR}"
  cp -rn "${TTS_RESOURCE_DIR}" "${TTS_RESOURCE_DIR}.bak"
  cp -f "${RESOURCE_DIR}/common.jet" "${TTS_RESOURCE_DIR}/"
  cp -f "${RESOURCE_DIR}/xiaoyan.jet" "${TTS_RESOURCE_DIR}/"
  cp -f "${RESOURCE_DIR}/xiaofeng.jet" "${TTS_RESOURCE_DIR}/"
  ls -lh "${TTS_RESOURCE_DIR}"
else
  echo "No resource directory supplied; keeping existing MSC TTS resources."
  echo "If QTTSAudioGet returns 11210, rerun with the 97d4e3db SDK resource path."
fi

echo "Clearing old MSC activation/cache directories"
rm -rf /home/ucar/.ros/msc
rm -rf /home/ucar/2026-xunfei-race/msc

echo "Rebuilding /home/ucar/ucar_ws"
cd "${UCAR_WS}"
catkin_make --pkg speech_command

echo "Done. Test with:"
echo "  source ${UCAR_WS}/devel/setup.bash"
echo "  rosrun speech_command voice_speak_node"
echo "Then publish in another terminal:"
echo "  source ${UCAR_WS}/devel/setup.bash"
echo "  rostopic pub -1 /speak std_msgs/String \"data: '你好，这是离线语音合成测试'\""

#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_WS="${ROBOT_WS:-/home/ucar/ucar_ws}"
REFERENCE_WS="${REFERENCE_ROBOT_WS:-/home/ucar/ori_ucar}"
CURRENT_PACKAGE="$CURRENT_WS/src/speech_command"
REFERENCE_PACKAGE="$REFERENCE_WS/src/speech_command"
GLOBAL_H="$CURRENT_PACKAGE/include/Global.h"
REFERENCE_GLOBAL_H="$REFERENCE_PACKAGE/include/Global.h"
AIUI_TESTER="$CURRENT_PACKAGE/src/AIUITester.cpp"
AIUI_MAIN="$CURRENT_PACKAGE/src/aiuiMain.cpp"
AIUI_CFG="$CURRENT_PACKAGE/config/AIUI/cfg/aiui.cfg"
COMPETITION_WS="${ROBOT_COMPETITION_WS:-/home/ucar/2026-xunfei-race}"
IAT_DOC="$COMPETITION_WS/docs/task1_llm_interface.md"
IAT_CREDENTIALS="${IAT_CREDENTIALS_FILE:-/home/ucar/.config/ucar_2026/iat_credentials.json}"
STAMP="$(date +%Y%m%d_%H%M%S)"

for file in "$GLOBAL_H" "$REFERENCE_GLOBAL_H" "$AIUI_TESTER" "$AIUI_MAIN" "$AIUI_CFG"; do
  [[ -f "$file" ]] || { echo "Missing required file: $file" >&2; exit 1; }
done
[[ -f "$IAT_DOC" ]] || { echo "Missing IAT credential source: $IAT_DOC" >&2; exit 1; }
[[ -f "$CURRENT_WS/build/Makefile" ]] || {
  echo "Missing configured build tree: $CURRENT_WS/build" >&2
  exit 1
}

cp -p "$GLOBAL_H" "$GLOBAL_H.before_asr_repair_$STAMP"
cp -p "$AIUI_TESTER" "$AIUI_TESTER.before_asr_repair_$STAMP"
cp -p "$AIUI_MAIN" "$AIUI_MAIN.before_asr_repair_$STAMP"
cp -p "$AIUI_CFG" "$AIUI_CFG.before_asr_repair_$STAMP"

# Keep the current hardware and topic changes, but restore the known working
# online AIUI appid/key pair from the robot's factory backup. Values are never
# printed or copied into this repository.
python3 - "$GLOBAL_H" "$REFERENCE_GLOBAL_H" <<'PY'
import re
import sys

current_path, reference_path = sys.argv[1:]
current = open(current_path, encoding="utf-8").read()
reference = open(reference_path, encoding="utf-8").read()

for name in ("appid", "key"):
    source_match = re.search(
        r"string\s+{}\s*=\s*(\"[^\"]*\")".format(name), reference
    )
    if source_match is None:
        raise SystemExit("missing {} in factory backup".format(name))
    pattern = r"(string\s+{}\s*=\s*)\"[^\"]*\"".format(name)
    current, count = re.subn(
        pattern, lambda match: match.group(1) + source_match.group(1), current, count=1
    )
    if count != 1:
        raise SystemExit("could not update {} in current source".format(name))

current, count = re.subn(
    r"bool\s+offline_mode\s*=\s*true", "bool offline_mode = false", current, count=1
)
if count not in (0, 1) or "bool offline_mode = false" not in current:
    raise SystemExit("could not disable expired offline grammar mode")

with open(current_path, "w", encoding="utf-8") as stream:
    stream.write(current)
PY

# Keep AIUI in READY after the hardware wake word. The competition controller
# first speaks “我在”, then calls start_listening, so TTS cannot be recognized as
# the user's command. The stop service similarly prevents “好的” from feeding
# back into ASR.
python3 - "$AIUI_TESTER" "$AIUI_MAIN" <<'PY'
import re
import sys

tester_path, main_path = sys.argv[1:]
tester = open(tester_path, encoding="utf-8").read()

pcm_marker = "Competition PCM bridge for fixed-command IAT"
if pcm_marker not in tester:
    include_anchor = "#include <AIUITester.h>"
    if include_anchor not in tester:
        raise SystemExit("AIUITester include anchor missing")
    tester = tester.replace(
        include_anchor,
        include_anchor + '\n#include "std_msgs/UInt8MultiArray.h"\n#include <atomic>',
        1,
    )

    global_anchor = "static RingBuffer buffer_source(MAX_BUFFER);"
    if global_anchor not in tester:
        raise SystemExit("AIUITester global anchor missing")
    tester = tester.replace(
        global_anchor,
        global_anchor
        + "\n// Competition PCM bridge for fixed-command IAT.\n"
        + "static std::atomic<bool> competition_audio_enabled(false);",
        1,
    )

    create_match = re.search(r"(?m)^(?P<indent>[ \t]*)createAgent\(\);[ \t]*$", tester)
    if create_match is None:
        raise SystemExit("AIUITester createAgent anchor missing")
    indent = create_match.group("indent")
    create_block = (
        create_match.group(0)
        + "\n"
        + indent + "ros::NodeHandle competition_audio_nh;\n"
        + indent + "ros::Publisher competition_audio_pub = competition_audio_nh.advertise<std_msgs::UInt8MultiArray>(\n"
        + indent + '    "/speech_command_node/audio_pcm", 100);'
    )
    tester = tester[:create_match.start()] + create_block + tester[create_match.end():]

    buffer_match = re.search(
        r"(?m)^(?P<indent>[ \t]*)Buffer \*buffer = Buffer::alloc\(buffer_frames\*frame_byte\*AUDIO_CHANNEL_SET\);[ \t]*$",
        tester,
    )
    if buffer_match is None:
        raise SystemExit("AIUITester PCM buffer anchor missing")
    indent = buffer_match.group("indent")
    inner = indent + "        "
    pcm_publish = (
        indent + "if (competition_audio_enabled.load())\n"
        + indent + "{\n"
        + inner + "const size_t audio_size = buffer_frames * frame_byte * AUDIO_CHANNEL_SET;\n"
        + inner + "const unsigned char *audio_bytes = reinterpret_cast<unsigned char *>(buffer1);\n"
        + inner + "std_msgs::UInt8MultiArray audio_msg;\n"
        + inner + "audio_msg.data.assign(audio_bytes, audio_bytes + audio_size);\n"
        + inner + "competition_audio_pub.publish(audio_msg);\n"
        + indent + "}\n"
    )
    tester = tester[:buffer_match.start()] + pcm_publish + tester[buffer_match.start():]

    tester, count = re.subn(
        r"(void gWakeup\(\)\s*\{)",
        r"\1\n        competition_audio_enabled.store(true);",
        tester,
        count=1,
    )
    if count != 1:
        raise SystemExit("AIUITester gWakeup anchor missing")
    tester, count = re.subn(
        r"(void gSleep\(\)\s*\{)",
        r"\1\n        competition_audio_enabled.store(false);",
        tester,
        count=1,
    )
    if count != 1:
        raise SystemExit("AIUITester gSleep anchor missing")

marker = "ROS controller starts AIUI after the wakeup reply"
if marker not in tester:
    tester, count = re.subn(
        r"(?m)^(\s*)gWakeup\(\);\s*$",
        r"\1// ROS controller starts AIUI after the wakeup reply.",
        tester,
        count=1,
    )
    if count != 1:
        raise SystemExit("could not defer hardware-triggered gWakeup()")
with open(tester_path, "w", encoding="utf-8") as stream:
    stream.write(tester)

main = open(main_path, encoding="utf-8").read()
declarations = """ros::Publisher pub_wakeup;
void gWakeup();
void gSleep();"""
if "void gWakeup();" not in main:
    main = main.replace("ros::Publisher pub_wakeup;", declarations, 1)

callback_marker = "AIUI listening started by competition controller"
if callback_marker not in main:
    callbacks = r'''
bool start_listening_server(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &response)
{
        gWakeup();
        response.success = true;
        response.message = "AIUI listening started by competition controller";
        return true;
}

bool stop_listening_server(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &response)
{
        gSleep();
        response.success = true;
        response.message = "AIUI listening stopped by competition controller";
        return true;
}

'''
    anchor = "void test_callback()"
    if anchor not in main:
        raise SystemExit("test_callback anchor missing in aiuiMain.cpp")
    main = main.replace(anchor, callbacks + anchor, 1)

service_marker = '"/speech_command_node/start_listening"'
if service_marker not in main:
    bind_match = re.search(r"(?m)^(?P<indent>[ \t]*)t\.bind\(test_callback\);[ \t]*$", main)
    if bind_match is None:
        raise SystemExit("t.bind anchor missing in aiuiMain.cpp")
    indent = bind_match.group("indent")
    services = (
        indent + "ros::ServiceServer start_listening_service = ndHandle.advertiseService(\n"
        + indent + '    "/speech_command_node/start_listening", start_listening_server);\n'
        + indent + "ros::ServiceServer stop_listening_service = ndHandle.advertiseService(\n"
        + indent + '    "/speech_command_node/stop_listening", stop_listening_server);\n\n'
    )
    main = main[:bind_match.start()] + services + main[bind_match.start():]

with open(main_path, "w", encoding="utf-8") as stream:
    stream.write(main)
PY

# The bundled local grammar resource is an expired trial. Use cloud IAT. Keep
# AIUI in continuous interaction after “我在”; the competition controller calls
# stop_listening only after it accepts a valid “取得xx” command. This prevents an
# incidental VAD turn from putting the microphone back to READY.
sed -i '/"iat":{/,/}/{s/"engine_type":"local"/"engine_type":"cloud"/;}' "$AIUI_CFG"
sed -i '/"speech":{/,/}/{s/"intent_engine_type":"local"/"intent_engine_type":"cloud"/;}' "$AIUI_CFG"
sed -i 's/"interact_timeout":"-1"/"interact_timeout":"60000"/' "$AIUI_CFG"
sed -i 's/"result_timeout":"2000"/"result_timeout":"5000"/' "$AIUI_CFG"
sed -i 's/"interact_mode":"oneshot"/"interact_mode":"continuous"/' "$AIUI_CFG"
sed -i 's/"vad_bos":"1000"/"vad_bos":"8000"/' "$AIUI_CFG"
sed -i 's/"vad_eos":"1000"/"vad_eos":"2500"/' "$AIUI_CFG"
sed -i 's/"cloud_vad_eos":"30000"/"cloud_vad_eos":"2000"/' "$AIUI_CFG"

# Store the already-provisioned WebAPI IAT credentials outside the repository.
# The runtime node also accepts XF_IAT_APPID/XF_IAT_API_SECRET/XF_IAT_API_KEY.
python3 - "$IAT_DOC" "$IAT_CREDENTIALS" <<'PY'
import json
import os
import re
import sys

source_path, output_path = sys.argv[1:]
source = open(source_path, encoding="utf-8").read()

def required(pattern, name):
    match = re.search(pattern, source)
    if match is None:
        raise SystemExit("missing {} in {}".format(name, source_path))
    return match.group(1).strip()

credentials = {
    "appid": required(r"APPID[：:]\s*([0-9a-fA-F]+)", "APPID"),
    "api_secret": required(r"APISecret[：:]\s*([^\s]+)", "APISecret"),
    "api_key": required(r"APIKey[：:]\s*([^\s]+)", "APIKey"),
}
directory = os.path.dirname(output_path)
os.makedirs(directory, mode=0o700, exist_ok=True)
temporary = output_path + ".tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(credentials, stream)
    stream.write("\n")
os.chmod(temporary, 0o600)
os.replace(temporary, output_path)
PY

# Do not reconfigure this legacy workspace: its source tree contains duplicate
# ROS packages. Build only the already-configured targets.
make -C "$CURRENT_WS/build" AIUITester/fast -j2
make -C "$CURRENT_WS/build" speech_command_node/fast -j2

echo "speech_command ASR repaired. Restart task1.launch to apply it."

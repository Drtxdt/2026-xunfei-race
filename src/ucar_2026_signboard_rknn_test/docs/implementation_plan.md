# Signboard RKNN Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a ROS package `ucar_2026_signboard_rknn_test` that runs RKNN/YOLO inference for three factory signboard classes, shows annotated X11 debug stream, and announces stable decisions via speech.

**Architecture:** Clone the structure of `ucar_2026_traffic_light_rknn_test` exactly, replacing traffic-light semantics with three signboard classes. Re-use the same consensus filtering, speech service fallback, and RKNN post-processing logic.

**Tech Stack:** ROS1 (rospy), OpenCV, RKNNLite, cv_bridge, ucar_2026_competition_speech.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `package.xml` | ROS package manifest (same deps as reference) |
| `CMakeLists.txt` | catkin build configuration |
| `config/signboard_rknn_test.yaml` | All tunable parameters |
| `launch/signboard_rknn_x11_speak_test.launch` | Launch camera, TTS, speech, viewer, and node |
| `scripts/signboard_rknn_test_node.py` | RKNN inference, consensus, debug image, speech |
| `scripts/check_signboard_rknn_test.py` | Runtime diagnostics |
| `README.md` | Usage instructions |

---

### Task 1: Package Metadata

**Files:**
- Create: `/Users/mikey/Downloads/ucar_2026_signboard_rknn_test/package.xml`
- Create: `/Users/mikey/Downloads/ucar_2026_signboard_rknn_test/CMakeLists.txt`

- [ ] **Step 1: Write package.xml**

```xml
<?xml version="1.0"?>
<package format="2">
  <name>ucar_2026_signboard_rknn_test</name>
  <version>0.1.0</version>
  <description>RKNN/NPU factory signboard camera, X11 view, and speech test package for U-CAR 2026.</description>
  <maintainer email="ucar@todo.todo">ucar</maintainer>
  <license>MIT</license>
  <buildtool_depend>catkin</buildtool_depend>
  <depend>cv_bridge</depend>
  <depend>rospy</depend>
  <depend>sensor_msgs</depend>
  <depend>std_msgs</depend>
  <depend>ucar_2026_competition_speech</depend>
  <exec_depend>image_view</exec_depend>
  <exec_depend>speech_command</exec_depend>
  <exec_depend>usb_cam</exec_depend>
  <exec_depend>yolo</exec_depend>
  <export></export>
</package>
```

- [ ] **Step 2: Write CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_signboard_rknn_test)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  ucar_2026_competition_speech
)
catkin_package()
catkin_install_python(PROGRAMS
  scripts/check_signboard_rknn_test.py
  scripts/signboard_rknn_test_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
```

---

### Task 2: Configuration & Launch

**Files:**
- Create: `/Users/mikey/Downloads/ucar_2026_signboard_rknn_test/config/signboard_rknn_test.yaml`
- Create: `/Users/mikey/Downloads/ucar_2026_signboard_rknn_test/launch/signboard_rknn_x11_speak_test.launch`

- [ ] **Step 1: Write YAML config**

```yaml
image_topic: /usb_cam/image_raw
detections_topic: /signboard_rknn_test/detections
debug_image_topic: /signboard_rknn_test/debug_image
status_topic: /signboard_rknn_test/status

model_path: $(find yolo)/models/cls_best.rknn
input_size: 640
confidence_threshold: 0.5
nms_iou_threshold: 0.45
inference_rate: 10.0
flip: false

consensus_confirm_frames: 3
consensus_release_frames: 3
consensus_timeout: 1.0
consensus_ema_alpha: 0.3

publish_debug: true
image_timeout: 5.0

enable_speech: true
announce_service: /competition_speech/announce
announce_service_timeout_sec: 0.5
speak_topic: /speak
use_announce_service: true
announce_event: custom
fallback_to_speak_topic: true
slow_speech: true
repeat_same: false
min_speech_interval_sec: 2.0
speech_wait: false
```

- [ ] **Step 2: Write launch file**

```xml
<launch>
  <arg name="start_camera" default="true" />
  <arg name="start_tts" default="true" />
  <arg name="start_competition_speech" default="true" />
  <arg name="start_viewer" default="true" />
  <arg name="camera_topic" default="/usb_cam/image_raw" />
  <arg name="debug_image_topic" default="/signboard_rknn_test/debug_image" />
  <arg name="config_file" default="$(find ucar_2026_signboard_rknn_test)/config/signboard_rknn_test.yaml" />
  <arg name="model_path" default="" />
  <arg name="flip" default="false" />
  <arg name="chars_per_second" default="2.0" />
  <arg name="required" default="false" />

  <group if="$(arg start_camera)">
    <include file="$(find usb_cam)/launch/usb_cam-test.launch" />
  </group>

  <group if="$(arg start_tts)">
    <node pkg="speech_command" type="voice_speak_node" name="voice_speak_node" output="screen"
          launch-prefix="bash -c 'sleep 1; exec &quot;$@&quot;' dummy" />
  </group>

  <group if="$(arg start_competition_speech)">
    <include file="$(find ucar_2026_competition_speech)/launch/competition_speech.launch">
      <arg name="chars_per_second" value="$(arg chars_per_second)" />
    </include>
  </group>

  <node pkg="ucar_2026_signboard_rknn_test"
        type="signboard_rknn_test_node.py"
        name="signboard_rknn_test_node"
        output="screen"
        required="$(arg required)"
        launch-prefix="bash -c 'sleep 2; exec &quot;$@&quot;' dummy">
    <rosparam command="load" file="$(arg config_file)" />
    <param name="image_topic" value="$(arg camera_topic)" />
    <param name="debug_image_topic" value="$(arg debug_image_topic)" />
    <param name="model_path" value="$(arg model_path)" />
    <param name="flip" value="$(arg flip)" />
  </node>

  <group if="$(arg start_viewer)">
    <node pkg="image_view" type="image_view" name="signboard_rknn_x11_viewer" output="screen"
          launch-prefix="bash -c 'sleep 4; exec &quot;$@&quot;' dummy">
      <remap from="image" to="$(arg debug_image_topic)" />
      <param name="image_transport" value="raw" />
    </node>
  </group>
</launch>
```

---

### Task 3: Core Inference Node

**Files:**
- Create: `/Users/mikey/Downloads/ucar_2026_signboard_rknn_test/scripts/signboard_rknn_test_node.py`

- [ ] **Step 1: Write complete node**

Copy the reference `traffic_light_rknn_test_node.py` verbatim, then change:
1. Node name → `signboard_rknn_test_node`
2. Class name → `SignboardRknnTestNode`
3. Topics prefix → `/signboard_rknn_test/`
4. `CLASS_NAMES` → `["food_processing", "daily_necessities", "electronics"]`
5. `CLASS_COLORS` → three distinct BGR colors
6. `SPEECH_TEXT` → Chinese signboard names
7. `ANNOUNCE_DECISION` → `{class: class}` (no special traffic decision mapping)
8. Remove the direction-lock logic in `update_consensus` (green_left/right oscillation guard is traffic-light specific)
9. Debug overlay text prefix → `SB:` instead of `TL:`

The node must keep all helper functions (`repair_logging_levels`, `letterbox`, `nms_boxes`, `infer_yolov5_input_size_from_count`, `resolve_model_path`, etc.) and both RKNN post-process paths (single-output and 3-head YOLOv5).

---

### Task 4: Diagnostics Script

**Files:**
- Create: `/Users/mikey/Downloads/ucar_2026_signboard_rknn_test/scripts/check_signboard_rknn_test.py`

- [ ] **Step 1: Write check script**

Copy the reference `check_traffic_light_rknn_test.py`, change model name to `cls_best.rknn`, node name to `signboard_rknn_test_node`, and topic names to `/signboard_rknn_test/*`.

---

### Task 5: Documentation

**Files:**
- Create: `/Users/mikey/Downloads/ucar_2026_signboard_rknn_test/README.md`

- [ ] **Step 1: Write README**

Include:
- Package purpose (signboard inference test)
- Quick start with `roslaunch`
- Parameter overrides (`start_camera:=false`, etc.)
- Topic list
- Diagnostics commands
- Class label mapping table

---

### Task 6: Validation

- [ ] **Step 1: File existence check**
  Run `find /Users/mikey/Downloads/ucar_2026_signboard_rknn_test -type f` and verify all 7 files exist.

- [ ] **Step 2: Python syntax check**
  Run `python3 -m py_compile` on both `.py` scripts.

- [ ] **Step 3: Launch XML lint**
  Verify launch file well-formedness.

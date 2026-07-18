#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise the deployed ROS node with a known corrected image."""

from __future__ import annotations

import argparse
import os
import shlex
import time
from pathlib import Path


def execute(client, command, allow_failure=False):
    stdin, stdout, stderr = client.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if code and not allow_failure:
        raise RuntimeError("remote command failed {}: {}\n{}".format(code, command, err))
    return code, out, err


def start_background(client, command, log_path):
    wrapped = "nohup setsid bash -lc {} >{} 2>&1 < /dev/null & echo $!".format(
        shlex.quote(command), shlex.quote(log_path)
    )
    _, out, _ = execute(client, wrapped)
    return int(out.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Corrected, non-mirrored 640x480 image")
    parser.add_argument("--expected", required=True)
    parser.add_argument("--host", default="192.168.1.6")
    parser.add_argument("--user", default="ucar")
    parser.add_argument("--password-env", default="UCAR_SSH_PASSWORD")
    parser.add_argument("--workspace", default="/home/ucar/2026-xunfei-race")
    parser.add_argument("--ros-setup", default="/home/ucar/ucar_ws/devel/setup.bash")
    parser.add_argument("--with-speech", action="store_true")
    args = parser.parse_args()
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError("missing {}".format(args.password_env))
    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(str(image))
    remote_corrected = "/tmp/traffic_ros_smoke_corrected.jpg"
    remote_raw = "/tmp/traffic_ros_smoke_raw.jpg"
    remote_publisher = "/tmp/traffic_ros_smoke_publish.py"
    launch_log = "/tmp/traffic_ros_smoke_launch.log"
    master_log = "/tmp/traffic_ros_smoke_master.log"

    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=10)
    pids = []
    try:
        sftp = client.open_sftp()
        try:
            sftp.put(str(image), remote_corrected)
            publisher_source = """#!/usr/bin/env python3
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
frame = cv2.imread('/tmp/traffic_ros_smoke_raw.jpg')
assert frame is not None
rospy.init_node('traffic_ros_smoke_publisher')
publisher = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=1)
bridge = CvBridge()
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    message = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
    message.header.stamp = rospy.Time.now()
    publisher.publish(message)
    rate.sleep()
"""
            with sftp.file(remote_publisher, "w") as handle:
                handle.write(publisher_source)
            sftp.chmod(remote_publisher, 0o755)
        finally:
            sftp.close()
        execute(
            client,
            "python3 -c \"import cv2; x=cv2.imread('{}'); assert x is not None; "
            "assert cv2.imwrite('{}', cv2.flip(x, 1))\"".format(remote_corrected, remote_raw),
        )
        env = "source {0} && source {1}/devel/setup.bash".format(args.ros_setup, args.workspace)
        master_ok, _, _ = execute(client, "bash -lc 'source {} && rosnode list'".format(args.ros_setup), True)
        if master_ok != 0:
            pids.append(start_background(client, env + " && exec roscore", master_log))
            time.sleep(3.0)
        start_tts = "true" if args.with_speech else "false"
        start_announcer = "true" if args.with_speech else "false"
        launch_command = (
            env
            + " && exec roslaunch ucar_2026_traffic_light_rknn_test "
            "traffic_light_rknn_x11_speak_test.launch start_camera:=false start_tts:={} "
            "start_competition_speech:={} start_viewer:=false enable_speech:=false required:=true"
        ).format(start_tts, start_announcer)
        pids.append(start_background(client, launch_command, launch_log))
        time.sleep(7.0)
        if args.with_speech:
            speech_code, speech_graph, speech_error = execute(
                client,
                "bash -lc '{} && rosnode list; rostopic info /speak; "
                "rosnode info /traffic_light_external_tts'".format(env),
                True,
            )
            print(speech_graph)
            if speech_error:
                print(speech_error)
            if (
                speech_code != 0
                or "/traffic_light_external_tts" not in speech_graph
                or "Subscribers:" not in speech_graph
                or "/voice_speak_node" not in speech_graph
            ):
                _, log_text, _ = execute(client, "tail -n 160 {}".format(launch_log), True)
                raise RuntimeError("TTS graph is incomplete; launch log:\n{}".format(log_text))
            announce_code, announce_out, announce_error = execute(
                client,
                "bash -lc '{} && rosrun ucar_2026_competition_speech announce_once.py "
                "_event:=custom _text:=traffic_light_voice_regression_fixed _wait:=false'".format(env),
                True,
            )
            print(announce_out)
            if announce_error:
                print(announce_error)
            if announce_code != 0:
                raise RuntimeError("announcement service smoke failed")
        execute(
            client,
            "pkill -TERM -f '[/]tmp/traffic_ros_smoke_raw.jpg' 2>/dev/null || true",
            True,
        )
        publisher_command = env + " && exec python3 {}".format(remote_publisher)
        pids.append(start_background(client, publisher_command, "/tmp/traffic_ros_smoke_publisher.log"))
        time.sleep(3.0)
        _, graph_info, graph_error = execute(
            client,
            "bash -lc '{} && rosnode list; rostopic list; "
            "rosnode info /traffic_light_rknn_test_node; "
            "rosnode info /traffic_ros_smoke_publisher'".format(env),
            True,
        )
        print(graph_info)
        if graph_error:
            print(graph_error)
        echo_code, detection, error = execute(
            client,
            "bash -lc '{} && timeout 10 rostopic echo -n1 /traffic_light_rknn_test/detections'".format(env),
            True,
        )
        print(detection)
        if error:
            print(error)
        compact = detection.replace(" ", "").replace("\\\"", '"')
        if echo_code != 0 or '"active":true' not in compact or args.expected not in compact:
            _, log_text, _ = execute(client, "tail -n 120 {}".format(launch_log), True)
            _, publisher_log, _ = execute(
                client, "tail -n 80 /tmp/traffic_ros_smoke_publisher.log", True
            )
            raise RuntimeError(
                "unexpected consensus; launch log:\n{}\npublisher log:\n{}".format(
                    log_text, publisher_log
                )
            )
        print("[OK] ROS consensus active:", args.expected)
    finally:
        for pid in reversed(pids):
            execute(client, "kill -TERM -- -{} 2>/dev/null || true".format(pid), True)
        time.sleep(1.0)
        execute(
            client,
            "rm -f /tmp/traffic_ros_smoke_corrected.jpg /tmp/traffic_ros_smoke_raw.jpg "
            "/tmp/traffic_ros_smoke_publish.py "
            "/tmp/traffic_ros_smoke_launch.log /tmp/traffic_ros_smoke_master.log "
            "/tmp/traffic_ros_smoke_publisher.log",
            True,
        )
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

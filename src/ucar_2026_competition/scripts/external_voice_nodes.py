#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run legacy ASR/TTS without sourcing the duplicate-filled legacy workspace."""

import os
import signal
import subprocess

import rospy


class ExternalVoiceNodes:
    def __init__(self):
        robot_ws = os.path.expanduser(rospy.get_param("~robot_ws", "/home/ucar/ucar_ws"))
        package_dir = os.path.join(robot_ws, "src", "speech_command")
        lib_dir = os.path.join(robot_ws, "devel", "lib", "speech_command")
        requested = []
        if rospy.get_param("~start_asr", False):
            requested.append(os.path.join(lib_dir, "speech_command_node"))
        if rospy.get_param("~start_tts", False):
            requested.append(os.path.join(lib_dir, "voice_speak_node"))
        if not requested:
            raise RuntimeError("no external voice process requested")
        if not os.path.isfile(os.path.join(package_dir, "package.xml")):
            raise RuntimeError("external speech_command package is unavailable: {}".format(package_dir))
        for path in requested:
            if not os.path.isfile(path) or not os.access(path, os.X_OK):
                raise RuntimeError("external voice executable is unavailable: {}".format(path))

        # The legacy nodes call ros::package::getPath("speech_command") to locate
        # aiui.cfg, audio files and SDK resources at runtime.  Starting only the
        # binaries therefore leaves AIUI unconfigured.  Add the package directory
        # itself instead of the whole legacy src tree, whose duplicate packages
        # would contaminate the current catkin workspace.
        child_env = os.environ.copy()
        current_package_path = child_env.get("ROS_PACKAGE_PATH", "")
        child_env["ROS_PACKAGE_PATH"] = os.pathsep.join(
            path for path in (package_dir, current_package_path) if path
        )
        self.processes = [
            subprocess.Popen([path], env=child_env, start_new_session=True) for path in requested
        ]
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "started legacy voice executable(s) with speech package path %s: %s",
            package_dir,
            requested,
        )

    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            for process in self.processes:
                if process.poll() is not None:
                    raise RuntimeError("legacy voice executable exited with code {}".format(process.returncode))
            rate.sleep()

    def shutdown(self):
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                except OSError:
                    process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    rospy.init_node("external_voice_nodes")
    node = ExternalVoiceNodes()
    try:
        node.run()
    finally:
        node.shutdown()

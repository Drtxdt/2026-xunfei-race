#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the legacy ASR/TTS binaries directly, without rospack scanning its workspace."""

import os
import signal
import subprocess

import rospy


class ExternalVoiceNodes:
    def __init__(self):
        robot_ws = os.path.expanduser(rospy.get_param("~robot_ws", "/home/ucar/ucar_ws"))
        lib_dir = os.path.join(robot_ws, "devel", "lib", "speech_command")
        requested = []
        if rospy.get_param("~start_asr", False):
            requested.append(os.path.join(lib_dir, "speech_command_node"))
        if rospy.get_param("~start_tts", False):
            requested.append(os.path.join(lib_dir, "voice_speak_node"))
        if not requested:
            raise RuntimeError("no external voice process requested")
        for path in requested:
            if not os.path.isfile(path) or not os.access(path, os.X_OK):
                raise RuntimeError("external voice executable is unavailable: {}".format(path))
        self.processes = [subprocess.Popen([path], start_new_session=True) for path in requested]
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("started legacy voice executable(s): %s", requested)

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best-effort ALSA volume setup before the speech node is used."""

import shutil
import subprocess

import rospy


def set_volume(control, volume):
    control = str(control or "").strip()
    volume = str(volume or "").strip()
    if not control or not volume:
        return
    if shutil.which("amixer") is None:
        rospy.logwarn("audio_setup: amixer not found, skip %s", control)
        return

    try:
        completed = subprocess.run(
            ["amixer", "set", control, volume],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2.0,
            check=False,
        )
    except Exception as exc:
        rospy.logwarn("audio_setup: failed to set %s: %s", control, exc)
        return

    if completed.returncode == 0:
        rospy.loginfo("audio_setup: set %s to %s", control, volume)
    else:
        rospy.logwarn(
            "audio_setup: amixer set %s failed: %s",
            control,
            completed.stderr.decode("utf-8", "ignore").strip(),
        )


def main():
    rospy.init_node("audio_setup_node")
    if not bool(rospy.get_param("~enabled", True)):
        rospy.loginfo("audio_setup: disabled")
        return

    set_volume("Master", rospy.get_param("~master_volume", "90%"))
    set_volume("PCM", rospy.get_param("~pcm_volume", "90%"))
    set_volume(rospy.get_param("~extra_control", ""), rospy.get_param("~extra_volume", ""))


if __name__ == "__main__":
    main()

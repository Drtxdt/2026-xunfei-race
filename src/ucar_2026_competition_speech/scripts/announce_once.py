#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call the competition announcement service once from roslaunch."""

from __future__ import annotations

import rospy
from ucar_2026_competition_speech.srv import Announce


def main() -> None:
    rospy.init_node("competition_announce_once")
    service_name = rospy.get_param("~service_name", "/competition_speech/announce")
    rospy.wait_for_service(service_name)
    announce = rospy.ServiceProxy(service_name, Announce)
    response = announce(
        rospy.get_param("~event", ""),
        rospy.get_param("~item", ""),
        rospy.get_param("~workshop", ""),
        rospy.get_param("~decision", ""),
        rospy.get_param("~text", ""),
        bool(rospy.get_param("~wait", True)),
    )
    if response.success:
        rospy.loginfo("Announcement succeeded: %s", response.speech_text)
    else:
        rospy.logerr("Announcement failed: %s", response.message)


if __name__ == "__main__":
    main()

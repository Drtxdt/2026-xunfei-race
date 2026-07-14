#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small client interface for the 2026 competition speech module.

Typical use in an existing rospy node:

    from ucar_2026_competition_speech.competition_speech_client import (
        CompetitionSpeechClient,
    )

    speech = CompetitionSpeechClient()
    speech.task2("香蕉", "食品加工车间")
    speech.task4("left")
    speech.task5()

This file does not call rospy.init_node().  The node that imports it should
already be a ROS node.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import rospy
from std_msgs.msg import String
from ucar_2026_competition_speech.srv import Announce

from ucar_2026_competition_speech.speech_templates import (
    build_announcement,
    estimate_duration,
)


class CompetitionSpeechClient:
    """Convenience wrapper around /competition_speech/announce.

    Parameters can be overridden in code or through private ROS params:
      ~announce_service
      ~speak_topic
      ~speech_wait
      ~speech_service_timeout_sec
      ~speech_fallback_to_topic
    """

    def __init__(
        self,
        service_name: Optional[str] = None,
        speak_topic: Optional[str] = None,
        wait: Optional[bool] = None,
        service_timeout_sec: Optional[float] = None,
        fallback_to_topic: Optional[bool] = None,
    ) -> None:
        self.service_name = service_name or rospy.get_param(
            "~announce_service", "/competition_speech/announce"
        )
        self.speak_topic = speak_topic or rospy.get_param("~speak_topic", "/speak")
        self.wait = bool(
            rospy.get_param("~speech_wait", True) if wait is None else wait
        )
        self.service_timeout_sec = float(
            rospy.get_param("~speech_service_timeout_sec", 1.0)
            if service_timeout_sec is None
            else service_timeout_sec
        )
        self.fallback_to_topic = bool(
            rospy.get_param("~speech_fallback_to_topic", True)
            if fallback_to_topic is None
            else fallback_to_topic
        )

        self._proxy = None
        self._speak_pub = None

    def announce(
        self,
        event: str,
        item: str = "",
        workshop: str = "",
        decision: str = "",
        text: str = "",
        wait: Optional[bool] = None,
    ):
        """Call the official competition speech service."""
        use_wait = self.wait if wait is None else bool(wait)

        try:
            self._ensure_proxy()
            response = self._proxy(event, item, workshop, decision, text, use_wait)
            if not response.success:
                rospy.logwarn("Competition speech failed: %s", response.message)
            return response
        except Exception as exc:
            rospy.logwarn(
                "Competition speech service unavailable (%s): %s",
                self.service_name,
                exc,
            )
            if not self.fallback_to_topic:
                return self._result(False, "", 0.0, str(exc))
            return self._publish_fallback(
                event,
                item=item,
                workshop=workshop,
                decision=decision,
                text=text,
                wait=use_wait,
            )

    def task1(self, text: str, wait: Optional[bool] = None):
        """Announce the complete task-1 reasoning text."""
        return self.announce("task1", text=text, wait=wait)

    def task2(self, item: str, workshop: str, wait: Optional[bool] = None):
        """Announce: 已将[货品名称]放入[仓库类别]."""
        return self.announce("task2", item=item, workshop=workshop, wait=wait)

    def task3(self, item: str, workshop: str, wait: Optional[bool] = None):
        """Announce: 仿真任务已完成，已将[货品名称]放入[仓库类别]."""
        return self.announce("task3", item=item, workshop=workshop, wait=wait)

    def task4(self, decision: str, wait: Optional[bool] = None):
        """Announce traffic decision: left/right/straight/stop."""
        return self.announce("task4", decision=decision, wait=wait)

    def task5(self, wait: Optional[bool] = None):
        """Announce final completion: 任务完成."""
        return self.announce("task5", wait=wait)

    def custom(self, text: str, wait: Optional[bool] = None):
        """Announce custom text through the same serialized speech gateway."""
        return self.announce("custom", text=text, wait=wait)

    def left(self, wait: Optional[bool] = None):
        return self.task4("left", wait=wait)

    def right(self, wait: Optional[bool] = None):
        return self.task4("right", wait=wait)

    def straight(self, wait: Optional[bool] = None):
        return self.task4("straight", wait=wait)

    def stop(self, wait: Optional[bool] = None):
        return self.task4("stop", wait=wait)

    def _ensure_proxy(self) -> None:
        if self._proxy is not None:
            return
        rospy.wait_for_service(self.service_name, timeout=self.service_timeout_sec)
        self._proxy = rospy.ServiceProxy(self.service_name, Announce)

    def _ensure_pub(self) -> None:
        if self._speak_pub is None:
            self._speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=10)

    def _publish_fallback(
        self,
        event: str,
        item: str = "",
        workshop: str = "",
        decision: str = "",
        text: str = "",
        wait: bool = True,
    ):
        try:
            event, speech_text = build_announcement(
                event, item=item, workshop=workshop, decision=decision, text=text
            )
        except ValueError as exc:
            return self._result(False, "", 0.0, str(exc))

        duration = estimate_duration(speech_text)
        self._ensure_pub()

        deadline = rospy.Time.now() + rospy.Duration(1.0)
        while (
            self._speak_pub.get_num_connections() == 0
            and rospy.Time.now() < deadline
            and not rospy.is_shutdown()
        ):
            rospy.sleep(0.05)

        self._speak_pub.publish(String(data=speech_text))
        if wait:
            rospy.sleep(duration)

        return self._result(
            True,
            speech_text,
            duration,
            "published to /speak fallback",
        )

    @staticmethod
    def _result(success: bool, speech_text: str, duration: float, message: str):
        return SimpleNamespace(
            success=success,
            speech_text=speech_text,
            estimated_duration=duration,
            message=message,
        )


_default_client = None


def get_default_client() -> CompetitionSpeechClient:
    global _default_client
    if _default_client is None:
        _default_client = CompetitionSpeechClient()
    return _default_client


def announce(
    event: str,
    item: str = "",
    workshop: str = "",
    decision: str = "",
    text: str = "",
    wait: Optional[bool] = None,
):
    return get_default_client().announce(
        event,
        item=item,
        workshop=workshop,
        decision=decision,
        text=text,
        wait=wait,
    )


def task1(text: str, wait: Optional[bool] = None):
    return get_default_client().task1(text, wait=wait)


def task2(item: str, workshop: str, wait: Optional[bool] = None):
    return get_default_client().task2(item, workshop, wait=wait)


def task3(item: str, workshop: str, wait: Optional[bool] = None):
    return get_default_client().task3(item, workshop, wait=wait)


def task4(decision: str, wait: Optional[bool] = None):
    return get_default_client().task4(decision, wait=wait)


def task5(wait: Optional[bool] = None):
    return get_default_client().task5(wait=wait)


def custom(text: str, wait: Optional[bool] = None):
    return get_default_client().custom(text, wait=wait)

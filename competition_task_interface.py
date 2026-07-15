#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface helpers for the 2026 smart-factory task flow.

This file is meant to be imported by your own ROS Python node.
It does not call rospy.init_node().

Recommended task-1 flow:

    from competition_task_interface import CompetitionTaskInterface

    task = CompetitionTaskInterface()

    result = task.task1_after_scan(
        item_a="香蕉",
        item_b="毛巾",
        item_c="手机",
        voice_instruction="小飞小飞，前往物品领取区，取得食品类，放置在对应仓库，并领取仿真环境中需要的日用品类放置在对应仓库",
    )

    print(result.pickup_item)
    print(result.pickup_workshop)

Important:
    task1_after_scan() should be called after QR scanning is finished and before
    subtask 2 navigation.  This matches the competition rule: task 1 is complete
    after the robot announces the reasoning result in the pickup area.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import rospy
from std_msgs.msg import String
from ucar_2026_competition_speech.srv import Announce
from ucar_2026_smart_factory_llm.srv import ReasonPickupOrder


class CompetitionSpeechClient:
    """Small wrapper around /competition_speech/announce."""

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
            rospy.get_param("~speech_service_timeout_sec", 2.0)
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
        use_wait = self.wait if wait is None else bool(wait)

        try:
            self._ensure_proxy()
            response = self._proxy(event, item, workshop, decision, text, use_wait)
            if not response.success:
                rospy.logwarn("Speech announce failed: %s", response.message)
            return response
        except Exception as exc:
            rospy.logwarn("Speech service unavailable: %s", exc)
            if not self.fallback_to_topic:
                return self._result(False, "", 0.0, str(exc))
            return self.publish_text_directly(text or self._simple_text(event, item, workshop, decision), wait=use_wait)

    def task1(self, announcement_full: str, wait: Optional[bool] = None):
        """Subtask 1: announce LLM reasoning result after QR scan."""
        return self.announce("task1", text=announcement_full, wait=wait)

    def task1_result(self, reason_result, wait: Optional[bool] = None):
        """Announce task-1 text from ReasonPickupOrder response."""
        return self.task1(reason_result.announcement_full, wait=wait)

    def task2(self, item: str, workshop: str, wait: Optional[bool] = None):
        """Subtask 2: announce physical warehouse placement."""
        return self.announce("task2", item=item, workshop=workshop, wait=wait)

    def task3(self, item: str, workshop: str, wait: Optional[bool] = None):
        """Subtask 3: announce simulation completion."""
        return self.announce("task3", item=item, workshop=workshop, wait=wait)

    def task4(self, decision: str, wait: Optional[bool] = None):
        """Subtask 4: announce traffic decision."""
        return self.announce("task4", decision=decision, wait=wait)

    def task5(self, wait: Optional[bool] = None):
        """Subtask 5: announce final completion."""
        return self.announce("task5", wait=wait)

    def custom(self, text: str, wait: Optional[bool] = None):
        return self.announce("custom", text=text, wait=wait)

    def left(self, wait: Optional[bool] = None):
        return self.task4("left", wait=wait)

    def right(self, wait: Optional[bool] = None):
        return self.task4("right", wait=wait)

    def straight(self, wait: Optional[bool] = None):
        return self.task4("straight", wait=wait)

    def stop(self, wait: Optional[bool] = None):
        return self.task4("stop", wait=wait)

    def publish_text_directly(self, text: str, wait: bool = True):
        """Fallback: publish directly to /speak when the gateway is unavailable."""
        if not text:
            return self._result(False, "", 0.0, "empty speech text")

        if self._speak_pub is None:
            self._speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=10)

        deadline = rospy.Time.now() + rospy.Duration(1.0)
        while (
            self._speak_pub.get_num_connections() == 0
            and rospy.Time.now() < deadline
            and not rospy.is_shutdown()
        ):
            rospy.sleep(0.05)

        self._speak_pub.publish(String(data=text))
        duration = max(2.0, len(text) / 3.0)
        if wait:
            rospy.sleep(duration)
        return self._result(True, text, duration, "published to /speak fallback")

    def _ensure_proxy(self) -> None:
        if self._proxy is not None:
            return
        rospy.wait_for_service(self.service_name, timeout=self.service_timeout_sec)
        self._proxy = rospy.ServiceProxy(self.service_name, Announce)

    @staticmethod
    def _simple_text(event: str, item: str, workshop: str, decision: str) -> str:
        if event == "task2":
            return "已将{}放入{}".format(item, workshop)
        if event == "task3":
            return "仿真任务已完成，已将{}放入{}".format(item, workshop)
        if event == "task4":
            return {
                "left": "左转",
                "right": "右转",
                "straight": "直行",
                "stop": "停止",
            }.get(decision, decision)
        if event == "task5":
            return "任务完成"
        return ""

    @staticmethod
    def _result(success: bool, speech_text: str, duration: float, message: str):
        return SimpleNamespace(
            success=success,
            speech_text=speech_text,
            estimated_duration=duration,
            message=message,
        )


class SmartFactoryLLMClient:
    """Wrapper around /smart_factory_llm/reason_pickup_order."""

    def __init__(
        self,
        service_name: Optional[str] = None,
        service_timeout_sec: Optional[float] = None,
    ) -> None:
        self.service_name = service_name or rospy.get_param(
            "~llm_service", "/smart_factory_llm/reason_pickup_order"
        )
        self.service_timeout_sec = float(
            rospy.get_param("~llm_service_timeout_sec", 90.0)
            if service_timeout_sec is None
            else service_timeout_sec
        )
        self._proxy = None

    def reason_pickup_order(
        self,
        item_a: str,
        item_b: str,
        item_c: str,
        voice_instruction: str,
    ):
        """Ask Spark X2 which item/category/workshop should be selected.

        Call this after all three QR codes are scanned.
        The response contains:
            pickup_item, pickup_major, pickup_workshop
            sim_item, sim_major, sim_workshop
            announcement_full
        """
        item_a = item_a.strip()
        item_b = item_b.strip()
        item_c = item_c.strip()
        voice_instruction = voice_instruction.strip()

        if not item_a or not item_b or not item_c:
            raise ValueError("item_a/item_b/item_c cannot be empty")
        if not voice_instruction:
            raise ValueError("voice_instruction cannot be empty")

        self._ensure_proxy()
        response = self._proxy(item_a, item_b, item_c, voice_instruction)
        if not response.success:
            raise RuntimeError(response.error_message)
        return response

    def _ensure_proxy(self) -> None:
        if self._proxy is not None:
            return
        rospy.wait_for_service(self.service_name, timeout=self.service_timeout_sec)
        self._proxy = rospy.ServiceProxy(self.service_name, ReasonPickupOrder)


class CompetitionTaskInterface:
    """High-level interface used by the main controller."""

    def __init__(
        self,
        llm: Optional[SmartFactoryLLMClient] = None,
        speech: Optional[CompetitionSpeechClient] = None,
    ) -> None:
        self.llm = llm or SmartFactoryLLMClient()
        self.speech = speech or CompetitionSpeechClient()

    def task1_after_scan(
        self,
        item_a: str,
        item_b: str,
        item_c: str,
        voice_instruction: str,
        announce: bool = True,
    ):
        """Subtask 1 correct rule flow.

        1. QR scanning has already produced three candidate item names.
        2. LLM extracts target category from voice_instruction.
        3. LLM maps the selected item to the target workshop.
        4. The robot announces the reasoning result while still in pickup area.
        5. Caller uses response.pickup_workshop to continue subtask 2.
        """
        result = self.llm.reason_pickup_order(
            item_a,
            item_b,
            item_c,
            voice_instruction,
        )

        if announce:
            speech_response = self.speech.task1(result.announcement_full, wait=True)
            if not speech_response.success:
                raise RuntimeError(speech_response.message)

        return result

    def announce_task2_from_task1(self, task1_result, wait: Optional[bool] = None):
        return self.speech.task2(
            task1_result.pickup_item,
            task1_result.pickup_workshop,
            wait=wait,
        )

    def announce_task3_from_task1(self, task1_result, wait: Optional[bool] = None):
        return self.speech.task3(
            task1_result.sim_item,
            task1_result.sim_workshop,
            wait=wait,
        )

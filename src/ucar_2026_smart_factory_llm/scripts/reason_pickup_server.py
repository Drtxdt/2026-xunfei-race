#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 服务：子任务1「智能接单与货品筛选」中大模型推理部分。

输入：三个二维码 JSON 中的子类名称 + 语音指令全文。
输出：结构化归类结果 + 赛方要求格式的两句播报文案。

依赖星火 Spark X2 HTTP 接口（OpenAI Chat Completions 兼容）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import rospy
from ucar_2026_smart_factory_llm.srv import (
    ReasonPickupOrder,
    ReasonPickupOrderRequest,
    ReasonPickupOrderResponse,
)

_LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是第21届全国大学生智能汽车竞赛「讯飞智慧工厂」赛项的调度推理模块。
你必须严格根据用户给出的三个货品名称（来自现场二维码返回的 JSON 字段 result）和用户语音指令，完成语义推理。

大类与目标车间对应关系（播报时必须使用下列车间全名）：
- 食品、食品加工类、生鲜、食材等相关大类 → 车间名：食品加工车间；对外表述可用「食品大类」。
- 日用品、日化、纺织、清洁用品等相关大类 → 车间名：日用品加工车间；对外表述可用「日用品大类」。
- 电子产品、数码、电器等相关大类 → 车间名：电子产品生产车间；对外表述可用「电子产品大类」。

推理要求：
1. 从语音中解析两个目标：①物品领取区要取的「目标大类」；②仿真环境中要取的「目标大类」。若语音未明确写出，依据指令里出现的「取得…」「放置在…」等语义尽力推断；仍无法确定时以 null 表示并在 err_hint 说明。
2. 判断三个货品各自属于哪一大类（食品/日用品/电子产品）。
3. 对①②各自在三个货品中选出唯一最匹配的一项；若多个候选，选与语音关键词最贴近的一项。

只输出一个 JSON 对象，不要 Markdown，不要代码围栏。键必须齐全，格式如下：
{
  "pickup_item": "字符串或null",
  "pickup_major": "食品大类|日用品大类|电子产品大类之一或null",
  "pickup_workshop": "食品加工车间|日用品加工车间|电子产品生产车间之一或null",
  "sim_item": "字符串或null",
  "sim_major": "同上或null",
  "sim_workshop": "同上或null",
  "announcement_physical": "取得X属于Y应放置在Z",
  "announcement_simulation": "仿真环境中取得X属于Y应放置在Z",
  "err_hint": "无问题时为空字符串"
}

announcement 句式必须与赛题一致：
- announcement_physical 必须以「取得」开头，包含「属于」「应放置在」，车间为上述三个全名之一。
- announcement_simulation 必须以「仿真环境中取得」开头（中间不要逗号），同样包含「属于」「应放置在」。
"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _parse_llm_json(content: str) -> Dict[str, Any]:
    raw = _strip_code_fence(content)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("模型输出中未找到 JSON 对象")
    return json.loads(raw[start : end + 1])


class SparkX2Client:
    """星火 X2 HTTP Chat Completions（Bearer APIPassword）。"""

    def __init__(
        self,
        api_password: str,
        base_url: str,
        model: str,
        timeout_sec: float,
    ) -> None:
        self._api_password = api_password.strip()
        self._url = base_url.strip()
        self._model = model.strip()
        self._timeout = timeout_sec
        if not self._api_password:
            raise ValueError("api_password 为空，请设置 XF_SPARK_API_PASSWORD 或 ROS 参数 api_password")

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={
                "Authorization": "Bearer {}".format(self._api_password),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError("Spark HTTP {}: {}".format(e.code, detail)) from e
        except urllib.error.URLError as e:
            raise RuntimeError("Spark 网络错误: {}".format(e)) from e

        decoded = json.loads(body)
        choices = decoded.get("choices") or []
        if not choices:
            raise RuntimeError("Spark 响应无 choices: {}".format(body[:800]))
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not content:
            raise RuntimeError("Spark 响应无 content: {}".format(body[:800]))
        return str(content)


def _build_user_prompt(items: List[str], voice: str) -> str:
    lines = [
        "三个货品名称（与车载视觉依次读取二维码的结果一致，可能对应食品/日用品/电子产品母类链接）：",
        "1) {}".format(items[0]),
        "2) {}".format(items[1]),
        "3) {}".format(items[2]),
        "",
        "用户语音指令全文：",
        voice.strip(),
    ]
    return "\n".join(lines)


def _fill_announcements(data: Dict[str, Any]) -> Tuple[str, str, str]:
    phy = (data.get("announcement_physical") or "").strip()
    sim = (data.get("announcement_simulation") or "").strip()

    def _line(prefix: str, item: str, major: str, workshop: str) -> str:
        item = item or ""
        major = major or ""
        workshop = workshop or ""
        if prefix == "sim":
            return "仿真环境中取得{}属于{}应放置在{}".format(item, major, workshop)
        return "取得{}属于{}应放置在{}".format(item, major, workshop)

    if not phy and data.get("pickup_item"):
        phy = _line(
            "phy",
            str(data.get("pickup_item") or ""),
            str(data.get("pickup_major") or ""),
            str(data.get("pickup_workshop") or ""),
        )
    if not sim and data.get("sim_item"):
        sim = _line(
            "sim",
            str(data.get("sim_item") or ""),
            str(data.get("sim_major") or ""),
            str(data.get("sim_workshop") or ""),
        )
    full = ""
    if phy and sim:
        full = "{}，{}".format(phy, sim)
    elif phy:
        full = phy
    else:
        full = sim
    return phy, sim, full


def _handle_request(req: ReasonPickupOrderRequest) -> ReasonPickupOrderResponse:
    res = ReasonPickupOrderResponse()
    res.success = False
    res.error_message = ""
    res.announcement_physical = ""
    res.announcement_simulation = ""
    res.announcement_full = ""
    res.pickup_item = ""
    res.pickup_major = ""
    res.pickup_workshop = ""
    res.sim_item = ""
    res.sim_major = ""
    res.sim_workshop = ""
    res.raw_model_reply = ""

    items = [req.item_a.strip(), req.item_b.strip(), req.item_c.strip()]
    if not all(items):
        res.error_message = "item_a/b/c 不能为空"
        return res
    voice = (req.voice_instruction or "").strip()
    if not voice:
        res.error_message = "voice_instruction 不能为空"
        return res

    api_password = rospy.get_param("~api_password", os.environ.get("XF_SPARK_API_PASSWORD", ""))
    base_url = rospy.get_param("spark_base_url", "https://spark-api-open.xf-yun.com/x2/chat/completions")
    model = rospy.get_param("spark_model", "spark-x")
    timeout = float(rospy.get_param("request_timeout_sec", 90.0))

    try:
        client = SparkX2Client(api_password, base_url, model, timeout)
    except ValueError as e:
        res.error_message = str(e)
        return res

    user_prompt = _build_user_prompt(items, voice)
    try:
        content = client.chat(SYSTEM_PROMPT, user_prompt)
    except Exception as e:  # noqa: BLE001
        rospy.logerr("Spark 调用失败: %s", e)
        res.error_message = str(e)
        return res

    res.raw_model_reply = content
    try:
        data = _parse_llm_json(content)
    except Exception as e:  # noqa: BLE001
        res.error_message = "解析模型 JSON 失败: {}".format(e)
        return res

    hint = (data.get("err_hint") or "").strip()
    if hint:
        rospy.logwarn("模型提示: %s", hint)

    phy, sim, full = _fill_announcements(data)

    res.pickup_item = str(data.get("pickup_item") or "")
    res.pickup_major = str(data.get("pickup_major") or "")
    res.pickup_workshop = str(data.get("pickup_workshop") or "")
    res.sim_item = str(data.get("sim_item") or "")
    res.sim_major = str(data.get("sim_major") or "")
    res.sim_workshop = str(data.get("sim_workshop") or "")
    res.announcement_physical = phy
    res.announcement_simulation = sim
    res.announcement_full = full

    if not res.pickup_item or not res.sim_item:
        res.error_message = "模型未给出完整的 pickup_item / sim_item"
        return res

    res.success = True
    return res


def main() -> None:
    rospy.init_node("smart_factory_llm_reason_pickup")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    rospy.Service("~reason_pickup_order", ReasonPickupOrder, _handle_request)
    pwd = rospy.get_param("~api_password", os.environ.get("XF_SPARK_API_PASSWORD", ""))
    if not str(pwd).strip():
        rospy.logwarn(
            "未配置 api_password / 环境变量 XF_SPARK_API_PASSWORD，服务将启动但调用时会失败。"
        )
    rospy.loginfo("smart_factory_llm: 服务已就绪 ~/reason_pickup_order")
    rospy.spin()


if __name__ == "__main__":
    main()

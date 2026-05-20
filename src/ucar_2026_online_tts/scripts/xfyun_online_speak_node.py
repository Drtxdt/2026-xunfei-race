#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subscribe to /speak and play iFlytek online TTS audio."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import subprocess
import threading
from datetime import datetime
from email.utils import formatdate
from time import mktime
from typing import List
from urllib.parse import urlencode, urlparse

import rospy
from std_msgs.msg import String

try:
    import websocket
except ImportError as exc:  # pragma: no cover - shown on robot if dependency is missing
    websocket = None
    _WEBSOCKET_IMPORT_ERROR = exc


class XfyunOnlineTts:
    def __init__(self) -> None:
        self._nh = rospy.get_name()
        self._host_url = rospy.get_param("~host_url", "wss://tts-api.xfyun.cn/v2/tts")
        self._speak_topic = rospy.get_param("~speak_topic", "/speak")
        self._app_id = rospy.get_param("~app_id", os.environ.get("XF_TTS_APPID", "")).strip()
        self._api_key = rospy.get_param("~api_key", os.environ.get("XF_TTS_API_KEY", "")).strip()
        self._api_secret = rospy.get_param("~api_secret", os.environ.get("XF_TTS_API_SECRET", "")).strip()
        self._voice_name = rospy.get_param("~voice_name", "x4_xiaoyan")
        self._speed = int(rospy.get_param("~speed", 50))
        self._volume = int(rospy.get_param("~volume", 80))
        self._pitch = int(rospy.get_param("~pitch", 50))
        self._sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self._audio_path = rospy.get_param("~audio_path", "/tmp/ucar_2026_online_tts.pcm")
        self._audio_device = rospy.get_param("~audio_device", "")
        self._lock = threading.Lock()

    def validate(self) -> bool:
        if websocket is None:
            rospy.logerr("Missing Python dependency: websocket-client (%s)", _WEBSOCKET_IMPORT_ERROR)
            rospy.logerr("Install on robot: python3 -m pip install --user websocket-client")
            return False
        missing = []
        if not self._app_id:
            missing.append("XF_TTS_APPID")
        if not self._api_key:
            missing.append("XF_TTS_API_KEY")
        if not self._api_secret:
            missing.append("XF_TTS_API_SECRET")
        if missing:
            rospy.logerr("Missing online TTS credentials: %s", ", ".join(missing))
            return False
        return True

    def auth_url(self) -> str:
        parsed = urlparse(self._host_url)
        date = formatdate(timeval=mktime(datetime.utcnow().timetuple()), localtime=False, usegmt=True)
        signature_origin = "host: {}\ndate: {}\nGET {} HTTP/1.1".format(
            parsed.netloc, date, parsed.path
        )
        signature_sha = hmac.new(
            self._api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature = base64.b64encode(signature_sha).decode("utf-8")
        authorization_origin = (
            'api_key="{}", algorithm="hmac-sha256", headers="host date request-line", '
            'signature="{}"'
        ).format(self._api_key, signature)
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
        query = urlencode({"authorization": authorization, "date": date, "host": parsed.netloc})
        return "{}://{}{}?{}".format(parsed.scheme, parsed.netloc, parsed.path, query)

    def synthesize(self, text: str) -> bytes:
        audio_chunks: List[bytes] = []
        errors: List[str] = []
        done = threading.Event()

        def on_open(ws) -> None:
            payload = {
                "common": {"app_id": self._app_id},
                "business": {
                    "aue": "raw",
                    "auf": "audio/L16;rate={}".format(self._sample_rate),
                    "vcn": self._voice_name,
                    "speed": self._speed,
                    "volume": self._volume,
                    "pitch": self._pitch,
                    "tte": "UTF8",
                },
                "data": {
                    "status": 2,
                    "text": base64.b64encode(text.encode("utf-8")).decode("utf-8"),
                },
            }
            ws.send(json.dumps(payload, ensure_ascii=False))

        def on_message(ws, message: str) -> None:
            try:
                response = json.loads(message)
            except json.JSONDecodeError as exc:
                errors.append("Invalid JSON from TTS: {}".format(exc))
                done.set()
                ws.close()
                return
            code = int(response.get("code", -1))
            if code != 0:
                errors.append("TTS API error {}: {}".format(code, response.get("message", "")))
                done.set()
                ws.close()
                return
            data = response.get("data") or {}
            audio = data.get("audio")
            if audio:
                audio_chunks.append(base64.b64decode(audio))
            if int(data.get("status", 0)) == 2:
                done.set()
                ws.close()

        def on_error(_ws, error) -> None:
            errors.append(str(error))
            done.set()

        def on_close(_ws, *_args) -> None:
            done.set()

        ws = websocket.WebSocketApp(
            self.auth_url(),
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        if not done.wait(float(rospy.get_param("~request_timeout_sec", 30.0))):
            ws.close()
            raise RuntimeError("TTS request timed out")
        if errors:
            raise RuntimeError("; ".join(errors))
        return b"".join(audio_chunks)

    def play_pcm(self, audio: bytes) -> None:
        with open(self._audio_path, "wb") as f:
            f.write(audio)
        cmd = ["aplay", "-q", "-f", "S16_LE", "-r", str(self._sample_rate), "-c", "1"]
        if self._audio_device:
            cmd.extend(["-D", self._audio_device])
        cmd.append(self._audio_path)
        subprocess.check_call(cmd)

    def speak_callback(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        with self._lock:
            try:
                rospy.loginfo("Online TTS speaking: %s", text)
                self.play_pcm(self.synthesize(text))
            except Exception as exc:  # noqa: BLE001
                rospy.logerr("Online TTS failed: %s", exc)

    def spin(self) -> None:
        rospy.Subscriber(self._speak_topic, String, self.speak_callback, queue_size=10)
        rospy.loginfo("Online TTS node ready on %s, voice=%s", self._speak_topic, self._voice_name)
        rospy.spin()


def main() -> None:
    rospy.init_node("xfyun_online_speak_node")
    node = XfyunOnlineTts()
    if not node.validate():
        return
    node.spin()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recognize the three fixed task commands from the legacy microphone PCM stream."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import queue
import ssl
import threading
import time
import urllib.parse
from email.utils import formatdate

import rospy
import websocket
from std_msgs.msg import String, UInt8MultiArray

from ucar_2026_competition.logic import parse_category, parse_task_categories


class FixedCommandIat:
    def __init__(self):
        self.audio_topic = rospy.get_param(
            "~audio_topic", "/speech_command_node/audio_pcm"
        )
        self.question_topic = rospy.get_param("~question_topic", "/question")
        self.text_topic = rospy.get_param("~text_topic", "/competition/iat_text")
        self.credentials_file = os.path.expanduser(
            rospy.get_param(
                "~credentials_file",
                "/home/ucar/.config/ucar_2026/iat_credentials.json",
            )
        )
        self.host = rospy.get_param("~host", "iat-api.xfyun.cn")
        self.path = rospy.get_param("~path", "/v2/iat")
        self.eos_ms = int(rospy.get_param("~eos_ms", 1500))
        self.stream_gap_sec = float(rospy.get_param("~stream_gap_sec", 0.8))
        self.max_session_sec = float(rospy.get_param("~max_session_sec", 55.0))
        self.result_wait_sec = float(rospy.get_param("~result_wait_sec", 3.0))
        self.reconnect_sec = float(rospy.get_param("~reconnect_sec", 0.5))

        self.appid, self.api_secret, self.api_key = self._load_credentials()
        self.audio_queue = queue.Queue(maxsize=512)
        self.shutdown_event = threading.Event()
        self.question_pub = rospy.Publisher(self.question_topic, String, queue_size=5)
        self.text_pub = rospy.Publisher(self.text_topic, String, queue_size=20)
        rospy.Subscriber(
            self.audio_topic, UInt8MultiArray, self._audio_cb, queue_size=100
        )
        rospy.on_shutdown(self.shutdown)

        # Do not route the robot's local speech connection through a desktop proxy.
        no_proxy = [value for value in os.environ.get("NO_PROXY", "").split(",") if value]
        if self.host not in no_proxy:
            no_proxy.append(self.host)
        os.environ["NO_PROXY"] = ",".join(no_proxy)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]

        self.worker = threading.Thread(target=self._worker, name="fixed-command-iat")
        self.worker.daemon = True
        self.worker.start()
        rospy.loginfo(
            "fixed command IAT ready: audio=%s question=%s",
            self.audio_topic,
            self.question_topic,
        )

    def _load_credentials(self):
        values = {
            "appid": os.environ.get("XF_IAT_APPID", "").strip(),
            "api_secret": os.environ.get("XF_IAT_API_SECRET", "").strip(),
            "api_key": os.environ.get("XF_IAT_API_KEY", "").strip(),
        }
        if not all(values.values()):
            try:
                with open(self.credentials_file, encoding="utf-8") as stream:
                    stored = json.load(stream)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "IAT credentials unavailable; run debug/repair_speech_command_asr.sh: {}".format(
                        exc
                    )
                )
            for key in values:
                values[key] = values[key] or str(stored.get(key, "")).strip()
        if not all(values.values()):
            raise RuntimeError("IAT credential file is incomplete: {}".format(self.credentials_file))
        return values["appid"], values["api_secret"], values["api_key"]

    def _audio_cb(self, msg):
        chunk = bytes(bytearray(msg.data))
        if not chunk or self.shutdown_event.is_set():
            return
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def _signed_url(self):
        date = formatdate(usegmt=True)
        signature_origin = "host: {}\ndate: {}\nGET {} HTTP/1.1".format(
            self.host, date, self.path
        )
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"),
                signature_origin.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("ascii")
        authorization_origin = (
            'api_key="{}", algorithm="hmac-sha256", '
            'headers="host date request-line", signature="{}"'
        ).format(self.api_key, signature)
        query = urllib.parse.urlencode(
            {
                "authorization": base64.b64encode(
                    authorization_origin.encode("utf-8")
                ).decode("ascii"),
                "date": date,
                "host": self.host,
            }
        )
        return "wss://{}{}?{}".format(self.host, self.path, query)

    def _frame(self, status, audio=b""):
        payload = {
            "data": {
                "status": status,
                "format": "audio/L16;rate=16000",
                "encoding": "raw",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        }
        if status == 0:
            payload["common"] = {"app_id": self.appid}
            payload["business"] = {
                "language": "zh_cn",
                "domain": "iat",
                "accent": "mandarin",
                "dwa": "wpgs",
                "eos": self.eos_ms,
            }
        return json.dumps(payload, separators=(",", ":"))

    def _publish_text(self, text, final=False):
        category = parse_category(text)
        payload = {
            "stamp": time.time(),
            "text": text,
            "category": category or "",
            "final": bool(final),
        }
        self.text_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        return category

    def _receive(self, ws, done, accepted):
        segments = {}
        try:
            while not self.shutdown_event.is_set() and not done.is_set():
                response = json.loads(ws.recv())
                code = int(response.get("code", -1))
                if code != 0:
                    rospy.logerr(
                        "fixed command IAT error %d: %s",
                        code,
                        response.get("message", "unknown error"),
                    )
                    break
                data = response.get("data") or {}
                result = data.get("result") or {}
                sn = int(result.get("sn", len(segments)))
                if result.get("pgs") == "rpl":
                    replace_range = result.get("rg") or []
                    if len(replace_range) == 2:
                        for old_sn in range(int(replace_range[0]), int(replace_range[1]) + 1):
                            segments.pop(old_sn, None)
                text = "".join(
                    str(candidates[0].get("w", ""))
                    for word in result.get("ws", [])
                    for candidates in [word.get("cw", [])]
                    if candidates
                )
                if text:
                    segments[sn] = text
                    full_text = "".join(segments[key] for key in sorted(segments))
                    is_final = data.get("status") == 2
                    self._publish_text(full_text, is_final)
                    physical, simulation = parse_task_categories(full_text)
                    command_complete = (
                        is_final
                        and physical is not None
                        and simulation is not None
                        and physical != simulation
                    )
                    if command_complete and not accepted.is_set():
                        accepted.set()
                        self.question_pub.publish(String(data=full_text))
                        rospy.loginfo(
                            "fixed command accepted: physical=%s simulation=%s",
                            physical,
                            simulation,
                        )
                if data.get("status") == 2:
                    break
        except (ValueError, websocket.WebSocketException, OSError) as exc:
            if not self.shutdown_event.is_set():
                rospy.logwarn("fixed command IAT receive ended: %s", exc)
        finally:
            done.set()

    def _run_session(self, first_chunk):
        ws = websocket.create_connection(
            self._signed_url(),
            timeout=10,
            sslopt={"cert_reqs": ssl.CERT_REQUIRED},
            http_proxy_host=None,
        )
        ws.settimeout(10)
        done = threading.Event()
        accepted = threading.Event()
        receiver = threading.Thread(
            target=self._receive, args=(ws, done, accepted), name="fixed-command-iat-rx"
        )
        receiver.daemon = True
        receiver.start()

        started = time.time()
        last_audio = started
        ws.send(self._frame(0, first_chunk))
        try:
            while not self.shutdown_event.is_set() and not done.is_set():
                if time.time() - started >= self.max_session_sec:
                    break
                try:
                    chunk = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    if time.time() - last_audio >= self.stream_gap_sec:
                        break
                    continue
                ws.send(self._frame(1, chunk))
                last_audio = time.time()
            if not done.is_set():
                ws.send(self._frame(2))
                done.wait(self.result_wait_sec)
        finally:
            done.set()
            try:
                ws.close(status=1000)
            except (TypeError, websocket.WebSocketException, OSError):
                ws.close()
            receiver.join(timeout=1.0)

    def _worker(self):
        while not self.shutdown_event.is_set() and not rospy.is_shutdown():
            try:
                first_chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run_session(first_chunk)
            except (ValueError, websocket.WebSocketException, OSError) as exc:
                if not self.shutdown_event.is_set():
                    rospy.logerr_throttle(5.0, "fixed command IAT connection failed: %s", exc)
                    self.shutdown_event.wait(self.reconnect_sec)

    def shutdown(self):
        self.shutdown_event.set()


def main():
    rospy.init_node("competition_fixed_command_iat")
    FixedCommandIat()
    rospy.spin()


if __name__ == "__main__":
    main()

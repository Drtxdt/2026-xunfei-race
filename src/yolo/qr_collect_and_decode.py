#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QR collector and decoder for ROS U-CAR competition.
- Subscribe /usb_cam/image_raw
- Save QR test images at fixed interval
- Decode one or multiple QR codes in camera image
- If QR content is URL, request it and parse returned JSON: {"code":200,"result":"xx"}
- Publish decoded result to /qr_code_results as std_msgs/String JSON
"""
import os
import time
import json
import argparse
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import requests
except Exception:
    requests = None

try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
except Exception:
    pyzbar = None
    ZBarSymbol = None


OFFLINE_ITEMS = {
    "food": ["苹果", "猪肉", "草莓", "饺子", "面条", "薯片", "馒头"],
    "daily": ["纸巾", "毛巾", "牙刷", "洗衣液", "T恤衫"],
    "electronic": ["手机", "耳机", "充电器", "鼠标", "数据线"],
}


class QRCollectAndDecode:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.detector = None
        try:
            self.detector = cv2.QRCodeDetector()
        except AttributeError:
            rospy.logwarn("cv2.QRCodeDetector not available (need opencv-contrib). "
                          "Will rely on pyzbar for QR decoding.")
        self.out_dir = os.path.expanduser(args.output)
        os.makedirs(self.out_dir, exist_ok=True)
        self.last_save_time = 0.0
        self.save_count = 0
        self.last_publish_text = ""
        self.last_publish_time = 0.0
        self.last_decode_time = 0.0
        self.url_cache = {}
        self.url_lock = threading.RLock()
        self.publish_lock = threading.Lock()
        self.pending_urls = set()
        self.url_next_allowed = {}
        self.fetch_executor = ThreadPoolExecutor(max_workers=3)
        self.decode_interval = max(0.0, float(args.decode_interval))
        self.decode_scales = self.parse_decode_scales(args.decode_scales)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.pub = rospy.Publisher(args.pub_topic, String, queue_size=10)
        self.status_pub = rospy.Publisher(
            args.status_topic, String, queue_size=10, latch=True)
        self.sub = rospy.Subscriber(args.topic, Image, self.image_cb, queue_size=1)
        rospy.on_shutdown(lambda: self.fetch_executor.shutdown(wait=False))

        rospy.loginfo("QR image topic: %s", args.topic)
        rospy.loginfo("QR images save to: %s", self.out_dir)
        rospy.loginfo("QR result topic: %s", args.pub_topic)
        rospy.loginfo(
            "QR decode interval: %.3fs, enhanced scales: %s",
            self.decode_interval,
            ",".join("{:.2f}".format(value) for value in self.decode_scales),
        )
        self.publish_decoder_status("ready")

    def publish_decoder_status(self, state):
        with self.url_lock:
            pending_count = len(self.pending_urls)
        payload = {
            "stamp": time.time(),
            "state": str(state),
            "pending_count": pending_count,
        }
        self.status_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))

    @staticmethod
    def parse_decode_scales(value):
        scales = []
        for token in str(value or "").split(','):
            try:
                scale = float(token.strip())
            except (TypeError, ValueError):
                continue
            if scale >= 1.0 and scale not in scales:
                scales.append(scale)
        if 1.0 not in scales:
            scales.insert(0, 1.0)
        return sorted(scales)

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.args.flip:
                img = cv2.flip(img, 1)

            now = time.time()
            if now - self.last_decode_time < self.decode_interval:
                return
            self.last_decode_time = now
            decoded_items = self.decode_qr(img)

            if decoded_items:
                results = []
                for text in decoded_items:
                    item = {
                        "raw": text,
                        "api": None,
                        "ok": False,
                        "result": None,
                        "error": None
                    }
                    if self.args.fetch and self.is_url(text):
                        self.schedule_url_resolution(text)
                        continue
                    results.append(item)

                if results:
                    self.publish_results(results, now)

                if self.args.save_on_detect:
                    self.save_image(img, prefix="qr_detect")

            if self.args.save_all and now - self.last_save_time >= self.args.interval:
                self.save_image(img, prefix="qr_raw")
                self.last_save_time = now

        except Exception as e:
            rospy.logerr("image_cb error: %s", str(e))

    def schedule_url_resolution(self, url):
        now = time.time()
        with self.url_lock:
            if url in self.pending_urls or now < self.url_next_allowed.get(url, 0.0):
                return
            self.pending_urls.add(url)
        self.publish_decoder_status("fetching")
        self.fetch_executor.submit(self.resolve_and_publish_url, url)

    def resolve_and_publish_url(self, url):
        started_at = time.monotonic()
        try:
            api_result = self.resolve_qr_url(url)
            item = {
                "raw": url,
                "api": api_result,
                "ok": bool(api_result and api_result.get("code") == 200),
                "result": api_result.get("result") if api_result else None,
                "error": None,
            }
            if api_result and api_result.get("code") != 200:
                item["error"] = "api_code_not_200"
            rospy.loginfo(
                "QR URL resolved in %.1fms: %s -> %s",
                (time.monotonic() - started_at) * 1000.0,
                url,
                item["result"],
            )
            self.publish_results([item], time.time())
        except Exception as exc:
            rospy.logwarn("QR URL resolution failed: %s -> %s", url, exc)
        finally:
            with self.url_lock:
                self.pending_urls.discard(url)
                self.url_next_allowed[url] = time.time() + self.args.repeat_period
                pending_count = len(self.pending_urls)
            self.publish_decoder_status(
                "fetching" if pending_count else "idle")

    def publish_results(self, results, stamp):
        payload = {"stamp": stamp, "count": len(results), "items": results}
        text_payload = json.dumps(payload, ensure_ascii=False)
        with self.publish_lock:
            rospy.loginfo("QR decoded: %s", text_payload)
            self.pub.publish(String(data=text_payload))
            self.last_publish_text = text_payload
            self.last_publish_time = stamp

    def decode_qr(self, img):
        texts = []

        # Fast path: retain the original OpenCV + pyzbar behavior on the
        # unmodified frame. Enhanced variants run only if this finds nothing.
        self.decode_opencv(img, texts)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.decode_pyzbar(gray, texts)
        if texts:
            return texts

        # Slow path for small/distant QR codes. CLAHE improves uneven light;
        # mild unsharp masking restores edges before cubic upscaling.
        enhanced = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)
        for scale in self.decode_scales:
            if scale == 1.0:
                candidate = sharpened
            else:
                candidate = cv2.resize(
                    sharpened,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            self.decode_opencv(candidate, texts)
            self.decode_pyzbar(candidate, texts)
        return texts

    def decode_opencv(self, image, texts):
        if self.detector is not None:
            try:
                ok, decoded_info, points, straight_qrcode = self.detector.detectAndDecodeMulti(image)
                if ok and decoded_info:
                    for s in decoded_info:
                        normalized = str(s or '').strip()
                        if normalized and normalized not in texts:
                            texts.append(normalized)
            except Exception:
                pass

    def decode_pyzbar(self, gray, texts):
        if pyzbar is not None:
            try:
                kwargs = {"symbols": [ZBarSymbol.QRCODE]} if ZBarSymbol is not None else {}
                codes = pyzbar.decode(gray, **kwargs)
                for code in codes:
                    s = code.data.decode('utf-8', errors='ignore').strip()
                    if s and s not in texts:
                        texts.append(s)
            except Exception:
                pass

    def is_url(self, text):
        return text.startswith('http://') or text.startswith('https://')

    def fetch_url(self, url):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "U-CAR-QR-Scanner/1.0"})
            with urllib.request.urlopen(req, timeout=self.args.timeout) as resp:
                body = resp.read().decode('utf-8', errors='ignore')
            try:
                return json.loads(body)
            except Exception:
                return {"code": -1, "result": None, "raw_body": body}
        except Exception as e:
            return {"code": -1, "result": None, "error": str(e)}

    def fetch_cached_url(self, url):
        with self.url_lock:
            if url in self.url_cache:
                return self.url_cache[url]

        attempts = max(1, int(self.args.fetch_retries))
        result = None
        for attempt in range(1, attempts + 1):
            result = self.fetch_url(url)
            if result and result.get("code") == 200:
                with self.url_lock:
                    self.url_cache[url] = result
                rospy.loginfo(
                    "QR URL cached on attempt %d/%d: %s -> %s",
                    attempt, attempts, url, result.get("result"))
                return result
            if attempt < attempts and not rospy.is_shutdown():
                rospy.logwarn(
                    "QR URL attempt %d/%d failed; retrying %s: %s",
                    attempt, attempts, url,
                    (result or {}).get("error") or "invalid response")
                time.sleep(max(0.0, float(self.args.retry_backoff)) * attempt)
        return result

    def resolve_qr_url(self, url):
        offline_mode = getattr(self.args, "offline_mode", "off")
        if offline_mode == "force":
            return self.fetch_offline_url(url, reason="forced")

        result = self.fetch_cached_url(url)
        if result and result.get("code") == 200:
            return result

        if offline_mode == "fallback":
            offline = self.fetch_offline_url(url, reason=(result or {}).get("error") or "fetch_failed")
            if offline:
                rospy.logwarn("QR URL fetch failed, using offline fallback: %s -> %s", url, offline.get("result"))
                return offline
        return result

    def fetch_offline_url(self, url, reason="offline"):
        category = self.category_from_url(url)
        if not category:
            return None
        items = OFFLINE_ITEMS[category]
        index = abs(hash(url)) % len(items)
        return {
            "code": 200,
            "result": items[index],
            "offline": True,
            "category": category,
            "reason": reason,
        }

    def category_from_url(self, url):
        try:
            parsed = urlparse(url)
            text = ("%s/%s" % (parsed.netloc, parsed.path)).lower()
        except Exception:
            text = str(url).lower()

        if "electronic" in text or "electronics" in text:
            return "electronic"
        if "daily" in text:
            return "daily"
        if "food" in text:
            return "food"
        return None

    def save_image(self, img, prefix):
        now = time.time()
        filename = "%s_%06d_%d.jpg" % (prefix, self.save_count, int(now * 1000))
        path = os.path.join(self.out_dir, filename)
        ok = cv2.imwrite(path, img)
        if ok:
            self.save_count += 1
            rospy.loginfo("saved %s", path)
        else:
            rospy.logwarn("failed to save image: %s", path)


def main():
    parser = argparse.ArgumentParser(description='QR code collector and decoder for ROS U-CAR')
    parser.add_argument('--topic', default='/usb_cam/image_raw', help='camera image topic')
    parser.add_argument('--output', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_dataset', 'qr_images'), help='QR image save dir')
    parser.add_argument('--pub-topic', default='/qr_code_data', help='publish decoded QR json to this topic')
    parser.add_argument('--status-topic', default='/qr_decoder/status',
                        help='publish decoder readiness and pending URL count')
    parser.add_argument('--fetch', action='store_true', help='if QR content is URL, request it and parse JSON')
    parser.add_argument('--save-on-detect', action='store_true', help='save image whenever QR is detected')
    parser.add_argument('--save-all', action='store_true', help='save raw images periodically even if no QR is detected')
    parser.add_argument('--interval', type=float, default=0.5, help='save interval when --save-all enabled')
    parser.add_argument('--repeat-period', type=float, default=2.0, help='republish same QR result after seconds')
    parser.add_argument('--timeout', type=float, default=3.0, help='HTTP timeout seconds')
    parser.add_argument('--fetch-retries', type=int, default=3,
                        help='HTTP attempts retained after a QR leaves the camera view')
    parser.add_argument('--retry-backoff', type=float, default=0.20,
                        help='base seconds between URL fetch retries')
    parser.add_argument('--decode-interval', type=float, default=0.10,
                        help='minimum seconds between decode attempts')
    parser.add_argument('--decode-scales', default='1.0,1.5,2.0',
                        help='comma-separated enhanced decode scales')
    parser.add_argument('--flip', action='store_true', help='horizontal flip image before decode')
    parser.add_argument('--offline-fallback', action='store_true', help='use local QR result if URL fetch fails')
    parser.add_argument('--offline-mode', choices=['off', 'fallback', 'force'], default=None,
                        help='off: only real URL; fallback: URL first then local; force: local only')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if args.offline_mode is None:
        args.offline_mode = 'fallback' if args.offline_fallback else 'off'

    rospy.init_node('qr_collect_and_decode')
    QRCollectAndDecode(args)
    rospy.spin()


if __name__ == '__main__':
    main()

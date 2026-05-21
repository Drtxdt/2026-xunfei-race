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
import urllib.request

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
except Exception:
    pyzbar = None


class QRCollectAndDecode:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.detector = None
        try:
            self.detector = cv2.QRCodeDetector()
        except AttributeError:
            rospy.logwarn(
                "cv2.QRCodeDetector is unavailable. Falling back to pyzbar."
            )
        if self.detector is None and pyzbar is None:
            rospy.logerr(
                "No QR decoder available. Install pyzbar/zbar or use OpenCV with QRCodeDetector."
            )
        self.out_dir = os.path.expanduser(args.output)
        os.makedirs(self.out_dir, exist_ok=True)
        self.last_save_time = 0.0
        self.save_count = 0
        self.last_publish_text = ""
        self.last_publish_time = 0.0

        self.pub = rospy.Publisher(args.pub_topic, String, queue_size=10)
        self.sub = rospy.Subscriber(args.topic, Image, self.image_cb, queue_size=1)

        rospy.loginfo("QR image topic: %s", args.topic)
        rospy.loginfo("QR images save to: %s", self.out_dir)
        rospy.loginfo("QR result topic: %s", args.pub_topic)

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.args.flip:
                img = cv2.flip(img, 1)

            now = time.time()
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
                        item["api"] = self.fetch_url(text)
                        if item["api"] and item["api"].get("code") == 200:
                            item["ok"] = True
                            item["result"] = item["api"].get("result")
                        elif item["api"]:
                            item["error"] = "api_code_not_200"
                    results.append(item)

                payload = {
                    "stamp": now,
                    "count": len(results),
                    "items": results
                }
                text_payload = json.dumps(payload, ensure_ascii=False)

                if text_payload != self.last_publish_text or now - self.last_publish_time > self.args.repeat_period:
                    rospy.loginfo("QR decoded: %s", text_payload)
                    self.pub.publish(String(data=text_payload))
                    self.last_publish_text = text_payload
                    self.last_publish_time = now

                if self.args.save_on_detect:
                    self.save_image(img, prefix="qr_detect")

            if self.args.save_all and now - self.last_save_time >= self.args.interval:
                self.save_image(img, prefix="qr_raw")
                self.last_save_time = now

        except Exception as e:
            rospy.logerr("image_cb error: %s", str(e))

    def decode_qr(self, img):
        texts = []

        # Method 1: OpenCV detectAndDecodeMulti
        if self.detector is not None:
            try:
                ok, decoded_info, points, straight_qrcode = self.detector.detectAndDecodeMulti(img)
                if ok and decoded_info:
                    for s in decoded_info:
                        if s and s.strip() and s not in texts:
                            texts.append(s.strip())
            except Exception:
                pass

        # Method 2: pyzbar fallback, often better for tilted QR codes
        if pyzbar is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                codes = pyzbar.decode(gray)
                for code in codes:
                    s = code.data.decode('utf-8', errors='ignore').strip()
                    if s and s not in texts:
                        texts.append(s)
            except Exception:
                pass

        return texts

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
    parser.add_argument('--output', default=os.path.expanduser('~/yolo_dataset/qr_images'), help='QR image save dir')
    parser.add_argument('--pub-topic', default='/qr_code_data', help='publish decoded QR json to this topic')
    parser.add_argument('--fetch', action='store_true', help='if QR content is URL, request it and parse JSON')
    parser.add_argument('--save-on-detect', action='store_true', help='save image whenever QR is detected')
    parser.add_argument('--save-all', action='store_true', help='save raw images periodically even if no QR is detected')
    parser.add_argument('--interval', type=float, default=0.5, help='save interval when --save-all enabled')
    parser.add_argument('--repeat-period', type=float, default=2.0, help='republish same QR result after seconds')
    parser.add_argument('--timeout', type=float, default=3.0, help='HTTP timeout seconds')
    parser.add_argument('--flip', action='store_true', help='horizontal flip image before decode')
    args = parser.parse_args()

    rospy.init_node('qr_collect_and_decode')
    QRCollectAndDecode(args)
    rospy.spin()


if __name__ == '__main__':
    main()

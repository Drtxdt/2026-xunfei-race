#!/usr/bin/env python3
"""Serve ROS camera topic as MJPEG over HTTP so VSCode can view it."""

import rospy
import cv2
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

latest_frame = None
bridge = CvBridge()
lock = threading.Lock()


flip_image = False


def image_cb(msg):
    global latest_frame
    try:
        img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        if flip_image:
            img = cv2.flip(img, 1)
        _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        with lock:
            latest_frame = jpeg.tobytes()
    except Exception:
        pass


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with lock:
                    frame = latest_frame
                if frame is not None:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                rospy.sleep(0.05)
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'<img src="/stream">')


def main():
    global flip_image
    rospy.init_node('camera_mjpeg_server')
    topic = rospy.get_param('~topic', '/usb_cam/image_raw')
    port = rospy.get_param('~port', 8080)
    flip_image = rospy.get_param('~flip', False)
    rospy.Subscriber(topic, Image, image_cb, queue_size=1)

    server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
    rospy.loginfo("MJPEG stream at http://<car_ip>:%s/stream", port)
    server.serve_forever()


if __name__ == '__main__':
    main()

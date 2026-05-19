#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError

# CvBridge对象，用于将ROS图像消息转换为OpenCV图像
bridge = CvBridge()

# 二维码扫描函数
def scan_qr_code(image_msg):
    # print("i get image msgs")
  
  
    try:
        # 将ROS图像消息转换为OpenCV格式
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        
        # 使用OpenCV的QRCodeDetector扫描二维码
        qr_decoder = cv2.QRCodeDetector()


        retval, decoded_info, points, straight_qrcode = qr_decoder.detectAndDecodeMulti(cv_image)
        print(f"1 {retval}")
        print(f"2 {decoded_info}")



        # if retval:
        #     rospy.loginfo("QR Code Detected: %s", decoded_info)
        #     # 将扫描到的信息发送到参数服务器
        #     # 使用绝对路径 /qr_code_data 确保不会被覆盖
        #     rospy.set_param('/qr_code_data', decoded_info)
        #     rospy.loginfo("QR Code data sent to parameter server: %s", decoded_info)
        #     print(decoded_info)
        # else:
        #     rospy.loginfo("No QR Code detected.")
        #     print("i get nothing")
    
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s", e)

# 回调函数，订阅图像消息
def image_callback(msg):
    scan_qr_code(msg)

if __name__ == "__main__":

    # 初始化节点
    rospy.init_node('qr_code_scanner', anonymous=True)
    
    # 订阅图像话题（假设话题是/usb_cam/image_raw）
    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
    
    # 持续运行ROS节点
    rospy.spin()
#!/usr/bin/env python3
# coding=utf8
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class Image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher('/stereo/image', Image, queue_size=1 )
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)
    def my_pub(self, img):
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tf
import rospy
from geometry_msgs.msg import PoseStamped
from ctf import TF

mytf = TF()
pos = PoseStamped()
tfbc = tf.TransformBroadcaster()

def get_msg(msg):
    pos.header.stamp = rospy.Time.now()
    pos.pose = msg.pose
    pass


rospy.init_node('demo_tf_broadcaster',anonymous=False)
rospy.Subscriber('tf_demo', PoseStamped, get_msg, queue_size=2)

rate = rospy.Rate(30)
while rospy.is_shutdown() == False:
    t = (pos.pose.position.x, pos.pose.position.y, pos.pose.position.z)

    mq = mytf.quaternion_from_euler(pos.pose.orientation.x, \
    pos.pose.orientation.y, pos.pose.orientation.z, 'sxyz')

    # mq = (pos.pose.orientation.x, pos.pose.orientation.y, pos.pose.orientation.z, pos.pose.orientation.w)
    

    print(t, mq)
    tfbc.sendTransform(t, mq, rospy.Time.now(), 'chess_frame', 'camera')
    rate.sleep()

rospy.spin()


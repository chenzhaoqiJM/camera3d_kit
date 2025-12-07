#!/usr/bin/env python3
# coding=utf8

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
from  creconstruction import  Calc3DPose


if __name__ == '__main__':

    ros_mode = True
    #3d位姿计算类
    my3d = Calc3DPose()

    pose_pub = 0
    if ros_mode == True:
        rospy.init_node('pubface_object', anonymous=True)
        pose_pub = rospy.Publisher('object_pose', PoseWithCovarianceStamped, queue_size=1)


    w1 = [144., -50., 100., 1]; w2 = [200.,50.,120., 1]
    w1 = np.array(w1); w2 = np.array(w2)
    w = (w1 + w2 )/2.0


    rate = rospy.Rate(10)
    while rospy.is_shutdown() == False:                      
        ob_pose = PoseWithCovarianceStamped()
        ob_pose = my3d.calc_3d_pose(w, w1,w2)
        ob_pose.header.frame_id = 'pen'
        pose_pub.publish(ob_pose)
        rate.sleep()

           
                   
                
     
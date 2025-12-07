#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from moveit_msgs.msg import RobotTrajectory
import numpy as np
import math
from cmoveit import MoveItIkDemo
from ctf import TF
    

if __name__ == "__main__":
    # 初始化ROS节点
    rospy.init_node('center_control')
    
    myik = MoveItIkDemo(False)
    

    positionm = [0.4, 0.3, 0.6]; angle = [0, 0, 0]

    pos = myik.pose_from_position_and_angles(positionm, angle)
    # print(pos)



    radiusm = 0.08
    myik.set_sphere_obstacle('sphere2', pos, radiusm)
    
    pose_list = myik.calc_mult_pose_by_sphere(positionm, [0,0,0], radiusm+ 0.003, 30, 90)
    # print(pose_list[4])
    print "************采样的步长为： 30度 与 90度。 个数为22个"
    

    loop_count = 0
    for i in range(0, len(pose_list)):
        print('\n')
        sss = "****************"+ "第 " + str(loop_count+1) + " 个位姿逆运动学求解"
        print sss
        
        traj = myik.get_traj_use_ik(pose_list[i])
        if len(traj.joint_trajectory.points)>0:
            print '^_^ ^_^ ^_^ ^_^找到解，跳出循环 !'
            break
        loop_count+=1
        
    if len(traj.joint_trajectory.points)>0:  
        l = len(traj.joint_trajectory.points)
        print "**************** 前往目标物体， 路径插值长度为：", l 
        place_state = myik.arm.execute(traj)
        myik.arm.stop()
        myik.arm.clear_pose_targets()


    rate = rospy.Rate(0.2)
    while rospy.is_shutdown() == False:
 
        rate.sleep()

    rospy.spin()

    
    

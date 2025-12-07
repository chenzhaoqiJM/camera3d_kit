#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from moveit_msgs.msg import RobotTrajectory
import numpy as np
import math


update_flag = False
same_flag = False
send_flag = False
old_end_position = []

def get_traj(msg):
    global update_flag, same_flag, old_end_position, send_flag
    global traj_msg
    
    if(len(traj_msg.joint_trajectory.points) != 0):
        old_end_position = traj_msg.joint_trajectory.points[-1].positions; old_end_position=list(old_end_position)
        new_end_position = msg.joint_trajectory.points[-1].positions; new_end_position=list(new_end_position)
        old_end_position = np.array(old_end_position)
        new_end_position = np.array(new_end_position)
        disk = old_end_position - new_end_position; length = math.sqrt(np.dot(disk, disk))
        print("distance is: ", length)
        if length < 0.1:
            same_flag = True
        else:
            same_flag = False
            send_flag = False
        
    traj_msg = RobotTrajectory()
    traj_msg.joint_trajectory = msg.joint_trajectory
    
    update_flag = True
    

if __name__ == "__main__":
    # 初始化ROS节点
    rospy.init_node('center_control')
    
    rospy.Subscriber('trajectory', RobotTrajectory, get_traj, queue_size=2)
    ang_pub = rospy.Publisher('/jointangles', Pose, queue_size=1)

    traj_msg = RobotTrajectory()

    rate = rospy.Rate(10)
    while rospy.is_shutdown() == False:
        if update_flag == True:
            update_flag = False
            
            a = traj_msg.joint_trajectory.points
            
            list_position = []
            for i in range(0, len(a)):
                cell = a[i].positions
                cell = list(cell)
                list_position.append(cell)
            
            if same_flag == False or send_flag == False:
                print("len of position : ",len(list_position))

            if same_flag == False or send_flag == False:
                pose = Pose()
                pose.position.x = list_position[i][0]; pose.position.y = list_position[i][1]
                pose.position.z = list_position[i][2]
                pose.orientation.x = list_position[i][3]; pose.orientation.y = list_position[i][4]
                pose.orientation.z = list_position[i][5]; pose.orientation.w = 50
                ang_pub.publish(pose)

                send_flag = True
                
        rate.sleep()

    rospy.spin()

    
    

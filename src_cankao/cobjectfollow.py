#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from moveit_msgs.msg import RobotTrajectory, PlanningScene, ObjectColor
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Pose, PoseWithCovarianceStamped
from math import acos, pi, sin, cos
import math
import numpy 
from tf import TransformListener
from ctf import TF



class MoveItIkFollow:
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)

        #这提供了一个远程界面，用于获取，设置和更新机器人对周围世界的内部了解：
        self.scene = moveit_commander.PlanningSceneInterface()
        # 创建一个发布场景变化信息的发布者
        self.scene_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=5)
        
        # 创建一个存储物体颜色的字典对象
        self.colors = dict()

        self.box1_size = [0.01, 0.01, 0.10]
        self.box1_pose = PoseStamped()
        self.box_count = 1

        #监听tf变换
        self.tf_listener = TransformListener()
        self.mytf = TF()
        self.dis_center = 0.05
        

    #更新物体信息并放入规划场景
    def update_box1(self, pwsc = PoseWithCovarianceStamped(),reference_frame = 'base_link'):
        
        ret = self.tf_listener.canTransform(reference_frame,'J6', rospy.Time(0))#查找变换是否存在
        if ret == True:
            trans = self.tf_listener.lookupTransform(reference_frame,'J6', rospy.Time(0)) #监听变换
            print(trans[1]) 
            mq1 = trans[1]   
            mq2 = self.mytf.quaternion_from_euler(0, math.pi/2.0, 0, 'rxyz') #在对象空间中绕y轴旋转90度
            mq = self.mytf.quaternion_multiply(mq1, mq2)#mq1 -> mq2 -> end #利用四元素乘法求最终的位姿
            
            theta1 = self.mytf.euler_from_quaternion(mq1, 'sxyz') #欧拉角，在直立空间中
            R_m1 = self.mytf.eulerAnglesToRotationMatrix(theta1)    #欧拉角到旋转矩阵，行向量表示，右手坐标系
            
            T = numpy.array(trans[0]) #平移向量
            
            #求对象空间中在Z轴上距离原点一定距离点在直立空间中的位置
            point_z = numpy.array([-0.0, 0, self.dis_center])
            point_z_r = numpy.dot(point_z, R_m1)
            point_z_r_t = point_z_r + T
        
            position = point_z_r_t
            
            self.box1_pose.header.frame_id = reference_frame
            self.box1_pose.header.stamp = rospy.Time.now()
            self.box1_pose.pose.position.x = position[0]; self.box1_pose.pose.position.y = position[1]
            self.box1_pose.pose.position.z = position[2]
            self.box1_pose.pose.orientation.x = mq[0]; self.box1_pose.pose.orientation.y = mq[1]
            self.box1_pose.pose.orientation.z = mq[2]; self.box1_pose.pose.orientation.w = mq[3]
            self.set_obstacle()
            
            self.box1_size[2] = pwsc.pose.covariance[0]#使用物体长度


    def set_obstacle(self):
        # 设置场景物体的名称
        box1_id = 'box_follow'
        # 移除场景中之前运行残留的物体
        self.scene.remove_world_object(box1_id)
        # rospy.sleep(1)
        # 将box设置成橙色
        self.setColor(box1_id, 0.8, 0.4, 0, 1.0)
        self.scene.add_box(box1_id, self.box1_pose, self.box1_size)
        # 将场景中的颜色设置发布
        self.sendColors()   
    
    def clear_obstacle(self):
        # 设置场景物体的名称
        box1_id = 'box_follow' 
        # 移除场景中之前运行残留的物体
        self.scene.remove_world_object(box1_id)
        

        
    def place_obstacle(self):
        box1_id2 = 'box_follow2' + str(self.box_count)
        self.scene.remove_world_object(box1_id2)
        self.setColor(box1_id2, 0.8, 0.4, 0, 1.0)
        self.box1_pose.pose.position.z = self.box1_pose.pose.position.z - 0.05
        self.scene.add_box(box1_id2, self.box1_pose, self.box1_size)
        # 将场景中的颜色设置发布
        self.sendColors()   
        self.box_count += 1
        
        
    # 设置场景物体的颜色
    def setColor(self, name, r, g, b, a = 0.9):
        # 初始化moveit颜色对象
        color = ObjectColor()
        color.id = name # 设置颜色值
        color.color.r = r
        color.color.g = g
        color.color.b = b
        color.color.a = a
        self.colors[name] = color # 更新颜色字典

    # 将颜色设置发送并应用到moveit场景当中
    def sendColors(self):
        # 初始化规划场景对象
        p = PlanningScene()

        # 需要设置规划场景是否有差异     
        p.is_diff = True
        
        # 从颜色字典中取出颜色设置
        for color in self.colors.values():
            p.object_colors.append(color)
        
        # 发布场景物体颜色设置
        self.scene_pub.publish(p)    

    




    

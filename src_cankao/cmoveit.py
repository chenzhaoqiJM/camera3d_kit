#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from moveit_msgs.msg import RobotTrajectory, PlanningScene, ObjectColor
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Pose, PoseWithCovarianceStamped
from std_msgs.msg import Bool
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import acos, pi, sin, cos
import math
import numpy 
# from tf import TransformListener

from ctf import TF


class MoveItIkDemo:
    def __init__(self, grapper_mode = True):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
                
        self.grapper_mode = grapper_mode
        # 初始化需要使用move group控制的机械臂中的arm group
        self.arm = moveit_commander.MoveGroupCommander('arm')
        if self.grapper_mode == True:
            self.gripper = moveit_commander.MoveGroupCommander('gripper')
                
        # 获取终端link的名称
        self.end_effector_link = self.arm.get_end_effector_link()
        sss_1 = "末端连杆名称为 ：" + self.end_effector_link
        print sss_1
        
        self.planning_frame = self.arm.get_planning_frame()
        sss_2 = "规划坐标系为： " + self.planning_frame
        print sss_2
                        
        # 当运动规划失败后，允许重新规划
        self.arm.allow_replanning(True)
        # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
        self.arm.set_goal_position_tolerance(0.01); self.arm.set_goal_orientation_tolerance(0.05)
        # self.arm.set_max_velocity_scaling_factor(0.5) 
        self.arm.set_num_planning_attempts(1)
        
        self.target_pose = PoseStamped() #抓取时的目标位姿
        self.surface_pose_list = [PoseStamped()] #可能的抓取位姿的列表
        self.place_pose = PoseStamped() #放置物体时的位姿
        self.set_place_pose() #设置放置位姿

        #这提供了一个远程界面，用于获取，设置和更新机器人对周围世界的内部了解：
        self.scene = moveit_commander.PlanningSceneInterface()
        # 创建一个发布场景变化信息的发布者
        self.scene_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=5)
        
        # 创建一个存储物体颜色的字典对象
        self.colors = dict()

        self.box1_size = [0.01, 0.01, 0.10]
        self.pedestal_size = [0.2, 0.2, 0.04]
        self.table_size = [1, 1, 0.02]
        self.box1_pose = PoseStamped()
        
        #创建tf对象，tf主要用于几何变换，声明要发布的轨迹消息
        self.mytf = TF()
        self.trajectory = RobotTrajectory()
        self.traj_pub = rospy.Publisher('trajectory', RobotTrajectory, queue_size=1)#轨迹消息发布者
        
        self.grasp_flag = True
        self.place_flag = False
        self.distance = 0.05 #求解位姿时离物体中心多远生成点集
        self.object_vector = 0
        
        # 等待场景准备就绪
        self.grasp_state_pub = rospy.Publisher('follow_state', Bool, queue_size=1)
        self.follow_state = Bool(); self.follow_state.data = False
        rospy.sleep(1)
        
    #更新物体信息并放入规划场景
    def update_box1(self, pswc = PoseWithCovarianceStamped(), reference_frame = 'base_link'):
        exp_flag = False
        if exp_flag == True:
        # 将物体加入场景当中
            self.box1_pose.header.frame_id = reference_frame
            self.box1_pose.pose.position.x = 0.16
            self.box1_pose.pose.position.y = 0.10
            self.box1_pose.pose.position.z = 0.30
            #pitch:x yaw:y roll:z   roll, pitch and yaw zxy
            #but 
            mq = self.mytf.quaternion_from_euler(0, pi/4, pi,'sxyz')
            print("################: ",mq)
            self.box1_pose.pose.orientation.x = mq[0]
            self.box1_pose.pose.orientation.y = mq[1]
            self.box1_pose.pose.orientation.z = mq[2]
            self.box1_pose.pose.orientation.w = mq[3] 
        else:
            #物体向量的欧拉角
            xyz_a = pswc.pose.covariance[1:4]
            # print('EL Angle IS :', xyz_a)
            theta = numpy.array(xyz_a); theta = theta/180*pi
             #物体上的两点
            wc = pswc.pose.covariance[4:7]; wl = pswc.pose.covariance[7:10]
            wr = pswc.pose.covariance[10:13]
            wc = numpy.array(wc); wl = numpy.array(wl)
            # print('WC IS: ', wc)
            # print('WL IS: ', wl)
            # print('WR IS: ', wr)

            self.box1_pose.header.frame_id = reference_frame
            mypose = PoseStamped()
            # el_angle = self.mytf.vector_to_angle(wl-wc)
            el_angle = theta
            # mypose = self.pose_from_position_and_angles(wc, el_angle, is_degree=False)
            mypose.pose.position.x = wc[0]; mypose.pose.position.y = wc[1]
            mypose.pose.position.z = wc[2]

            #保证旋转的正确
            mq = [0, 0, 0 ,1]
            if abs(el_angle[2]- pi)< 0.01:
                mq = self.mytf.quaternion_from_euler(el_angle[0], el_angle[1], el_angle[2], 'rxyz')
            else:
                mq = self.mytf.quaternion_from_euler(el_angle[0], el_angle[1], el_angle[2], 'sxyz')
            
            mypose.pose.orientation.x = mq[0]; mypose.pose.orientation.y= mq[1]
            mypose.pose.orientation.z = mq[2]; mypose.pose.orientation.w = mq[3]
            self.box1_pose.pose = mypose.pose
            # self.box1_pose.pose = pswc.pose.pose
            self.box1_size[2] = pswc.pose.covariance[0]

    def set_obstacle_fixed(self):
        pedestal_id = 'pedestal'
        pedestal_pose = self.pose_from_position_and_angles([0,0,-self.pedestal_size[2]/2], [0,0,0])
        self.scene.remove_world_object(pedestal_id)
        self.scene.add_box(pedestal_id, pedestal_pose, self.pedestal_size)
        self.setColor(pedestal_id, 0.0, 0.9, 0, 1.0)
        self.sendColors() 
        table_id = 'table'
        table_pose = self.pose_from_position_and_angles([0,0,-self.pedestal_size[2]-self.table_size[2]/2], [0,0,0])
        self.scene.remove_world_object(table_id)
        self.scene.add_box(table_id, table_pose, self.table_size)
        self.setColor(table_id, 0.5, 0.9, 0, 1.0)
        self.sendColors() 

    def set_obstacle(self):
        # 设置场景物体的名称
        box1_id = 'box1'
        # 移除场景中之前运行残留的物体
        self.scene.remove_world_object(box1_id)
        # rospy.sleep(1)
        # 将box设置成橙色
        self.setColor(box1_id, 0.8, 0.4, 0, 1.0)
        self.scene.add_box(box1_id, self.box1_pose, self.box1_size)

        # 将场景中的颜色设置发布
        self.sendColors()   

    def set_sphere_obstacle(self, name='sphere1', pos = PoseStamped(), radius=0):
        self.scene.remove_world_object(name)
        self.setColor(name, 0.0, 1.0, 0.1, 1.0)
        # pos = PoseStamped()
        # pos.header.frame_id = 'base_link'; pos.header.stamp = rospy.Time.now()
        # pos.pose.position.x = position[0]; pos.pose.position.y = position[1]; pos.pose.position.z = position[2]
        self.scene.add_sphere(name, pos, radius)
        self.sendColors()  
        
    def clear_obstacle(self):
        # 设置场景物体的名称
        box1_id = 'box1'
        # 移除场景中之前运行残留的物体
        self.scene.remove_world_object(box1_id)

    #到达初始位姿，抓取物体
    def ik_plan_use_pose(self, pswc = PoseWithCovarianceStamped(), real_mode = False, reference_frame = 'base_link'):
        # self.follow_state.data = False
        # self.grasp_state_pub.publish(self.follow_state)
        
        if real_mode == False:   #设置起始位姿
            position = [0.2,0.2,0.2]; angles = [180,0,0]    # 设置机械臂工作空间中的目标位姿，位置使用x、y、z坐标描述，
            self.target_pose = self.pose_from_position_and_angles(position, angles) # 姿态使用四元数描述，基于base_link坐标系
            self.target_pose.header.frame_id = reference_frame
            self.target_pose.header.stamp = rospy.Time.now()  
            
        if real_mode == True and pswc.pose.pose.position.x != 0: #
            
            #物体向量的欧拉角
            xyz_a = pswc.pose.covariance[1:4]
            theta = numpy.array(xyz_a); theta = theta/180*pi

            #物体上的两点
            wc = pswc.pose.covariance[4:7]; wl = pswc.pose.covariance[7:10]
            wc = numpy.array(wc); wl = numpy.array(wl)
            self.object_vector = wl - wc
            # obj_vec = wl-wc
            # self.check_rotation_matrix(obj_vec, theta)
            print('\n')
            b= u"物体中心点坐标为 : "; b= b.encode('utf-8')
            rrc = pswc.pose.covariance[33:36]
            print b,rrc
            print('\n')

            #求距离物体中心一定距离的点#物体表面的点; use circle
            self.distance = 0.05
            self.surface_pose_list = self.calc_mult_pose(wc, theta, self.distance, 90)
        
        if real_mode == True:
            loop_count = 0
            self.trajectory = RobotTrajectory()

            if self.grapper_mode == True:
                jp= [0.012, 0.012]
                zhuaqu = self.gripper.set_joint_value_target(jp) #张开抓手
                self.gripper.go()  

            for i in range(0, len(self.surface_pose_list)): #尝试每一个位姿
                print('\n')
                sss = "****************"+ "第 " + str(loop_count+1) + " 个位姿逆运动学求解"
                print sss
            
                target = PoseStamped(); target.header.stamp = rospy.Time.now(); target.header.frame_id = reference_frame
                target.pose = self.surface_pose_list[i].pose

                angle_j6 = self.calc_gripper_angle(target, self.object_vector)#抓手要旋转的角度
                sss2 = "**************** 抓手要旋转的角度为： " + str(angle_j6)
                print sss2
                target = self.get_new_pose(target, angle_j6) #计算旋转后的位姿
                print "****************** 位姿已调整 ^_^"

                self.arm.set_start_state_to_current_state() #设置机械臂的初始状态为当前状态
                self.arm.set_pose_target(target, self.end_effector_link) #设置机械臂的目标位姿

                traj = self.arm.plan() #逆运动学规划
                if len(traj.joint_trajectory.points)>0: #如果成功规划出轨迹
                    print '^_^ ^_^ ^_^ ^_^找到解，跳出循环 !'
                    # print('\n')
                    self.target_pose = target
                    self.trajectory = traj #传递变量，后面要用
                    break

                self.arm.clear_pose_targets() #清除所有位姿
                loop_count = loop_count+1
            
            check_flag = False
            if len(self.trajectory.joint_trajectory.points)>0: #如果轨迹为有效轨迹
                l = len(self.trajectory.joint_trajectory.points)
                print "**************** 前往目标物体， 路径插值长度为：", l
                check_flag = True
                self.traj_pub.publish(self.trajectory) #发布轨迹

            # self.ik_plan_use_angle()               
            grasp_state = self.arm.execute(self.trajectory) #执行该轨迹

            if self.grapper_mode == True:
                jp= [0.007, 0.007]
                zhuaqu = self.gripper.set_joint_value_target(jp) #合抓手
                self.gripper.go()  

            if check_flag == False:
                grasp_state = False
            # print("THE Arrive STATE IS: ", grasp_state)
            if grasp_state == True:
                print "^_^ ^_^ ********** 成功到达！！"
            if grasp_state == False:
                print "!!!!!!!!!抓取失败，尝试重新规划"
            if grasp_state == True and real_mode == True: #设置相关的标志位
                self.grasp_flag = False
                self.place_flag = True 
                self.follow_state.data = True
            
        else:
            # 设置机器臂当前的状态作为运动初始状态
            self.arm.set_start_state_to_current_state()
            # 设置机械臂终端运动的目标位姿
            self.arm.set_pose_target(self.target_pose, self.end_effector_link)
            
            #规划运动路径
            traj = self.arm.plan()
            self.trajectory = traj
            
            suc_flag = False
            if len(self.trajectory.joint_trajectory.points)>0:  #检查轨迹有效性
                print " ************前往起始位置, 路径插值长度为：", len(self.trajectory.joint_trajectory.points)
                self.traj_pub.publish(self.trajectory)
                suc_flag = True
            
            grasp_state = self.arm.execute(traj)    #执行轨迹
            if self.grapper_mode == True:       #合拢抓手
                jp= [0.0, 0.0]
                zhuaqu = self.gripper.set_joint_value_target(jp)
                self.gripper.go() 
            if suc_flag == True:
                print " *************到达起始位置！！！"
            if suc_flag == False:
                grasp_state = False
                print('\n')

        self.arm.stop()
        self.arm.clear_pose_targets()
        rospy.sleep(1)


    def ik_plan_use_pose_place(self, pose = PoseStamped(), reference_frame = 'base_link'):
        pose.header.stamp = rospy.Time.now(); pose.header.frame_id = reference_frame
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(pose, self.end_effector_link)
        #规划运动路径
        traj = self.arm.plan()
        self.trajectory = traj
        
        check_flag = False; place_state = False
        if len(traj.joint_trajectory.points)>0:     #检查轨迹有效性
            check_flag = True
            # print('TO DESTINATION : ', len(traj.joint_trajectory.points))
            print('\n')
            print "**********  开始放置物体 *********"
            self.traj_pub.publish(self.trajectory)
        
        
        self.grasp_state_pub.publish(self.follow_state)     #发布跟随命令
        
        place_state = self.arm.execute(traj)    #执行规划好的轨迹
        
        if place_state == True and check_flag == True: #设置相关标志位
            self.place_flag = False
            self.grasp_flag = True
            print("^^^^^^^^^^ 放置物体成功!! ^^^^^^^^^")
            print('\n')
        self.arm.stop() #停止机械臂
        self.arm.clear_pose_targets() #清空所有目标
        
        self.follow_state.data = False
        self.grasp_state_pub.publish(self.follow_state)
        rospy.sleep(1)
    
    #计算到达正确位姿抓手要转过的角度    
    def calc_gripper_angle(self, pose, object_vec):
        pos = PoseStamped(); pos = pose
        mq = [pos.pose.orientation.x, pos.pose.orientation.y, pos.pose.orientation.z,pos.pose.orientation.w]
        th = self.mytf.euler_from_quaternion(mq, 'sxyz')
        R_m = self.mytf.eulerAnglesToRotationMatrix(th)
        # print("RMMM", R_m)
        v_y = numpy.array([0, 1, 0]); v_y_r = numpy.dot(v_y, R_m)
        # print("THE VY IS : ", v_y_r)
        # print('THE TH is: ', th)
        object_vec = numpy.array(object_vec)
        cosalphay = numpy.dot(object_vec, v_y_r)/\
        math.sqrt(numpy.dot(object_vec, object_vec)* numpy.dot(v_y_r, v_y_r))
        # print('@@@@@@@@@@@@@@@@: ',cosalphay)
        if abs(cosalphay)>1.0:
            if cosalphay > 1.0:
                cosalphay = 1.0 
            if cosalphay < -1.0:
                cosalphay = -1.0
        alphay = math.acos(cosalphay); alphady = alphay/math.pi * 180
        # if alphad > 90:
        #     alphad = 180. - alphad
        v_x = numpy.array([1, 0, 0]); v_x_r = numpy.dot(v_x, R_m)
        object_vec = numpy.array(object_vec)
        cosalphax = numpy.dot(object_vec, v_x_r)/\
        math.sqrt(numpy.dot(object_vec, object_vec)* numpy.dot(v_x_r, v_x_r))
        alphax = math.acos(cosalphax); alphadx = alphax/math.pi * 180
        if alphady > 90:
            alphadx = -alphadx
        # print('#########ANGEL IS: ', alphadx)
        return alphadx

    def get_traj_use_ik(self, pose, reference_frame = 'base_link'):
        self.arm.set_start_state_to_current_state()
        target = pose; target.header.stamp = rospy.Time.now(); target.header.frame_id = reference_frame
        self.arm.set_pose_target(target, self.end_effector_link)
        traj = self.arm.plan()
        self.arm.clear_pose_targets()
        return traj
    
    #根据物体向量计算抓取姿态,过给定抓取点的平面上的圆环
    #wc：抓取位置（中心），theta：物体的欧拉角， base_vec：标准坐标中的一个朝向
    def calc_pose_from_vector(self,  wc, theta, base_vec=[0.02, 0.02, 0]):
        base_vec = numpy.array(base_vec); wc = numpy.array(wc)
        R_m = self.mytf.eulerAnglesToRotationMatrix(theta)
        r_vec = numpy.dot(base_vec, R_m)
        t_vec = r_vec + wc
        # print("TransandR: ", t_vec)
        # print("%%%%%%dis: ", math.sqrt(numpy.dot(t_vec-wc, t_vec-wc)))
        
        #计算机械臂朝向向量的欧拉角与旋转四元数
        new_vec = - t_vec + wc; el_angle = self.mytf.vector_to_angle(new_vec)
        surface_pose = self.pose_from_position_and_angles(t_vec, el_angle, is_degree=False)
        return surface_pose

    def calc_pose_from_vector_sphere(self,  wc, theta, base_vec=[0.02, 0.02, 0]):
        base_vec = numpy.array(base_vec); wc = numpy.array(wc)
        R_m = self.mytf.eulerAnglesToRotationMatrix(theta)
        r_vec = numpy.dot(base_vec, R_m)
        t_vec = r_vec + wc
        # print('\n')
        # print(base_vec)
        # print(t_vec)
        # print('\n')
        
        #计算机械臂朝向向量的欧拉角与旋转四元数
        new_vec = - t_vec + wc
        el_angle = self.mytf.vector_to_angle_sphere(new_vec)
        surface_pose = self.pose_from_position_and_angles(t_vec, el_angle, is_degree=False)
        return surface_pose
    
    #根据给定的半径在平面圆环上生成多组抓取位姿
    def calc_mult_pose(self, wc, theta, r=0.02, step = 30):
        base_vec = []
        for i in range(0, 360//step):
            th = i * step / 180.0 * math.pi
            x = r * math.cos(th); y = r * math.sin(th); z=0
            b = [x, y, z]; base_vec.append(b)

        pose_list = []
        for i in range(0, len(base_vec)):
            pose = self.calc_pose_from_vector(wc, theta, base_vec[i])
            pose_list.append(pose)
        return pose_list
    
    def calc_mult_pose_by_sphere(self, wc, theta, r=0.01, step1 = 30, step2 = 90):
        base_vec = []
        
        for i in range(0, 180//step1+1):
            fai = i* step1/180.0 * math.pi 
            z = r * math.cos(fai)
            r1 = r * math.sin(fai)
            for j in range(0, 360//step2):
                th = j * step2 / 180.0 * math.pi
                x = r1* math.cos(th); y = r1 * math.sin(th)
                b = [x, y, z]; base_vec.append(b)
                if i == 0 or i==(180//step1+1):
                    break
                # if i == 3:
                #     print(b)

        # print(base_vec)
        pose_list = []
        # print(wc, theta)
        # print("$#$$$$$$$$$$$$$$%$$$%%")
        # print(base_vec[0])
        for i in range(0, len(base_vec)):
            pose = self.calc_pose_from_vector_sphere(wc, theta, base_vec[i])
            pose_list.append(pose)
        return pose_list
    
    
    #从位置和欧拉角计算位姿
    def pose_from_position_and_angles(self, position=[0.2, 0.2, 0.1], angle= [0, 0, 0], is_degree = True, reference_frame = 'base_link'): 
        mytf = TF()       
        angle = numpy.array(angle)
        if is_degree == True:
            angle = angle/180.0 * math.pi
        pos = PoseStamped()
        pos.header.frame_id = reference_frame
        pos.header.stamp = rospy.Time.now()
        pos.pose.position.x = position[0]; pos.pose.position.y = position[1]
        pos.pose.position.z = position[2]
        mq = mytf.quaternion_from_euler(angle[0], angle[1], angle[2], 'sxyz')
        pos.pose.orientation.x = mq[0]; pos.pose.orientation.y = mq[1]
        pos.pose.orientation.z = mq[2]; pos.pose.orientation.w = mq[3]
        return pos
    
    
    
    
    #检查旋转矩阵是否正确
    def check_rotation_matrix(self, w_vec, theta):
        
        theta_vec = [0,0,1];theta_vec = numpy.array(theta_vec)
        R_m = self.mytf.eulerAnglesToRotationMatrix(theta)
        theta_vec = numpy.dot(theta_vec, R_m)
        theta_vec = theta_vec/ math.sqrt(numpy.dot(theta_vec, theta_vec))
        print('theta_v from el: ', theta_vec)
        w_vec = w_vec/ math.sqrt(numpy.dot(w_vec, w_vec)) #normalnaze
        print("theta_v from w: ", w_vec)
    

    #设置物体的放置位姿
    def set_place_pose(self):
        position = [0.2,0.2,0.2]; angles = [180,0,0]
        self.place_pose = self.pose_from_position_and_angles(position, angles)
            
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

    def ik_plan_use_angle(self, angel_j6 = 90, is_degree = True):
            # We can get the joint values from the group and adjust some of the values:
        # 设置机器臂当前的状态作为运动初始状态
        # cstate = self.arm.get_current_state(); print("&&&&&&&&&&%%% : ", cstate)
        # list_goal = [-0.41, 0.24, 1.57, 0, 1.57, -0.53]
        # list_goal = [-7.77, 16.98, 68.95, -30.46, 5.76, -29.22]
        # for i in range(0, len(list_goal)):
        #     list_goal[i] = list_goal[i]/180.0 * math.pi

        self.arm.set_start_state_to_current_state()
        self.joint_goal = self.arm.get_current_joint_values()
        if is_degree == True:
            angel_j6 = angel_j6 / 180 * math.pi
        
        self.joint_goal[5] = self.joint_goal[5] + angel_j6

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.arm.go(self.joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.arm.stop()
        # rospy.sleep(1)

    def ik_plan_by_joint6(self, angels_state):
        self.arm.set_start_state_to_current_state()
        self.joint_goal = angels_state
        self.arm.go(self.joint_goal, wait=True)
        self.arm.stop()

    def get_new_pose(self, pose_old, angle_z, is_degree = True):
        new_pose = pose_old
        mq1 = [new_pose.pose.orientation.x,new_pose.pose.orientation.y\
        ,new_pose.pose.orientation.z, new_pose.pose.orientation.w ]

        if is_degree == True:
            angle_z = angle_z/180.0 * math.pi
        
        mq2 = self.mytf.quaternion_from_euler(0, 0, angle_z, 'rxyz')
        mq3 = self.mytf.quaternion_multiply(mq1, mq2)
        new_pose.pose.orientation.x = mq3[0]; new_pose.pose.orientation.y = mq3[1]
        new_pose.pose.orientation.z = mq3[2]; new_pose.pose.orientation.w = mq3[3]

        return new_pose

    def shut_down(self):
            # 关闭并退出moveit
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
    




    

#!/usr/bin/env python3
# coding=utf8

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from cimageconverter import Image_converter
import numpy as np
from cv2 import cv2
from ccalibration import CCameraCalibration
from creconstruction import Reconstruction
from  creconstruction import  Calc3DPose
from cccparameters import CParameters
from cboxdeal import boxDeal

def box_callback(msg):
    global  label, obj_list
    label_list=['pen','stents']; label = label_list
    list_box = msg.poses
    box = []
    for i in range(0, len(list_box)):
        pose1 = PoseStamped()
        pose1 = list_box[i]
        linshi=[0,0,0,0,0,0]
        linshi[0] = int(pose1.pose.position.x) #class
        linshi[1] = pose1.pose.position.y #confidence
        linshi[2] = pose1.pose.orientation.x; linshi[3] = pose1.pose.orientation.y #left_top
        linshi[4] = pose1.pose.orientation.z; linshi[5] = pose1.pose.orientation.w #right_bottom

        box.append(linshi)
    # print(box)
    obj_list = box


if __name__ == '__main__':
    cv_img = 0
    obj_list = []
    label = []
    ros_mode = True
    


    #************** start 相机标定 **********************
    
    #初始化自定义相机标定对象，默认立体标定模式
    mycc = CCameraCalibration() 
     
    #设置标定图像的路径、图片数量、图片的高和宽、棋盘格的尺寸、每个格子的实际长度，单位mm
    mycc.set_calibration_images("/home/zq/Desktop/1280/91/", 15, 480, 1280, [6,8], 24.6)
     
    #进行双目立体标定，得到内参矩阵、畸变系数、重映射矩阵map、重投影矩阵Q等。默认加载标定文件以避免反复标定
    mycc.stereo_calibration(True) 
    
    #************** end 标定完成 **********************

    #相机参数的储存对象
    myp = CParameters()
    myp.init_by_cc(mycc)

    #相机三维重构类
    myrec = Reconstruction(myp)
    #输入标定得到的重投影矩阵Q
    myrec.inputQ(mycc._stereoCommParameters["Q"]) 


    #3d位姿计算类
    my3d = Calc3DPose()
    #输入重构类对象
    my3d.input_Rec(myrec) 
    #计算手眼标定的旋转矩阵R和平移向量T
    my3d.calc_R_and_T() 
    
    boxdealer = boxDeal()

    pose_pub = 0
    if ros_mode == True:
        rospy.init_node('pubImg_and_sbox', anonymous=False)
        rospy.Subscriber('/boxs', Path, box_callback)
        pose_pub = rospy.Publisher('object_pose', PoseWithCovarianceStamped, queue_size=1)
        imgCvt = Image_converter()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    name_count = 1
    while(True):
        ret, frame = cap.read()
        left = frame[0:mycc._height, 0: mycc._width//2] #拆分左右图像
        right = frame[0:mycc._height, mycc._width//2: mycc._width]
        
        #立体校正
        left = cv2.remap(left, mycc._leftParameters["map1"], mycc._leftParameters["map2"], \
        cv2.INTER_LINEAR)
        right = cv2.remap(right, mycc._rightParameters["map1"], mycc._rightParameters["map2"], \
        cv2.INTER_LINEAR)
        dst =  cv2.hconcat([left, right]) #合并校正后图像

        #立体匹配计算视差图
        myrec.calc_dis_BM(left, right) 
        # myrec.calc_dis_SGBM(left, right)

        if ros_mode == True:
            cv_img = dst.copy()
            # imgCvt.my_pub(left)
            imgCvt.my_pub(dst)
        
        center_w = [0, 0, 0]
        center_w_stents = [0,0,0]
        if ros_mode == True and len(obj_list)>0:
            # print(obj_list)

            box = obj_list; label_list = label
            boxdealer.update_box(label_list, box)
            boxdealer.find_respone()
            boxdealer.search_dis(myrec)
               
            #恢复物体在相机坐标系中的三维坐标
            if bool(boxdealer.disparity_bm) == True:
                pen = boxdealer.disparity_bm[label_list[0]]
                stents = boxdealer.disparity_bm[label_list[1]]
                
                if bool(pen) == True:
                    temp = pen['center']; pl = temp[0:2]; dis = temp[2]; w = 0
                    temp1 = pen['left_up']; pl1 = temp1[0:2]; dis1 = temp1[2]; w1 = 0
                    if dis != -1 :
                        w = myrec.img_to_world_by_dis(pl, dis)
                        # w1 = myrec.img_to_world_by_dis(pl1, dis1)
                        # print(w1)
                        center_w = w[0][0:3]
                    
            #     if bool(stents) == True:
            #         temp = stents['center']; pl = temp[0:2]; dis = temp[2]; w=1
            #         if dis != -1:
            #             w = myrec.img_to_world_by_dis(pl, dis)
            #             center_w_stents = w[0][0:3]
            #             print(w)
                        
            #恢复物体在相机坐标系中的三维坐标
            if bool(boxdealer.disparity_dict) == True:
                pen = boxdealer.disparity_dict[label_list[0]]
                stents = boxdealer.disparity_dict[label_list[1]]
                
                if bool(pen) == True:
                    temp = pen['center']; pl = temp[0:2]; dis = temp[2]; w = 0
                    temp1 = pen['left_up']; pl1 = temp1[0:2]; dis1 = temp1[2]; w1 = 0
                    temp2 = pen['right_down']; pl2 = temp2[0:2]; dis2 = temp2[2]; w2 = 0
                    if dis != -1 and dis1 != -1 and dis2 != -1:
                        w = myrec.img_to_world_by_dis(pl, dis)
                        w1 = myrec.img_to_world_by_dis(pl1, dis1)
                        w2 = myrec.img_to_world_by_dis(pl2, dis2)
                        
                        ob_pose = PoseWithCovarianceStamped()
                        ob_pose = my3d.calc_3d_pose(w[0], w1[0],w2[0])
                        ob_pose.header.frame_id = 'pen'
                        ob_pose.pose.covariance[33:36] = np.array(center_w)/1000.
                        pose_pub.publish(ob_pose)
                        
                        # print(w[0], w1[0], w2[0])
                   
                
            # print(box)
            #画出检测框
            for i in range(0, len(box)):
                center_x = (box[i][2] + box[i][4])/2.0; center_x = int(center_x)
                center_y = (box[i][3] + box[i][5])/2.0; center_y= int(center_y)
                
                cv_img = cv2.rectangle(cv_img, (int(box[i][2]), int(box[i][3])), \
                (int(box[i][4]), int(box[i][5])), (255, 0, 0), 2)
                cv_img = cv2.circle(cv_img,(center_x,center_y), 3 ,(0,0,255),2)
                font = cv2.FONT_HERSHEY_TRIPLEX
                text = label_list[int(box[i][0]) ] + " " + (str(box[i][1]))[0:5]
                cv2.putText(cv_img, text, (int(box[i][2]), int(box[i][3])-5 ), \
                font, 1, (255, 255, 0), 1)
                
                if center_w[0] != 0 and label_list[int(box[i][0]) ] == 'pen':
                    text_c = '('+str(center_w[0])[0:5] + ','+ str(center_w[1])[0:5] + ',' + str(center_w[2])[0:6]+')'
                    cv2.putText(cv_img, text_c, (center_x, center_y-15 ), \
                font, 0.7, (122, 255, 0), 1)
                
                if center_w_stents[0]!=0 and label_list[int(box[i][0]) ] == 'stents':
                    text_c = '('+str(center_w_stents[0])[0:5] + ' ,'+ str(center_w_stents[1])[0:5] + ' ,' +\
                        str(center_w_stents[2])[0:6]+')'
                    cv2.putText(cv_img, text_c, (center_x, center_y-15 ), \
                    font, 0.7, (122, 255, 0), 1)  
                    print('flag^^^^^^^^^^^^_^^^^^^^')
                # print(label_list[int(box[i][0]) ])            
        
        #从相机坐标系转换到自定义三维坐标系
        # myw = my3d.camera_to_world(w0)
        # print(myw)
        # print(np.dot(my3d.Rotation, my3d.Rotation.T))
        # print(my3d.Rotation)
                                                          
        # cv2.namedWindow("frame")
        # cv2.namedWindow("LEFT")
        # cv2.namedWindow("RIGHT")
        # cv2.namedWindow("DST")
        
        # cv2.imshow('frame',frame)
        # cv2.imshow('LEFT',left)
        # cv2.imshow('RIGHT',right)
        # cv2.imshow('DST',dst)

        if ros_mode == True:
            cv2.imshow('result', cv_img)
        
        key = cv2.waitKey(1)
        # print(key)
        if(ros_mode == True):
            if rospy.is_shutdown() :
                break

        if key & 0xFF == ord('q'):
            # cv2.imwrite('C:\\Users\\hp\\Desktop\\2.jpg', frame)
            break
        if key == 9:
            cv2.imwrite('/home/zq/Desktop/trian/640-480/'+str(name_count)+".jpg", left)
            name_count = name_count+1
        
    rospy.spin()  
    
    cap.release()
    cv2.destroyAllWindows()
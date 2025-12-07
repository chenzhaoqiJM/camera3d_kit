#!/usr/bin/env python
# coding=utf8

import numpy as np
from cv2 import cv2
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from ccalibration import CCameraCalibration
from creconstruction import Reconstruction
from  creconstruction import  Calc3DPose
from cccparameters import CParameters
try:
    ros_env = True
    import rospy
except:
    ros_env = False
    print('NO MODULE Named rospy')
    pass

images_path_win = 'C:\\Users\\hp\\Desktop\\1280\\91\\'
images_path_ros = '/home/zq/Desktop/1280/91/'
images_path = images_path_ros   #默认
win_mode = False

if ros_env == True:
    rospy.init_node('demo_stereo', anonymous=False)
else:
    images_path = images_path_win
    win_mode = True



#相机标定类
mycc = CCameraCalibration()  #初始化
mycc.set_calibration_images(images_path, 15, 480, 1280, [6,8], 24.6) #设置标定图像的路径和图片数量
mycc.stereo_calibration(True) #进行双目立体标定，得到内参矩阵、畸变系数、重映射矩阵map、重投影矩阵Q等
mycc.print_p()

#相机参数的储存类
myp = CParameters()
myp.init_by_cc(mycc)

#相机三维重构类
myrec = Reconstruction(myp)
# myrec.inputQ(mycc._stereoCommParameters["Q"]) #输入标定得到的重投影矩阵Q
# myrec.calc_M_by_RT(mycc) #通过标定得到的旋转矩阵和平移向量计算两个摄像头的投影矩阵，以左相机坐标系为基准
img_pnp = cv2.imread(images_path + '1.jpg')
myrec.inputPnPImg(img_pnp, 480, 1280, (6,8), 24.6)
myrec.calc_r_and_t_by_PnP(myp, True)

print("Q Matrix is: ", "\n",mycc._stereoCommParameters["Q"])

#3d位姿计算类
my3d = Calc3DPose()
my3d.input_Rec(myrec) #输入重构类对象
my3d.calc_R_and_T() #计算手眼标定的旋转矩阵R和平移向量T


cap = 0
if win_mode == True:
    cap = cv2.VideoCapture(1 , cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


name_count = 301
count = 1
while(True):
    if cap.isOpened() == False:
        print('can not open camera')
        break
    ret, frame = cap.read()
    if ret == False:
        continue
    left = frame[0:mycc._height, 0: mycc._width//2]
    right = frame[0:mycc._height, mycc._width//2: mycc._width]
    
    left = cv2.remap(left, mycc._leftParameters["map1"], mycc._leftParameters["map2"], \
    cv2.INTER_LINEAR)
    right = cv2.remap(right, mycc._rightParameters["map1"], mycc._rightParameters["map2"], \
    cv2.INTER_LINEAR)
    
    dst =  cv2.hconcat([left, right])
    dst1 = dst.copy()
    dst1 = cv2.Canny(dst1, 50, 200)
    
    #立体匹配计算视差图
    myrec.calc_dis_BM(left, right)
    # myrec.calc_dis_SGBM(left, right)
    disparity = myrec.disparity
    
    _3dimg = cv2.reprojectImageTo3D(disparity, myrec.Q.T) * 16
   
    # print(_3dimg[300][190])

    #检测立体校正后的左右图像棋盘角点
    chess_size = (4, 6)
    # chess_size = (6, 8)
    ret_l, corners_left, ret_r, corners_right  = myrec.check_chess_corners(left, right, chess_size)
    
    #利用重投影矩阵计算棋盘第一个角点的三维坐标
    w0 =  myrec.print_zero(left, right, ret_l, corners_left, ret_r, corners_right)
    # print("THE NO MATHCH IS : ",w0)
    if ret_l and ret_r:
        point = [corners_left[0][0][0], corners_left[0][0][1]]
        dis = myrec.get_bm_dis(point)
        w1 = myrec.img_to_world_by_dis(point, dis)
        # print("BM : ", w1)
        print('DERCTOE DIS is :', corners_left[0][0][0]-corners_right[0][0][0])
        print('BM DIS is ', dis)
        
    # myrec.print_len(ret_l, corners_left, ret_r, corners_right)
    
    if ret_l and ret_r:
        myrec.draw_cube_left(left, corners_left, corners_right, chess_size)
        
    
    #利用投影矩阵解算三维坐标
    # if(ret_l == True and ret_r == True):    
    #     myrec.img_to_world_by_m(corners_left[0][0], corners_right[0][0])
    
    #找手眼标定的点，基于Q矩阵
    # myrec.hand_eye_calibration(left, right, ret_l, corners_left, ret_r, corners_right)
    
    
    #关键点检测和匹配
    #right = myrec.feature_check(right)
    #img3 = myrec.feature_check_and_match(left, right)
    
    #可视化视差的精确度
    # if count<=200:
    #     my3d.calibra_depth_by_chess(left, right, ret_l, corners_left, ret_r, corners_right)
    #     count = count +1
    
    #从相机坐标系转换到自定义三维坐标系
    # myw = my3d.camera_to_world(w0)
    # print(myw)
    # print(np.dot(my3d.Rotation, my3d.Rotation.T))
    # print(my3d.Rotation)

    # cv2.namedWindow("frame")
    cv2.namedWindow("LEFT")
    # cv2.namedWindow("RIGHT")
    # cv2.namedWindow("DST")
    # cv2.namedWindow("DIS")
    
    # cv2.imshow('frame',frame)
    cv2.imshow('LEFT',left)
    # cv2.imshow('RIGHT',right)
    # cv2.imshow('DST',dst)
    # cv2.imshow('DIS', _3dimg)
    # cv2.imshow('DSTCanny',disparity/16)
    
    mykey = cv2.waitKey(1)
    if mykey & 0xFF == ord('q'):
        # cv2.imwrite('C:\\Users\\hp\\Desktop\\1.jpg', frame)
        pass
        break
    # if key == 9:
    #     cv2.imwrite('C:\\Users\\hp\\Desktop\\train\\'+str(name_count)+".jpg", right)
    #     print(name_count)
    #     name_count = name_count+1


# When everything done, release the capture
if ros_env == True:
    rospy.spin()
cap.release()
cv2.destroyAllWindows()
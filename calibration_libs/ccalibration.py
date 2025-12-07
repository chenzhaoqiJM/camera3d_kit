#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
import os

class CCameraCalibration:

    def __init__(self):
        """相机的标定类，OpenCV的版本建议在4.5.0以上
        >>> 立体标定输入：合在一起的左右图像，文件名必须为数字+.jpg 如：1.jpg
        >>> 单目相机的标定相关函数都带有前缀名monocular
        >>> 单目相机标定：已集成
        """
        
        self.images_path = ' '     #图像路径初始化
        self._width = 1280  #图像的宽度
        self._height = 480  #图像的高度
        self._stereoMode = True     #是否使用立体标定
        self._imagesReady = False; self.file_flag = False   #文件读取的标志位
        self._chessCellLen = 24.6   #棋盘格单元的大小，单位mm
        self._chessSize = [4,6]     #棋盘格的尺寸
        self._imagesMerge = []      #标定图像的列表，读取后的文件存储于此
        
        #标定后得到的左右相机的参数字典，以及公共参数
        self._leftParameters = dict()
        self._rightParameters = dict()
        self._stereoCommParameters = dict()

        self._imageSize = (self._width//2, self._height)    #单副左图或右图像的大小
        
        #**********************************************$#
        #以下为单目标定的变量
        self._monocular_path = ''
        self._monocular_images = []
        self._monocular_image_width = 640
        self._monocular_image_height = 480
        self._monocular_chessCellLen = 24.6 #mm
        self._monocular_chessSize = [4,6]
        self._monocular_images_num = 10
        self._monocular_mode = 'square'
        self._monocular_file_flag = False
        self._monocular_image_ready = False
        self._monocularParameters = dict()
        self._monocular_calibration_finished = False
        
        
    def print_p(self):
        """打印左右图像的内参数矩阵，以及重投影矩阵;只针对立体标定
        """
        print("左图对应相机的内参..................................")
        print(self._leftParameters['cameraMatrix'])
        print("\n右图对应相机的内参..................................")
        print(self._rightParameters['cameraMatrix'])
        print("\n重投影矩阵Q..................................")
        print(self._stereoCommParameters['Q'])

    #从标定文件中读取参数文件
    def from_file_read(self, path):
        self._leftParameters  = np.load(path + 'left.npy', allow_pickle=True).item()
        self._rightParameters  = np.load(path + 'right.npy', allow_pickle=True).item()
        self._stereoCommParameters  = np.load(path + 'comm.npy', allow_pickle=True).item()

    #设置标定文件路径、数量、图像尺寸、标定板大小、单元实际长度
    def set_calibration_images(self, images_path, numbers=10, height=480, width=1280, size=[4,6], length=24.6):
        """只针对立体标定；
        设置标定图像文件路径、数量、图像尺寸、标定板大小、单元实际长度mm
        """

        # 1. 检查文件夹是否存在
        if not os.path.isdir(images_path):
            print(f"错误：路径 '{images_path}' 不是一个有效的文件夹。")

        listd = os.listdir(images_path)
        
        #重置变量
        self.images_path = images_path
        self._imagesMerge = []
        self._height = height; self._width = width; self._chessSize = size; self._chessCellLen=length
        
        #如果存在立体标定的文件就进行读取
        if listd.count('left.npy')==1  and  listd.count('right.npy')==1 and listd.count('right.npy')==1:
            self.from_file_read(images_path)
            self.file_flag = True

        #读取标定图片
        # 定义所有可接受的扩展名集合（使用小写）
        valid_extensions = {'.jpg', '.png', '.bmp', '.jpeg', '.gif'}
        image_files = []

        for item_name in os.listdir(images_path):
            # 3. 构造完整路径并检查它是否是文件
            full_path = os.path.join(images_path, item_name)
            if os.path.isfile(full_path):
                # 4. 获取文件的扩展名并转换为小写（以便不区分大小写地匹配）
                # os.path.splitext(path) 将路径分割为 (root, ext)
                root, ext = os.path.splitext(item_name)
                
                # 5. 检查扩展名是否在有效列表中
                if ext.lower() in valid_extensions:
                    # 只添加文件名（不包含路径）
                    image_files.append(item_name)

        if len(image_files)<numbers:
            print("标定图像的数量不达标, 退出..........")
            return 
        
        print(image_files)
        for img_name in image_files:
            img = cv2.imread(os.path.join(images_path, img_name), 1)
            self._imagesMerge.append(img)

        self._imagesReady = True#设置文件读取完毕标志位

    def stereo_calibration(self, use_c_file = False, reversal=False):
        """只针对立体标定；
        >>> 形参：use_c_file设置为True表示尝试读取标定文件以避免重复标定，一般为True
        >>> 得到的标定参数的说明
        >>> 左相机： 相机内参：_leftParameters["cameraMatrix"]； 畸变向量：_leftParameters["distCoeffs"]
        >>>         矫正+校正映射： _leftParameters["map1"]； _leftParameters["map2"]
        >>> “
        >>> 右相机： 相机内参：_rightParameters["cameraMatrix"]； 畸变向量：_rightParameters["distCoeffs"]
        >>>         矫正+校正映射： _rightParameters["map1"]； _rightParameters["map2"]  
        >>> '
        >>> 公共参数：重投影矩阵： _stereoCommParameters["Q"]； 旋转：key->"R"； 平移； key->"T"
        """
        
        if self._stereoMode != True or self._imagesReady!=True:
            print("stereoMode is not be set True or no calibration's images recevied !!")
            return
        if self.file_flag == True and use_c_file == True:
            print('you have calibrated !')
            return
        #得到左右拆分图像
        src_left = [] ; src_right = []
        for i in range(0, len(self._imagesMerge)):
            item = self._imagesMerge[i]
            left = item[0:self._height, 0: self._width//2]
            right = item[0:self._height, self._width//2: self._width]
            leftg = cv2.cvtColor(left, cv2.COLOR_RGBA2GRAY)
            rightg = cv2.cvtColor(right, cv2.COLOR_RGBA2GRAY)
            src_left.append(leftg); src_right.append(rightg)

        #棋盘格上角点的真实坐标
        objp = np.zeros((self._chessSize[0]*self._chessSize[1],3), np.float32)
        for i in range(0, self._chessSize[1]):#row
            for j in range(0, self._chessSize[0]): #col
                point = [j*self._chessCellLen, i*self._chessCellLen, 0.]
                objp[i*self._chessSize[0]+j][0] = point[0]
                objp[i*self._chessSize[0]+j][1] = point[1]
                objp[i*self._chessSize[0]+j][2] = point[2]
        apoints = objp
            
        
        # termination criteria， 迭代的终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        img_points_left = []
        img_points_right = []
        
        objectpoints = [] #存储每一个棋盘格真实坐标的列表
        for i in range(0, len(self._imagesMerge)):
            # Find the chess board corners
            col = self._chessSize[0]; row = self._chessSize[1]
            ret_l, corners_left = cv2.findChessboardCorners(src_left[i], (col,row), None)
            ret_r, corners_right = cv2.findChessboardCorners(src_right[i], (col,row), None)
            if(ret_l == True and ret_r == True):
                # corners_left = cv2.cornerSubPix(src_left[i],corners_left, (11,11), (-1,-1), criteria)
                # corners_left = cv2.cornerSubPix(src_left[i],corners_left, (11,11), (-1,-1), criteria)
                if not reversal:
                    img_points_left.append(corners_left)
                    img_points_right.append(corners_right)
                else:
                    img_points_left.append(corners_right)
                    img_points_right.append(corners_left)

                objectpoints.append(apoints)


        # 单独标定左右摄像头, reversal 反转左右的定义
        _ret_l, mtx_l, dist_l, _rvecs_l, _tvecs_l = cv2.calibrateCamera(objectpoints, img_points_left, src_left[0].shape[::-1], None, None)
        _ret_r, mtx_r, dist_r, _rvecs_r, _tvecs_r = cv2.calibrateCamera(objectpoints, img_points_right, src_right[0].shape[::-1], None, None)

        # print(mtx_l)
        
        #单独畸变矫正，得到单独的畸变矫正映射
        map_sl1, map_sl2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, None, None, src_left[0].shape[::-1], cv2.CV_16SC2)
        map_sr1, map_sr2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, None, None, src_right[0].shape[::-1], cv2.CV_16SC2)
        self._leftParameters["map_s1"] = map_sl1; self._leftParameters["map_s2"] = map_sl2
        self._rightParameters["map_s1"] = map_sr1; self._rightParameters["map_s2"] = map_sr2


        #立体标定摄像头
        imageSize = src_right[0].shape[::-1]
        _retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate(objectpoints, img_points_left, img_points_right, \
        mtx_l, dist_l, mtx_r, dist_r, imageSize	)


        # 3D校正
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T)

        mapl1, mapl2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1,\
         R1, P1, imageSize, cv2.CV_16SC2)
        mapr1, mapr2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2,\
         R2, P2, imageSize, cv2.CV_16SC2)
        # print(Q)
        # print(cameraMatrix1)

        self._leftParameters["map1"] = mapl1; self._leftParameters["map2"] = mapl2
        self._leftParameters["cameraMatrix"] = cameraMatrix1; self._leftParameters["distCoeffs"]=distCoeffs1

        self._rightParameters["map1"] = mapr1; self._rightParameters["map2"] = mapr2
        self._rightParameters["cameraMatrix"] = cameraMatrix2; self._rightParameters["distCoeffs"]=distCoeffs2

        self._stereoCommParameters["Q"] = Q; self._stereoCommParameters["R"] = R; self._stereoCommParameters["T"] = T
        self._imageSize = imageSize
        #存储参数文件
        np.save(self.images_path+'left.npy', self._leftParameters)
        np.save(self.images_path+'right.npy', self._rightParameters)
        np.save(self.images_path+'comm.npy', self._stereoCommParameters)

    def calc_map_by_matrix(self, left,right ):
        """从左右相机的标定参数字典中获得畸变矫正映射；
        >>> 形参：left：左参数字典； right：右参数字典
        >>> 一般从C++迁移到Python需要用到此函数
        """
        cameraMatrix1 = left["cameraMatrix"] ; distCoeffs1 = left["distCoeffs"]
        R1 = left["R"]; P1 = left["P"]

        cameraMatrix2 = right["cameraMatrix"] ; distCoeffs2 = right["distCoeffs"]
        R2 = right["R"]; P2 = right["P"]

        mapl1, mapl2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1,\
         R1, P1, self._imageSize, cv2.CV_16SC2)
        mapr1, mapr2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2,\
         R2, P2, self._imageSize, cv2.CV_16SC2)

        self._leftParameters["map1"] = mapl1; self._leftParameters["map2"] = mapl2
        self._rightParameters["map1"] = mapr1; self._rightParameters["map2"] = mapr2
        
        
        #从标定文件中读取参数文件
    
    
    #*************以下为单目相机的标定函数，主要得到相机的内参数、畸变向量等***************#
    def monocular_print(self):
        print('\n','Internal matrix IS', '\n', self._monocularParameters["cameraMatrix"], '\n')
        print('Distortion vector is', '\n', self._monocularParameters["distCoeffs"], '\n')
    
    def monocular_from_file_read(self, path, c_file_name = 'monocular.npy'):
        self._monocularParameters  = np.load(path + c_file_name, allow_pickle=True).item()
    
    def monocular_set_calibration_images(self, path, c_file_name = 'monocular.npy' ,only_use_c_file = False,\
        numbers=10, height=480, width=640, size=[4,6], length=24.6, mode = 'square'):
        """只针对单目标定；
        设置标定图像文件路径、标定好的文件名、是否只使用标定文件、图像数量、图像尺寸、标定板大小、单元实际长度mm
        """
        
        listd = os.listdir(path) #检查文件夹下是否有指定数量的标定文件
        if len(listd)<numbers:
            print("this director is not have so many images")
            print('You will be checked to see if you have the calibrated parameters file!!!')
        
        self._monocular_path = path
        self._monocular_image_width = width
        self._monocular_image_height = height
        self._monocular_chessSize = size
        self._monocular_chessCellLen = length
        self._monocular_mode = mode
        
        #如果存在单目标定的文件就进行读取
        if listd.count( c_file_name )==1 :
            self.monocular_from_file_read(path, c_file_name)
            self._monocular_file_flag = True
            print('Check your calibration file ^_^')
            if only_use_c_file == True: #仅仅使用标定文件且检测到标定文件就在此跳出
                return True
        else:
            print('You do not have a calibration file')

        list_images = []
        for item in listd:
            k = item.split('.')[-1]
            if k == 'jpg' or k == 'jpeg' or k == 'png' or k=='bmp':
                list_images.append(item)
        try:
            #读取制定数量的标定图片
            for i in range(1, numbers):
                file_name = os.path.join(path, list_images[i])
                img = cv2.imread(file_name, 1)
                self._monocular_images.append(img)
            
            self._monocular_image_ready = True
        except :
            print('An error occurred. Please check the path and quantity of your pictures')
            print('You will work with the calibration file')
            if self._monocular_file_flag == False:
                print('But you do not have a calibration file. Please calibrate your camera first')
        
    
    def monocular_calibration(self, c_file_name = 'monocular.npy', use_c_file = True):
        """只针对单目标定；
        >>> 形参：use_c_file设置为True表示尝试读取标定文件以避免重复标定，一般为True; c_file_name:存储相机参数文件名
        >>> 得到的标定参数的说明；注意：以列向量的形式组织
        >>> 相机内参：_monocularParameters["cameraMatrix"]； 畸变向量：_monocularParameters["distCoeffs"]
        >>>         矫正映射： _monocularParameters["map1"]； _monocularParameters["map2"]
        """
        if self._monocular_file_flag == True and use_c_file == True:
            print('you have calibrated for monocular !')
            self._monocular_calibration_finished = True
            return True
        
        if self._monocular_image_ready == False:
            print(" No calibration's images recevied !!, Please check your picture path and quantity Settings")
            return False
        
        src_left = []
        for i in range(0, len(self._monocular_images)):
            item = self._monocular_images[i]
            leftg = cv2.cvtColor(item, cv2.COLOR_RGBA2GRAY)
            src_left.append(leftg)
        
        
        #棋盘格上角点的真实坐标
        objp = np.zeros((self._monocular_chessSize[0]*self._monocular_chessSize[1],3), np.float32)
        for i in range(0, self._monocular_chessSize[1]):#row
            for j in range(0, self._monocular_chessSize[0]): #col
                point = [j*self._monocular_chessCellLen, i*self._monocular_chessCellLen, 0.]
                objp[i*self._monocular_chessSize[0]+j][0] = point[0]
                objp[i*self._monocular_chessSize[0]+j][1] = point[1]
                objp[i*self._monocular_chessSize[0]+j][2] = point[2]
        apoints = objp
        
        
        # termination criteria， 迭代的终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        img_points_left = []
        objectpoints = [] #存储每一个棋盘格真实坐标的列表
        for i in range(0, len(self._monocular_images)):
            # Find the chess board corners
            col = self._monocular_chessSize[0]; row = self._monocular_chessSize[1]
            if self._monocular_mode == 'square':
                ret_l, corners_left = cv2.findChessboardCorners(src_left[i], (col,row), None)
            if self._monocular_mode == 'circle':
                flags = cv2.CALIB_CB_SYMMETRIC_GRID
                params = cv2.SimpleBlobDetector_Params()
                # params.maxCircularity = 1000
                params.maxArea = 100000
                blob = cv2.SimpleBlobDetector_create(params)
                pra = cv2.CirclesGridFinderParameters()
                flags = cv2.CALIB_CB_SYMMETRIC_GRID #对称的圆网格
                ret_l, corners_left = cv2.findCirclesGrid(src_left[i], (col,row), flags)
                
            if(ret_l == True ):
                # corners_left = cv2.cornerSubPix(src_left[i],corners_left, (11,11), (-1,-1), criteria)
                img_points_left.append(corners_left)
                objectpoints.append(apoints)
        
        if len(img_points_left) <= 2:
            print('标定板角点检测异常2345 !!!!!!!!!!!')
        # 标定摄像头
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(\
        objectpoints, img_points_left, src_left[0].shape[::-1], None, None)
        
        #单独畸变矫正，得到单独的畸变矫正映射
        
        map_sl1, map_sl2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, None, mtx_l, src_left[0].shape[::-1], cv2.CV_16SC2)
        self._monocularParameters["map1"] = map_sl1; self._monocularParameters["map2"] = map_sl2
        self._monocularParameters["cameraMatrix"] = mtx_l; self._monocularParameters["distCoeffs"]=dist_l


        #存储参数文件
        np.save(self._monocular_path + c_file_name , self._monocularParameters)
        
        self._monocular_calibration_finished = True
        return True
        
        
    def monocular_remap(self, src_img):
        if self._monocular_calibration_finished == True:
            pass
        else:
            print('Please calibrate your camera first!!!')
            return False
        
        dst_img = cv2.remap(src_img, self._monocularParameters["map1"], self._monocularParameters["map2"], \
        cv2.INTER_LINEAR)
        return dst_img
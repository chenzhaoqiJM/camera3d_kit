#!/usr/bin/env python3
# coding=utf8
import  rospkg
import numpy as np
import cv2
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from ccalibration import CCameraCalibration
from creconstruction import Reconstruction
from  creconstruction import  Calc3DPose
from cccparameters import CParameters

class stereoWrapper():
    def __init__(self):
        self.myp = CParameters()

        #相机标定类
        self.mycc = CCameraCalibration()  #初始化
        #设置标定图像的路径和图片数量,图片大小、标定板尺寸等
        self.mycc.set_calibration_images("C:\\Users\\hp\\Desktop\\1280\\91\\", 15, 480, 1280, [6,8], 24.6)
        #进行双目立体标定，得到内参矩阵、畸变系数、重映射矩阵map、重投影矩阵Q等 
        self.mycc.stereo_calibration(True) 

        #相机三维重构类
        self.myrec = Reconstruction()
        # myrec.inputQ(myp._comm["Q"])
        self.myrec.inputQ(mycc._stereoCommParameters["Q"]) #输入标定得到的重投影矩阵Q
        # myrec.calc_M_by_RT(mycc) #通过标定得到的旋转矩阵和平移向量计算两个摄像头的投影矩阵，以左相机坐标系为基准
        img_pnp = cv2.imread('C:\\Users\\hp\\Desktop\\1280\\91\\1.jpg')
        self.myrec.inputPnPImg(img_pnp, 480, 1280, (6,8), 24.6)
        self.myrec.calc_r_and_t_by_PnP(mycc, True)

        #3d位姿计算类
        self.my3d = Calc3DPose()
        self.my3d.input_Rec(myrec) #输入重构类对象
        self.my3d.calc_R_and_T() #计算手眼标定的旋转矩阵R和平移向量T

        self.cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        
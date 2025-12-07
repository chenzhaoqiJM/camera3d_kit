#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from cv2 import cv2
import  math
from numpy import ma
from scipy.optimize import leastsq
from math import sin, cos
from ctf import TF


class MonocularRec:
    def __init__(self, ccp):
        
        """需要输入单目标定参数, 字典形式，与标定类中的格式一样
        >>> 可以使用其它方法标定到的参数，但键必需为：内参：ccp['cameraMatrix']； 畸变向量：ccp['distCoeffs']
        >>> 输入的内参矩阵必须以列向量形式组织。
        >>> 计算说明： 全部使用行向量形式进行计算
        """
        self._monocularParameters = dict()
        self._monocularParameters = ccp
        self._InternalMatrix = self._monocularParameters['cameraMatrix']

    def img_point_to_x_angle(self, p, is_degree = False):
        """计算在XZ平面上，（像素点对应的世界点与相机的连线）与光轴的夹角
        >>> 输入p[x,y]像素坐标; is_degree:返回角度设置为True，设置为False将返回弧度
        """
        u = p[0]
        cx = self._InternalMatrix[0][2]
        fx = self._InternalMatrix[0][0]
        
        theta = math.atan2(u-cx, fx)
        if is_degree == False:
            return theta
        if is_degree == True:
            return theta * 180.0 / math.pi
        
    def img_point_to_y_angle(self, p, is_degree = False):
        """计算在YZ平面上，（像素点对应的世界点与相机的连线）与光轴的夹角
        >>> 输入p[x,y]像素坐标; is_degree:返回角度设置为True，设置为False将返回弧度
        """
        v= p[1]
        cy = self._InternalMatrix[1][2]
        fy = self._InternalMatrix[1][1]
        
        theta = math.atan2(v-cy, fy)
        if is_degree == False:
            return theta
        if is_degree == True:
            return theta * 180.0 / math.pi
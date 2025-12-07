#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import os

class CParameters:
    def __init__(self):
        """
        >>> 实现标定与重构的解耦，你可以使用其它标定软件，然后把参数输入到该类
        >>> 所有参数以列向量形式存储
        """  
        self._leftParameters = dict()
        self._rightParameters = dict()
        self._stereoCommParameters = dict()

    def init_by_cc(self, cc):
        """用立体标定对象初始化参数类；
        """
        self._leftParameters = cc._leftParameters
        self._rightParameters = cc._rightParameters
        self._stereoCommParameters = cc._stereoCommParameters
        
    def init_by_hand(self):
        """使用手动输入的参数，需要在类里面设置
        """
        self.init_leftParameters()
        self.init_rightParameters()
        self.init_stereoCommParameters()

    def init_stereoCommParameters(self):
        Q = [[1, 0, 0, -317.1514282226562],\
        [0, 1, 0, -264.7153873443604],\
        [0, 0, 0, 479.8703544516589],\
        [0, 0, 0.01669070971845944, -0]]
        self._stereoCommParameters["Q"] = np.array(Q)

    def init_leftParameters(self):
        cameraMatrix = [[480.6342566971228, 0, 278.0589273369235],\
        [0, 479.8703544516589, 268.7681944747201],\
        [0, 0, 1]]
        self._leftParameters["cameraMatrix"] = np.array(cameraMatrix)

        distCoeffs = [0.09293069293251677, -0.06715517584335046, \
        -0.002162906462548637, -0.001991891799426002, -0.07705557332711026]
        self._leftParameters["distCoeffs"] = np.array(distCoeffs)

        R = [[0.9990590784452132, 0.002811640934202277, -0.04327877599307366],\
            [-0.002781689782516758, 0.9999958481458816, 0.0007522585676369707],\
            [0.04328071138689119, -0.000631162622356141, 0.9990627516105226]]
        self._leftParameters["R"] = np.array(R)

        P = [[479.8703544516589, 0, 317.1514282226562, 0],\
        [0, 479.8703544516589, 264.7153873443604, 0]\
        ,[0, 0, 1, 0]]
        self._leftParameters["P"] = np.array(P)

    def init_rightParameters(self):
        cameraMatrix = [[488.082113823578, 0, 304.2083621693459],\
        [0, 488.1764598831438, 262.7688625177545],\
        [0, 0, 1]]
        self._rightParameters["cameraMatrix"] = np.array(cameraMatrix)

        distCoeffs = [0.09847204607049365, -0.1314452320682229,\
         -0.002904018839132114, -0.0004338561092788612, 0.1338313090750929]
        self._rightParameters["distCoeffs"] = np.array(distCoeffs)

        R = [[0.9991154084302503, 0.003908603879825309, -0.04187031708698821],\
            [-0.003937578309584333, 0.9999920619276906, -0.0006095561184391152],\
            [0.04186760220397405, 0.0007738845626126001, 0.9991228678137507]]
        self._rightParameters["R"] = np.array(R)

        P = [[479.8703544516589, 0, 317.1514282226562, -28750.74592669576],\
        [0, 479.8703544516589, 264.7153873443604, 0]\
        ,[0, 0, 1, 0]]
        self._rightParameters["P"] = np.array(P)
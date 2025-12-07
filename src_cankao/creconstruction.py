#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
from cv2 import cv2
import  math
import matplotlib.pyplot as plt
from numpy import ma
from scipy.optimize import leastsq
from math import sin, cos
from cccparameters import CParameters
from ctf import TF

try:
    from geometry_msgs.msg import PoseStamped ,PoseWithCovarianceStamped
    import rospy
except:
    print('导入ROS模块出错，可能是非ROS环境')

class Reconstruction:
    def __init__(self,ccp):
        """需要输入参数类对象
        >>> 全部使用行向量形式进行计算，因此会转置重投影矩阵Q
        """
        self.QS = ccp._stereoCommParameters["Q"]
        self.Q = np.array(self.QS).T; self.QInv = (np.linalg.inv(self.Q))
        
        self.chess_size = (4, 6); self.chess_cell = 24.6
        
        #使用左右内参数，以及PNP或者R T计算左右相机投影矩阵
        self.pnp_img = 0 ;self.pheight = 480; self.pwidth = 1280; self.psize = (4,6); self.plen = 24.6
        self.leftM = np.ones([3,4],dtype=np.float64); self.rightM = np.ones([3,4],dtype=np.float64)
        self.left_mtx = ccp._leftParameters["cameraMatrix"]
        self.right_mtx = ccp._rightParameters["cameraMatrix"]
        
        #初始化立体匹配对象
        self.stereoBM = cv2.StereoBM_create(numDisparities=128, blockSize=15)
        self.stereoSGBM = cv2.StereoSGBM_create()
        self.disparity = 0
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.mytf = TF()    #初始化几何变换对象

        try:
            self.tf_pub_init_flag = True
            self.tf_pub = rospy.Publisher('tf_demo', PoseStamped, queue_size=1) 
        except :
            print('publisher init failed!')
            self.tf_pub_init_flag = False
        self.real_dis = fitLine()
        
        
    #输入Q矩阵
    def inputQ(self, q):
        """直接输入重投影矩阵，以列向量形式组织  
        """
        self.Q = np.array(q).T
        self.QInv = (np.linalg.inv(self.Q))
        
    def inputPnPImg(self, img, height=480, width=1280, size=(4,6), length=24.6):
        """输入PnP的图像，图像高与宽，棋盘格尺寸、棋盘格单元大小 
        """
        self.pnp_img = img; self.pheight = height; self.pwidth = width; self.psize = size; self.plen = length
    
    #利用左图和右图的相应点计算三维坐标，输入为一对点的列表  
    def img_to_world(self, pl, pr):
        """利用左图和右图的相应点计算三维坐标，基于视差
        >>> 输入: pl:[x, y]; pr[x, y]
        >>> 以行向量的方式计算
        >>> 返回三维齐次坐标，形式：[[x, y, z, 1]]
        """
        img_p = [ 300. ,300. ,32. ,1.]
        if(len(pl)!=2 or len(pr)!=2):
            return []
        img_p[0] = pl[0]; img_p[1] = pl[1]; img_p[2] = pl[0]-pr[0]
        img_p = np.array(img_p) 
        img_p.resize(1,4)
        w = np.dot(img_p ,self.Q)
        w = w/w[0][3]
        return w
    
    def img_to_world_by_m(self,pl,pr):
        """利用左右投影矩阵计算三维坐标，输入为一对点的列表  
        >>> 以行向量的方式计算
        """
        A = np.ones([4,3],dtype=np.float64)
        B = np.ones([4,1],dtype=np.float64)

        u1 = pl[0]; v1 = pl[1]
        u2 = pr[0]; v2 = pr[1]
        m1 = self.leftM; m2 = self.rightM
        
        A[0][0] = u1* m1[2][0] - m1[0][0] ; A[0][1] = u1*m1[2][1] - m1[0][1] ; A[0][2] = u1*m1[2][2] - m1[0][2]
        A[1][0] = v1* m1[2][0] - m1[1][0] ; A[1][1] = v1*m1[2][1] - m1[1][1] ; A[1][2] = v1*m1[2][2] - m1[1][2]
        A[2][0] = u2* m2[2][0] - m2[0][0] ; A[2][1] = u2*m2[2][1] - m2[0][1] ; A[2][2] = u2*m2[2][2] - m2[0][2]
        A[3][0] = v2* m2[2][0] - m2[1][0] ; A[3][2] = v2*m2[2][1] - m2[1][1] ; A[3][2] = v2*m2[2][2] - m2[1][2]
        
        # print(A)
        B[0][0] = m1[0][3] - u1*m1[2][3]; B[1][0] = m1[1][3] - v1*m1[2][3]
        B[2][0] = m2[0][3] - u2*m2[2][3]; B[3][0] = m2[1][3] - v2*m2[2][3]
        
        u = np.linalg.lstsq(A, B, rcond=None)
        
        a = np.ones([3,3],dtype=np.float64) ; a[0] = A[0];a[1] = A[1]; a[2] = A[2]
        b = np.ones([3,1],dtype=np.float64) ; b[0] = B[0];b[1] = B[1]; b[2] = B[2] 
        v1 = np.linalg.solve(a, b) 
        
        a[0] = A[0];a[1] = A[1]; a[2] = A[3]; b[0] = B[0];b[1] = B[1]; b[2] = B[3]
        v2 = np.linalg.solve(a, b)
        
        a[0] = A[0];a[1] = A[2]; a[2] = A[3]; b[0] = B[0];b[1] = B[2]; b[2] = B[3]
        v3 = np.linalg.solve(a, b)
        
        a[0] = A[1];a[1] = A[2]; a[2] = A[3]; b[0] = B[1];b[1] = B[2]; b[2] = B[3]
        v4 = np.linalg.solve(a, b)    
        # print(u)
        print('\n', v1, '\n', v2, '\n', v3, '\n', v4)
        
    def calc_r_and_t_by_PnP(self, cc, nondis = False):
        """利用PnP计算左右相机之间的R与T（旋转与平移）及投影矩阵
        >>> 输入：参数类对象； 是否做畸变矫正
        """
        left = self.pnp_img[0:self.pheight, 0: self.pwidth//2]
        right = self.pnp_img[0:self.pheight, self.pwidth//2: self.pwidth]
        if nondis == True:  
            left = cv2.remap(left, cc._leftParameters["map1"], cc._leftParameters["map2"], \
            cv2.INTER_LINEAR)
            right = cv2.remap(right, cc._rightParameters["map1"], cc._rightParameters["map2"], \
            cv2.INTER_LINEAR)
                
        left = cv2.cvtColor(left, cv2.COLOR_RGBA2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_RGBA2GRAY)
        
        col = self.psize[0]; row = self.psize[1]
        ret_l, corners_left = cv2.findChessboardCorners(left, (col,row), None)
        ret_r, corners_right = cv2.findChessboardCorners(right, (col,row), None)
        if(ret_l != True or ret_r != True):
            return False
        
        objp = np.zeros((self.psize[0]*self.psize[1],3), np.float32)
        for i in range(0, self.psize[1]):#row
            for j in range(0, self.psize[0]): #col
                point = [j*self.plen, i*self.plen, 0.]
                objp[i*self.psize[0]+j][0] = point[0]
                objp[i*self.psize[0]+j][1] = point[1]
                objp[i*self.psize[0]+j][2] = point[2]
                
        cameraMatrix1 =  cc._leftParameters["cameraMatrix"]; distCoeffs1 = cc._leftParameters["distCoeffs"]
        cameraMatrix2 =  cc._rightParameters["cameraMatrix"]; distCoeffs2 = cc._rightParameters["distCoeffs"]
        retval_l, rvec_l, tvec_l = cv2.solvePnP(objp, corners_left, cameraMatrix1, distCoeffs1)
        retval_r, rvec_r, tvec_r = cv2.solvePnP(objp, corners_right, cameraMatrix2, distCoeffs2)
        rl ,  jacobian = cv2.Rodrigues(rvec_l); rr,  jacobian = cv2.Rodrigues(rvec_r)
        
        # print(rl, "\n", tvec_l, "\n")
        leftRT = np.ones([4,4],dtype=np.float64); rightRT = np.ones([4,4],dtype=np.float64)
        leftRT[0:3, 0:3]=rl; leftRT[0:3, 3] = tvec_l.T ;leftRT[3, 0:3]=[0,0,0]; leftRT[3, 3] = 1.0
        rightRT[0:3, 0:3]=rr; rightRT[0:3, 3] = tvec_r.T ;rightRT[3, 0:3]=[0,0,0]; rightRT[3, 3] = 1.0
        
        #直接用得到的投影矩阵：
        # self.leftM = np.dot( cameraMatrix1, leftRT[0:3])
        # self.rightM = np.dot( cameraMatrix2, rightRT[0:3])
        
        # print(leftRT, '\n', rightRT)
        r_to_lM = np.dot(rightRT, np.linalg.inv(leftRT))
        self.rightM =  np.dot(cameraMatrix2, r_to_lM[0:3])
        # print('\n', r_to_lM)
        
        left_R = [[1,0,0],[0,1,0],[0,0,1]]; left_T = [0,0,0]
        leftRT = np.ones([3,4],dtype=np.float64)
        leftRT[0:3, 0:3]=left_R; leftRT[0:3, 3]=left_T
        self.leftM = np.dot( cameraMatrix1, leftRT) #左投影矩阵
        return True

        
        
    def calc_M_by_RT(self, cc):
        """利用标定得到的R与T（旋转与平移）计算投影矩阵
        >>> 输入：参数类对象
        """
        self.left_mtx = cc._leftParameters["cameraMatrix"]
        self.right_mtx = cc._rightParameters["cameraMatrix"]
        
        left_R = [[1,0,0],[0,1,0],[0,0,1]]; left_T = [0,0,0]
        leftRT = np.ones([3,4],dtype=np.float64)
        leftRT[0:3, 0:3]=left_R; leftRT[0:3, 3]=left_T
        self.leftM = np.dot( self.left_mtx, leftRT) #左投影矩阵
        
        R = cc._stereoCommParameters['R']  
        T = cc._stereoCommParameters['T']
        # T = - np.dot(R.T, T) ;print(T, "\n")
        # R = R.T
        rightRT = np.ones([3,4],dtype=np.float64)
        rightRT[0:3, 0:3]=R; rightRT[0:3, 3]=T.T 
        self.rightM = np.dot( self.right_mtx, rightRT)#生成右摄像头投影矩阵

    
    #利用左图上对应的一点以及立体匹配得到的视差计算三维坐标
    def img_to_world_by_dis(self, pl, dis):
        """利用左图上对应的一点以及立体匹配得到的视差计算三维坐标
        >>> 输入： pl[x,y]; dis:double
        >>> 以行向量形式计算
        >>> 正常返回值：三维齐次坐标；[[x, y, z, 1]]
        """
        img_p = [ 300. ,300. ,32. ,1.]
        if(len(pl)!=2 ):
            return []
        img_p[0] = pl[0]; img_p[1] = pl[1]; img_p[2] = dis
        img_p = np.array(img_p)
        img_p.resize(1,4)
        w = np.dot(img_p ,self.Q  )
        w = w/w[0][3]
        return w

    #输入三维坐标计算其在左图中的对应点，输入为列表
    def world_to_img_l(self, pw):
        """利用三维坐标计算其在左图中的对应点，
        >>> 输入为:[x, y, z], [x, y, z, 1]
        >>> 以行向量形式计算
        >>> 正常返回值：二维齐次坐标；[[x, y, 1]]
        """
        w_p = [0., 0. ,550. ,1]
        if(len(pw)!=3 and len(pw)!=4):
            return []
        w_p[0] = pw[0]; w_p[1] = pw[1] ; w_p[2] = pw[2]; w_p[3] = pw[3]
        w_p = np.array(w_p)
        w_p.resize(1,4)
        img_p = np.dot(w_p, self.QInv )
        img_p = img_p/img_p[0][3]
        return img_p
    
    #检测棋盘格角点并返回
    def check_chess_corners(self, left, right, chess_size, draw_flag=True):
        """检测棋盘格角点并返回
        >>> 输入为左右图像、棋盘格大小、是否绘制检测结果默认为True
        >>> 正常返回值：是否检测到的标志Bool类型；棋盘角点：[ [[x,y,z]], [[x,y,z]], .....]
        """
        self.chess_size = chess_size 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret_l, corners_left = cv2.findChessboardCorners(left, chess_size, None)
        ret_r, corners_right = cv2.findChessboardCorners(right, chess_size, None)
        if(ret_l == True and ret_r == True):
            grayl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            corners_left = cv2.cornerSubPix(grayl,corners_left, (11,11), (-1,-1), criteria)
            grayr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            corners_right = cv2.cornerSubPix(grayr,corners_right, (11,11), (-1,-1), criteria)
            if draw_flag == True: 
                cv2.drawChessboardCorners(left, chess_size, corners_left, ret_l)
                cv2.drawChessboardCorners(right, chess_size, corners_right, ret_r)
        return ret_l ,corners_left, ret_r, corners_right
    
    #打印棋盘格第一个点的视差或者其三维坐标
    def print_zero(self, left, right, ret_l, corners_left, ret_r, corners_right ):
        if(ret_l and ret_r):
            w = self.img_to_world(corners_left[0][0], corners_right[0][0])
            a = corners_left[0][0][0]; b = corners_left[0][0][1]
            left_s = (int(a), int(b))
            print(left_s)
            cv2.circle(left, left_s ,3, (255,0,0),  2)
        
            return w
        else:
            return []
        
    def print_len(self, ret_l, corners_left, ret_r, corners_right ):
        if(ret_l and ret_r):
            w0 = self.img_to_world(corners_left[0][0], corners_right[0][0])
            w1 = self.img_to_world(corners_left[3][0], corners_right[3][0])
            w = w1-w0; w = np.array(w[0, 0:3])
            print( math.sqrt(np.dot(w,w)) )
            
    def hand_eye_calibration(self, left, right, ret_l, corners_left, ret_r, corners_right ):
        if(ret_l and ret_r):
            index0 = 0; indexx = self.chess_size[0]-1; indexy = self.chess_size[0]*(self.chess_size[1]-1)
            w0 = self.img_to_world(corners_left[index0][0], corners_right[index0][0])
            wx = self.img_to_world(corners_left[indexx][0], corners_right[indexx][0])
            wy = self.img_to_world(corners_left[indexy][0], corners_right[indexy][0])
            
            vx = wx - w0;  vx = vx / np.linalg.norm(vx) 
            vy = wy - w0; vy = vy / np.linalg.norm(vy)
            w1 = w0 - vx * self.chess_cell  - vy * self.chess_cell
            
            w1 = [w1[0][0], w1[0][1], w1[0][2], w1[0][3]]
            imp1 = self.world_to_img_l(w1);  p_l = [int(imp1[0][0]), int(imp1[0][1])]
            cv2.circle(left, tuple(p_l) ,3, (0,255,0),  2)
            # print(w1)
            
    
    #利用立体匹配得到视差图并计算棋盘格角点的世界坐标    
    def calc_dis_BM(self, left, right):
        grayl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        grayr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        self.disparity = self.stereoBM.compute(grayl,grayr)
        
    def calc_dis_SGBM(self, left, right):
        self.stereoSGBM.setNumDisparities(128)
        grayl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        grayr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        self.disparity = self.stereoSGBM.compute(grayl, grayr)

    def get_bm_dis(self, pl):
        """从立体匹配图中获得视差
        >>> 输入为:pl[x, y]
        >>> 正常返回值：视差：double
        """
        dis = self.disparity[int(pl[1])][int(pl[0])] /16.0 
        return dis
    
    def get_int_corners(self, ret_l, left, ret_r, right):
        """获得整数级的棋盘格角点，精度将会有一点损失，用于绘图
        """
        left_c = []
        right_c = []
        if ret_l == True and ret_r == True:
            for i in range(0, len(left)):
                a = left[i][0][0]; b = left[i][0][1]
                temp_left = [[int(a), int(b)]]
                left_c.append(temp_left)
                a = right[i][0][0]; b = right[i][0][1]
                temp_right = [[int(a), int(b)]]
                right_c.append(temp_right)
        return left_c, right_c
    
    def draw_cube_left(self, left, left_corners, right_corners, chess_size, cube=False):
        corners_left_int, corners_right_int = self.get_int_corners(True, left_corners, True, right_corners)
        
        index1 = 0; index2 = chess_size[0]-1; index3 = chess_size[0]*(chess_size[1]-1 ); index4 = index3 + index2
        oringin = corners_left_int[index1][0]; x_end = corners_left_int[index2][0]
        y_end = corners_left_int[index3][0]
        cv2.arrowedLine(left, tuple(oringin), tuple(x_end), (0,0,255), 3)
        cv2.arrowedLine(left, tuple(oringin), tuple(y_end), (255,0,0), 3)
        w1 = self.img_to_world(left_corners[index1][0], right_corners[index1][0])
        w2 = self.img_to_world(left_corners[index2][0], right_corners[index2][0])
        w3 = self.img_to_world(left_corners[index3][0], right_corners[index3][0])
        vx = (w2-w1)[0][0:3]; vy = (w3-w1)[0][0:3]; vx = np.array(vx); vy = np.array(vy)
        
        
        len_x = math.sqrt(np.dot(vx, vx)); len_y = math.sqrt(np.dot(vy, vy))
        text_x = 'len:'+ str(len_x)[0:5]+'mm'; text_x_fact = 'real len:74mm'
        position_x = (int((oringin[0]+x_end[0])/2.0), int((oringin[1]+x_end[1])/2.0)-10)
        position_x_fact = (position_x[0], position_x[1] - 20)
        
        text_y = 'len:'+ str(len_y)[0:5]+'mm'; text_y_fact= 'real len:123mm'
        position_y = (int((oringin[0]+y_end[0])/2.0)-160, int((oringin[1]+y_end[1])/2.0)+40 )
        position_y_fact = (position_y[0], position_y[1]-20)
        
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(left, text_x_fact, position_x_fact, font, 0.7, (0, 255, 255), 1)
        cv2.putText(left, text_x, position_x, font, 0.7, (255, 255, 0), 1)
        
        cv2.putText(left, text_y_fact, position_y_fact, font, 0.7, (0, 255, 255), 1)
        cv2.putText(left, text_y, position_y, font, 0.7, (255, 255, 0), 1)
        
        text_o = 'The depth of origin: ' + str(w1[0][2])[0:6] + 'mm'
        cv2.putText(left, text_o, (30, 30), font, 0.7, (255, 255, 0), 1)
        text_c_o = "Oringin's 3D coordinate:" +\
            '(' + str(w1[0][0])[0:6] + ',' + str(w1[0][1])[0:6]+',' + str(w1[0][2])[0:6] + ')'
        cv2.putText(left, text_c_o, (30, 30+20), font, 0.7, (255, 255, 0), 1)   
        text_oo = 'oringin'
        cv2.putText(left, text_oo, (oringin[0]-90, oringin[1]-10), font, 0.7, (255, 0, 255), 1)
        
        vz = np.cross(vy, vx); vz = vz / math.sqrt(np.dot(vz,vz))
        w_z = w1[0][0:3] + vz * 50; 
        w_z = w_z.tolist(); w_z.append(1.0)
        p_z = self.world_to_img_l(w_z); p_z = (int(p_z[0][0]), int(p_z[0][1]))
        cv2.arrowedLine(left, tuple(oringin), p_z, (0,255,0), 3)
        
        if self.tf_pub_init_flag == True:
            try:
                vx = vx / math.sqrt(np.dot(vx, vx)); vy = vy / math.sqrt(np.dot(vy, vy))
                rotation = np.eye(3)
                rotation[0, 0:3] = vx; rotation[1, 0:3] = vy; rotation[2, 0:3] = vz
                # rotation = rotation.T
                rotation = self.mytf.matrix_orthogonalization(rotation)
                four_rotation = self.mytf.rotation_to_homogeneous(rotation)

                # mq = self.mytf.quaternion_from_matrix(four_rotation)
                el = self.mytf.euler_from_matrix(four_rotation)
                # mq = self.mytf.quaternion_from_euler(el[0], el[1], el[2], 'sxyz')
                mq = [el[0], el[1], el[2], 1]
                # length = math.sqrt(np.dot(mq, mq))

                t = w1[0][0:3]/1000.0
                pos = PoseStamped()
                pos.header.stamp = rospy.Time.now()
                pos.pose.position.x = t[0]; pos.pose.position.y = t[1]; pos.pose.position.z = t[2]
                pos.pose.orientation.x = mq[0]; pos.pose.orientation.y=mq[1]; pos.pose.orientation.z=mq[2]
                pos.pose.orientation.w = mq[3]
                self.tf_pub.publish(pos)
                print('mq', mq)
                print('T', t)
            except:
                print('TF BC failed!')
        else:
            pass
        
        if cube==True:
            pass        
        
        
    #检测ORB关键点并画出
    def feature_check(self, img):
        kp = self.orb.detect(img, None)
        kp, des = self.orb.compute(img, kp)
        img2 = cv2.drawKeypoints(img, kp, None, color =(0,255,0), flags = 0)
        return img2
    
    #检测并匹配ORB关键点
    def feature_check_and_match(self, left, right):
        kp1, des1 = self.orb.detectAndCompute(left ,None)
        kp2, des2 = self.orb.detectAndCompute(right ,None)
        matches = self.bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(left ,kp1, right,kp2,matches[:40],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img3
        


class Calc3DPose():
    def __init__(self):
        self.left_v = [[212,352], [256,310], [239,368]]   #左图像里面的像素坐标
        self.right_v = [[162,352], [206,310], [170, 368]]    #右图对应的像素坐标
        self.oringin = [[255,353], [205,352]]   #向量起点像素坐标
        
        self.w_v = np.array([[-50,0,0], [0,-50,0], [0,0,157]]) #自定义坐标系里的三个向量，起点这里为（0，0，0）
        
        self.Rec = 0
        self.Rotation = np.ones([3,3], dtype=np.float64)
        self.Trans = np.ones([3,1], dtype=np.float64)
        self.RT = np.ones([4,4],dtype=np.float64)
        self.c_o = np.ones([1,3], dtype=np.float64)
        self.shear = np.ones([3,3], dtype=np.float64)
        
        self.x_list_x = []
        self.x_list_y = []
        self.fit = fitLine()
        
        self.mytf = TF()
    

        
    def calc_3d_pose(self, wc, wl, wr, danwei = 0.001):
        wc = wc[:-1]; wl = wl[:-1]; wr = wr[:-1]
        wc = np.array(wc); wl = np.array(wl); wr = np.array(wr)
        vlr = wl-wr; distance = math.sqrt(np.dot(vlr, vlr))
        # print("OBJECT_LENGTH: ", distance)
        
        pos = PoseWithCovarianceStamped()
        pos.pose.pose.position.x = wc[0]*danwei; pos.pose.pose.position.y = wc[1]*danwei
        pos.pose.pose.position.z = wc[2]*danwei
        
        e_ang = self.mytf.vector_to_angle( wl-wc)
        print("el_anfles is : ", np.array(e_ang)/math.pi*180)
        mq = self.mytf.quaternion_from_euler(e_ang[0], e_ang[1], e_ang[2], 'sxyz')
        # mq = self.mytf.quaternion_from_euler(math.pi/4, 0, math.pi/4, 'sxyz')

        pos.pose.pose.orientation.x = mq[0]; pos.pose.pose.orientation.y = mq[1]
        pos.pose.pose.orientation.z = mq[2]; pos.pose.pose.orientation.w = mq[3]
        
        pos.pose.covariance[0] = distance*danwei
        pos.pose.covariance[1] = e_ang[0]/math.pi*180; pos.pose.covariance[2] = e_ang[1]/math.pi*180
        pos.pose.covariance[3] = e_ang[2]/math.pi*180
        pos.pose.covariance[4:7] = wc*danwei; pos.pose.covariance[7:10]=wl*danwei; pos.pose.covariance[10:13]=wr*danwei
        return pos
         
    def input_Rec(self,rec):
        self.Rec = rec
    
    def calibra_depth_by_chess(self, left, right, ret_l, corners_left, ret_r, corners_right ):
        if ret_l == False or ret_r == False:
            return
            
        x_list_x = []; x_list_y = []
        xp_list_x = []; xp_list_y = []
        
        y_list_x = []; y_list_y = []
        yp_list_x = []; yp_list_y = []
        
        len_x= 4; len_y = 6
        for i in range(0, len_x):
            dis = corners_left[i][0][0]-corners_right[i][0][0]
            w = self.Rec.img_to_world(corners_left[i][0],corners_right[i][0])
            a1 = [corners_left[i][0][0], dis]
            x_list_x.append(w[0][0]); x_list_y.append(dis)
            self.x_list_x.append( corners_left[i][0][0] ); self.x_list_y.append(dis)
            xp_list_x.append(corners_left[i][0][0]); xp_list_y.append(dis)
            
        for i in range(0, len_y):
            index = len_x * i
            dis = corners_left[index][0][0]-corners_right[index][0][0]
            w = self.Rec.img_to_world(corners_left[index][0],corners_right[index][0])
            a1 = [corners_left[index][0][1], dis]
            y_list_x.append(w[0][1]); y_list_y.append(dis)
            yp_list_x.append(corners_left[index][0][1]); yp_list_y.append(dis)
            
        
        # plt.plot(x_list_x ,x_list_y, linewidth=5)
        # plt.scatter(x_list_x, x_list_y, s=200)
        plt.scatter(self.x_list_x, self.x_list_y, s=100)
        plt.title("XPIX - DIS", fontsize=24); plt.xlabel("X",fontsize=14); plt.ylabel('DIS',fontsize=14)
        plt.show()
        self.fit.input_data(self.x_list_x, self.x_list_y)
        self.fit.solution()
        
        # plt.plot(xp_list_x ,xp_list_y, linewidth=5); plt.scatter(xp_list_x, xp_list_y, s=200)
        # plt.title("XP - DIS", fontsize=24); plt.xlabel("XP",fontsize=14); plt.ylabel('DIS',fontsize=14)
        # plt.show()
        
        # plt.plot(y_list_x ,y_list_y, linewidth=5); plt.scatter(y_list_x, y_list_y, s=200)
        # plt.title("YTrue - DIS", fontsize=24); plt.xlabel("Y",fontsize=14); plt.ylabel('DIS',fontsize=14)
        # plt.show()

        # plt.plot(yp_list_x ,yp_list_y, linewidth=5); plt.scatter(yp_list_x, yp_list_y, s=200)
        # plt.title("YP - DIS", fontsize=24); plt.xlabel("YP",fontsize=14); plt.ylabel('DIS',fontsize=14)
        # plt.show()
        
    #计算旋转矩阵和平移向量，使用行向量
    def calc_R_and_T(self):
        v_img = []  #相机坐标系里面的向量
        p_img = []  #相机坐标系中的点
        oringin = self.Rec.img_to_world(self.oringin[0], self.oringin[1])  #与自定义坐标系的原点相对应的相机坐标系点
        self.c_o = np.array([oringin[0][0], oringin[0][1], oringin[0][2] ])

        for i in range(0, len(self.left_v)):
            img_cd = self.Rec.img_to_world(self.left_v[i], self.right_v[i]) #计算相机坐标系中的点的坐标
            img_cd = [img_cd[0][0], img_cd[0][1], img_cd[0][2]] #丢弃齐次坐标，原本长度为4
            p_img.append(img_cd)
            # print(img_cd)
            img_cd = [img_cd[0]-oringin[0][0], img_cd[1]-oringin[0][1], img_cd[2]-oringin[0][2]] #相机坐标系里的三个向量
            v_img.append(img_cd)
            
        p_img = np.array(p_img)
        # print(p_img)
        v_img = np.array(v_img)
        # print(v_img)
        
        xytheta = np.dot(v_img[0],v_img[1])/math.sqrt( (np.dot(v_img[0],v_img[0])) * (np.dot(v_img[1],v_img[1])) )
        xytheta = math.acos(xytheta) * 180 / math.pi
        
        v_xycross = np.cross(v_img[1], v_img[0])
        ztheta = np.dot(v_img[2], v_xycross)/math.sqrt( (np.dot(v_img[2],v_img[2])) * (np.dot(v_xycross, v_xycross)) )
        ztheta = math.acos(ztheta) * 180 / math.pi
        # print(xytheta)
        # print(ztheta)
        # print(math.sqrt(np.dot(v_img[0],v_img[0])))
        # print(math.sqrt(np.dot(v_img[1],v_img[1])))
        # print(math.sqrt(np.dot(v_img[2],v_img[2])))
        
       
        self.Rotation = np.dot(np.linalg.inv(v_img), self.w_v) #用相机坐标系中三个向量组成的矩阵的逆与自定义坐标系对应矩阵相乘
        self.Trans = self.w_v[0] - np.dot(p_img[0], self.Rotation) #任取四个点中一点，计算平移向量
        
        self.RT[0:3, 0:3]=self.Rotation; self.RT[3, 0:3]=self.Trans #把旋转矩阵与平移向量合并成一个齐次矩阵，合并乘法与加法
        self.RT[0:3, 3] = 0
        
    def camera_to_world(self, w_c):
        if(len(w_c )== 0):
            print('error')
            return
        w_c = np.array(w_c[0, 0:3]) 
        myw = np.dot(w_c, self.Rotation) + self.Trans  #拆分计算的方法，结果与下面的一样
        # myw = np.dot(w_c, self.RT)  #从相机坐标系变化到自定义坐标系
        return myw
    
    def expr(self):
        a  = np.array([[1,2,3], [4, 5, 6], [7, 8, 9]])
        b = np.array([[1,1,1], [1,2,1], [2,3,2]])  
        print(b.dot(a))
        

class fitLine():
    
    def __init__(self):
        self.x = []
        self.y = []
        self.x_min = 0
        self.x_max = 800
        
    def z_corect(self, col, z):
        k = (z - 648.1919724248596) / 35436.52107081033 
        b = (z - 973.7276917970362) / -7.876657389425634
        # k = -0.0020482118890336872; b=50.903140145346505
        return (k * 320 + b) #true dis
        
        
    def input_data(self, x , y):
        self.x = np.array(x)
        self.y = np.array(y)
     
    #函数的标准形式
    def func(self, params, x):
        a, b = params
        return a + b * x
    
    # 误差函数，即拟合曲线所求的值与实际值的差
    def error(self, params, x, y):
        return self.func(params, x) - y
    
    # 对参数求解
    def slovePara(self):
        p0 = [10, 10]
        Para = leastsq(self.error, p0, args=(self.x, self.y))
        return Para
    
    # 输出最后的结果
    def solution(self):
        Para = self.slovePara()
        a, b = Para[0]
        print("a=",a," b=",b)
        print( "cost:" + str(Para[1]))
        print( "求解的曲线是:")
        print("y="+str(a)+" + "+str(b)+" * x")
    
        plt.figure(figsize=(16,12))
    
        # 画拟合线
        x = np.linspace(self.x_min ,self.x_max ,1000) 
        y = a + b* x ##函数式
    

        plt.scatter(self.x, self.y, color="green", label="sample data", linewidth=2)
        plt.plot(x,y,color="red",label="Depth line" ,linewidth=2)
        plt.xlabel("X", fontsize=18)
        plt.ylabel("DISP", rotation='horizontal', fontsize=18)
        plt.legend()
        plt.show()
        



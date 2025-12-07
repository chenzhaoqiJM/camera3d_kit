from cv2 import cv2
import numpy as np
import math

class boxDeal():
    def __init__(self):
        self.boxes = []
        self.labels = []
        self.boxes_dict = dict()
        self.left_img = 0
        self.right_img = 0
        self.init_flag = False
        self.img_w = 1280
        self.disparity_dict = dict()
        self.disparity_bm = dict()
    
    # label : ['pen', stents]
    #box:
    # [[class, confidence, left_up_x, left_up_y, right_down_x, right_down_y],....] list of list and only single
    #仅允许一个类别的一个物体出现，若有多个物体应提前划分
    
    #输入标签和物体框列表
    def update_box(self,labels,box):
        self.boxes = box; self.labels = labels
        self.init_flag = True
        
    def get_results(self):
        pass
    
    #将列表形式的框转换成对应标签的字典，便于管理
    def find_respone(self):
        self.boxes_dict = dict()
        for i in range(0, len(self.labels)):
            self.boxes_dict[self.labels[i]] = []
        
        for i in range(0, len(self.boxes)):
            self.boxes_dict[self.labels[ self.boxes[i][0] ] ].append(self.boxes[i])
            
        # print(self.boxes_dict)
        self.disparity_dict = dict()
        for i in range(0, len(self.labels)):
            if len(self.boxes_dict[self.labels[i] ]) == 2:
                temp = self.boxes_dict[self.labels[i] ]
                left = []; right = []
                if temp[0][2] > self.img_w//2 :
                    right = temp[0]; left = temp[1]
                else:
                    right = temp[1]; left = temp[0]
                    
                disp = dict()
                disp['class'] = left[0]; disp['confidence'] = [left[1], right[1]]; 
                
                left_up_dis = left[2] + self.img_w//2 - right[2]
                disp['left_up'] = [left[2], left[3], left_up_dis]
                
                right_down_dis = left[4] + self.img_w//2 - right[4]
                disp['right_down'] = [left[4], left[5], right_down_dis ]
                
                c_x_l = (left[2]+left[4])/2.0; c_y_l = (left[3]+left[5])/2.0
                c_x_r = (right[2]+right[4])/2.0; c_y_r = (right[3]+right[5])/2.0
                disp['center'] =   [c_x_l, c_y_l, c_x_l + self.img_w//2 - c_x_r ]
                self.disparity_dict[self.labels[i] ] = disp
            else:
                self.disparity_dict[self.labels[i] ] = dict()
                
        # print(self.disparity_dict)
            
            
    
    def find_dis_by_canny(self):
        if self.init_flag == False:
            return False
    
    #沿垂直线查找视差  
    def get_dis_list(self,rec, pl, th, length=20):
        left = []; right = []
        if abs(th) > 0.1:
            k = math.tan(th + math.pi/2)
        else:
            k = 10000
        x0 = pl[0]; y0 = pl[1]
        for i in range(0, length):
            t = i
            x_positive = x0 + t * math.cos(th + math.pi/2.0); x_negative = x0 - t * math.cos(th + math.pi/2.0)
            y_positive = y0 + t * math.sin(th + math.pi/2.0); y_negative = y0 - t * math.sin(th + math.pi/2.0)
            if x_positive >= self.img_w//2 or x_negative < 0 or y_positive >= 480 or y_negative<0:
                break
            if x_positive < 0 or x_negative >=self.img_w//2 or y_positive < 0 or y_negative>=480:
                break
            dis_positive = rec.get_bm_dis([x_positive, y_positive])
            dis_negative = rec.get_bm_dis([x_negative, y_negative])
            left.append(dis_positive)
            right.append(dis_negative)
        result = [left, right]
        return result
    
    #从垂直线视差列表中查找最佳视差
    def find_best_dis(self, dis_list):
        if len(dis_list)<2:
            return False
        new_list = []
        for i in range(0, len(dis_list[0])):
            new_list.append( (dis_list[0][i]+dis_list[1][i])/ 2.0 )
        dis = -1.0
        
        for i in range(0, len(new_list)):
            if new_list[i]>0 and new_list[i]>=dis_list[0][i]/2.0 and new_list[i] >= dis_list[1][i]/2.0:
                dis = new_list[i]
                return dis
        return dis
            
    #查找某个点附件的最佳视差，用于左上角、右下角、中点视差的确定。
    def search_dis(self, rec):
        self.disparity_bm = dict()
        for i in range(0, len(self.labels)):
            self.disparity_bm[self.labels[i] ] = dict()
            if len(self.boxes_dict[self.labels[i] ]) > 0:
                left = []; flag = False
                if len(self.boxes_dict[self.labels[i] ]) == 2:
                    temp = self.boxes_dict[self.labels[i]]
                    flag = True
                    if temp[0][2] < self.img_w//2:
                        left = temp[0]
                    else:
                        left = temp[1]
                if len(self.boxes_dict[self.labels[i] ]) == 1:
                    left = []; temp = self.boxes_dict[self.labels[i]]
                    if temp[0][2] < self.img_w//2:
                        # print('1111111111111')
                        left = temp[0]
                        flag = True
                    else:
                        pass
                if flag == True:
                    x1 = left[2]; y1 = left[3]; x2 = left[4]; y2 = left[5]
                    xc = (x1 + x2)/2.0; yc = (y1 + y2)/2.0
                    theta = math.atan2(y2-y1, x2-x1)
                    x1 = (x1+xc)/2.0; y1 = (y1+yc)/2.0
                    x2 = (x2+xc)/2.0; y2 = (y2+yc)/2.0
                    
                    left_bottom = self.get_dis_list(rec, [x1,y1], theta, 4)
                    right_bottom = self.get_dis_list(rec, [x2, y2], theta, 4)
                    center = self.get_dis_list(rec, [xc, yc], theta, 2)
                    
                    # print(self.find_best_dis(center))
                    left_bottom_dis = self.find_best_dis(left_bottom)
                    right_bottom_dis = self.find_best_dis( right_bottom)
                    center_dis = self.find_best_dis(center)
                    
                    self.disparity_bm[self.labels[i] ]['left_up'] = [x1, y1, left_bottom_dis]
                    self.disparity_bm[self.labels[i] ]['right_down'] = [x2, y2, right_bottom_dis]
                    self.disparity_bm[self.labels[i] ]['center'] = [xc, yc, center_dis]
                    
                    
            else:
                pass
        
        
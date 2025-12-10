import sys
import os
import shutil
code_dir = os.path.dirname(os.path.realpath(__file__))

_module_path = os.path.abspath(os.path.join(code_dir, "../../"))
print(_module_path)
sys.path.append(_module_path)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibration_libs.ccalibration import CCameraCalibration

# 导入库
from calibration_libs.utils import get_chess_corners, get_chess_corners_world, img_to_world_by_dis, world_to_img_l, vis_disparity

def clear_folder(folder):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

images_path =  os.path.abspath(os.path.join(code_dir, './images_640x480'))


# 标定板参数
chessSize=[11, 8] # col, row, 长边设置为col, 短边row , col作为x轴， row作为y轴
chessCellLen=20 # 单位 mm

#相机标定类
mycc = CCameraCalibration()  #初始化
mycc.set_calibration_images(images_path, 15, 480, 1280, size=chessSize, length=chessCellLen) #设置标定图像的路径和图片数量

mycc.stereo_calibration(True, reversal=False) #进行双目立体标定，得到内参矩阵、畸变系数、重映射矩阵map、重投影矩阵Q等
mycc.print_p()

# 立体校正
img_src = cv2.imread(os.path.join(code_dir, './images_640x480/WIN_20251207_14_54_12_Pro.jpg'))
print("图像形状:", img_src.shape)
_height, _width, _c = img_src.shape
left_src = img_src[:, 0: _width//2] #拆分左右图像
right_src = img_src[:, _width//2:]

#立体校正
left_remap = cv2.remap(left_src, mycc._leftParameters["map1"], mycc._leftParameters["map2"], cv2.INTER_LINEAR)
right_remap = cv2.remap(right_src, mycc._rightParameters["map1"], mycc._rightParameters["map2"], cv2.INTER_LINEAR)
img_remap =  cv2.hconcat([left_remap, right_remap]) #合并校正后图像

# 棋盘格点检测
_ret, left_src_corners, _left_src_corners_draw = get_chess_corners(left_src, chessSize, draw_flag=True)
_ret, right_src_corners, _right_src_corners_draw = get_chess_corners(right_src, chessSize, draw_flag=True)
_src_draw_concat = cv2.hconcat([_left_src_corners_draw, _right_src_corners_draw]) #合并绘图

# remap 图
_ret, left_remap_corners, _left_remap_corners_draw = get_chess_corners(left_remap, chessSize, draw_flag=True)
_ret, right_remap_corners, _right_remap_corners_draw = get_chess_corners(right_remap, chessSize, draw_flag=True)
_remap_draw_concat = cv2.hconcat([_left_remap_corners_draw, _right_remap_corners_draw]) #合并绘图


# 极线验证
print("\n极线校正验证------------------------------------------------------------")
## 原图
_draw_src = img_src.copy()
_width = _draw_src.shape[1]
_points_src_left = left_src_corners[:chessSize[0],:,:] # shape: (col 1 2)
_points_src_right = right_src_corners[:chessSize[0],:,:] # shape: (col 1 2)

_src_error = 0.0
for _point_left, _point_right in zip(_points_src_left, _points_src_right):
    _pt1 = (int(_point_left[0][0]), int(_point_left[0][1]))
    _pt2 = (_width-1, int(_point_left[0][1]))
    _src_error += abs(_point_left[0][1] - _point_right[0][1])
    cv2.line(_draw_src, _pt1, _pt2, (0,0,255), thickness=1)
print("原图在y坐标上的总误差(绝对值之和): ", _src_error)

## remap后
_draw_remap = img_remap.copy()
_width = img_remap.shape[1]
_points_remap_left = left_remap_corners[:chessSize[0],:,:] # shape: (col 1 2)
_points_remap_right = right_remap_corners[:chessSize[0],:,:] # shape: (col 1 2)
_remap_error = 0.0
for _point_left, _point_right in zip(_points_remap_left, _points_remap_right):
    _pt1 = (int(_point_left[0][0]), int(_point_left[0][1]))
    _pt2 = (_width-1, int(_point_left[0][1]))
    _remap_error += abs(_point_left[0][1] - _point_right[0][1])
    cv2.line(_draw_remap, _pt1, _pt2, (0,0,255), thickness=1)
print("Remap校正图在y坐标上的总误差(绝对值之和): ", _remap_error)


# 从像素坐标到实际坐标的检验
# 7 个格子
print("\n从像素坐标到实际坐标的检验------------------------------------------------")
_cell_nums = 7
pt1_left = left_remap_corners[0][0]
pt1_right = right_remap_corners[0][0]
_world_pt1 = img_to_world_by_dis(pt1_left, pt1_left[0]-pt1_right[0], mycc._stereoCommParameters["Q"])
print("真实坐标点1:", _world_pt1)

pt2_left = left_remap_corners[_cell_nums][0]
pt2_right = right_remap_corners[_cell_nums][0]
_world_pt2 = img_to_world_by_dis(pt2_left, pt2_left[0]-pt2_right[0], mycc._stereoCommParameters["Q"])
print("真实坐标点2:", _world_pt2)

print(f"{_cell_nums}个格子的实际距离: {_cell_nums*chessCellLen} mm")
print(f"使用矩阵Q计算得到的距离: {np.linalg.norm(_world_pt2-_world_pt1)} mm")


# SGBM算法初始化
left_matcher  = cv2.StereoSGBM_create(
    minDisparity=0, # 最小视差值。通常为 0，但如果图像经过校正后可能会有偏移，则可能需要调整
    numDisparities=16 * 7, # 最大视差减去最小视差。该值必须大于 0，并且在当前实现中，必须能被 16 整除。设置的视差范围越大，可以检测到的深度范围也越大，但计算量也会增加。
    blockSize=3, # 匹配块的大小 (Matched block size)。它必须是一个 大于等于 1 的奇数。较小保留更多细节，但更容易受噪声影响；较大结果更平滑，但可能丢失细节
    P1=8 * 3 * 3, # 第一个控制视差平滑度的参数。用于计算相邻像素之间视差变化为 1 时的惩罚值
    P2=32 * 3 * 3, # 第二个控制视差平滑度的参数。用于计算相邻像素之间视差变化大于 1 时的惩罚值, 通常，P2 > P1。值越大，得到的视差图越平滑。
    disp12MaxDiff=12, # 左右视差检查的最大容许差值
    uniquenessRatio=10, 
    speckleWindowSize=50, # 用于斑点过滤 (Speckle filtering) 的最大平滑视差区域尺寸
    speckleRange=32, # 斑点过滤中最大允许的视差变化范围, 隐式乘以 16
    preFilterCap=63  # 该值越大，对纹理较少区域的响应越强。
    ) 

# WLS滤波器
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher )


grayl = cv2.cvtColor(left_remap, cv2.COLOR_BGR2GRAY)
grayr = cv2.cvtColor(right_remap, cv2.COLOR_BGR2GRAY)

disparity_left  = left_matcher.compute(grayl, grayr)
disparity_right = right_matcher.compute(grayr, grayl)

_dis1 = disparity_left[int(pt1_left[1])][int(pt1_left[0])] /16.0 
_dis1_real = pt1_left[0] - pt1_right[0]
print(f"pt1的stereoSGBM求解视差:{_dis1}, 真实视差:{_dis1_real}")

_dis2 = disparity_left[int(pt2_left[1])][int(pt2_left[0])] /16.0 
_dis2_real = pt2_left[0] - pt2_right[0]
print(f"pt2的stereoSGBM求解视差:{_dis2}, 真实视差:{_dis2_real}")

plt.imshow(disparity_left / 16, cmap='jet')
plt.colorbar()
plt.show()


# 创建 WLS 滤波器
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)        # 越大越平滑，典型值 800~20000
wls_filter.setSigmaColor(1.5)     # 颜色敏感度 0.8~2 之间

# 执行滤波
# 注意：disparity_filtered 是 float32，单位仍然是 "乘了 16" 的 fixed-point。
disparity_filtered = wls_filter.filter(
    disparity_left, 
    grayl, 
    None,
    disparity_right
)

_dis1 = disparity_filtered[int(pt1_left[1])][int(pt1_left[0])] /16.0 
_dis1_real = pt1_left[0] - pt1_right[0]
print(f"pt1的stereoSGBM求解视差:{_dis1}, 真实视差:{_dis1_real}")

_dis2 = disparity_filtered[int(pt2_left[1])][int(pt2_left[0])] /16.0 
_dis2_real = pt2_left[0] - pt2_right[0]
print(f"pt2的stereoSGBM求解视差:{_dis2}, 真实视差:{_dis2_real}")

plt.imshow(disparity_filtered / 16, cmap='jet')
plt.colorbar()
plt.show()



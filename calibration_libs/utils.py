    
import cv2
import numpy as np

#检测棋盘格角点并返回
def get_chess_corners(img, chess_size, draw_flag=True):
    """检测棋盘格角点并返回
    >>> 输入为左右图像、棋盘格大小、是否绘制检测结果默认为True
    >>> 正常返回值：是否检测到的标志Bool类型; 棋盘角点: shape:(col*row, 1, 2), col约定为标定板长边
    """
    draw_img = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img, chess_size, None)

    if(ret):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        if draw_flag == True: 
            cv2.drawChessboardCorners(draw_img, chess_size, corners, ret)

    return ret ,corners, draw_img

def get_chess_corners_world(chessSize=[11, 8], chessCellLen=20):
    """计算标定板的角点真实坐标
    >>> 输入 chessSize, 列表，[标定板长边角点数(col)(x), 标定板短边角点数(row)(y)]
    >>> 输入 chessCellLen, 角点之间距离，单位 mm
    >>> 输出 numpy, shape: (row*col, 3)
    """
    #棋盘格上角点的真实坐标
    objp = np.zeros((chessSize[0]*chessSize[1],3), np.float32)
    for i in range(0, chessSize[1]):#row
        for j in range(0, chessSize[0]): #col
            point = [j*chessCellLen, i*chessCellLen, 0.]
            objp[i*chessSize[0]+j][0] = point[0]
            objp[i*chessSize[0]+j][1] = point[1]
            objp[i*chessSize[0]+j][2] = point[2]
    apoints = objp

    return apoints


#利用左图上对应的一点以及立体匹配得到的视差计算三维坐标
def img_to_world_by_dis(pl, dis, Q):
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
    w = np.dot(img_p ,Q.T)
    w = w/w[0][3]
    return w

#输入三维坐标计算其在左图中的对应点，输入为列表
def world_to_img_l(pw, Q):
    """利用三维坐标计算其在左图中的对应点，
    >>> 输入为:[x, y, z, 1]
    >>> 以行向量形式计算
    >>> 返回值：二维齐次坐标；[[x, y, 视差, 1]]
    """
    if(len(pw)!=4):
        print("输入格式错误")
        return []
   
    w_p = np.array(pw)
    w_p.resize(1,4)
    img_p = np.dot(w_p, np.linalg.inv(Q.T))
    img_p = img_p/img_p[0][3]
    return img_p

def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)
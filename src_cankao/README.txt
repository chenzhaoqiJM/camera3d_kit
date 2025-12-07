#运行环境
ubuntu18.04+ROS
需要安装PaddlePaddle机器学习框架

#程序说明
由多个模块构成：立体标定、三维重建、目标检测、机械臂抓取与放置、其它几何变换

立体标定类：
ccalibration.py

三维重构类：
creconstruction.py

机械臂抓取与放置：
cmoveit、cobjectfollow

启动文件：
img_pub_and_sub.py 捕获图像、立体矫正、三维重建、订阅物体框、画框
infer_ros.py 加载预训练模型、发布检测框

moveit_ik.py 规划并执行路径
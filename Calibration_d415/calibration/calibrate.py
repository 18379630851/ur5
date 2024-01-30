import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import cv2.aruco as aruco
from scipy import optimize
import os
import sys
sys.path.append(os.getcwd())
from equipment.Robot import Robot
from equipment.Camera import Camera
from equipment.Gripper import Gripper 

from utils.compute import get_rigid_transform,plane_norm
  
# mtx = np.array([[613.89953613,   0,         311.69946289],[  0,         613.8996582,  237.63970947],[  0,           0,           1        ]])
# dist = np.array(([[0.028784404196194664, 0.7481844381754564, 0.0028707604214314336, -0.0032153215725527914, -3.1796489988923713]]))
mtx = np.array([[909.65582275,   0,         652.90588379],
 [  0,         907.89691162, 365.95373535],
 [  0,           0,           1.        ]])
dist = np.array(([[0.0000001, 0, 0, 0, 0]]))

# --------------- Setup options ---------------
# 求desk平面的法向量
point_a = np.array([0.00230, -0.8042, 0.005])
point_b = np.array([-0.28090, -0.80415, 0.00659])
point_c = np.array([0.01026, -0.34282, -0.00931])
vector_norm = plane_norm(point_a,point_b,point_c) #plane_norm在compute里定义，为根据空间中3个点的坐标，获取平面的法向量

#感觉y范围可以是-0.661, -0.136
calibspace_limits = np.asarray([[-0.25, -0.05], [-0.84, -0.4], [0.00, 0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
calib_grid_step = 0.1
# left-up point??
checkerboard_offset_from_tool = [-0.035,0,0.005]                     #参考坐标系是robotbase，这里的第三个数调整夹具末端到方块底的高度
# checkerboard_offset_from_tool = [-0.015, 0.02, 0.005]
tool_orientation = [2.221,2.221,0]                                 
# ---------------------------------------------

# Construct 3D calibration grid across workspace
#np.linspace：生成一个指定大小，指定数据区间的均匀分布序列
# （1）start：序列中数据的下界。
# （2）end：序列中数据的上界。
# （3）num：生成序列包含num个元素；其值默认为50。
# np.round：取整
# 第一句为什么不是gridspace_x = np.linspace(calibspace_limits[0][0], calibspace_limits[0][1],int(np.round(1 + (calibspace_limits[0][1] - calibspace_limits[0][0])/calib_grid_step))) 
# gridspace_x = np.linspace(calibspace_limits[0][0], calibspace_limits[0][1],int(np.round(1 + (calibspace_limits[1][1] - calibspace_limits[1][0])/calib_grid_step)))    #9
gridspace_x = np.linspace(calibspace_limits[0][0], calibspace_limits[0][1],int(np.round(1 + (calibspace_limits[0][1] - calibspace_limits[0][0])/calib_grid_step))) 
gridspace_y = np.linspace(calibspace_limits[1][0], calibspace_limits[1][1],int(np.round(1 + (calibspace_limits[1][1] - calibspace_limits[1][0])/calib_grid_step)))    #9
gridspace_z = np.linspace(calibspace_limits[2][0], calibspace_limits[2][1],int(np.round(1 + (calibspace_limits[2][1] - calibspace_limits[2][0])/calib_grid_step)))    #3
# meshgrid是生成点阵坐标，xyz呈一一对应的状态，此处相当于根据坐标范围和步长表示出了整个标定空间里的标定点坐标
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]      #9*9*3
# print(num_calib_grid_pts)

calib_grid_x.shape = (num_calib_grid_pts,1)
calib_grid_y.shape = (num_calib_grid_pts,1)
calib_grid_z.shape = (num_calib_grid_pts,1)
desk_height = point_a[2] + (vector_norm[0]*(point_a[0]-calib_grid_x)+vector_norm[1]*(point_a[1]-calib_grid_y))/vector_norm[2]
calib_grid_z = calib_grid_z + desk_height
calib_pts_world = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)         #对对应行的xyz分别进行拼接，world坐标系下具体的9*9*3个3D点


measured_pts = []                   #测量的点
observed_pts = []                   #观察到的点（像素到空间点的转换）
observed_pix = []                   #观察到的像素

# Move robot to home pose
print('Connecting to robot...')
robot_ur5 = Robot()

# Move robot to each calibration point in workspace
print('Collecting data...')

for calib_pt_idx in range(num_calib_grid_pts):              #num_calib_grid_pts
    print(calib_pt_idx)
    tool_position = calib_pts_world[calib_pt_idx,:]          #3D点为每一行
    
    robot_ur5.linear_move(tool_position,tool_orientation)    #6D位姿
    time.sleep(1)
    #num_calib_grid_pts
    
    # Find checkerboard center
    camera_sr300 = Camera()

    #这里内参可以直接等于mtx
    cam_intrinsics = camera_sr300.get_intrinsics()      #获取相机内参矩阵
    # cam_intrinsics = mtx
    depth_scale = camera_sr300.get_depth_scale()        #获取相机深度尺寸
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)   #迭代！？
    
    camera_color_img, camera_depth_img = camera_sr300.get_array()    
    camera_sr300.close_camera()
    #从camera中取出的原始的彩色图像和单位为mile的深度图像
    cv2.imshow('color',camera_color_img)
    #cv2.waitKey(0)
    camera_depth_img = camera_depth_img.astype(float) * depth_scale 

    h1, w1 = camera_color_img.shape[:2]     #获取彩色图片的高、宽，并且赋值给h1和w1
    # print(h1, w1)

    # 纠正畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(camera_color_img, mtx, dist, None, newcameramtx)

    frame=dst1

    #灰度化，检测aruco标签，
    gray_data = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)            #转换为灰度图
    cv2.imshow('gray', gray_data)
    
    # 开始检测aruco码
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict, parameters=parameters) #左上、右上、右下、左下
    print("========================================")
    cv2.waitKey(5)
    if len(corners)==0: #没有检测到角点
        continue 
    else:
        # print(ids)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist)  #角点坐标，aruco码尺寸单位m，内参矩阵，畸变参数；返回：旋转向量，平移向量
        # print(rvec)
        # print(tvec)

        corners1 = np.reshape(corners[0], (4,2))    #把矩阵corners[0]变成4行2列

        #码的中点坐标（相机or像素！！！！）应该是像素吧
        x_pic = (corners1[1][0]+corners1[3][0])/2   
        y_pic = (corners1[0][1]+corners1[2][1])/2
        checkerboard_pix = np.array([x_pic, y_pic])                                               #找得到内点，就进一步精细化寻找
        # print(checkerboard_pix)

        checkerboard_pix=checkerboard_pix.astype(int)     
        checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]         #证明了camera_depth_img应该是与camera_color_img是匹配对应的关系（如果不是很匹配呢）
        checkerboard_x = np.multiply(checkerboard_pix[0]-cam_intrinsics[0][2],checkerboard_z/cam_intrinsics[0][0])      #从2D像素转换为camera坐标系下的3D
        checkerboard_y = np.multiply(checkerboard_pix[1]-cam_intrinsics[1][2],checkerboard_z/cam_intrinsics[1][1])

        tool_position = tool_position + checkerboard_offset_from_tool                       #实际世界坐标系真实测量到的3D点，夹具末端坐标
        observed_pts.append([checkerboard_x,checkerboard_y,checkerboard_z])
        measured_pts.append(tool_position)  #measured_pts是工件坐标
        observed_pix.append(checkerboard_pix)   #observed_pix是相机坐标系
        # print(camera_depth_img)
        # print(measured_pts)
        # print(observed_pix)
        # print(checkerboard_pix)
        # print(camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]])
        # print(checkerboard_z)
        print(observed_pts)

        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(camera_color_img, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.04)
            aruco.drawDetectedMarkers(camera_color_img, corners)
        cv2.imshow("frame", camera_color_img)
        cv2.waitKey(5)

    # print(observed_pts)
    
    
    

# Move robot back to home pose
robot_ur5.joint_move(robot_ur5.home_pos)                                            #收集点完毕，机械臂返回home位置

measured_pts = np.asarray(measured_pts)                     #list转变为array:[[1, 2, 3], [4, 5, 6]]>>[[1 2 3];  一维len变为二维nx3
observed_pts = np.asarray(observed_pts)                                                             #[4 5 6]]
observed_pix = np.asarray(observed_pix)

world2camera = np.eye(4)                                    #这个表示初始化摄像头坐标系与世界坐标系之间的坐标系转换关系

def get_rigid_transform_error(z_scale):
    global measured_pts, observed_pts, observed_pix, world2camera,cam_intrinsics
    # print(observed_pts)
    # Apply z offset and compute new observed points using camera intrinsics(使用z的偏移量并使用相机内参计算新的观察点)
    observed_z = observed_pts[:,2:] * z_scale               #取第三列的元素checkerboard_z
    observed_x = np.multiply(observed_pix[:,[0]]-cam_intrinsics[0][2],observed_z/cam_intrinsics[0][0])
    observed_y = np.multiply(observed_pix[:,[1]]-cam_intrinsics[1][2],observed_z/cam_intrinsics[1][1])
    new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

    # Estimate rigid transform between measured points and new observed points
    R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))  #以两个nx3的二维矩阵为输入，每一行表示camera坐标系下观察点的3D坐标系
    t.shape = (3,1)
    world2camera = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)    #矩阵合并(world到camera的转换)

    # Compute rigid transform error     R.shape=[3x3],measured_pts.shape=[n,3];重复数组t(1,n)次来构建新的数组，行数不变，列数xn，为了与前面shape保持一致;得到的结果是measured_pts在camera坐标系下的表示
    registered_pts = np.dot(R,np.transpose(measured_pts)) + np.tile(t,(1,measured_pts.shape[0]))
    error = np.transpose(registered_pts) - new_observed_pts         #[3,n]>>[n,3]
    error = np.sum(np.multiply(error,error))                #对应位置每个元素平方然后相加
    rmse = np.sqrt(error/measured_pts.shape[0]);            #每个点的平均平方误差，再开方
    return rmse                         #optim_result.fun

# Optimize z scale w.r.t. rigid transform error
print('Calibrating...')
z_scale_init = 1                                            #初始化z的尺度为1
# 参数：要最小化的目标参数、最初的猜测（大小为（n，）的实数元素数组）、解算器类型为Nelder-Mead的Simplex algorithm
#返回的是优化结果，重要的属性：x(解的数组)、success(一个布尔标志，表示优化器是否成功退出)、message(描述终止的原因)
optim_result = optimize.minimize(get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
camera_depth_offset = optim_result.x

# Save camera optimized offset and camera pose
print('Saving...')
np.savetxt('real/camera_depth_calib.txt', camera_depth_offset, delimiter=' ')       #优化的z比例因子，该比例因子应与从相机捕获的每个深度像素相乘。这一步与RealSense SR300相机更为相关，后者通常面临严重的缩放问题，其中3D数据通常比真实世界坐标小15-20%。
get_rigid_transform_error(camera_depth_offset)              #最小化z_scale的优化结果
camera2world = np.linalg.inv(world2camera)       #目的是真实的进行计算，得到  world2camera                #矩阵求逆（结果应该是camera坐标系转换为world坐标系的转换矩阵，相当于camera在world下的位姿表示）
np.savetxt('real/camera2world.txt', camera2world, delimiter=' ')
print('Done.')

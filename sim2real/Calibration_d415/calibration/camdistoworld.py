import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import cv2.aruco as aruco
from scipy import optimize
import math as m
import os
import sys
sys.path.append(os.getcwd())
from equipment.Camera import Camera
from utils.compute import get_rigid_transform,plane_norm
from scipy.spatial.transform import Rotation as R
  
# mtx = np.array([[613.89953613,   0,         311.69946289],[  0,         613.8996582,  237.63970947],[  0,           0,           1        ]])
# dist = np.array(([[0.028784404196194664, 0.7481844381754564, 0.0028707604214314336, -0.0032153215725527914, -3.1796489988923713]]))
mtx = np.array([[909.65582275,   0,         652.90588379],
 [  0,         907.89691162, 365.95373535],
 [  0,           0,           1.        ]])
dist = np.array(([[0.0000001, 0.00000001, 0.0, 0, 0]]))
M = None

point_a = np.array([0.00230, -0.8042, 0.005]) # -0.391, -0.661, 0             -16.07 -844.17 11.43
point_b = np.array([-0.28090, -0.80415, 0.00659]) # -0.391, -0.136, -0.013       -338.90 -844.19 13.56
point_c = np.array([0.01026, -0.34282, -0.00931]) # 
obj_x = []
obj_y = []
obj_theta = []

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            observed_pts = []                   #观察到的点（像素到空间点的转换）
            observed_pix = []                   #观察到的像素
            robot_pts = []                      #aruco to robot

            #这里内参可以直接等于mtx
            cam_intrinsics = mtx      #获取相机内参矩阵
                # cam_intrinsics = mtx
            depth_scale = 0.00012498664727900177        #获取相机深度尺寸
            checkerboard_size = (3,3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)   #迭代！？
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_depth_img = camera_depth_img.astype(float) * depth_scale 
            camera_color_img = np.asanyarray(color_frame.get_data())
            cv2.imshow('color',camera_color_img)

            h1, w1 = camera_color_img.shape[:2]     #获取彩色图片的高、宽，并且赋值给h1和w1
                # print(h1, w1)

                # 纠正畸变
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
            dst1 = cv2.undistort(camera_color_img, mtx, dist, None, newcameramtx)

            frame=dst1

            #灰度化，检测aruco标签，
            gray_data = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)            #转换为灰度图
            cv2.imshow('gray', gray_data)
                #cv2.waitKey(1)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 开始检测aruco码
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict, parameters=parameters) #左上、右上、右下、左下

            if ids is not None: #检测到角点
                print('1')
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist)  #角点坐标，aruco码尺寸单位m，内参矩阵，畸变参数；返回：旋转向量，平移向量
                rot_mat, _ = cv2.Rodrigues(rvec)
                tvec1 = np.reshape(tvec,(3,1))
                M = np.append(rot_mat, tvec1, axis=1)
                E = np.asarray([[0,0,0,1]])
                M = np.append(M,E,axis=0)
                
                for i in range(rvec.shape[0]):
                    cv2.drawFrameAxes(camera_color_img, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.04)
                    aruco.drawDetectedMarkers(camera_color_img, corners)
                cv2.imshow("frame", camera_color_img)
                cv2.waitKey(5)

            camera2world = np.asarray([
                [-4.576686961397543121e-03, -8.225492109762101434e-01, 5.686755221203669830e-01, -5.892528463357350477e-01],
[-9.989653598685415137e-01, -2.197049475730761953e-02, -3.981842717679989763e-02, -5.400838946708869104e-01],
[4.524669843394511087e-02, -5.682693840798866969e-01, -8.215976164755434130e-01, 5.842051462048796218e-01],
[0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
            ])



            robot_pts = np.dot(camera2world,M)
            # aruco_pts = robot_pts+[[0,0,0,0.015],[0,0,0,0],[0,0,0,-0.005],[0,0,0,0]]
            # print(np.shape(observed_pts))
            # r1 = robot_pts[0:3,0:3]
            # r = R.from_matrix(r1)
            # robot_rotate = np.asarray(r.as_euler('zyx'))
            # obj_theta = robot_rotate[0]

            # obj_x = robot_pts[0][3]-((1.414*0.04*m.sin(obj_theta-45*180/m.pi))/2)
            # obj_y = robot_pts[1][3]-((1.414*0.035*m.cos(obj_theta-45*180/m.pi))/2)
            obj_x = robot_pts[0][3]
            obj_y = robot_pts[1][3]
            r1 = robot_pts[0:3,0:3]
            r = R.from_matrix(r1)
            robot_rotate = np.asarray(r.as_euler('zyx')*180/3.1415926)
            obj_theta = robot_rotate[0]


            obj = [obj_x+0.035,obj_y, robot_pts[2][3]-0.005, robot_rotate]
            # obj = [obj_x+0.035,obj_y, calib_grid_z, robot_rotate]
            print(obj)
            # print(observed_pts)

    finally:
        # Stop streaming
        pipeline.stop()


            


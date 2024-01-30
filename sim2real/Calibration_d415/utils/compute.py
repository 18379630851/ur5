import numpy as np
import math 
import cv2 

import os 
import sys
sys.path.append(os.getcwd())

depth_scale = np.loadtxt('real/depth_scale.txt',delimiter=' ')
camera_depth_calib = np.loadtxt('real/camera_depth_calib.txt',delimiter=' ')
camera_intrinsics = np.loadtxt('real/camera_intrinsics.txt',delimiter=' ')
cam_world = np.loadtxt('real/camera2world.txt', delimiter=' ')

# 1.转移矩阵的估计（求的是A相对于B坐标系的位姿（A在B下的表示））
def get_rigid_transform(A, B):                              #A.shape=B.shape=[n,3]
    assert len(A) == len(B)
    N = A.shape[0]; # Total points
    centroid_A = np.mean(A, axis=0)                         #压缩行，对各列求均值，返回 1* 3 矩阵
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))                    #每个点减去均值
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB)                        #[3,n]x[n,3]
    #对H进行奇异值分解，返回U, S, Vt;U大小为(M,M)，S大小为(M,N)，Vt大小为(N,N);H=UxSxVt
    #其中s是对矩阵a的奇异值分解。s除了对角元素不为0，其他元素都为0，并且对角元素从大到小排列。s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值
    U, S, Vt = np.linalg.svd(H)                             
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0: # Special reflection case（如果R的行列式<0）
       Vt[2,:] *= -1            #第三行乘以-1
       R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t

# 2.得到相机坐标系下的坐标
def get_cam_pts(depth_img, camera_intrinsics):
    # Get depth image size
    im_h = depth_img.shape[0]               #480
    im_w = depth_img.shape[1]               #640

    # Project depth into 3D point cloud in camera coordinates(相机坐标系中将深度图投影到点云中)
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))           #得到的是像平面大小的位置
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)
    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)         #im_h*im_w x 3
    
    return cam_pts

# 3.rgb与相机坐标系下的数组结构匹配
def get_rgb_pts(color_img):
    # get color image part size
    im_h = color_img.shape[0]               #480
    im_w = color_img.shape[1]               #640

    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)         #im_h*im_w x 3

    return rgb_pts                                                              #二维数组[im_h*im_w,3];分别表示camera坐标系下的每个pixel的位置与颜色（对应行对应）从左到右从上到下的顺序 

# 4.获取点云数据基于camera坐标系（为获取高度图做准备;参数从get_heightmap函数中传入;）
def get_pointcloud(color_img, depth_img, camera_intrinsics):

    cam_pts = get_cam_pts(depth_img, camera_intrinsics)
    rgb_pts = get_rgb_pts(color_img)

    return cam_pts, rgb_pts                                                        #二维数组[im_h*im_w,3];分别表示camera坐标系下的每个pixel的位置与颜色（对应行对应）从左到右从上到下的顺序

# 5.从4x4坐标变换矩阵中获取位置信息
def getPosition(rotMat):
    position = rotMat[0:3,3]
    return position

# 6.从4x4坐标变换矩阵中获取旋转向量
def getRotVec(Mat):
    rotMat=Mat[0:3,0:3]
    rotVec=cv2.Rodrigues(rotMat)[0]
    rotVec_f=rotVec.flatten()
    return rotVec_f

# 7.从4x4坐标变换矩阵中获取1x6的位置和姿态信息
def get_position_and_pose(Mat):
    position = getPosition(Mat)
    rotvect = getRotVec(Mat)
    position_pose = np.hstack((position,rotvect))
    return position_pose

# 8.从欧拉角计算旋转矩阵(逆正顺负)
#从右向左运动是相对于固定参考系（左乘）、从左向右运动是相对于运动坐标系（右乘）
def euler2rotm(theta):                      #theta必须是弧度表示
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


# 9.BGR转化为hsv图像，再归一化处理(3维变2维)
def bgr2hsv(color_img):
    color_img_hsv = cv2.cvtColor(color_img,cv2.COLOR_BGR2HSV)
    color_img_h = color_img_hsv[:,:,0]/180
    color_img_s = color_img_hsv[:,:,1]/255
    color_img_v = color_img_hsv[:,:,2]/255
    color_img_h.shape = (color_img_hsv.shape[0]*color_img_hsv.shape[1],1)
    color_img_s.shape = (color_img_hsv.shape[0]*color_img_hsv.shape[1],1)
    color_img_v.shape = (color_img_hsv.shape[0]*color_img_hsv.shape[1],1)
    color_pts_hsv =  np.concatenate((color_img_h, color_img_s, color_img_v), axis=1)
    return color_pts_hsv

# 10.根据空间中3个点的坐标，获取平面的法向量
def plane_norm(point_a,point_b,point_c):
    vector_ab = point_b-point_a
    vector_ac = point_c-point_a
    plane_norm = np.cross(vector_ab,vector_ac)
    return plane_norm

# 11.根据空场景图片和实时采集的场景图片，获取物体标记（去除深度为0的边缘标记）;
# (要考虑reference和输入是否需要cvtcolor?)
def get_object_index(rgb_input_img,dep_input_img):
    depth_refer_path = 'Hitpgd/samples/background/raw/background_depth_m.png'
    # depth_refer_path = 'experiment/single/background_depth.png'
    depth_refer = cv2.imread(depth_refer_path,cv2.IMREAD_ANYDEPTH)                  #获取背景深度图
    depth_refer_real = depth_refer.astype(float) * depth_scale * camera_depth_calib
    color_refer_path = depth_refer_path.replace('depth','RGB')
    color_refer = cv2.imread(color_refer_path)                                      #获取背景RGB图
    color_refer_hsv = bgr2hsv(color_refer)                                          #RGB转HSV
    camera_pts_refer = get_cam_pts(depth_refer_real, camera_intrinsics)             #获取背景图相机坐标系下的坐标
    world_pts_refer = np.transpose(np.dot(cam_world[0:3,0:3],np.transpose(camera_pts_refer)) + np.tile(cam_world[0:3,3:],(1,camera_pts_refer.shape[0])))
    #获取背景图世界坐标系下的坐标（坐标系转换）
    distance_pts = dep_input_img.flatten() #输入深度图降为一维
    color_img_hsv = bgr2hsv(rgb_input_img) #RGB转HSV
    # cv2.imwrite('experiment/single/hsv_46.png',color_img_hsv)
    camera_pts = get_cam_pts(dep_input_img, camera_intrinsics)  #获取输入图相机坐标系下的坐标
    world_pts = np.transpose(np.dot(cam_world[0:3,0:3],np.transpose(camera_pts)) + np.tile(cam_world[0:3,3:],(1,camera_pts.shape[0])))
    # print('world_pts',world_pts)
    differ_hsv = color_img_hsv - color_refer_hsv   #减去背景color
    differ_pts = world_pts - world_pts_refer       #减去背景depth

    differ_colorspace = np.sqrt(differ_hsv[:,0]**2+differ_hsv[:,1]**2+differ_hsv[:,2]**2)
    differ_distances = np.sqrt(differ_pts[:,0]**2+differ_pts[:,1]**2+differ_pts[:,2]**2)

    # 对应位置坐标的差距不大，对应位置hsv的差距不大，深度为0的区域，都设置为背景 00005
    mask = (differ_distances<0.00005)|(differ_colorspace<0.33)|(distance_pts==0) 
    # mask = (differ_distances<0.0001)|(differ_colorspace<0.05)|(distance_pts==0)
    # print('mask shape =',mask.shape())
    # mask = (differ_distances<0.0008)|(differ_colorspace<0.2)
    mask.shape = dep_input_img.shape
    object_index = np.array(np.argwhere(mask==False)) #获取非背景的数组元素的索引并存为nx2矩阵
    
    return(object_index)



#========================================================================
# point_a = np.array([-0.298,-0.3,0.01])
# point_b = np.array([0.102,-0.3,0.007])
# point_c = np.array([-0.298,-0.7,0.02])
# point_d = np.array([0.102,-0.7,0.017])

# abc = plane_equation(point_a,point_b,point_c)
# abd = plane_equation(point_a,point_b,point_d)
# acd = plane_equation(point_a,point_c,point_d)
# bcd = plane_equation(point_b,point_c,point_d)
# print(abc)
# print(abd)
# print(bcd)
# print(acd)
# 法向量[3,10,400]
# 基于robot_base coordinate frame平面的方程为3x+10y+400z-0.106=0
#----------------------------------------------------------
# get the translation from desk to robot base
'''
-------------------------------------------------
    desk coordinate     |   robot coordinate     
-------------------------------------------------

'''
# theta_desk = np.array([math.pi,0,math.pi/2])
# rotm_desk2world = euler2rotm(theta_desk)
# # print(rotm_desk2world)
# rotm_desk2world[ np.abs(rotm_desk2world) < 0.000001] = 0
# # print(rotm_desk2world)
# tram_desk2world = np.array([[0.346],[-0.15],[-0.025]])
# ones = np.array([[0,0,0,1]])
# desk2world = np.concatenate((np.concatenate((rotm_desk2world,tram_desk2world),axis=1),ones),axis=0)
# print(desk2world)
# np.savetxt('real/desk2world.txt', desk2world, delimiter=' ')

# desk2robot = np.loadtxt("real/desk2world_new.txt",delimiter=" ")
# one_desk_pt = np.array([[0],[0],[0],[1]])
# one_robot_pt = np.dot(desk2robot,one_desk_pt)
# print(one_robot_pt)



#----------------------------------------------------------
# get the rotation vector of calibration and desk

# theta_calib = np.array([math.pi,0,0])
# world2calib = euler2rotm(theta_calib)
# rotvect1 = getRotVec(world2calib)
# print(rotvect1)                                     #[3.14159265,0,         0        ]

# theta_desk = np.array([math.pi,0,math.pi/2])
# world2desk = euler2rotm(theta_desk)
# print(world2desk)
# rotvect2 = getRotVec(world2desk)
# print(rotvect2)                                     #[2.22144147,2.22144147,0        ]                                                   


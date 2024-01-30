import os
import numpy as np
import h5py
import urx
import cv2
import time
#from action_transformation import *
import threading
import pybullet as pb
import math

from Calibration_d415.equipment.Gripper import Gripper
global real_action

flag=1
# tactile.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# tactile.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# tactile.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
# tactile.set(cv2.CAP_PROP_FPS,30)
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# fps = 30 
# out = cv2.VideoWriter('tactile1.avi', fourcc, fps, (640, 480))

ac_dir = '/home/zhou/tactile_gym/tactile_gym/vae/train_data/rolate_swing_all_pair.npy'
#ac_dir = '/home/zhou/tactile_gym/tactile_gym/Privileged_learning/action/surface/surface_action.npy'
real_dir='/home/zhou/tactile_gym/tactile_gym/vae/real_train_data/images'
swing=np.load(ac_dir)
robot = urx.Robot("192.168.1.100")
#robot.set_tcp((0.011,-0.007,-0.4,0.06,-0.03,0))
robot.set_tcp((0,0,0.0,0,0,0))
robot.set_payload(2,(0,0,0.1))
reset_position=[-0.11262,-0.48969,0.4494+0.13,2.2159,2.2077,0.0049] #0.050
#reset_position=[-0.33220,-0.38510,0.250,1.1868,2.4982,0.0267]

min_action, max_action = -0.25,0.25
max_pos_vel = 0.01  # m/s
max_ang_vel = 5.0 * (np.pi / 180)  # rad/s
x_act_min, x_act_max = -max_pos_vel, max_pos_vel
y_act_min, y_act_max = -max_pos_vel, max_pos_vel
z_act_min, z_act_max = -max_pos_vel, max_pos_vel
roll_act_min, roll_act_max = -max_ang_vel, max_ang_vel
pitch_act_min, pitch_act_max =  -max_ang_vel, max_ang_vel
yaw_act_min, yaw_act_max = -max_ang_vel, max_ang_vel
close_act_min,close_act_max = -max_ang_vel, max_ang_vel
init_pos = [-0.15123,-0.57240,0.06092]
init_rpy = [3.131571178388143, 0.017776347872228968, -0.07070360740624171]
init_orn = pb.getQuaternionFromEuler(init_rpy)

def robot_reset():
    robot.movel(reset_position,vel=0.3)

# while(1):
#     print(robot.getl())
#     time.sleep(1)

#robot.speedl([0,0,0,0,0,-0.05],0.5,31.4)
# while(1):
#     pass
f = h5py.File('/home/zhou/tactile_gym/tactile_gym/vae/real_train_data/simulation_all_pair.h5', 'w')


def scale_actions(actions):
        
        # would prefer to enforce action bounds on algorithm side, but this is ok for now
        actions = np.clip(actions, min_action, max_action)

        input_range = max_action - min_action

        new_x_range = x_act_max - x_act_min
        new_y_range = y_act_max - y_act_min
        new_z_range = z_act_max - z_act_min
        new_roll_range = roll_act_max - roll_act_min
        new_pitch_range = pitch_act_max - pitch_act_min
        new_yaw_range = yaw_act_max - yaw_act_min
        new_close_range = close_act_max - close_act_min

        scaled_actions = [
            (((actions[0] - min_action) * new_x_range) / input_range) + x_act_min,
            (((actions[1] - min_action) * new_y_range) / input_range) + y_act_min,
            (((actions[2] - min_action) * new_z_range) / input_range) + z_act_min,
            (((actions[3] - min_action) * new_roll_range) / input_range) + roll_act_min,
            (((actions[4] - min_action) * new_pitch_range) / input_range) + pitch_act_min,
            (((actions[5] - min_action) * new_yaw_range) / input_range) + yaw_act_min,
            (((actions[6] - min_action) * new_close_range) / input_range) + close_act_min,
        ]

        return np.array(scaled_actions)

def encode_TCP_frame_actions(actions):
        encoded_actions = np.zeros(7)
        encoded_actions = actions
 

        return encoded_actions

def robot_j_trans(robot_j):
    tf_j=np.zeros(6)
    tf_j[0]=robot_j[0]
    tf_j[1]=robot_j[1]+1.57
    tf_j[2]=robot_j[2]
    tf_j[3]=robot_j[3]+1.57
    tf_j[4]=robot_j[4]
    tf_j[5]=robot_j[5]-1.57

    return tf_j

def goal_pose(delta_pose):
     # 获取当前机械臂的位置
    current_pose = robot.getl().array
    current_pose=np.append(current_pose,0)
    zf = [-1,-1,1,1,1,1,1]
    delta_pose = [delta_pose[i] * zf[i] for i in range(7)]
    # 计算目标位置
    target_pose = [current_pose[i] + delta_pose[i] for i in range(7)]
    return target_pose

def delta_x_tf_speed(delta_pose):
    speed_control = np.zeros(6)
    zf = [-1, -1, 1, -1, -1, 1, 1]
    delta_pose = [delta_pose[i] * zf[i] for i in range(7)]
    speed_control = [1.1*delta_pose[i] / 0.1 for i in range(7)]
    # for i in range(len(speed_control)):
    #      if(np.abs(speed_control[i])>0.05):
    #           speed_control[i]=0.01
    return speed_control
     
def robot_tcp_trans(robot_tcp):
    tf_tcp = np.zeros(7)
    tf_tcp[0] = -robot_tcp[0]+0.006
    tf_tcp[1] = -robot_tcp[1]
    tf_tcp[2] =  robot_tcp[2]+0.85
    quaternion = quaternion_from_rotation_vector(robot_tcp[3:6])
    tf_tcp[3] = quaternion[0]
    tf_tcp[4] = quaternion[1]
    tf_tcp[5] = quaternion[2]
    tf_tcp[6] = quaternion[3]
    return tf_tcp

#旋转矢量转四元数
def normalize_vector(vector):
    magnitude = math.sqrt(sum(component ** 2 for component in vector))
    return [component / magnitude for component in vector]


def quaternion_from_rotation_vector(rotation_vector):
    rotation_angle = math.sqrt(sum(component ** 2 for component in rotation_vector))
    rotation_axis = normalize_vector(rotation_vector)
    half_angle = rotation_angle / 2.0
    sin_half_angle = math.sin(half_angle)
    quaternion = [component * sin_half_angle for component in rotation_axis]
    quaternion.append(math.cos(half_angle))
    return quaternion

def finger_sensor_pose(tcp_pose,grasp_num):
    lf_pose = np.zeros(3)
    rf_pose = np.zeros(3)
    theta = robot_j_trans(robot.getj())[5]
    gama = (-0.3*grasp_num+50)/180 * np.pi
    h = 0.175+0.1*math.cos(gama)
    r = 0.1*math.sin(gama)-0.002
    lf_pose[2] = tcp_pose[2]-h
    rf_pose[2] = tcp_pose[2]-h
    lf_pose[0] = tcp_pose[0]+math.sin(np.pi-theta)*r
    lf_pose[1] = tcp_pose[1]-math.cos(np.pi-theta)*r
    rf_pose[0] = tcp_pose[0]+math.sin(0-theta)*r
    rf_pose[1] = tcp_pose[1]-math.cos(0-theta)*r

    #对应仿真
    lf_pose[0] = -lf_pose[0]+0.0089
    lf_pose[1] = -lf_pose[1]-0.003
    lf_pose[2] = lf_pose[2]+0.786
    rf_pose[0] = -rf_pose[0]+0.0131
    rf_pose[1] = -rf_pose[1]-0.008
    rf_pose[2] = lf_pose[2]+0.7803

    return lf_pose,rf_pose


def control_thread():
    while True:
        if real_action is not None:
            #time3=time.time()
            #robot.speedl([-0.02,0,0,0,0,0],0.5,0.2)
            #robot.speedl([real_action[0],real_action[1],real_action[2],real_action[3],real_action[4],real_action[5]],0.5,0.2)
            #print(real_action)
            #robot.movel([real_action[0],real_action[1],real_action[2],real_action[3],real_action[4],real_action[5]],0.5,0.01)
            #time4 = time.time()
            #print("1")
            pass  
            #time.sleep(0.008)

swing = np.loadtxt('/home/zhou/VisualTactile-main/sim2real/action_record.txt')
if __name__ == "__main__":
    robot_reset()
    gripper_2f140 = Gripper()
    gripper_2f140.gripper_action(28)
    grasp_num = 28
    #print(robot.getl())
    print("reset done!")
    real_action = None
    thread = threading.Thread(target=control_thread, daemon=True)
    thread.start()
    for i in range(len(swing)):  #动作序列的长度
        time1=time.time()

        
        #这里保存图像frame
        #f.write
        #这里执行下一步action
        #print([swing[i][0]/10,-swing[i][1]/10,0,0,0,swing[i][2]/10])
        #action = [0.2, 0, 0., 0., 0., 0.,0]
        action = swing[i]
        cmd_limit =[0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05]
        action = [action[i]*cmd_limit[i] for i in range(7)]
        #cur_tcp_orn = pb.getQuaternionFromEuler([robot.getl()[3],robot.getl()[4],robot.getl()[5]])
        encoded_action = encode_TCP_frame_actions(action)

        #delta_action = scale_actions(encoded_action)

        delta_action = encoded_action
        real_action = goal_pose(delta_action)
        robot.movel([real_action[0],real_action[1],real_action[2],real_action[3],real_action[4],real_action[5]],0.5,0.01)
        if(real_action[6]!=0):
            delta_grasp = real_action[6]/0.07/0.05
            grasp_num += int(delta_grasp)
            if(grasp_num>20 and grasp_num<240):
                gripper_2f140.gripper_action(grasp_num)
            else:
                print("grasp error")
        #real_action = delta_x_tf_speed(delta_action)
        #print(real_action)
        #real_action=swing[i]        #仿真环境动作实现

        current_pose = robot.getl().array
        current_pose = robot_tcp_trans(current_pose)
        print(i,current_pose[:3])
        lf_pose,rf_pose = finger_sensor_pose(robot.getl().array,grasp_num)
        #print("lf_pose",lf_pose,"rf_pose:",rf_pose,"delta_pose:",lf_pose[1]-rf_pose[1])
        time2 = time.time()
        frequence = 0.1
        time_wait = frequence - (time2-time1)

        if time_wait > 0:
            time.sleep(time_wait)
        else:
            pass
        #print(real_action)

        # robot_j=np.zeros(6)
        # robot_j[0]=robot.getj()[0]/3.14*180
        # robot_j[1]=robot.getj()[1]/3.14*180
        # robot_j[2]=robot.getj()[2]/3.14*180
        # robot_j[3]=robot.getj()[3]/3.14*180
        # robot_j[4]=robot.getj()[4]/3.14*180
        # robot_j[5]=robot.getj()[5]/3.14*180
        #robot_j = robot_j_trans(robot.getj())

        # 获取位置信息
        if cv2.waitKey(1) ==ord('q'):
            break
    # out.release()
    cv2.destroyAllWindows()
    robot.close()
    time.sleep(0.1)
    real_action=None



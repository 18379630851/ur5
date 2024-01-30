import socket 
import struct 
import math 
import collections
import numpy as np 
import time

class Robot():
    # 1.构造函数
    def __init__(self):
        # set relavant parameters 
        self.robot_host = "192.168.1.100"
        self.robot_port = 30003
        self.joint_acc = 1.4
        self.joint_vel = 0.6
        self.tool_acc = 1
        self.tool_vel = 0.25
        self.joint_tolerance = 0.01
        self.tool_pose_tolerance = [0.002,0.002,0.002,0.01,0.01,0.01]
        ##定义放置的位置（这里需要修改）
        self.target_pos = np.array([-11*math.pi/12,-math.pi/2,-math.pi/2,-math.pi/2,math.pi/2,0])
        self.home_pos = np.array([-math.pi/2,-math.pi/4,-2*math.pi/3,-math.pi/2,math.pi/2,0]) 
        self.dic= collections.OrderedDict([('MessageSize','i'), ('Time','d'), ('q target','6d'), ('qd target','6d'), ('qdd target','6d'),('I target','6d'),
                                           ('M target','6d'), ('q actual','6d'), ('qd actual','6d'), ('I actual','6d'), ('I control','6d'),
                                           ('Tool vector actual','6d'), ('TCP speed actual','6d'), ('TCP force','6d'), ('Tool vector target','6d'),
                                           ('TCP speed target','6d'), ('Digital input bits','d'), ('Motor temperatures','6d'), ('Controller Timer','d'),
                                           ('Test value','d'), ('Robot Mode','d'), ('Joint Modes','6d'), ('Safety Mode','d'), ('empty1','6d'), ('Tool Accelerometer values','3d'),
                                           ('empty2','6d'), ('Speed scaling','d'), ('Linear momentum norm','d'), ('SoftwareOnly','d'), ('softwareOnly2','d'), ('V main','d'),
                                           ('V robot','d'), ('I robot','d'), ('V actual','6d'), ('Digital outputs','d'), ('Program state','d'), ('Elbow position','3d'), ('Elbow velocity','3d')])
        
        self.joint_move(self.home_pos)
       

    # 2.关节移动
    def joint_move(self,joint_vector):
        self.robot_tcp = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.robot_tcp.connect((self.robot_host,self.robot_port))
        robot_message = b"movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f)\n" % (joint_vector[0],joint_vector[1],joint_vector[2],joint_vector[3],joint_vector[4],joint_vector[5],self.joint_acc,self.joint_vel)
        self.robot_tcp.send(robot_message)
        self.robot_tcp.close()

        # Block until robot reaches target tool position
        actual_tool_pose = self.get_robot_state("q actual")
        while not all([np.abs(actual_tool_pose[j] - joint_vector[j]) < self.joint_tolerance for j in range(6)]): 
            actual_tool_pose = self.get_robot_state("q actual")
        time.sleep(0.1)
         
        

    # 在推移物体时，使用movel，可以保证空间内的平行直线推动;但是在移动机械臂是，使用了大量的movej(适用性更广更强)
    # 3.坐标系移动
    def linear_move(self,tool_position,tool_orientation):
        self.robot_tcp = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.robot_tcp.connect((self.robot_host,self.robot_port))
        robot_message = b"movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f)\n" % (tool_position[0],tool_position[1],tool_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
        self.robot_tcp.send(robot_message)
        self.robot_tcp.close()

        # Block until robot reaches target tool position
        actual_tool_pose = self.get_robot_state("Tool vector actual")
        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):                       
            actual_tool_pose = self.get_robot_state("Tool vector actual")
        time.sleep(0.1)
        
        

    # 4.获取机械臂的状态信息
    def get_robot_state(self,key_word):
        self.robot_tcp = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.robot_tcp.connect((self.robot_host,self.robot_port))
        data = self.robot_tcp.recv(1108)
        self.robot_tcp.close()
        # ii = range(len(self.dic))
        sum = 0
        for key in self.dic:
            fmtsize=struct.calcsize(self.dic[key])
            if key !=  key_word:
                sum += fmtsize
            else:
                specific_data = data[sum:sum+fmtsize]
                fmt = "!"+self.dic[key]
                key_word_data = self.dic[key],struct.unpack(fmt, specific_data)
        state_data = np.array(key_word_data[1])
        return state_data


if __name__ == "__main__":
    robot_ur5 = Robot()
    robot_ur5.linear_move([ 0.05228997,-0.62712123,0.03255],[1.80194352,2.57344204,0])


    

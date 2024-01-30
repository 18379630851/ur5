# coding=utf8
# 此段代码有用，但是需要整理寄存器的各个数据
import serial              # pip3 install pyserial
import time
import binascii
import cv2

class Gripper():
    def __init__(self):
        # communicate and initialize the gripper 
        self.ser = serial.Serial(port='/dev/ttyUSB0',
                             baudrate=115200,
                             timeout=1,
                             parity=serial.PARITY_NONE,
                             stopbits=serial.STOPBITS_ONE,
                             bytesize=serial.EIGHTBITS)
        self.activate_gripper_stream = b"\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30" #page 78
        # self.open_text = b"\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1"
        # self.read_status = "\x09\x01\x07\xD0\x00\x01\x85\xCF"
        # self.read_status2 = "\x09\x01\x07\xD0\x00\x03\x04\x0E"
        self.open_gripper_stream = b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19"
        self.close_gripper_stream = b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29"
        #(0xFF:full closing of the Gripper;0xFF:full speed;0xFF:full force)

        # activate the gripper
        self.ser.write(self.activate_gripper_stream)
        # self.ser.write(self.read_status)
        # self.ser.write(self.read_status2)
        self.ser.readline()                 #保证预留充足的时间，使得夹爪完成上一步的指令
        print("the gripper is activated.")
        time.sleep(0.1)    
    # def activate_gripper(self):
    #     self.ser.write(self.activate_gripper_stream)
    #     self.ser.readline()
    #     print("the grippeer is activated")
    #     time.sleep(0.1)    

    # open the gripper
    def open_gripper(self):
        self.ser.write(self.open_gripper_stream)
        #self.ser.write(self.read_status)
        self.ser.readline()
        print("the gripper is opening.")
        time.sleep(0.1)

    # close the gripper
    def close_gripper(self):
        self.ser.write(self.close_gripper_stream)
        self.ser.readline()                 
        print("the gripper is closed.")
        time.sleep(0.1)
    def gripper_action(self,x):
        action = hex(x)
        action = action[2:]
        action = action.upper()
        #print(action)
        sd = '09 10 03 E8 00 03 06 09 00 00 '+str(action)+' 80 80'
        crc = calc_crc(sd.replace(' ', ''))
        #print(crc)
        begin = b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00" + bytes([x]) + b"\x80\x80" +bytearray.fromhex(crc[0])+bytearray.fromhex(crc[1])
        #print(begin)
        # middle = "\xFF\xFF"
        # print(begin)
        #action_gripper_stream = str.encode(fr" x09 x10 x03 xE8 x00\x03\x06\x09\x00\x00\x{action}\xFF\xFF\x{crc[0]}\x{crc[1]}")
        #print(action_gripper_stream)
        self.ser.write(begin)
        loop=True
        while(loop):
            # request = b"\x09\x01\x07\xD0\x00\x01\x85\xCF"
            # self.ser.write(request)
            s = self.ser.readline()
            #print(s)
            if s !=b'':
                loop=True
            else:
                loop=False        
        time.sleep(0.1)

def calc_crc(string):
    data = bytearray.fromhex(string)
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for i in range(8):
            if (crc & 1) != 0:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    hex_crc = hex(((crc & 0xff) << 8) + (crc >> 8)) # 返回十六进制
    crc_0 = crc & 0xff
    crc_1 = crc >> 8
    str_crc_0 = '{:02x}'.format(crc_0).upper()
    str_crc_1 = '{:02x}'.format(crc_1).upper()
    return str_crc_0, str_crc_1 # 返回两部分十六进制字符

# if __name__ == '__main__':
#     sd = '09 10 03 E8 00 03 06 09 00 00 A0 FF FF'
#     crc = calc_crc(sd.replace(' ', ''))
#     print(crc)
#50    105
#100   80  
#150   45
#200   20 
if __name__ == "__main__":
    #cv2.namedWindow('image')
    gripper_2f140 = Gripper()
    gripper_2f140.open_gripper()
    gripper_2f140.gripper_action(28)
    #gripper_2f140.gripper_action(58.5)
    # gripper_2f140.gripper_action(88)
    # gripper_2f140.gripper_action(118)

    #print("grasp num:",num)
    #gripper_2f140.close_gripper()
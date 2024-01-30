import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from utils.compute import get_object_index,get_cam_pts,camera_intrinsics,cam_world

background = [255,255,255]                 #标记各个像素的颜色        
book = [255,0  ,0  ]            
concave_cuboid = [255,165,0  ]            
cube = [255,110,180]
cylinder = [0  ,0  ,255]
tri_prism = [143,188,143]
cuboid_short = [0,0,0]


DSET_MEAN = [0.27356932,0.44076302,0.47741872]
DSET_STD = [0.15257772,0.16305404,0.19518022]

label_colours = np.array([background,book,concave_cuboid,
                        cube,cylinder,tri_prism,cuboid_short])


def view_annotated(tensor, plot=True):          #视图注释
    temp = tensor.numpy()                       #将tensor张量转化为numpy矩阵;得到的结果应该是h,w的二维
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,7):                       #l是12个类别的标记，整个图像上遍历12个类别的像素，分别赋值
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]              #为什么要除以255（为什么要归一化）？
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:                                    #展示语义分割结果
        plt.imshow(rgb)
        plt.show()
    else:                                       #返回结果，是为了要与真实标注作对比？，显示最终表现效果
        return rgb

def decode_image(tensor):
    inp = tensor.numpy().transpose((1, 2, 0))   #这里表示坐标轴互换，因为tensor的格式是channel、height、width,这里还是变回原来的格式吧
    mean = np.array(DSET_MEAN)
    std = np.array(DSET_STD)
    inp = std * inp + mean                      #编码是(pixel-mean)/std
    return inp

def view_image(tensor):
    inp = decode_image(tensor)
    inp = np.clip(inp, 0, 1)                    #将inp中的元素限制在0-1之间，大于1的设为1,小于0的设为0
    plt.imshow(inp)
    plt.show()



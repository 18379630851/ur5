import transforms3d as tfs
import numpy as np 

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as pb

# quat2mat(good)
R_calib = [0.03920891, 0.02203872, 0.59794866, 0.80027143]
M_base_cam = np.array(pb.getMatrixFromQuaternion(R_calib)).reshape(3,3)
r = R.from_matrix(M_base_cam)

# mat2euler
r = R.from_matrix([[-7.740712266707006593e-03, -8.119593808756747633e-01, 5.836626124582530162e-01],
[-9.993016190860873893e-01, -1.505663251290337810e-02, -3.419900450738040665e-02],
[3.655619599368210154e-02, -5.835197182832477791e-01, -8.112757748812114977e-01]])
print("r.as_euler('zyx')*180/pi: {}".format(r.as_euler('zyx')*180/3.1415926))


import transforms3d as tfs
import numpy as np 
import pybullet as pb
import math
from scipy.spatial.transform import Rotation as R

# 四元数转旋转矩阵
r = tfs.quaternions.quat2mat([ 0.22727021, -0.65645264,  0.66286255, -0.27934104])

# 旋转矩阵转四元数
R = [[-7.740712266707006593e-03, -8.119593808756747633e-01, 5.836626124582530162e-01],
[-9.993016190860873893e-01, -1.505663251290337810e-02, -3.419900450738040665e-02],
[3.655619599368210154e-02, -5.835197182832477791e-01, -8.112757748812114977e-01]]
r = tfs.quaternions.mat2quat(np.asarray(R))
print(r)

import math
import os
import sys

import matplotlib.pyplot as plt
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from PIL import Image as Im
from tasks.base.base_task import BaseTask
from utils import o3dviewer
from utils.mimic_util import actuate, find_joints_with_dof, mimic_clip, position_check
from utils.torch_jit_utils import quat_mul, tensor_clamp, to_torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../Pointnet2_PyTorch/pointnet2_ops_lib"))
import torch
#print(sys.path)
from torchvision import transforms
from pointnet2_ops import pointnet2_utils

#import numpy as np

gym_BLUE = gymapi.Vec3(0., 0., 1.)

def get_UR5_asset(gym, sim, asset_root, asset_file):
    """Create a UR5 asset with a linear slider."""
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    ur5_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return ur5_asset


class Ur5pickup(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.device_type = device_type
        self.device_id = device_id
        self.debug = cfg["env"]["debug"]
        self.debug_view_type = self.cfg["env"]["debug_camera_type"]


        self.obs_type = self.cfg["env"]["obs_type"]
            
        self.transforms_depth = transforms.CenterCrop((240,320))
        if "oracle" in self.obs_type:    
            self.num_obs = 30  # 23+ 12

        #     self.num_obs = 30+1024*3
        elif "pointcloud" in self.obs_type:
        # task-specific parameters
            self.point_cloud_debug = self.cfg["env"]["Pointcloud_Visualize"]
            self.camera_view_debug = self.cfg["env"]["Camera_Visualize"]
        elif "tactile" in self.obs_type:
        # task-specific parameters
            self.point_cloud_debug = self.cfg["env"]["Pointcloud_Visualize"]
            self.camera_view_debug = self.cfg["env"]["Camera_Visualize"]
        else: 
            print("choose a obs type")

        if self.debug == False:
            self.point_cloud_debug = False
            self.camera_view_debug = False
        else:
            plt.ion()

        self.num_obs = 30
        self.num_act = 7  # force applied on the pole (-1 to 1)

        self.reset_dist = 3.0  # when to reset
        self.max_push_effort = 400.0  # the range of force applied to the ur5reach
        self.max_episode_length = 500  # maximum episode length

        self.pointCloudDownsampleNum = self.cfg["env"]["PCDownSampleNum"]
        self.sensor_downsample_num = self.cfg["env"]["TDownSampleNum"]
        self.all_downsample_num = self.cfg["env"]["AllDownSampleNum"]
        
        #for saving visualized pointcloud
        self.save_pc = False


        # Tensor placeholders
        self.states = {} 
        
        self.cfg["env"]["numObservations"] = self.num_obs
        self.cfg["env"]["numActions"] = self.num_act
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.cfg["headless"] = headless

        self.num_envs = self.cfg["env"]["numEnvs"]

        self.arm_dof = self.cfg["env"]["arm_dof"]
        self.action_scale = self.cfg["env"]["actionScale"]

        

        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)

        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self.start_position_noise = 0.15
        self.start_rotation_noise = 0.785
        self._pos_control = None            # Position actions
        self.num_force_sensor = 2
        

        #self.device = self.cfg["env"]["sim_device"]
        super().__init__(cfg=self.cfg)
        #self.joint_limits = [[]]

        

    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, spacing=2.5, num_per_row=int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        # define environment space (for visualisation)
        lower = gymapi.Vec3(0, 0, 0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        #camera and pointcloud
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        #default pos
        self.default_dof_pos = to_torch(
            [-1.57, 0, -1.57, 0, 1.57, 0, 
             0., 0., 0., 0., 0., 0.], device=self.device
        )
        

        self.position_limits = to_torch([[-4.,-1.5,-2.355,-0.785,-3.14,-3.14,0   ],
                                         [-1.6,1.5, 0.,    1.5,   3.14, 3.14,0.57]], device=self.device)
        self.osc_limits = to_torch([[-0.05,0.05,0.87],
                                    [0.65,0.8,1.85]], device=self.device)
        self.init_goal_pos = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        # Set control limits
        self.cmd_limit = to_torch([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05], device=self.device).unsqueeze(0)


        asset_root = 'assets'
        asset_file = 'ur5_rq_gsmini_wo_cover.urdf'

        ur5_assert = get_UR5_asset(self.gym, self.sim, asset_root, asset_file)
        self.num_dof = self.gym.get_asset_dof_count(ur5_assert)
        ur5_dof_names = self.gym.get_asset_dof_names(ur5_assert)
        
        self.all_limits = torch.zeros((2,self.num_dof),device=self.device)

        # Create table asset
        self.table_stand_height = 0.83
        table_pos = [0.30, 0.365, self.table_stand_height/2]
        
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.8, 1.0, self.table_stand_height], table_opts)

        # Create table connector asset
        table_con_height = 0.02
        table_con_pos = [0.0, 0.0, self.table_stand_height + table_con_height / 2]
        table_con_opts = gymapi.AssetOptions()
        table_con_opts.fix_base_link = True
        table_con_asset = self.gym.create_box(self.sim, *[0.2, 0.15, table_con_height], table_opts)

        #create cube asset
        self.cubeA_size = 0.06
        cubeA_pos = [0.30, 0.365, self.table_stand_height + self.cubeA_size / 2]
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)
        self._init_cubeA_state = torch.zeros((self.num_envs, 13), device=self.device)

        self.revolute_joints, self.mimic_joints, self.actuator_joints, dof = find_joints_with_dof(asset_root, asset_file, ur5_dof_names)

        self.all_limits = mimic_clip(self.actuator_joints, self.mimic_joints,self.arm_dof, self.all_limits, self.position_limits)
        self.num_state = 2 * self.num_dof #dof -> position speed

        urdf_root = "/home/zhou/VisualTactile-main/assets"
        urdf_file = "small_ball.urdf"
        model_pose = gymapi.Transform()
        model_pose.p = gymapi.Vec3(0.39, 0.365, self.table_stand_height +  0.03)  # 模型的位置
        model_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 模型的旋转
        model_color = gymapi.Vec3(0.6, 0.1, 0.0)
        ball_options = gymapi.AssetOptions()
        #ball_options.fix_base_link = True
        model_asset = self.gym.load_asset(self.sim,urdf_root, urdf_file,ball_options)


        # define ur5reach pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, table_con_height + self.table_stand_height)  # generate the ur5reach 1m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(*cubeA_pos)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*table_pos)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_con_pose = gymapi.Transform()
        table_con_pose.p = gymapi.Vec3(*table_con_pos)
        table_con_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        right_sensor_idx = self.gym.find_asset_rigid_body_index(ur5_assert, "right_box")
        left_sensor_idx = self.gym.find_asset_rigid_body_index(ur5_assert, "left_box")

        sensor_pose1 = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.02))
        sensor_pose2 = gymapi.Transform(gymapi.Vec3(-0.0, 0.0, 0.02))

        sensor_idx1 = self.gym.create_asset_force_sensor(ur5_assert, right_sensor_idx, sensor_pose1)
        sensor_idx2 = self.gym.create_asset_force_sensor(ur5_assert, left_sensor_idx, sensor_pose2)
        
        
        # define ur5reach dof properties
        dof_props = self.gym.get_asset_dof_properties(ur5_assert)
        dof_props["driveMode"][:self.arm_dof].fill(gymapi.DOF_MODE_POS)
        dof_props["driveMode"][self.arm_dof:self.num_dof].fill(gymapi.DOF_MODE_EFFORT)

        dof_props["stiffness"][:self.arm_dof].fill(5000.0)
        dof_props["stiffness"][self.arm_dof:self.num_dof].fill(7000.0)

        dof_props["damping"][:self.arm_dof].fill(40.0)
        dof_props["damping"][self.arm_dof:self.num_dof].fill(1.0e2)

        num_robot_bodies = self.gym.get_asset_rigid_body_count(ur5_assert)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(ur5_assert)
        max_agg_bodies = num_robot_bodies + 3     # 1 for table, table stand, cubeA
        max_agg_shapes = num_robot_shapes + 3     # 1 for table, table stand, cubeA

        # generate environments
        self.envs = []
        self.targ_handles = []
        self.ur5_handles = []
        self.targ_idxs = []
        print(f'Creating {self.num_envs} environments.')


        self.all_pointcloud_flatten = torch.zeros((self.num_envs, self.all_downsample_num * 3), device=self.device)
        if  "pointcloud" in self.obs_type:

            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True


            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            
            self.pointcloud_flatten = torch.zeros((self.num_envs, self.pointCloudDownsampleNum * 3), device=self.device)
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')


        else:
            self.pointcloud_flatten = None

        
        if "tactile" in self.obs_type:
            self.sensors = []     # 包含所有 sensors
            self.projs = []
            self.vinvs = []
            self.visualizers = []
            self.sensor_width = 320  # 1.65 320
            self.sensor_height = 240
            self.sensors_camera_props = gymapi.CameraProperties()
            self.sensors_camera_props.enable_tensors = True
            self.sensors_camera_props.horizontal_fov = 56
            self.sensors_camera_props.width = self.sensor_width
            self.sensors_camera_props.height = self.sensor_height
            
            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.sensor_pointcloud_flatten = torch.zeros((self.num_envs, self.sensor_downsample_num * 3), device=self.device)

            sensor_u = torch.arange(0, self.sensor_width, device=self.device)
            sensor_v = torch.arange(0, self.sensor_height, device=self.device)
            self.sensor_v2, self.sensor_u2 = torch.meshgrid(sensor_v, sensor_u, indexing='ij')
        else:
            self.sensor_pointcloud_flatten = None


        if self.save_pc:
            self.pc_data = torch.zeros((self.num_envs , self.pointCloudDownsampleNum, 3), device=self.device)
        else:
            self.pc_data = None

        if self.point_cloud_debug:
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else:
            self.pointCloudVisualizer = None


        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            # add ur5reach here in each environment
            ur5_handle = self.gym.create_actor(env_ptr, ur5_assert, pose, "ur5reach", i, 1, 0)
            self.ur5_handles.append(ur5_handle)

            self.gym.set_actor_dof_properties(env_ptr, ur5_handle, dof_props)

            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            model_actor = self.gym.create_actor(env_ptr, model_asset, model_pose, "Model",i, 0,0)
            table_con_actor = self.gym.create_actor(env_ptr, table_con_asset, table_con_pose, "table_con", i, 1, 0)


            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 0, 0)
            

            #obj props
            object_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._cubeA_id)
            object_props[0].mass = 0.1
            self.gym.set_actor_rigid_body_properties(env_ptr, self._cubeA_id, object_props)
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)

            model_props = self.gym.get_actor_rigid_body_properties(env_ptr, model_actor)
            model_props[0].mass = 0.1
            self.gym.set_actor_rigid_body_properties(env_ptr, model_actor, model_props)
            self.gym.set_rigid_body_color(env_ptr, model_actor, 0, gymapi.MESH_VISUAL, model_color)


            #camera
            if "pointcloud" in self.obs_type or "tactile" in self.obs_type:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z

            if  "pointcloud" in self.obs_type:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.64, 0.485, self.table_stand_height+0.6), gymapi.Vec3(0.38, 0.485, self.table_stand_height))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)

                
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)


            if "tactile" in self.obs_type:
                # 创建 传感器 相机 handle
                # sensor_camera

                sensor_handle_1 = self.gym.create_camera_sensor(env_ptr, self.sensors_camera_props)
                right_sensor_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_handle, "right_gelsight_mini")
                camera_offset1 = gymapi.Vec3(0.0, -0.00, 0.00)
                camera_rotation1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.deg2rad(-90))
                actor_handle1 = self.gym.get_actor_handle(env_ptr, 0)

                body_handle1 = self.gym.get_actor_rigid_body_handle(env_ptr, actor_handle1, right_sensor_handle)  # right sensor
                # 将相机 handle 附着到sensor的 base_link 上
                self.gym.attach_camera_to_body(sensor_handle_1, env_ptr, body_handle1,
                                          gymapi.Transform(camera_offset1, camera_rotation1), gymapi.FOLLOW_TRANSFORM)
                self.sensors.append(sensor_handle_1)


                # 创建相机 handle sensor_camera
                sensor_handle_2 = self.gym.create_camera_sensor(env_ptr, self.sensors_camera_props)
                left_sensor_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_handle, "left_gelsight_mini")
                camera_offset2 = gymapi.Vec3(0.0, -0.00, 0.00)
                camera_rotation2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.deg2rad(-90))
                actor_handle2 = self.gym.get_actor_handle(env_ptr, 0)

                body_handle2 = self.gym.get_actor_rigid_body_handle(env_ptr, actor_handle2, left_sensor_handle)  # left sensor
                # 将相机 handle 附着到sensor的 base_link 上
                self.gym.attach_camera_to_body(sensor_handle_2, env_ptr, body_handle2,
                                          gymapi.Transform(camera_offset2, camera_rotation2), gymapi.FOLLOW_TRANSFORM)
                self.sensors.append(sensor_handle_2)



            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
        
        self.init_data()

    def init_data(self):
        env_ptr = self.envs[0] 
        ur5_handle = 0
        #check your urdf
        num_ur5_rigid_bodies = self.gym.get_actor_rigid_body_count(env_ptr, ur5_handle)
        

        self.handles = {
            # Ur5
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, ur5_handle, "ee_link"),
            "hand_left": self.gym.find_actor_rigid_body_handle(env_ptr, ur5_handle, "left_box"),
            "hand_right": self.gym.find_actor_rigid_body_handle(env_ptr, ur5_handle, "right_box"),

        }

        #for sensor index
        #num_rigid_bodies = num_ur5_rigid_bodies + 3 #table tablestand cube
        # self.index_rigid_bodies = {
        #     "hand": [self.handles["hand"] + i * num_rigid_bodies - 1 for i in range(self.num_envs)],
        #     "hand_left": [self.handles["hand_left"] + i * num_rigid_bodies -1 for i in range(self.num_envs)],
        #     "hand_right": [self.handles["hand_right"] + i * num_rigid_bodies -1 for i in range(self.num_envs)],
        # }

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)  #including objs
        self._init_cubeA_state[:, :3] = torch.tensor([0.30, 0.365, self.table_stand_height + self.cubeA_size / 2], device=self.device)
        self._init_cubeA_state[:, 6] = torch.tensor([1], device=self.device)

        self.init_goal_pos[:,:3] = to_torch([0.30, 0.365, self.table_stand_height + self.cubeA_size / 2 + 0.5], device=self.device)
        self.init_goal_pos[:,6] = to_torch([1], device=self.device)    

        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)      #only dof
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  #arm_hand
        
        #_net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.fs_tensor = gymtorch.wrap_tensor(force_sensor_tensor)


        #The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), 
        # linear velocity([7:10]), and angular velocity([10:13]).


        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)  #pos speed
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["hand_left"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["hand_right"], :]
        self._q = self._dof_state[..., 0]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :] 
        self.goal_pos = self.init_goal_pos[:,:3]

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5reach")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        self._j_eef = jacobian[:, self.arm_dof, :, :]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)

        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)  #3 num of actor
        

    def _update_states(self):
        self.states.update({
            "q": self._q[:, :7],
            "eef_pos": self._eef_state[:, :3],  #3
            "eef_quat": self._eef_state[:, 3:7],  #4
            "eef_lin_vel": self._eef_state[:, 7:10],  #3
            "eef_ang_vel": self._eef_state[:, 10:13],  #3
            "middle_gripper_state": (self._eef_lf_state[:,:3] + self._eef_rf_state[:,:3]) / 2. ,
            "eef_lf_pos": self._eef_lf_state[:, :3],   #3
            #"eef_lf_quat": self._eef_lf_state[:, 3:7],   #4
            "eef_rf_pos": self._eef_rf_state[:, :3], #3
            #"eef_rf_quat": self._eef_rf_state[:, 3:7],   #4
            "goal_pos": self.goal_pos[:, :3],
            "cube_pos": self._cubeA_state[:, :3],
            "cube_quat": self._cubeA_state[:, 3:7],
            "cube_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            "pointcloud": self.pointcloud_flatten,
            "tactile": self.sensor_pointcloud_flatten,
            "all_pc": self.all_pointcloud_flatten
        })    
        
    # def save_pc_data(self, pointcloud_data):
    #     split_pc = torch.split(pointcloud_data.cpu(), split_size_or_sections=1, dim=0)
    #     for i, pc in enumerate(split_pc):
    #         np.savetxt(f'PCmodule/data/output_{i}.txt', pc.squeeze().numpy(), fmt='%.6f')


    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        #self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        if "pointcloud" in self.obs_type or "tactile" in self.obs_type:
            self.compute_point_cloud_observation()
        else:
            pass

        self._update_states()

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.successes[:] = compute_reach_reward(   self.reset_buf,
                                                                        self.progress_buf,
                                                                        self.states,
                                                                        self.max_episode_length)

    def compute_observations(self):
        self._refresh()
        obs = ["q", "eef_pos", "eef_quat", "eef_lf_pos", "eef_rf_pos","goal_pos", "cube_pos", "cube_quat"]
        states = ["q", "eef_pos", "eef_quat", "eef_lf_pos", "eef_rf_pos","goal_pos", "cube_pos", "cube_quat", "all_pc"]
        #prioperception ["q", "eef_pos", "eef_quat", "eef_lf_pos", "eef_rf_pos"]
        student = ["q","eef_pos", "eef_quat", "eef_lf_pos", "eef_rf_pos","goal_pos","all_pc"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        self.states_buf = torch.cat([self.states[state] for state in states], dim=-1)


    def reset(self, env_ids):

        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()


        #sampled_cube_state = self._random_cubeA_state(self._init_cubeA_state, env_ids)
        sampled_cube_state = self._init_cubeA_state

        #sampled_goal_state = self._random_goal_state(self.init_goal_pos)
        sampled_goal_state = self.init_goal_pos
       
        dof_state_reset = torch.zeros_like(self._dof_state, device=self.device)
        dof_state_reset[:,:,0] = self.default_dof_pos
        self._pos_control  = self.default_dof_pos
        self._q = self._dof_state[..., 0]

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                               gymtorch.unwrap_tensor(dof_state_reset),
                                               gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        

        self._root_state[env_ids, self._cubeA_id, :] = sampled_cube_state[env_ids,:]  #TODO:debug
 
        self.goal_pos = sampled_goal_state
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        #print(self.goal_pos)
        # clear up desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.successes[env_ids] = 0

        self._refresh()
        # refresh new observation after reset
        self.compute_observations()
        # 

    def compute_point_cloud_observation(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        
        point_clouds = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
        sensors_point_clouds = torch.zeros((self.num_envs, 2 * self.sensor_downsample_num, 3), device=self.device)
        #all_point_clouds = torch.zeros((self.num_envs, self.all_downsample_num, 3), device=self.device)
        
        for i in range(self.num_envs):
            if  "pointcloud" in self.obs_type:
                # Here is an example. In practice, it's better not to convert tensor from GPU to CPU
                points = depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i], 
                                                        self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2, 
                                                        self.camera_props.width, self.camera_props.height, 10, self.device).contiguous()
                #print(points.shape)
                if points.shape[0] > 0:
                    selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_mathed='random')
                else:
                    selected_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
                
                point_clouds[i] = selected_points #centorids

            if  "tactile" in self.obs_type:
                for index in range(2):
                    real_index = 2*i + index

                    if torch.det(torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.sensors[real_index]))) == 0:
                        cam_vinv = torch.zeros(4,4).to(self.device)
                    else:
                        cam_vinv = torch.inverse(
                            (torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.sensors[real_index])))).to(self.device)
                    cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.envs[i], self.sensors[real_index]),
                                            device=self.device)

                    camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i],
                                                                        self.sensors[real_index], gymapi.IMAGE_DEPTH)
                    torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                    torch_cam_tensor = self.transforms_depth(torch_cam_tensor)

                    points = sensor_depth_image_to_point_cloud_GPU(torch_cam_tensor, cam_vinv,
                                                    cam_proj, self.sensor_u2, self.sensor_v2,
                                                    self.sensor_width, self.sensor_height, 0.1, self.device).contiguous()
                    if points.numel() != 0:
                        points = self.sample_points(points, sample_num=self.sensor_downsample_num, sample_mathed='random')
                        # 存储points pair
                        start_index = index * self.sensor_downsample_num
                        sensors_point_clouds[i, start_index:start_index + self.sensor_downsample_num, :] = points
            #print(point_clouds.size())
       #compute_tactile
       
        if self.camera_view_debug:
            if self.debug_view_type == "camera" and "pointcloud" in self.obs_type:
                self.camera_window = plt.figure("CAMERA_DEBUG")
                camera_rgba_image = self.camera_visulization(camera = self.cameras[0], is_depth_image=False)
                plt.imshow(camera_rgba_image)
                plt.pause(1e-9)
                plt.cla()
            

            elif self.debug_view_type == "sensor" and "tactile" in self.obs_type:
                self.camera_window = plt.figure("SENSOR_DEBUG")
                sensor_rgba_image = self.camera_visulization(camera = self.sensors[0], is_depth_image=True)
                plt.imshow(sensor_rgba_image)
                plt.pause(1e-9)      
                plt.cla()
            else:
                print("obs_type error!")

        if self.pointCloudVisualizer != None :
            import open3d as o3d
            #points = point_clouds[0, :, :3].cpu().numpy()
            #point_clouds[0, :,2]=1
            points = np.concatenate(( point_clouds[0, :, :3].cpu().numpy() , sensors_point_clouds[0, :, :3].cpu().numpy()) , axis=0)
            if(sensors_point_clouds != None):
                lf_x_result = np.sum(sensors_point_clouds[0, :64, 0].cpu().numpy())/64.0
                lf_y_result = np.sum(sensors_point_clouds[0, :64, 1].cpu().numpy())/64.0
                lf_z_result = np.sum(sensors_point_clouds[0, :64, 2].cpu().numpy())/64.0
                rf_x_result = np.sum(sensors_point_clouds[0, 64:128, 0].cpu().numpy())/64.0
                rf_y_result = np.sum(sensors_point_clouds[0, 64:128, 1].cpu().numpy())/64.0
                rf_z_result = np.sum(sensors_point_clouds[0, 64:128, 2].cpu().numpy())/64.0
                #print("lf:",lf_x_result,lf_y_result,lf_z_result,"rf:",rf_x_result,rf_y_result,rf_z_result)
            #points = sensors_point_clouds[0, :, :3].cpu().numpy()
            #print(sensors_point_clouds[0, :, :3].cpu().numpy())
            
            #colors = plt.get_cmap(points)
            self.o3d_pc.points = o3d.utility.Vector3dVector(points)
            #self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])

            # np_sensors_point_clouds = sensors_point_clouds.cpu().numpy()
            # # 颜色该为灰色
            # if len(np_sensors_point_clouds) != 0:
            #     depth = np_sensors_point_clouds[0, :, -1]
            #     # 深度归一化到 0.2 ---> 0.8
            #     np_colors = (depth - depth.min()) / (depth.max() - depth.min()) * 0.6 + 0.2
            #     self.sensor_o3d_pc.points = o3d.utility.Vector3dVector(np_sensors_point_clouds[0, :, :])
            #     colors = np.zeros([np_sensors_point_clouds[0, :, :].shape[0], 3])
            #     for _ in range(3):
            #         colors[:, _] = np_colors
            # self.sensor_o3d_pc.colors = o3d.utility.Vector3dVector(colors)

            if self.pointCloudVisualizerInitialized == False :
                self.pointCloudVisualizer.add_geometry(self.o3d_pc)
                self.pointCloudVisualizerInitialized = True
            else :
                self.pointCloudVisualizer.update(self.o3d_pc)

        self.gym.end_access_image_tensors(self.sim)

        point_clouds -= self.env_origin.view(self.num_envs, 1, 3)
        self.pointcloud_flatten = point_clouds.view(self.num_envs, self.pointCloudDownsampleNum * 3)

        sensors_point_clouds -= self.env_origin.view(self.num_envs, 1, 3)

        self.sensor_pointcloud_flatten = sensors_point_clouds.view(self.num_envs, 2 * self.sensor_downsample_num * 3)

        self.all_pointcloud_flatten[:,:self.pointCloudDownsampleNum * 3] = self.pointcloud_flatten
        self.all_pointcloud_flatten[:,self.pointCloudDownsampleNum * 3:] = self.sensor_pointcloud_flatten

        #self.pc_data = point_clouds.view(self.num_envs , self.pointCloudDownsampleNum,  3)
        

    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]
    
    def sample_points(self, points, sample_num=1000, sample_mathed='random'):
        eff_points = points[points[:, 2]>0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

    def camera_visulization(self, camera, is_depth_image=False):
        
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], camera, gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            
            
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -0.15, -0.)

            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                        to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], camera, gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)           
        
        return camera_image
       
    def safety_check(self, actions, eef_state, osc_limit):
        force = self.fs_tensor.view(self.num_envs,self.num_force_sensor,6)
        print("force:",force[0,0,:3] * 1000)

        result = eef_state + actions
        exceed_limit = torch.any(result > osc_limit[1]) or torch.any(result < osc_limit[0])
        if exceed_limit:
            clamped_result = torch.clamp(result, osc_limit[0], osc_limit[1])
            adjusted_action = clamped_result - eef_state
            actions = adjusted_action

        return actions


    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        if len(env_ids) > 0:
            self.reset(env_ids)
        # apply action
        #actions = torch.zeros_like(actions,device=self.device)  #test

        actions = actions * self.cmd_limit / self.action_scale 
        #print(ee_end_state)
        actions[:,:3] = self.safety_check(actions[:,:3], self.states["middle_gripper_state"], self.osc_limits)
         
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs,self.num_dof,2)
        dof_pos = dof_states[:,:,0]
        #check = dof_pos.clone()

        
        u_delta = control_ik(self._j_eef, actions[:,:6].unsqueeze(-1), self.num_envs, num_dofs=self.num_dof)
        u_delta = actuate(self.actuator_joints, self.mimic_joints, self.arm_dof, u_delta, actions[:,self.arm_dof:])
        
        
        check = (u_delta + dof_pos).clone()
        u_offset = position_check(self.actuator_joints, self.mimic_joints, self.arm_dof, check)
        
        #print(u_offset)

        self._pos_control = (u_delta + dof_pos + u_offset)
        
        self._pos_control = torch.clamp(self._pos_control, min=self.all_limits[0],max=self.all_limits[1])

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))



    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward()
        
    def _random_cubeA_state(self, _init_cubeA_state, env_ids):
        #_init_cubeA_state [num_envs, 13]

        centered_cube_xy_state = torch.tensor([0.30, 0.365, 0.83], device=self.device, dtype=torch.float32)
        _init_cubeA_state[env_ids, 2] = self.table_stand_height + self.cubeA_size / 2
        _init_cubeA_state[env_ids, :2] = centered_cube_xy_state[0:2] + 2.0 * self.start_position_noise * \
                                        (torch.rand(len(env_ids), 2, device=self.device) - 0.5)

        # Sample rotation value

        aa_rot = torch.zeros(len(env_ids), 3, device=self.device)
        aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(len(env_ids), device=self.device) - 0.5)
        _init_cubeA_state[env_ids, 3:7] = quat_mul(axisangle2quat(aa_rot), _init_cubeA_state[env_ids, 3:7])
        return _init_cubeA_state
    
    def _random_goal_state(self, init_goal_state):
        centered_goal_xy_state = torch.tensor([0.30, 0.365, 1.36], device=self.device, dtype=torch.float32)
        init_goal_state[:, 2] = self.table_stand_height + self.cubeA_size / 2 + 0.5
        init_goal_state[:, :2] = centered_goal_xy_state[0:2] + 0.5 * self.start_position_noise * \
                                        (torch.rand(self.num_envs, 2, device=self.device) - 0.5)
        return init_goal_state


   

# define reward function using JIT
@torch.jit.script
def control_ik(j_eef, dpose, num_envs, num_dofs, damping=0.05):
    """Solve damped least squares, from `franka_cube_ik_osc.py` in Isaac Gym.

    Returns: Change in DOF positions, [num_envs,num_dofs], to add to current positions.
    """
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6).to(j_eef_T.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, num_dofs)
    return u

@torch.jit.script
def compute_reach_reward(reset_buf, progress_buf, states, max_episode_length):

    # type: (Tensor, Tensor, Dict[str, Tensor], float) -> Tuple[Tensor, Tensor, Tensor]
    
    d_lf = torch.norm(states["cube_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["cube_pos"] - states["eef_rf_pos"], dim=-1)
    d_ff = torch.norm(states["eef_lf_pos"] - states["eef_rf_pos"], dim=-1)

    #goal_reward = torch.norm(states["goal_pos"] - states["cube_pos"], dim=-1)
    #print(d_lf)
    #print(d_rf)
    # reward for lifting cubeA
    cubeA_height = states["cube_pos"][:, 2] - 0.83
    cubeA_lifted = (cubeA_height - 0.03) > 0.045
    cubeA_droped = (cubeA_height - 0.03) < -0.01
    success_buf = cubeA_lifted


    rew_buf = 1 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) + cubeA_lifted * cubeA_height * 20

    #reset_buf = torch.where((progress_buf >= (max_episode_length - 1)) | (rewards > 0.8), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((progress_buf >= (max_episode_length - 1)) | (cubeA_droped), torch.ones_like(reset_buf), reset_buf)
    return rew_buf, reset_buf, success_buf

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

@torch.jit.script
def sensor_depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width: float,
                                   height: float, depth_bar: float, device: torch.device):
    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y = (v - centerV) / height * Z * fv

    Z = Z.view(-1)

    valid = ((-(0.022) > Z) & (Z > -(0.028)))
    # valid = (Z > -0.1)

    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position @ vinv

    points = position[:, 0:3]

    return points

import rospy
from std_msgs.msg import String, Float32MultiArray, Float64MultiArray
from sensor_msgs.msg import JointState, Image


from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
import torch

from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

import cv2

from pynput.keyboard import Key, Listener

from scipy.spatial.transform import Rotation as R

FREQ = 30# Hz
PHYSICS_FREQ = 30#Hz
PHYSICS_SUBSTEPS = 20#4
CAM_UPDATE_FREQ = 30 #Hz
REFRESH_RATE = 20 #s


'''
subscribe from topic
/data/action

publish to topic
data/left_wrist_image
data/right_wrist_image
data/left_eye_image
data/right_eye_image
data/past_action
data/joint_position
data/joint_velocity

'''


class Sim:
    def __init__(self,
                 print_freq=False):
        #ROS
        rospy.init_node('teleop', anonymous=True)
        self.rate = rospy.Rate(PHYSICS_FREQ)
        self.joint_command = JointState()
        self.joint_command.header.stamp = rospy.Time.now()
        self.joint_command.name = [
            "FoldingModularJoint02_Joint",
            "FoldingModularJoint03_Joint",
            "Trunk_Joint",
            "ArmL02_Joint",
            "ArmL03_Joint",
            "ArmL04_Joint",
            "ArmL05_Joint",
            "ArmL06_Joint",
            "ArmL07_Joint",
            "ArmL08_Joint",            
            "JawBlock01_Joint",
            "JawBlock02_Joint",
            "ArmR02_Joint",
            "ArmR03_Joint",
            "ArmR04_Joint",
            "ArmR05_Joint",
            "ArmR06_Joint",
            "ArmR07_Joint",
            "ArmR08_Joint",
            "JawBlock03_Joint",
            "JawBlock04_Joint",
            "Head02_Joint",
            "Head03_Joint",
        ]

        # publish data for training
        self.left_wrist_image_pub = rospy.Publisher('data/left_wrist_image', Image, queue_size=10)
        self.right_wrist_image_pub = rospy.Publisher('data/right_wrist_image', Image, queue_size=10)
        self.left_eye_image_pub = rospy.Publisher('data/left_eye_image', Image, queue_size=10)
        self.right_eye_image_pub = rospy.Publisher('data/right_eye_image', Image, queue_size=10)
        self.past_action_pub = rospy.Publisher('data/past_action', Float64MultiArray, queue_size=10)
        self.joint_position_pub = rospy.Publisher('data/joint_position', Float64MultiArray, queue_size=10)
        self.joint_velocity_pub = rospy.Publisher('data/joint_velocity', Float64MultiArray, queue_size=10)

        self.active_dof_num = 23 #3+7+7+2+2+2
        self.total_dof_num = 33
        self.past_robot_q_pos = [0]*self.total_dof_num
        self.upper_body_command_num = 21 #3+7+7+1+1+2
        self.action = np.zeros(self.active_dof_num)
        self.past_action = np.zeros(self.active_dof_num)
        self.joint_position = np.zeros(self.active_dof_num)
        self.joint_velocity = np.zeros(self.active_dof_num)

        self.eye_image_resolution = np.array([360, 640])
        self.wrist_image_resolution = np.array([360, 640])
        self.left_wrist_image = np.zeros(self.wrist_image_resolution)
        self.right_wrist_image = np.zeros(self.wrist_image_resolution)
        self.left_eye_image = np.zeros(self.eye_image_resolution)
        self.right_eye_image = np.zeros(self.eye_image_resolution)
        
        rospy.Subscriber("data/action", Float64MultiArray, self.action_callback)

        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / PHYSICS_FREQ
        sim_params.substeps = PHYSICS_SUBSTEPS
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.static_friction = 0.2 #1
        plane_params.dynamic_friction = 0.2 #1
        plane_params.restitution = 0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # # load cube1 asset
        # cube_asset_options = gymapi.AssetOptions()
        # cube_asset_options.density = 10
        # cube_asset1 = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        # # load cube2 asset
        # cube_asset_options = gymapi.AssetOptions()
        # cube_asset_options.density = 10
        # cube_asset2 = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)
        # load cube assets from URDF
        asset_root = "assets"
        cube_path = "cube.urdf"
        cube_asset_options = gymapi.AssetOptions()
        cube_asset1 = self.gym.load_asset(self.sim, asset_root, cube_path, cube_asset_options)
        cube_asset2 = self.gym.load_asset(self.sim, asset_root, cube_path, cube_asset_options)
        # cube_asset3 = self.gym.load_asset(self.sim, asset_root, cube_path, cube_asset_options)

        # load sphere asset
        sphere_asset_options = gymapi.AssetOptions()
        sphere_asset_options.density = 5.0  # 设置密度
        sphere_radius = 0.03
        sphere_asset = self.gym.create_sphere(self.sim, sphere_radius, sphere_asset_options)


        bucket_path = "bucket/bucket.urdf"

        bucket_asset_options = gymapi.AssetOptions()
        bucket_asset_options.disable_gravity = False
        bucket_asset_options.fix_base_link = True
        bucket_asset_options.collapse_fixed_joints = True
        bucket_asset_options.vhacd_enabled = True
        bucket_asset_options.vhacd_params = gymapi.VhacdParams()
        bucket_asset_options.vhacd_params.resolution = 500000
        bucket_asset_options.vhacd_params.max_num_vertices_per_ch = 32
        bucket_asset_options.vhacd_params.min_volume_per_ch = 0.001
        bucket_asset = self.gym.load_asset(self.sim, asset_root, bucket_path, bucket_asset_options)
        

        robot_asset_path = "humanoid/BRX042501/BRX042501_wheel.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = 1 #position target #gymapi.DOF_MODE_POS
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_path, asset_options)


        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed()#np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.3, 0, 2.15)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube1
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.55, 0, 2.3)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cube_handle1 = self.gym.create_actor(self.env, cube_asset1, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, self.cube_handle1, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        # Get the rigid shape properties
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, self.cube_handle1)
        # Set friction for all shapes (assuming a single shape here)
        for prop in shape_props:
            prop.friction = 1.0  # Desired friction coefficient        

        # cube2
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.55, 0.2, 2.3)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cube_handle2 = self.gym.create_actor(self.env, cube_asset2, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, self.cube_handle2, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        # Get the rigid shape properties
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, self.cube_handle2)
        # Set friction for all shapes (assuming a single shape here)
        for prop in shape_props:
            prop.friction = 1.0  # Desired friction coefficient     

        # # cube3
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(-0.55, -0.2, 2.3)
        # pose.r = gymapi.Quat(0, 0, 0, 1)
        # cube_handle3 = self.gym.create_actor(self.env, cube_asset3, pose, 'cube3', 0)
        # color = gymapi.Vec3(1, 0.5, 0.5)
        # self.gym.set_rigid_body_color(self.env, cube_handle3, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        #  # sphere
        # sphere_pose = gymapi.Transform()
        # sphere_pose.p = gymapi.Vec3(-0.55, -0.2, 2.3)  # 放在桌子中心上方
        # sphere_pose.r = gymapi.Quat(0, 0, 0, 1)  # 初始朝向
        # sphere_handle = self.gym.create_actor(self.env, sphere_asset, sphere_pose, 'sphere', 0)
        # color = gymapi.Vec3(1, 1, 0)
        # self.gym.set_rigid_body_color(self.env, sphere_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # bucket
        bucket_pose = gymapi.Transform()
        bucket_pose.p = gymapi.Vec3(-0.3, 0, 2.2)  
        bucket_pose.r = gymapi.Quat(0, 0, 0, 1)
        bucket_handle = self.gym.create_actor(self.env, bucket_asset, bucket_pose, 'bucket', 0)
        color = gymapi.Vec3(0.7, 0.7, 1)
        self.gym.set_rigid_body_color(self.env, bucket_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)  
        # self.bucket_pose = gymapi.Transform()
        # self.bucket_pose.p = gymapi.Vec3()
        # self.bucket_pose.p.x = arm_pose.p.x - 0.6
        # self.bucket_pose.p.y = arm_pose.p.y - 1
        # self.bucket_pose.p.z = arm_pose.p.z + 0.45

        # bucket_rb_count = self.gym.get_asset_rigid_body_count(bucket_asset)
        # bucket_shapes_count = self.gym.get_asset_rigid_shape_count(bucket_asset)


        # robot
        pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(-1.0, 0, 1.6) #1.6
        pose.p = gymapi.Vec3(-1.1, 0, 1.6) #1.6
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.robot_handle = self.gym.create_actor(self.env, robot_asset, pose, 'robot', 0, 1)
        self.gym.set_actor_dof_states(self.env, self.robot_handle, np.zeros(self.gym.get_asset_dof_count(robot_asset), gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        robot_idx = self.gym.get_actor_index(self.env, self.robot_handle, gymapi.DOMAIN_SIM)
        

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)


        # Get DOF names and properties
        dof_names = self.gym.get_actor_dof_names(self.env, self.robot_handle)
        dof_props = self.gym.get_actor_dof_properties(self.env, self.robot_handle)

        print(dof_names)
        print(dof_props["stiffness"])
        print(dof_props["damping"])

        '''
        [
        0 'BLWheel01_Joint', 
        1 'BLWheel02_Joint', 
        2 'BRWheel01_Joint', 
        3 'BRWheel02_Joint', 
        4 'FLWheel01_Joint', 
        5 'FLWheel02_Joint', 
        6 'FRWheel01_Joint', 
        7 'FRWheel02_Joint', 
        8 'FoldingModularJoint02_Joint', 
        9 'FoldingModularJoint03_Joint', 
        10'Trunk_Joint', 
        11 'ArmL02_Joint', 
        12 'ArmL03_Joint', 
        13 'ArmL04_Joint', 
        14 'ArmL05_Joint', 
        15 'ArmL06_Joint', 
        16 'ArmL07_Joint', 
        17 'ArmL08_Joint', 
        18 'JawBlock03_Joint', 
        19 'JawBlock04_Joint', 
        20 'ArmR02_Joint', 
        21 'ArmR03_Joint', 
        22 'ArmR04_Joint', 
        23 'ArmR05_Joint', 
        24 'ArmR06_Joint', 
        25 'ArmR07_Joint', 
        26 'ArmR08_Joint', 
        27 'JawBlock01_Joint', 
        28 'JawBlock02_Joint', 
        29 'Head02_Joint', 
        30 'Head03_Joint', 
        31 'WheelL_Joint', 
        32 'WheelR_Joint'
        ]
        '''

        # Modify DOF properties for each joint
        for i, name in enumerate(dof_names):
            if i < 8:
                mode = gymapi.DOF_MODE_EFFORT
                dof_props["driveMode"][i] = mode
            elif i >= 8 and i < 11:
                mode = gymapi.DOF_MODE_POS
                dof_props["driveMode"][i] = mode
                # use default setting
                # dof_props["stiffness"][i] = 20000.0  # Proportional gain
                # dof_props["damping"][i] = 2000.0     # Derivative gain
            elif i >= 11 and i < 18:
                mode = gymapi.DOF_MODE_POS
                dof_props["driveMode"][i] = mode
                dof_props["stiffness"][i] = 2000.0  # Proportional gain
                dof_props["damping"][i] = 200.0     # Derivative gain
            elif i >= 18 and i < 20:
                mode = gymapi.DOF_MODE_POS
                dof_props["driveMode"][i] = mode
                dof_props["stiffness"][i] = 2000.0  # Proportional gain
                dof_props["damping"][i] = 200.0     # Derivative gain
            elif i >= 20 and i < 27:
                mode = gymapi.DOF_MODE_POS
                dof_props["driveMode"][i] = mode
                dof_props["stiffness"][i] = 2000.0  # Proportional gain
                dof_props["damping"][i] = 200.0     # Derivative gain
            elif i >= 27 and i < 31:
                mode = gymapi.DOF_MODE_POS
                dof_props["driveMode"][i] = mode
                dof_props["stiffness"][i] = 2000.0  # Proportional gain
                dof_props["damping"][i] = 200.0     # Derivative gain
            elif i>=31 and i < 33:
                mode = gymapi.DOF_MODE_VEL
                dof_props["driveMode"][i] = mode
                # dof_props["stiffness"][i] = 0.0  # Proportional gain
                dof_props["damping"][i] = 200.0     # Derivative gain

            # if name in drive_modes:
            #     mode = drive_modes[name]
            #     dof_props["driveMode"][i] = mode
                
            #     # Set PD gains for position/velocity control
            #     if mode == gymapi.DOF_MODE_POS:
            #         dof_props["stiffness"][i] = 1000.0  # Proportional gain
            #         dof_props["damping"][i] = 100.0     # Derivative gain
            #     elif mode == gymapi.DOF_MODE_VEL:
            #         dof_props["damping"][i] = 50.0      # Velocity tracking gain

        # Apply modified DOF properties to the robot
        self.gym.set_actor_dof_properties(self.env, self.robot_handle, dof_props)

        # self.left_root_states = self.root_states[left_idx]
        # self.right_root_states = self.root_states[right_idx]

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        # self.left_cam_offset = np.array([0, 0.033, 0])
        # self.right_cam_offset = np.array([0, -0.033, 0])
        self.left_cam_offset = np.array([0.01, 0.033, 0])
        self.right_cam_offset = np.array([0.01, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280//2 #1280
        camera_props.height = 720//2 #720
        camera_props.horizontal_fov = 90.0 #125.0
        print(camera_props.horizontal_fov, "horizontal_fov")
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        # self.gym.set_camera_location(self.left_camera_handle,
        #                              self.env,
        #                              gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
        #                              gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*self.cam_pos)
        transform.r = gymapi.Quat.from_euler_zyx(0,0,0)
        self.gym.set_camera_transform(self.left_camera_handle, self.env, transform)
        # body_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_handle, "EyeL_Link")
        # self.gym.attach_camera_to_body(self.left_camera_handle, self.env, body_handle, transform, gymapi.FOLLOW_TRANSFORM)

        # local_transform = gymapi.Transform()
        # local_transform.p = (x,y,z)
        # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
        # gym.attach_camera_to_body(camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280//2 #1280
        camera_props.height = 720//2 #720
        camera_props.horizontal_fov = 90.0 #125.0
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        # self.gym.set_camera_location(self.right_camera_handle,
        #                              self.env,
        #                              gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
        #                              gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*self.cam_pos)
        transform.r = gymapi.Quat.from_euler_zyx(0,0,0)
        self.gym.set_camera_transform(self.right_camera_handle, self.env, transform)
        # body_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_handle, "EyeR_Link")
        # self.gym.attach_camera_to_body(self.right_camera_handle, self.env, body_handle, transform, gymapi.FOLLOW_TRANSFORM)


        self.left_image = np.zeros((720,1280,3),dtype=np.uint8)
        self.right_image = np.zeros((720,1280,3),dtype=np.uint8)

        # create left wrist camera viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = int(1280//4)
        camera_props.height = int(720//4)
        camera_props.horizontal_fov = 90.0
        self.left_wrist_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        # self.gym.set_camera_location(self.left_wrist_camera_handle,
        #                              self.env,
        #                              gymapi.Vec3(*(self.cam_pos)),
        #                              gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset)))        
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*self.cam_pos)
        transform.r = gymapi.Quat.from_euler_zyx(0,0,0)
        self.gym.set_camera_transform(self.left_wrist_camera_handle, self.env, transform)
        # body_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_handle, "HandCam02_Link")
        # self.gym.attach_camera_to_body(self.left_wrist_camera_handle, self.env, body_handle, transform, gymapi.FOLLOW_TRANSFORM)

        # create right wrist camera viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = int(1280//4)
        camera_props.height = int(720//4)
        camera_props.horizontal_fov = 90.0
        self.right_wrist_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        # self.gym.set_camera_location(self.right_wrist_camera_handle,
        #                              self.env,
        #                              gymapi.Vec3(*(self.cam_pos)),
        #                              gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset)))     
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*self.cam_pos)
        transform.r = gymapi.Quat.from_euler_zyx(0,0,0)
        self.gym.set_camera_transform(self.right_wrist_camera_handle, self.env, transform)
        # body_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_handle, "HandCam01_Link")
        # self.gym.attach_camera_to_body(self.right_wrist_camera_handle, self.env, body_handle, transform, gymapi.FOLLOW_TRANSFORM)


        self.counter = 0

        self.control_flag = False
        self.key_press = "n"
        # Start listening for key presses in non blocking fashion
        self.listener = Listener(
            on_press=self.on_press)
        self.listener.start()


    def on_press(self, key):
        try:
            key_char = key.char.lower()  # Get lowercase character
            self.key_press = key_char
            if key_char == 'y':
                print("'Y' was pressed")
                # set control flag to true, start receiving commands
                self.control_flag = True
            elif key_char == 'n':
                print("'N' was pressed")
                # set control flag to false, maintain previous pose
                self.control_flag = False
        except AttributeError:
            pass  # Ignore special keys (e.g., Shift, Ctrl)



    def step(self):
        if self.counter%(PHYSICS_FREQ*REFRESH_RATE) == 0: #reset cube color and location
            #obtain root states of all actors
            root_states_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            root_states = gymtorch.wrap_tensor(root_states_tensor)

            # cube1
            pose = [-0.55+np.random.uniform(-0.1, 0.1), 0+np.random.uniform(-0.05, 0.05), 2.3]
            orn = [0, 0, 0, 1]
            color = gymapi.Vec3(np.random.rand(1), np.random.rand(1), np.random.rand(1))
            self.gym.set_rigid_body_color(self.env, self.cube_handle1, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            object_idx = self.gym.get_actor_index(self.env, self.cube_handle1, gymapi.DOMAIN_SIM)
            root_states[object_idx, 0:3] = torch.tensor(pose, device=root_states.device)
            root_states[object_idx, 3:7] = torch.tensor(orn, device=root_states.device)
            root_states[object_idx, 7:13] = 0  # Reset velocities (linear + angular)


            # cube2
            pose = [-0.55+np.random.uniform(-0.1, 0.1), 0.2+np.random.uniform(-0.05, 0.05), 2.3]
            orn = [0, 0, 0, 1]
            color = gymapi.Vec3(np.random.rand(1), np.random.rand(1), np.random.rand(1))
            self.gym.set_rigid_body_color(self.env, self.cube_handle2, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)      
            object_idx = self.gym.get_actor_index(self.env, self.cube_handle2, gymapi.DOMAIN_SIM)
            root_states[object_idx, 0:3] = torch.tensor(pose, device=root_states.device)
            root_states[object_idx, 3:7] = torch.tensor(orn, device=root_states.device)
            root_states[object_idx, 7:13] = 0  # Reset velocities (linear + angular)      

            # Push changes back to simulator
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))   

        if self.print_freq:
            start = time.time()


        if self.control_flag == False:
                # do not update control when control flag is false
            robot_qpos = [0]*self.total_dof_num
            robot_qpos[:] = self.past_robot_q_pos[:]
        else:
            # pos control
            robot_qpos = [0]*self.total_dof_num
            robot_qpos[8:8+self.active_dof_num] = self.action[:]

            # update past robot_qpos
            self.past_robot_q_pos[:] = robot_qpos[:]


        # Joint related datas
        robot_dof_state = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
        for i in range(self.active_dof_num):
            self.joint_position[i] = robot_dof_state[i+8][0]
            self.joint_velocity[i] = robot_dof_state[i+8][1]
        self.past_action = np.copy(self.action)

        # publish joint related data
        joint_position_msg = Float64MultiArray(data=self.joint_position)
        joint_velocity_msg = Float64MultiArray(data=self.joint_velocity)
        past_action_msg = Float64MultiArray(data=self.past_action)
        self.joint_position_pub.publish(joint_position_msg)
        self.joint_velocity_pub.publish(joint_velocity_msg)
        self.past_action_pub.publish(past_action_msg)

        # vel control
        vel_target = [0]*self.total_dof_num
        vel_target[31] = 3.14
        vel_target[32] = -3.14
        # effort control
        effort_target = [0]*self.total_dof_num

        # self.gym.set_actor_dof_states(self.env, self.robot_handle, robot_states, gymapi.STATE_POS) #TODO check gripper issue when this line is commented out
        
        self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, robot_qpos)
        self.gym.set_actor_dof_velocity_targets(self.env, self.robot_handle, vel_target)
        self.gym.apply_actor_dof_efforts(self.env, self.robot_handle, effort_target)
        robot_state = self.gym.get_actor_rigid_body_states(self.env, self.robot_handle, gymapi.STATE_POS)
        

        robot_dof_state = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_POS)

        time_physics_start = time.time()
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        #set camera location before stepping physics
        if self.counter % int(PHYSICS_FREQ/CAM_UPDATE_FREQ) == 0:
            robot_state = self.gym.get_actor_rigid_body_states(self.env, self.robot_handle, gymapi.STATE_POS)

            left_eye_camera_link = robot_state[40] #EyeL_Link
            left_eye_camera_quat = left_eye_camera_link[0][1]
            left_eye_camera_pos =  left_eye_camera_link[0][0]     
            r = R.from_quat([left_eye_camera_quat[0], left_eye_camera_quat[1], left_eye_camera_quat[2], left_eye_camera_quat[3]])
            left_eye_rmat = r.as_matrix()
            self.left_eye_cam_pos = np.array([left_eye_camera_pos[0], left_eye_camera_pos[1], left_eye_camera_pos[2]])

            right_eye_camera_link = robot_state[42] #EyeR_Link
            right_eye_camera_quat = right_eye_camera_link[0][1]
            right_eye_camera_pos =  right_eye_camera_link[0][0]     
            r = R.from_quat([right_eye_camera_quat[0], right_eye_camera_quat[1], right_eye_camera_quat[2], right_eye_camera_quat[3]])
            right_eye_rmat = r.as_matrix()
            self.right_eye_cam_pos = np.array([right_eye_camera_pos[0], right_eye_camera_pos[1], right_eye_camera_pos[2]])

            left_wrist_camera_link = robot_state[22] #HeadCam02_Link
            left_wrist_camera_quat = left_wrist_camera_link[0][1]
            left_wrist_camera_pos =  left_wrist_camera_link[0][0]     
            r = R.from_quat([left_wrist_camera_quat[0], left_wrist_camera_quat[1], left_wrist_camera_quat[2], left_wrist_camera_quat[3]])
            left_wrist_rmat = r.as_matrix()
            self.left_wrist_cam_pos = np.array([left_wrist_camera_pos[0], left_wrist_camera_pos[1], left_wrist_camera_pos[2]])

            right_wrist_camera_link = robot_state[34] #HeadCam01_Link
            right_wrist_camera_quat = right_wrist_camera_link[0][1]
            right_wrist_camera_pos =  right_wrist_camera_link[0][0]     
            r = R.from_quat([right_wrist_camera_quat[0], right_wrist_camera_quat[1], right_wrist_camera_quat[2], right_wrist_camera_quat[3]])
            right_wrist_rmat = r.as_matrix()
            self.right_wrist_cam_pos = np.array([right_wrist_camera_pos[0], right_wrist_camera_pos[1], right_wrist_camera_pos[2]])

            transform = gymapi.Transform()
            transform.p = left_eye_camera_pos
            transform.r = left_eye_camera_quat
            self.gym.set_camera_transform(self.left_camera_handle, self.env, transform)

            transform = gymapi.Transform()
            transform.p = right_eye_camera_pos
            transform.r = right_eye_camera_quat
            self.gym.set_camera_transform(self.right_camera_handle, self.env, transform)

            transform = gymapi.Transform()
            transform.p = left_wrist_camera_pos
            transform.r = left_wrist_camera_quat
            self.gym.set_camera_transform(self.left_wrist_camera_handle, self.env, transform)

            transform = gymapi.Transform()
            transform.p = right_wrist_camera_pos
            transform.r = right_wrist_camera_quat
            self.gym.set_camera_transform(self.right_wrist_camera_handle, self.env, transform)


        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        time_physics_end = time.time()
        print("time physics time", time_physics_end-time_physics_start)

        # retrive image after simulation steps
        if self.counter % int(PHYSICS_FREQ/CAM_UPDATE_FREQ) == 0:                      
            get_image_start = time.time()
            left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
            right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
            left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
            right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]
            self.left_image = left_image
            self.right_image = right_image


            left_wrist_image = self.gym.get_camera_image(self.sim, self.env, self.left_wrist_camera_handle, gymapi.IMAGE_COLOR)
            left_wrist_image = left_wrist_image.reshape(left_wrist_image.shape[0], -1, 4)[..., :3]


            right_wrist_image = self.gym.get_camera_image(self.sim, self.env, self.right_wrist_camera_handle, gymapi.IMAGE_COLOR)
            right_wrist_image = right_wrist_image.reshape(right_wrist_image.shape[0], -1, 4)[..., :3]
            get_image_end = time.time()
            print("get image time", get_image_end-get_image_start)

            image_ros_start = time.time()
            self.left_eye_image = cv2.resize(left_image, (1280//2, 720//2)) #(horizontal vertical)
            self.right_eye_image = cv2.resize(right_image, (1280//2, 720//2))
            self.left_wrist_image = cv2.resize(left_wrist_image, (1280//4, 720//4))
            self.right_wrist_image = cv2.resize(right_wrist_image, (1280//4, 720//4))


            def cv_to_ros(cv_image, encoding='bgr8'):
                msg = Image()
                msg.height = cv_image.shape[0]
                msg.width = cv_image.shape[1]
                msg.encoding = encoding
                msg.step = int(cv_image.strides[0])
                msg.data = cv_image.tobytes()
                return msg

            left_eye_image_msg = cv_to_ros(self.left_eye_image, 'rgb8')
            right_eye_image_msg = cv_to_ros(self.right_eye_image, 'rgb8')
            left_wrist_image_msg = cv_to_ros(self.left_wrist_image, 'rgb8')
            right_wrist_image_msg = cv_to_ros(self.right_wrist_image, 'rgb8')
            stamp = rospy.Time.now()
            left_eye_image_msg.header.stamp = stamp
            right_eye_image_msg.header.stamp = stamp
            left_wrist_image_msg.header.stamp = stamp
            right_wrist_image_msg.header.stamp = stamp
            self.left_eye_image_pub.publish(left_eye_image_msg)
            self.right_eye_image_pub.publish(right_eye_image_msg)
            self.left_wrist_image_pub.publish(left_wrist_image_msg)
            self.right_wrist_image_pub.publish(right_wrist_image_msg)
            image_ros_end = time.time()
            print("image ros time", image_ros_end-image_ros_start)

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)


        self.rate.sleep()
        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        self.counter = self.counter+1

        return self.left_image, self.right_image

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def action_callback(self, data):
        self.action = data.data


if __name__ == '__main__':

    simulator = Sim()

    try:

        counter = 0
        while True:
            
            left_img, right_img = simulator.step()

            counter = counter+1

            if simulator.key_press == "c":
                break

    except KeyboardInterrupt:
        simulator.end()
        exit(0)

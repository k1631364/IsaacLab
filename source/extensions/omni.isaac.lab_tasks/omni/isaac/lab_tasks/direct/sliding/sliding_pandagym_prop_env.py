# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg,DirectRLEnvFeedback
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns

from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

import numpy as np
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import pickle
from sklearn.neighbors import NearestNeighbors

import source.offline_learning.model.RNNPropertyEstimator as rnnmodel

def normalize(tensor, min_val, max_val, new_min, new_max):
    return (tensor - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

def denormalize(tensor, min_val, max_val, new_min, new_max):
    return (tensor - new_min) / (new_max - new_min) * (max_val - min_val) + min_val

@configclass
class EventCfg:
  robot_physics_material = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("cylinderpuck2"),
          "static_friction_range": (0.05, 0.05),
          "dynamic_friction_range": (0.05, 0.3),
          "restitution_range": (1.0, 1.0),  # (1.0, 1.0),  
        #   "com_rad": 0.032, 
        #   "com_range_x": (-0.01, 0.01), # (-0.02, 0.02),
        #   "com_range_y": (-0.01, 0.01), # (-0.02, 0.02),
        #   "com_range_z": (0.0, 0.0), 
          "mass_range": (0.15, 0.15), 
          "num_buckets": 250, 
      },
  )

#   @configclass
#   class EventCfg:
#     robot_physics_material_table = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("cuboidtable2"),
#             "static_friction_range": (0.5, 0.5),
#           "dynamic_friction_range": (0.5, 0.5),
#           "restitution_range": (0.6, 0.6),
#             "mass_range": (30.0, 30.0),
#             "num_buckets": 250,
#         },
#     )

# class EventCfg:
#   robot_physics_material = EventTerm(
#       func=mdp.randomize_rigid_body_material,
#       mode="reset",
#       params={
#           "asset_cfg": SceneEntityCfg("cylinderpuck2"),
#           "static_friction_range": (0.05, 0.05),
#           "dynamic_friction_range": (0.05, 0.3),
#           "restitution_range": (0.4, 0.4),  # (1.0, 1.0),  
#           "com_range_x": (-0.01, 0.01), # (-0.02, 0.02),
#           "com_range_y": (-0.01, 0.01), # (-0.02, 0.02),
#           "com_range_z": (0.0, 0.0), 
#           "mass_range": (0.15, 0.15),
#           "num_buckets": 250,
#       },
#   )

#   robot_physics_material2 = EventTerm(
#       func=mdp.randomize_rigid_body_material,
#       mode="reset",
#       params={
#           "asset_cfg": SceneEntityCfg("cuboidpusher2"),
#           "static_friction_range": (0.5, 0.5),
#           "dynamic_friction_range": (0.5, 0.5),
#           "restitution_range": (0.4, 0.6),
#         #   "com_range_x": (-0.00, 0.00), # (-0.02, 0.02),
#         #   "com_range_y": (-0.00, 0.00), # (-0.02, 0.02),
#         #   "com_range_z": (0.0, 0.0), 
#           "mass_range": (0.5, 3.0),
#           "num_buckets": 250,
#       },
#   )
  
  # Default material proeprties tensor([[[0.5000, 0.5000, 0.0000]]])
  # Juan's pushing randomisation: distribution_parameters: [[0.5, 0.2, 0.4], [0.7, 0.4, 0.6]]

@configclass
class SlidingPandaGymPropEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    events: EventCfg = EventCfg()

    # Noise
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    # )
    # # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )

    obs_pos_noise_mean = 0.0
    obs_pos_noise_std = 0.0025 # 0.0025  # sigma = 0.0025m = 2.5mm
    obs_vel_noise_mean = 0.0
    obs_vel_noise_std = 0.06 # 0.06  # sigma = 0.06m/s
    obs_rot_noise_mean = 0.0
    obs_rot_noise_std = 0.01

    obs_fric_noise_mean = 0.0
    obs_fric_noise_std = 0.05 # 0.06  # sigma = 0.06m/s

    # Table
    table_length = 4.0
    cuboidtable2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboidtable2",
        spawn=sim_utils.CuboidCfg(
            size = [table_length, 1.0, 1.0], 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True, 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.85, 0.9), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Puck
    puck_length = 0.032
    puck_default_pos = 1.35
    cylinderpuck2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinderpuck2",
        spawn=sim_utils.CylinderCfg(
            radius = puck_length, 
            height = 0.05, 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),  # mass=0.23
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True, 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(puck_default_pos, 0.0, 1.025), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # puck_length = 0.1
    # puck_default_pos = 1.35
    # cylinderpuck2_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cylinderpuck2",
    #     spawn=sim_utils.CuboidCfg(
    #         size = [0.05, puck_length, 0.05], 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         activate_contact_sensors=True, 
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(puck_default_pos, 0.0, 1.05), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # Elle's example
    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.05, 0.05, 0.05),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
    #         collision_props=CollisionPropertiesCfg(collision_enabled=True)
    #     ),
    # )

    # # Puck
    # puck_length = 0.032
    # puck_default_pos = 1.1
    # cylinderpuck2_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cylinderpuck2",
    #     spawn=sim_utils.SphereCfg(
    #         radius = puck_length, 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.15),  # mass=0.23
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         activate_contact_sensors=True, 
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(puck_default_pos, 0.0, 1.025), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # # Pusher
    # pusher_length = 0.026 # 0.1
    # pusher_default_pos = 1.2 # 1.4
    # cuboidpusher2_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboidpusher2",
    #     spawn=sim_utils.CuboidCfg(
    #         size = [pusher_length, pusher_length, pusher_length], # [pusher_length, 0.2, 0.05], 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         activate_contact_sensors=True, 
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(pusher_default_pos, 0.0, 1.075), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # Pusher
    # pusher_length = 0.03
    # pusher_default_pos = 1.3
    # cuboidpusher2_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboidpusher2",
    #     spawn=sim_utils.SphereCfg(
    #         radius=pusher_length, 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         activate_contact_sensors=True, 
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(pusher_default_pos, 0.0, 1.025), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # pusher_length = 0.013
    # pusher_default_pos = 1.45
    # cuboidpusher2_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboidpusher2",
    #     spawn=sim_utils.SphereCfg(
    #         radius=pusher_length, 
    #         # height = 0.05, 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         activate_contact_sensors=True, 
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(pusher_default_pos, 0.0, 1.025), rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    pusher_length = 0.032
    pusher_default_pos = 1.45
    cuboidpusher2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboidpusher2",
        spawn=sim_utils.CylinderCfg(
            radius = pusher_length, 
            height = 0.04, 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # mass=0.23
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True, 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(pusher_default_pos, 0.0, 1.02), rot=(1.0, 0.0, 0.0, 0.0)),
    )
 
    # Goal
    goal_location = 0.5  # the cart is reset if it exceeds that position [m]
    goal_location = [0.5, 0.0, 1.0]
    goal_length = 0.1
    max_puck_goalcount = 10
    max_estimation_goalcount = 5

    # markergoal1_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visual/Goal1",
    #     markers={
    #         "cylinder": sim_utils.CylinderCfg(
    #             radius = 0.05, 
    #             height = 0.05, 
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #         ),
    #     },
    # )

    markergoal1_cfg = VisualizationMarkersCfg(
        prim_path="/Visual/Start1",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(goal_length*2, goal_length*2, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )


    # Start region
    max_pusher_posx_bound = 1.6  # the cart is reset if it exceeds that position [m]
    min_pusher_posx_bound = 1.0  # the cart is reset if it exceeds that position [m]
    max_pusher_posx = max_pusher_posx_bound-(pusher_length/2.0)  # the cart is reset if it exceeds that position [m] (0.95)
    min_pusher_posx = min_pusher_posx_bound+(pusher_length/2.0)   # the cart is reset if it exceeds that position [m] (0.05)
    max_pusher_posy_bound = 0.3  # the cart is reset if it exceeds that position [m]
    min_pusher_posy_bound = -0.3  # the cart is reset if it exceeds that position [m]
    min_pusher_posy = min_pusher_posy_bound+(pusher_length/2.0)   # the cart is reset if it exceeds that position [m] (0.05)
    max_pusher_posy = max_pusher_posy_bound-(pusher_length/2.0)  # the cart is reset if it exceeds that position [m] (0.95)
    start_length = abs(max_pusher_posx_bound - min_pusher_posx_bound)
    start_location = (max_pusher_posx_bound + min_pusher_posx_bound)/2.0
    print("Start locationnnnnn")
    print(start_location)
    markerstart1_cfg = VisualizationMarkersCfg(
        prim_path="/Visual/Start1",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(start_length, 1.0, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    decimation = 5
    episode_length_s = 3.0
    action_scale = 1.0
    num_actions = 2 # action dim
    num_observations = 12
    num_states = 2

    max_puck_posx = 2.0  # the cart is reset if it exceeds that position [m]
    min_puck_posx = -2.0  # the cart is reset if it exceeds that position [m]
    max_puck_posy = 0.5  # the cart is reset if it exceeds that position [m]
    min_puck_posy = -0.5  # the cart is reset if it exceeds that position [m]
    min_puck_velx = 0.01
    max_puck_restcount = 10
    
    # reward scales
    rew_scale_terminated = -15.0
    rew_scale_distance = 0.05
    rew_scale_goal = 30.0
    rew_scale_timestep = 0.001
    rew_scale_pushervel = -0.1

class SlidingPandaGymPropEnv(DirectRLEnvFeedback):
    cfg: SlidingPandaGymPropEnvCfg

    def __init__(self, cfg: SlidingPandaGymPropEnvCfg, render_mode: str | None = None, **kwargs):
        # print("Env init called!!!!")
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale

        # Root state (default state)
        self.cuboidpusher2_state = self.cuboidpusher2.data.root_state_w.clone()
        self.cylinderpuck2_state = self.cylinderpuck2.data.root_state_w.clone()
        self.cuboidtable2_state = self.cuboidtable2.data.root_state_w.clone()

        # Out of bound counter: min puck velocity (puck at rest)
        self.out_of_bounds_min_puck_velx_count = torch.zeros((self.scene.env_origins.shape[0]), device=self.scene.env_origins.device)

        # Out of bound counter: goal puck (puck in goal)
        self.out_of_bounds_goal_puck_posx_count = torch.zeros((self.scene.env_origins.shape[0]), device=self.scene.env_origins.device)
        self.out_of_bounds_goal_prop_estimate_count = torch.zeros((self.scene.env_origins.shape[0]), device=self.scene.env_origins.device)

        # Recent episode success/failure tracking (1: success, 0: failure)
        self.goal_bounds = torch.zeros((self.scene.env_origins.shape[0]), dtype=torch.bool)
        # self.inside_goal = torch.zeros((self.scene.env_origins.shape[0]), dtype=torch.bool)

        self.goal_bounds_exp = torch.zeros((self.scene.env_origins.shape[0]), dtype=torch.bool)
        self.task_phase = torch.zeros((self.scene.env_origins.shape[0]), dtype=torch.bool)

        # Past puck velocity and acceleration tracking
        self.prev_puck_vel = torch.zeros((self.scene.env_origins.shape[0]), device=self.scene.env_origins.device)
        self.prev_puck_acc = torch.zeros_like(self.prev_puck_vel)
        
        # Goal location tensor tracking (goal shape = (num_envs, 3))
        init_goal_location = torch.tensor(self.cfg.goal_location, device=self.scene.env_origins.device)
        self.goal_locations = init_goal_location.repeat(self.scene.env_origins.shape[0], 1)

        self.goal_length = self.cfg.goal_length
        self.success_threshold = 0.2
        self.maxgoal_locations = self.goal_locations[:,0]+(self.goal_length/2.0)-(self.cfg.puck_length/2.0)  # the cart is reset if it exceeds that position [m] (-0.7)
        self.mingoal_locations = (self.goal_locations[:,0]-(self.goal_length/2.0))+(self.cfg.puck_length/2.0)
        self.goal_threshold = 0.1

        # property estimation goal
        self.prop_estimate_threshold = 0.05
        
        # Goal randomisation range
        self.goal_location_min = 0.25
        self.goal_location_max = 0.75
        self.goal_location_min_x = 0.0
        self.goal_location_max_x = 0.75
        self.goal_location_min_y = -0.3
        self.goal_location_max_y = 0.3
        self.discrete_goals = torch.tensor([0.75, 0.5, 0.25, 0.0], device=self.device)
        self.discrete_goals_x = torch.tensor([0.5, 0.7, 0.9], device=self.device)    # Nearby goals: [0.7, 0.9]
        self.discrete_goals_y = torch.tensor([0.1, -0.1], device=self.device)   # Nearby goals: [0.1, -0.1]
        self.discrete_goal = False
        
        # Normalisaion range: goal
        # self.goal_location_normmax = 2.0
        # self.goal_location_normmin = -2.0
        self.goal_location_normmax = 1.5
        self.goal_location_normmin = -1.5
        
        # Normalisaion range: object
        # self.object_location_normmax = self.goal_location_normmax
        # self.object_location_normmin = self.goal_location_normmin
        self.object_location_normmax = 2.0
        self.object_location_normmin = -2.0
        self.object_vel_normmax = 3.0
        self.object_vel_normmin = -3.0
        self.object_rot_normmax = 2.0
        self.object_rot_normmin = -2.0

        # Success rate tracking
        self.success_rates = []

        state_pos_2d = [0,1]
        state_pos_1d = [0]
        self.state_pos_idx = state_pos_2d

        state_vel_2d = [7,8]
        state_vel_1d = [7]
        self.state_vel_idx = state_vel_2d

        self.state_rot_idx = [6]

        self.state_norm_max = 2
        self.state_norm_min = -2

        # Past state tracking
        self.past_timestep = 1
        initial_pos_tensor = torch.zeros(self.past_timestep, self.num_envs, len(self.state_pos_idx), device=self.scene.env_origins.device)
        initial_vel_tensor = torch.zeros(self.past_timestep, self.num_envs, len(self.state_vel_idx), device=self.scene.env_origins.device)
        initial_rot_tensor = torch.zeros(self.past_timestep, self.num_envs, len(self.state_rot_idx), device=self.scene.env_origins.device)
        self.past_pusher_pos = [initial_pos_tensor]
        self.past_puck_pos = [initial_pos_tensor]
        self.past_pusher_vel = [initial_vel_tensor]
        self.past_puck_vel = [initial_vel_tensor]
        self.past_puck_rot = [initial_rot_tensor]
        self.past_puckpusher_relative = [initial_pos_tensor]
        self.past_puckgoal_relative = [initial_pos_tensor]

        # Past state tracking for RNN prop estimation
        self.past_timestep_prop = 15
        self.initial_pos_tensor_prop = torch.zeros(1, self.num_envs, len(self.state_pos_idx), device=self.scene.env_origins.device)
        self.past_pusher_pos_prop = [self.initial_pos_tensor_prop.clone() for _ in range(self.past_timestep_prop)]
        self.past_puck_pos_prop = [self.initial_pos_tensor_prop.clone() for _ in range(self.past_timestep_prop)]
        self.initial_obs_tensor_prop = torch.zeros(1, self.num_envs, 5, device=self.scene.env_origins.device)
        self.past_obs_prop = [self.initial_obs_tensor_prop.clone() for _ in range(self.past_timestep_prop)]
        self.initial_action_tensor_prop = torch.zeros(1, self.num_envs, 2, device=self.scene.env_origins.device)
        self.past_action_prop = [self.initial_action_tensor_prop.clone() for _ in range(self.past_timestep_prop)]

        self.curriculum_count = 0

        ### Embeddings ###
        # embedding_lookuptable_path = "/workspace/isaaclab/logs/exp_lookuptable/predefined/dynamicfriction_z2.pkl"
        # with open(embedding_lookuptable_path, "rb") as fp: 
        #     self.embedding_lookuptable = pickle.load(fp)

        # # min_values_all = self.embedding_lookuptable.min()
        # # max_values_all = self.embedding_lookuptable.max()

        # # self.min_values = min_values_all[[0, 1]].values
        # # self.max_values = max_values_all[[0, 1]].values

        # self.min_value_global = self.embedding_lookuptable[[0, 1]].min().min()
        # self.max_value_global = self.embedding_lookuptable[[0, 1]].max().max()

        # # print("Min max valuessss")
        # # print(self.min_value_global)
        # # print(self.max_value_global)

        # # print(self.embedding_lookuptable.shape) 

        ### Embeddings ###  

        # ### Prop estimation (offline) ###

        # # Load the learnt model info
        # prop_estimator_dict_path = "/workspace/isaaclab/logs/prop_estimation/offline_prop_estimation/2024-09-13_13-27-32/model/model_params_dict.pkl"
        # with open(prop_estimator_dict_path, "rb") as fp: 
        #     self.prop_estimator_dict = pickle.load(fp)
        # print(self.prop_estimator_dict)

        # rnn_input_size = self.prop_estimator_dict["input_size"]
        # rnn_hidden_size = self.prop_estimator_dict["hidden_size"]
        # rnn_num_layers = self.prop_estimator_dict["num_layers"]
        # rnn_output_size = self.prop_estimator_dict["output_size"]

        # self.rnn_prop_model = rnnmodel.RNNPropertyEstimator(rnn_input_size, rnn_hidden_size, rnn_num_layers, rnn_output_size).to(self.scene.env_origins.device)

        # # Load the learnt model
        # self.rnn_prop_model.load_state_dict(torch.load(self.prop_estimator_dict["model_path"], map_location=torch.device(self.scene.env_origins.device)))
        # self.rnn_prop_model.eval()

        # self.pos_min = self.prop_estimator_dict["pos_min"] 
        # self.pos_max = self.prop_estimator_dict["pos_max"] 
        # self.rot_min = self.prop_estimator_dict["rot_min"] 
        # self.rot_max = self.prop_estimator_dict["rot_max"] 
        # self.vel_min = self.prop_estimator_dict["vel_min"] 
        # self.vel_max = self.prop_estimator_dict["vel_max"] 
        # self.feature_target_min = self.prop_estimator_dict["feature_target_min"] 
        # self.feature_target_max = self.prop_estimator_dict["feature_target_max"]

        # self.fric_min = self.prop_estimator_dict["fric_min"] 
        # self.fric_max = self.prop_estimator_dict["fric_max"] 
        # self.com_min = self.prop_estimator_dict["com_min"] 
        # self.com_max = self.prop_estimator_dict["com_max"] 
        # self.action_target_min = self.prop_estimator_dict["action_target_min"] 
        # self.action_target_max = self.prop_estimator_dict["action_target_max"]

        # ### Prop estimation (offline) END ###

        # Episodic noises
        self.obs_pos_noise_epi = torch.normal(self.cfg.obs_pos_noise_mean, self.cfg.obs_pos_noise_std, size=(self.scene.env_origins.shape[0],1)).to(self.scene.env_origins.device)
        self.obs_vel_noise_epi = torch.normal(self.cfg.obs_vel_noise_mean, self.cfg.obs_vel_noise_std, size=(self.scene.env_origins.shape[0],1)).to(self.scene.env_origins.device)
        self.obs_rot_noise_epi = torch.normal(self.cfg.obs_rot_noise_mean, self.cfg.obs_rot_noise_std, size=(self.scene.env_origins.shape[0],1)).to(self.scene.env_origins.device)
        self.obs_fric_noise_epi = torch.normal(self.cfg.obs_fric_noise_mean, self.cfg.obs_fric_noise_std, size=(self.scene.env_origins.shape[0],1)).to(self.scene.env_origins.device)

        # Property estimation
        self.rnn_rmse = None
        self.denormalsied_output = None
        self.denormalsied_target = None
        self.prop_rmse_eachenv = None       

    def _setup_scene(self):
        # print("Env setup scene called!!!!")

        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

        self.markerstart1 = VisualizationMarkers(cfg=self.cfg.markerstart1_cfg)
        self.markergoal1 = VisualizationMarkers(cfg=self.cfg.markergoal1_cfg) 

        self.cylinderpuck2 = RigidObject(self.cfg.cylinderpuck2_cfg)
        self.cuboidpusher2 = RigidObject(self.cfg.cuboidpusher2_cfg)
        self.cuboidtable2 = RigidObject(self.cfg.cuboidtable2_cfg)

        # Start marker positioning
        goal_pos_offset = torch.zeros(self.scene.env_origins.shape)
        goal_pos_offset[:, 0] = self.cfg.start_location
        goal_pos_offset[:, 2] = 1.0
        goal_pos_offset = goal_pos_offset.to(self.scene.env_origins.device)
        goal_pos = self.scene.env_origins + goal_pos_offset
        goal_pos = goal_pos.to(self.scene.env_origins.device)
        goal_rot = torch.zeros((self.scene.env_origins.shape[0], 4))
        goal_rot = goal_rot.to(self.scene.env_origins.device)
        self.markerstart1.visualize(goal_pos, goal_rot) 

        # Goal marker positioning
        goal_pos_offset = torch.zeros(self.scene.env_origins.shape)
        goal_pos_offset[:, 0] = self.cfg.goal_location[0]
        goal_pos_offset[:, 1] = self.cfg.goal_location[1]
        goal_pos_offset[:, 2] = self.cfg.goal_location[2]
        goal_pos_offset = goal_pos_offset.to(self.scene.env_origins.device)
        goal_pos = self.scene.env_origins + goal_pos_offset
        goal_pos = goal_pos.to(self.scene.env_origins.device)
        goal_rot = torch.zeros((self.scene.env_origins.shape[0], 4))
        goal_rot = goal_rot.to(self.scene.env_origins.device)
        self.markergoal1.visualize(goal_pos, goal_rot) 
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.rigid_objects["cylinderpuck2"] = self.cylinderpuck2
        self.scene.rigid_objects["cuboidpusher2"] = self.cuboidpusher2
        self.scene.rigid_objects["cuboidtable2"] = self.cuboidtable2
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_rnn_output(self, rnn_output: dict) -> None:
        print("Check estimated prop")
        print(self.episode_length_buf)
        print(rnn_output["denormalsied_output"])

        curr_materials = self.scene.rigid_objects["cylinderpuck2"].root_physx_view.get_material_properties()
        # static_frictions = curr_materials.squeeze().reshape((-1,3))[:,0].to(self.scene.env_origins.device)
        dynamic_frictions = curr_materials.squeeze().reshape((-1,3))[:,1].to(self.scene.env_origins.device)
        # restitutions = curr_materials.squeeze().reshape((-1,3))[:,2].to(self.scene.env_origins.device)
        print(dynamic_frictions)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        # print("Env pre-physics called!!!!")
        self.actions = self.action_scale * actions.clone()
        # self.actions[self.actions < -2.0] = -2.0
        # self.actions[:,0][self.episode_length_buf <4] = -1.0
        # self.actions[:,1] = -0.0
        # self.actions[self.actions < 0.0] = -0.0
        # self.actions[self.actions > 0.0] = -0.0
        # print(self.actions.shape)
        # print("Action vel")
        # print(self.actions.shape)
        # pass

        ### Action data tracking for rnn prop estimation ###
        # print("Actions data")
        # print(self.actions.unsqueeze(0).shape)
        self.past_action_prop.append(self.actions.unsqueeze(0))
        self.past_action_prop = self.past_action_prop[-self.past_timestep_prop:]

    def _apply_action(self) -> None:
        # print("Env apply action called!!!!")
        new_linvel = torch.zeros((self.scene.num_envs, 3))
        new_linvel = new_linvel.to(self.scene.env_origins.device)
        # new_linvel[:,0] = new_linvel[:,0]-0.2
        # print("Actionsssssss")
        # print(self.actions.shape)
        xvel = self.actions[:,[0,1]]
        # xvel = -0.0
        new_linvel[:,[0,1]] = new_linvel[:,[0,1]]+xvel
        new_angvel = torch.zeros((self.scene.num_envs, 3))
        new_angvel = new_angvel.to(self.scene.env_origins.device)
        # print("New Lin vellllll")
        # print(new_linvel.shape)
        # Pusher state
        # curr_cuboidpusher2_state = self.cuboidpusher2_state.clone()
        # curr_cuboidpusher2_state[:, 0:3] = (
        #     curr_cuboidpusher2_state[:, 0:3] - self.scene.env_origins
        # )
        # next_position = self.compute_next_position(curr_cuboidpusher2_state[:, 0:3], new_linvel)
        # # print(next_position)
        # exceeds_min = curr_cuboidpusher2_state[:, 0] < self.cfg.min_pusher_posx
        # exceeds_max = curr_cuboidpusher2_state[:, 0] > self.cfg.max_pusher_posx
        # print("Exceeed")
        # print(exceeds_min)
        # print(exceeds_max)
        # new_linvel[:, 0][exceeds_min | exceeds_max] = 0.0
        # print(new_linvel)
        # exceeds = exceeds_min | exceeds_max
        # new_linvel[exceeds, 0] = 0.0
        self.cuboidpusher2.set_velocities(new_linvel, new_angvel)
        # pass

    def compute_next_position(self, curr_pos, velocity_command, dt=1/120):
        # print("CHekckk")
        # print(curr_pos)
        # print(velocity_command)
        # print(dt)
        control_dt = self.cfg.decimation * dt
        next_position = curr_pos + velocity_command * control_dt
        return next_position
        # pass

    def _get_observations(self) -> dict:
        # print("Env get observations called!!!!")
        
        # Pusher state
        curr_cuboidpusher2_state = self.cuboidpusher2_state.clone()
        curr_cuboidpusher2_state[:, 0:3] = (
            curr_cuboidpusher2_state[:, 0:3] - self.scene.env_origins
        )

        # Pusher pos
        # normalized_curr_pusher_pos = (curr_cuboidpusher2_state[:, self.state_pos_idx] - self.object_location_normmin) / (self.object_location_normmax - self.object_location_normmin)
        normalized_curr_pusher_pos = normalize(curr_cuboidpusher2_state[:, self.state_pos_idx], self.object_location_normmin, self.object_location_normmax, self.state_norm_min, self.state_norm_max)
        self.past_pusher_pos.append(normalized_curr_pusher_pos.unsqueeze(0))
        self.past_pusher_pos = self.past_pusher_pos[-self.past_timestep:]
        past_pusher_pos_tensor = torch.cat(self.past_pusher_pos, dim=0)
        normalized_past_pusher_pos_obs  = past_pusher_pos_tensor[-self.past_timestep:, :, :]

        # Pusher vel
        # normalized_curr_pusher_vel = (curr_cuboidpusher2_state[:, self.state_vel_idx] - self.object_vel_normmin) / (self.object_vel_normmax - self.object_vel_normmin)
        normalized_curr_pusher_vel = normalize(curr_cuboidpusher2_state[:, self.state_vel_idx], self.object_vel_normmin, self.object_vel_normmax, self.state_norm_min, self.state_norm_max)
        self.past_pusher_vel.append(normalized_curr_pusher_vel.unsqueeze(0))
        self.past_pusher_vel = self.past_pusher_vel[-self.past_timestep:]
        past_pusher_vel_tensor = torch.cat(self.past_pusher_vel, dim=0)
        normalized_past_pusher_vel_obs  = past_pusher_vel_tensor[-self.past_timestep:, :, :]
        
        # Puck state
        curr_cylinderpuck2_state = self.cylinderpuck2_state.clone()
        curr_cylinderpuck2_state[:, 0:3] = (
            curr_cylinderpuck2_state[:, 0:3] - self.scene.env_origins
        )

        # Puck pos
        # normalized_curr_puck_pos = (curr_cylinderpuck2_state[:, self.state_pos_idx] - self.object_location_normmin) / (self.object_location_normmax - self.object_location_normmin)
        normalized_curr_puck_pos = normalize(curr_cylinderpuck2_state[:, self.state_pos_idx], self.object_location_normmin, self.object_location_normmax, self.state_norm_min, self.state_norm_max)
        self.past_puck_pos.append(normalized_curr_puck_pos.unsqueeze(0))
        self.past_puck_pos = self.past_puck_pos[-self.past_timestep:]
        past_puck_pos_tensor = torch.cat(self.past_puck_pos, dim=0)
        normalized_past_puck_pos_obs  = past_puck_pos_tensor

        # Puck vel
        # normalized_curr_puck_vel = (curr_cylinderpuck2_state[:, self.state_vel_idx] - self.object_vel_normmin) / (self.object_vel_normmax - self.object_vel_normmin)
        normalized_curr_puck_vel = normalize(curr_cylinderpuck2_state[:, self.state_vel_idx], self.object_vel_normmin, self.object_vel_normmax, self.state_norm_min, self.state_norm_max)
        self.past_puck_vel.append(normalized_curr_puck_vel.unsqueeze(0))
        self.past_puck_vel = self.past_puck_vel[-self.past_timestep:]
        past_puck_vel_tensor = torch.cat(self.past_puck_vel, dim=0)
        normalized_past_puck_vel_obs  = past_puck_vel_tensor

        # Puck orientation
        normalized_curr_puck_rot = normalize(curr_cylinderpuck2_state[:, self.state_rot_idx], self.object_rot_normmin, self.object_rot_normmax, self.state_norm_min, self.state_norm_max)
        self.past_puck_rot.append(normalized_curr_puck_rot.unsqueeze(0))
        self.past_puck_rot = self.past_puck_rot[-self.past_timestep:]
        past_puck_rot_tensor = torch.cat(self.past_puck_rot, dim=0)
        normalized_past_puck_rot_obs  = past_puck_rot_tensor

        # Puck-pusher relative position 
        curr_puckpusher_relative = curr_cylinderpuck2_state[:, self.state_pos_idx] - curr_cuboidpusher2_state[:, self.state_pos_idx] 
        normalized_curr_puckpusher_relative = normalize(curr_puckpusher_relative, self.object_location_normmin, self.object_location_normmax, self.state_norm_min, self.state_norm_max)
        self.past_puckpusher_relative.append(normalized_curr_puckpusher_relative.unsqueeze(0))
        self.past_puckpusher_relative = self.past_puckpusher_relative[-self.past_timestep:]
        past_puckpusher_relative_tensor = torch.cat(self.past_puckpusher_relative, dim=0)
        normalized_past_puckpusher_relative_obs  = past_puckpusher_relative_tensor

        curr_puckgoal_relative = curr_cylinderpuck2_state[:, self.state_pos_idx] - self.goal_locations[:, self.state_pos_idx] 
        normalized_curr_puckgoal_relative = normalize(curr_puckgoal_relative, self.object_location_normmin, self.object_location_normmax, self.state_norm_min, self.state_norm_max)
        self.past_puckgoal_relative.append(normalized_curr_puckgoal_relative.unsqueeze(0))
        self.past_puckgoal_relative = self.past_puckgoal_relative[-self.past_timestep:]
        past_puckgoal_relative_tensor = torch.cat(self.past_puckgoal_relative, dim=0)
        normalized_past_puckgoal_relative_obs  = past_puckgoal_relative_tensor
        # print("Current relative position")
        # print(curr_puckpusher_relative)
        # print(curr_puckgoal_relative)
        
        # Goal
        # goal_tensor = self.goal_locations[:,0].clone()
        # normalized_goal_tensor = (goal_tensor - self.goal_location_normmin) / (self.goal_location_normmax - self.goal_location_normmin)
        goal_tensor_x = self.goal_locations[:,0].clone()
        # normalized_goal_tensor_x = (goal_tensor_x - self.goal_location_normmin) / (self.goal_location_normmax - self.goal_location_normmin)
        normalized_goal_tensor_x = normalize(goal_tensor_x, self.goal_location_normmin, self.goal_location_normmax, self.state_norm_min, self.state_norm_max)
        normalized_goal_tensor_x = normalized_goal_tensor_x.view(-1,1)
        goal_tensor_y = self.goal_locations[:,1].clone()
        # normalized_goal_tensor_y = (goal_tensor_y - self.goal_location_normmin) / (self.goal_location_normmax - self.goal_location_normmin)
        normalized_goal_tensor_y = normalize(goal_tensor_y, self.goal_location_normmin, self.goal_location_normmax, self.state_norm_min, self.state_norm_max)
        normalized_goal_tensor_y = normalized_goal_tensor_y.view(-1,1)

        # Properties
        # Materials
        curr_materials = self.scene.rigid_objects["cylinderpuck2"].root_physx_view.get_material_properties()
        static_frictions = curr_materials.squeeze().reshape((-1,3))[:,0].to(self.scene.env_origins.device)
        dynamic_frictions = curr_materials.squeeze().reshape((-1,3))[:,1].to(self.scene.env_origins.device)
        restitutions = curr_materials.squeeze().reshape((-1,3))[:,2].to(self.scene.env_origins.device)

        dynamic_frictions_min = 0.05
        dynamic_frictions_max = 0.3
        # normalized_dynamic_frictions = (dynamic_frictions - dynamic_frictions_min) / (dynamic_frictions_max - dynamic_frictions_min)
        normalized_dynamic_frictions = normalize(dynamic_frictions, dynamic_frictions_min, dynamic_frictions_max, self.state_norm_min, self.state_norm_max)
        normalized_dynamic_frictions = normalized_dynamic_frictions.view(-1,1)

        # CoM
        curr_coms = self.scene.rigid_objects["cylinderpuck2"].root_physx_view.get_coms()
        com_x = curr_coms[:,0].to(self.scene.env_origins.device)
        com_y = curr_coms[:,1].to(self.scene.env_origins.device)
        com_z = curr_coms[:,2].to(self.scene.env_origins.device)
        com_min = -0.02
        com_max = 0.02
        normalized_com_x = (com_x - com_min) / (com_max - com_min)
        normalized_com_y = (com_y - com_min) / (com_max - com_min)
        normalized_com_z = (com_z - com_min) / (com_max - com_min)

        # Estimated prop
        if self.denormalsied_output == None: 
            # print("None")
            normalized_estimated_dynamic_frictions = torch.zeros_like(normalized_dynamic_frictions)
            # pass
        else: 
            # print(self.denormalsied_target)
            # print(self.denormalsied_output)
            # print(dynamic_frictions)
            normalized_estimated_dynamic_frictions = (self.denormalsied_output - dynamic_frictions_min) / (dynamic_frictions_max - dynamic_frictions_min)

        if self.prop_rmse_eachenv == None: 
            estimation_errors = torch.zeros(self.num_envs).to(self.device)
            # print("None")
        else: 
            estimation_errors = self.prop_rmse_eachenv
            # print(self.prop_rmse_eachenv.shape)
        # print(estimation_errors)
        estimation_errors = estimation_errors.view(-1,1)

        ### Embeddings ### 
        # # print("Dynamic Frictionsssss: should be (32,1)")
        # dynamic_frictions_np = dynamic_frictions.cpu().detach().numpy().reshape(self.num_envs, -1)
        # # print(dynamic_frictions_np.shape)
        # # print(self.embedding_lookuptable[['dynamic friction']].shape)

        # neigh_dynamicfric = NearestNeighbors(n_neighbors=1)
        # neigh_dynamicfric.fit(self.embedding_lookuptable[['dynamic friction']].to_numpy())

        # dynamicfric_match_indices = neigh_dynamicfric.kneighbors(dynamic_frictions_np, return_distance=False)
        # # print("Look up table")
        # # print(self.embedding_lookuptable)
        # dynamicfric_matches = self.embedding_lookuptable.iloc[dynamicfric_match_indices.flatten()][[0, 1]]
        # dynamicfric_matches_np= dynamicfric_matches.to_numpy()

        # dynamicfric_matches_normalized_np = (dynamicfric_matches_np - self.min_value_global) / (self.max_value_global - self.min_value_global)

        # # print("This should be all sameeeee")
        # # print(dynamicfric_matches_normalized_np)

        # dynamicfric_matches_tensor = torch.from_numpy(dynamicfric_matches_normalized_np).to(self.device)

        ### Embeddings (finish) ###

        # Check if Puck unreachable
        unreachable_max_puck_posx = curr_cylinderpuck2_state[:,0] > self.cfg.max_pusher_posx
        unreachable_min_puck_posx = curr_cylinderpuck2_state[:,0] < self.cfg.min_pusher_posx
        unreachable_max_puck_pos = unreachable_max_puck_posx | unreachable_min_puck_posx
        
        # Step-wise noise
        self.obs_pos_noise_step = torch.normal(self.cfg.obs_pos_noise_mean, self.cfg.obs_pos_noise_std, size=(self.scene.env_origins.shape[0],1)).to(self.scene.env_origins.device)
        self.obs_vel_noise_step = torch.normal(self.cfg.obs_vel_noise_mean, self.cfg.obs_vel_noise_std, size=(self.scene.env_origins.shape[0],1)).to(self.scene.env_origins.device)
        self.obs_rot_noise_step = torch.normal(self.cfg.obs_rot_noise_mean, self.cfg.obs_rot_noise_std, size=(self.scene.env_origins.shape[0],1)).to(self.scene.env_origins.device)
        self.obs_fric_noise_step = torch.normal(self.cfg.obs_fric_noise_mean, self.cfg.obs_fric_noise_std, size=(self.scene.env_origins.shape[0],1)).to(self.scene.env_origins.device)

        normalized_past_puck_pos_obs_x = normalized_past_puck_pos_obs[:,:,0].T + self.obs_pos_noise_step + self.obs_pos_noise_epi
        normalized_past_puck_pos_obs_y = normalized_past_puck_pos_obs[:,:,1].T + self.obs_pos_noise_step + self.obs_pos_noise_epi
        normalized_past_puck_vel_obs_x = normalized_past_puck_vel_obs[:,:,0].T + self.obs_vel_noise_step + self.obs_vel_noise_epi 
        normalized_past_puck_vel_obs_y = normalized_past_puck_vel_obs[:,:,1].T + self.obs_vel_noise_step + self.obs_vel_noise_epi 
        normalized_past_puck_rot_obs_yaw = normalized_past_puck_rot_obs[:,:,0].T + self.obs_rot_noise_step + self.obs_rot_noise_epi 
        normalized_past_pusher_pos_obs_x = normalized_past_pusher_pos_obs[:,:,0].T + self.obs_pos_noise_step + self.obs_pos_noise_epi
        normalized_past_pusher_pos_obs_y = normalized_past_pusher_pos_obs[:,:,1].T + self.obs_pos_noise_step + self.obs_pos_noise_epi
        normalized_past_pusher_vel_obs_x = normalized_past_pusher_vel_obs[:,:,0].T + self.obs_vel_noise_step + self.obs_vel_noise_epi 
        normalized_past_pusher_vel_obs_y = normalized_past_pusher_vel_obs[:,:,1].T + self.obs_vel_noise_step + self.obs_vel_noise_epi 

        # Noise on groundtruth properties
        # normalized_dynamic_frictions = normalized_dynamic_frictions + self.obs_fric_noise_step + self.obs_fric_noise_epi 

        # Offline prop estimation

        curr_obs_prop = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y), dim=1)
        self.past_obs_prop.append(curr_obs_prop.unsqueeze(0))
        self.past_obs_prop = self.past_obs_prop[-self.past_timestep_prop:]
        past_obs_prop_tensor = torch.cat(self.past_obs_prop, dim=0)

        # position_index = [0,1,3,4]
        # rotation_index = 2

        # # online_rnn_prop_info = {
        # #     "position_index": position_index, 
        # #     "rotation_index": rotation_index, 
        # # }

        # positions = past_obs_prop_tensor[:, :, position_index]
        # rotation = past_obs_prop_tensor[:, :, rotation_index]

        # normalized_positions = normalize(positions, self.pos_min, self.pos_max, self.feature_target_min, self.feature_target_max)
        # normalized_rotation = normalize(rotation, self.rot_min, self.rot_max, self.feature_target_min, self.feature_target_max)

        # normalized_past_obs_prop = past_obs_prop_tensor.clone()
        # normalized_past_obs_prop[:, :, position_index] = normalized_positions
        # normalized_past_obs_prop[:, :, rotation_index] = normalized_rotation

        # # print("Normaliseeee past obs prop")
        # # print(normalized_past_obs_prop.shape)
        # # print(past_obs_prop_tensor.shape)

        past_action_prop_tensor = torch.cat(self.past_action_prop, dim=0)

        # velocities = past_action_prop_tensor
        # normalized_velocities = normalize(velocities, self.vel_min, self.vel_max, self.feature_target_min, self.feature_target_max)
        # normalized_past_action_prop = past_action_prop_tensor.clone()
        # normalized_past_action_prop = normalized_velocities

        # # print("Normaliseeee past obs prop")
        # # print(normalized_past_action_prop.shape)
        # # print(past_action_prop_tensor.shape)

        # # print("Check past obs shape")
        # # print(self.episode_length_buf)
        # # print(self.past_obs_prop[10][0,1,:])
        # # print(self.past_obs_prop[11][0,1,:])
        # # print(self.past_obs_prop[12][0,1,:])
        # # print(self.past_obs_prop[13][0,1,:])
        # # print(self.past_obs_prop[14][0,1,:])

        online_rnn_prop_input = torch.cat((past_obs_prop_tensor, past_action_prop_tensor), dim=2)
        online_rnn_prop_input = online_rnn_prop_input.swapaxes(0, 1)

        # rnn_prop_input = torch.cat((normalized_past_obs_prop, normalized_past_action_prop), dim=2)
        # rnn_prop_input = rnn_prop_input.swapaxes(0, 1)

        # Offline inference
        # self.rnn_prop_model.eval()
        # with torch.no_grad():
        #     normalsied_estimated_prop = self.rnn_prop_model(rnn_prop_input)
        
        # denormalsied_estimated_prop = denormalize(normalsied_estimated_prop, self.fric_min, self.fric_max, self.action_target_min, self.action_target_max)
        
        # normalized_estimated_prop_rl = (denormalsied_estimated_prop - dynamic_frictions_min) / (dynamic_frictions_max - dynamic_frictions_min)

        normalized_past_puckpusher_relative_obs = normalized_past_puckpusher_relative_obs.squeeze(0)
        normalized_past_puckgoal_relative_obs = normalized_past_puckgoal_relative_obs.squeeze(0)
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, normalized_com_x.view(-1, 1), norma        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y), dim=1)lized_com_y.view(-1, 1), normalized_dynamic_frictions.view(-1,1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, denormalsied_estimated_prop), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, normalized_estimated_prop_rl), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, normalized_dynamic_frictions), dim=1)        
        obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, normalized_past_puck_rot_obs_yaw, normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, normalized_dynamic_frictions), dim=1)   
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, normalized_past_puck_rot_obs_yaw, normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, normalized_dynamic_frictions, normalized_past_puckpusher_relative_obs, normalized_past_puckgoal_relative_obs), dim=1)        
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, normalized_past_puck_rot_obs_yaw, normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y), dim=1)        
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, normalized_estimated_dynamic_frictions), dim=1)        
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, normalized_dynamic_frictions, estimation_errors), dim=1)        
        # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, curr_cylinderpuck2_state[:, 6].view(-1,1), normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x, normalized_goal_tensor_y, normalized_estimated_dynamic_frictions, estimation_errors), dim=1)        
        # obs = torch.cat((normalized_past_puck_pos_obs, normalized_past_puck_vel_obs, normalized_past_pusher_pos_obs, normalized_past_pusher_vel_obs, normalized_goal_tensor.view(-1, 1), normalized_com_x.view(-1, 1), normalized_com_y.view(-1, 1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs, normalized_past_puck_vel_obs, normalized_past_pusher_pos_obs, normalized_past_pusher_vel_obs, normalized_goal_tensor.view(-1, 1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs, normalized_past_puck_vel_obs, normalized_past_pusher_pos_obs, normalized_past_pusher_vel_obs, normalized_goal_tensor.view(-1, 1), dynamic_frictions.view(-1,1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs, normalized_past_puck_vel_obs, normalized_past_pusher_pos_obs, normalized_past_pusher_vel_obs, normalized_goal_tensor.view(-1, 1), static_frictions.view(-1,1), dynamic_frictions.view(-1,1), restitutions.view(-1,1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puc?k_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puck_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1), dynamicfric_matches_tensor), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puck_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1), normalized_com_x.view(-1, 1), normalized_com_y.view(-1, 1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puck_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puck_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1), normalized_com_x.view(-1, 1), normalized_com_y.view(-1, 1), normalized_dynamic_frictions.view(-1,1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puck_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1), normalized_dynamic_frictions.view(-1,1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puck_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1), normalized_com_x.view(-1, 1), normalized_com_y.view(-1, 1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puck_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1), dynamicfric_matches_tensor), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs[:,:,0].T, normalized_past_puck_pos_obs[:,:,1].T, normalized_past_puck_vel_obs[:,:,0].T, normalized_past_puck_vel_obs[:,:,1].T, normalized_past_pusher_pos_obs[:,:,0].T, normalized_past_pusher_pos_obs[:,:,1].T, normalized_past_pusher_vel_obs[:,:,0].T, normalized_past_pusher_vel_obs[:,:,1].T, normalized_goal_tensor.view(-1, 1), normalized_dynamic_frictions.view(-1,1)), dim=1)

        # print("PROP shapeeeee")
        # print(dynamic_frictions.view(-1,1).shape)
        # print(com_x.view(-1,1).shape)
        # print(com_y.view(-1,1).shape)
        # groundtruth_prop = {
        #     "dynamic_frictions": dynamic_frictions.view(-1,1), 
        #     "com_x": com_x.view(-1,1), 
        #     "com_y": com_y.view(-1,1), 
        # }
        groundtruth_prop = torch.cat((dynamic_frictions.view(-1,1), com_x.view(-1,1), com_y.view(-1,1)), dim=1)
        observations = {"policy": obs, 
                        "prop": groundtruth_prop, 
                        "rnn_input": online_rnn_prop_input}
        
        return observations

    def _get_estimation(self, prop_info: dict) -> None:
        # Call the parent class's _get_estimation method
        super()._get_estimation(prop_info)
        
        # prop info
        # print("Prop info")
        # print(self.prop_info)
        self.rnn_rmse = self.prop_info["prop_estimator_output"]["rnn_rmse"]
        self.denormalsied_output = self.prop_info["prop_estimator_output"]["denormalsied_output"]
        self.denormalsied_target = self.prop_info["prop_estimator_output"]["denormalsied_target"]
        # print("rmse")
        # print(self.rnn_rmse)
        # print(self.denormalsied_output.shape)
        # print(self.denormalsied_target.shape)
        # print(self.denormalsied_output)
        # print(self.denormalsied_target)
        # print(self.groundtruth_prop)
        squared_error = (self.denormalsied_output - self.denormalsied_target) ** 2
        self.prop_rmse_eachenv = torch.sqrt(squared_error).squeeze() 
        if self.prop_rmse_eachenv.numel() == 1 and self.prop_rmse_eachenv.dim() == 0:
            self.prop_rmse_eachenv = self.prop_rmse_eachenv.unsqueeze(0)

        # print(prop_rmse_eachenv) 

    def _get_rewards(self) -> torch.Tensor:
        # print("Env get rewards called!!!!")

        # Puch state
        curr_cylinderpuck2_state = self.cylinderpuck2_state.clone()
        curr_cylinderpuck2_state[:, 0:3] = (
            curr_cylinderpuck2_state[:, 0:3] - self.scene.env_origins
        )

        # Pusher state
        curr_cuboidpusher2_state = self.cuboidpusher2_state.clone()
        curr_cuboidpusher2_state[:, 0:3] = (
            curr_cuboidpusher2_state[:, 0:3] - self.scene.env_origins
        )

        normalised_curr_distance = torch.abs(curr_cylinderpuck2_state[:,0] - self.goal_locations[:,0])/self.cfg.table_length

        goal_tensor = self.goal_locations[:,0].clone()
        normalized_goal_tensor = (goal_tensor - self.goal_location_normmin) / (self.goal_location_normmax - self.goal_location_normmin)

        # Compute acceleration and jerk
        physics_time_step = 1 / 120
        control_dt = self.cfg.decimation * physics_time_step

        # Acceleration
        acc = (curr_cuboidpusher2_state[:, 7]-self.prev_puck_vel)/control_dt
        self.prev_puck_vel = curr_cuboidpusher2_state[:, 7].clone()

        # Jerk
        jerk = (acc-self.prev_puck_acc)/control_dt
        self.prev_puck_acc = acc.clone()

        total_reward = compute_rewards(
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_distance, 
            self.cfg.rew_scale_goal, 
            self.cfg.rew_scale_timestep, 
            self.cfg.rew_scale_pushervel, 
            normalised_curr_distance, 
            curr_cuboidpusher2_state[:, 7], 
            self.episode_failed,
            self.goal_bounds, 
            normalized_goal_tensor, 
            self.prop_rmse_eachenv
        )
        return total_reward
        # pass

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print("Env get dones called!!!!")

        # Update state information
        self.cuboidpusher2_state = self.cuboidpusher2.data.root_state_w.clone()
        self.cylinderpuck2_state = self.cylinderpuck2.data.root_state_w.clone()

        # Check for Pusher out of bound (i.e., out of pushing area)
        curr_cuboidpusher2_state = self.cuboidpusher2_state.clone()
        curr_cuboidpusher2_state[:, 0:3] = (
            curr_cuboidpusher2_state[:, 0:3] - self.scene.env_origins
        )        
        out_of_bounds_max_pusher_posx = curr_cuboidpusher2_state[:,0] > self.cfg.max_pusher_posx
        out_of_bounds_min_pusher_posx = curr_cuboidpusher2_state[:,0] < self.cfg.min_pusher_posx

        out_of_bounds_max_pusher_posy = curr_cuboidpusher2_state[:,1] > self.cfg.max_pusher_posy
        out_of_bounds_min_pusher_posy = curr_cuboidpusher2_state[:,1] < self.cfg.min_pusher_posy

        out_of_bounds_max_pusher_pos = out_of_bounds_max_pusher_posx | out_of_bounds_max_pusher_posy
        out_of_bounds_min_pusher_pos = out_of_bounds_min_pusher_posx | out_of_bounds_min_pusher_posy

        # Check for Puck out of bound (i.e., out of table)
        curr_cylinderpuck2_state = self.cylinderpuck2_state.clone()
        curr_cylinderpuck2_state[:, 0:3] = (
            curr_cylinderpuck2_state[:, 0:3] - self.scene.env_origins
        )        
        out_of_bounds_max_puck_posx = curr_cylinderpuck2_state[:,0] > self.cfg.max_puck_posx
        out_of_bounds_min_puck_posx = curr_cylinderpuck2_state[:,0] < self.cfg.min_puck_posx

        out_of_bounds_max_puck_posy = curr_cylinderpuck2_state[:,1] > self.cfg.max_puck_posy
        out_of_bounds_min_puck_posy = curr_cylinderpuck2_state[:,1] < self.cfg.min_puck_posy

        out_of_bounds_max_puck_pos = out_of_bounds_max_puck_posx | out_of_bounds_max_puck_posy
        out_of_bounds_min_puck_pos = out_of_bounds_min_puck_posx | out_of_bounds_min_puck_posy

        # Check for Puck overshoot (i.e., over goal region) "Be careful the sign!!!!"
        overshoot_max_puck_posx = curr_cylinderpuck2_state[:,0] < self.mingoal_locations

        # Check if Puck unreachable
        unreachable_max_puck_posx = curr_cylinderpuck2_state[:,0] > self.cfg.max_pusher_posx
        unreachable_min_puck_posx = curr_cylinderpuck2_state[:,0] < self.cfg.min_pusher_posx

        # Check for Puck at rest (i.e., puck velocity is zero for sometime)
        # check if currently at rest
        curr_out_of_bounds_min_puck_velx_count = abs(curr_cylinderpuck2_state[:,7]) < self.cfg.min_puck_velx
        self.out_of_bounds_min_puck_velx_count += curr_out_of_bounds_min_puck_velx_count.int()
        # check if it reaches rest count
        out_of_bounds_min_puck_velx = self.out_of_bounds_min_puck_velx_count>self.cfg.max_puck_restcount
        # Reset count once it reaches rest count or if it is no longer at rest currently
        self.out_of_bounds_min_puck_velx_count[self.out_of_bounds_min_puck_velx_count>self.cfg.max_puck_restcount] = 0
        self.out_of_bounds_min_puck_velx_count[~curr_out_of_bounds_min_puck_velx_count] = 0
        
        # Check if it reaches the goal
        goal_bounds_max_puck_posx = curr_cylinderpuck2_state[:,0] < self.maxgoal_locations    # becomes false if overshoot
        goal_bounds_min_puck_posx = curr_cylinderpuck2_state[:,0] > self.mingoal_locations    # becomes true once in goal region

        euclid_distance = torch.norm(curr_cylinderpuck2_state[:,[0, 1]] - self.goal_locations[:,[0,1]], dim=1)

        # print("Euclid distance")
        # print(euclid_distance)

        # # Reset goal
        # curr_success_rate = self.extras.get('log')
        # if curr_success_rate is not None: 
        #     # print(curr_success_rate["success_rate"])  
        #     if curr_success_rate["success_rate"] > self.success_threshold and self.goal_threshold > 0.11 and self.curriculum_count>self.max_episode_length: 
        #         self.goal_threshold -= 0.05
        #         print(self.goal_threshold)
        #         # self.rew_scale_goal += 5
        #         # self.success_threshold += 0.1
        #         # self.goal_length -= 0.1 
        #         self.curriculum_count = 0
        #         pass
        #     else:
        #         self.curriculum_count+=1

        # curr_out_of_bounds_goal_puck_posx_count = euclid_distance < 0.15
        curr_out_of_bounds_goal_puck_posx_count = euclid_distance < self.goal_threshold
        # print(curr_out_of_bounds_goal_puck_posx_count)

        # print("INside goallll")
        # print(euclid_distance)

        # self.inside_goal = euclid_distance < 0.05 

        # curr_out_of_bounds_goal_puck_posx_count = goal_bounds_max_puck_posx & goal_bounds_min_puck_posx 
        self.out_of_bounds_goal_puck_posx_count+= curr_out_of_bounds_goal_puck_posx_count.int()

        self.goal_bounds = self.out_of_bounds_goal_puck_posx_count>self.cfg.max_puck_goalcount
        self.out_of_bounds_goal_puck_posx_count[self.out_of_bounds_goal_puck_posx_count>self.cfg.max_puck_goalcount] = 0
        self.out_of_bounds_goal_puck_posx_count[~curr_out_of_bounds_goal_puck_posx_count] = 0

        # Goal for exploration
        curr_out_of_bounds_goal_prop_estimate_count = self.prop_rmse_eachenv < self.prop_estimate_threshold

        self.out_of_bounds_goal_prop_estimate_count+= curr_out_of_bounds_goal_prop_estimate_count.int()

        self.goal_bounds_exp = self.out_of_bounds_goal_prop_estimate_count>self.cfg.max_estimation_goalcount
        self.out_of_bounds_goal_prop_estimate_count[self.out_of_bounds_goal_prop_estimate_count>self.cfg.max_estimation_goalcount] = 0
        self.out_of_bounds_goal_prop_estimate_count[~curr_out_of_bounds_goal_prop_estimate_count] = 0

        self.task_phase[self.goal_bounds_exp] = True
        # print("Task phase checker")
        # print(self.task_phase)

        # out_of_bounds = out_of_bounds_max_pusher_posx | out_of_bounds_min_pusher_posx | out_of_bounds_max_puck_posx | out_of_bounds_min_puck_posx | overshoot_max_puck_posx | self.goal_bounds | out_of_bounds_min_puck_velx     
        out_of_bounds = out_of_bounds_max_pusher_pos | out_of_bounds_min_pusher_pos | out_of_bounds_max_puck_pos | out_of_bounds_min_puck_pos | self.goal_bounds | out_of_bounds_min_puck_velx     

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # print("out_of_bounds_max_pusher_posx") 
        # print(out_of_bounds_max_pusher_posx) 
        # print("out_of_bounds_min_pusher_posx") 
        # print(out_of_bounds_min_pusher_posx) 
        # print("out_of_bounds_max_puck_posx") 
        # print(out_of_bounds_max_puck_posx) 
        # print("out_of_bounds_min_puck_posx") 
        # print(out_of_bounds_min_puck_posx) 
        # print("overshoot_max_puck_posx")  
        # print(overshoot_max_puck_posx)  
        # print("goal_bounds")    
        # print(self.goal_bounds)   
        # print("out_of_bounds_min_puck_velx")  
        # print(out_of_bounds_min_puck_velx)        
        
        # if out_of_bounds_max_pusher_posx[0]:
        #     print("out_of_bounds_max_pusher_posx") 
        # if out_of_bounds_min_pusher_posx[0]: 
        #     print("out_of_bounds_min_pusher_posx") 
        # if out_of_bounds_max_puck_posx[0]: 
        #     print("out_of_bounds_max_puck_posx") 
        # if out_of_bounds_min_puck_posx[0]: 
        #     print("out_of_bounds_min_puck_posx") 
        # if overshoot_max_puck_posx[0]: 
        #     print("overshoot_max_puck_posx")  
        # if self.goal_bounds[0]: 
        #     print("goal_bounds")    
        # if out_of_bounds_min_puck_velx[0]: 
        #     print("out_of_bounds_min_puck_velx")   

        # if time_out[0]: 
        #     print("time_out")  

        false_tensor = torch.zeros(self.scene.num_envs, dtype=torch.bool)
        true_tensor = torch.ones(self.scene.num_envs, dtype=torch.bool)

        # episode_failed = out_of_bounds_max_pusher_posx | out_of_bounds_min_pusher_posx | out_of_bounds_max_puck_posx | out_of_bounds_min_puck_posx | overshoot_max_puck_posx | time_out | out_of_bounds_min_puck_velx
        episode_failed = out_of_bounds_max_pusher_pos | out_of_bounds_min_pusher_pos | out_of_bounds_max_puck_pos | out_of_bounds_min_puck_pos | time_out | out_of_bounds_min_puck_velx

        done_info = {}
        done_info = {
            "curr_rmse": self.prop_rmse_eachenv, 
            "goal_bounds_exp": self.goal_bounds_exp, 
            "task_phase": self.task_phase
        }

        return out_of_bounds, time_out, self.goal_bounds, episode_failed, done_info
        # return false_tensor, time_out, self.goal_bounds, episode_failed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # print("Env reset idx called!!!!")

        if env_ids is None:
            env_ids = self.cylinderpuck2._ALL_INDICES
        super()._reset_idx(env_ids)

        # Episode noise
        self.obs_pos_noise_epi_new = torch.normal(self.cfg.obs_pos_noise_mean, self.cfg.obs_pos_noise_std, size=(len(env_ids),1)).to(self.scene.env_origins.device)
        self.obs_vel_noise_epi_new = torch.normal(self.cfg.obs_vel_noise_mean, self.cfg.obs_vel_noise_std, size=(len(env_ids),1)).to(self.scene.env_origins.device)
        self.obs_rot_noise_epi_new = torch.normal(self.cfg.obs_rot_noise_mean, self.cfg.obs_rot_noise_std, size=(len(env_ids),1)).to(self.scene.env_origins.device)
        self.obs_fric_noise_epi_new = torch.normal(self.cfg.obs_fric_noise_mean, self.cfg.obs_fric_noise_std, size=(len(env_ids),1)).to(self.scene.env_origins.device)
        self.obs_pos_noise_epi[env_ids, :] = self.obs_pos_noise_epi_new
        self.obs_vel_noise_epi[env_ids, :] = self.obs_vel_noise_epi_new
        self.obs_rot_noise_epi[env_ids, :] = self.obs_rot_noise_epi_new
        self.obs_fric_noise_epi[env_ids, :] = self.obs_fric_noise_epi_new

        # Past obs (prop) 
        # print("Past obs shapeee")
        # print(len(self.past_obs_prop))
        # print("Past actions shapeee")
        # print(len(self.past_action_prop))
        # print(self.past_obs_prop[0].shape)
        # print(self.past_action_prop[0].shape)
        # print("reset env ids")
        # print(env_ids.shape)
        selected_envs = env_ids.tolist()
        # print(selected_envs)
        for tensor in self.past_obs_prop:
            tensor[0, selected_envs, :] = 0  # Here, 0 is for dim0, `selected_envs` is for dim1, and `:` is for dim2
        for tensor in self.past_action_prop:
            tensor[0, selected_envs, :] = 0 # 0  # Here, 0 is for dim0, `selected_envs` is for dim1, and `:` is for dim2
        
        if self.denormalsied_output is not None: 
            self.denormalsied_output = self.denormalsied_output.clone()
            self.denormalsied_target = self.denormalsied_target.clone()
            self.denormalsied_output[selected_envs, :] = 1.0
            self.denormalsied_target[selected_envs, :] = 1.0
            self.prop_rmse_eachenv[selected_envs] = 1.0
            # print(self.denormalsied_output.shape)
            # print(self.denormalsied_target.shape)
            # print(self.prop_rmse_eachenv.shape)

        self.task_phase[selected_envs] = False

        # self.past_obs_prop = [self.initial_obs_tensor_prop.clone() for _ in range(self.past_timestep_prop)]
        # self.past_action_prop = [self.initial_action_tensor_prop.clone() for _ in range(self.past_timestep_prop)]

        # Reset goal
        # curr_success_rate = self.extras.get('log')
        # if curr_success_rate is not None: 
        #     # print(curr_success_rate["success_rate"])
        #     if curr_success_rate["success_rate"] > self.success_threshold: 
        #         # self.rew_scale_goal += 5
        #         self.success_threshold += 0.1
        #         self.goal_length -= 0.1 

        if self.discrete_goal:
            random_indices = torch.randint(len(self.discrete_goals), size=(len(env_ids),), device=self.device)
            goal_noise = self.discrete_goals[random_indices]
            random_indices_x = torch.randint(len(self.discrete_goals_x), size=(len(env_ids),), device=self.device)
            goal_noise_x = self.discrete_goals_x[random_indices_x]
            random_indices_y = torch.randint(len(self.discrete_goals_y), size=(len(env_ids),), device=self.device)
            goal_noise_y = self.discrete_goals_y[random_indices_y]
        else:
            goal_noise = sample_uniform(self.goal_location_max, self.goal_location_min, (len(env_ids)), device=self.device)
            goal_noise_x = sample_uniform(self.goal_location_max_x, self.goal_location_min_x, (len(env_ids)), device=self.device)
            goal_noise_y = sample_uniform(self.goal_location_max_y, self.goal_location_min_y, (len(env_ids)), device=self.device)
        goal_pos_offset = self.goal_locations.clone()
        # goal_pos_offset[env_ids, 0] = goal_noise
        goal_pos_offset[env_ids, 0] = goal_noise_x
        goal_pos_offset[env_ids, 1] = goal_noise_y

        self.goal_locations = goal_pos_offset.clone()
        # self.maxgoal_locations = self.goal_locations[:,0]+(self.goal_length/2.0)-(self.cfg.puck_length/2.0)  # the cart is reset if it exceeds that position [m] (-0.7)
        # self.mingoal_locations = (self.goal_locations[:,0]-(self.goal_length/2.0))+(self.cfg.puck_length/2.0)
        goal_pos_offset = goal_pos_offset.to(self.scene.env_origins.device)

        goal_pos = self.scene.env_origins + goal_pos_offset
        goal_pos = goal_pos.to(self.scene.env_origins.device)
        goal_rot = torch.zeros((self.scene.env_origins.shape[0], 4))
        goal_rot = goal_rot.to(self.scene.env_origins.device)
        self.markergoal1.visualize(goal_pos, goal_rot) 

        # Reset puck
        cylinderpuck2_default_state = self.cylinderpuck2.data.default_root_state.clone()[env_ids]
        pos_noise_x = sample_uniform(1.0, 1.5, (len(env_ids), 1), device=self.device)
        pos_noise_y = sample_uniform(-0.05, 0.05, (len(env_ids), 1), device=self.device)
        cylinderpuck2_default_state[:, 0] =  pos_noise_x.squeeze()
        cylinderpuck2_default_state[:, 1] =  pos_noise_y.squeeze()
        cylinderpuck2_local_state = cylinderpuck2_default_state.clone()
        cylinderpuck2_default_state[:, 0:3] = (
            cylinderpuck2_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        cylinderpuck2_default_state[:, 7:] = torch.zeros_like(self.cylinderpuck2.data.default_root_state[env_ids, 7:])
        # cylinderpuck2_default_state[:, 7] = -1.0
        self.cylinderpuck2_state[env_ids] = cylinderpuck2_default_state.clone()
        self.cylinderpuck2.write_root_state_to_sim(cylinderpuck2_default_state, env_ids)

        # Reset pusher
        cuboidpusher2_default_state = self.cuboidpusher2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        # cuboidpusher2_default_state[:, 0] = cylinderpuck2_local_state[:, 0]+0.1
        # cuboidpusher2_default_state[:, 1] = cylinderpuck2_local_state[:, 1]
        cuboidpusher2_default_state[:, 0:3] = (
            cuboidpusher2_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        cuboidpusher2_default_state[:, 7:] = torch.zeros_like(self.cuboidpusher2.data.default_root_state[env_ids, 7:])
        self.cuboidpusher2_state[env_ids] = cuboidpusher2_default_state.clone()
        self.cuboidpusher2.write_root_state_to_sim(cuboidpusher2_default_state, env_ids)

        # reset table
        cuboidtable2_default_state = self.cuboidtable2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        cuboidtable2_default_state[:, 0:3] = (
            cuboidtable2_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        cuboidtable2_default_state[:, 7:] = torch.zeros_like(self.cuboidtable2.data.default_root_state[env_ids, 7:])
        self.cuboidtable2_state[env_ids] = cuboidtable2_default_state.clone()
        self.cuboidtable2.write_root_state_to_sim(cuboidtable2_default_state, env_ids) 

        # Reset episode count
        self.episode_length_buf[env_ids] = 0

        pass


@torch.jit.script
def compute_rewards(
    rew_scale_terminated: float,
    rew_scale_distance: float, 
    rew_scale_goal: float, 
    rew_scale_timestep: float, 
    rew_scale_pushervel: float, 
    normalised_curr_distance: torch.Tensor,
    curr_vel: torch.Tensor,  
    reset_terminated: torch.Tensor,
    goal_bounds: torch.Tensor, 
    goal_locations: torch.Tensor, 
    prop_rmse_eachenv: torch.Tensor, 
):

    # Positive reward for reaching goal
    rew_goal = rew_scale_goal * goal_bounds.int() 
    # rew_goal = rew_scale_goal * goal_bounds.int() * (1.0 - normalised_curr_distance)
    
    # Negative reward for failed episode
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # Small positive reward for lower distance between goal and puck
    rew_distance = rew_scale_distance * (1.0 - normalised_curr_distance)
    # rew_distance = -rew_scale_distance * normalised_curr_distance

    # Small negative reward for velocity/acc/jerk 
    # normalized_pushervel = torch.abs(jerk) / 4000.0
    # normalized_pushervel = torch.abs(acc) / 10.0
    normalized_pushervel = torch.abs(curr_vel) / 10.0
    modified_tensor = torch.where(normalized_pushervel < 0.01, torch.tensor(1.0), torch.tensor(0.0))
    rew_pushervel = rew_scale_pushervel * normalized_pushervel
    rew_pushervel0 = 0.05 * modified_tensor

    # rew_inside_goal = inside_goal.int()
    # rew_outside_goal = (1-inside_goal.int())*-1
    # print("Rew goallll")
    # print(rew_inside_goal)
    # print(rew_outside_goal)

    # total_reward = rew_inside_goal.float() + rew_outside_goal.float()
    # print(total_reward.shape)

    rew_scale_props_estimate = -1.0
    rew_prop_estimate = rew_scale_props_estimate * prop_rmse_eachenv.float()

    total_reward = rew_goal + rew_termination # + rew_distance # + rew_pushervel0 # + rew_pushervel0 # + rew_timestep
    # total_reward = rew_prop_estimate + rew_goal + rew_termination # + rew_distance # + rew_pushervel0 # + rew_pushervel0 # + rew_timestep

    return total_reward
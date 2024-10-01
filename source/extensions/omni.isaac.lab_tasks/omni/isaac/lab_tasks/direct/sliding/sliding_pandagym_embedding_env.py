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
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
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

import numpy as np
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import pickle
import source.offline_learning.model.VAE as vaemodel
import torch.optim as optim
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

import time

@configclass
class EventCfg:
  robot_physics_material = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("cylinderpuck2"),
          "static_friction_range": (0.05, 0.05),
          "dynamic_friction_range": (0.05, 0.3),
          "restitution_range": (1.0, 1.0),
          "com_range_x": (-0.00, 0.00), # (-0.02, 0.02),
          "com_range_y": (-0.00, 0.00), # (-0.02, 0.02),
          "com_range_z": (0.0, 0.0),
          "num_buckets": 250,
      },
  )

@configclass
class SlidingPandaGymEmbeddingEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    events: EventCfg = EventCfg()
    
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
    puck_length = 0.05
    puck_default_pos = 1.2
    cylinderpuck2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinderpuck2",
        spawn=sim_utils.CylinderCfg(
            radius = puck_length, 
            height = 0.05, 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True, 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(puck_default_pos, 0.0, 1.025), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # # Pusher
    # pusher_length = 0.1
    # pusher_default_pos = 1.4
    # cuboidpusher2_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboidpusher2",
    #     spawn=sim_utils.CuboidCfg(
    #         size = [pusher_length, 0.2, 0.05], 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         activate_contact_sensors=True, 
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(pusher_default_pos, 0.0, 1.075), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # Pusher
    pusher_length = 0.03
    pusher_default_pos = 1.3
    cuboidpusher2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboidpusher2",
        spawn=sim_utils.SphereCfg(
            radius=pusher_length, 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True, 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(pusher_default_pos, 0.0, 1.025), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Goal
    goal_location = 0.5  # the cart is reset if it exceeds that position [m]
    goal_length = 0.5
    max_puck_goalcount = 10

    markergoal1_cfg = VisualizationMarkersCfg(
        prim_path="/Visual/Goal1",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius = puck_length+0.05, 
                height = 0.05, 
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    # Start region
    max_pusher_posx_bound = 2.0  # the cart is reset if it exceeds that position [m]
    min_pusher_posx_bound = 1.0  # the cart is reset if it exceeds that position [m]
    max_pusher_posx = max_pusher_posx_bound-(pusher_length/2.0)  # the cart is reset if it exceeds that position [m] (0.95)
    min_pusher_posx = min_pusher_posx_bound+(pusher_length/2.0)   # the cart is reset if it exceeds that position [m] (0.05)
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
    episode_length_s = 2.0
    action_scale = 1.0
    num_actions = 2 # action dim
    num_observations = 5
    num_states = 2

    max_puck_posx = 2.0  # the cart is reset if it exceeds that position [m]
    min_puck_posx = -2.0  # the cart is reset if it exceeds that position [m]
    min_puck_velx = 0.01
    max_puck_restcount = 10
    
    # reward scales
    rew_scale_terminated = -15.0
    rew_scale_distance = 0.02
    rew_scale_goal = 30.0
    rew_scale_timestep = 0.001
    rew_scale_pushervel = -0.1


class SlidingPandaGymEmbeddingEnv(DirectRLEnv):
    cfg: SlidingPandaGymEmbeddingEnvCfg

    def __init__(self, cfg: SlidingPandaGymEmbeddingEnvCfg, render_mode: str | None = None, **kwargs):
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

        # Recent episode success/failure tracking (1: success, 0: failure)
        self.goal_bounds = torch.zeros((self.scene.env_origins.shape[0]), dtype=torch.bool)
        self.inside_goal = torch.zeros((self.scene.env_origins.shape[0]), dtype=torch.bool)

        # Past puck velocity and acceleration tracking
        self.prev_puck_vel = torch.zeros((self.scene.env_origins.shape[0]), device=self.scene.env_origins.device)
        self.prev_puck_acc = torch.zeros_like(self.prev_puck_vel)
        
        # Goal location tensor tracking
        init_goal_location = torch.tensor([self.cfg.goal_location, 0.0, 1.0], device=self.scene.env_origins.device)
        self.goal_locations = init_goal_location.repeat(self.scene.env_origins.shape[0], 1)

        self.goal_length = self.cfg.goal_length
        self.success_threshold = 0.9
        self.maxgoal_locations = self.goal_locations[:,0]+(self.goal_length/2.0)-(self.cfg.puck_length/2.0)  # the cart is reset if it exceeds that position [m] (-0.7)
        self.mingoal_locations = (self.goal_locations[:,0]-(self.goal_length/2.0))+(self.cfg.puck_length/2.0)
        self.goal_threshold = 0.1
        
        # Goal randomisation range
        self.goal_location_min = 0.5
        self.goal_location_max = 0.9
        self.discrete_goals = torch.tensor([0.75], device=self.device)
        self.discrete_goal = True
        
        # Normalisaion range: goal
        self.goal_location_normmax = 2.0
        self.goal_location_normmin = -2.0
        
        # Normalisaion range: object
        self.object_location_normmax = self.goal_location_normmax
        self.object_location_normmin = self.goal_location_normmin
        self.object_vel_normmax = 4.0
        self.object_vel_normmin = -4.0

        # Success rate tracking
        self.success_rates = []

        # Past state tracking
        self.past_timestep = 1
        self.past_pusher_pos = torch.zeros((self.scene.env_origins.shape[0], self.past_timestep), device=self.scene.env_origins.device)
        self.past_puck_pos = torch.zeros((self.scene.env_origins.shape[0], self.past_timestep), device=self.scene.env_origins.device)
        self.past_pusher_vel = torch.zeros((self.scene.env_origins.shape[0], self.past_timestep), device=self.scene.env_origins.device)
        self.past_puck_vel = torch.zeros((self.scene.env_origins.shape[0], self.past_timestep), device=self.scene.env_origins.device)

        # Embeddings
        embedding_lookuptable_path = "/workspace/isaaclab/logs/exp_lookuptable/predefined/dynamicfriction_z2.pkl"
        with open(embedding_lookuptable_path, "rb") as fp: 
            self.embedding_lookuptable = pickle.load(fp)

        print(self.embedding_lookuptable.shape) 

        
        # model_params_path = "/workspace/isaaclab/logs/exp_model/exploration_sslmodel_params_dict.pkl"
        # with open(model_params_path, "rb") as fp: 
        #     model_params_dict = pickle.load(fp)

        # # model_params_path = "/workspace/isaaclab/logs/exp_model/exploration_sslmodel_params_dict.pkl"
        # model_params_path = "/workspace/isaaclab/logs/exp_model/exploration_sslmodel_params_dict.pkl"
        # with open(model_params_path, "rb") as fp: 
        #     model_params_dict = pickle.load(fp)

        # input_dim = model_params_dict["input_dim"]
        # latent_dim = model_params_dict["latent_dim"]
        # char_dim = model_params_dict["char_dim"]
        # output_dim = model_params_dict["output_dim"]
        # dropout = model_params_dict["dropout"]        
        # KLbeta = model_params_dict["KLbeta"]
        # rec_weight = model_params_dict["rec_weight"]
        # learning_rate = model_params_dict["learning_rate"]
        # model_path = model_params_dict["model_path"]
        # max_count = model_params_dict["max_count"]
        # exp_traj = model_params_dict["exp_traj"]
        
        # # Create VAE model
        # model = vaemodel.VAE(x_dim=input_dim, z_dim=latent_dim, char_dim=char_dim, y_dim=output_dim, dropout=dropout, beta=KLbeta, alpha=rec_weight).to(self.scene.env_origins.device)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # # Load the learnt model
        # model.load_state_dict(torch.load(model_path, map_location=torch.device(self.scene.env_origins.device)))

        # model.eval()

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
        goal_pos_offset[:, 0] = self.cfg.goal_location
        goal_pos_offset[:, 2] = 1.0
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


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print("Env pre-physics called!!!!")
        self.actions = self.action_scale * actions.clone()
        # print(self.actions.shape)
        # pass

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
        # self.past_pusher_pos = torch.cat((self.past_pusher_pos,  curr_cuboidpusher2_state[:, 0].reshape((-1,1))), dim=1)
        # past_pusher_pos_obs  = self.past_pusher_pos[:, -self.past_timestep:]
        # normalized_past_pusher_pos_obs = (past_pusher_pos_obs - self.object_location_normmin) / (self.object_location_normmax - self.object_location_normmin)
        
        normalized_curr_pusher_pos = (curr_cuboidpusher2_state[:, 0] - self.object_location_normmin) / (self.object_location_normmax - self.object_location_normmin)
        self.past_pusher_pos = torch.cat((self.past_pusher_pos, normalized_curr_pusher_pos.reshape((-1,1))), dim=1)
        normalized_past_pusher_pos_obs  = self.past_pusher_pos[:, -self.past_timestep:]

        # self.past_pusher_vel = torch.cat((self.past_pusher_vel,  curr_cuboidpusher2_state[:, 7].reshape((-1,1))), dim=1)
        # past_pusher_vel_obs  = self.past_pusher_vel[:, -self.past_timestep:]
        # normalized_past_pusher_vel_obs = (past_pusher_vel_obs - self.object_vel_normmin) / (self.object_vel_normmax - self.object_vel_normmin)

        normalized_curr_pusher_vel = (curr_cuboidpusher2_state[:, 7] - self.object_vel_normmin) / (self.object_vel_normmax - self.object_vel_normmin)
        self.past_pusher_vel = torch.cat((self.past_pusher_vel, normalized_curr_pusher_vel.reshape((-1,1))), dim=1)
        normalized_past_pusher_vel_obs  = self.past_pusher_vel[:, -self.past_timestep:]
        
        # Puck state
        curr_cylinderpuck2_state = self.cylinderpuck2_state.clone()
        curr_cylinderpuck2_state[:, 0:3] = (
            curr_cylinderpuck2_state[:, 0:3] - self.scene.env_origins
        )
        # self.past_puck_pos = torch.cat((self.past_puck_pos,  curr_cylinderpuck2_state[:, 0].reshape((-1,1))), dim=1)
        # past_puck_pos_obs  = self.past_puck_pos[:, -self.past_timestep:]
        # normalized_past_puck_pos_obs = (past_puck_pos_obs - self.object_location_normmin) / (self.object_location_normmax - self.object_location_normmin)

        normalized_curr_puck_pos = (curr_cylinderpuck2_state[:, 0] - self.object_location_normmin) / (self.object_location_normmax - self.object_location_normmin)
        self.past_puck_pos = torch.cat((self.past_puck_pos, normalized_curr_puck_pos.reshape((-1,1))), dim=1)
        normalized_past_puck_pos_obs  = self.past_puck_pos[:, -self.past_timestep:]
        
        # self.past_puck_vel = torch.cat((self.past_puck_vel,  curr_cylinderpuck2_state[:, 7].reshape((-1,1))), dim=1)
        # past_puck_vel_obs  = self.past_puck_vel[:, -self.past_timestep:]
        # normalized_past_puck_vel_obs = (past_puck_vel_obs - self.object_vel_normmin) / (self.object_vel_normmax - self.object_vel_normmin)
        
        normalized_curr_puck_vel = (curr_cylinderpuck2_state[:, 7] - self.object_vel_normmin) / (self.object_vel_normmax - self.object_vel_normmin)
        self.past_puck_vel = torch.cat((self.past_puck_vel, normalized_curr_puck_vel.reshape((-1,1))), dim=1)
        normalized_past_puck_vel_obs  = self.past_puck_vel[:, -self.past_timestep:]
        
        # Goal
        goal_tensor = self.goal_locations[:,0].clone()
        normalized_goal_tensor = (goal_tensor - self.goal_location_normmin) / (self.goal_location_normmax - self.goal_location_normmin)

        # Friction
        curr_materials = self.scene.rigid_objects["cylinderpuck2"].root_physx_view.get_material_properties()
        static_frictions = curr_materials.squeeze().reshape((-1,3))[:,0].to(self.scene.env_origins.device)
        dynamic_frictions = curr_materials.squeeze().reshape((-1,3))[:,1].to(self.scene.env_origins.device)
        restitutions = curr_materials.squeeze().reshape((-1,3))[:,2].to(self.scene.env_origins.device)

        curr_coms = self.scene.rigid_objects["cylinderpuck2"].root_physx_view.get_coms()
        com_x = curr_coms[:,0].to(self.scene.env_origins.device)
        com_y = curr_coms[:,1].to(self.scene.env_origins.device)
        com_z = curr_coms[:,2].to(self.scene.env_origins.device)
        com_min = -0.02
        com_max = 0.02
        normalized_com_x = (com_x - com_min) / (com_max - com_min)
        normalized_com_y = (com_y - com_min) / (com_max - com_min)
        normalized_com_z = (com_z - com_min) / (com_max - com_min)
        # dynamic_frictions = curr_materials.squeeze().reshape((-1,3))[:,1].to(self.scene.env_origins.device)
        # restitutions = curr_materials.squeeze().reshape((-1,3))[:,2].to(self.scene.env_origins.device)
        # print("Current materislaaass")
        # print(com_z)
        # torch.Size([32, 1, 3])

        # Embeddings
        print("Dynamic Frictionsssss: should be (32,1)")
        dynamic_frictions_np = dynamic_frictions.cpu().detach().numpy().reshape(self.num_envs, -1)
        print(dynamic_frictions_np.shape)
        print(self.embedding_lookuptable[['dynamic friction']].shape)

        neigh_dynamicfric = NearestNeighbors(n_neighbors=1)
        neigh_dynamicfric.fit(self.embedding_lookuptable[['dynamic friction']].to_numpy())

        dynamicfric_match_indices = neigh_dynamicfric.kneighbors(dynamic_frictions_np, return_distance=False)
        print("Look up table")
        print(self.embedding_lookuptable)
        dynamicfric_matches = self.embedding_lookuptable.iloc[dynamicfric_match_indices.flatten()][[0, 1]]
        dynamicfric_matches_np= dynamicfric_matches.to_numpy()

        # obs = torch.cat((normalized_past_puck_pos_obs, normalized_past_puck_vel_obs, normalized_past_pusher_pos_obs, normalized_past_pusher_vel_obs, normalized_goal_tensor.view(-1, 1), normalized_com_x.view(-1, 1), normalized_com_y.view(-1, 1)), dim=1)
        obs = torch.cat((normalized_past_puck_pos_obs, normalized_past_puck_vel_obs, normalized_past_pusher_pos_obs, normalized_past_pusher_vel_obs, normalized_goal_tensor.view(-1, 1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs, normalized_past_puck_vel_obs, normalized_past_pusher_pos_obs, normalized_past_pusher_vel_obs, normalized_goal_tensor.view(-1, 1), dynamic_frictions.view(-1,1)), dim=1)
        # obs = torch.cat((normalized_past_puck_pos_obs, normalized_past_puck_vel_obs, normalized_past_pusher_pos_obs, normalized_past_pusher_vel_obs, normalized_goal_tensor.view(-1, 1), static_frictions.view(-1,1), dynamic_frictions.view(-1,1), restitutions.view(-1,1)), dim=1)

        observations = {"policy": obs}

        return observations

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
            self.inside_goal, 
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

        # Check for Puck out of bound (i.e., out of table)
        curr_cylinderpuck2_state = self.cylinderpuck2_state.clone()
        curr_cylinderpuck2_state[:, 0:3] = (
            curr_cylinderpuck2_state[:, 0:3] - self.scene.env_origins
        )        
        out_of_bounds_max_puck_posx = curr_cylinderpuck2_state[:,0] > self.cfg.max_puck_posx
        out_of_bounds_min_puck_posx = curr_cylinderpuck2_state[:,0] < self.cfg.min_puck_posx

        # Check for Puck overshoot (i.e., over goal region) "Be careful the sign!!!!"
        overshoot_max_puck_posx = curr_cylinderpuck2_state[:,0] < self.mingoal_locations

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

        # Reset goal
        curr_success_rate = self.extras.get('log')
        if curr_success_rate is not None: 
            # print(curr_success_rate["success_rate"])  
            if curr_success_rate["success_rate"] > self.success_threshold and self.goal_threshold > 0.0499: 
                self.goal_threshold -= 0.01
                print(self.goal_threshold)
                # self.rew_scale_goal += 5
                # self.success_threshold += 0.1
                # self.goal_length -= 0.1 
                pass

        # curr_out_of_bounds_goal_puck_posx_count = euclid_distance < 0.15
        curr_out_of_bounds_goal_puck_posx_count = euclid_distance < self.goal_threshold
        # print(curr_out_of_bounds_goal_puck_posx_count)

        # print("INside goallll")
        # print(euclid_distance)

        self.inside_goal = euclid_distance < 0.05 

        # curr_out_of_bounds_goal_puck_posx_count = goal_bounds_max_puck_posx & goal_bounds_min_puck_posx 
        self.out_of_bounds_goal_puck_posx_count+= curr_out_of_bounds_goal_puck_posx_count.int()

        self.goal_bounds = self.out_of_bounds_goal_puck_posx_count>self.cfg.max_puck_goalcount
        self.out_of_bounds_goal_puck_posx_count[self.out_of_bounds_goal_puck_posx_count>self.cfg.max_puck_goalcount] = 0
        self.out_of_bounds_goal_puck_posx_count[~curr_out_of_bounds_goal_puck_posx_count] = 0

        out_of_bounds = out_of_bounds_max_pusher_posx | out_of_bounds_min_pusher_posx | out_of_bounds_max_puck_posx | out_of_bounds_min_puck_posx | overshoot_max_puck_posx | self.goal_bounds | out_of_bounds_min_puck_velx     

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

        episode_failed = out_of_bounds_max_pusher_posx | out_of_bounds_min_pusher_posx | out_of_bounds_max_puck_posx | out_of_bounds_min_puck_posx | overshoot_max_puck_posx | time_out | out_of_bounds_min_puck_velx

        return out_of_bounds, time_out, self.goal_bounds, episode_failed
        # return false_tensor, time_out, self.goal_bounds, episode_failed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # print("Env reset idx called!!!!")

        if env_ids is None:
            env_ids = self.cylinderpuck2._ALL_INDICES
        super()._reset_idx(env_ids)

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
        else:
            goal_noise = sample_uniform(self.goal_location_max, self.goal_location_min, (len(env_ids)), device=self.device)
        goal_pos_offset = self.goal_locations.clone()
        goal_pos_offset[env_ids, 0] = goal_noise

        self.goal_locations = goal_pos_offset.clone()
        self.maxgoal_locations = self.goal_locations[:,0]+(self.goal_length/2.0)-(self.cfg.puck_length/2.0)  # the cart is reset if it exceeds that position [m] (-0.7)
        self.mingoal_locations = (self.goal_locations[:,0]-(self.goal_length/2.0))+(self.cfg.puck_length/2.0)
        goal_pos_offset = goal_pos_offset.to(self.scene.env_origins.device)

        goal_pos = self.scene.env_origins + goal_pos_offset
        goal_pos = goal_pos.to(self.scene.env_origins.device)
        goal_rot = torch.zeros((self.scene.env_origins.shape[0], 4))
        goal_rot = goal_rot.to(self.scene.env_origins.device)
        self.markergoal1.visualize(goal_pos, goal_rot) 

        # Reset puck
        cylinderpuck2_default_state = self.cylinderpuck2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        cylinderpuck2_default_state[:, 0:3] = (
            cylinderpuck2_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        cylinderpuck2_default_state[:, 7:] = torch.zeros_like(self.cylinderpuck2.data.default_root_state[env_ids, 7:])
        # cylinderpuck2_default_state[:, 7] = -2.5
        self.cylinderpuck2_state[env_ids] = cylinderpuck2_default_state.clone()
        self.cylinderpuck2.write_root_state_to_sim(cylinderpuck2_default_state, env_ids)

        # Reset pusher
        cuboidpusher2_default_state = self.cuboidpusher2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
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
    inside_goal: torch.Tensor, 
):

    # Positive reward for reaching goal
    rew_goal = rew_scale_goal * goal_bounds.int()
    
    # Negative reward for failed episode
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # Small positive reward for lower distance between goal and puck
    # rew_distance = rew_scale_distance * (1.0 - normalised_curr_distance)
    rew_distance = -rew_scale_distance * normalised_curr_distance

    # Small negative reward for velocity/acc/jerk 
    # normalized_pushervel = torch.abs(jerk) / 4000.0
    # normalized_pushervel = torch.abs(acc) / 10.0
    normalized_pushervel = torch.abs(curr_vel) / 10.0
    modified_tensor = torch.where(normalized_pushervel < 0.01, torch.tensor(1.0), torch.tensor(0.0))
    rew_pushervel = rew_scale_pushervel * normalized_pushervel
    rew_pushervel0 = 0.05 * modified_tensor

    rew_inside_goal = inside_goal.int()
    rew_outside_goal = (1-inside_goal.int())*-1
    # print("Rew goallll")
    # print(rew_inside_goal)
    # print(rew_outside_goal)

    # total_reward = rew_inside_goal.float() + rew_outside_goal.float()
    # print(total_reward.shape)

    total_reward = rew_goal + rew_termination # + rew_distance # + rew_pushervel0 # + rew_pushervel0 # + rew_timestep

    return total_reward
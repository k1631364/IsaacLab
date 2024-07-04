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

from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg



# from omni.isaac.core.objects import DynamicCuboid
# import omni.isaac.core.utils.prims as prim_utils

# import omni.isaac.lab.sim as sim_utils
import numpy as np
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class SlidingEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    # # robot
    # robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # cart_dof_name = "slider_to_cart"
    # pole_dof_name = "cart_to_pole"

    # # cone object
    # cone_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Cone",
    #     spawn=sim_utils.ConeCfg(
    #         radius=0.1,
    #         height=0.2,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(),
    # )

    cuboidtable_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size = [2.0, 1.0, 1.0], 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.85, 0.9), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 0.0)),
    )    


    cuboidpuck_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Puck",
        spawn=sim_utils.CuboidCfg(
            size = [0.1, 0.2, 0.1], 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 1.05), rot=(0.0, 0.0, 0.0, 0.0)),
    ) 

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.CuboidCfg(
            size = [0.1, 0.2, 0.05], 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 1.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset

    # Puck
    puck_length = 0.1
    cuboidpuck2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboidpuck2",
        spawn=sim_utils.CuboidCfg(
            size = [puck_length, 0.2, 0.1], 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 1.075), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Pusher
    pusher_length = 0.1
    cuboidpusher2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboidpusher2",
        spawn=sim_utils.CuboidCfg(
            size = [pusher_length, 0.2, 0.05], 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.0, 1.075), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cuboidtable2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboidtable2",
        spawn=sim_utils.CuboidCfg(
            size = [2.0, 1.0, 1.0], 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.85, 0.9), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    spherepusher_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pusher",
        spawn=sim_utils.SphereCfg(
            radius = 0.025, 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 1.0625), rot=(0.0, 0.0, 0.0, 0.0)),
    ) 

    cuboidpusher_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CuboidPusher",
        spawn=sim_utils.CuboidCfg(
            size = [0.1, 0.2, 0.05], 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.0, 1.075), rot=(0.0, 0.0, 0.0, 0.0)),
    ) 

    # Goal
    goal_location = -0.45  # the cart is reset if it exceeds that position [m]
    gaol_length = 0.5
    max_goal_posx = (goal_location-(gaol_length/2.0))+(puck_length/2.0)  # the cart is reset if it exceeds that position [m] (-1.0)
    min_goal_posx = goal_location+(gaol_length/2.0)-(puck_length/2.0)  # the cart is reset if it exceeds that position [m] (-0.7)
    max_puck_goalcount = 20

    markergoal1_cfg = VisualizationMarkersCfg(
        prim_path="/Visual/Goal1",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(gaol_length, 1.0, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    
    markergoal2_cfg = VisualizationMarkersCfg(
        prim_path="/Visual/Goal2",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(0.3, 1.0, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )

    # Start region
    max_pusher_posx_bound = 1.0  # the cart is reset if it exceeds that position [m]
    min_pusher_posx_bound = 0.0  # the cart is reset if it exceeds that position [m]
    max_pusher_posx = max_pusher_posx_bound-(pusher_length/2.0)  # the cart is reset if it exceeds that position [m] (0.95)
    min_pusher_posx = min_pusher_posx_bound+(pusher_length/2.0)   # the cart is reset if it exceeds that position [m] (0.05)
    start_length = abs(max_pusher_posx_bound - min_pusher_posx_bound)
    start_location = (max_pusher_posx_bound - min_pusher_posx_bound)/2.0
    markerstart1_cfg = VisualizationMarkersCfg(
        prim_path="/Visual/Start1",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(start_length, 1.0, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        },
    )

    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.UsdFileCfg(
    #         # usd_path=f"Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=567.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # # Rigid Object
    # cone_cfg = RigidObjectCfg(
    #     # prim_path="/World/Origin.*/Cone",
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.ConeCfg(
    #         radius=0.1,
    #         height=0.2,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(),
    # )

    # # Rigid Object
    # box_cfg = RigidObjectCfg(
    #     # prim_path="/World/Origin.*/Cone",
    #     prim_path="/World/envs/env_.*/Table",
    #     spawn=sim_utils.CuboidCfg(
    #         size = [0.1, 0.2, 0.1], 
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
    #     ),
    #     # spawn=sim_utils.ConeCfg(
    #     #     radius=0.1,
    #     #     height=0.2,
    #     #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #     #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #     #     collision_props=sim_utils.CollisionPropertiesCfg(),
    #     #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     # ),
    #     init_state=RigidObjectCfg.InitialStateCfg(),
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # env
    # episode length is determined by decimation (e.g., 2), physics timestep (e.g., dt=1 / 120), and episode_length_s (e.g., 5)
    # In this case, 300 = ceil(5.0 / (2 * 1/120))
    # episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))
    # control_frequency = 1 / (decimation_rate * physics_time_step) lower control freq like 30Hz leads to more stable learning
    # See /workspace/isaaclab/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/rl_env_cfg.py for more details
    decimation = 5
    episode_length_s = 5.0
    # action_scale = 100.0  # [N]
    action_scale = 1.0
    num_actions = 1 # action dim
    num_observations = 2
    num_states = 2

    # # reset
    # max_pusher_posx = 0.9  # the cart is reset if it exceeds that position [m]
    # min_pusher_posx = 0.1  # the cart is reset if it exceeds that position [m]
    max_puck_posx = 0.9  # the cart is reset if it exceeds that position [m]
    min_puck_posx = -0.95  # the cart is reset if it exceeds that position [m]
    min_puck_velx = 0.01
    max_puck_restcount = 30
    # goal_location = -0.85  # the cart is reset if it exceeds that position [m]
    # gaol_length = 0.3
    # max_goal_posx = goal_location-(gaol_length/2.0)  # the cart is reset if it exceeds that position [m] (-1.0)
    # min_goal_posx = goal_location+(gaol_length/2.0)  # the cart is reset if it exceeds that position [m] (-0.7)
    # max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    # initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # # reward scales
    rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    rew_scale_terminated = -70.0
    rew_scale_distance = 0.01
    rew_scale_goal = 100.0
    rew_scale_timestep = 0.01
    rew_scale_pushervel = 1.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005

class SlidingEnv(DirectRLEnv):
    cfg: SlidingEnvCfg

    def __init__(self, cfg: SlidingEnvCfg, render_mode: str | None = None, **kwargs):
        # print("Env init called!!!!")
        super().__init__(cfg, render_mode, **kwargs)

        # self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        # self.joint_pos = self.cartpole.data.joint_pos
        # self.joint_vel = self.cartpole.data.joint_vel
        # self.puck_state = self.cuboidpuck.data.default_root_state
        # self.puck_state = self.cuboidpuck.data.root_state_w
        # self.cuboidpusher_state = self.cuboidpusher.data.root_state_w
        # 
        # print("Root state wwwwwwwwww")
        self.cuboidpusher2_state = self.cuboidpusher2.data.root_state_w.clone()
        # print(self.cuboidpusher2_state)
        self.cuboidpuck2_state = self.cuboidpuck2.data.root_state_w.clone()
        # print(self.cuboidpuck2_state)
        self.cuboidtable2_state = self.cuboidtable2.data.root_state_w.clone()
        # print(self.cuboidtable2_state)

        # # of envs = self.scene.env_origins.shape[0]
        self.out_of_bounds_min_puck_velx_count = torch.zeros((self.scene.env_origins.shape[0]))
        self.out_of_bounds_min_puck_velx_count = self.out_of_bounds_min_puck_velx_count.to(self.scene.env_origins.device)

        self.out_of_bounds_goal_puck_posx_count = torch.zeros((self.scene.env_origins.shape[0]))
        self.out_of_bounds_goal_puck_posx_count = self.out_of_bounds_goal_puck_posx_count.to(self.scene.env_origins.device)

        self.goal_bounds = torch.zeros((self.scene.env_origins.shape[0]), dtype=torch.bool)

    def _setup_scene(self):
        # print("Env setup scene called!!!!")
        # self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

        # self.cuboidpuck = RigidObject(cfg=self.cfg.cuboidpuck_cfg)
        # self.cuboidtable = RigidObject(cfg=self.cfg.cuboidtable_cfg)
        # self.spherepusher = RigidObject(cfg=self.cfg.spherepusher_cfg)
        # self.cuboidpusher = RigidObject(cfg=self.cfg.cuboidpusher_cfg)
        self.marker1 = VisualizationMarkers(cfg=self.cfg.markergoal1_cfg) 
        # self.marker2 = VisualizationMarkers(cfg=self.cfg.markergoal2_cfg)
        self.markerstart1 = VisualizationMarkers(cfg=self.cfg.markerstart1_cfg)
        # self.object = RigidObject(self.cfg.box_cfg)
        # self.cuboidpuck = RigidObject(self.cfg.cuboidpuck_cfg)
        # self.cuboidpusher = RigidObject(cfg=self.cfg.cuboidpusher_cfg)
        # self.object = RigidObject(self.cfg.object_cfg)
        self.cuboidpuck2 = RigidObject(self.cfg.cuboidpuck2_cfg)
        self.cuboidpusher2 = RigidObject(self.cfg.cuboidpusher2_cfg)
        self.cuboidtable2 = RigidObject(self.cfg.cuboidtable2_cfg)

        print("Env originannnnnn")
        print(self.scene.env_origins.shape)
        goal_pos_offset = torch.zeros(self.scene.env_origins.shape)
        goal_pos_offset[:, 0] = self.cfg.goal_location
        goal_pos_offset[:, 2] = 1.0
        goal_pos_offset = goal_pos_offset.to(self.scene.env_origins.device)
        goal_pos = self.scene.env_origins + goal_pos_offset
        goal_pos = goal_pos.to(self.scene.env_origins.device)
        goal_rot = torch.zeros((self.scene.env_origins.shape[0], 4))
        goal_rot = goal_rot.to(self.scene.env_origins.device)
        self.marker1.visualize(goal_pos, goal_rot) 

        # goal_pos_offset = torch.zeros((32, 3))
        # goal_pos_offset[:, 0] = -0.55
        # goal_pos_offset[:, 2] = 1.05
        # goal_pos_offset = goal_pos_offset.to(self.scene.env_origins.device)
        # goal_pos = self.scene.env_origins + goal_pos_offset
        # goal_pos = goal_pos.to(self.scene.env_origins.device)
        # goal_rot = torch.zeros((self.scene.env_origins.shape[0], 4))
        # goal_rot = goal_rot.to(self.scene.env_origins.device)
        # self.marker2.visualize(goal_pos, goal_rot) 

        goal_pos_offset = torch.zeros(self.scene.env_origins.shape)
        goal_pos_offset[:, 0] = self.cfg.start_location
        goal_pos_offset[:, 2] = 1.0
        goal_pos_offset = goal_pos_offset.to(self.scene.env_origins.device)
        goal_pos = self.scene.env_origins + goal_pos_offset
        goal_pos = goal_pos.to(self.scene.env_origins.device)
        goal_rot = torch.zeros((self.scene.env_origins.shape[0], 4))
        goal_rot = goal_rot.to(self.scene.env_origins.device)
        self.markerstart1.visualize(goal_pos, goal_rot) 

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        # self.scene.articulations["cartpole"] = self.cartpole
        # self.scene.rigid_objects["pusher"] = self.spherepusher
        # self.scene.rigid_objects["cuboidpusher"] = self.cuboidpusher
        # self.scene.rigid_objects["cuboidpuck"] = self.cuboidpuck
        # self.scene.rigid_objects["table"] = self.cuboidtable
        # self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["cuboidpuck2"] = self.cuboidpuck2
        self.scene.rigid_objects["cuboidpusher2"] = self.cuboidpusher2
        self.scene.rigid_objects["cuboidtable2"] = self.cuboidtable2
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # env_ids = torch.arange(0, 32).to(self.scene.env_origins.device)
        # self._reset_idx(env_ids)
        


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print("Env pre-physics called!!!!")
        self.actions = self.action_scale * actions.clone()
        # print("Actionnnnnnn")
        # print(self.actions.shape)
        # pass

    def _apply_action(self) -> None:
        # print("Env apply action called!!!!")
        # self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
        new_linvel = torch.zeros((self.scene.num_envs, 3))
        new_linvel = new_linvel.to(self.scene.env_origins.device)
        # new_linvel[:,0] = new_linvel[:,0]-0.2
        xvel = self.actions[:,0]
        new_linvel[:,0] = new_linvel[:,0]+xvel
        new_angvel = torch.zeros((self.scene.num_envs, 3))
        new_angvel = new_angvel.to(self.scene.env_origins.device)
        # self.spherepusher.set_velocities(new_linvel, new_angvel)
        self.cuboidpusher2.set_velocities(new_linvel, new_angvel)
        # self.spherepusher.write_data_to_sim()
        # pass

    def _get_observations(self) -> dict:
        # print("Env get observations called!!!!")
        # print(self.cuboidpusher2_state[0,0])

        curr_cuboidpusher2_state = self.cuboidpusher2_state.clone()
        # print(object_default_state.shape)
        # print(self.scene.env_origins.shape)
        # print(self.cuboidpusher2_state.shape)
        # global object positions
        curr_cuboidpusher2_state[:, 0:3] = (
            curr_cuboidpusher2_state[:, 0:3] - self.scene.env_origins
        )
        # curr_cuboidpusher2_state[:, 7:] = torch.zeros_like(self.cuboidpusher2.data.default_root_state[:, 7:])

        curr_cuboidpuck2_state = self.cuboidpuck2_state.clone()
        # print(object_default_state.shape)
        # print(self.scene.env_origins.shape)
        # print(self.cuboidpusher2_state.shape)
        # global object positions
        curr_cuboidpuck2_state[:, 0:3] = (
            curr_cuboidpuck2_state[:, 0:3] - self.scene.env_origins
        )
        # curr_cuboidpuck2_state[:, 7:] = torch.zeros_like(self.cuboidpuck2.data.default_root_state[:, 7:])
        # print(curr_cuboidpuck2_state[0,0])

        # obs = curr_cuboidpusher2_state[:,0]
        obs = torch.stack((curr_cuboidpusher2_state[:,0], curr_cuboidpuck2_state[:,0]), dim=1)

        observations = {"policy": obs}
        # print("Printing observations!!!!")
        # print(observations['policy'][0])
        # # # observations['policy'].shape = (# of environments, state dim)
        return observations

        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )

        # observations = {"policy": obs}
        # print("Printing observations!!!!")
        # print(observations['policy'].shape)
        # # observations['policy'].shape = (# of environments, state dim)
        # return observations
        # pass

    def _get_rewards(self) -> torch.Tensor:
        # print("Env get rewards called!!!!")

        curr_cuboidpuck2_state = self.cuboidpuck2_state.clone()
        curr_cuboidpuck2_state[:, 0:3] = (
            curr_cuboidpuck2_state[:, 0:3] - self.scene.env_origins
        )
        # curr_cuboidpuck2_state[:, 7:] = torch.zeros_like(self.cuboidpuck2.data.default_root_state[:, 7:])
        # print(curr_cuboidpuck2_state[0,0])

        curr_cuboidpusher2_state = self.cuboidpusher2_state.clone()
        curr_cuboidpusher2_state[:, 0:3] = (
            curr_cuboidpusher2_state[:, 0:3] - self.scene.env_origins
        )

        # print(curr_cuboidpusher2_state[:,7])

        # print("Pusher velocity!!!!!!!!")
        # print(curr_cuboidpusher2_state[0,7])

        # goal_bounds_max_puck_posx = curr_cuboidpuck2_state[:,0] > self.cfg.max_goal_posx
        # goal_bounds_min_puck_posx = curr_cuboidpuck2_state[:,0] < self.cfg.min_goal_posx

        # curr_out_of_bounds_goal_puck_posx_count = goal_bounds_max_puck_posx & goal_bounds_min_puck_posx 
        # self.out_of_bounds_goal_puck_posx_count+= curr_out_of_bounds_goal_puck_posx_count.int()

        # goal_bounds = self.out_of_bounds_goal_puck_posx_count>self.cfg.max_puck_goalcount
        # self.out_of_bounds_goal_puck_posx_count[self.out_of_bounds_goal_puck_posx_count>self.cfg.max_puck_goalcount] = 0
        # self.out_of_bounds_goal_puck_posx_count[~curr_out_of_bounds_goal_puck_posx_count] = 0

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_distance, 
            self.cfg.rew_scale_goal, 
            self.cfg.rew_scale_timestep, 
            self.cfg.rew_scale_pushervel, 
            # self.cfg.rew_scale_pole_pos,
            # self.cfg.rew_scale_cart_vel,
            # self.cfg.rew_scale_pole_vel,
            # self.joint_pos[:, self._pole_dof_idx[0]],
            # self.joint_vel[:, self._pole_dof_idx[0]],
            # self.joint_pos[:, self._cart_dof_idx[0]],
            # self.joint_vel[:, self._cart_dof_idx[0]],
            curr_cuboidpuck2_state[:,0], 
            curr_cuboidpusher2_state[:,7], 
            self.reset_terminated,
            self.cfg.goal_location, 
            # self.cfg.max_goal_posx,
            # self.cfg.min_goal_posx,
            self.goal_bounds, 
            self.episode_length_buf,
            self.max_episode_length, 
        )
        # total_reward = compute_rewards(
        #     self.cfg.rew_scale_alive,
        #     self.cfg.rew_scale_terminated,
        #     self.cfg.rew_scale_pole_pos,
        #     self.cfg.rew_scale_cart_vel,
        #     self.cfg.rew_scale_pole_vel,
        #     self.joint_pos[:, self._pole_dof_idx[0]],
        #     self.joint_vel[:, self._pole_dof_idx[0]],
        #     self.joint_pos[:, self._cart_dof_idx[0]],
        #     self.joint_vel[:, self._cart_dof_idx[0]],
        #     self.reset_terminated,
        # )
        # print("Reward!!!!")
        # print(total_reward.shape)
        # # total_reward.shape = (# of environments)
        # print(total_reward.shape)
        return total_reward
        # pass

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print("Env get dones called!!!!")
        self.cuboidpusher2_state = self.cuboidpusher2.data.root_state_w.clone()
        self.cuboidpuck2_state = self.cuboidpuck2.data.root_state_w.clone()
        # self.puck_state = self.cuboidpuck.data.root_state_w
        # self.cuboidpusher_state = self.cuboidpusher.data.root_state_w
        # # self.joint_pos = self.cartpole.data.joint_pos
        # # self.joint_vel = self.cartpole.data.joint_vel
        # # print(self.cuboidpusher_state[0,1])
        # # print(self.scene.env_origins)
        # self.cuboidpusher_state_local = self.cuboidpusher_state
        # self.cuboidpusher_state_local[:,0:3] -= self.scene.env_origins[:,0:3]
        # # # print(self.cuboidpusher_state[:,0])
        # print(self.cuboidpusher2_state[0,0])

        # Check for Pusher out of bound (i.e., out of pushing area)
        curr_cuboidpusher2_state = self.cuboidpusher2_state.clone()
        # # global object positions
        curr_cuboidpusher2_state[:, 0:3] = (
            curr_cuboidpusher2_state[:, 0:3] - self.scene.env_origins
        )
        # curr_cuboidpusher2_state[:, 7:] = torch.zeros_like(self.cuboidpusher2.data.default_root_state[:, 7:])
        
        out_of_bounds_max_pusher_posx = curr_cuboidpusher2_state[:,0] > self.cfg.max_pusher_posx
        out_of_bounds_min_pusher_posx = curr_cuboidpusher2_state[:,0] < self.cfg.min_pusher_posx

        # Check for Puck out of bound (i.e., out of table)
        curr_cuboidpuck2_state = self.cuboidpuck2_state.clone()
        # # global object positions
        curr_cuboidpuck2_state[:, 0:3] = (
            curr_cuboidpuck2_state[:, 0:3] - self.scene.env_origins
        )
        # curr_cuboidpuck2_state[:, 7:] = torch.zeros_like(self.cuboidpuck2.data.default_root_state[:, 7:])
        
        out_of_bounds_max_puck_posx = curr_cuboidpuck2_state[:,0] > self.cfg.max_puck_posx
        out_of_bounds_min_puck_posx = curr_cuboidpuck2_state[:,0] < self.cfg.min_puck_posx

        # Check for Puck overshoot (i.e., over goal region) "Be careful the sign!!!!"
        overshoot_max_puck_posx = curr_cuboidpuck2_state[:,0] < self.cfg.max_goal_posx

        # Check for Puck at rest (i.e., puck velocity is zero for sometime)
        # print("Puck velocityyyyyy")
        # print(curr_cuboidpuck2_state[:,7])
        # print(curr_cuboidpusher2_state[:,7])
        # print(abs(curr_cuboidpuck2_state[:,7])<0.01)
        
        curr_out_of_bounds_min_puck_velx_count = abs(curr_cuboidpuck2_state[:,7]) < self.cfg.min_puck_velx
        self.out_of_bounds_min_puck_velx_count += curr_out_of_bounds_min_puck_velx_count.int()

        out_of_bounds_min_puck_velx = self.out_of_bounds_min_puck_velx_count>self.cfg.max_puck_restcount
        self.out_of_bounds_min_puck_velx_count[self.out_of_bounds_min_puck_velx_count>self.cfg.max_puck_restcount] = 0
        self.out_of_bounds_min_puck_velx_count[~curr_out_of_bounds_min_puck_velx_count] = 0
        
        # print(self.out_of_bounds_min_puck_velx_count)
        # print(self.out_of_bounds_min_puck_velx_count>self.cfg.max_puck_restcount)
        # print(out_of_bounds_min_puck_velx)

        # puck_velocity = state['puck_velocity']  # Assuming 'puck_velocity' is part of state
        # if np.linalg.norm(puck_velocity) < self.threshold_velocity:
        #     self.consecutive_steps_count += 1
        # else:
        #     self.consecutive_steps_count = 0

        # Check if it reaches the goal
        goal_bounds_max_puck_posx = curr_cuboidpuck2_state[:,0] > self.cfg.max_goal_posx    # becomes false if overshoot
        goal_bounds_min_puck_posx = curr_cuboidpuck2_state[:,0] < self.cfg.min_goal_posx    # becomes true once in goal region

        curr_out_of_bounds_goal_puck_posx_count = goal_bounds_max_puck_posx & goal_bounds_min_puck_posx 
        self.out_of_bounds_goal_puck_posx_count+= curr_out_of_bounds_goal_puck_posx_count.int()

        self.goal_bounds = self.out_of_bounds_goal_puck_posx_count>=self.cfg.max_puck_goalcount
        self.out_of_bounds_goal_puck_posx_count[self.out_of_bounds_goal_puck_posx_count>self.cfg.max_puck_goalcount] = 0
        self.out_of_bounds_goal_puck_posx_count[~curr_out_of_bounds_goal_puck_posx_count] = 0

        # print("Goallllll")
        # if self.goal_bounds[0]:
        #     print(self.goal_bounds)

        # print("chekcing puck goal ")
        # print(self.out_of_bounds_goal_puck_posx_count)
        # print(self.cfg.max_puck_goalcount)
        # print(self.goal_bounds)

        # out_of_bounds = out_of_bounds_max_pusher_posx | out_of_bounds_min_pusher_posx | out_of_bounds_max_puck_posx | out_of_bounds_min_puck_posx | out_of_bounds_min_puck_velx | overshoot_max_puck_posx # | self.goal_bounds
        out_of_bounds = out_of_bounds_max_pusher_posx | out_of_bounds_min_pusher_posx | out_of_bounds_max_puck_posx | out_of_bounds_min_puck_posx | overshoot_max_puck_posx # | self.goal_bounds

        # print(out_of_bounds)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # # # out_of_bounds = torch.any(self.cuboidpusher_state_local[:, 0] > self.cfg.max_pusher_posx)
        # out_of_bounds_max_pusher_posx = self.cuboidpusher_state_local[:, 0] > self.cfg.max_pusher_posx
        # out_of_bounds_min_pusher_posx = self.cuboidpusher_state_local[:, 0] < self.cfg.min_pusher_posx
        # out_of_bounds = out_of_bounds_max_pusher_posx | out_of_bounds_min_pusher_posx
        # # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        # # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        # # print("Out of bounds")
        # # print(out_of_bounds.shape)
        # # print("timeout")
        # # print(time_out.shape)
        # # # out_of_bounds.shape, boolean of tensor (# of environments)
        # # # time_out.shape, boolean of tensor (# of environments)
        # # num_envs = self.joint_pos.shape[0]
        false_tensor = torch.zeros(self.scene.num_envs, dtype=torch.bool)
        true_tensor = torch.ones(self.scene.num_envs, dtype=torch.bool)
        # return out_of_bounds, time_out
        # return false_tensor, false_tensor
        # return true_tensor, true_tensor
        # print(self.cuboidpusher2_state[0,0])

        episode_failed = out_of_bounds_max_pusher_posx | out_of_bounds_min_pusher_posx | out_of_bounds_max_puck_posx | out_of_bounds_min_puck_posx | overshoot_max_puck_posx # | time_out

        return out_of_bounds, time_out, self.goal_bounds, episode_failed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # print("Env reset idx called!!!!")
        # print(env_ids)

        # # # print(self.spherepusher._ALL_INDICES)
        if env_ids is None:
            env_ids = self.cuboidpuck2._ALL_INDICES
        super()._reset_idx(env_ids)

        # # # Initialise cuboidpusher position
        # # cuboidpusher_default_state = self.cuboidpusher.data.default_root_state.clone()[env_ids]
        # # # test_sample = sample_uniform(
        # # #     -0.3,
        # # #     0.3,
        # # #     cuboidpusher_default_state[:, 1].shape,
        # # #     cuboidpusher_default_state.device,
        # # # )
        # # # object_default_state[:, 0] += 0.4
        # # # cuboidpusher_default_state[:, 0] += test_sample
        # # cuboidpusher_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        # # self.cuboidpusher_state = cuboidpusher_default_state
        # # print(cuboidpusher_default_state)
        # # self.cuboidpusher.write_root_state_to_sim(cuboidpusher_default_state, env_ids)

        # # Initialise puck position
        # object_default_state = self.cuboidpuck.data.default_root_state.clone()[env_ids]
        # test_sample = sample_uniform(
        #     -0.3,
        #     0.3,
        #     object_default_state[:, 1].shape,
        #     object_default_state.device,
        # )
        # object_default_state[:, 0] += -0.2
        # # object_default_state[:, 0] += test_sample
        # object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        # self.puck_state = object_default_state
        # self.cuboidpuck.write_root_state_to_sim(object_default_state, env_ids)

        # # Initialise pusher position
        # pusher_default_state = self.cuboidpusher.data.default_root_state.clone()[env_ids]
        # # test_sample = sample_uniform(
        # #     -0.3,
        # #     0.3,
        # #     pusher_default_state[:, 1].shape,
        # #     pusher_default_state.device,
        # # )
        # # pusher_default_state[:, 0] += 0.0
        # # pusher_default_state[:, 0] += test_sample
        # pusher_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        # self.cuboidpusher_state = pusher_default_state
        # self.cuboidpusher.write_root_state_to_sim(pusher_default_state, env_ids)

        # # reset object
        # object_default_state = self.object.data.default_root_state.clone()[env_ids]
        # pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        # # global object positions
        # object_default_state[:, 0:3] = (
        #     object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        # )
        # object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        # self.object.write_root_state_to_sim(object_default_state, env_ids)

        # # reset object
        # object_default_state = self.cuboidpuck2.data.default_root_state.clone()[env_ids]
        # pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        # # global object positions
        # object_default_state[:, 0:3] = (
        #     object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        # )
        # object_default_state[:, 7:] = torch.zeros_like(self.cuboidpuck2.data.default_root_state[env_ids, 7:])
        # self.cuboidpuck2.write_root_state_to_sim(object_default_state, env_ids)

        # reset object
        cuboidpuck2_default_state = self.cuboidpuck2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        # global object positions
        cuboidpuck2_default_state[:, 0:3] = (
            cuboidpuck2_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        cuboidpuck2_default_state[:, 7:] = torch.zeros_like(self.cuboidpuck2.data.default_root_state[env_ids, 7:])
        self.cuboidpuck2_state[env_ids] = cuboidpuck2_default_state.clone()
        self.cuboidpuck2.write_root_state_to_sim(cuboidpuck2_default_state, env_ids)

        # reset object
        cuboidpusher2_default_state = self.cuboidpusher2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        # global object positions
        cuboidpusher2_default_state[:, 0:3] = (
            cuboidpusher2_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        cuboidpusher2_default_state[:, 7:] = torch.zeros_like(self.cuboidpusher2.data.default_root_state[env_ids, 7:])
        self.cuboidpusher2_state[env_ids] = cuboidpusher2_default_state.clone()
        self.cuboidpusher2.write_root_state_to_sim(cuboidpusher2_default_state, env_ids)

        cuboidtable2_default_state = self.cuboidtable2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        # global object positions
        cuboidtable2_default_state[:, 0:3] = (
            cuboidtable2_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        cuboidtable2_default_state[:, 7:] = torch.zeros_like(self.cuboidtable2.data.default_root_state[env_ids, 7:])
        self.cuboidtable2_state[env_ids] = cuboidtable2_default_state.clone()
        self.cuboidtable2.write_root_state_to_sim(cuboidtable2_default_state, env_ids) 
        



        self.episode_length_buf[env_ids] = 0
        

        pass


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_distance: float, 
    rew_scale_goal: float, 
    rew_scale_timestep: float, 
    rew_scale_pushervel: float, 
    # rew_scale_pole_pos: float,
    # rew_scale_cart_vel: float,
    # rew_scale_pole_vel: float,
    # pole_pos: torch.Tensor,
    # pole_vel: torch.Tensor,
    # cart_pos: torch.Tensor,
    # cart_vel: torch.Tensor,
    curr_cuboidpuck2_state: torch.Tensor,  
    curr_cuboidpusher2_state: torch.Tensor,  
    reset_terminated: torch.Tensor,
    goal_location: float, 
    # max_goal_posx: float, 
    # min_goal_posx: float, 
    goal_bounds: torch.Tensor, 
    episode_length_buf: torch.Tensor,
    max_episode_length: float, 
):
    # print("Env compute rewards called!!!!")
    # print(max_goal_posx)
    # print(min_goal_posx)
    # print(curr_cuboidpuck2_state)
    # print(goal_location)
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    # distance_to_goal = torch.abs(curr_cuboidpuck2_state - goal_x_positions)
    rew_distance = rew_scale_distance * (1.0 - torch.abs(curr_cuboidpuck2_state - goal_location))

    rew_timestep = rew_scale_timestep * (1.0-(episode_length_buf/max_episode_length))

    # Check for Puck reaches goal (i.e., in goal region)
    # print("Check goal conditoinnnnnnn")
    # print(max_goal_posx)
    # print(min_goal_posx)
    # goal_bounds_max_puck_posx = curr_cuboidpuck2_state > max_goal_posx
    # goal_bounds_min_puck_posx = curr_cuboidpuck2_state < min_goal_posx

    # goal_bounds = goal_bounds_max_puck_posx & goal_bounds_min_puck_posx 
    rew_goal = rew_scale_goal * goal_bounds.int()
    # print(goal_bounds)

    normalized_pushervel = torch.abs(curr_cuboidpusher2_state) / 3.0
    rew_pushervel = rew_scale_pushervel * (1.0-normalized_pushervel)
    # print(rew_pushervel)


    # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    # total_reward = rew_alive + rew_termination 
    # total_reward = rew_alive + rew_termination + rew_goal
    total_reward = rew_goal + rew_termination + rew_distance # + rew_pushervel + rew_timestep
    # print(rew_distance)
    # print(curr_cuboidpuck2_state)
    # print(goal_location)
    # print(total_reward)
    return total_reward
    # print(total_reward)
    # pass

# @configclass
# class EventCfg:
#   robot_physics_material = EventTerm(
#       func=mdp.randomize_rigid_body_material,
#       mode="reset",
#       params={
#           "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#           "static_friction_range": (0.7, 1.3),
#           "dynamic_friction_range": (1.0, 1.0),
#           "restitution_range": (1.0, 1.0),
#           "num_buckets": 250,
#       },
#   )
#   robot_joint_stiffness_and_damping = EventTerm(
#       func=mdp.randomize_actuator_gains,
#       mode="reset",
#       params={
#           "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
#           "stiffness_distribution_params": (0.75, 1.5),
#           "damping_distribution_params": (0.3, 3.0),
#           "operation": "scale",
#           "distribution": "log_uniform",
#       },
#   )
#   reset_gravity = EventTerm(
#       func=mdp.randomize_physics_scene_gravity,
#       mode="interval",
#       is_global_time=True,
#       interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
#       params={
#           "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
#           "operation": "add",
#           "distribution": "gaussian",
#       },
#   )
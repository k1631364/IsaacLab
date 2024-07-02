# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


from omni.isaac.lab.sensors import RayCasterCfg, patterns

##
# Configuration
##

VELODYNE_VLP_16_CFG =  RayCasterCfg(
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(channels=16, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=0.2),
    debug_vis=True,
)
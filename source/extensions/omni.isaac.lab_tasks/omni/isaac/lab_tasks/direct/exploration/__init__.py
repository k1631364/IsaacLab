# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shuffleboard sliding environment.
"""

import gymnasium as gym

from . import agents
from .exploration_env import ExplorationEnv, ExplorationEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Exploration-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.exploration:ExplorationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExplorationEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


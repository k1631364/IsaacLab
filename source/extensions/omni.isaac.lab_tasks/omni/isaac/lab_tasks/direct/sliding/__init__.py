# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shuffleboard sliding environment.
"""

import gymnasium as gym

from . import agents
from .sliding_env import SlidingEnv, SlidingEnvCfg
from .sliding_long_env import SlidingLongEnv, SlidingLongEnvCfg
from .sliding_long_exp_env import SlidingLongExpEnv, SlidingLongExpEnvCfg
from .sliding_twophase_env import SlidingTwoPhaseEnv, SlidingTwoPhaseEnvCfg
from .sliding_pandagym_env import SlidingPandaGymEnv, SlidingPandaGymEnvCfg
from .sliding_pandagym_exp_env import SlidingPandaGymExpEnv, SlidingPandaGymExpEnvCfg
from .sliding_pandagym_embedding_env import SlidingPandaGymEmbeddingEnv, SlidingPandaGymEmbeddingEnvCfg
from .sliding_pandagym_prop_env import SlidingPandaGymPropEnv, SlidingPandaGymPropEnvCfg
from .sliding_pandagym_exp2_env import SlidingPandaGymExp2Env, SlidingPandaGymExp2EnvCfg

from . import run_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Sliding-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SlidingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sliding-Direct-v1",
    entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingLongEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SlidingLongEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sliding-Direct-v2",
    entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingLongExpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SlidingLongExpEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_exp_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sliding-Direct-v3",
    entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingTwoPhaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SlidingTwoPhaseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_twophase_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sliding-Direct-v4",
    entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingPandaGymEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SlidingPandaGymEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_franka_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sliding-Direct-v5",
    entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingPandaGymExpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SlidingPandaGymExpEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_exp_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sliding-Direct-v6",
    entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingPandaGymEmbeddingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SlidingPandaGymEmbeddingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sliding-Direct-v7",
    entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingPandaGymPropEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SlidingPandaGymPropEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_franka_cfg.yaml",
        "skrl_exp_cfg_entry_point": f"{agents.__name__}:skrl_ppo_preexp_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "run_env_cfg": "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/sliding/run_env_cfg/sliding_env_cfg.yaml"
    },
)

# gym.register(
#     id="Isaac-Sliding-Direct-v8",
#     entry_point="omni.isaac.lab_tasks.direct.sliding:SlidingPandaGymExp2Env",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": SlidingPandaGymExp2EnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_franka_cfg.yaml",
#         "skrl_exp_cfg_entry_point": f"{agents.__name__}:skrl_ppo_preexp_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
# )


# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.

# skrl in /workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/utils/wrappers/skrl.py
# PPO in /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/skrl/agents/torch/ppo/ppo.py
# PPO config for slider in /workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/sliding/agents/skrl_ppo_cfg.yaml
# To find pakcage location, find / -name "skrl" or find / -name "gymnasium"
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG #,PPO_RNN
from source.skrl_custom.ppo_rnn_prop import PPO_RNN_PROP
from source.skrl_custom.ppo_rnn_propexp import PPO_RNN_PROPEXP
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlSequentialLogTrainer, SkrlVecEnvWrapper, process_skrl_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl_rnn import SkrlSequentialLogTrainer_RNN #, SkrlVecEnvWrapper, process_skrl_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl_feedback import SkrlSequentialLogTrainer_Feedback #, SkrlVecEnvWrapper, process_skrl_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl_feedback_propexp import SkrlSequentialLogTrainer_FeedbackPropExp #, SkrlVecEnvWrapper, process_skrl_cfg

# from source.offline_learning.model import RNNPropertyEstimator 
# import source.offline_learning.model.RNNPropertyEstimator as rnnmodel

# def create_agent(experiment_cfg, env): 
#     print("exoeriment cfg")
#     print(experiment_cfg["models"]["policy"])
#     models = {}
#     # non-shared models
#     if experiment_cfg["models"]["separate"]:
#         models["policy"] = gaussian_model(
#             observation_space=env.observation_space,
#             action_space=env.action_space,
#             device=env.device,
#             **process_skrl_cfg(experiment_cfg["models"]["policy"]),
#         )
#         models["value"] = deterministic_model(
#             observation_space=env.observation_space,
#             action_space=env.action_space,
#             device=env.device,
#             **process_skrl_cfg(experiment_cfg["models"]["value"]),
#         )
#     # shared models
#     else:
#         models["policy"] = shared_model(
#             observation_space=env.observation_space,
#             action_space=env.action_space,
#             device=env.device,
#             structure=None,
#             roles=["policy", "value"],
#             parameters=[
#                 process_skrl_cfg(experiment_cfg["models"]["policy"]),
#                 process_skrl_cfg(experiment_cfg["models"]["value"]),
#             ],
#         )
#         models["value"] = models["policy"]

#     memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
#     memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)
#     # Memory to save all transitions
#     max_episode_length = env.unwrapped.max_episode_length
#     memory_all = RandomMemory(memory_size=max_episode_length*100, num_envs=env.num_envs, device=env.device)

#     agent_cfg = PPO_DEFAULT_CONFIG.copy()
#     experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
#     agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))
#     agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
#     agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

#     agent_cfg["prop_estimator"] = experiment_cfg["prop_estimator"]

#     agent = PPO_RNN_PROPEXP(
#         models=models,
#         memory=memory,
#         cfg=agent_cfg,
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         device=env.device,
#         # memory_all=memory_all, 
#     )

#     agent.init()
#     # agent.load(resume_path)
#     # set agent to evaluation mode
#     agent.set_running_mode("eval")

#     return agent

# Test comment (Working)
def main():

    """Train with skrl agent."""
    # read the seed from command line
    args_cli_seed = args_cli.seed

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    prop_experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_exp_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    experiment_cfg["agent"]["experiment"]["wandb_kwargs"]["name"] = log_dir
    
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        experiment_cfg["trainer"]["timesteps"] = args_cli.max_iterations * experiment_cfg["agent"]["rollouts"]

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`
    
    # set seed for the experiment (override from command line)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/api/utils/model_instantiators.html

    import copy
    exp_experiment_cfg = copy.deepcopy(experiment_cfg)
    exp_env = SkrlVecEnvWrapper(env)

    models = {}
    # non-shared models
    if experiment_cfg["models"]["separate"]:
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["policy"]),
        )
        models["value"] = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["value"]),
        )
    # shared models
    else:
        models["policy"] = shared_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(experiment_cfg["models"]["policy"]),
                process_skrl_cfg(experiment_cfg["models"]["value"]),
            ],
        )
        models["value"] = models["policy"]

    # instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    # https://skrl.readthedocs.io/en/latest/api/memories/random.html
    memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)
    # Memory to save all transitions
    max_episode_length = env.unwrapped.max_episode_length
    memory_all = RandomMemory(memory_size=max_episode_length*100, num_envs=env.num_envs, device=env.device)

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))
    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    agent_cfg["prop_estimator"] = experiment_cfg["prop_estimator"]

    agent = PPO_RNN_PROPEXP(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        # memory_all=memory_all, 
    )

    ### Exp model ###
    exp_models = {}
    # non-shared exp_models
    if exp_experiment_cfg["models"]["separate"]:
        exp_models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(exp_experiment_cfg["models"]["policy"]),
        )
        exp_models["value"] = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(exp_experiment_cfg["models"]["value"]),
        )
    # shared exp_models
    else:
        exp_models["policy"] = shared_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(exp_experiment_cfg["models"]["policy"]),
                process_skrl_cfg(exp_experiment_cfg["models"]["value"]),
            ],
        )
        exp_models["value"] = exp_models["policy"]

    exp_memory_size = exp_experiment_cfg["agent"]["rollouts"]  # exp_exp_memory_size is the agent's number of rollouts
    exp_memory = RandomMemory(memory_size=exp_memory_size, num_envs=env.num_envs, device=env.device)

    exp_agent_cfg = copy.deepcopy(agent_cfg)
    exp_agent_cfg["prop_estimator"] = prop_experiment_cfg["prop_estimator"]
    exp_agent = PPO_RNN_PROPEXP(
        models=exp_models,
        memory=exp_memory,
        cfg=exp_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        # memory_all=memory_all, 
    )

    ### Exp model ends ###

    # configure and instantiate a custom RL trainer for logging episode events
    # https://skrl.readthedocs.io/en/latest/api/trainers.html
    trainer_cfg = experiment_cfg["trainer"]
    #     cfg=agent_cfg,
    trainer = SkrlSequentialLogTrainer_FeedbackPropExp(cfg=trainer_cfg, env=env, agents=agent, exp_agent=exp_agent)

    print("Before expeirmnet start!!!!!!!!!!!!!!!!!!!!!!!")
    print("Log dir")
    print(log_dir)

    # train the agent
    trainer.train()

    # memory_path = os.path.join(log_dir, "memory", "ppo_memory.pt")
    # agent.memory.save(memory_path)
    
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    checkpoint_path = os.path.join(log_dir, "checkpoints", "best_agent.pt")
    print(f"Checkpoint: {checkpoint_path}")#
    
    # print(agent.memory)
    # agent.memory.save(memory_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
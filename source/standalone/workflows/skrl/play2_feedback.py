# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

from datetime import datetime

from source.skrl_custom.ppo_rnn_prop import PPO_RNN_PROP
from source.skrl_custom.ppo_rnn_propexp import PPO_RNN_PROPEXP

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg


def main():
    """Play with skrl agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    
    print(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'

    # print("Checkpoint path")
    # print("Video log dir")
    # print(log_dir)
    # # logs
    # import sys
    # sys.exit(0)
    # os.path.join("logs", "videos", log_dir, "videos", "train")

    if args_cli.video:
        video_kwargs = {
            # "video_folder": os.path.join(log_dir, "videos", "train"),
            "video_folder": os.path.join("logs", "videos", log_dir, "videos", "train"), 
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }

        print("[INFO] Recording videos during play.")

        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/api/utils/model_instantiators.html
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

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["experiment"]["write_interval"] = 0  # don't log to Tensorboard
    agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    agent_cfg["prop_estimator"] = experiment_cfg["prop_estimator"]
    agent_cfg["prop_estimator"]["train"] = False

    agent = PPO_RNN_PROPEXP(
        models=models,
        memory=None,  # memory is optional during evaluation
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # initialize agent
    agent.init()
    agent.load(resume_path)
    # set agent to evaluation mode
    agent.set_running_mode("eval")

    # # set default values
    # import copy
    # wandb_kwargs = copy.deepcopy(experiment_cfg)
    # wandb_kwargs.setdefault("name", os.path.split(log_root_path)[-1])
    # wandb_kwargs.setdefault("sync_tensorboard", True)
    # wandb_kwargs.setdefault("config", {})

    # init Weights & Biases
    import wandb
    run = wandb.init(
        # Set the project where this run will be logged
        project="",
        # Track hyperparameters and run metadata
        config={},
    )

    # reset environment
    obs, infos = env.reset()
    prev_total_episode_num = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # actions = agent.act(obs, timestep=0, timesteps=0)[0]
            actions, log_prob, outputs, prop_estimator_output = agent.act(obs, infos, timestep=0, timesteps=0)
            actions = outputs["mean_actions"]
            
            # get prop estimate
            prop_info = {}
            prop_info["prop_estimator_output"] = prop_estimator_output
            # print("Prop info")
            # print(prop_info)
            env._set_estimation(prop_info)

            if "prop_estimation" in infos: 
                print("curr rmse")
                print(infos["prop_estimation"])
                # print("Passed info")
                # print(prop_estimator_output)
                pass

            # env stepping
            obs, _, _, _, infos = env.step(actions)
            # print("Infos keys")
            # print(infos.keys())
            # print(prop_estimator_output.keys())
            # print("Current episode")
            # print(env.episode_length_buf)
            # print("Groundtruth")
            # print(infos["prop"][:,0]) 
            # print("Estimated")
            # print(prop_estimator_output["denormalsied_output"])
            # print("Should be same as GroundTruth")
            # print(prop_estimator_output["denormalsied_target"])
            # print(infos["prop"].shape)  # (num_envs, num_char)


            # infos["log"] = {"tttt": torch.tensor(3.14)}
            # print(infos)
            # if "log" in infos:
            #     for k, v in infos["log"].items():
            #         if isinstance(v, torch.Tensor) and v.numel() == 1:
            #             agent.track_data(f"EpisodeInfo / {k}", v.item())
            
            # agent.write_tracking_data()
            # print(infos["log"]["success_rate"])
            # wandb.log({"success_rate": infos["log"]["success_rate"]})

            total_episode_num = infos["log_eval"]["num_success"]+infos["log_eval"]["num_failure"]
            
            if total_episode_num!=0 and prev_total_episode_num!=total_episode_num:
                success_rate_1env = (infos["log_eval"]["num_success"]/(infos["log_eval"]["num_success"]+infos["log_eval"]["num_failure"]))*100.0
                print("Success num")
                print(total_episode_num)
                print(infos["log_eval"]["num_success"])
                print(infos["log_eval"]["num_failure"])
                print(success_rate_1env)
                # wandb.log({"Episode_num": total_episode_num})  
                # wandb.log({"success_rate": success_rate_1env}) 

                if "log" in infos and "end_rmse" in infos["log"]:
                    end_rmse = infos["log"]["end_rmse"]
                    # print(infos["log"]["end_rmse"])
                    # print(end_rmse)       
                    wandb.log({"episode_num": total_episode_num, "success_rate": success_rate_1env, "end_rmse": end_rmse})
                else: 
                    wandb.log({"episode_num": total_episode_num, "success_rate": success_rate_1env})
                prev_total_episode_num = total_episode_num 

            # print(total_episode_num)
            # if total_episode_num == 100:
            #     memory_path = os.path.join(log_dir, "memory", "ppo_memory.pt")
            #     agent.memory.save(memory_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

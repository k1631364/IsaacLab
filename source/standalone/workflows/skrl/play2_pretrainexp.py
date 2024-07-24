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

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
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
import torch
import time
import torch.optim as optim
import pickle

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from source.skrl.agents.torch.ppo.ppo_rnn_exp import PPO_RNN_EXP
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg

import source.offline_learning.model.VAE as vaemodel

def main():
    """Play with skrl agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
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

    agent = PPO_RNN_EXP(
        models=models,
        memory=None,  # memory is optional during evaluation
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

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

    # Load model
    model_params_path = "/workspace/isaaclab/logs/exp_model/exploration_sslmodel_params_dict.pkl"
    with open(model_params_path, "rb") as fp: 
        model_params_dict = pickle.load(fp)

    input_dim = model_params_dict["input_dim"]
    latent_dim = model_params_dict["latent_dim"]
    char_dim = model_params_dict["char_dim"]
    output_dim = model_params_dict["output_dim"]
    dropout = model_params_dict["dropout"]        
    KLbeta = model_params_dict["KLbeta"]
    rec_weight = model_params_dict["rec_weight"]
    learning_rate = model_params_dict["learning_rate"]
    model_path = model_params_dict["model_path"]
    max_count = model_params_dict["max_count"]
    exp_traj_action = model_params_dict["exp_traj"]
    
    print("Env infoooooo")
    print(env.exp_max_count)
    if env.exp_max_count != max_count or exp_traj_action.shape[0] != max_count:
        print("MAX COUNT FOR EXPLORATION NOT MATCHING")
        time.sleep(300)
    # print(if env.exp_episode_length_buf!=self.max_count: )

    exp_traj_action = exp_traj_action.reshape((max_count,-1))
    exp_traj_action = torch.from_numpy(exp_traj_action).to(torch.float32).to(env.device)

    z_embeddings = torch.zeros((env.num_envs, latent_dim)).to(env.device)


    # reset environment
    obs, infos = env.reset()
    prev_total_episode_num = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        curr_step = None
        if infos == {}:
            curr_step = torch.zeros(env.num_envs, dtype=torch.int).to(env.device)
        else:
            curr_step = infos["two_phase"]["episode_length_buf"]
            exp_traj = infos["two_phase"]["exp_traj"]
            # print("Exp traj")
            # print(curr_step)
            # print(exp_traj)
            # print(exp_traj.shape)
            check_exp_end_id = curr_step==max_count-1
            exp_traj_batch = exp_traj[check_exp_end_id]
            check_episode_end_id = curr_step==0
            z_embeddings[check_episode_end_id] = 0

            # Create VAE model
            model = vaemodel.VAE(x_dim=input_dim, z_dim=latent_dim, char_dim=char_dim, y_dim=output_dim, dropout=dropout, beta=KLbeta, alpha=rec_weight).to(env.device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            model.load_state_dict(torch.load(model_path, map_location=torch.device(env.device)))
                    
            # print("Exp end")
            # print(check_exp_end_id)
            # print(check_episode_end_id)
            if exp_traj_batch.shape[0] != 0: 
                x_char = torch.zeros(exp_traj_batch.shape[0])
                lower_bound, z, y, y_char = model(exp_traj_batch, x_char, exp_traj_batch, env.device)
                # print(z.shape)
                z_embeddings[check_exp_end_id, :] = z

                # Fig file path (to save)
                log_fig_eval_dir = os.path.join("logs", "exp_data", "exploration_rl")
                if not os.path.exists(log_fig_eval_dir):
                    os.makedirs(log_fig_eval_dir)
                
                # xpoints = np.arange(0, exp_traj_batch.shape[1])
                # ypoints = exp_traj_batch[0,:].cpu().detach().numpy()
                # plt.plot(xpoints, ypoints, label="dynamic friction: ")

                # plt.legend()
                # plt.xlabel("Time (step)")
                # plt.ylabel("Puck pos x (m)")

                # log_fig_path = os.path.join(log_fig_eval_dir, 'test3.png')
                # plt.savefig(log_fig_path)
                
            temp_states = obs.clone()
            temp_states[:, -z_embeddings.shape[1]:] = z_embeddings
            obs = temp_states

        mask = curr_step<max_count
        final_actions = torch.zeros(env.num_envs, env.action_space.shape[0], dtype=torch.float32).to(env.device)

        with torch.inference_mode():
            # agent stepping
            actions = agent.act(obs, timestep=0, timesteps=0)[0]
            
            final_actions[mask] = exp_traj_action[curr_step[mask]]
            final_actions[~mask] = actions[~mask]

            # env stepping
            obs, _, _, _, infos = env.step(final_actions)
            infos["phase"] = {"mask": mask}

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
                wandb.log({"Episode_num": total_episode_num})  
                wandb.log({"success_rate": success_rate_1env}) 
                wandb.log({"episode_num": total_episode_num, "success_rate": success_rate_1env})
                prev_total_episode_num = total_episode_num 




    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

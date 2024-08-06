# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

from skrl.memories.torch import RandomMemory

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import os
import time
from datetime import datetime
import numpy as np
import h5py

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

# from skrl.resources.preprocessors.torch import RunningStandardScaler 

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    log_root_path = os.path.join("logs", "exp_data", "predefined_slide")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    num_itr = 50
    
    max_episode_length = env.unwrapped.max_episode_length
    memory = RandomMemory(memory_size=(max_episode_length-1)*num_itr, num_envs=env.num_envs, device=env.device)

    observation_shape = env.observation_space.shape  # This should be (num_envs, 8)
    action_shape = env.action_space.shape  # This should be (num_envs, 1)
    # print(vars(env))
    # print(observation_shape)

    memory.create_tensor(name="states", size=observation_shape[1], dtype=torch.float32)
    memory.create_tensor(name="actions", size=action_shape[1], dtype=torch.float32)
    memory.create_tensor(name="rewards", size=env.num_envs, dtype=torch.float32)
    memory.create_tensor(name="terminated", size=env.num_envs, dtype=torch.bool)
    memory.create_tensor(name="truncated", size=env.num_envs, dtype=torch.bool)
    memory.create_tensor(name="log_prob", size=env.num_envs, dtype=torch.float32)
    memory.create_tensor(name="values", size=env.num_envs, dtype=torch.float32)
    memory.create_tensor(name="returns", size=env.num_envs, dtype=torch.float32)
    memory.create_tensor(name="advantages", size=env.num_envs, dtype=torch.float32)

    # print(memory)
    # print("Memory size check")
    # print(memory.get_tensor_by_name("states").shape)
    # print(memory.get_tensor_by_name("actions").shape)
    # print(memory.get_tensor_by_name("rewards").shape)
    # print(memory.get_tensor_by_name("terminated").shape)
    # print(memory.get_tensor_by_name("log_prob").shape)
    # print(memory.get_tensor_by_name("values").shape)
    # print(memory.get_tensor_by_name("returns").shape)
    # print(memory.get_tensor_by_name("advantages").shape)

    # running_standard_scaler = RunningStandardScaler(size=2)
    # data = torch.rand(3, 2)  # tensor of shape (N, 2)
    # running_standard_scaler(data)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    states_dict, infos = env.reset()
    states = states_dict['policy'].clone()
    print("State should look like thissssssssss")
    print(states.shape)
    # simulate environment
    count = 0
    max_count = (max_episode_length-1)*num_itr
    exp_traj = []
    while simulation_app.is_running() and count<max_count:
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            # pusher_velocity = -2.5 #(Working initially)
            pusher_velocity = -3.0
            # pusher_velocity_tensor = torch.tensor(pusher_velocity, device=env.unwrapped.device).clone()
            pusher_velocity_tensor = torch.full((env.num_envs, 1), pusher_velocity, device=env.unwrapped.device)
            actions = pusher_velocity_tensor.reshape((-1,1))

            if infos == {}:
                print("Noneeeee")
            else:
                mask = infos["two_phase"]["episode_length_buf"] > 0
                mask = mask.unsqueeze(1)  # Shape becomes [32, 1]
                actions[mask] = 0.0
                # print(infos["two_phase"]["episode_length_buf"] )

            if count<(max_episode_length-1):
                exp_traj.append(actions[0,0].item())

            # 2d action space (pad with zeros)
            zeros_tensor = torch.zeros(actions.shape[0], 1, device=env.unwrapped.device)
            actions = torch.cat((actions, zeros_tensor), dim=1)

            # print("Action space")
            # print(actions)
            # time.sleep(500)

            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            next_states_dict, rewards, terminated, truncated, infos = env.step(actions)

            next_states = next_states_dict['policy'].clone()

            # print("State shape")
            # print(states.shape)
            # print(actions.shape)
            # print(rewards.shape)
            # print(next_states.shape)
            # print(terminated.shape)
            # print(truncated.shape)
            # memory.add_samples(terminated=terminated)

            memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated)
            states = next_states

            # print("Memory")
            # tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages"]
            tensors_names = ["states", "actions"]
            # print(memory.sample_all(tensors_names))
            # if truncated[0]==True:
            #     print(memory.get_tensor_by_name("states"))

            # print("Memory size")
            # print(memory.sample_all(tensors_names))
            # print(memory.get_tensor_by_name("states").shape)
            # print(count)
            count+=1
    
    print("Exp traj shapeee")
    print(np.array(exp_traj).shape)
    log_h5_path = os.path.join(log_dir, 'test.hdf5')
    with h5py.File(log_h5_path, 'w') as f: 
        dset = f.create_dataset("states", data = memory.get_tensor_by_name("states").cpu().numpy())
        dset = f.create_dataset("truncated", data = memory.get_tensor_by_name("truncated").cpu().numpy())
        dset = f.create_dataset("exp_traj", data = np.array(exp_traj))
        dset = f.create_dataset("num_itr", data = num_itr)

    print("Log path")
    print(log_h5_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

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

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    # env.reset()
    # reset env
    states, infos = env.reset()

    control_dt = 5 * (1 / 120)
    puck_pos = ((states["policy"][:,0])*-1.0)+1.0
    pusher_vel = states["policy"][:,3]
    fric = states["policy"][:,5]
    goal_pos = ((states["policy"][:,4])*-1.0)+1.0
    
    initial_puck_position = ((states["policy"][:,0])*-1.0)+1.0
    initial_pusher_position = ((states["policy"][:,1])*-1.0)+1.0
    goal_position = ((states["policy"][:,4])*-1.0)+1.0
    mu_k = states["policy"][:,5]
    g = 9.81
    delta_t = 5 * (1 / 120)
    start_position = 1.0

    initial_puck_velocity = 0.0
    initial_pusher_velocity = 0.0
    
    distance_to_goal = goal_position - initial_puck_position

    import math
    v_required = math.sqrt(2.0*mu_k*g*distance_to_goal)

    interaction_distance = initial_puck_position - initial_pusher_position

    current_puck_position = initial_puck_position
    current_puck_velocity = initial_puck_velocity
    current_pusher_position = initial_pusher_position
    current_pusher_velocity = initial_pusher_velocity

    # print(states["policy"])
    # print(initial_puck_position)
    # print(initial_pusher_position)
    # print(goal_position)
    # print(mu_k)
    # print(g)
    # print(delta_t)
    # print(start_position)

    # simulate environment
    sim_count = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            # if current_puck_position<goal_position:

            #     # # Update the positions
            #     # current_puck_position += current_puck_velocity * delta_t
            #     # current_pusher_position += current_pusher_velocity * delta_t 

            #     # Update the velocities considering friction
            #     v_required -= mu_k * g * delta_t
            #     # if current_puck_velocity < 0:
            #     #     current_puck_velocity = 0
            #     #     print("Negative velocity")
            #     v_required = -1.0 * v_required

            #     # Ensure the pusher does not move beyond the start position
            #     if current_pusher_position >= start_position:
            #         current_pusher_position = start_position
            #         current_pusher_velocity = 0
            #         print("Pusher beyond start position")

            #     # Ensure the pusher stops at the start position
            #     elif current_pusher_position == start_position:
            #         current_pusher_velocity = 0
            #         print("Pusher at start position")

            #     # elif current_pusher_velocity < 0:
            #     #     current_pusher_velocity = 0
            #     #     print("Negative velocity")
            
            #     # If the pusher is in contact with the puck, apply the required velocity
            #     elif current_pusher_position + interaction_distance >= current_puck_position:
            #         current_pusher_velocity = v_required
            #         print("Commanding velocity")
            #         print(current_pusher_velocity)

            # else:
            #     current_pusher_velocity = 0
            #     print("Task finished")

            # print(current_pusher_velocity)
            # # current_pusher_velocity = -2.676940132397926
            # # current_pusher_velocity = -1.0*current_pusher_velocity

            current_pusher_velocity_tensor = torch.tensor(current_pusher_velocity, device=env.unwrapped.device).clone()
            actions = current_pusher_velocity_tensor.reshape((-1,1))
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            # Update the positions
            current_puck_position = ((next_states["policy"][:,0])*-1.0)+1.0
            current_pusher_position = ((next_states["policy"][:,1])*-1.0)+1.0
            


            # sample actions from -1 to 1
            # puck_pos = 0.1
            # pusher_vel = 0.1
            pusher_vel = -1.0*(pusher_vel-fric*(-9.81)*control_dt)*10.0
            # print(pusher_vel)
            # print("Pusher vel")
            # print(pusher_vel.shape)
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            # if sim_count<10:
            #     actions = pusher_vel.reshape((-1,1))
            # else: 
            #     actions = torch.zeros((states["policy"].shape[0],1), device=env.unwrapped.device)
            # actions = torch.zeros((states["policy"].shape[0],1), device=env.unwrapped.device)
            actions = pusher_vel.reshape((-1,1))
            actions = current_pusher_velocity_tensor.reshape((-1,1))
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            puck_pos = next_states["policy"][:,0]
            pusher_vel = next_states["policy"][:,3]   
            fric = next_states["policy"][:,5]
            goal_pos = next_states["policy"][:,4]  
            # sim_count+=1       

            

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

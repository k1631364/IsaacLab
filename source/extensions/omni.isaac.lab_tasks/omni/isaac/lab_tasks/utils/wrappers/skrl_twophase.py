# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env)

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env

    env = wrap_env(env, wrapper="isaac-orbit")

"""

# needed to import for type hinting: Agent | list[Agent]
from __future__ import annotations

import copy
import torch
import tqdm
import pickle
import torch.optim as optim
import time

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper, wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: F401
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: F401
from skrl.trainers.torch import Trainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.utils.model_instantiators.torch import Shape  # noqa: F401

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

import source.offline_learning.model.VAE as vaemodel

"""
Configuration Parser.
"""

def process_skrl_cfg(cfg: dict) -> dict:
    """Convert simple YAML types to skrl classes/components.

    Args:
        cfg: A configuration dictionary.

    Returns:
        A dictionary containing the converted configuration.
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "state_preprocessor",
        "value_preprocessor",
        "input_shape",
        "output_shape",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, timestep, timesteps):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)

        return d

    # parse agent configuration and convert to classes
    return update_dict(cfg)


"""
Vectorized environment wrapper.
"""


def SkrlVecEnvWrapper(env: ManagerBasedRLEnv):
    """Wraps around Isaac Lab environment for skrl.

    This function wraps around the Isaac Lab environment. Since the :class:`ManagerBasedRLEnv` environment
    wrapping functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Args:
        env: The environment to wrap around.

    Raises:
        ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/envs/wrapping.html
    """
    # check that input is valid
    if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
        raise ValueError(
            f"The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type: {type(env)}"
        )
    # wrap and return the environment
    return wrap_env(env, wrapper="isaac-orbit")


"""
Custom trainer for skrl.
"""


class SkrlSequentialLogTrainer_TwoPhase(Trainer):
    """Sequential trainer with logging of episode information.

    This trainer inherits from the :class:`skrl.trainers.base_class.Trainer` class. It is used to
    train agents in a sequential manner (i.e., one after the other in each interaction with the
    environment). It is most suitable for on-policy RL agents such as PPO, A2C, etc.

    It modifies the :class:`skrl.trainers.torch.sequential.SequentialTrainer` class with the following
    differences:

    * It also log episode information to the agent's logger.
    * It does not close the environment at the end of the training.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/trainers.html#base-class
    """

    def __init__(
        self,
        env: Wrapper,
        agents: Agent | list[Agent],
        agents_scope: list[int] | None = None,
        cfg: dict | None = None,
    ):
        """Initializes the trainer.

        Args:
            env: Environment to train on.
            agents: Agents to train.
            agents_scope: Number of environments for each agent to
                train on. Defaults to None.
            cfg: Configuration dictionary. Defaults to None.
        """
        # update the config
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        # store agents scope
        agents_scope = agents_scope if agents_scope is not None else []
        # initialize the base class
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)
        # init agents
        if self.env.num_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

        # Use gpu if available. Otherwise, use cpu. 
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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
        self.max_count = model_params_dict["max_count"]
        self.exp_traj_action = model_params_dict["exp_traj"]        

        print("Env infoooooo")
        print(env.exp_max_count)
        if env.exp_max_count != self.max_count or self.exp_traj_action.shape[0] != self.max_count:
            print("MAX COUNT FOR EXPLORATION NOT MATCHING")
            time.sleep(300)
        # print(if env.exp_episode_length_buf!=self.max_count: )

        self.exp_traj_action = self.exp_traj_action.reshape((self.max_count,-1))
        self.exp_traj_action = torch.from_numpy(self.exp_traj_action).to(torch.float32).to(self.torch_device)

        self.z_embeddings = torch.zeros((self.env.num_envs, latent_dim)).to(self.torch_device)

        # Create VAE model
        self.model = vaemodel.VAE(x_dim=input_dim, z_dim=latent_dim, char_dim=char_dim, y_dim=output_dim, dropout=dropout, beta=KLbeta, alpha=rec_weight).to(self.torch_device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.torch_device)))

    def train(self):
        """Train the agents sequentially.

        This method executes the training loop for the agents. It performs the following steps:

        * Pre-interaction: Perform any pre-interaction operations.
        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        * Post-interaction: Perform any post-interaction operations.
        * Reset the environments: Reset the environments if they are terminated or truncated.

        """
        # init agent
        self.agents.init(trainer_cfg=self.cfg)
        self.agents.set_running_mode("train")
        # reset env
        states, infos = self.env.reset()
        # training loop
        for timestep in tqdm.tqdm(range(self.timesteps), disable=self.disable_progressbar):
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            # compute actions
            curr_step = None
            if infos == {}:
                curr_step = torch.zeros(self.env.num_envs, dtype=torch.int).to(self.torch_device)
            else:
                curr_step = infos["two_phase"]["episode_length_buf"]
                exp_traj = infos["two_phase"]["exp_traj"]
                # print("Exp traj")
                # print(curr_step)
                # print(exp_traj)
                # print(exp_traj.shape)
                check_exp_end_id = curr_step==self.max_count-1
                exp_traj_batch = exp_traj[check_exp_end_id]
                check_episode_end_id = curr_step==0
                self.z_embeddings[check_episode_end_id] = 0
                # print("Exp end")
                # print(check_exp_end_id)
                # print(check_episode_end_id)
                if exp_traj_batch.shape[0] != 0: 
                    x_char = torch.zeros(exp_traj_batch.shape[0])
                    lower_bound, z, y, y_char = self.model(exp_traj_batch, x_char, exp_traj_batch, self.torch_device)
                    # print(z.shape)
                    self.z_embeddings[check_exp_end_id, :] = z
                    
                temp_states = states.clone()
                temp_states[:, -self.z_embeddings.shape[1]:] = self.z_embeddings
                states = temp_states

            mask = curr_step<self.max_count
            final_actions = torch.zeros(self.env.num_envs, self.env.action_space.shape[0], dtype=torch.float32).to(self.torch_device)

            with torch.no_grad():
                # print(states.shape)  # torch.Size([# of env, obs dim])
                # print(timestep)   # 0
                # print(self.timesteps)   # 1600
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                # actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
            # print(curr_step[mask])
            # masked_curr_step = curr_step[mask]
            final_actions[mask] = self.exp_traj_action[curr_step[mask]]
            final_actions[~mask] = actions[~mask]
            # print("Final actionnnn")
            # print(final_actions)
            # step the environments
            # next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            next_states, rewards, terminated, truncated, infos = self.env.step(final_actions)
            # note: here we do not call render scene since it is done in the env.step() method
            # record the environments' transitions
            with torch.no_grad():
                self.agents.record_transition(
                    states=states,
                    actions=final_actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )
            # log custom environment data
            # print("Infosssss")
            # print(infos)
            # if terminated[0,0]:
            #     print(terminated)
            # single_value_tensor = torch.tensor(3.14)
            # infos["log"] = {"tttt": single_value_tensor}
            # print(infos)
            if "log" in infos:
                for k, v in infos["log"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        self.agents.track_data(f"EpisodeInfo / {k}", v.item())
            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
            # reset the environments
            # note: here we do not call reset scene since it is done in the env.step() method
            # update states
            states.copy_(next_states)

    def eval(self) -> None:
        """Evaluate the agents sequentially.

        This method executes the following steps in loop:

        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        """
        # # set running mode
        # if self.num_agents > 1:
        #     for agent in self.agents:
        #         agent.set_running_mode("eval")
        # else:
        #     self.agents.set_running_mode("eval")
        # # single agent
        # if self.num_agents == 1:
        #     self.single_agent_eval()
        #     return

        self.single_agent_eval()

        # reset env
        states, infos = self.env.reset()
        # evaluation loop
        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):
            # compute actions
            with torch.no_grad():
                actions = torch.vstack([
                    agent.act(states[scope[0] : scope[1]], timestep=timestep, timesteps=self.timesteps)[0]
                    for agent, scope in zip(self.agents, self.agents_scope)
                ])

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            with torch.no_grad():
                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    # track data
                    agent.record_transition(
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )
                    # log custom environment data
                    if "log" in infos:
                        for k, v in infos["log"].items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                agent.track_data(k, v.item())
                    # perform post-interaction
                    super(type(agent), agent).post_interaction(timestep=timestep, timesteps=self.timesteps)

                # reset environments
                # note: here we do not call reset scene since it is done in the env.step() method
                states.copy_(next_states)

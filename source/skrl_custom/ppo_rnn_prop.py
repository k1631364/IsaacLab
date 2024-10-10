from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

import source.nn_models.RNN as rnnmodel
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib

def normalize(tensor, min_val, max_val, new_min, new_max):
    return (tensor - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

def denormalize(tensor, min_val, max_val, new_min, new_max):
    return (tensor - new_min) / (new_max - new_min) * (max_val - min_val) + min_val


# [start-config-dict-torch]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]

class PPO_RNN_PROP(Agent):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Proximal Policy Optimization (PPO) with support for Recurrent Neural Networks (RNN, GRU, LSTM, etc.)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """

        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                  lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor
        
        # print("Running ppo rnn proppppppp")
        # print(self.cfg["prop_estimator"]["input_size"])
        
        input_size = self.cfg["prop_estimator"]["input_size"]  # State and action concatenated size
        hidden_size = self.cfg["prop_estimator"]["hidden_size"]    # Number of features in hidden state
        num_layers = self.cfg["prop_estimator"]["num_layers"]      # Number of LSTM layers
        output_size = self.cfg["prop_estimator"]["output_size"]     # Number of physical properties (e.g., friction, CoM)
        self.num_epochs = self.cfg["prop_estimator"]["num_epochs"]
        prop_learning_rate = self.cfg["prop_estimator"]["learning_rate"]
        self.position_index = self.cfg["prop_estimator"]["position_index"]
        self.rotation_index = self.cfg["prop_estimator"]["rotation_index"]
        self.velocity_index = self.cfg["prop_estimator"]["velocity_index"]
        self.pos_min = self.cfg["prop_estimator"]["pos_min"]
        self.pos_max = self.cfg["prop_estimator"]["pos_max"]
        self.rot_min = self.cfg["prop_estimator"]["rot_min"]
        self.rot_max = self.cfg["prop_estimator"]["rot_max"] 
        self.vel_min = self.cfg["prop_estimator"]["vel_min"] 
        self.vel_max = self.cfg["prop_estimator"]["vel_max"] 
        self.feature_target_min = self.cfg["prop_estimator"]["feature_target_min"]   
        self.feature_target_max = self.cfg["prop_estimator"]["feature_target_max"]   
        self.fric_min = self.cfg["prop_estimator"]["fric_min"] 
        self.fric_max = self.cfg["prop_estimator"]["fric_max"] 
        self.estimate_target_min = self.cfg["prop_estimator"]["estimate_target_min"]   
        self.estimate_target_max = self.cfg["prop_estimator"]["estimate_target_max"]    

        # print(input_size)
        # print(learning_rate)

        self.prop_model = rnnmodel.RNN(input_size, hidden_size, num_layers, output_size).to(self.device)
        self.prop_criterion = nn.MSELoss()
        self.prop_optimizer = optim.Adam(self.prop_model.parameters(), lr=prop_learning_rate)

        if self.cfg["prop_estimator"]["train"]: 
            self.prop_model.train()
            print("Train prop model from scratchs")
        else: 
            # self.prop_model.eval()
            # # trained_model_path = "/workspace/isaaclab/logs/skrl/shortpushing_direct/2024-10-07_22-26-19/checkpoints_prop/LSTM_best.pth"
            # trained_model_path = "/workspace/isaaclab/logs/skrl/shortpushing_direct/2024-10-08_21-04-10/checkpoints_prop/LSTM_best.pth"
            # self.prop_model.load_state_dict(torch.load(trained_model_path, map_location=torch.device(self.device)))
            # print("Load prop model")
            pass

        # trained_model_path = "/workspace/isaaclab/logs/skrl/shortpushing_direct/2024-10-08_21-04-10/checkpoints_prop/LSTM_best.pth"   # short pushing
        # trained_model_path = "/workspace/isaaclab/logs/skrl/exploration_direct/2024-10-09_20-41-41/checkpoints_prop/LSTM_best.pth"  # exploration
        trained_model_path = "/workspace/isaaclab/logs/skrl/shortpushing_direct/2024-10-10_13-53-39/checkpoints_prop/LSTM_best.pth"
        self.prop_model.load_state_dict(torch.load(trained_model_path, map_location=torch.device(self.device)))
        print("Load prop model")

        self.prop_model.eval()

        self.curr_rollout_rnn_input = []
        self.curr_rollout_rnn_target = []

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # print("RNn param received in agent")
            # print(self.policy.get_rnn_param())
            # self.rnn_param = self.policy.get_rnn_param()
            # self.memory.create_tensor(name="state_hist", size=(self.rnn_param["rnn_num_envs"], self.rnn_param["rnn_sequence_length"], self.observation_space.shape[0]), dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "terminated", "log_prob", "values", "returns", "advantages"]

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"policy": [], "value": []}
        self._rnn_initial_states = {"policy": [], "value": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)

        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True)
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # value
        if self.value is not None:
            if self.policy is self.value:
                self._rnn_initial_states["value"] = self._rnn_initial_states["policy"]
            else:
                for i, size in enumerate(self.value.get_specification().get("rnn", {}).get("sizes", [])):
                    self._rnn = True
                    # create tensors in memory
                    if self.memory is not None:
                        self.memory.create_tensor(name=f"rnn_value_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True)
                        self._rnn_tensors_names.append(f"rnn_value_{i}")
                    # default RNN states
                    self._rnn_initial_states["value"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, infos: dict, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """

        curr_rnn_prop_input = infos["rnn_input"]
        # print("Current input")
        # print(curr_rnn_prop_input)

        normalized_curr_rnn_prop_input = curr_rnn_prop_input.clone()

        position = curr_rnn_prop_input[:, :, self.position_index]
        rotation = curr_rnn_prop_input[:, :, self.rotation_index]

        normalized_position = normalize(position, self.pos_min, self.pos_max, self.feature_target_min, self.feature_target_max)
        normalized_rotation = normalize(rotation, self.rot_min, self.rot_max, self.feature_target_min, self.feature_target_max)
        
        velocities = curr_rnn_prop_input[:, :, self.velocity_index]
        normalized_velocities = normalize(velocities, self.vel_min, self.vel_max, self.feature_target_min, self.feature_target_max)

        normalized_curr_rnn_prop_input[:, :, self.position_index] = normalized_position
        normalized_curr_rnn_prop_input[:, :, self.rotation_index] = normalized_rotation
        normalized_curr_rnn_prop_input[:, :, self.velocity_index] = normalized_velocities

        self.curr_rollout_rnn_input.append(normalized_curr_rnn_prop_input)
        # self.curr_rollout_rnn_input.append(curr_rnn_prop_input)

        curr_rnn_prop_target = infos["prop"][:,0].reshape(-1,1)
        frictions = curr_rnn_prop_target
        normalized_friction = normalize(frictions, self.fric_min, self.fric_max, self.estimate_target_min, self.estimate_target_max)
        normalized_curr_rnn_prop_target = normalized_friction

        self.curr_rollout_rnn_target.append(normalized_curr_rnn_prop_target)

        # print("Current shapeeeee")
        # print(curr_rnn_prop_input.shape)
        # print(curr_rnn_prop_target.shape)

        normalized_output = self.prop_model(normalized_curr_rnn_prop_input)

        denormalsied_output = denormalize(normalized_output, self.fric_min, self.fric_max, self.estimate_target_min, self.estimate_target_max)
        denormalsied_target = denormalize(normalized_curr_rnn_prop_target, self.fric_min, self.fric_max, self.estimate_target_min, self.estimate_target_max)

        # mean_friction = 0.5
        # weight = 1+ torch.abs(normalized_curr_rnn_prop_target-mean_friction)
        # rnn_loss = torch.mean(weight*(normalized_output-normalized_curr_rnn_prop_target)**2)
        
        rnn_loss = self.prop_criterion(normalized_output, normalized_curr_rnn_prop_target)
        rnn_rmse = torch.sqrt(self.prop_criterion(denormalsied_output, curr_rnn_prop_target))
        # print(output)
        # print(loss)

        # print("Estimation loss")
        # print(denormalsied_target)
        # print("Denormalised output")
        # print(denormalsied_output)
        # print("Normalised output")
        # print(normalized_output)
        # print(rnn_rmse)

        prop_estimator_output = {
            "rnn_loss": rnn_loss, 
            "rnn_rmse": rnn_rmse, 
            "normalized_output": normalized_output, 
            "denormalsied_output": denormalsied_output, 
            "denormalsied_target": denormalsied_target
        }

        # with torch.no_grad():
        #     for batch_idx, (inputs, targets) in enumerate(test_loader):
        #         inputs = inputs.to(torch_device)
        #         targets = targets.to(torch_device)
        #         # targets = targets[:, -1, :]
        #         targets = targets[:, -1, 0].view(-1,1)
        #         for i in range(len(inputs)):
        #             input = inputs[i].unsqueeze(0)  # Add batch dimension
        #             target = targets[i].unsqueeze(0)  # Add batch dimension

        #             output = model(input)
        #             loss = criterion(output, target)

    
        
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}

        # sample random actions
        # TODO: fix for stochasticity, rnn and log_prob
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states), **rnn}, role="policy")

        # sample stochastic actions
        # print("State processorrrrrr")
        # print(self._state_preprocessor)
        # print(self._state_preprocessor(states)[0,0])
        # print(states[0,0]) 

        # print("act shapeeeee")
        # print(self._state_preprocessor(states).shape)

        # print("Agent act called")
        actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states), **rnn}, role="policy")
        self._current_log_prob = log_prob

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

        return actions, log_prob, outputs, prop_estimator_output

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            values, _, outputs = self.value.act({"states": self._state_preprocessor(states), **rnn}, role="value")
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update({f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])})
                if self.policy is not self.value:
                    rnn_states.update({f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])})

            # storage transition in memory
            # print("Check add sampleessss")
            # print(states.shape)
            # print(actions.shape)
            # print(rnn_states.keys())
            # print(rnn_states["rnn_policy_0"].shape)
            # print(rnn_states["rnn_policy_1"].shape)
            # print(rnn_states["rnn_value_0"].shape)
            # print(rnn_states["rnn_value_1"].shape)
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values, **rnn_states)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values, **rnn_states)

        # update RNN states
        if self._rnn:
            self._rnn_final_states["value"] = self._rnn_final_states["policy"] if self.policy is self.value else outputs.get("rnn", [])

            # reset states if the episodes have ended
            finished_episodes = terminated.nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0
                if self.policy is not self.value:
                    for rnn_state in self._rnn_final_states["value"]:
                        rnn_state[:, finished_episodes[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.prop_model.train()
            self._update_prop_estimator(timestep, timesteps)
            self.prop_model.eval()
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            import os
            log_model_dir = os.path.join(self.experiment_dir, "checkpoints_prop", "LSTM_best.pth")
            log_model_dir = os.path.join(self.experiment_dir, "checkpoints_prop")
            if not os.path.exists(log_model_dir):
                os.makedirs(log_model_dir)
            best_model_path = log_model_dir + "/LSTM_best.pth"
            torch.save(self.prop_model.to(self.device).state_dict(), best_model_path)

            curr_model_path = log_model_dir + "/LSTM_"+str(timestep)+".pth"
            torch.save(self.prop_model.to(self.device).state_dict(), curr_model_path)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        def compute_gae(rewards: torch.Tensor,
                        dones: torch.Tensor,
                        values: torch.Tensor,
                        next_values: torch.Tensor,
                        discount_factor: float = 0.99,
                        lambda_coefficient: float = 0.95) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad():
            self.value.train(False)
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            last_values, _, _ = self.value.act({"states": self._state_preprocessor(self._current_next_states.float()), **rnn}, role="value")
            self.value.train(True)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        # print("Memoryyyyy")
        # print(self.memory.get_tensor_by_name("rewards").shape)
        returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                          dones=self.memory.get_tensor_by_name("terminated"),
                                          values=values,
                                          next_values=last_values,
                                          discount_factor=self._discount_factor,
                                          lambda_coefficient=self._lambda)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        # print("Check mini batches")
        # print(self._mini_batches)
        # print(self._rnn_sequence_length)
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)
            # print("Check sample batch shape")
            # # first mini-batch = sampled_batches[0]
            # curr_mini_batch = sampled_batches[0]
            # curr_mini_batch_states = curr_mini_batch[0]
            # print(curr_mini_batch_states.shape)
        # import sys
        # sys.exit(0)

        rnn_policy, rnn_value = {}, {}
        if self._rnn:
            # print("Mini batch rnn")
            # print(self._mini_batches)
            # print(self._rnn_sequence_length)
            sampled_rnn_batches = self.memory.sample_all(names=self._rnn_tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)
            # print("RNN bacthess")
            # print(sampled_rnn_batches[0][0].shape)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            # test_sample = sampled_batches[i][0]
            # print(test_sample.shape)
            for i, (sampled_states, sampled_actions, sampled_dones, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages) in enumerate(sampled_batches):
                # print("Sampled states")
                # print(sampled_states.shape)
                # print(i)

                import numpy as np
                xpoints = np.arange(0, self._mini_batches) 

                if self._rnn:
                    # print("RNN enableddd!!")
                    if self.policy is self.value:
                        rnn_policy = {"rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]], "terminated": sampled_dones}
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "policy" in n], "terminated": sampled_dones}
                        rnn_value = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "value" in n], "terminated": sampled_dones}

                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                # print("update shapeeeee")
                # print(sampled_states.shape)
                # print("Agent update called")
                _, next_log_prob, _ = self.policy.act({"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="policy")

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                predicted_values, _, _ = self.value.act({"states": sampled_states, **rnn_value}, role="value")

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip)
                self.optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    
    def _update_prop_estimator(self, timestep: int, timesteps: int) -> None:
        # print("Prop estimator update")

        # print(len(self.curr_rollout_rnn_target))
        # print(self.curr_rollout_rnn_target[13].shape)

        # print("Current rnn prop input xxx")
        # print(len(self.curr_rollout_rnn_input))
        # print(self.curr_rollout_rnn_input[14].shape)

        # print(self.num_epochs)
        
        for epoch in range(self.num_epochs):
            for i, rnn_input in enumerate(self.curr_rollout_rnn_input): 
                # print("RNN input shapeeeeeeeeeeeeeeeeeeee")
                # print(rnn_input.shape)
                # print(i)
                # print(self.curr_rollout_rnn_target[i].shape)

                targets = self.curr_rollout_rnn_target[i]
                
                outputs = self.prop_model(rnn_input)

                # mean_friction = 0.5
                # weight = 1+ torch.abs(targets-mean_friction)
                # loss = torch.mean(weight*(outputs-targets)**2)
                
                loss = self.prop_criterion(outputs, targets)
                
                # Backward pass and optimization
                self.prop_optimizer.zero_grad()
                loss.backward()
                self.prop_optimizer.step()
            
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # # Record average loss after each epoch to weights and biases
            # wandb.log({"loss": loss.item()})



        self.curr_rollout_rnn_input = []
        self.curr_rollout_rnn_target = []

        # rewards = self.memory.get_tensor_by_name("rewards")
        # states = self.memory.get_tensor_by_name("states")

        # print(rewards.shape)
        # print(states.shape)
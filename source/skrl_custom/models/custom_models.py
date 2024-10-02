from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import textwrap
import gym
import gymnasium

import torch
import torch.nn as nn  # noqa

# from skrl.models.torch import GaussianMixin  # noqa
# from skrl.models.torch import Model

from skrl.models.torch import Model  # noqa
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, MultivariateGaussianMixin, MultiCategoricalMixin  # noqa

from skrl.utils.model_instantiators.torch.common import convert_deprecated_parameters, generate_containers

activations = {
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}

def build_sequential_network(inputs, hiddens, outputs, hidden_activation, output_activation):
    layers = []

    # First hidden layer: from inputs to the first hidden layer
    layers.append(nn.Linear(inputs, hiddens[0]))
    layers.append(activations[hidden_activation[0]])  # Add activation

    # Hidden layers: loop over hidden layers
    for i in range(len(hiddens) - 1):
        layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
        layers.append(activations[hidden_activation[i + 1]])  # Add activation

    # Output layer: from the last hidden layer to the output
    layers.append(nn.Linear(hiddens[-1], outputs))
    # layers.append(activations[output_activation])

    if output_activation and output_activation in activations:
        layers.append(activations[output_activation])  # Add output activation if specified

    return nn.Sequential(*layers)

def custom_gaussian_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   device: Optional[Union[str, torch.device]] = None,
                   clip_actions: bool = False,
                   clip_log_std: bool = True,
                   min_log_std: float = -20,
                   max_log_std: float = 2,
                   initial_log_std: float = 0,
                   network: Sequence[Mapping[str, Any]] = [],
                   output: Union[str, Sequence[str]] = "",
                   return_source: bool = False,
                   *args,
                   **kwargs) -> Union[Model, str]:
    """Instantiate a Gaussian model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
    :type clip_log_std: bool, optional
    :param min_log_std: Minimum value of the log standard deviation (default: -20)
    :type min_log_std: float, optional
    :param max_log_std: Maximum value of the log standard deviation (default: 2)
    :type max_log_std: float, optional
    :param initial_log_std: Initial value for the log standard deviation (default: 0)
    :type initial_log_std: float, optional
    :param network: Network definition (default: [])
    :type network: list of dict, optional
    :param output: Output expression (default: "")
    :type output: list or str, optional
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Gaussian model instance or definition source
    :rtype: Model
    """

    class GaussianModel(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions,
                        clip_log_std, min_log_std, max_log_std, metadata, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

            self.num_observations = observation_space.shape[0]
            self.num_actions = action_space.shape[0]

            hiddens = metadata["hiddens"]
            hidden_activation = metadata["hidden_activation"]
            output_activation = metadata["output_activation"]
            
            # self.test_nn = build_sequential_network(self.num_observations, hiddens, self.num_actions, hidden_activation, output_activation)

            self.net_container = nn.Sequential(
                nn.Linear(self.num_observations, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, self.num_actions)
            )

            self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(self.num_actions))

        def compute(self, inputs, role=""):

            # states = inputs["states"]

            # output = self.test_nn(states)    

            output = self.net_container(inputs["states"])        
    
            return output, self.log_std_parameter, {}

    metadata = {
        "input_shape": kwargs.get('input_shape', None), 
        "hiddens": kwargs.get('hiddens', None), 
        "hidden_activation": kwargs.get('hidden_activation', None), 
        "output_shape": kwargs.get('output_shape', None), 
        "output_activation": kwargs.get('output_activation', None), 
        "output_scale": kwargs.get('output_scale', None), 
        "initial_log_std": kwargs.get('initial_log_std', None), 
    }

    return GaussianModel(observation_space=observation_space,
                                    action_space=action_space,
                                    device=device,
                                    clip_actions=clip_actions,
                                    clip_log_std=clip_log_std,
                                    min_log_std=min_log_std,
                                    max_log_std=max_log_std, 
                                    metadata=metadata)

def custom_gaussian_model2(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   device: Optional[Union[str, torch.device]] = None,
                   clip_actions: bool = False,
                   clip_log_std: bool = True,
                   min_log_std: float = -20,
                   max_log_std: float = 2,
                   initial_log_std: float = 0,
                   network: Sequence[Mapping[str, Any]] = [],
                   output: Union[str, Sequence[str]] = "",
                   return_source: bool = False,
                   *args,
                   **kwargs) -> Union[Model, str]:
    """Instantiate a Gaussian model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
    :type clip_log_std: bool, optional
    :param min_log_std: Minimum value of the log standard deviation (default: -20)
    :type min_log_std: float, optional
    :param max_log_std: Maximum value of the log standard deviation (default: 2)
    :type max_log_std: float, optional
    :param initial_log_std: Initial value for the log standard deviation (default: 0)
    :type initial_log_std: float, optional
    :param network: Network definition (default: [])
    :type network: list of dict, optional
    :param output: Output expression (default: "")
    :type output: list or str, optional
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Gaussian model instance or definition source
    :rtype: Model
    """

    class GaussianModel(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions,
                        clip_log_std, min_log_std, max_log_std, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

            self.net_container = nn.Sequential(
                nn.LazyLinear(out_features=256),
                nn.ELU(),
                nn.LazyLinear(out_features=128),
                nn.ELU(),
                nn.LazyLinear(out_features=64),
                nn.ELU(),
                # nn.LazyLinear(out_features=2),
                nn.Linear(64, self.num_actions)
            )

            self.log_std_parameter = nn.Parameter(0 * torch.ones(2))

        def compute(self, inputs, role=""):
            output = self.net_container(inputs["states"])
            return output, self.log_std_parameter, {}

    return GaussianModel(observation_space=observation_space,
                                    action_space=action_space,
                                    device=device,
                                    clip_actions=clip_actions,
                                    clip_log_std=clip_log_std,
                                    min_log_std=min_log_std,
                                    max_log_std=max_log_std)



def custom_gaussian_model_rnn(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   device: Optional[Union[str, torch.device]] = None,
                   clip_actions: bool = False,
                   clip_log_std: bool = True,
                   min_log_std: float = -20,
                   max_log_std: float = 2,
                   initial_log_std: float = 0,
                   network: Sequence[Mapping[str, Any]] = [],
                   output: Union[str, Sequence[str]] = "",
                   return_source: bool = False,
                   *args,
                   **kwargs) -> Union[Model, str]:
    """Instantiate a Gaussian model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
    :type clip_log_std: bool, optional
    :param min_log_std: Minimum value of the log standard deviation (default: -20)
    :type min_log_std: float, optional
    :param max_log_std: Maximum value of the log standard deviation (default: 2)
    :type max_log_std: float, optional
    :param initial_log_std: Initial value for the log standard deviation (default: 0)
    :type initial_log_std: float, optional
    :param network: Network definition (default: [])
    :type network: list of dict, optional
    :param output: Output expression (default: "")
    :type output: list or str, optional
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Gaussian model instance or definition source
    :rtype: Model
    """


    rnn_hidden_size = kwargs.get('rnn_hidden_size', None)
    rnn_num_layers = kwargs.get('rnn_num_layers', None)
    rnn_num_envs = kwargs.get('num_envs', None)
    rnn_sequence_length = kwargs.get('rnn_sequence_length', None)
    rnn_param = {
        "rnn_hidden_size": rnn_hidden_size, 
        "rnn_num_layers": rnn_num_layers, 
        "rnn_num_envs": rnn_num_envs, 
        "rnn_sequence_length": rnn_sequence_length
    }

    class GaussianModel(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions,
                        clip_log_std, min_log_std, max_log_std, rnn_param, metadata, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

            print("observation_space")
            print(observation_space.shape[0])

            # print(rnn_param)

            self.rnn_param = rnn_param
            
            self.num_observations = observation_space.shape[0]
            self.num_actions = action_space.shape[0]
            self.feature_extractor_size=568
            self.sequence_length = self.rnn_param["rnn_sequence_length"]
            self.num_envs = self.rnn_param["rnn_num_envs"]
            self.num_layers = self.rnn_param["rnn_num_layers"]
            self.hidden_size = self.rnn_param["rnn_hidden_size"]
            print(self.sequence_length)
            print(self.num_envs)
            print(self.num_layers)
            print(self.hidden_size)

            print("NUm pbervation")
            print(self.num_observations)
            print(self.feature_extractor_size)

            self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(self.num_actions))

            print("Inptu shape")
            print(metadata)
            hiddens = metadata["hiddens"]
            hidden_activation = metadata["hidden_activation"]
            output_activation = metadata["output_activation"]

            self.test_nn = build_sequential_network(self.num_observations, hiddens, self.num_actions, hidden_activation, output_activation)

            self.feature_extractor = nn.Sequential(nn.Linear(self.num_observations, self.feature_extractor_size),
                                                nn.Tanh())

            self.lstm = nn.LSTM(input_size=self.feature_extractor_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)

            self.post_lstm = nn.Sequential(nn.Linear(self.hidden_size, 568),
                                    nn.Tanh(),
                                    nn.Linear(568, self.num_actions))

            self.mlp_nn = nn.Sequential(nn.Linear(self.num_observations, 568, ),
                                                nn.Linear(568, self.num_actions))

        def get_specification(self):
            return {"rnn": {"sequence_length": self.sequence_length,
                            "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
                                    (self.num_layers, self.num_envs, self.hidden_size)]}}

        def compute(self, inputs, role=""):

            states = inputs["states"]
            terminated = inputs.get("terminated", None)
            hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

            # print("Memory input checkkkkk")
            # print(self.training)
            # print(states.shape)
            # if terminated is not None: 
            #     print("terminated")
            #     print(terminated.shape)
            # print(hidden_states.shape)
            # print(cell_states.shape)

            # print("state shape")
            # print(states.shape)

            features_states = self.feature_extractor(states)

            # print("Feature state shapeee")
            # print(features_states.shape)

            # batch_size = self.num_envs  # Number of parallel environments
            features_states = features_states.view(-1, 1, self.feature_extractor_size)  # shape: (batch_size, 1, input_size)
            # print("features_states shape")
            # print(features_states.shape)

            if terminated is not None and torch.any(terminated):
                terminated_indices = terminated.squeeze().nonzero(as_tuple=True)[0]  # Get indices of terminated environments
                
                # Reset the states for terminated environments
                hidden_states[0, terminated_indices, :] = 0
                cell_states[0, terminated_indices, :] = 0

            # rnn_output, (self.hidden_states, self.cell_states) = self.lstm(features_states, (self.hidden_states, self.cell_states))
            # rnn_output, rnn_states = self.lstm(features_states, (hidden_states, cell_states))
            rnn_output, (hidden_states, cell_states) = self.lstm(features_states, (hidden_states.contiguous(), cell_states.contiguous()))

            rnn_states = (hidden_states, cell_states)
            
            # print("rnn_output shape")
            # print(rnn_output.shape)
            # print(rnn_states[0].shape)
            # print(rnn_states[1].shape)
    
            output = self.post_lstm(rnn_output)
            output = torch.squeeze(output)

            return output, self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

            # states = inputs["states"]
            # terminated = inputs.get("terminated", None)
            # hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

            # features_states = self.feature_extractor(states)

            # if self.training:            
            #     rnn_input = features_states.view(-1, self.sequence_length, features_states.shape[-1])
            #     hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])
            #     cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])
                
            #     # hidden and cell states for initial sequence
            #     hidden_states = hidden_states[:,:,0,:].contiguous()
            #     cell_states = cell_states[:,:,0,:].contiguous()

            #     # check if RNN state needs to be reset
            #     if terminated is not None and torch.any(terminated):
            #         rnn_outputs = []
            #         terminated = terminated.view(-1, self.sequence_length)
            #         indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

            #         for i in range(len(indexes) - 1):
            #             i0, i1 = indexes[i], indexes[i + 1]
            #             rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
            #             hidden_states[:, (terminated[:,i1-1]), :] = 0
            #             cell_states[:, (terminated[:,i1-1]), :] = 0
            #             rnn_outputs.append(rnn_output)

            #         rnn_states = (hidden_states, cell_states)
            #         rnn_output = torch.cat(rnn_outputs, dim=1)
            #     else:
            #         rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
            # else:
            #     rnn_input = features_states.view(-1, 1, features_states.shape[-1])
            #     rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

            # rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

            # return self.post_lstm(rnn_output), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

    metadata = {
        "input_shape": kwargs.get('input_shape', None), 
        "hiddens": kwargs.get('hiddens', None), 
        "hidden_activation": kwargs.get('hidden_activation', None), 
        "output_shape": kwargs.get('output_shape', None), 
        "output_activation": kwargs.get('output_activation', None), 
        "output_scale": kwargs.get('output_scale', None), 
        "initial_log_std": kwargs.get('initial_log_std', None), 
    }

    return GaussianModel(observation_space=observation_space,
                                    action_space=action_space,
                                    device=device,
                                    clip_actions=clip_actions,
                                    clip_log_std=clip_log_std,
                                    min_log_std=min_log_std,
                                    max_log_std=max_log_std, 
                                    rnn_param=rnn_param, 
                                    metadata=metadata)

# def custom_gaussian_model_rnn2(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                    action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                    device: Optional[Union[str, torch.device]] = None,
#                    clip_actions: bool = False,
#                    clip_log_std: bool = True,
#                    min_log_std: float = -20,
#                    max_log_std: float = 2,
#                    initial_log_std: float = 0,
#                    network: Sequence[Mapping[str, Any]] = [],
#                    output: Union[str, Sequence[str]] = "",
#                    return_source: bool = False,
#                    *args,
#                    **kwargs) -> Union[Model, str]:
#     """Instantiate a Gaussian model

#     :param observation_space: Observation/state space or shape (default: None).
#                               If it is not None, the num_observations property will contain the size of that space
#     :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
#     :param action_space: Action space or shape (default: None).
#                          If it is not None, the num_actions property will contain the size of that space
#     :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
#     :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
#                    If None, the device will be either ``"cuda"`` if available or ``"cpu"``
#     :type device: str or torch.device, optional
#     :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
#     :type clip_actions: bool, optional
#     :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
#     :type clip_log_std: bool, optional
#     :param min_log_std: Minimum value of the log standard deviation (default: -20)
#     :type min_log_std: float, optional
#     :param max_log_std: Maximum value of the log standard deviation (default: 2)
#     :type max_log_std: float, optional
#     :param initial_log_std: Initial value for the log standard deviation (default: 0)
#     :type initial_log_std: float, optional
#     :param network: Network definition (default: [])
#     :type network: list of dict, optional
#     :param output: Output expression (default: "")
#     :type output: list or str, optional
#     :param return_source: Whether to return the source string containing the model class used to
#                           instantiate the model rather than the model instance (default: False).
#     :type return_source: bool, optional

#     :return: Gaussian model instance or definition source
#     :rtype: Model
#     """


#     rnn_hidden_size = kwargs.get('rnn_hidden_size', None)
#     rnn_num_layers = kwargs.get('rnn_num_layers', None)
#     rnn_num_envs = kwargs.get('num_envs', None)
#     rnn_sequence_length = kwargs.get('rnn_sequence_length', None)
#     rnn_param = {
#         "rnn_hidden_size": rnn_hidden_size, 
#         "rnn_num_layers": rnn_num_layers, 
#         "rnn_num_envs": rnn_num_envs, 
#         "rnn_sequence_length": rnn_sequence_length
#     }

#     class GaussianModel(GaussianMixin, Model):
#         def __init__(self, observation_space, action_space, device, clip_actions,
#                         clip_log_std, min_log_std, max_log_std, reduction="sum"):
#             Model.__init__(self, observation_space, action_space, device)
#             GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#             rnn_hidden_size = kwargs.get('rnn_hidden_size', None)
#             rnn_num_layers = kwargs.get('rnn_num_layers', None)
#             rnn_num_envs = kwargs.get('num_envs', None)
#             rnn_sequence_length = kwargs.get('rnn_sequence_length', None)
            
#             self.net_container1 = nn.Sequential(
#                 nn.LazyLinear(out_features=128),
#                 nn.ELU(),
#             )

#             self.lstm = nn.LSTM(input_size=self.feature_extractor_size,
#                                 hidden_size=self.hidden_size,
#                                 num_layers=self.num_layers,
#                                 batch_first=True)

#             self.net_container2 = nn.Sequential(
#                 nn.LazyLinear(out_features=64), 
#                 nn.ELU(),
#                 nn.LazyLinear(out_features=self.num_actions), 
#             )

#             self.log_std_parameter = nn.Parameter(0 * torch.ones(2))

#         def get_specification(self):
#             return {"rnn": {"sequence_length": self.sequence_length,
#                             "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
#                                     (self.num_layers, self.num_envs, self.hidden_size)]}}

#         def compute(self, inputs, role=""):

#             states = inputs["states"]
#             terminated = inputs.get("terminated", None)
#             hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

#             features_states = self.net_container1(states)

#             features_states = features_states.view(-1, 1, self.feature_extractor_size)  # shape: (batch_size, 1, input_size)

#             if terminated is not None and torch.any(terminated):
#                 terminated_indices = terminated.squeeze().nonzero(as_tuple=True)[0]  # Get indices of terminated environments
                
#                 # Reset the states for terminated environments
#                 hidden_states[0, terminated_indices, :] = 0
#                 cell_states[0, terminated_indices, :] = 0

#             rnn_output, (hidden_states, cell_states) = self.lstm(features_states, (hidden_states.contiguous(), cell_states.contiguous()))

#             rnn_states = (hidden_states, cell_states)
            
#             output = self.post_lstm(rnn_output)
#             output = torch.squeeze(output)

#             return output, self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

#     return GaussianModel(observation_space=observation_space,
#                                     action_space=action_space,
#                                     device=device,
#                                     clip_actions=clip_actions,
#                                     clip_log_std=clip_log_std,
#                                     min_log_std=min_log_std,
#                                     max_log_std=max_log_std, 
#                                     rnn_param=rnn_param)


def custom_deterministic_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        clip_actions: bool = False,
                        network: Sequence[Mapping[str, Any]] = [],
                        output: Union[str, Sequence[str]] = "",
                        return_source: bool = False,
                        *args,
                        **kwargs) -> Union[Model, str]:
    """Instantiate a deterministic model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param network: Network definition (default: [])
    :type network: list of dict, optional
    :param output: Output expression (default: "")
    :type output: list or str, optional
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Deterministic model instance or definition source
    :rtype: Model
    """

    class DeterministicModel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions, metadata):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions) 

            self.num_observations = observation_space.shape[0]
            self.num_actions = action_space.shape[0]

            hiddens = metadata["hiddens"]
            hidden_activation = metadata["hidden_activation"]
            output_activation = metadata["output_activation"]
            output_shape = metadata["output_shape"].value

            # self.test_nn = build_sequential_network(self.num_observations, hiddens, output_shape, hidden_activation, output_activation)

            self.net_container = nn.Sequential(
                nn.Linear(self.num_observations, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, output_shape)
            )

        def compute(self, inputs, role=""):

            # states = inputs["states"]

            # output = self.test_nn(states)

            output = self.net_container(inputs["states"])    

            return output, {}

    metadata = {
        "input_shape": kwargs.get('input_shape', None), 
        "hiddens": kwargs.get('hiddens', None), 
        "hidden_activation": kwargs.get('hidden_activation', None), 
        "output_shape": kwargs.get('output_shape', None), 
        "output_activation": kwargs.get('output_activation', None), 
        "output_scale": kwargs.get('output_scale', None), 
        "initial_log_std": kwargs.get('initial_log_std', None), 
    }

    return DeterministicModel(observation_space=observation_space,
                                         action_space=action_space,
                                         device=device,
                                         clip_actions=clip_actions, 
                                         metadata=metadata)


def custom_deterministic_model_rnn(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        clip_actions: bool = False,
                        network: Sequence[Mapping[str, Any]] = [],
                        output: Union[str, Sequence[str]] = "",
                        return_source: bool = False,
                        *args,
                        **kwargs) -> Union[Model, str]:
    """Instantiate a deterministic model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param network: Network definition (default: [])
    :type network: list of dict, optional
    :param output: Output expression (default: "")
    :type output: list or str, optional
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Deterministic model instance or definition source
    :rtype: Model
    """
 
    rnn_hidden_size = kwargs.get('rnn_hidden_size', None)
    rnn_num_layers = kwargs.get('rnn_num_layers', None)
    rnn_num_envs = kwargs.get('num_envs', None)
    rnn_sequence_length = kwargs.get('rnn_sequence_length', None)
    rnn_param = {
        "rnn_hidden_size": rnn_hidden_size, 
        "rnn_num_layers": rnn_num_layers, 
        "rnn_num_envs": rnn_num_envs, 
        "rnn_sequence_length": rnn_sequence_length
    }

    class DeterministicModel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions, rnn_param, metadata):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions) 

            self.rnn_param = rnn_param

            self.num_observations = observation_space.shape[0]
            self.num_actions = action_space.shape[0]
            self.feature_extractor_size=568
            self.sequence_length = self.rnn_param["rnn_sequence_length"]
            self.num_envs = self.rnn_param["rnn_num_envs"]
            self.num_layers = self.rnn_param["rnn_num_layers"]
            self.hidden_size = self.rnn_param["rnn_hidden_size"]
            print(self.sequence_length)
            print(self.num_envs)
            print(self.num_layers)
            print(self.hidden_size)

            print("NUm pbervation")
            print(self.num_observations)
            print(self.feature_extractor_size)

            # self.feature_extractor_size=568
            # self.sequence_length = self.rnn_param["rnn_sequence_length"]
            # self.num_envs = self.rnn_param["rnn_num_envs"]
            # self.num_layers = self.rnn_param["rnn_num_layers"]
            # self.hidden_size = self.rnn_param["rnn_hidden_size"]
            # print(self.sequence_length)
            # print(self.num_envs)
            # print(self.num_layers)
            # print(self.hidden_size)

            # print("NUm pbervation")
            # print(self.num_observations)
            # print(self.feature_extractor_size)

            # hiddens = metadata["hiddens"]
            # hidden_activation = metadata["hidden_activation"]
            # output_activation = metadata["output_activation"]
            # output_shape = metadata["output_shape"].value

            # self.test_nn = build_sequential_network(self.num_observations, hiddens, output_shape, hidden_activation, output_activation)

            print("Inptu shape")
            print(metadata)
            hiddens = metadata["hiddens"]
            hidden_activation = metadata["hidden_activation"]
            output_activation = metadata["output_activation"]
            output_shape = metadata["output_shape"].value

            self.test_nn = build_sequential_network(self.num_observations, hiddens, output_shape, hidden_activation, output_activation)

            self.feature_extractor = nn.Sequential(nn.Linear(self.num_observations, self.feature_extractor_size),
                                                nn.Tanh())

            self.lstm = nn.LSTM(input_size=self.feature_extractor_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)

            self.post_lstm = nn.Sequential(nn.Linear(self.hidden_size, 568),
                                    nn.Tanh(),
                                    nn.Linear(568, output_shape))

            self.mlp_nn = nn.Sequential(nn.Linear(self.num_observations, 568, ),
                                                nn.Linear(568, output_shape))

        def get_specification(self):
            return {"rnn": {"sequence_length": self.sequence_length,
                            "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
                                    (self.num_layers, self.num_envs, self.hidden_size)]}}
        
        def compute(self, inputs, role=""):

            # states = inputs["states"]

            # output = self.test_nn(states)

            # return output, {}

            states = inputs["states"]
            terminated = inputs.get("terminated", None)
            hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

            # print("State shape")
            # print(states.shape)

            features_states = self.feature_extractor(states)

            # print("Feature state shapeee")
            # print(features_states.shape)

            batch_size = self.num_envs  # Number of parallel environments
            features_states = features_states.view(-1, 1, self.feature_extractor_size)  # shape: (batch_size, 1, input_size)

            if terminated is not None and torch.any(terminated):
                terminated_indices = terminated.squeeze().nonzero(as_tuple=True)[0]  # Get indices of terminated environments
                
                # Reset the states for terminated environments
                hidden_states[0, terminated_indices, :] = 0
                cell_states[0, terminated_indices, :] = 0

            # rnn_output, (self.hidden_states, self.cell_states) = self.lstm(features_states, (self.hidden_states, self.cell_states))
            rnn_output, rnn_states = self.lstm(features_states, (hidden_states.contiguous(), cell_states.contiguous()))

            # print("RNN shape")
            # print(rnn_output.shape)

            rnn_output = torch.squeeze(rnn_output)
    
            output = self.post_lstm(rnn_output)
            # output = torch.squeeze(output)

            # print("Check value output size")
            # print(output.shape)
            # print(rnn_states[0].shape)
            
            # print("Output shapeeee")
            # print(output.shape)

            return output, {"rnn": [rnn_states[0], rnn_states[1]]}

    metadata = {
        "input_shape": kwargs.get('input_shape', None), 
        "hiddens": kwargs.get('hiddens', None), 
        "hidden_activation": kwargs.get('hidden_activation', None), 
        "output_shape": kwargs.get('output_shape', None), 
        "output_activation": kwargs.get('output_activation', None), 
        "output_scale": kwargs.get('output_scale', None), 
        "initial_log_std": kwargs.get('initial_log_std', None), 
    }

    return DeterministicModel(observation_space=observation_space,
                                         action_space=action_space,
                                         device=device,
                                         clip_actions=clip_actions, 
                                         rnn_param=rnn_param, 
                                         metadata=metadata)


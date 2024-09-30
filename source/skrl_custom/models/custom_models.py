from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import textwrap
import gym
import gymnasium

import torch
import torch.nn as nn  # noqa

# from skrl.models.torch import GaussianMixin  # noqa
# from skrl.models.torch import Model

from skrl.models.torch import Model  # noqa
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, MultivariateGaussianMixin  # noqa

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
            
            self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(self.num_actions))

            self.test_nn = build_sequential_network(self.num_observations, hiddens, self.num_actions, hidden_activation, output_activation)

        def compute(self, inputs, role=""):

            states = inputs["states"]

            output = self.test_nn(states)

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
            self.feature_extractor_size=128
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

            self.post_lstm = nn.Sequential(nn.Linear(self.hidden_size, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, self.num_actions))

            self.mlp_nn = nn.Sequential(nn.Linear(self.num_observations, 128, ),
                                                nn.Linear(128, self.num_actions))

        # def get_rnn_param(self): 
        #     return self.rnn_param
        
        def get_specification(self):
            return {"rnn": {"sequence_length": self.sequence_length,
                            "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
                                    (self.num_layers, self.num_envs, self.hidden_size)]}}

        def compute(self, inputs, role=""):

            # print("Roll? ")
            # # print(inputs["rnn"][0].shape)
            # # print(inputs["rnn"][1].shape)
            # print(inputs.keys())
            # print(inputs["rnn"][0].shape)

            # states = inputs["states"]

            # output = self.test_nn(states)

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

            # rnn_output, (self.hidden_states, self.cell_states) = self.lstm(features_states, (self.hidden_states, self.cell_states))
            rnn_output, rnn_states = self.lstm(features_states, (hidden_states, cell_states))
    
            output = self.post_lstm(rnn_output)
            output = torch.squeeze(output)

            # print("Check policy output size")
            # print(output.shape)
            # print(rnn_states[0].shape)

            return output, self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

            # print("Feature state shapee")
            # print(features_states.shape)
            # print(features_states.shape[-1])
            # print("Sequence length")
            # print(self.sequence_length)
            # print(features_states.view(-1, self.sequence_length, features_states.shape[-1]).shape)

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

            # return output, self.log_std_parameter, {{}}
            return output, self.log_std_parameter, {}

    # # compatibility with versions prior to 1.3.0
    # if not network and kwargs:
    #     network, output = convert_deprecated_parameters(kwargs)

    # # parse model definition
    # containers, output = generate_containers(network, output, embed_output=True, indent=1)

    # # network definitions
    # networks = []
    # forward: list[str] = []
    # for container in containers:
    #     networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
    #     forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    # # process output
    # if output["modules"]:
    #     networks.append(f'self.output_layer = {output["modules"][0]}')
    #     forward.append(f'output = self.output_layer({container["name"]})')
    # if output["output"]:
    #     forward.append(f'output = {output["output"]}')
    # else:
    #     forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

    # # build substitutions and indent content
    # networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    # forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

    # template = f"""class GaussianModel(GaussianMixin, Model):
    # def __init__(self, observation_space, action_space, device, clip_actions,
    #                 clip_log_std, min_log_std, max_log_std, reduction="sum"):
    #     Model.__init__(self, observation_space, action_space, device)
    #     GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

    #     {networks}
    #     self.log_std_parameter = nn.Parameter({initial_log_std} * torch.ones({output["size"]}))

    # def compute(self, inputs, role=""):
    #     {forward}
    #     return output, self.log_std_parameter, {{}}
    # """
    # # return source
    # if return_source:
    #     return template

    # instantiate model
    # _locals = {}
    # exec(template, globals(), _locals)
    # return _locals["GaussianModel"](observation_space=observation_space,
    #                                 action_space=action_space,
    #                                 device=device,
    #                                 clip_actions=clip_actions,
    #                                 clip_log_std=clip_log_std,
    #                                 min_log_std=min_log_std,
    #                                 max_log_std=max_log_std)

    metadata = {
        # "input_shape": input_shape,
        # "hiddens": hiddens,
        # "hidden_activation": hidden_activation,
        # "output_shape": output_shape,
        # "output_activation": output_activation,
        # "output_scale": output_scale,
        # "initial_log_std": initial_log_std,
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
    # compatibility with versions prior to 1.3.0
    if not network and kwargs:
        network, output = convert_deprecated_parameters(kwargs)

    # parse model definition
    containers, output = generate_containers(network, output, embed_output=True, indent=1)

    # network definitions
    networks = []
    forward: list[str] = []
    for container in containers:
        networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
        forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    # process output
    if output["modules"]:
        networks.append(f'self.output_layer = {output["modules"][0]}')
        forward.append(f'output = self.output_layer({container["name"]})')
    if output["output"]:
        forward.append(f'output = {output["output"]}')
    else:
        forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

    # build substitutions and indent content
    networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

    template = f"""class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        {networks}

    def compute(self, inputs, role=""):
        {forward}
        print("Output shapeeeeee")
        print(output.shape)
        return output, {{}}
    """
    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["DeterministicModel"](observation_space=observation_space,
                                         action_space=action_space,
                                         device=device,
                                         clip_actions=clip_actions)




def custom_deterministic_model2(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
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
    # # compatibility with versions prior to 1.3.0
    # if not network and kwargs:
    #     network, output = convert_deprecated_parameters(kwargs)

    # # parse model definition
    # containers, output = generate_containers(network, output, embed_output=True, indent=1)

    # # network definitions
    # networks = []
    # forward: list[str] = []
    # for container in containers:
    #     networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
    #     forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    # # process output
    # if output["modules"]:
    #     networks.append(f'self.output_layer = {output["modules"][0]}')
    #     forward.append(f'output = self.output_layer({container["name"]})')
    # if output["output"]:
    #     forward.append(f'output = {output["output"]}')
    # else:
    #     forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

    # # build substitutions and indent content
    # networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    # forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

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

            self.test_nn = build_sequential_network(self.num_observations, hiddens, output_shape, hidden_activation, output_activation)

        def compute(self, inputs, role=""):

            states = inputs["states"]

            output = self.test_nn(states)

            print("Outpus shapee??")
            print(output.shape)

            return output, {}

    # template = f"""class DeterministicModel(DeterministicMixin, Model):
    # def __init__(self, observation_space, action_space, device, clip_actions):
    #     Model.__init__(self, observation_space, action_space, device)
    #     DeterministicMixin.__init__(self, clip_actions)

    #     {networks}

    # def compute(self, inputs, role=""):
    #     {forward}
    #     return output, {{}}
    # """


    # # return source
    # if return_source:
    #     return template

    # # instantiate model
    # _locals = {}
    # exec(template, globals(), _locals)
    # return _locals["DeterministicModel"](observation_space=observation_space,
    #                                      action_space=action_space,
    #                                      device=device,
    #                                      clip_actions=clip_actions)

    metadata = {
        "input_shape": kwargs.get('input_shape', None), 
        "hiddens": kwargs.get('hiddens', None), 
        "hidden_activation": kwargs.get('hidden_activation', None), 
        "output_shape": kwargs.get('output_shape', None), 
        "output_activation": kwargs.get('output_activation', None), 
        "output_scale": kwargs.get('output_scale', None), 
    }

    return DeterministicModel(observation_space=observation_space,
                                         action_space=action_space,
                                         device=device,
                                         clip_actions=clip_actions, 
                                         metadata=metadata)




def custom_deterministic_model3(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
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
            
            # self.num_observations = observation_space.shape[0]
            # self.num_actions = action_space.shape[0]

            self.rnn_param = rnn_param
            
            self.num_observations = observation_space.shape[0]
            self.num_actions = action_space.shape[0]
            self.feature_extractor_size=128
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

            # self.feature_extractor_size=128
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

            self.post_lstm = nn.Sequential(nn.Linear(self.hidden_size, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, output_shape))

            self.mlp_nn = nn.Sequential(nn.Linear(self.num_observations, 128, ),
                                                nn.Linear(128, output_shape))

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

            # rnn_output, (self.hidden_states, self.cell_states) = self.lstm(features_states, (self.hidden_states, self.cell_states))
            rnn_output, rnn_states = self.lstm(features_states, (hidden_states, cell_states))

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





# def custom_gaussian_model4(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
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
#                         clip_log_std, min_log_std, max_log_std, rnn_param, metadata, reduction="sum"):
#             Model.__init__(self, observation_space, action_space, device)
#             GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#             print("observation_space")
#             print(observation_space.shape[0])

#             # print(rnn_param)

#             self.rnn_param = rnn_param
            
#             self.num_observations = observation_space.shape[0]
#             self.num_actions = action_space.shape[0]
#             self.feature_extractor_size=128
#             self.sequence_length = self.rnn_param["rnn_sequence_length"]
#             self.num_envs = self.rnn_param["rnn_num_envs"]
#             self.num_layers = self.rnn_param["rnn_num_layers"]
#             self.hidden_size = self.rnn_param["rnn_hidden_size"]
#             print(self.sequence_length)
#             print(self.num_envs)
#             print(self.num_layers)
#             print(self.hidden_size)

#             print("NUm pbervation")
#             print(self.num_observations)
#             print(self.feature_extractor_size)

#             self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(self.num_actions))

#             print("Inptu shape")
#             print(metadata)
#             hiddens = metadata["hiddens"]
#             hidden_activation = metadata["hidden_activation"]
#             output_activation = metadata["output_activation"]

#             self.test_nn = build_sequential_network(self.num_observations, hiddens, self.num_actions, hidden_activation, output_activation)

#             self.feature_extractor = nn.Sequential(nn.Linear(self.num_observations, self.feature_extractor_size),
#                                                 nn.Tanh())

#             self.lstm = nn.LSTM(input_size=self.feature_extractor_size,
#                                 hidden_size=self.hidden_size,
#                                 num_layers=self.num_layers,
#                                 batch_first=True)

#             self.post_lstm = nn.Sequential(nn.Linear(self.hidden_size, 128),
#                                     nn.Tanh(),
#                                     nn.Linear(128, self.num_actions))

#             self.mlp_nn = nn.Sequential(nn.Linear(self.num_observations, 128, ),
#                                                 nn.Linear(128, self.num_actions))

#         def get_rnn_param(self): 
#             return self.rnn_param
        
#         def get_specification(self):
#             return {"rnn": {"sequence_length": self.sequence_length,
#                             "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
#                                     (self.num_layers, self.num_envs, self.hidden_size)]}}

#         def compute(self, inputs, role=""):

#             # print("Roll? ")
#             # # print(inputs["rnn"][0].shape)
#             # # print(inputs["rnn"][1].shape)
#             # print(inputs.keys())
#             # print(inputs["rnn"][0].shape)

#             # states = inputs["states"]

#             # output = self.test_nn(states)

#             states = inputs["states"]
#             terminated = inputs.get("terminated", None)
#             hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

#             # print("State shape")
#             # print(states.shape)

#             features_states = self.feature_extractor(states)

#             # print("Feature state shapeee")
#             # print(features_states.shape)

#             batch_size = self.num_envs  # Number of parallel environments
#             features_states = features_states.view(-1, 1, self.feature_extractor_size)  # shape: (batch_size, 1, input_size)

#             # rnn_output, (self.hidden_states, self.cell_states) = self.lstm(features_states, (self.hidden_states, self.cell_states))
#             rnn_output, rnn_states = self.lstm(features_states, (hidden_states, cell_states))
    
#             output = self.post_lstm(rnn_output)
#             output = torch.squeeze(output)

#             # print("Check policy output size")
#             # print(output.shape)
#             # print(rnn_states[0].shape)

#             return output, self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

#             # print("Feature state shapee")
#             # print(features_states.shape)
#             # print(features_states.shape[-1])
#             # print("Sequence length")
#             # print(self.sequence_length)
#             # print(features_states.view(-1, self.sequence_length, features_states.shape[-1]).shape)

#             # if self.training:            
#             #     rnn_input = features_states.view(-1, self.sequence_length, features_states.shape[-1])
#             #     hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])
#             #     cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])
                
#             #     # hidden and cell states for initial sequence
#             #     hidden_states = hidden_states[:,:,0,:].contiguous()
#             #     cell_states = cell_states[:,:,0,:].contiguous()

#             #     # check if RNN state needs to be reset
#             #     if terminated is not None and torch.any(terminated):
#             #         rnn_outputs = []
#             #         terminated = terminated.view(-1, self.sequence_length)
#             #         indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

#             #         for i in range(len(indexes) - 1):
#             #             i0, i1 = indexes[i], indexes[i + 1]
#             #             rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
#             #             hidden_states[:, (terminated[:,i1-1]), :] = 0
#             #             cell_states[:, (terminated[:,i1-1]), :] = 0
#             #             rnn_outputs.append(rnn_output)

#             #         rnn_states = (hidden_states, cell_states)
#             #         rnn_output = torch.cat(rnn_outputs, dim=1)
#             #     else:
#             #         rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
#             # else:
#             #     rnn_input = features_states.view(-1, 1, features_states.shape[-1])
#             #     rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

#             # rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

#             # return self.post_lstm(rnn_output), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

#             # return output, self.log_std_parameter, {{}}
#             # return output, self.log_std_parameter, {}

#     # # compatibility with versions prior to 1.3.0
#     # if not network and kwargs:
#     #     network, output = convert_deprecated_parameters(kwargs)

#     # # parse model definition
#     # containers, output = generate_containers(network, output, embed_output=True, indent=1)

#     # # network definitions
#     # networks = []
#     # forward: list[str] = []
#     # for container in containers:
#     #     networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
#     #     forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
#     # # process output
#     # if output["modules"]:
#     #     networks.append(f'self.output_layer = {output["modules"][0]}')
#     #     forward.append(f'output = self.output_layer({container["name"]})')
#     # if output["output"]:
#     #     forward.append(f'output = {output["output"]}')
#     # else:
#     #     forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

#     # # build substitutions and indent content
#     # networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
#     # forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

#     # template = f"""class GaussianModel(GaussianMixin, Model):
#     # def __init__(self, observation_space, action_space, device, clip_actions,
#     #                 clip_log_std, min_log_std, max_log_std, reduction="sum"):
#     #     Model.__init__(self, observation_space, action_space, device)
#     #     GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#     #     {networks}
#     #     self.log_std_parameter = nn.Parameter({initial_log_std} * torch.ones({output["size"]}))

#     # def compute(self, inputs, role=""):
#     #     {forward}
#     #     return output, self.log_std_parameter, {{}}
#     # """
#     # # return source
#     # if return_source:
#     #     return template

#     # instantiate model
#     # _locals = {}
#     # exec(template, globals(), _locals)
#     # return _locals["GaussianModel"](observation_space=observation_space,
#     #                                 action_space=action_space,
#     #                                 device=device,
#     #                                 clip_actions=clip_actions,
#     #                                 clip_log_std=clip_log_std,
#     #                                 min_log_std=min_log_std,
#     #                                 max_log_std=max_log_std)

#     metadata = {
#         # "input_shape": input_shape,
#         # "hiddens": hiddens,
#         # "hidden_activation": hidden_activation,
#         # "output_shape": output_shape,
#         # "output_activation": output_activation,
#         # "output_scale": output_scale,
#         # "initial_log_std": initial_log_std,
#         "input_shape": kwargs.get('input_shape', None), 
#         "hiddens": kwargs.get('hiddens', None), 
#         "hidden_activation": kwargs.get('hidden_activation', None), 
#         "output_shape": kwargs.get('output_shape', None), 
#         "output_activation": kwargs.get('output_activation', None), 
#         "output_scale": kwargs.get('output_scale', None), 
#         "initial_log_std": kwargs.get('initial_log_std', None), 
#     }

#     return GaussianModel(observation_space=observation_space,
#                                     action_space=action_space,
#                                     device=device,
#                                     clip_actions=clip_actions,
#                                     clip_log_std=clip_log_std,
#                                     min_log_std=min_log_std,
#                                     max_log_std=max_log_std, 
#                                     rnn_param=rnn_param, 
#                                     metadata=metadata)


# def custom_gaussian_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
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

#     # compatibility with versions prior to 1.3.0
#     if not network and kwargs:
#         network, output = convert_deprecated_parameters(kwargs)

#     # parse model definition
#     containers, output = generate_containers(network, output, embed_output=True, indent=1)

#     # network definitions
#     networks = []
#     forward: list[str] = []
#     for container in containers:
#         networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
#         forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
#     # process output
#     if output["modules"]:
#         networks.append(f'self.output_layer = {output["modules"][0]}')
#         forward.append(f'output = self.output_layer({container["name"]})')
#     if output["output"]:
#         forward.append(f'output = {output["output"]}')
#     else:
#         forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

#     # build substitutions and indent content
#     networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
#     forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

#     # Access kwargs
#     # test_var = kwargs.get('test_var', None)
#     # print(test_var)

#     # RNN params
#     rnn_hidden_size = kwargs.get('rnn_hidden_size', None)
#     rnn_num_layers = kwargs.get('rnn_num_layers', None)
#     rnn_num_envs = kwargs.get('num_envs', None)
#     rnn_param = {
#         "rnn_hidden_size": rnn_hidden_size, 
#         "rnn_num_layers": rnn_num_layers, 
#         "rnn_num_envs": rnn_num_envs
#     }

#     template = f"""class GaussianModel(GaussianMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions,
#                     clip_log_std, min_log_std, max_log_std, rnn_param, sequence_length = 10, reduction="sum"):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#         print("observation_space")
#         print(observation_space.shape[0])

#         # print(rnn_param)

#         self.sequence_length = sequence_length
#         self.rnn_num_envs = rnn_param["rnn_num_envs"]
#         self.rnn_num_layers = rnn_param["rnn_num_layers"]
#         self.rnn_hidden_size = rnn_param["rnn_hidden_size"]
#         print(self.sequence_length)
#         print(self.rnn_num_envs)
#         print(self.rnn_num_layers)
#         print(self.rnn_hidden_size)

#         # # Assuming rnn_param is defined somewhere in your code
#         # hidden_state = torch.zeros(rnn_param["rnn_num_layers"], batch_size, rnn_param["rnn_hidden_size"]).to(device)  # (num_layers, batch_size, hidden_size)
#         # cell_state = torch.zeros(rnn_param["rnn_num_layers"], batch_size, rnn_param["rnn_hidden_size"]).to(device)  # LSTM also needs cell state
        
#         self.rnn = nn.LSTM(input_size={observation_space.shape[0]}, hidden_size=self.rnn_hidden_size, num_layers=self.rnn_num_layers, batch_first=True)

#         {networks}

#         self.log_std_parameter = nn.Parameter({initial_log_std} * torch.ones({output["size"]}))

#     def compute(self, inputs, role=""):

#         states = inputs["states"]
#         terminated = inputs.get("terminated", None)
#         hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

#         print("Statesss")
#         print(states.shape)
#         # print(terminated.shape)
#         print(hidden_states.shape)
#         # print(cell_states.shape)

#         # rnn_input = inputs["states"]  # Extract the state from the dictionary
#         # rnn_input = rnn_input.view(-1, 1, rnn_input.shape[1])

#         # rnn_output, _ = self.rnn(rnn_input)  # Pass through the RNN

#         # # Use the last output of RNN as input to the RL network
#         # rnn_output = rnn_output[:, -1, :]  # Select the last time step
#         # # print("RNN outputttt")
#         # # print(rnn_output.shape)

#         # # {forward.replace(container["input"], "rnn_output")}

#         {forward}
#         return output, self.log_std_parameter, {{}}

#     def get_specification(self):
#         return {{"rnn": {{"sequence_length": self.sequence_length,
#                         "sizes": [(self.rnn_num_layers, self.rnn_num_envs, self.rnn_hidden_size),
#                                   (self.rnn_num_layers, self.rnn_num_envs, self.rnn_hidden_size)]}}}}
    
#     # def print_hello(self, inputs, role=""):

#     #     print("hello")    
#     """
    
#     # return source
#     if return_source:
#         return template

#     # instantiate model
#     _locals = {}
#     exec(template, globals(), _locals)
#     return _locals["GaussianModel"](observation_space=observation_space,
#                                     action_space=action_space,
#                                     device=device,
#                                     clip_actions=clip_actions,
#                                     clip_log_std=clip_log_std,
#                                     min_log_std=min_log_std,
#                                     max_log_std=max_log_std, 
#                                     rnn_param=rnn_param)


# def custom_gaussian_model2(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
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
#     # compatibility with versions prior to 1.3.0
#     if not network and kwargs:
#         network, output = convert_deprecated_parameters(kwargs)

#     # parse model definition
#     containers, output = generate_containers(network, output, embed_output=True, indent=1)

#     # network definitions
#     networks = []
#     forward: list[str] = []
#     for container in containers:
#         networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
#         forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
#     # process output
#     if output["modules"]:
#         networks.append(f'self.output_layer = {output["modules"][0]}')
#         forward.append(f'output = self.output_layer({container["name"]})')
#     if output["output"]:
#         forward.append(f'output = {output["output"]}')
#     else:
#         forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

#     # build substitutions and indent content
#     networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
#     forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

#     template = f"""class GaussianModel(GaussianMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions,
#                     clip_log_std, min_log_std, max_log_std, reduction="sum"):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#         {networks}
#         self.log_std_parameter = nn.Parameter({initial_log_std} * torch.ones({output["size"]}))

#     # def get_specification(self):
#     #     print("hello")

#     #     return {{'rnn': {{'sizes': [(1, 4, 64), (1, 4, 64)]}}}}

#     def compute(self, inputs, role=""):
#         {forward}
#         return output, self.log_std_parameter, {{}}
#     """
#     # return source
#     if return_source:
#         return template

#     # instantiate model
#     _locals = {}
#     exec(template, globals(), _locals)
#     return _locals["GaussianModel"](observation_space=observation_space,
#                                     action_space=action_space,
#                                     device=device,
#                                     clip_actions=clip_actions,
#                                     clip_log_std=clip_log_std,
#                                     min_log_std=min_log_std,
#                                     max_log_std=max_log_std)


# def custom_deterministic_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                         action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                         device: Optional[Union[str, torch.device]] = None,
#                         clip_actions: bool = False,
#                         network: Sequence[Mapping[str, Any]] = [],
#                         output: Union[str, Sequence[str]] = "",
#                         return_source: bool = False,
#                         *args,
#                         **kwargs) -> Union[Model, str]:
#     """Instantiate a deterministic model

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
#     :param network: Network definition (default: [])
#     :type network: list of dict, optional
#     :param output: Output expression (default: "")
#     :type output: list or str, optional
#     :param return_source: Whether to return the source string containing the model class used to
#                           instantiate the model rather than the model instance (default: False).
#     :type return_source: bool, optional

#     :return: Deterministic model instance or definition source
#     :rtype: Model
#     """
#     # # compatibility with versions prior to 1.3.0
#     # if not network and kwargs:
#     #     network, output = convert_deprecated_parameters(kwargs)

#     # # parse model definition
#     # containers, output = generate_containers(network, output, embed_output=True, indent=1)

#     # # network definitions
#     # networks = []
#     # forward: list[str] = []
#     # for container in containers:
#     #     networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
#     #     forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
#     # # process output
#     # if output["modules"]:
#     #     networks.append(f'self.output_layer = {output["modules"][0]}')
#     #     forward.append(f'output = self.output_layer({container["name"]})')
#     # if output["output"]:
#     #     forward.append(f'output = {output["output"]}')
#     # else:
#     #     forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

#     # # build substitutions and indent content
#     # networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
#     # forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

#     # # RNN params
#     # rnn_hidden_size = kwargs.get('rnn_hidden_size', None)
#     # rnn_num_layers = kwargs.get('rnn_num_layers', None)
#     # rnn_param = {
#     #     "rnn_hidden_size": rnn_hidden_size, 
#     #     "rnn_num_layers": rnn_num_layers
#     # }

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

#     class DeterministicModel(DeterministicMixin, Model):
#         def __init__(self, observation_space, action_space, device, clip_actions, rnn_param, metadata):
#             Model.__init__(self, observation_space, action_space, device)
#             DeterministicMixin.__init__(self, clip_actions) 

#             self.rnn_param = rnn_param
            
#             # self.num_observations = observation_space.shape[0]
#             # self.num_actions = action_space.shape[0]

#             self.rnn_param = rnn_param
            
#             self.num_observations = observation_space.shape[0]
#             self.num_actions = action_space.shape[0]
#             self.feature_extractor_size=128
#             self.sequence_length = self.rnn_param["rnn_sequence_length"]
#             self.num_envs = self.rnn_param["rnn_num_envs"]
#             self.num_layers = self.rnn_param["rnn_num_layers"]
#             self.hidden_size = self.rnn_param["rnn_hidden_size"]
#             print(self.sequence_length)
#             print(self.num_envs)
#             print(self.num_layers)
#             print(self.hidden_size)

#             print("NUm pbervation")
#             print(self.num_observations)
#             print(self.feature_extractor_size)

#             # self.feature_extractor_size=128
#             # self.sequence_length = self.rnn_param["rnn_sequence_length"]
#             # self.num_envs = self.rnn_param["rnn_num_envs"]
#             # self.num_layers = self.rnn_param["rnn_num_layers"]
#             # self.hidden_size = self.rnn_param["rnn_hidden_size"]
#             # print(self.sequence_length)
#             # print(self.num_envs)
#             # print(self.num_layers)
#             # print(self.hidden_size)

#             # print("NUm pbervation")
#             # print(self.num_observations)
#             # print(self.feature_extractor_size)

#             # hiddens = metadata["hiddens"]
#             # hidden_activation = metadata["hidden_activation"]
#             # output_activation = metadata["output_activation"]
#             # output_shape = metadata["output_shape"].value

#             # self.test_nn = build_sequential_network(self.num_observations, hiddens, output_shape, hidden_activation, output_activation)

#             print("Inptu shape")
#             print(metadata)
#             hiddens = metadata["hiddens"]
#             hidden_activation = metadata["hidden_activation"]
#             output_activation = metadata["output_activation"]

#             self.test_nn = build_sequential_network(self.num_observations, hiddens, self.num_actions, hidden_activation, output_activation)

#             self.feature_extractor = nn.Sequential(nn.Linear(self.num_observations, self.feature_extractor_size),
#                                                 nn.Tanh())

#             self.lstm = nn.LSTM(input_size=self.feature_extractor_size,
#                                 hidden_size=self.hidden_size,
#                                 num_layers=self.num_layers,
#                                 batch_first=True)

#             self.post_lstm = nn.Sequential(nn.Linear(self.hidden_size, 128),
#                                     nn.Tanh(),
#                                     nn.Linear(128, self.num_actions))

#             self.mlp_nn = nn.Sequential(nn.Linear(self.num_observations, 128, ),
#                                                 nn.Linear(128, self.num_actions))

#         def get_specification(self):
#             return {"rnn": {"sequence_length": self.sequence_length,
#                             "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
#                                     (self.num_layers, self.num_envs, self.hidden_size)]}}
        
#         def compute(self, inputs, role=""):

#             # states = inputs["states"]

#             # output = self.test_nn(states)

#             # return output, {}

#             states = inputs["states"]
#             terminated = inputs.get("terminated", None)
#             hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

#             # print("State shape")
#             # print(states.shape)

#             features_states = self.feature_extractor(states)

#             # print("Feature state shapeee")
#             # print(features_states.shape)

#             batch_size = self.num_envs  # Number of parallel environments
#             features_states = features_states.view(-1, 1, self.feature_extractor_size)  # shape: (batch_size, 1, input_size)

#             # rnn_output, (self.hidden_states, self.cell_states) = self.lstm(features_states, (self.hidden_states, self.cell_states))
#             rnn_output, rnn_states = self.lstm(features_states, (hidden_states, cell_states))
    
#             output = self.post_lstm(rnn_output)
#             output = torch.squeeze(output)

#             # print("Check value output size")
#             # print(output.shape)
#             # print(rnn_states[0].shape)

#             return output, {"rnn": [rnn_states[0], rnn_states[1]]}

#     metadata = {
#         "input_shape": kwargs.get('input_shape', None), 
#         "hiddens": kwargs.get('hiddens', None), 
#         "hidden_activation": kwargs.get('hidden_activation', None), 
#         "output_shape": kwargs.get('output_shape', None), 
#         "output_activation": kwargs.get('output_activation', None), 
#         "output_scale": kwargs.get('output_scale', None), 
#         "initial_log_std": kwargs.get('initial_log_std', None), 
#     }

#     return DeterministicModel(observation_space=observation_space,
#                                          action_space=action_space,
#                                          device=device,
#                                          clip_actions=clip_actions, 
#                                          rnn_param=rnn_param, 
#                                          metadata=metadata)


# def custom_deterministic_model2(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                         action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                         device: Optional[Union[str, torch.device]] = None,
#                         clip_actions: bool = False,
#                         network: Sequence[Mapping[str, Any]] = [],
#                         output: Union[str, Sequence[str]] = "",
#                         return_source: bool = False,
#                         *args,
#                         **kwargs) -> Union[Model, str]:
#     """Instantiate a deterministic model

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
#     :param network: Network definition (default: [])
#     :type network: list of dict, optional
#     :param output: Output expression (default: "")
#     :type output: list or str, optional
#     :param return_source: Whether to return the source string containing the model class used to
#                           instantiate the model rather than the model instance (default: False).
#     :type return_source: bool, optional

#     :return: Deterministic model instance or definition source
#     :rtype: Model
#     """
#     # compatibility with versions prior to 1.3.0
#     if not network and kwargs:
#         network, output = convert_deprecated_parameters(kwargs)

#     # parse model definition
#     containers, output = generate_containers(network, output, embed_output=True, indent=1)

#     # network definitions
#     networks = []
#     forward: list[str] = []
#     for container in containers:
#         networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
#         forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
#     # process output
#     if output["modules"]:
#         networks.append(f'self.output_layer = {output["modules"][0]}')
#         forward.append(f'output = self.output_layer({container["name"]})')
#     if output["output"]:
#         forward.append(f'output = {output["output"]}')
#     else:
#         forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

#     # build substitutions and indent content
#     networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
#     forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

#     template = f"""class DeterministicModel(DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions):
#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions)

#         {networks}

#     def compute(self, inputs, role=""):
#         {forward}
#         return output, {{}}
#     """
#     # return source
#     if return_source:
#         return template

#     # instantiate model
#     _locals = {}
#     exec(template, globals(), _locals)
#     return _locals["DeterministicModel"](observation_space=observation_space,
#                                          action_space=action_space,
#                                          device=device,
#                                          clip_actions=clip_actions)

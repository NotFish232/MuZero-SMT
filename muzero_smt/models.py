import math

import torch as T
from torch import nn
from typing_extensions import Any, Self, Type

from games.abstract_game import MuZeroConfig


def dict_to_cpu(dictionary: dict[str, Any]) -> dict[str, Any]:
    cpu_dict: dict[str, Any] = {}
    for key, value in dictionary.items():
        if isinstance(value, T.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class MuZeroNetwork(nn.Module):
    @staticmethod
    def from_config(config: MuZeroConfig) -> "MuZeroNetwork":
        return MuZeroNetwork(
            config.observation_shape,
            config.stacked_observations,
            len(config.action_space),
            config.encoding_size,
            config.fc_reward_layers,
            config.fc_value_layers,
            config.fc_policy_layers,
            config.fc_representation_layers,
            config.fc_dynamics_layers,
            config.support_size,
        )

    def __init__(
        self: Self,
        observation_shape: tuple[int, ...],
        stacked_observations,
        action_space_size: int,
        encoded_state_size: int,
        fc_reward_layers: list[int],
        fc_value_layers: list[int],
        fc_policy_layers: list[int],
        fc_representation_layers: list[int],
        fc_dynamics_layers: list[int],
        support_size: int,
    ) -> None:
        super().__init__()

        self.action_space_size = action_space_size
        self.encoded_state_size = encoded_state_size

        # Size of entire support with a support for values in range -[support_size, support_size]
        self.full_support_size = 2 * support_size + 1

        # Representation network
        # Input is a stack of `stacked_observations` number previous observations
        # + the current observation
        # + a `stacked_observations` number of frames where all elements are the action taken
        self.representation_network = mlp(
            (stacked_observations + 1) * math.prod(observation_shape)
            + stacked_observations * math.prod(observation_shape[1:]),
            fc_representation_layers,
            self.encoded_state_size,
        )

        # Dynamics state transition network
        # Input is the encoded space + an action
        self.dynamics_state_network = mlp(
            self.encoded_state_size + self.action_space_size,
            fc_dynamics_layers,
            self.encoded_state_size,
        )

        # Dynamics reward network
        # Input is the encoded space
        # Output is a support of a scalar representing the reward
        self.dynamics_reward_network = mlp(
            self.encoded_state_size, fc_reward_layers, self.full_support_size
        )

        # Prediction policy network
        # Input is the encoded space
        # Output is logits over the action space
        self.prediction_policy_network = mlp(
            self.encoded_state_size, fc_policy_layers, self.action_space_size
        )

        # Prediction value network
        # Input is the encoded space
        # Output is a support of a scalar representing the value
        self.prediction_value_network = mlp(
            self.encoded_state_size, fc_value_layers, self.full_support_size
        )

    def prediction(self: Self, encoded_state: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        """
        Predicts the policy logits and value of the encode_state through the prediction network

        Args:
            encoded_state (T.Tensor): The hidden state representation of the current state

        Returns:
            tuple[T.Tensor, T.Tensor]: A tuple of the policy logits and the value tensor
        """

        # encoded_state: (batch size, encoded state size)

        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)

        # policy_logits: (batch size, action space size)
        # value: (batch size, full supports size)

        return policy_logits, value

    def representation(self: Self, observation: T.Tensor) -> T.Tensor:
        """
        Creates the representation of the observation using the representation network

        Args:
            observation (T.Tensor): The observation associated with the current state

        Returns:
            T.Tensor: The encoded state representation of the current observation
        """

        # observation: (batch size, *observation shape)
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )

        # Scale encoded state between [0, 1] (See appendix paper Training)

        # Find min and max values along non-batch dimension
        min_encoded_state = T.min(encoded_state, dim=1, keepdim=True)[0]
        max_encoded_state = T.max(encoded_state, dim=1, keepdim=True)[0]

        # For numerical stability add a small epsilon
        encoded_state_normalized = (encoded_state - min_encoded_state) / (
            max_encoded_state - min_encoded_state + 1e-8
        )

        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            T.zeros((action.shape[0], self.action_space_size)).to(action.device).float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = T.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = T.log(
            (
                T.zeros(1, self.full_support_size)
                .scatter(1, T.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


def mlp(
    input_size: int,
    layer_sizes: list[int],
    output_size: int,
    activation: Type[nn.Module] = nn.ELU,
    output_activation: Type[nn.Module] = nn.Identity,
) -> nn.Module:
    """
    Constructs a MLP outlined by the passed in sizes

    Args:
        input_size (int): The size of the input layer
        layer_sizes (list[int]): The size of the hidden layers
        output_size (int): The size of the output layer
        activation (nn.Module, optional): The activation function for hidden layers. Defaults to nn.ELU().
        output_activation (nn.Module, optional): The action function for output layer. Defaults to nn.Identity().

    Returns:
        nn.Module: The MLP comprised of the specified parameters
    """

    sizes = [input_size] + layer_sizes + [output_size]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]

    return nn.Sequential(*layers)


def support_to_scalar(
    logits: T.Tensor, support_size: int, eps: float = 1e-3
) -> T.Tensor:
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture

    Args:
        logits (torch.Tensor - shape (x_1, ..., x_n, 2 * support_size + 1): The tensor with last dimension of supports
        support_size (int): The number of supports
        eps (float): A small epsilon used for inverting the scaling of x

    Returns:
        torch.Tensor - shape (x_1, ..., x_n): Tensor of scaler values recovered from supports
    """

    # Softmax the logits so it sums up to one
    probs = T.softmax(logits, dim=-1)

    # Generate supports and sum up the element wise multiplication of support values and probs
    support = T.arange(
        -support_size, support_size + 1, dtype=T.float32, device=probs.device
    )
    x = T.sum(support * probs, dim=-1)

    # Invert the scaling (defined in https://arxivTrue.org/abs/1805.11593)
    x = (
        T.sign(x) * ((T.sqrt(1 + 4 * eps * (T.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2
        - 1
    )

    return x


def scalar_to_support(x: T.Tensor, support_size: int, eps: float = 1e-3) -> T.Tensor:
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture

    Args:
        x (torch.Tensor - shape (x_1, ..., x_n)): The tensor of scalars to conver to supports
        support_size (int): The number of supports
        eps (float): An epsilon for the scaling of x

    Returns:
        torch.Tensor - shape (x_1, ..., x_n, 2 * support_size + 1): A tensor encoded in supports
    """

    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = T.sign(x) * (T.sqrt(T.abs(x) + 1) - 1) + eps * x

    # Clamp x to the range [-support_size, support_size] and convert it to an integer
    x = T.clamp(x, -support_size, support_size)

    # Get value below corresponding to left-side support
    x_floor = x.floor()

    # Logits of size (W, H, 2 * support_size + 1) that contains the actual supports
    logits = T.zeros((*x.shape, 2 * support_size + 1), device=x.device)

    # Get values and idxs for first supports and second supports
    # Value of first support is how close it is to floor, and value of second is how close it is to the ceiling
    # Index is the floor offset by the support size to deal with negative values
    support_1_vals = (x_floor - x + 1).unsqueeze_(-1)
    support_1_idxs = (x_floor + support_size).to(T.int32).unsqueeze_(-1)

    support_2_vals = 1 - support_1_vals
    support_2_idxs = support_1_idxs + 1
    # If support 2 index overflows past 2 * support_size, wrap it around to index 0
    support_2_idxs.masked_fill_(2 * support_size < support_2_idxs, 0)

    # Scatter the support values into the corresponding idxs
    logits.scatter_(-1, support_1_idxs, support_1_vals)
    logits.scatter_(-1, support_2_idxs, support_2_vals)

    return logits

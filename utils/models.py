import torch as T
from torch import nn
from typing_extensions import Any, Self
import torch

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


##################################
######## Fully Connected #########


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
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            T.zeros((action.shape[0], self.action_space_size)).to(action.device).float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = T.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

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
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ELU,
):
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
    """
    # Decode to a scalar
    probabilities = T.softmax(logits, dim=1)
    support = (
        T.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = T.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = T.sign(x) * (
        ((T.sqrt(1 + 4 * eps * (T.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )
    return x


def scalar_to_support(x: T.Tensor, support_size: int, eps: float = 1e-3) -> T.Tensor:
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture

    Args:
        x (torch.Tensor - shape (W, H)): The tensor of scalars to conver to supports
        support_size (int): The number of supports
        eps (float): An epsilon for the scaling of x

    Returns:
        torch.Tensor - shape (W, H, 2 * support_size + 1): A tensor encoded in supports
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

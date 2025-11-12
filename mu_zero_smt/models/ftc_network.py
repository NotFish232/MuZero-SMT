import math

import torch as T
from torch.nn import functional as F
from typing_extensions import Self, override

from mu_zero_smt.utils.config import MuZeroConfig

from .mu_zero_network import MuZeroNetwork
from .utils import mlp


class FTCNetwork(MuZeroNetwork):
    @override
    @staticmethod
    def from_config(config: MuZeroConfig) -> MuZeroNetwork:
        return FTCNetwork(
            config.observation_shape,
            config.stacked_observations,
            config.discrete_action_space,
            config.continuous_action_space,
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
        stacked_observations: int,
        discrete_action_size: int,
        continuous_action_size: int,
        encoded_state_size: int,
        fc_reward_layers: list[int],
        fc_value_layers: list[int],
        fc_policy_layers: list[int],
        fc_representation_layers: list[int],
        fc_dynamics_layers: list[int],
        support_size: int,
    ) -> None:
        super().__init__()

        self.discrete_action_size = discrete_action_size
        self.continuous_action_size = continuous_action_size

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
            self.encoded_state_size
            + self.discrete_action_size
            + self.continuous_action_size,
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
            self.encoded_state_size,
            fc_policy_layers,
            self.discrete_action_size + 2 * continuous_action_size,
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
            encoded_state (torch.Tensor): The hidden state representation of the current state

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of the policy logits and the value tensor
        """

        # encoded_state: (batch size, encoded state size)

        policy_logits = self.prediction_policy_network(encoded_state)

        # make continuous standard deviations positive
        mask = T.zeros_like(policy_logits, dtype=T.bool)
        mask[:, self.discrete_action_size + 1 :: 2] = True

        policy_logits = T.where(mask, F.softplus(policy_logits), policy_logits)

        value = self.prediction_value_network(encoded_state)

        # policy_logits: (batch size, action space size)
        # value: (batch size, full supports size)

        return policy_logits, value

    def representation(self: Self, observation: T.Tensor) -> T.Tensor:
        """
        Creates the representation of the observation using the representation network

        Args:
            observation (torch.Tensor): The observation associated with the current state

        Returns:
            T.Tensor: The encoded state representation of the current observation
        """

        # observation: (batch size, *observation shape)

        # encoded_state: (batch size, encoded state size)
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

    def dynamics(
        self: Self, encoded_state: T.Tensor, action: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor]:
        """
        Using the dynamics network, predicts the next hidden state and reward associated with the ucrrent state

        Args:
            encoded_state (torch.Tensor): The current hidden state
            action (torch.Tensor): The action being taken in the current state

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The next hidden state and the current reward
        """

        # encoded_state: (batch size, encoded state size)
        # action: (batch size, action space size)

        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        x = T.cat((encoded_state, action), dim=1)

        # Using the dynamics networks get both the next state and reward
        next_encoded_state = self.dynamics_state_network(x)
        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)

        # Find min and max values along non-batch dimensions
        min_next_encoded_state = T.min(next_encoded_state, dim=1, keepdim=True)[0]
        max_next_encoded_state = T.max(next_encoded_state, dim=1, keepdim=True)[0]

        # for numerical stability add a small epsilon
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / (max_next_encoded_state - min_next_encoded_state + 1e-8)

        return next_encoded_state_normalized, reward

    @override
    def initial_inference(
        self: Self, observation: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        """
        Runs the intial inference based on the starting observation

        Args:
            observation (torch.Tensor): The initial observation

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The value, reward, policy, and state
        """

        # observation: (batch size, *observation shape)

        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)

        # reward equal to 0 for consistency, log it to turn it into logits which will be softmaxed later
        reward = T.zeros(
            (observation.shape[0], self.full_support_size),
            dtype=T.float32,
            device=observation.device,
        )
        reward[:, self.full_support_size // 2] = 1
        reward = T.log(reward)

        # encoded_state: (batch size, encoded state size)
        # policy_logits: (batch size, action space size)
        # value: (batch size, full support size)
        # reward: (batch size, full support size)

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    @override
    def recurrent_inference(
        self: Self, encoded_state: T.Tensor, action: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        """
        Runs the recurrent inference based on the previous hidden state and an action

        Args:
            encoded_state (torch.Tensor): The current hidden state
            action (torch.Tensor): The action taken at that state

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The value, reward, policy, and state
        """

        # encoded_state: (batch size, encoded state size)
        # action: (batch size, action size)

        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)

        # next_encoded_state: (batch size, encoded state size)
        # reward: (batch size, full support size)
        # policy_logits (batch size, action space size)
        # value: (batch size, full support size)

        return value, reward, policy_logits, next_encoded_state

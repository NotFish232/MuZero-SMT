import torch as T
from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as TGnn  # type: ignore
from torch_geometric.data import Batch  # type: ignore
from typing_extensions import Self, override

from mu_zero_smt.utils.config import MuZeroConfig

from .mu_zero_network import MuZeroNetwork
from .utils import mlp


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv1 = TGnn.GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            train_eps=True,
        )

        self.conv2 = TGnn.GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            train_eps=True,
        )

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self: Self, x, edge_index: T.Tensor, batch: T.Tensor) -> T.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = TGnn.global_mean_pool(x, batch)
        x = self.lin(x)

        return x


class GraphNetwork(MuZeroNetwork):
    @staticmethod
    def from_config(config: MuZeroConfig) -> "GraphNetwork":
        return GraphNetwork(
            observation_size=config.observation_size,
            discrete_action_size=config.discrete_action_space,
            continuous_action_size=config.continuous_action_space,
            support_size=config.support_size,
            **config.network_args
        )

    def __init__(
        self: Self,
        observation_size: int,
        discrete_action_size: int,
        continuous_action_size: int,
        encoded_state_size: int,
        fc_reward_layers: list[int],
        fc_value_layers: list[int],
        fc_policy_layers: list[int],
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
        self.representation_network = GraphEncoder(
            observation_size, observation_size // 4, self.encoded_state_size
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
            self.encoded_state_size,
        )
        self.prediction_policy_discrete_network = mlp(
            self.encoded_state_size,
            fc_policy_layers,
            self.discrete_action_size,
        )
        self.prediction_policy_continuous_network = mlp(
            self.encoded_state_size,
            fc_policy_layers,
            self.continuous_action_size,
        )

        # Prediction value network
        # Input is the encoded space
        # Output is a support of a scalar representing the value
        self.prediction_value_network = mlp(
            self.encoded_state_size, fc_value_layers, self.full_support_size
        )

        self.prediction_policy_network.children

    def prediction(
        self: Self, encoded_state: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        """
        Predicts the policy logits and value of the encode_state through the prediction network

        Args:
            encoded_state (T.Tensor): The hidden state representation of the current state

        Returns:
            tuple[T.Tensor, T.Tensor, T.Tensor]: A tuple of the discrete logits, continuous logits, and the value tensor
        """

        # encoded_state: (batch size, encoded state size)

        policy_logits = self.prediction_policy_network(encoded_state)

        policy_discrete_logits = self.prediction_policy_discrete_network(policy_logits)
        policy_continuous_logits = self.prediction_policy_continuous_network(
            policy_logits
        )

        value = self.prediction_value_network(encoded_state)

        # policy_logits: (batch size, action space size)
        # value: (batch size, full supports size)

        return policy_discrete_logits, policy_continuous_logits, value

    def representation(self: Self, observation: Batch) -> T.Tensor:
        """
        Creates the representation of the observation using the representation network

        Args:
            observation (Data): The observation associated with the current state

        Returns:
            T.Tensor: The encoded state representation of the current observation
        """

        # observation: (batch size, *observation shape)

        # encoded_state: (batch size, encoded state size)
        encoded_state = self.representation_network(
            observation.x, observation.edge_index, observation.batch
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
        self: Self,
        observation: Batch,
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        """
        Runs the intial inference based on the starting observation

        Args:
            observation (Batch): The initial observation

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The value, reward, policy, and state
        """

        # observation: (batch size, *observation shape)

        encoded_state = self.representation(observation)
        discrete_logits, continuous_logits, value = self.prediction(encoded_state)

        # reward equal to 0 for consistency, log it to turn it into logits which will be softmaxed later
        reward = T.zeros(
            (encoded_state.shape[0], self.full_support_size),
            dtype=T.float32,
            device=encoded_state.device,
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
            discrete_logits,
            continuous_logits,
            encoded_state,
        )

    @override
    def recurrent_inference(
        self: Self, encoded_state: T.Tensor, action: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
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
        discrete_logits, continuous_logits, value = self.prediction(next_encoded_state)

        # next_encoded_state: (batch size, encoded state size)
        # reward: (batch size, full support size)
        # policy_logits (batch size, action space size)
        # value: (batch size, full support size)

        return value, reward, discrete_logits, continuous_logits, next_encoded_state

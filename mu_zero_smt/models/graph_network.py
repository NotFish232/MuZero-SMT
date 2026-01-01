import torch as T
from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as TGnn  # type: ignore
from torch_geometric.data import Batch  # type: ignore
from typing_extensions import Self, override

from mu_zero_smt.utils import MuZeroConfig, mlp

from .mu_zero_network import MuZeroNetwork


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
            action_space=config.action_space,
            support_size=config.support_size,
            **config.model_config
        )

    def __init__(
        self: Self,
        observation_size: int,
        action_space: list[int],
        encoded_state_size: int,
        fc_reward_layers: list[int],
        fc_value_layers: list[int],
        fc_policy_layers: list[int],
        fc_dynamics_layers: list[int],
        support_size: int,
    ) -> None:
        super().__init__()

        self.action_space = action_space

        self.encoded_state_size = encoded_state_size

        # Size of entire support with a support for values in range -[support_size, support_size]
        self.full_support_size = 2 * support_size + 1

        # Representation model
        self.representation_model = GraphEncoder(
            observation_size, observation_size // 4, self.encoded_state_size
        )

        # Dynamics Model
        self.dynamics_model = mlp(
            self.encoded_state_size,
            fc_dynamics_layers,
            self.encoded_state_size,
        )
        self.dynamics_heads = nn.ModuleList(
            [
                mlp(self.encoded_state_size + dim, [], self.encoded_state_size)
                for dim in self.action_space
            ]
        )

        # Policy model
        self.policy_model = mlp(
            self.encoded_state_size,
            fc_policy_layers,
            self.encoded_state_size,
        )
        self.policy_discrete_head = mlp(
            self.encoded_state_size,
            fc_policy_layers,
            len(self.action_space),
        )
        self.policy_continuous_heads = nn.ModuleList(
            [mlp(self.encoded_state_size, [], 2 * dim) for dim in self.action_space]
        )

        # Value model
        self.value_model = mlp(
            self.encoded_state_size, fc_value_layers, self.full_support_size
        )

        # Reward model
        self.reward_model = mlp(
            self.encoded_state_size, fc_reward_layers, self.full_support_size
        )

    def prediction(
        self: Self, encoded_state: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        """
        Predicts the discrete policy logits, a continuous parameter distribution,
         and value of the encode_state through the prediction network

        Args:
            encoded_state (T.Tensor): The hidden state representation of the current state

        Returns:
            tuple[T.Tensor, T.Tensor, T.Tensor]: A tuple of the discrete logits, continuous logits, and the value tensor
        """

        # encoded_state: (batch_size, encoded_state_size)

        policy_latent_state = self.policy_model(encoded_state)

        # policy_latent_state: (batch_size, encoded_state_size)

        policy_discrete_logits = self.policy_discrete_head(policy_latent_state)

        policy_continuous_logits = T.concat(
            [
                h(policy_latent_state).reshape(-1, dim, 2)
                for h, dim in zip(self.policy_continuous_heads, self.action_space)
            ],
            dim=1,
        )

        # Ensure variances are strictly positive
        policy_continuous_means = policy_continuous_logits[:, :, 0]
        policy_continuous_stds = F.softplus(policy_continuous_logits[:, :, 1])

        policy_continuous_logits = T.stack(
            (policy_continuous_means, policy_continuous_stds), dim=-1
        )

        # policy_discrete_logits: (batch_size, len(action_space))
        # policy_continuous_logits: (batch_size, sum(action_space), 2)

        value = self.value_model(encoded_state)

        # value: (batch_size, full_support_size)

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
        encoded_state = self.representation_model(
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
        self: Self, encoded_state: T.Tensor, action: T.Tensor, parameters: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor]:
        """
        Using the dynamics network, predicts the next hidden state and reward associated with the next state

        Args:
            encoded_state (T.Tensor): The current hidden state
            action (T.Tensor): The action being taken in the current state
            parameters (T.Tensor): The parameters associated with the provided action

        Returns:
            tuple[T.Tensor, T.Tensor]: The next hidden state and the next reward
        """

        # encoded_state: (batch size, encoded state size)
        # action: (batch size, 1)

        # Using the dynamics networks get both the next state and reward

        dynamics_latent_state = self.dynamics_model(encoded_state)

        next_encoded_state = T.zeros_like(encoded_state)

        cur_param_idx = 0

        # For each possible action, mask out state action pairs that use this action and apply the corresponding head
        for cur_action, (head, dim) in enumerate(
            zip(self.dynamics_heads, self.action_space)
        ):
            action_mask = (action == cur_action).squeeze(1)

            if action_mask.any():
                batch_states = dynamics_latent_state[action_mask]
                batch_params = parameters[
                    action_mask, cur_param_idx : cur_param_idx + dim
                ]

                batch = T.concat((batch_states, batch_params), dim=-1)

                next_encoded_state[action_mask] = head(batch)

            cur_param_idx += dim

        reward = self.reward_model(next_encoded_state)

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
            observation (T.Tensor): The initial observation

        Returns:
            tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]: The value, reward, discrete policy, continuous policy, and initial state
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
        self: Self,
        encoded_state: T.Tensor,
        action: T.Tensor,
        parameters: T.Tensor,
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        """
        Runs the recurrent inference based on the previous hidden state and an action

        Args:
            encoded_state (T.Tensor): The current hidden state
            action (T.Tensor): The action taken at that state
            parameters (T.Tensor): The parameters associated with the action

        Returns:
            tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]: The value, reward, discrete policy, continuous policy, and new state
        """

        # encoded_state: (batch size, encoded state size)
        # action: (batch size, action size)

        next_encoded_state, reward = self.dynamics(encoded_state, action, parameters)
        discrete_logits, continuous_logits, value = self.prediction(next_encoded_state)

        # next_encoded_state: (batch size, encoded state size)
        # reward: (batch size, full support size)
        # policy_logits (batch size, action space size)
        # value: (batch size, full support size)

        return value, reward, discrete_logits, continuous_logits, next_encoded_state

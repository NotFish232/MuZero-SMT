from abc import ABC, abstractmethod

import torch as T
from torch import nn
from torch_geometric.data import Batch  # type: ignore
from typing_extensions import Self

from mu_zero_smt.utils.config import MuZeroConfig
from mu_zero_smt.utils.utils import CollatedObservation

"""
Base Class for a MuZero Network
"""


class MuZeroNetwork(ABC, nn.Module):
    @staticmethod
    def from_config(config: MuZeroConfig) -> "MuZeroNetwork":
        """
        Constructs the network based on the config
        """

        if config.network_type == "ftc":
            from .ftc_network import FTCNetwork

            return FTCNetwork.from_config(config)
        if config.network_type == "graph":
            from .graph_network import GraphNetwork

            return GraphNetwork.from_config(config)

        raise ValueError(f'Invalid network type: "{config.network_type}"')

    @abstractmethod
    def initial_inference(
        self: Self, observation: CollatedObservation
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        """
        Runs the intial inference based on the starting observation

        Args:
            observation (torch.Tensor): The initial observation

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The value, reward, policy, and state
        """

        ...

    @abstractmethod
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

        ...

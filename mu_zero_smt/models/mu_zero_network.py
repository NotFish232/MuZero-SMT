import torch as T
from torch import nn
from typing_extensions import Self

from mu_zero_smt.utils.config import MuZeroConfig

from abc import ABC, abstractmethod


"""
Base Class for a MuZero Network
"""


class MuZeroNetwork(ABC, nn.Module):
    @staticmethod
    @abstractmethod
    def from_config(config: MuZeroConfig) -> "MuZeroNetwork": ...

    @abstractmethod
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

        ...

    @abstractmethod
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

        ...

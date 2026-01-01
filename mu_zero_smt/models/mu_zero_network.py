from abc import ABC, abstractmethod

import torch as T
from torch import nn
from typing_extensions import Self

from mu_zero_smt.utils import CollatedObservation, MuZeroConfig


class MuZeroNetwork(ABC, nn.Module):
    """
    Base Class for a MuZero Network
    """

    @staticmethod
    def from_config(config: MuZeroConfig) -> "MuZeroNetwork":
        """
        Constructs the network based on the config
        """
        if config.model_type == "graph":
            from .graph_network import GraphNetwork

            return GraphNetwork.from_config(config)

        raise ValueError(f'Invalid network type: "{config.model_type}"')

    @abstractmethod
    def initial_inference(
        self: Self, observation: CollatedObservation
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        """
        Runs the intial inference based on the starting observation

        Args:
            observation (T.Tensor): The initial observation

        Returns:
            tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]: The value, reward, discrete policy, continuous policy, and initial state
        """

        ...

    @abstractmethod
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
            parameters (T.Tensor): The parameter associated with the actions

        Returns:
            tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]: The value, reward, discrete policy, continuous policy, and new state
        """

        ...

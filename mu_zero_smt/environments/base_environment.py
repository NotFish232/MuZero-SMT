from abc import ABC, abstractmethod

import numpy as np
from torch_geometric.data import Data  # type: ignore
from typing_extensions import Any, Self

from mu_zero_smt.utils.utils import RawObservation, RunMode


class BaseEnvironment(ABC):
    """
    Inherit this class for muzero to play
    """

    @abstractmethod
    def __init__(
        self: Self,
        mode: RunMode,
        seed: int | None = None,
    ) -> None:
        pass

    @abstractmethod
    def step(
        self: Self, action: int, params: np.ndarray
    ) -> tuple[RawObservation, float, bool]:
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.
            params: continuous params in the range [0, 1]

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

    @abstractmethod
    def reset(self: Self, episode_id: int | None = None) -> RawObservation:
        """
        Reset the game for a new game. If id is passed it should be the id of the current game

        Returns:
            Initial observation of the game.
        """

    @abstractmethod
    def get_action_mask(self: Self, action: int) -> np.ndarray:
        """
        A mask of which continuous parameters each action uses for loss calculations
        """

    @abstractmethod
    def unique_episodes(self: Self) -> list[int]:
        """
        A list of ids representing unique episodes
        """

    @abstractmethod
    def episode_stats(self: Self) -> dict[str, Any]:
        """
        A dictionary representing arbitray stats of the previously run episode in the environment
        """

    def close(self: Self) -> None:
        """
        Properly close the game.
        """

from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Any, Self

from mu_zero_smt.utils.config import MuZeroConfig
from mu_zero_smt.utils.utils import Mode


class AbstractEnvironment(ABC):
    """
    Inherit this class for muzero to play
    """

    @abstractmethod
    def __init__(
        self: Self,
        mode: Mode,
        seed: int | None = None,
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_config() -> MuZeroConfig:
        """
        Get the MuZeroConfig for this game. Used for training
        """

    @abstractmethod
    def step(
        self: Self, action: int, params: np.ndarray
    ) -> tuple[np.ndarray, float, bool]:
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.
            params: continuous params in the range [0, 1]

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

    @abstractmethod
    def reset(self: Self, id: int | None = None) -> np.ndarray:
        """
        Reset the game for a new game. If id is passed it should be the id of the current game

        Returns:
            Initial observation of the game.
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

    def cleanup(self: Self) -> None:
        """
        Cleans up the internal state of the environment so it can be serialized
        """

    def close(self: Self) -> None:
        """
        Properly close the game.
        """

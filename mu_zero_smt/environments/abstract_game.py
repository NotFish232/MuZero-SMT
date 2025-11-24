from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Any, Self

from mu_zero_smt.utils.config import MuZeroConfig


class AbstractGame(ABC):
    """
    Inherit this class for muzero to play
    """

    @abstractmethod
    def __init__(self: Self, seed: int | None = None) -> None:
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
    def reset(self: Self) -> np.ndarray:
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """

    @abstractmethod
    def task_stats(self: Self) -> dict[str, Any]:
        """
        A dcitionary representing arbitray stats of the previously run task
        """

    def close(self: Self) -> None:
        """
        Properly close the game.
        """

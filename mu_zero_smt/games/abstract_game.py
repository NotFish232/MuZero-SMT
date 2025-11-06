from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Self

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
    def step(self: Self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

    @abstractmethod
    def legal_actions(self: Self) -> list[int]:
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """

    @abstractmethod
    def reset(self: Self) -> np.ndarray:
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """

    def close(self: Self) -> None:
        """
        Properly close the game.
        """

    @abstractmethod
    def render(self: Self) -> None:
        """
        Display the game observation.
        """

    def action_to_string(self: Self, action: int) -> str:
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """

        return str(action)

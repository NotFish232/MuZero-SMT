from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from typing_extensions import Callable, Self


@dataclass
class MuZeroConfig:
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

    seed: int  # Seed for numpy, torch and the game
    max_num_gpus: (
        int | None
    )  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

    ### Game
    observation_shape: tuple[
        int, int, int
    ]  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    action_space: list[
        int
    ]  # Fixed list of all possible actions. You should only edit the length
    players: list[int]  # List of players. You should only edit the length
    stacked_observations: int  # Number of previous observations and previous actions to add to the current observation

    # Evaluate
    muzero_player: int  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    opponent: (
        str | None
    )  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

    ### Self-Play
    num_workers: int  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    selfplay_on_gpu: bool
    max_moves: int  # Maximum number of moves if game is not finished before
    num_simulations: int  # Number of future moves self-simulated
    discount: float  # Chronological discount of the reward
    temperature_threshold: (
        int | None
    )  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

    # Root prior exploration noise
    root_dirichlet_alpha: float
    root_exploration_fraction: float

    # UCB formula
    pb_c_base: float
    pb_c_init: float

    ### Network
    support_size: int  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
    # Fully Connected Network
    encoding_size: int
    fc_representation_layers: list[
        int
    ]  # Define the hidden layers in the representation network
    fc_dynamics_layers: list[int]  # Define the hidden layers in the dynamics network
    fc_reward_layers: list[int]  # Define the hidden layers in the reward network
    fc_value_layers: list[int]  # Define the hidden layers in the value network
    fc_policy_layers: list[int]  # Define the hidden layers in the policy network

    ### Training
    results_path: Path  # Path to store the model weights and TensorBoard logs
    save_model: bool  # Save the checkpoint in results_path as model.checkpoint

    training_steps: (
        int  # Total number of training steps (ie weights update according to a batch)
    )
    batch_size: int  # Number of parts of games to train on at each training step

    checkpoint_interval: (
        int  # Number of training steps before using the model for self-playing
    )

    value_loss_weight: float  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
    train_on_gpu: bool  # Train on GPU if available

    optimizer: str  # "Adam" or "SGD". Paper uses SGD
    weight_decay: float  # L2 weights regularization
    momentum: float  # Used only if optimizer is SGD

    # Exponential learning rate schedule
    lr_init: float  # Initial learning rate
    lr_decay_rate: float  # Set it to 1 to use a constant learning rate
    lr_decay_steps: int

    ### Replay Buffer
    replay_buffer_size: int  # Number of self-play games to keep in the replay buffer

    num_unroll_steps: int  # Number of game moves to keep for every batch element

    td_steps: int  # Number of steps in the future to take into account for calculating the target value
    PER: bool  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    PER_alpha: float  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    # Reanalyze (See paper appendix Reanalyse)
    use_last_model_value: bool  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
    reanalyse_on_gpu: bool

    ### Adjust the self play / training ratio to avoid over/underfitting
    self_play_delay: float  # Number of seconds to wait after each played game
    training_delay: float  # Number of seconds to wait after each training step
    ratio: (
        float | None
    )  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
    """
    visit_softmax_temperature_fn: Callable[[Self, int], float]


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

        pass

    @abstractmethod
    def step(self: Self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        pass

    def to_play(self: Self) -> int:
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0

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
        pass

    @abstractmethod
    def reset(self: Self) -> np.ndarray:
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        pass

    def close(self: Self) -> None:
        """
        Properly close the game.
        """
        pass

    @abstractmethod
    def render(self: Self) -> None:
        """
        Display the game observation.
        """
        pass

    def action_to_string(self: Self, action: int) -> str:
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """

        return str(action)

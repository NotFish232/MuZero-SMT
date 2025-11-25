from dataclasses import dataclass
from pathlib import Path

from typing_extensions import TYPE_CHECKING, Callable, Self, Type

if TYPE_CHECKING:
    from mu_zero_smt.models import MuZeroNetwork

"""
A data class holding the config for MuZero
"""


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
    discrete_action_space: (
        int  # Fixed list of all possible actions. You should only edit the length
    )
    continuous_action_space: int  # Additional actions to consider
    stacked_observations: int  # Number of previous observations and previous actions to add to the current observation

    ### Self-Play
    num_self_play_workers: int  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    num_validate_workers: (
        int  # Number of simultaneous threads / workers for validating the current model
    )
    max_moves: int  # Maximum number of moves if game is not finished before
    num_simulations: int  # Number of future moves self-simulated
    num_continuous_samples: int  # Number of samples of continuous parameters to take
    discount: float  # Chronological discount of the reward
    # Root prior exploration noise
    root_dirichlet_alpha: float
    root_exploration_fraction: float

    # UCB formula
    pb_c_base: float
    pb_c_init: float

    ### Network
    network: Type["MuZeroNetwork"]  # Which network is used
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
    training_steps: (
        int  # Total number of training steps (ie weights update according to a batch)
    )
    batch_size: int  # Number of parts of games to train on at each training step

    checkpoint_interval: (
        int  # Number of training steps before using the model for self-playing
    )

    value_loss_weight: float  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
    train_on_gpu: bool  # Train on GPU if available

    weight_decay: float  # L2 weights regularization

    # Exponential learning rate schedule
    lr_init: float  # Initial learning rate
    lr_decay_rate: float  # Set it to 1 to use a constant learning rate
    lr_decay_steps: int

    ### Replay Buffer
    replay_buffer_size: int  # Number of self-play games to keep in the replay buffer

    num_unroll_steps: int  # Number of game moves to keep for every batch element

    td_steps: int  # Number of steps in the future to take into account for calculating the target value
    priority_alpha: float  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    ### Adjust the self play / training ratio to avoid over/underfitting
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

import json
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any

"""
A data class holding the config for MuZero
"""


@dataclass
class MuZeroConfig:
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

    seed: int  # Seed for numpy, torch and the game

    env_config: dict[str, Any]

    ### Game
    observation_size: int  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    discrete_action_space: (
        int  # Fixed list of all possible actions. You should only edit the length
    )
    continuous_action_space: int  # Additional actions to consider

    ### Self-Play
    num_self_play_workers: int  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    num_eval_workers: (
        int  # Number of simultaneous threads / workers for validating the current model
    )
    num_test_workers: int
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
    experiment_name: str
    training_steps: (
        int  # Total number of training steps (ie weights update according to a batch)
    )
    batch_size: int  # Number of parts of games to train on at each training step

    checkpoint_interval: (
        int  # Number of training steps before using the model for self-playing
    )

    value_loss_weight: float  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)

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
    Linearly interpolates between temperature start and end over the course of training
    """
    temperature_start: float
    temperature_end: float


def load_config(experiment_name: str) -> MuZeroConfig:
    config_path = Path(__file__).parents[2] / "experiments" / f"{experiment_name}.json"

    config = MuZeroConfig(**json.load(open(config_path)))

    return config

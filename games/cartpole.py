import datetime
import pathlib

import gymnasium as gym  # type: ignore
import numpy
import torch
from typing_extensions import Self, override

from games.abstract_game import AbstractGame, MuZeroConfig


def visit_softmax_temperature_fn(self: MuZeroConfig, trained_steps: int) -> float:
    """
    Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
    The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

    Returns:
        Positive float.
    """
    if trained_steps < 0.5 * self.training_steps:
        return 1.0
    elif trained_steps < 0.75 * self.training_steps:
        return 0.5
    else:
        return 0.25


EPISODE_LENGTH = 1_000


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self: Self, seed: int | None = None):
        self.env = gym.make("CartPole-v1", max_episode_steps=EPISODE_LENGTH)
        print("SPEC", self.env.spec)
        if seed is not None:
            self.env.reset(seed=seed)

    @override
    @staticmethod
    def get_config() -> MuZeroConfig:
        return MuZeroConfig(
            # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
            seed=0,  # Seed for numpy, torch and the game
            max_num_gpus=None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
            ### Game
            ,
            observation_shape=(
                1,
                1,
                4,
            ),  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
            action_space=list(
                range(2)
            ),  # Fixed list of all possible actions. You should only edit the length
            stacked_observations=0  # Number of previous observations and previous actions to add to the current observation
            ### Self-Play
            ,
            num_workers=1,  # Number of simultaneous threads/workers self-playing to feed the replay buffer
            selfplay_on_gpu=False,
            max_moves=EPISODE_LENGTH,  # Maximum number of moves if game is not finished before
            num_simulations=50,  # Number of future moves self-simulated
            discount=0.997,  # Chronological discount of the reward
            # Root prior exploration noise
            root_dirichlet_alpha=0.25,
            root_exploration_fraction=0.25
            # UCB formula
            ,
            pb_c_base=19652,
            pb_c_init=1.25
            ### Network
            ,
            support_size=10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
            # Fully Connected Network
            ,
            encoding_size=8,
            fc_representation_layers=[],  # Define the hidden layers in the representation network
            fc_dynamics_layers=[16],  # Define the hidden layers in the dynamics network
            fc_reward_layers=[16],  # Define the hidden layers in the reward network
            fc_value_layers=[16],  # Define the hidden layers in the value network
            fc_policy_layers=[16]  # Define the hidden layers in the policy network
            ### Training
            ,
            results_path=pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime(
                "%Y-%m-%d--%H-%M-%S"
            ),  # Path to store the model weights and TensorBoard logs
            save_model=True,  # Save the checkpoint in results_path as model.checkpoint
            training_steps=10000,  # Total number of training steps (ie weights update according to a batch)
            batch_size=128,  # Number of parts of games to train on at each training step
            checkpoint_interval=10,  # Number of training steps before using the model for self-playing
            value_loss_weight=1,  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
            train_on_gpu=torch.cuda.is_available(),  # Train on GPU if available
            optimizer="Adam",  # "Adam" or "SGD". Paper uses SGD
            weight_decay=1e-4,  # L2 weights regularization
            momentum=0.9  # Used only if optimizer is SGD
            # Exponential learning rate schedule
            ,
            lr_init=0.02,  # Initial learning rate
            lr_decay_rate=0.8,  # Set it to 1 to use a constant learning rate
            lr_decay_steps=1000
            ### Replay Buffer
            ,
            replay_buffer_size=500,  # Number of self-play games to keep in the replay buffer
            num_unroll_steps=10,  # Number of game moves to keep for every batch element
            td_steps=50,  # Number of steps in the future to take into account for calculating the target value
            PER=True,  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
            PER_alpha=0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
            # Reanalyze (See paper appendix Reanalyse)
            ,
            use_last_model_value=True,  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            reanalyse_on_gpu=False
            ### Adjust the self play / training ratio to avoid over/underfitting
            ,
            self_play_delay=0,  # Number of seconds to wait after each played game
            training_delay=0,  # Number of seconds to wait after each training step
            ratio=1.5,  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
            visit_softmax_temperature_fn=visit_softmax_temperature_fn,
        )

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, terminated, truncated, _ = self.env.step(action)
        return numpy.array([[observation]]), reward, terminated or truncated

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return numpy.array([[self.env.reset()[0]]])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Push cart to the left",
            1: "Push cart to the right",
        }
        return f"{action_number}. {actions[action_number]}"

import datetime
import logging
import pathlib
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import z3  # type: ignore
from typing_extensions import Self, override

from mu_zero_smt.models import FTCNetwork

from .abstract_game import AbstractGame, MuZeroConfig

logging.basicConfig(
    filename="information.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


SOLVING_TIMEOUT = 60

MAX_NUM_TACTICS = 10


BENCHMARK_DIR = "data/non-incremental/QF_NIA/20170427-VeryMax/CInteger"


TACTICS = [
    "simplify",
    "smt",
    "bit-blast",
    "propagate-values",
    "ctx-simplify",
    "elim-uncnstr",
    "solve-eqs",
    "qfnia",
    "lia2card",
    "max-bv-sharing",
    "nla2bv",
    "qfnra-nlsat",
    "cofactor-term-ite",
]

PROBES = [
    "is-unbounded",
    "arith-max-deg",
    "arith-avg-deg",
    "arith-max-bw",
    "arith-avg-bw",
    "is-qfnia",
    "is-qfbv-eq",
    "memory",
    "size",
    "num-exprs",
    "num-consts",
    "num-bool-consts",
    "num-arith-consts",
    "num-bv-consts",
    "is-propositional",
    "is-qfbv",
]


def create_probe_embedding(
    goal: z3.Goal, probes: list[z3.Probe], time_used: float
) -> np.ndarray:
    values = np.zeros(len(probes) + 1, dtype=np.float64)

    for i, probe in enumerate(probes):
        probe_res = probe(goal)

        values[i] = probe_res

    values[len(probes)] = time_used

    return values


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


class Game(AbstractGame):
    """
    Game wrapper.
    """

    @override
    @staticmethod
    def get_config() -> MuZeroConfig:
        return MuZeroConfig(
            seed=0,  # Seed for numpy, torch and the game
            max_num_gpus=None,  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
            ### Game
            observation_shape=(
                1,
                1,
                len(PROBES) + 1,
            ),  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
            action_space=list(
                range(len(TACTICS))
            ),  # Fixed list of all possible actions. You should only edit the length
            stacked_observations=0,  # Number of previous observations and previous actions to add to the current observation
            ### Self-Play
            num_workers=5,  # Number of simultaneous threads/workers self-playing to feed the replay buffer
            selfplay_on_gpu=False,
            max_moves=MAX_NUM_TACTICS,  # Maximum number of moves if game is not finished before
            num_simulations=10,  # Number of future moves self-simulated
            discount=1,  # Chronological discount of the reward
            # Root prior exploration noise
            root_dirichlet_alpha=0.25,
            root_exploration_fraction=0.25,
            # UCB formula
            pb_c_base=19652,
            pb_c_init=1.25,
            ### Network
            conversion_fn=lambda gh, _: gh.observation_history[
                -1
            ],  # All processing happens in network
            network=FTCNetwork,
            support_size=10,  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
            # Fully Connected Network
            encoding_size=8,
            fc_representation_layers=[
                16
            ],  # Define the hidden layers in the representation network
            fc_dynamics_layers=[16],  # Define the hidden layers in the dynamics network
            fc_reward_layers=[16],  # Define the hidden layers in the reward network
            fc_value_layers=[16],  # Define the hidden layers in the value network
            fc_policy_layers=[16],  # Define the hidden layers in the policy network
            ### Training
            results_path=pathlib.Path(__file__).resolve().parents[2]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime(
                "%Y-%m-%d--%H-%M-%S"
            ),  # Path to store the model weights and TensorBoard logs
            save_model=True,  # Save the checkpoint in results_path as model.checkpoint
            training_steps=100_000,  # Total number of training steps (ie weights update according to a batch)
            batch_size=32,  # Number of parts of games to train on at each training step
            checkpoint_interval=10,  # Number of training steps before using the model for self-playing
            value_loss_weight=1,  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
            train_on_gpu=torch.cuda.is_available(),  # Train on GPU if available
            optimizer="Adam",  # "Adam" or "SGD". Paper uses SGD
            weight_decay=1e-4,  # L2 weights regularization
            momentum=0.9,  # Used only if optimizer is SGD
            # Exponential learning rate schedule
            lr_init=0.0064,  # Initial learning rate
            lr_decay_rate=1,  # Set it to 1 to use a constant learning rate
            lr_decay_steps=1000,
            ### Replay Buffer
            replay_buffer_size=5000,  # Number of self-play games to keep in the replay buffer
            num_unroll_steps=7,  # Number of game moves to keep for every batch element
            td_steps=7,  # Number of steps in the future to take into account for calculating the target value
            PER=True,  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
            PER_alpha=0.5,  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
            # Reanalyze (See paper appendix Reanalyse)
            use_last_model_value=True,  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            reanalyse_on_gpu=False,
            ### Adjust the self play / training ratio to avoid over/underfitting
            self_play_delay=0,  # Number of seconds to wait after each played game
            training_delay=0,  # Number of seconds to wait after each training step
            ratio=None,  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
            visit_softmax_temperature_fn=visit_softmax_temperature_fn,
        )

    def __init__(self: Self, seed: int | None = None, test_mode: bool = False) -> None:
        self.files = [*Path(BENCHMARK_DIR).rglob("*.smt2")]

        self.probes = [z3.Probe(p) for p in PROBES]
        self.tactics = [z3.Tactic(t) for t in TACTICS]

        self.current_goal: z3.Goal = z3.Goal()
        self.time_spent = 0.0
        self.num_tactics_applied = 0

        self.test_mode = test_mode
        self.selected_idx = -1

        random.seed(seed)

    def _get_observation(self: Self) -> np.ndarray:
        return create_probe_embedding(
            self.current_goal, self.probes, self.time_spent
        ).reshape(1, 1, -1)

    def step(self: Self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        current_file = self.files[self.selected_idx]
        tactic = self.tactics[action]

        reward = 0.0
        done = False

        self.num_tactics_applied += 1

        acc_tactic = z3.TryFor(tactic, int((SOLVING_TIMEOUT - self.time_spent) * 1_000))

        try:
            start = perf_counter()

            sub_goals = acc_tactic(self.current_goal)

            end = perf_counter()

            self.time_spent += end - start

            if len(sub_goals) != 1:
                raise Exception(
                    f"Expected 1 subgoal but found {len(sub_goals)} subgoals instead"
                )

            logging.info(
                f'{current_file.stem} | Ran tactic "{TACTICS[action]}" successfully'
            )

            self.current_goal = sub_goals[0]

            if len(self.current_goal) == 0 or self.current_goal.inconsistent():
                reward = 2 - self.time_spent / SOLVING_TIMEOUT
                done = True

                logging.info(
                    f"{current_file.stem} | Found SAT / UNSAT ({self.time_spent:.3f}s)"
                )
            elif self.num_tactics_applied >= MAX_NUM_TACTICS:
                reward = -1
                done = True

                logging.info(f"{current_file.stem} | TERM - Ran max number tactics")

        except z3.Z3Exception as e:
            msg = e.args[0].decode()

            if msg == "canceled":
                done = True
                reward = -1

                logging.info(f"{current_file.stem} | TERM - Timing out")
            elif self.num_tactics_applied >= MAX_NUM_TACTICS:
                reward = -1
                done = True

                logging.info(f"{current_file.stem} | TERM - Ran max number tactics")
            else:
                reward = -0.1
                logging.info(
                    f'{current_file.stem} | Error encountered running tactic "{TACTICS[action]}", {e}'
                )

        return self._get_observation(), reward, done

    def legal_actions(self: Self) -> list[int]:
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return [*range(len(self.tactics))]

    def reset(self: Self) -> np.ndarray:
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """

        # Run each benchmark sequentially in test mode
        if self.test_mode:
            self.selected_idx += 1
        else:
            self.selected_idx = random.randint(0, len(self.files) - 1)

        self.current_goal = z3.Goal()
        self.current_goal.add(z3.parse_smt2_file(str(self.files[self.selected_idx])))

        self.time_spent = 0.0
        self.num_tactics_applied = 0

        return self._get_observation()

    def render(self):
        """
        Display the game observation.
        """
        print(self.current_goal)
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return ""

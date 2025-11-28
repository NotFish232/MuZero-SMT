import os
import pathlib
import pickle
import time

import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Any, Self, Type

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

import ray

from mu_zero_smt.environments.abstract_environment import AbstractEnvironment
from mu_zero_smt.environments.smt import SMTEnvironment
from mu_zero_smt.models import dict_to_cpu
from mu_zero_smt.replay_buffer import Reanalyse, ReplayBuffer
from mu_zero_smt.self_play import GameHistory, SelfPlay
from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.trainer import Trainer


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(
        self: Self,
        Environment: Type[AbstractEnvironment],
    ) -> None:

        self.Environment = Environment
        self.config = self.Environment.get_config()

        # Preload the data if its being downloaded so it doesn't happen in each actor
        self.Environment(mode="train")

        # Fix random generator seed
        np.random.seed(self.config.seed)
        T.manual_seed(self.config.seed)

        ray.init(
            num_cpus=self.config.num_self_play_workers
            + self.config.num_eval_workers
            + 2
        )

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint: dict[str, Any] = {
            "weights": None,
            "optimizer_state": None,
            # Metrics
            "finished_eval_workers": [],
            "self_play_results": [],
            "eval_results": [],
            "training_step": 0,
            # Statistics
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "lr": 0,
            # Loss
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            # Termination condition
            "terminate": False,
        }

        self.replay_buffer: dict[int, GameHistory] = {}

        model = self.config.network.from_config(self.config)

        self.checkpoint["weights"] = dict_to_cpu(model.state_dict())

    def train(self: Self) -> None:
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        self.config.results_path.mkdir(parents=True, exist_ok=True)

        # Initialize workers
        self.shared_storage_worker = (
            ray.remote(SharedStorage)
            .options(name="shared_storage_worker", num_cpus=0)
            .remote(
                self.checkpoint,
                self.config,
            )
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = (
            ray.remote(ReplayBuffer)
            .options(name="replay_buffer_worker", num_cpus=0)
            .remote(self.checkpoint, self.replay_buffer, self.config)
        )

        self.reanalyse_worker = (
            ray.remote(Reanalyse)
            .options(name="reanalyse_worker", num_cpus=1)
            .remote(self.checkpoint, self.config)
        )

        self.training_worker = (
            ray.remote(Trainer)
            .options(name="trainer_worker", num_cpus=1)
            .remote(self.checkpoint, self.config)
        )

        self.self_play_workers = [
            ray.remote(SelfPlay)
            .options(name=f"self_play_worker_{i + 1}", num_cpus=1)
            .remote(
                self.checkpoint,
                self.Environment,
                "train",
                self.config,
                self.config.seed + i,
                i,
            )
            for i in range(self.config.num_self_play_workers)
        ]

        self.eval_workers = [
            ray.remote(SelfPlay)
            .options(name=f"eval_worker_{i + 1}", num_cpus=1)
            .remote(
                self.checkpoint,
                self.Environment,
                "eval",
                self.config,
                self.config.seed + self.config.num_self_play_workers + i,
                i,
            )
            for i in range(self.config.num_eval_workers)
        ]

        # Launch workers
        self.reanalyse_worker.reanalyse.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )

        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )

        for self_play_worker in self.self_play_workers:
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker,
                self.replay_buffer_worker,
            )

        for eval_worker in self.eval_workers:
            eval_worker.continuous_self_play.remote(self.shared_storage_worker, None)

        self.logging_loop()

    def logging_loop(self: Self) -> None:
        """
        Keep track of the training performance.
        """

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        # Loop for updating the training performance
        counter = 0
        keys = [
            "training_step",
            # Metrics
            "finished_eval_workers",
            "self_play_results",
            "eval_results",
            # Stats
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
            "lr",
            # Loss Metrics
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
        ]
        info = ray.get(self.shared_storage_worker.get_info_batch.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info_batch.remote(keys))

                writer.add_scalar(
                    "1.Metrics/1.Self_Play_Percent_Solved",
                    (
                        (
                            sum(x["successful"] for x in info["self_play_results"])
                            / len(info["self_play_results"])
                        )
                        if len(info["self_play_results"]) > 0
                        else 0
                    ),
                    counter,
                )

                if len(info["finished_eval_workers"]) == self.config.num_eval_workers:
                    writer.add_scalar(
                        "1.Metrics/2.Eval_Percent_Solved",
                        (
                            (
                                sum(x["successful"] for x in info["eval_results"])
                                / len(info["eval_results"])
                            )
                        ),
                        counter,
                    )
                    self.shared_storage_worker.set_info_batch.remote(
                        {"eval_results": [], "finished_eval_workers": []}
                    )

                writer.add_scalar(
                    "2.Workers/1.Self_played_games",
                    info["num_played_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Eval_played_games",
                    len(info["eval_results"]),
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/3.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Self_played_steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/5.Reanalysed_games",
                    info["num_reanalysed_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/6.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/7.Learning_rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
                print(
                    f'Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                counter += 1

                time.sleep(1)

        except KeyboardInterrupt:
            pass

        self.terminate_workers()

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = T.load(checkpoint_path, weights_only=False)
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0


def main() -> None:
    muzero = MuZero(SMTEnvironment)
    muzero.train()


if __name__ == "__main__":
    main()

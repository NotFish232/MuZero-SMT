import copy
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # type: ignore
from typing_extensions import Any, Self, Type

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

import ray
from ray.actor import ActorProxy

from mu_zero_smt.environments.base_environment import BaseEnvironment
from mu_zero_smt.environments.smt import SMTEnvironment
from mu_zero_smt.models.graph_network import MuZeroNetwork
from mu_zero_smt.replay_buffer import ReplayBuffer
from mu_zero_smt.self_play import GameHistory, SelfPlay
from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.trainer import Trainer
from mu_zero_smt.utils import MuZeroConfig, dict_to_cpu, load_config


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        Environment (Type[BaseEnvironment]): The Environment used for training
        config (MuZeroConfig, optional): The configuration for training
    """

    def __init__(
        self: Self,
        Environment: Type[BaseEnvironment],
        config: MuZeroConfig,
    ) -> None:

        self.Environment = Environment
        self.config = config

        # Preload the data if its being downloaded so it doesn't happen in each actor
        self.Environment(mode="train", **self.config.env_config)

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
            "train_weights": None,
            "optimizer_state": None,
            "eval_weights": None,
            "best_weights": None,
            "best_weights_percent": 0,
            # Metrics
            "finished_eval_workers": [],
            "train_results": [],
            "eval_results": [],
            "eval_results_history": [],
            "training_step": 0,
            # Statistics
            "num_played_games": 0,
            "num_played_steps": 0,
            "lr": 0,
            # Loss
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "param_loss": 0,
            # Termination condition
            "terminate": False,
        }

        self.replay_buffer: dict[int, GameHistory] = {}

        model = MuZeroNetwork.from_config(self.config)

        self.checkpoint["train_weights"] = dict_to_cpu(model.state_dict())
        self.checkpoint["eval_weights"] = copy.deepcopy(
            dict_to_cpu(self.checkpoint["train_weights"])
        )

        self.shared_storage_worker: ActorProxy[SharedStorage] | None = None
        self.replay_buffer_worker: ActorProxy[ReplayBuffer] | None = None
        self.training_worker: ActorProxy[Trainer] | None = None
        self.self_play_workers: list[ActorProxy[SelfPlay]] | None = None
        self.eval_workers: list[ActorProxy[SelfPlay]] | None = None

    def train(self: Self) -> None:
        """
        Spawn ray workers and launch the training.
        """

        self.results_path = (
            Path(__file__).parent
            / "results"
            / self.config.experiment_name
            / datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )

        Path(self.results_path).mkdir(exist_ok=True, parents=True)

        # Initialize workers
        self.shared_storage_worker = (
            ray.remote(SharedStorage)
            .options(name="shared_storage_worker", num_cpus=0)
            .remote(self.checkpoint)
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = (
            ray.remote(ReplayBuffer)
            .options(name="replay_buffer_worker", num_cpus=0)
            .remote(self.checkpoint, self.replay_buffer, self.config)
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
                False,
                self.config,
                i,
                self.config.num_self_play_workers,
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
                True,
                self.config,
                i,
                self.config.num_eval_workers,
            )
            for i in range(self.config.num_eval_workers)
        ]

        # Launch workers
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

        # All workers must be setup by this point
        assert self.shared_storage_worker is not None

        # Write everything in TensorBoard
        writer = SummaryWriter(self.results_path)

        # Loop for updating the training performance
        p_bar = tqdm(total=self.config.training_steps)

        counter = 0
        keys = [
            "training_step",
            "train_weights",
            "eval_weights",
            "best_weights_percent",
            # Metrics
            "finished_eval_workers",
            "train_results",
            "eval_results",
            "eval_results_history",
            # Stats
            "num_played_games",
            "num_played_steps",
            "lr",
            # Loss Metrics
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "param_loss",
        ]
        info = ray.get(self.shared_storage_worker.get_info_batch.remote(keys))

        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info_batch.remote(keys))

                writer.add_scalar(
                    "1.Metrics/1.Self_Play_Percent_Solved",
                    (
                        np.mean([x["successful"] for x in info["train_results"]])
                        if len(info["train_results"]) > 0
                        else 0
                    ),
                    counter,
                )

                if len(info["finished_eval_workers"]) == self.config.num_eval_workers:
                    percent_solved = np.mean(
                        [x["successful"] for x in info["eval_results"]]
                    )

                    if percent_solved > info["best_weights_percent"]:
                        self.shared_storage_worker.set_info_batch.remote(
                            {
                                "best_weights": info["eval_weights"],
                                "best_weights_percent": percent_solved,
                            }
                        )

                    writer.add_scalar(
                        "1.Metrics/2.Eval_Percent_Solved", percent_solved, counter
                    )

                    self.shared_storage_worker.update_info.remote(
                        "eval_results_history", info["eval_results"]
                    )
                    self.shared_storage_worker.set_info.remote(
                        "eval_weights", info["train_weights"]
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
                    sum(len(e) for e in info["eval_results_history"])
                    + len(info["eval_results"]),
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/3.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Self_played_steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/2.Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/3.Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/4.Policy_loss", info["policy_loss"], counter)
                writer.add_scalar("3.Loss/5.Param_loss", info["param_loss"], counter)

                if info["training_step"] % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()

                counter += 1

                p_bar.n = info["training_step"]
                p_bar.update()

                time.sleep(1)

        except KeyboardInterrupt:
            pass

        self.terminate_workers()

    def terminate_workers(self: Self) -> None:
        """
        Softly terminate the running tasks and garbage collect the workers.
        """

        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )

        self.self_play_workers = None
        self.eval_workers = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def save_checkpoint(self: Self) -> None:
        assert (
            self.shared_storage_worker is not None
            and self.replay_buffer_worker is not None
        )

        # Save model weights
        checkpoint_keys = [
            "train_weights",
            "optimizer_state",
            "eval_weights",
            "best_weights",
            "best_weights_percent",
            "train_results",
            "eval_results_history",
            "training_step",
            "num_played_games",
            "num_played_steps",
        ]
        checkpoint = ray.get(
            self.shared_storage_worker.get_info_batch.remote(checkpoint_keys)
        )
        T.save(checkpoint, self.results_path / "model.checkpoint")

        # Save replay buffer
        # with open(self.results_path / "replay_buffer.pkl", "wb") as f:
        #     pickle.dump(ray.get(self.replay_buffer_worker.get_buffer.remote()), f)

    def load_checkpoint(self: Self, checkpoint_path: Path) -> None:
        """
        Load a model and  saved replay buffer.

        Args:
            checkpoint_path (Path): Path to directory containing model.checkpoint and replay_buffer.pkl
        """

        # Load checkpoint
        self.checkpoint.update(
            T.load(checkpoint_path / "model.checkpoint", weights_only=False)
        )

        # Load replay buffer
        with open(checkpoint_path / "replay_buffer.pkl", "rb") as f:
            self.replay_buffer = pickle.load(f)


def main() -> None:

    config = load_config()

    muzero = MuZero(SMTEnvironment, config)
    muzero.train()


if __name__ == "__main__":
    main()

import copy
import time

import numpy as np
import ray
import torch as T
from ray.actor import ActorProxy
from typing_extensions import TYPE_CHECKING, Any, Self

from mu_zero_smt.models import support_to_scalar
from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.utils.config import MuZeroConfig

if TYPE_CHECKING:
    from mu_zero_smt.self_play import GameHistory


class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batches.
    """

    def __init__(
        self: Self,
        initial_checkpoint: dict[str, Any],
        initial_buffer: dict[int, "GameHistory"],
        config: MuZeroConfig,
    ) -> None:
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)

        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]

        self.total_samples = sum(
            len(game_history.root_values) for game_history in self.buffer.values()
        )

        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        np.random.seed(self.config.seed)

    @ray.method
    def save_game(
        self: Self,
        game_history: "GameHistory",
        shared_storage: ActorProxy[SharedStorage],
    ) -> None:
        """
        Saves a game in GameHistory to the ReplayBuffer

        Args:
            game_history (GameHistory): The history of a single game
            shared_storage (ActorHandle[SharedStorage] | None, optional): SharedStorage to save stats to. Defaults to None.
        """

        if len(game_history.priorities) > 0:
            # Avoid read only array when loading replay buffer from disk
            game_history.priorities = np.copy(game_history.priorities)
        else:
            # Initial priorities for the prioritized replay (See paper appendix Training)
            priorities = []
            for i, root_value in enumerate(game_history.root_values):
                priority = (
                    np.abs(root_value - self.compute_target_value(game_history, i))
                    ** self.config.priority_alpha
                )
                priorities.append(priority)

            game_history.priorities = np.array(priorities, dtype=np.float32)
            game_history.game_priority = np.max(game_history.priorities)

        # Add game to buffer and update stats
        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        # If too many games in buffer remove the earliest one
        if len(self.buffer) > self.config.replay_buffer_size:
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        shared_storage.set_info.remote("num_played_games", self.num_played_games)
        shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self: Self) -> dict[int, "GameHistory"]:
        return self.buffer

    @ray.method
    def get_batch(
        self: Self,
    ) -> tuple[
        list[list[int]],
        tuple[
            list[np.ndarray],
            list[list[int]],
            list[list[np.ndarray]],
            list[list[float]],
            list[list[float]],
            list[list[list[float]]],
            np.ndarray,
            list[list[int]],
        ],
    ]:
        (
            index_batch,
            observation_batch,
            action_batch,
            param_batch,
            reward_batch,
            value_batch,
            policy_batch,
            l_weight_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [], [], [])

        for game_id, game_history, game_prob in self.sample_n_games(self.config.batch_size, False):  # type: ignore
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions, params = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    self.config.discrete_action_space,
                )
            )
            action_batch.append(actions)
            param_batch.append(params)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )

            l_weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        weight_batch = np.array(l_weight_batch, dtype=np.float32) / max(l_weight_batch)

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                param_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    @ray.method
    def sample_n_games(
        self: Self, n_games: int, uniform: bool
    ) -> list[tuple[int, "GameHistory", float]]:

        game_ids = list(self.buffer.keys())
        game_probs = None
        game_prob_dict = {}

        # Sample based on game priorities
        if not uniform:
            game_probs = np.array(
                [self.buffer[id].game_priority for id in game_ids], dtype=np.float32
            )
            game_probs /= np.sum(game_probs)

            # Update prob dict with the calculated probabilities
            for game_id, prob in zip(game_ids, game_probs):
                game_prob_dict[game_id] = prob

        selected_games = np.random.choice(game_ids, n_games, p=game_probs)

        # return the game_id, GameHistory, and the probability (-1 if uniform)
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id, -1))
            for game_id in selected_games
        ]

        return ret

    def sample_position(
        self: Self, game_history: "GameHistory", uniform: bool = False
    ) -> tuple[int, float]:
        """
        Sample position from game either uniformly or according to some priority from the game history.
        See paper appendix Training.

        Returns:
            tuple[int, float]: The index of the game history sampled along with the probability
        """

        position_probs = None

        if not uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)

        position_index = np.random.choice(
            len(game_history.root_values), p=position_probs
        )

        return position_index, (
            position_probs[position_index] if position_probs is not None else -1
        )

    @ray.method
    def update_game_history(
        self: Self, game_id: int, game_history: "GameHistory"
    ) -> None:
        # The element could have been removed since its selection and update
        if next(iter(self.buffer.keys())) <= game_id:
            # Avoid read only array when loading replay buffer from disk
            game_history.priorities = np.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    @ray.method
    def update_priorities(
        self: Self, priorities: np.ndarray, index_info: list[list[int]]
    ) -> None:
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """

        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer.keys())) <= game_id:
                priority = priorities[i, :]

                # Update priorities starting from the game position
                start_idx = game_pos
                end_idx = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )

                # Update position priorities of game_id in range of start_index to end_index
                self.buffer[game_id].priorities[start_idx:end_idx] = priority[
                    : end_idx - start_idx
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = np.max(
                    self.buffer[game_id].priorities
                )

    def compute_target_value(
        self: Self, game_history: "GameHistory", idx: int
    ) -> float:
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.

        # Find the actual reward at the end or 0 if outside of range of game
        bootstrap_index = idx + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = root_values[bootstrap_index]

            # Discount it by how far in the future
            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[idx + 1 : bootstrap_index + 1]
        ):
            # Accumulate the total reward discounted by the dicsunt factor
            value += reward * self.config.discount**i

        return value

    def make_target(
        self: Self, game_history: "GameHistory", state_index: int
    ) -> tuple[
        list[float], list[float], list[list[float]], list[int], list[np.ndarray]
    ]:
        """
        Generate targets for every unroll steps.
        """

        target_values, target_rewards, target_policies, actions, params = (
            [],
            [],
            [],
            [],
            [],
        )
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
                params.append(game_history.param_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
                params.append(game_history.param_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(
                    np.random.choice(range(self.config.discrete_action_space))
                )
                params.append(np.zeros(self.config.continuous_action_space))

        return target_values, target_rewards, target_policies, actions, params


class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config: MuZeroConfig) -> None:
        self.config = config

        # Fix random generator seed
        np.random.seed(self.config.seed)
        T.manual_seed(self.config.seed)

        # Initialize the network
        self.model = self.config.network.from_config(config)
        self.model.load_state_dict(initial_checkpoint["weights"])
        self.model.to(T.device("cpu"))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    @ray.method
    def reanalyse(
        self,
        replay_buffer: ActorProxy[ReplayBuffer],
        shared_storage: ActorProxy[SharedStorage],
    ) -> None:
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.load_state_dict(
                ray.get(shared_storage.get_info.remote("weights"))
            )

            game_id, game_history, _ = ray.get(
                replay_buffer.sample_n_games.remote(1, True)
            )[0]

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            l_observations = np.array(
                [
                    game_history.get_stacked_observations(
                        i,
                        self.config.stacked_observations,
                        self.config.discrete_action_space,
                    )
                    for i in range(len(game_history.root_values))
                ]
            )

            observations = T.tensor(
                l_observations,
                dtype=T.float32,
                device=next(self.model.parameters()).device,
            )

            values = support_to_scalar(
                self.model.initial_inference(observations)[0],
                self.config.support_size,
            )
            game_history.reanalysed_predicted_root_values = (
                T.squeeze(values).detach().cpu().numpy()
            )

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )

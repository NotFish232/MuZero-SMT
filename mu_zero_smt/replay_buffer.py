import copy

import numpy as np
import ray
from ray.actor import ActorProxy
from typing_extensions import TYPE_CHECKING, Any, Self

from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.utils import MuZeroConfig

if TYPE_CHECKING:
    from mu_zero_smt.self_play import GameHistory


class ReplayBufferEntry:
    """
    Class representing an entry into the replay buffer
    """

    def __init__(
        self: Self, game_history: "GameHistory", priorities: np.ndarray | None = None
    ) -> None:
        self.game_history = game_history

        self.update_priorities(
            priorities
            if priorities is not None
            else np.zeros_like(game_history.root_values)
        )

        # For PER
        self.reanalysed_predicted_root_values = None

    def update_priorities(self: Self, new_priorities: np.ndarray) -> None:
        self.priorities = new_priorities
        self.game_priority = np.max(self.priorities)


class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batches.
    """

    def __init__(
        self: Self,
        initial_checkpoint: dict[str, Any],
        initial_buffer: dict[int, ReplayBufferEntry],
        config: MuZeroConfig,
    ) -> None:
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)

        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]

        self.total_samples = sum(
            len(buffer_entry.game_history.root_values)
            for buffer_entry in self.buffer.values()
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

        # Initial priorities for the prioritized replay (See paper appendix Training)
        entry = ReplayBufferEntry(game_history)

        priorities = []
        for i, root_value in enumerate(game_history.root_values):
            priority = (
                np.abs(root_value - self.compute_target_value(entry, i))
                ** self.config.priority_alpha
            )
            priorities.append(priority)

        entry.update_priorities(np.array(priorities, dtype=np.float32))

        # Add game to buffer and update stats
        self.buffer[self.num_played_games] = entry
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        # If too many games in buffer remove the earliest one
        if len(self.buffer) > self.config.replay_buffer_size:
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].game_history.root_values)
            del self.buffer[del_id]

        shared_storage.set_info.remote("num_played_games", self.num_played_games)
        shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    @ray.method
    def get_buffer(self: Self) -> dict[int, ReplayBufferEntry]:
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

        games = self.sample_n_games(self.config.batch_size, uniform=False)

        for buffer_id, entry, game_prob in games:
            game_pos, pos_prob = self.sample_position(entry)

            values, rewards, policies, actions, params = self.make_target(
                entry, game_pos
            )
            index_batch.append([buffer_id, game_pos])
            observation_batch.append(
                entry.game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
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
                        len(entry.game_history.action_history) - game_pos,
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

    def sample_n_games(
        self: Self, n_games: int, uniform: bool
    ) -> list[tuple[int, ReplayBufferEntry, float]]:

        buffer_ids = list(self.buffer.keys())
        game_probs = None
        game_prob_dict = {}

        # Sample based on game priorities
        if not uniform:
            game_probs = np.array(
                [self.buffer[buffer_id].game_priority for buffer_id in buffer_ids],
                dtype=np.float32,
            )
            game_probs /= np.sum(game_probs)

            # Update prob dict with the calculated probabilities
            for buffer_id, prob in zip(buffer_ids, game_probs):
                game_prob_dict[buffer_id] = prob

        selected_games = np.random.choice(buffer_ids, n_games, p=game_probs)

        # return the game_id, GameHistory, and the probability (-1 if uniform)
        ret = [
            (
                buffer_id,
                self.buffer[buffer_id],
                game_prob_dict.get(buffer_id, -1),
            )
            for buffer_id in selected_games
        ]

        return ret

    def sample_position(
        self: Self, entry: ReplayBufferEntry, uniform: bool = False
    ) -> tuple[int, float]:
        """
        Sample position from game either uniformly or according to some priority from the game history.
        See paper appendix Training.

        Returns:
            tuple[int, float]: The index of the game history sampled along with the probability
        """

        position_probs = None

        if not uniform:
            position_probs = entry.priorities / sum(entry.priorities)

        position_index = np.random.choice(
            len(entry.game_history.root_values), p=position_probs
        )

        return position_index, (
            position_probs[position_index] if position_probs is not None else -1
        )

    @ray.method
    def update_buffer_entry(
        self: Self, buffer_id: int, entry: ReplayBufferEntry
    ) -> None:
        # The element could have been removed since its selection and update
        if next(iter(self.buffer.keys())) <= buffer_id:
            self.buffer[buffer_id] = entry

    @ray.method
    def update_priorities(
        self: Self, priorities: np.ndarray, index_info: list[list[int]]
    ) -> None:
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """

        for i in range(len(index_info)):
            buffer_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer.keys())) <= buffer_id:
                priority = priorities[i, :]

                # Update priorities starting from the game position
                start_idx = game_pos
                end_idx = min(
                    game_pos + len(priority), len(self.buffer[buffer_id].priorities)
                )

                # Update position priorities of game_id in range of start_index to end_index
                new_priorities = self.buffer[buffer_id].priorities.copy()
                new_priorities[start_idx:end_idx] = priority[: end_idx - start_idx]

                self.buffer[buffer_id].update_priorities(new_priorities)

    def compute_target_value(self: Self, entry: ReplayBufferEntry, idx: int) -> float:
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.

        # Find the actual reward at the end or 0 if outside of range of game
        bootstrap_index = idx + self.config.td_steps
        if bootstrap_index < len(entry.game_history.root_values):
            root_values = (
                entry.game_history.root_values
                if entry.reanalysed_predicted_root_values is None
                else entry.reanalysed_predicted_root_values
            )
            last_step_value = root_values[bootstrap_index]

            # Discount it by how far in the future
            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            entry.game_history.reward_history[idx + 1 : bootstrap_index + 1]
        ):
            # Accumulate the total reward discounted by the dicsunt factor
            value += reward * self.config.discount**i

        return value

    def make_target(self: Self, entry: ReplayBufferEntry, state_index: int) -> tuple[
        list[float],
        list[float],
        list[list[float]],
        list[int],
        list[np.ndarray],
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
            value = self.compute_target_value(entry, current_index)

            if current_index < len(entry.game_history.root_values):
                target_values.append(value)
                target_rewards.append(entry.game_history.reward_history[current_index])
                target_policies.append(entry.game_history.child_visits[current_index])
                actions.append(entry.game_history.action_history[current_index])
                params.append(entry.game_history.param_history[current_index])
            elif current_index == len(entry.game_history.root_values):
                target_values.append(0)
                target_rewards.append(entry.game_history.reward_history[current_index])
                target_policies.append([0] * len(self.config.action_space))
                actions.append(np.random.choice(range(len(self.config.action_space))))
                params.append(np.zeros(sum(self.config.action_space)))
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append([0] * len(self.config.action_space))
                actions.append(np.random.choice(range(len(self.config.action_space))))
                params.append(np.zeros(sum(self.config.action_space)))

        return (
            target_values,
            target_rewards,
            target_policies,
            actions,
            params,
        )

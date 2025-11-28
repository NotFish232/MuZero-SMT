import math
import random
import time

import numpy as np
import ray
import torch as T
from ray.actor import ActorProxy
from typing_extensions import Any, Self, Type

from mu_zero_smt.environments.abstract_environment import AbstractEnvironment
from mu_zero_smt.models import (
    MuZeroNetwork,
    one_hot_encode,
    sample_continuous_params,
    support_to_scalar,
)
from mu_zero_smt.replay_buffer import ReplayBuffer
from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.utils.config import MuZeroConfig
from mu_zero_smt.utils.utils import Mode


class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(
        self: Self,
        initial_checkpoint: dict[str, Any],
        Environment: Type[AbstractEnvironment],
        mode: Mode,
        config: MuZeroConfig,
        seed: int,
        worker_id: int,
    ) -> None:
        self.config = config
        self.seed = seed
        self.worker_id = worker_id

        # Fix random generator seed
        np.random.seed(seed)
        T.manual_seed(seed)

        self.env = Environment(mode, seed=self.seed)
        self.mode = mode

        # Initialize the network
        self.model = self.config.network.from_config(self.config)
        self.model.load_state_dict(initial_checkpoint["weights"])
        self.model.to(T.device("cpu"))
        self.model.eval()

    @ray.method
    def continuous_self_play(
        self: Self,
        shared_storage: ActorProxy[SharedStorage],
        replay_buffer: ActorProxy[ReplayBuffer] | None,
    ) -> None:
        """
        Runs continuous self play with an environment
        """

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.load_state_dict(
                ray.get(shared_storage.get_info.remote("weights"))
            )

            temperature = (
                self.config.visit_softmax_temperature_fn(
                    self.config,
                    ray.get(shared_storage.get_info.remote("training_step")),
                )
                if self.mode == "train"
                else 0
            )

            if self.mode == "train":
                game_history = self.play_game(temperature)

                shared_storage.update_info.remote(
                    "self_play_results", self.env.episode_stats()
                )

                if replay_buffer is not None:
                    replay_buffer.save_game.remote(game_history, shared_storage)
            else:
                ids = self.env.unique_episodes()

                batch_size = math.ceil(len(ids) / self.config.num_eval_workers)

                batch_start = self.worker_id * batch_size
                batch_end = min(batch_start + batch_size, len(ids))

                for i in range(batch_start, batch_end):
                    self.play_game(0, ids[i])

                    shared_storage.update_info.remote(
                        f"{self.mode}_results", self.env.episode_stats()
                    )

                shared_storage.update_info.remote(
                    f"finished_{self.mode}_workers", self.worker_id
                )

                # Wait for all eval actors to finish and the main thread to clear it
                while (
                    len(
                        ray.get(
                            shared_storage.get_info.remote(
                                f"finished_{self.mode}_workers"
                            )
                        )
                    )
                    != 0
                ):
                    time.sleep(1)

            # Managing the self-play / training ratio
            if self.mode == "train" and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.env.close()

    def play_game(
        self: Self,
        temperature: float,
        env_id: int | None = None,
    ) -> "GameHistory":
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """

        # Initial observation
        observation = self.env.reset(env_id)

        # Initial game history with a dummy entry for the root node
        game_history = GameHistory()

        game_history.action_history.append(0)
        game_history.param_history.append(np.zeros(self.config.continuous_action_space))
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)

        done = False

        with T.no_grad():
            while not done:
                # Stack the last `self.config.stacked_observations` number of observations
                stacked_observations = game_history.get_stacked_observations(
                    -1,
                    self.config.stacked_observations,
                    self.config.discrete_action_space,
                )

                # Choose the next action based on MCTS' visit distributions and a temperature parameter
                root = MCTS(self.config).run(
                    self.model,
                    stacked_observations,
                    True,
                )
                action, params = SelfPlay.select_action(root, temperature)

                observation, reward, done = self.env.step(
                    action, 1 / (1 + np.exp(-params))
                )

                game_history.store_search_statistics(
                    root, self.config.discrete_action_space
                )

                # Next batch
                game_history.action_history.append(action)
                game_history.param_history.append(params)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)

        return game_history

    @staticmethod
    def select_action(node: "MCTSNode", temperature: float) -> tuple[int, np.ndarray]:
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """

        actions = []
        l_visit_counts = []

        for action, lst in node.children.items():
            actions.append(action)
            l_visit_counts.append(sum(n.visit_count for n, _ in lst))

        visit_counts = np.array(l_visit_counts)

        if temperature == 0:
            # Greedly select the best action
            choice = np.argmax(visit_counts)
        elif temperature == float("inf"):
            # Select each action with equal probability
            choice = np.random.choice(range(len(visit_counts)))
        else:
            # See paper appendix Data Generation

            # Select each action with a probability given by a weight of the visit count and temperature
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )

            choice = np.random.choice(
                range(len(visit_counts)), p=visit_count_distribution
            )

        action = actions[choice]
        # Weight params by visit count to form our param policy
        params = np.sum(
            [param * n.visit_count for n, param in node.children[action]],
            axis=0,
        ) / sum(n.visit_count for n, _ in node.children[action])

        return action, params


class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self: Self, config: MuZeroConfig) -> None:
        self.config = config

    def run(
        self: Self,
        model: MuZeroNetwork,
        raw_observation: np.ndarray,
        add_exploration_noise: bool,
    ) -> "MCTSNode":
        """
        Runs the MCTS algorithm starting from a root node and expanding outward to find good moves

        Args:
            raw_observation (np.ndarray): The observation from the game
            add_exploration_noise (bool): Whether to add Dirichlet noise to the root node

        Returns:
            tuple[MCTSNode, dict[str, Any]]: The root node of the MCTS algorithm along with some data
        """

        device = next(iter(model.parameters())).device

        # Initialize root of MCTS search with no priors
        root = MCTSNode(0)

        # Convert the observation from a numpy array to a tensor
        observation = T.tensor(
            raw_observation, dtype=T.float32, device=next(model.parameters()).device
        )

        # Get inital inference from model
        (
            value_support,
            reward_support,
            policy_logits,
            hidden_state,
        ) = model.initial_inference(observation)

        continuous_params = sample_continuous_params(
            policy_logits,
            self.config.continuous_action_space,
            self.config.num_continuous_samples,
        )

        # Convert model predicted root value and reward to an actual scalar
        value = support_to_scalar(value_support, self.config.support_size).item()
        reward = support_to_scalar(reward_support, self.config.support_size).item()

        # Expand Root
        root.expand(
            hidden_state,
            reward,
            self.config.discrete_action_space,
            policy_logits,
            continuous_params,
        )

        # Add dirichlet exploration noise to root to make children selection non-deterministic
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            while node.is_expanded():
                node, action, continuous_params = self.select_child(node, min_max_stats)
                search_path.append(node)

            # Get the parent of the last node which is not expanded
            # Calculate the new hidden state, and the policy / reward of that node based on the parent's hiden state
            parent = search_path[-2]

            value_support, reward_support, policy_logits, hidden_state = (
                model.recurrent_inference(
                    parent.hidden_state,
                    T.concat(
                        (
                            one_hot_encode(
                                T.tensor([[action]], device=device),
                                self.config.discrete_action_space,
                            ),
                            continuous_params.unsqueeze(0),
                        ),
                        dim=1,
                    ),
                )
            )

            continuous_params = sample_continuous_params(
                policy_logits,
                self.config.continuous_action_space,
                self.config.num_continuous_samples,
            )

            # Convert model predicted value and reward to scalars
            value = support_to_scalar(value_support, self.config.support_size).item()
            reward = support_to_scalar(reward_support, self.config.support_size).item()

            # Expand the first unexpanded node
            node.expand(
                hidden_state,
                reward,
                self.config.discrete_action_space,
                policy_logits,
                continuous_params,
            )

            # Propagate up the search path with the value received
            self.backpropagate(search_path, value, min_max_stats)

        return root

    def select_child(
        self: Self, node: "MCTSNode", min_max_stats: "MinMaxStats"
    ) -> tuple["MCTSNode", int, T.Tensor]:
        """
        Select the child with the highest UCB score.

        Args:
            node (MCTSNode): The current node that we are exploring its children
            min_max_stats (MinMaxStats): Stats of the tree for normalizing

        Returns:
            tuple[MCTSNode, int, T.Tensor]: tuple of the child, associated action, and continuous params
        """

        # Find the child with the maximum PUCT score
        node_action_tuples = []
        puct_scores = []

        for action, lst in node.children.items():
            for child, params in lst:
                node_action_tuples.append((child, action, params))
                puct_scores.append(self.puct_score(node, child, min_max_stats))

        max_puct_score = max(puct_scores)

        # Only selecting the first introduces bias that is actually noticable for performance
        # Break this bias by selecting a random choice
        return random.choice(
            [
                tup
                for tup, puct_score in zip(node_action_tuples, puct_scores)
                if puct_score == max_puct_score
            ]
        )

    def puct_score(
        self: Self, parent: "MCTSNode", child: "MCTSNode", min_max_stats: "MinMaxStats"
    ) -> float:
        """
        Calculates the Polynomial Upper Confidence Tree score for the passed child

        Args:
            parent (MCTSNode): The parent of the node
            child (MCTSNode): The node whose PUCT score is being calculated
            min_max_stats (MinMaxStats): Stats of the tree for normalizing
        """

        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_prob

        if child.visit_count > 0:
            # Mean value of Q
            value_score = min_max_stats.normalize(child.value())
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(
        self: Self,
        search_path: list["MCTSNode"],
        value: float,
        min_max_stats: "MinMaxStats",
    ) -> None:
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.

        Args:
            search_path (list[MCTSNode]): the path of nodes visited during each rollout
            value (float): The end value when the game was terminated
            min_max_stats (MinMaxStats): The stats of the tree for normalizing
        """

        for node in reversed(search_path):
            # Bootstrap the rewards backwards using the predicted rewards and end value
            value = node.reward + self.config.discount * value

            # Update node values
            node.value_sum += value
            node.visit_count += 1

            min_max_stats.update(node.value())


class MCTSNode:
    def __init__(self: Self, prior_prob: float) -> None:
        """
        Args:
            prior_prob (float): The initial probability of selecting this node
        """

        self.prior_prob = prior_prob

        # Node stats
        self.visit_count = 0
        self.value_sum = 0.0

        # Information about the current state
        self.hidden_state: T.Tensor = T.empty(0)
        self.reward = 0.0

        self.children: dict[int, list[tuple[MCTSNode, T.Tensor]]] = {}

    def is_expanded(self: Self) -> bool:
        """
        Whether this node has been expanded yet
        """

        return len(self.children) > 0

    def value(self: Self) -> float:
        """
        The value of this node, which is the mean value from simulations
        """

        if self.visit_count == 0:
            return 0

        return self.value_sum / self.visit_count

    def expand(
        self: Self,
        hidden_state: T.Tensor,
        reward: float,
        action_space: int,
        policy_logits: T.Tensor,
        continous_params: T.Tensor,
    ) -> None:
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.

        Args:
            hidden_state (torch.Tensor): The tensor encoding of the current state from the representation network
            reward (float): The reward received at this state
            action_space (int) The available actions at the current state (might not all be legal)
            policy_logits (torch.Tensor): Logits of the policy for the current state
        """

        self.hidden_state = hidden_state
        self.reward = reward

        # Convert logits to values through softmax and mask out actions not in action space
        policy_values = T.softmax(policy_logits[0, :action_space], dim=0).tolist()

        # Initialize children with the probability predicted by the policy
        for action, policy_prob in zip(range(action_space), policy_values):
            if continous_params.numel() == 0:
                self.children[action] = [(MCTSNode(policy_prob), T.empty(0))]
            else:
                self.children[action] = [
                    (MCTSNode(policy_prob / continous_params.shape[0]), c)
                    for c in continous_params
                ]

    def add_exploration_noise(
        self: Self, dirichlet_alpha: float, exploration_fraction: float
    ) -> None:
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.

        Args:
            dirichlet_alpha (float): The alpha parameter for the dirichlet distribution
            exploration_fraction (float): The fraction of noise to use
        """

        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        f = exploration_fraction

        # Set the prior probabilities of children as a mix of the network prediction and noise
        for a, n in zip(actions, noise):
            for node, _ in self.children[a]:
                node.prior_prob = node.prior_prob * (1 - f) + n * f


class GameHistory:
    """
    Store only useful information of a self-play game.
    """

    def __init__(self: Self) -> None:
        self.observation_history: list[np.ndarray] = []
        self.action_history: list[int] = []
        self.param_history: list[np.ndarray] = []
        self.reward_history: list[float] = []
        self.child_visits: list[list[float]] = []
        self.root_values: list[float] = []

    def store_search_statistics(
        self: Self, root: MCTSNode | None, action_space: int
    ) -> None:
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(
                n.visit_count for lst in root.children.values() for n, _ in lst
            )
            self.child_visits.append(
                [
                    sum(n.visit_count for n, _ in root.children[a]) / sum_visits
                    for a in range(action_space)
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(-1)

    def get_stacked_observations(
        self: Self, index: int, num_stacked_observations: int, action_space_size: int
    ) -> np.ndarray:
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            # Previous observation is the last observation + an array of the previous actions of same size as the observation
            if past_observation_index >= 0:
                previous_observation = np.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            np.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = np.concatenate(
                    (
                        np.zeros_like(self.observation_history[index]),
                        [np.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = np.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self: Self) -> None:
        self.maximum = float("-inf")
        self.minimum = float("inf")

    def update(self: Self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self: Self, value: float) -> float:
        """
        Normalizes value based on the current minimum and maximum of the tree
        """

        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

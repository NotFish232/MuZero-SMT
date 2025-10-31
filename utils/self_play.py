import math
import time

import numpy
import numpy as np
import ray
import torch as T
from typing_extensions import Any, Self, Type

from games.abstract_game import AbstractGame, MuZeroConfig
from utils.models import MuZeroNetwork, support_to_scalar


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(
        self: Self,
        initial_checkpoint: dict[str, Any],
        Game: Type[AbstractGame],
        config: MuZeroConfig,
        seed: int,
    ) -> None:
        self.config = config
        self.game = Game()

        # Fix random generator seed
        numpy.random.seed(seed)
        T.manual_seed(seed)

        # Initialize the network
        self.model = MuZeroNetwork.from_config(self.config)
        self.model.load_state_dict(initial_checkpoint["weights"])
        self.model.to(T.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.load_state_dict(
                ray.get(shared_storage.get_info.remote("weights"))
            )

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        self.config,
                        ray.get(shared_storage.get_info.remote("training_step")),
                    ),
                    self.config.temperature_threshold,
                    False,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
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

        self.close_game()

    def play_game(self, temperature, temperature_threshold, render):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)

        done = False

        if render:
            self.game.render()

        with T.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                # Choose the action
                root, mcts_info = MCTS(self.config).run(
                    self.model,
                    stacked_observations,
                    self.game.legal_actions(),
                    True,
                )
                action = self.select_action(
                    root,
                    (
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0
                    ),
                )

                observation, reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)

        return game_history

    def close_game(self):
        self.game.close()

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
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
        legal_actions: list[int],
        add_exploration_noise: bool,
    ) -> tuple["MCTSNode", dict[str, Any]]:
        """
        Runs the MCTS algorithm starting from a root node and expanding outward to find good moves

        Args:
            raw_observation (np.ndarray): The observation from the game
            legal_actions (list[int]): The actions an agent can take in the environment

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
            root_predicted_value,
            reward,
            policy_logits,
            hidden_state,
        ) = model.initial_inference(observation)

        # Convert model predicted root value and reward to an actual scalar
        root_predicted_value = support_to_scalar(
            root_predicted_value, self.config.support_size
        ).item()
        reward = support_to_scalar(reward, self.config.support_size).item()

        # Expand Root
        root.expand(
            hidden_state,
            reward,
            legal_actions,
            policy_logits,
        )

        # Add dirichlet exploration noise to root to make children selection non-deterministic
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.is_expanded():
                current_tree_depth += 1
                node, action = self.select_child(node, min_max_stats)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                T.tensor([[action]]).to(device),
            )
            value = support_to_scalar(value, self.config.support_size).item()
            reward = support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                hidden_state,
                reward,
                self.config.action_space,
                policy_logits,
            )

            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(
        self: Self, node: "MCTSNode", min_max_stats: "MinMaxStats"
    ) -> tuple["MCTSNode", int]:
        """
        Select the child with the highest UCB score.

        Args:
            node (MCTSNode): The current node that we are exploring its children
            min_max_stats (MinMaxStats): Stats of the tree for normalizing

        Returns:
            tuple[MCTSNode, int]: tuple of the child and associated action
        """


        selected_action = -1
        max_puct_score = -1.0

        # Find the child with the maximum PUCT score
        # Don't worry about multiple having same score since that never happens
        for action, child in node.children.items():
            puct_score = self.puct_score(node, child, min_max_stats)

            if puct_score > max_puct_score:
                selected_action = action
                max_puct_score = puct_score


        return node.children[selected_action], selected_action

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
        self.hidden_state: T.Tensor | None = None
        self.reward = 0.0

        self.children: dict[int, MCTSNode] = {}

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
        action_space: list[int],
        policy_logits: T.Tensor,
    ) -> None:
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.

        Args:
            hidden_state (torch.Tensor): The tensor encoding of the current state from the representation network
            reward (float): The reward received at this state
            action_space (list[int]): The available actions at the current state (might not all be legal)
            policy_logits (torch.Tensor): Logits of the policy for the current state
        """

        self.hidden_state = hidden_state
        self.reward = reward

        # Convert logits to values through softmax and mask out actions not in action space
        policy_values = T.softmax(policy_logits[0, action_space], dim=0).tolist()

        # Initialize children with the probability predicted by the policy
        for action, policy_prob in zip(action_space, policy_values):
            self.children[action] = MCTSNode(policy_prob)

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
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        f = exploration_fraction

        # Set the prior probabilities of children as a mix of the network prediction and noise
        for a, n in zip(actions, noise):
            self.children[a].prior_prob = self.children[a].prior_prob * (1 - f) + n * f


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    (
                        root.children[a].visit_count / sum_visits
                        if a in root.children
                        else 0
                    )
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
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
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
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

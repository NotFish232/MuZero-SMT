import random
from time import perf_counter

import numpy as np
import z3  # type: ignore
from typing_extensions import Any, Self, override

from mu_zero_smt.utils import RawObservation, RunMode

from ..base_environment import BaseEnvironment
from .dataset import SMTDataset
from .embeddings import SMTEmbeddings


def map_raw_val(raw_val: float, typ: str) -> Any:
    """
    Maps a raw value with a type to the actual value fed into the network
    For instance if the typ is bool, true if the raw_val >= 0.5 else false
    """

    if typ == "bool":
        return round(raw_val) == 1

    raise NotImplementedError(f'Unknown value type: "{typ}"')


class SMTEnvironment(BaseEnvironment):
    """
    Game wrapper.
    """

    def __init__(
        self: Self,
        mode: RunMode = "train",
        shuffle: bool = True,
        seed: int | None = None,
        *,
        benchmark: str,
        embedding_type: str,
        embedding_config: dict[str, Any],
        tactics: dict[str, dict[str, str]],
        solving_timeout: float,
        max_num_tactics: int,
        split: dict[RunMode, float],
    ) -> None:
        self.mode = mode
        self.shuffle = shuffle

        random.seed(seed)

        self.benchmark = benchmark

        self.tactics = [*tactics.keys()]
        self.tactic_parameters = tactics

        # Embeds the formula
        self.embedder = SMTEmbeddings.new(embedding_type, embedding_config)

        self.solving_timeout = solving_timeout
        self.max_num_tactics = max_num_tactics

        self.dataset = SMTDataset(self.benchmark, self.mode, split)

        # Environment data that resets each episode
        self.time_spent = 0.0
        self.tactics_applied: list[tuple[str, dict[str, Any]]] = []

        self.selected_idx = -1

        self.current_goal: z3.Goal

    def _get_observation(self: Self) -> RawObservation:
        return self.embedder.embed(
            self.current_goal, self.time_spent / self.solving_timeout
        )

    @override
    def step(
        self: Self, action: int, raw_params: np.ndarray
    ) -> tuple[RawObservation, float, bool]:
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        tactic = self.tactics[action]

        param_to_val = {}

        # Map the raw parameter value passed into the actual value used in the solver
        for idx, (p_name, typ) in enumerate(self.tactic_parameters[tactic].items()):
            # Timeout is special, its a fraction of the remaining time
            if p_name == "timeout":
                param_to_val[p_name] = max(
                    int(
                        raw_params[idx]
                        * (self.solving_timeout - self.time_spent)
                        * 1_000
                    ),
                    10,
                )
            else:
                param_to_val[p_name] = map_raw_val(raw_params[idx], typ)

        reward = 0.0
        done = False

        self.tactics_applied.append((self.tactics[action], param_to_val))

        params_ref = z3.ParamsRef()
        for param, val in param_to_val.items():
            if param == "timeout":
                continue

            params_ref.set(param, val)

        acc_tactic = z3.TryFor(
            z3.WithParams(z3.Tactic(self.tactics[action]), params_ref),
            param_to_val["timeout"],
        )

        start = perf_counter()

        try:
            sub_goals = acc_tactic(self.current_goal)

            if len(sub_goals) != 1:
                raise Exception(
                    f"Expected 1 subgoal but found {len(sub_goals)} subgoals instead"
                )

            self.current_goal = sub_goals[0]

            if len(self.current_goal) == 0 or self.current_goal.inconsistent():
                reward = 2 - self.time_spent / self.solving_timeout
                done = True

        except z3.Z3Exception as e:
            msg = e.args[0].decode()

            if msg != "canceled":
                reward = -0.1

        finally:
            end = perf_counter()

            self.time_spent += end - start

            if self.time_spent > self.solving_timeout:
                reward = -1
                done = True

            elif len(self.tactics_applied) >= self.max_num_tactics:
                reward = -1
                done = True

        return self._get_observation(), reward, done

    @override
    def reset(self: Self, episode_id: int | None = None) -> RawObservation:
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """

        # Run each benchmark sequentially in shuffle mode
        if episode_id is not None:
            self.selected_idx = self.dataset.id_to_idx[episode_id]
        else:
            if self.shuffle:
                self.selected_idx = random.randint(0, len(self.dataset) - 1)
            else:
                self.selected_idx += 1

        self.current_goal = z3.Goal()
        self.current_goal.add(z3.parse_smt2_file(str(self.dataset[self.selected_idx])))

        self.time_spent = 0.0
        self.tactics_applied = []

        return self._get_observation()

    @override
    def unique_episodes(self: Self) -> list[int]:
        return self.dataset.idxs

    @override
    def episode_stats(self: Self) -> dict[str, Any]:
        result = ""

        if len(self.current_goal) == 0:
            result = "SAT"
        elif self.current_goal.inconsistent():
            result = "UNSAT"
        elif self.time_spent > self.solving_timeout:
            result = "TIMEOUT"
        elif len(self.tactics_applied) >= self.max_num_tactics:
            result = "MAX_NUM_TACTICS"

        return {
            "id": self.dataset.idxs[self.selected_idx],
            "name": self.dataset[self.selected_idx].stem,
            "tactic_history": self.tactics_applied,
            "time": self.time_spent,
            "result": result,
            "successful": result in ("SAT", "UNSAT"),
        }

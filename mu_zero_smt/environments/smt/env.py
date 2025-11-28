import random
from time import perf_counter

import numpy as np
import z3  # type: ignore
from typing_extensions import Any, Self, override

from mu_zero_smt.utils.utils import Mode

from ..abstract_environment import AbstractEnvironment
from .dataset import SMTDataset


class SMTEnvironment(AbstractEnvironment):
    """
    Game wrapper.
    """

    def __init__(
        self: Self,
        mode: Mode = "train",
        seed: int | None = None,
        *,
        tactics: list[str],
        probes: list[str],
        solving_timeout: float,
        max_num_tactics: int,
        split: dict[Mode, float],
    ) -> None:
        self.mode = mode
        random.seed(seed)

        self.dataset = SMTDataset("QF_NIA/CInteger", self.mode, split)

        self.tactics = tactics
        self.probes = probes
        self.solving_timeout = solving_timeout
        self.max_num_tactics = max_num_tactics

        self.time_spent = 0.0
        self.tactics_applied: list[tuple[str, list[float]]] = []

        self.selected_idx = -1

        self.current_goal: z3.Goal

    def _get_observation(self: Self) -> np.ndarray:
        values = np.zeros(len(self.probes) + 1, dtype=np.float64)

        for i, probe in enumerate(self.probes):
            probe_res = z3.Probe(probe)(self.current_goal)

            values[i] = probe_res

        values[len(self.probes)] = self.time_spent

        return values.reshape(1, 1, -1)

    @override
    def step(
        self: Self, action: int, params: np.ndarray
    ) -> tuple[np.ndarray, float, bool]:
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        timeout = params[0]

        reward = 0.0
        done = False

        self.tactics_applied.append((self.tactics[action], params.tolist()))

        acc_tactic = z3.TryFor(
            z3.Tactic(self.tactics[action]),
            max(int(timeout * (self.solving_timeout - self.time_spent) * 1_000), 10),
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
    def reset(self: Self, episode_id: int | None = None) -> np.ndarray:
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """

        # Run each benchmark sequentially in test mode
        if episode_id is not None:
            self.selected_idx = self.dataset.id_to_idx[episode_id]
        else:
            if self.mode in ["test", "eval"]:
                self.selected_idx += 1
            else:
                self.selected_idx = random.randint(0, len(self.dataset) - 1)

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
            "name": self.dataset[self.selected_idx].name,
            "tactic_history": self.tactics_applied,
            "time": self.time_spent,
            "result": result,
            "successful": result in ("SAT", "UNSAT"),
        }

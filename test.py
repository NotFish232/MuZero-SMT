import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import torch as T
from tqdm import tqdm  # type: ignore

from mu_zero_smt.environments.smt.smt import TACTICS
from mu_zero_smt.environments.smt.smt import Game as SMTGame
from mu_zero_smt.self_play import MCTS, GameHistory, SelfPlay

CHECKPOINT = (
    f"{next(f for f in Path("results/smt").rglob("*") if f.is_dir())}/model.checkpoint"
)


@T.no_grad()
def main() -> None:
    game = SMTGame(test_mode=True)

    config = SMTGame.get_config()

    checkpoint = T.load(CHECKPOINT, weights_only=False)

    model = config.network.from_config(config)
    model.load_state_dict(checkpoint["weights"])
    model.eval()

    for _ in tqdm(range(len(game.dataset))):
        observation = game.reset()

        done = False

        game_history = GameHistory()

        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.param_history.append(np.zeros(config.continuous_action_space))

        start = perf_counter()

        while not done:
            stacked_observations = game_history.get_stacked_observations(
                -1, config.stacked_observations, config.discrete_action_space
            )

            # Choose the action
            root = MCTS(config).run(
                model,
                stacked_observations,
                True,
            )
            action, params = SelfPlay.select_action(root, 0)

            observation, reward, done = game.step(action, 1 / (1 + np.exp(-params)))

            game_history.store_search_statistics(root, config.discrete_action_space)

            # Next batch
            game_history.action_history.append(action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
            game_history.param_history.append(1 / (1 + np.exp(-params)))

        end = perf_counter()

        total_time = end - start

        logging.info(
            f"EVAL | {game.dataset[game.selected_idx].stem} | {"SOLVED" if reward >= 1 else "UNSOLVED"} ({total_time:.2f}s)\n"
            f"TACTICS: {[str(TACTICS[a]) + " (" + str(p) + ")"  for a, p in zip(game_history.action_history[1:], game_history.param_history[1:])]}\n\n"
        )


if __name__ == "__main__":
    main()

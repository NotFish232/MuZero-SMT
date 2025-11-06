import logging
from time import perf_counter

import torch as T
from tqdm import tqdm  # type: ignore

from mu_zero_smt.games.smt import Game as SMTGame
from mu_zero_smt.self_play import MCTS, GameHistory, SelfPlay

CHECKPOINT = "results/smt/2025-11-01--22-29-26/model.checkpoint"


@T.no_grad()
def main() -> None:
    game = SMTGame(test_mode=True)

    config = SMTGame.get_config()

    checkpoint = T.load(CHECKPOINT, weights_only=False)

    model = config.network.from_config(config)
    model.load_state_dict(checkpoint["weights"])
    model.eval()

    for _ in tqdm(range(len(game.files))):
        observation = game.reset()

        done = False

        game_history = GameHistory()

        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)

        start = perf_counter()

        while not done:
            stacked_observations = game_history.get_stacked_observations(
                -1, config.stacked_observations, len(config.action_space)
            )

            # Choose the action
            root = MCTS(config).run(
                model,
                stacked_observations,
                game.legal_actions(),
                True,
            )
            action = SelfPlay.select_action(root, 0)

            observation, reward, done = game.step(action)

            game_history.store_search_statistics(root, config.action_space)

            # Next batch
            game_history.action_history.append(action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)

        end = perf_counter()

        total_time = end - start

        logging.info(
            f"EVAL | {game.files[game.selected_idx].stem} | {"SOLVED" if reward >= 1 else "UNSOLVED"} ({total_time:.2f}s)\n"
            f"TACTICS: {[game.tactics[a] for a in game_history.action_history[1:]]}\n\n"
        )


if __name__ == "__main__":
    main()

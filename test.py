from pathlib import Path

from train import MuZero

CHECKPOINT_DIR = "results/smt/2025-10-28--13-13-50"

GAME_NAME = "smt"


def main() -> None:
    muzero = MuZero(GAME_NAME)

    if CHECKPOINT_DIR is not None:
        checkpoint_path = Path(CHECKPOINT_DIR) / "model.checkpoint"
        replay_buffer_path = Path(CHECKPOINT_DIR) / "replay_buffer.pkl"

        muzero.load_model(checkpoint_path, replay_buffer_path)

    res = muzero.test()

    print(res)


if __name__ == "__main__":
    main()

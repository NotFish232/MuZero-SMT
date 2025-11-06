import copy
from pathlib import Path

import ray
import torch as T
from typing_extensions import Any, Self

from mu_zero_smt.utils.config import MuZeroConfig


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self: Self, checkpoint: dict[str, Any], config: MuZeroConfig) -> None:
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self: Self, path: Path | None = None) -> None:
        if not path:
            path = self.config.results_path / "model.checkpoint"

        T.save(self.current_checkpoint, path)

    def get_checkpoint(self: Self) -> dict[str, Any]:
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self: Self, keys: str | list[str]) -> Any | dict[str, Any]:
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(
        self: Self, keys: str | dict[str, Any], values: Any | None = None
    ) -> None:
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

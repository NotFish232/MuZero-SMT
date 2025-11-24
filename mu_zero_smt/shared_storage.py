import copy
from pathlib import Path

import ray
import torch as T
from typing_extensions import Any, Self

from mu_zero_smt.utils.config import MuZeroConfig


class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self: Self, checkpoint: dict[str, Any], config: MuZeroConfig) -> None:
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    @ray.method
    def save_checkpoint(self: Self, path: Path | None = None) -> None:
        if not path:
            path = self.config.results_path / "model.checkpoint"

        T.save(self.current_checkpoint, path)

    def get_checkpoint(self: Self) -> dict[str, Any]:
        return copy.deepcopy(self.current_checkpoint)

    @ray.method
    def get_info(self: Self, key: str) -> Any:
        return self.current_checkpoint[key]

    @ray.method
    def get_info_batch(self: Self, keys: list[str]) -> dict[str, Any]:
        return {key: self.current_checkpoint[key] for key in keys}

    @ray.method
    def set_info(
        self: Self,
        key: str,
        value: Any,
    ) -> None:
        self.current_checkpoint[key] = value

    @ray.method
    def update_info(self: Self, key: str, value: Any) -> None:
        if key not in self.current_checkpoint:
            self.current_checkpoint[key] = []
        self.current_checkpoint[key].append(value)

    @ray.method
    def set_info_batch(self: Self, key_and_values: dict[str, Any]) -> None:
        self.current_checkpoint.update(key_and_values)

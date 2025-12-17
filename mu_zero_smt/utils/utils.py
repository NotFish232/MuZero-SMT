import numpy as np
import torch as T
from torch_geometric.data import Batch, Data  # type: ignore
from typing_extensions import Literal, Union

RawObservation = Union[np.ndarray, Data]
CollatedObservation = Union[T.Tensor, Batch]
RunMode = Literal["train"] | Literal["eval"] | Literal["test"]


def collate_observations(observations: list[RawObservation]) -> CollatedObservation:
    """
    Collate the observations together into a "batch". Different observation types collate differently
    """

    if isinstance(observations[0], np.ndarray):
        return T.tensor(np.array(observations), dtype=T.float32)
    elif isinstance(observations[0], Data):
        return Batch.from_data_list(observations)

    raise ValueError(f"Unknown observation type: {type(observations[0])}")

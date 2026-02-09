import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as T
from torch import nn
from torch_geometric.data import Batch, Data  # type: ignore
from typing_extensions import Any, Literal, Type, Union

# A raw observation is the result of a call to `step` or `reset` on an environment
# A collated observation is the result of batching a list of raw observations
# Run mode is to set the environment / dataset into various modes
RawObservation = Union[np.ndarray, Data]
CollatedObservation = Union[T.Tensor, Batch]
RunMode = Literal["train"] | Literal["eval"] | Literal["test"]


@dataclass
class MuZeroConfig:
    """
    A data class holding the config for MuZero
    """

    ### Basic config

    # The name of the experiment for checkpointing / saving
    experiment_name: str
    # Random see for numpy, pytorch, and the envirionment
    seed: int

    ### Environment config

    # Extra arguments passed to the envirionment constructor
    env_config: dict[str, Any]
    # Game dimensions, size of observation as well as action space,
    # which are discrete actions tied to some number of continuous parameters
    observation_size: int
    action_space: list[int]

    ### Model config

    # Model type and extra arguments passed to the model constructor
    model_type: str
    model_config: dict[str, Any]
    # The support size for the value and reward models which predict supports instead of scalars
    support_size: int

    ### Ray config

    # Number of workers for various code functions, like self play, eval, and test
    num_self_play_workers: int
    num_eval_workers: int
    num_test_workers: int

    ### MCTS config

    # Number of MCTS simulations to run
    num_simulations: int
    # Number of samples of continuous parameters during MCTS branching
    num_continuous_samples: int
    # Number of stacked observations to feed into the model
    stacked_observations: int
    # Chronological reward discount
    discount: float
    # Root prior exploration noise
    root_dirichlet_alpha: float
    root_exploration_fraction: float
    # Hyperparameters for the UCB formula
    pb_c_base: float
    pb_c_init: float
    # Temperature for action selection during self play, linearly interpolated between these two values
    temperature_start: float
    temperature_end: float

    ### Training config

    # Split for train / eval / test
    split_ratios: dict[RunMode, float]
    # Total number of training steps
    training_steps: int
    # Batch size for training
    batch_size: int
    # Number of training steps per checkpoint
    checkpoint_interval: int
    # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
    value_loss_weight: float
    # L2 weights regularization
    weight_decay: float
    # Exponential learning rate schedule
    lr_init: float
    lr_decay_rate: float
    lr_decay_steps: int
    # Maximum number of self-play games to keep in the replay buffer
    replay_buffer_size: int
    # Number of steps the game is enrolled during training to create a batch
    num_unroll_steps: int
    # Number of steps used to calculate the target value
    td_steps: int
    # How much priority is given during game selection from the replay buffer
    priority_alpha: float
    # Number of seconds to weight after each training step
    training_delay: float


def load_config() -> MuZeroConfig:
    if len(sys.argv) < 2:
        raise ValueError("config path must be passed as first argument")

    path = Path(sys.argv[1])

    if not path.exists() or not path.is_file():
        raise ValueError(f"{path} is not a valid path to a config")

    config = MuZeroConfig(**json.load(open(path)))

    return config


def load_dataset_split(
    config: MuZeroConfig, remove_info: bool = True
) -> dict[str, list[int]]:
    split_path = Path(f"splits/{config.experiment_name}.json")

    if not split_path.exists() or not split_path.is_file():
        raise ValueError(f"No dataset split found (generate with gen_split.py)")

    with open(split_path) as f:
        dataset_split = json.load(f)
    
    if remove_info:
        del dataset_split["info"]

    return dataset_split


def collate_observations(observations: list[RawObservation]) -> CollatedObservation:
    """
    Collate the raw observations from the environment together into a "batch".
      Different observation types collate differently
    """

    if isinstance(observations[0], np.ndarray):
        return T.tensor(np.array(observations), dtype=T.float32)
    elif isinstance(observations[0], Data):
        return Batch.from_data_list(observations)

    raise ValueError(f"Unknown observation type: {type(observations[0])}")


def dict_to_cpu(dictionary: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively moves all tensors in the dictionary onto the cpu

    Args:
        dictionary (dict[str, Any]): The dictionary

    Returns:
        dict[str, Any]: The dictionary with all tensors on cpu
    """

    cpu_dict: dict[str, Any] = {}

    for key, value in dictionary.items():
        if isinstance(value, T.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value

    return cpu_dict


def support_to_scalar(
    logits: T.Tensor, support_size: int, eps: float = 1e-3
) -> T.Tensor:
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture

    Args:
        logits (T.Tensor - shape (x_1, ..., x_n, 2 * support_size + 1): The tensor with last dimension of supports
        support_size (int): The number of supports
        eps (float): A small epsilon used for inverting the scaling of x

    Returns:
        T.Tensor - shape (x_1, ..., x_n): Tensor of scaler values recovered from supports
    """

    # Softmax the logits so it sums up to one
    probs = T.softmax(logits, dim=-1)

    # Generate supports and sum up the element wise multiplication of support values and probs
    support = T.arange(
        -support_size, support_size + 1, dtype=T.float32, device=probs.device
    )
    x = T.sum(support * probs, dim=-1)

    # Invert the scaling (defined in https://arxivTrue.org/abs/1805.11593)
    # x = T.sign(x) * (
    #     ((T.sqrt(1 + 4 * eps * (T.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    # )

    return x


def scalar_to_support(x: T.Tensor, support_size: int, eps: float = 1e-3) -> T.Tensor:
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture

    Args:
        x (T.Tensor - shape (x_1, ..., x_n)): The tensor of scalars to conver to supports
        support_size (int): The number of supports
        eps (float): An epsilon for the scaling of x

    Returns:
        T.Tensor - shape (x_1, ..., x_n, 2 * support_size + 1): A tensor encoded in supports
    """

    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    # x = T.sign(x) * (T.sqrt(T.abs(x) + 1) - 1) + eps * x

    # Clamp x to the range [-support_size, support_size] and convert it to an integer
    x = T.clamp(x, -support_size, support_size)

    # Get value below corresponding to left-side support
    x_floor = x.floor()

    # Logits of size (W, H, 2 * support_size + 1) that contains the actual supports
    logits = T.zeros((*x.shape, 2 * support_size + 1), device=x.device)

    # Get values and idxs for first supports and second supports
    # Value of first support is how close it is to floor, and value of second is how close it is to the ceiling
    # Index is the floor offset by the support size to deal with negative values
    support_1_vals = (x_floor - x + 1).unsqueeze_(-1)
    support_1_idxs = (x_floor + support_size).to(T.int64).unsqueeze_(-1)

    support_2_vals = 1 - support_1_vals
    support_2_idxs = support_1_idxs + 1
    # If support 2 index overflows past 2 * support_size, wrap it around to index 0
    support_2_idxs.masked_fill_(2 * support_size < support_2_idxs, 0)

    # Scatter the support values into the corresponding idxs
    logits.scatter_(-1, support_1_idxs, support_1_vals)
    logits.scatter_(-1, support_2_idxs, support_2_vals)

    return logits


def sample_continuous_params(
    continuous_logits: T.Tensor, num_continuous_samples: int
) -> T.Tensor:
    """
    Samples continuous params from the policy logits, given mean and standard deviations

    Args:
        continuous_logits (T.Tensor): The tensor of policy logits of shape (continuous_action_space, 2)
        num_continuous_samples (int): How many samples to make

    Returns:
        T.Tensor: Sampled values of dimension (num_samples, continuous_action_space)
    """

    means = continuous_logits[:, 0]

    # Converts from log variance to standard deviation
    stds = T.exp(0.5 * continuous_logits[:, 1])

    return means + stds * T.randn(num_continuous_samples, continuous_logits.shape[0])


def get_param_mask(actions: T.Tensor, action_space: list[int]) -> T.Tensor:
    """
    Constructs a boolean mask for the actions based on the provided action space

    Args:
        actions (T.Tensor): A tensor of shape (N, 1)
        action_space (list[int]): A list of how many parameters per action

    Returns:
        T.Tensor: A tensor of shape (N, sum(action_space)) which is True where an action requires that parameter
    """

    action_space_tensor = T.tensor(action_space, device=actions.device)

    action_offsets = T.cumsum(action_space_tensor, dim=0)

    end_idxs = action_offsets[actions]
    start_idxs = action_offsets[actions] - action_space_tensor[actions]

    param_indices = T.arange(
        T.sum(action_space_tensor).item(), device=actions.device
    ).unsqueeze(0)

    mask = (param_indices >= start_idxs) & (param_indices < end_idxs)

    return mask


def mlp(
    input_size: int,
    layer_sizes: list[int],
    output_size: int,
    activation: Type[nn.Module] = nn.ELU,
    output_activation: Type[nn.Module] = nn.Identity,
) -> nn.Module:
    """
    Constructs a MLP outlined by the passed in sizes

    Args:
        input_size (int): The size of the input layer
        layer_sizes (list[int]): The size of the hidden layers
        output_size (int): The size of the output layer
        activation (nn.Module, optional): The activation function for hidden layers. Defaults to nn.ELU().
        output_activation (nn.Module, optional): The action function for output layer. Defaults to nn.Identity().

    Returns:
        nn.Module: The MLP comprised of the specified parameters
    """

    sizes = [input_size] + layer_sizes + [output_size]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]

    return nn.Sequential(*layers)

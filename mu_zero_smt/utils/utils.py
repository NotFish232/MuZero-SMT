import numpy as np
import torch as T
from torch import nn
from torch_geometric.data import Batch, Data  # type: ignore
from typing_extensions import Any, Literal, Type, Union

RawObservation = Union[np.ndarray, Data]
CollatedObservation = Union[T.Tensor, Batch]
RunMode = Literal["train"] | Literal["eval"] | Literal["test"]


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
        logits (torch.Tensor - shape (x_1, ..., x_n, 2 * support_size + 1): The tensor with last dimension of supports
        support_size (int): The number of supports
        eps (float): A small epsilon used for inverting the scaling of x

    Returns:
        torch.Tensor - shape (x_1, ..., x_n): Tensor of scaler values recovered from supports
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
        x (torch.Tensor - shape (x_1, ..., x_n)): The tensor of scalars to conver to supports
        support_size (int): The number of supports
        eps (float): An epsilon for the scaling of x

    Returns:
        torch.Tensor - shape (x_1, ..., x_n, 2 * support_size + 1): A tensor encoded in supports
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
    stds = continuous_logits[:, 1]

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


def one_hot_encode(x: T.Tensor, n: int) -> T.Tensor:
    """
    One hot encodes a nx1 vector x
    """

    one_hot = T.zeros(
        (x.shape[0], n),
        dtype=T.float32,
        device=x.device,
    )
    one_hot.scatter_(1, x.to(T.int64), 1.0)

    return one_hot


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

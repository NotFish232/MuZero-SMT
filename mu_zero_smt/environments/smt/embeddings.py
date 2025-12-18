from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch as T
import z3  # type: ignore
from torch.nn import functional as F
from torch_geometric.data import Data  # type: ignore
from typing_extensions import Any, Self, override

from mu_zero_smt.utils.utils import RawObservation


class SMTEmbeddings(ABC):
    """
    Abstract class representing various smt formula embeddings
    """

    @staticmethod
    def new(embedding_type: str, embedding_args: dict[str, Any]) -> "SMTEmbeddings":
        if embedding_type == "probe":
            return ProbeSMTEmbeddings(**embedding_args)
        elif embedding_type == "graph":
            return GraphSMTEmbeddings(**embedding_args)

        raise ValueError(f'Unknown embedding type: "{embedding_type}"')

    @abstractmethod
    def embed(self: Self, goal: z3.Goal, time: float) -> RawObservation:
        """
        Embeds the given goal object

        Args:
            goal (z3.Goal): The current SMT formula
            time (float): The percentage of time currently used
        """


class ProbeSMTEmbeddings(SMTEmbeddings):
    def __init__(self: Self, probes: dict[str, tuple[int, int]]) -> None:
        self.probes = probes

    @override
    def embed(self: Self, goal: z3.Goal, time: float) -> RawObservation:
        values = np.zeros(len(self.probes) + 1, dtype=np.float64)

        for i, (probe, (min_val, max_val)) in enumerate(self.probes.items()):
            probe_res = z3.Probe(probe)(goal)

            values[i] = (probe_res - min_val) / (max_val - min_val)

        values[len(self.probes)] = time

        return values.reshape(1, 1, -1)


Z3_OP_TYPES = sorted(v for k, v in z3.__dict__.items() if k.startswith("Z3_OP"))

# Fast lookup from ast type and op type to id
Z3_OP_TO_ID = dict(zip(Z3_OP_TYPES, range(len(Z3_OP_TYPES))))


class GraphSMTEmbeddings(SMTEmbeddings):
    def __init__(self: Self, embedding_size: int, max_num_nodes: int) -> None:
        self.embedding_size = embedding_size
        self.max_num_nodes = max_num_nodes

        assert len(Z3_OP_TYPES) < self.embedding_size

    @override
    def embed(self: Self, goal: z3.Goal, _time: float) -> RawObservation:
        """
        Constructs a graph from a smt formula

        Args:
            s (z3.AstVector): The AstVector representation of the SMT formula

        Returns:
            Data: The embeddings of each node and the edges.
        """

        num_variable_names = self.embedding_size - len(Z3_OP_TYPES)

        nodes: list[tuple[str | None, int]] = []
        edges: set[tuple[int, int]] = set()
        visited: set[int] = set()

        var_freq: dict[str, int] = {}

        queue: deque[tuple[z3.ExprRef, int]] = deque()

        for ref in goal:
            queue.append((ref, -1))

        while len(queue) > 0 and len(visited) < self.max_num_nodes:
            ref, parent = queue.popleft()

            if ref.get_id() in visited:
                continue
            visited.add(ref.get_id())

            expr_str = None

            # A function (or variable which is a function with arity 0)
            if ref.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                expr_str = ref.decl().name()

                var_freq[expr_str] = var_freq.get(expr_str, 0) + 1

            op_id = Z3_OP_TO_ID[ref.decl().kind()]

            # New expression
            node_idx = len(nodes)

            nodes.append((expr_str, op_id))

            if parent != -1:
                edges.add((node_idx, parent))

            for child_ref in ref.children():
                queue.append((child_ref, node_idx))

        var_to_id = {}

        # Sort based on frequency since we want to truncate the less used variables if we have to
        for i, var_name in enumerate(
            reversed(sorted(var_freq.keys(), key=var_freq.__getitem__))
        ):
            if i < num_variable_names:
                var_to_id[var_name] = i
            else:
                var_to_id[var_name] = num_variable_names - 1

        node_embeddings = []

        for name, op_id in nodes:
            op_embedding = F.one_hot(T.tensor(op_id), len(Z3_OP_TYPES))

            var_embedding = T.zeros(num_variable_names)

            if name is not None:
                var_id = var_to_id[name]
                var_embedding = F.one_hot(T.tensor(var_id), num_variable_names)

            node_embedding = T.concat((op_embedding, var_embedding))

            node_embeddings.append(node_embedding)

        if len(node_embeddings) > 0:
            node_embeddings_tensor = T.stack(node_embeddings)
        else:
            node_embeddings_tensor = T.empty(0, self.embedding_size)

        # Add the other direction of edges and transpose + make it contiguous
        if len(edges) > 0:
            edges_tensor = T.tensor(list(edges), dtype=T.int64)
            edges_tensor = edges_tensor.T.contiguous()
        else:
            edges_tensor = T.empty(2, 0)

        graph = Data(x=node_embeddings_tensor, edge_index=edges_tensor)

        return graph

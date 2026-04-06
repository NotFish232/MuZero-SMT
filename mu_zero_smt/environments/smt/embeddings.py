from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch as T
import z3  # type: ignore
from torch_geometric.data import Data  # type: ignore
from typing_extensions import Any, Self, override

from mu_zero_smt.utils import RawObservation


class SMTEmbeddings(ABC):
    """
    Abstract class representing various smt formula embeddings
    """

    @staticmethod
    def new(embedding_type: str, embedding_config: dict[str, Any]) -> "SMTEmbeddings":
        if embedding_type == "probe":
            return ProbeSMTEmbeddings(**embedding_config)
        elif embedding_type == "graph":
            return GraphSMTEmbeddings(**embedding_config)

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


class GraphSMTEmbeddings(SMTEmbeddings):
    def __init__(
        self: Self, max_num_nodes: int, max_num_ops: int, max_num_vars: int
    ) -> None:
        # We make simple size 3 embeddings
        # Its up to the graph neural net to learn embeddings based on our numbers
        # First 50% of embeddings are reffered to top n most frequent and will have no collisions
        # Other ones are modded and might

        self.max_num_nodes = max_num_nodes

        self.greedy_percentage = 0.5

        self.max_num_ops = max_num_ops
        self.max_num_vars = max_num_vars

        self.num_ops_greedy = int(self.greedy_percentage * self.max_num_ops)
        self.num_ops_remaining = self.max_num_ops - self.num_ops_greedy

        self.num_vars_greedy = int(self.greedy_percentage * self.max_num_vars)
        self.num_vars_remaining = self.max_num_vars - self.num_vars_greedy

        z3_ops = [v for k, v in z3.__dict__.items() if k.startswith("Z3_OP")]

        self.z3_op_to_id = dict(zip(sorted(z3_ops), range(len(z3_ops))))

    @override
    def embed(self: Self, goal: z3.Goal, time: float) -> RawObservation:
        """
        Constructs a graph from a smt formula

        Args:
            s (z3.AstVector): The AstVector representation of the SMT formula

        Returns:
            Data: The embeddings of each node and the edges.
        """

        nodes: list[tuple[str | None, int]] = []
        edges: set[tuple[int, int]] = set()
        visited: set[int] = set()

        queue: deque[tuple[z3.ExprRef, int]] = deque()

        # Embeddings for variable names
        var_to_freq: dict[str, int] = {}

        for ref in goal:
            queue.append((ref, -1))

        # Runs a BFS on the AST of the formula
        while len(queue) > 0:
            ref, parent = queue.popleft()

            if ref.get_id() in visited:
                continue

            visited.add(ref.get_id())

            expr_str = None

            # A function (or variable which is a function with arity 0)
            if ref.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                expr_str = ref.decl().name()

                var_to_freq[expr_str] = var_to_freq.get(expr_str, 0) + 1

            # New expression

            # if we are greater than max num of nodes prune here
            # we don't prune earlier because we still want the var visit counts
            node_idx = len(nodes)

            if node_idx < self.max_num_nodes:
                nodes.append((expr_str, ref.decl().kind()))

                if parent != -1:
                    # store edges as (parent -> child) and (child -> parent)
                    edges.add((parent, node_idx))
                    edges.add((node_idx, parent))

                for child_ref in ref.children():
                    queue.append((child_ref, node_idx))

        sorted_vars = sorted(
            var_to_freq.keys(),
            key=var_to_freq.__getitem__,
            reverse=True,
        )
        var_to_id = dict(zip(sorted_vars, range(len(var_to_freq))))

        node_embeddings = []

        for name, op_id in nodes:
            op_id = self.z3_op_to_id[op_id]

            # If we have too many op embeddings hash the ones higher
            # First k are gauranteed to be collision free
            if op_id >= self.num_ops_greedy:
                op_id = (
                    self.num_ops_greedy
                    + (op_id - self.num_ops_greedy) % self.num_ops_remaining
                )

            var_id = var_to_id[name] if name is not None else self.max_num_vars - 1

            # Same idea here
            if var_id >= self.num_vars_greedy:
                var_id = (
                    self.num_vars_greedy
                    + (var_id - self.num_vars_greedy) % self.num_vars_remaining
                )

            # Full embedding is our op id, var id, and time
            node_embeddings.append(T.tensor([op_id, var_id, time]))

        if len(node_embeddings) > 0:
            node_embeddings_tensor = T.stack(node_embeddings)
        else:
            node_embeddings_tensor = T.empty(0, 3)

        # Transpose the edges and make them contiguous
        if len(edges) > 0:
            edges_tensor = T.tensor(list(edges), dtype=T.int64)
            edges_tensor = edges_tensor.T.contiguous()
        else:
            edges_tensor = T.empty(2, 0, dtype=T.int64)

        graph = Data(x=node_embeddings_tensor, edge_index=edges_tensor)

        return graph

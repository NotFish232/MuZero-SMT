import hashlib
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
    def __init__(self: Self, embedding_size: int, max_num_nodes: int) -> None:
        self.embedding_size = embedding_size
        self.max_num_nodes = max_num_nodes

        # We need one space for time
        self.op_embedding_size = embedding_size // 2
        self.var_name_embedding_size = embedding_size - self.op_embedding_size - 1

        # Embeddings for z3 operators
        self.z3_op_to_proj = {}

        for k, v in z3.__dict__.items():
            if k.startswith("Z3_OP"):
                self.z3_op_to_proj[v] = self._random_projection(
                    k, self.op_embedding_size
                )

    def _stable_hash(self: Self, key: str) -> int:
        """
        Stable deterministic hash of a key to an integer
        """
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
        return int.from_bytes(digest, "big")

    def _random_projection(self: Self, key: str, dim: int) -> T.Tensor:
        """
        Produces a deterministic projection of key into a vector of dimension dim with
        norm ~1

        Returns:
            T.Tensor: tensor of dimension of (dim, )
        """
        seed = self._stable_hash(key) & 0xFFFFFFFF

        proj = []

        for _ in range(dim):
            seed = (seed + 0x9E3779B9) & 0xFFFFFFFF

            # splitmix32 alg
            z = seed
            z = (z ^ (z >> 15)) * 0x85EBCA6B & 0xFFFFFFFF
            z = (z ^ (z >> 13)) * 0xC2B2AE35 & 0xFFFFFFFF
            z ^= z >> 16

            proj.append(1.0 if z & 1 else -1.0)

        # Normalize norm to be ~1
        return T.tensor(proj, dtype=T.float32) / T.sqrt(T.tensor(dim))

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
        var_to_embedding = {}
        empty_var_embedding = self._random_projection("", self.var_name_embedding_size)

        for ref in goal:
            queue.append((ref, -1))

        # Runs a BFS on the AST of the formula until either all nodes are visited
        # or until we reached max_num_nodes
        while len(queue) > 0 and len(visited) < self.max_num_nodes:
            ref, parent = queue.popleft()

            if ref.get_id() in visited:
                continue
            visited.add(ref.get_id())

            expr_str = None

            # A function (or variable which is a function with arity 0)
            if ref.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                expr_str = ref.decl().name()

                if expr_str not in var_to_embedding:
                    var_to_embedding[expr_str] = self._random_projection(
                        expr_str, self.var_name_embedding_size
                    )

            # New expression
            node_idx = len(nodes)

            nodes.append((expr_str, ref.decl().kind()))

            if parent != -1:
                # store edges as (parent -> child) and (child -> parent)
                edges.add((parent, node_idx))
                edges.add((node_idx, parent))

            for child_ref in ref.children():
                queue.append((child_ref, node_idx))

        node_embeddings = []

        for name, op_id in nodes:
            op_embedding = self.z3_op_to_proj[op_id]

            var_embedding = (
                var_to_embedding[name] if name is not None else empty_var_embedding
            )

            time_embedding = T.zeros(1)

            # Full embedding is just one-hot encoded operator + one-hot encoded variable name + space for time
            node_embeddings.append(
                T.concat((op_embedding, var_embedding, time_embedding))
            )

        if len(node_embeddings) > 0:
            node_embeddings_tensor = T.stack(node_embeddings)
        else:
            node_embeddings_tensor = T.empty(0, self.embedding_size)

        # Add a dedicated time node so temporal budget is available to the GNN
        time_node = T.zeros(self.embedding_size)
        time_node[-1] = float(time)

        if node_embeddings_tensor.shape[0] > 0:
            node_embeddings_tensor = T.cat(
                (node_embeddings_tensor, time_node.unsqueeze(0)), dim=0
            )

            # Connect time node to root to enable message passing
            time_node_idx = node_embeddings_tensor.shape[0] - 1
            edges.add((0, time_node_idx))
            edges.add((time_node_idx, 0))
        else:
            node_embeddings_tensor = time_node.unsqueeze(0)

        # Transpose the edges and make them contiguous
        if len(edges) > 0:
            edges_tensor = T.tensor(list(edges), dtype=T.int64)
            edges_tensor = edges_tensor.T.contiguous()
        else:
            edges_tensor = T.empty(2, 0, dtype=T.int64)

        graph = Data(x=node_embeddings_tensor, edge_index=edges_tensor)

        return graph

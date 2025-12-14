import z3  # type: ignore
from torch_geometric.data import Data  # type: ignore
import torch as T
from torch.nn import functional  as F


z3_op_types = sorted(v for k, v in z3.__dict__.items() if k.startswith("Z3_OP"))

# Fast lookup from ast type and op type to id
z3_op_to_id = dict(zip(z3_op_types, range(len(z3_op_types))))


num_variable_names = 2**9 - len(z3_op_to_id)


def build_graph(ast: z3.AstVector) -> Data:
    """
    Constructs a graph from a smt formula

    Args:
        s (z3.AstVector): The AstVector representation of the SMT formula

      Returns:
          Data: The embeddings of each node and the edges.
    """

    def _traverse_tree(
        expr_ref: z3.ExprRef,
        nodes: list[tuple[str | None, int]],
        edges: list[tuple[int, int]],
        var_freq: dict[str, int],
    ) -> None:
        """
        Traverses the syntax tree recursively

        Args:
            expr_ref (z3.ExprRef): The current expression being parsed
            nodes (list[tuple[str | None, int]]): Previous nodes, where each node is a string repr if its an identifier and the op id
            edges (list[tuple[int, int]]): The edges between nodes
            var_freq (dict[str, int]): The frequency of each variable, truncated variables are those with less freq
        """

        expr_str = None

        # A function (or variable which is a function with arity 0)
        if expr_ref.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            expr_str = expr_ref.decl().name()

            if expr_str not in var_freq:
                var_freq[expr_str] = 0
            var_freq[expr_str] += 1

        op_id = z3_op_to_id[expr_ref.decl().kind()]

        node_idx = len(nodes)
        nodes.append((expr_str, op_id))

        for child_ref in expr_ref.children():
            next_idx = len(nodes)

            _traverse_tree(child_ref, nodes, edges, var_freq)

            edges.append((node_idx, next_idx))

    nodes: list[tuple[str | None, int]] = []
    edges: list[tuple[int, int]] = []

    var_freq: dict[str, int] = {}

    for ref in ast:
        _traverse_tree(ref, nodes, edges, var_freq)

    var_to_id = {}

    # Sort based on frequency since we want to truncate the less used variables if we have to
    for i, var_name in enumerate(sorted(var_freq.keys(), key=var_freq.__getitem__)):
        if i < num_variable_names:
            var_to_id[var_name] = i
        else:
            var_to_id[var_name] = num_variable_names - 1

    node_embeddings = []

    for name, op_id in nodes:
        op_embedding = F.one_hot(T.tensor(op_id), len(z3_op_types))

        var_embedding = T.zeros(num_variable_names)

        if name is not None:
            var_id = var_to_id[name]
            var_embedding = F.one_hot(T.tensor(var_id), num_variable_names)

        node_embedding = T.concat((op_embedding, var_embedding))

        node_embeddings.append(node_embedding)
    

    if len(node_embeddings) > 0:
        node_embeddings_tensor = T.stack(node_embeddings)
    else:
        node_embeddings_tensor = T.empty(0, 2**9)

    # Add the other direction of edges and transpose + make it contiguous
    if len(edges) > 0:
        edges_tensor = T.tensor(edges)
        edges_tensor = T.concat((edges_tensor, edges_tensor.flip(-1)))
        edges_tensor = edges_tensor.T.contiguous()
    else:
        edges_tensor = T.empty(2, 0)

    graph = Data(x=node_embeddings_tensor, edge_index=edges_tensor)

    return graph

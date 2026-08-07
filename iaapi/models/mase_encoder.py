"""
MASE (Mechanism-Aware Structure Encoder) - Module A.

This module encodes SBML reaction networks as heterogeneous graphs
using edge-type-aware message passing and graph-level attention pooling.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch, HeteroData
except ImportError as exc:
    raise ImportError("PyTorch and torch-geometric are required") from exc


NODE_TYPES = ("species", "reaction", "parameter", "compartment", "observable")
EDGE_TYPES = (
    ("species", "reactant", "reaction"),
    ("reaction", "product", "species"),
    ("species", "modifier", "reaction"),
    ("parameter", "used_in", "reaction"),
    ("species", "observed_by", "observable"),
    ("parameter", "used_in_observable", "observable"),
    ("species", "in_compartment", "compartment"),
)


class MASEEncoder(nn.Module):
    """
    Mechanism-Aware Structure Encoder (Module A).

    Encodes SBML/PEtab heterogeneous graphs into model and parameter embeddings.
    The forward pass always returns batched tensors:
        - z_M: (batch, d_embed)
        - z_theta: (batch, max_n_params, d_embed)
    """

    def __init__(
        self,
        d_embed: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_model: int = 256,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_embed = d_embed
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.node_types = NODE_TYPES

        self.type_embeddings = nn.ParameterDict(
            {node_type: nn.Parameter(torch.randn(d_model) * 0.02) for node_type in NODE_TYPES}
        )
        self.scalar_feature_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_norm = nn.ModuleDict({node_type: nn.LayerNorm(d_model) for node_type in NODE_TYPES})

        self.gnn_layers = nn.ModuleList(
            [EdgeTypeAwareHeteroLayer(d_model, d_ff, dropout) for _ in range(n_layers)]
        )

        self.graph_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.readout = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Linear(d_model, d_embed)
        self.parameter_proj = nn.Linear(d_model, d_embed)

    def forward(self, graph_data: HeteroData | Batch | List[HeteroData]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            graph_data: Single HeteroData, batched HeteroData/Batch, or list of HeteroData.

        Returns:
            z_M: Model embeddings (batch, d_embed)
            z_theta: Padded parameter embeddings (batch, max_n_params, d_embed)
        """
        if graph_data is None:
            raise ValueError("graph_data must not be None")
        if isinstance(graph_data, list):
            graph_data = Batch.from_data_list(graph_data)

        device = self.graph_query.device
        x_dict, batch_dict, batch_size = self._initial_node_states(graph_data, device)
        for layer in self.gnn_layers:
            x_dict = layer(x_dict, graph_data, device)

        z_M = self._pool_model_embedding(x_dict, batch_dict, batch_size)
        z_theta = self._pad_parameter_embeddings(x_dict, batch_dict, batch_size)
        return z_M, z_theta

    def _initial_node_states(
        self,
        graph_data: HeteroData,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
        """Build initial node states and batch vectors for every supported node type."""
        batch_size = self._infer_batch_size(graph_data)
        degrees = self._compute_degrees(graph_data, device)
        x_dict: Dict[str, torch.Tensor] = {}
        batch_dict: Dict[str, torch.Tensor] = {}

        for node_type in self.node_types:
            n_nodes = self._num_nodes(graph_data, node_type)
            if n_nodes == 0:
                x_dict[node_type] = torch.zeros((0, self.d_model), device=device)
                batch_dict[node_type] = torch.zeros((0,), dtype=torch.long, device=device)
                continue

            batch_vec = self._node_batch(graph_data, node_type, n_nodes, device)
            local_index = self._local_index(batch_vec, batch_size).to(device)
            counts = torch.bincount(batch_vec, minlength=batch_size).float().clamp_min(1.0).to(device)
            count_feature = torch.log1p(counts[batch_vec]).unsqueeze(-1)
            index_feature = local_index.float().unsqueeze(-1) / counts[batch_vec].unsqueeze(-1)
            in_degree = torch.log1p(degrees[node_type]["in"]).unsqueeze(-1)
            out_degree = torch.log1p(degrees[node_type]["out"]).unsqueeze(-1)
            scalar_features = torch.cat([index_feature, count_feature, in_degree, out_degree], dim=-1)

            type_embedding = self.type_embeddings[node_type].to(device).unsqueeze(0).expand(n_nodes, -1)
            x = type_embedding + self.scalar_feature_encoder(scalar_features)
            x_dict[node_type] = self.input_norm[node_type](x)
            batch_dict[node_type] = batch_vec

        return x_dict, batch_dict, batch_size

    def _infer_batch_size(self, graph_data: HeteroData) -> int:
        """Infer number of graphs in a HeteroData/Batch object."""
        num_graphs = getattr(graph_data, "num_graphs", None)
        if num_graphs is not None:
            return int(num_graphs)
        max_batch = -1
        for node_type in graph_data.node_types:
            store = graph_data[node_type]
            if hasattr(store, "batch") and store.batch.numel() > 0:
                max_batch = max(max_batch, int(store.batch.max().item()))
        return max_batch + 1 if max_batch >= 0 else 1

    def _num_nodes(self, graph_data: HeteroData, node_type: str) -> int:
        if node_type not in graph_data.node_types:
            return 0
        return int(graph_data[node_type].num_nodes or 0)

    def _node_batch(
        self,
        graph_data: HeteroData,
        node_type: str,
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        store = graph_data[node_type]
        if hasattr(store, "batch"):
            return store.batch.to(device=device, dtype=torch.long)
        return torch.zeros(n_nodes, dtype=torch.long, device=device)

    def _local_index(self, batch_vec: torch.Tensor, batch_size: int) -> torch.Tensor:
        local_index = torch.zeros_like(batch_vec)
        for graph_idx in range(batch_size):
            mask = batch_vec == graph_idx
            local_index[mask] = torch.arange(int(mask.sum().item()), device=batch_vec.device)
        return local_index

    def _compute_degrees(self, graph_data: HeteroData, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
        degrees = {
            node_type: {
                "in": torch.zeros(self._num_nodes(graph_data, node_type), device=device),
                "out": torch.zeros(self._num_nodes(graph_data, node_type), device=device),
            }
            for node_type in self.node_types
        }
        for edge_type in graph_data.edge_types:
            src_type, _, dst_type = edge_type
            if src_type not in degrees or dst_type not in degrees:
                continue
            edge_index = graph_data[edge_type].edge_index.to(device)
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            degrees[src_type]["out"].index_add_(0, src, torch.ones_like(src, dtype=torch.float))
            degrees[dst_type]["in"].index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        return degrees

    def _pool_model_embedding(
        self,
        x_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        node_tensors = []
        batch_tensors = []
        for node_type in self.node_types:
            if x_dict[node_type].numel() == 0:
                continue
            node_tensors.append(x_dict[node_type])
            batch_tensors.append(batch_dict[node_type])
        if not node_tensors:
            pooled = torch.zeros((batch_size, self.d_model), device=self.graph_query.device)
            return self.output_proj(pooled)

        all_nodes = torch.cat(node_tensors, dim=0)
        all_batch = torch.cat(batch_tensors, dim=0)
        padded, key_padding_mask = self._pad_nodes_by_batch(all_nodes, all_batch, batch_size)
        query = self.graph_query.expand(batch_size, -1, -1)
        readout, _ = self.readout(query, padded, padded, key_padding_mask=key_padding_mask, need_weights=False)
        return self.output_proj(readout.squeeze(1))

    def _pad_nodes_by_batch(
        self,
        nodes: torch.Tensor,
        batch_vec: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        counts = torch.bincount(batch_vec, minlength=batch_size)
        max_count = int(counts.max().item()) if counts.numel() else 0
        max_count = max(max_count, 1)
        padded = torch.zeros((batch_size, max_count, nodes.shape[-1]), device=nodes.device, dtype=nodes.dtype)
        key_padding_mask = torch.ones((batch_size, max_count), device=nodes.device, dtype=torch.bool)
        for graph_idx in range(batch_size):
            graph_nodes = nodes[batch_vec == graph_idx]
            n_nodes = graph_nodes.shape[0]
            if n_nodes > 0:
                padded[graph_idx, :n_nodes] = graph_nodes
                key_padding_mask[graph_idx, :n_nodes] = False
        return padded, key_padding_mask

    def _pad_parameter_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        param_features = x_dict.get("parameter")
        if param_features is None or param_features.numel() == 0:
            return torch.zeros((batch_size, 0, self.d_embed), device=self.graph_query.device)
        param_embeddings = self.parameter_proj(param_features)
        param_batch = batch_dict["parameter"]
        counts = torch.bincount(param_batch, minlength=batch_size)
        max_params = int(counts.max().item()) if counts.numel() else 0
        output = torch.zeros(
            (batch_size, max_params, self.d_embed),
            device=param_embeddings.device,
            dtype=param_embeddings.dtype,
        )
        for graph_idx in range(batch_size):
            graph_params = param_embeddings[param_batch == graph_idx]
            if graph_params.numel() > 0:
                output[graph_idx, : graph_params.shape[0]] = graph_params
        return output


class EdgeTypeAwareHeteroLayer(nn.Module):
    """Edge-type-aware heterogeneous message passing layer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.relation_linears = nn.ModuleDict()
        for src_type, relation, dst_type in EDGE_TYPES:
            self.relation_linears[self._relation_key(src_type, relation, dst_type)] = nn.Linear(d_model, d_model)
            self.relation_linears[self._relation_key(dst_type, f"rev_{relation}", src_type)] = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.message_norm = nn.ModuleDict({node_type: nn.LayerNorm(d_model) for node_type in NODE_TYPES})
        self.ff_norm = nn.ModuleDict({node_type: nn.LayerNorm(d_model) for node_type in NODE_TYPES})
        self.ff = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                )
                for node_type in NODE_TYPES
            }
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        graph_data: HeteroData,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        aggregated = {node_type: torch.zeros_like(x) for node_type, x in x_dict.items()}
        counts = {
            node_type: torch.zeros((x.shape[0], 1), device=device, dtype=x.dtype)
            for node_type, x in x_dict.items()
        }

        for edge_type in graph_data.edge_types:
            src_type, relation, dst_type = edge_type
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            edge_index = graph_data[edge_type].edge_index.to(device)
            if edge_index.numel() == 0:
                continue
            edge_weight = self._edge_weight(graph_data[edge_type], edge_index.shape[1], device, x_dict[src_type].dtype)
            self._aggregate_relation(
                x_dict,
                aggregated,
                counts,
                src_type,
                dst_type,
                relation,
                edge_index,
                edge_weight,
                reverse=False,
            )
            self._aggregate_relation(
                x_dict,
                aggregated,
                counts,
                src_type,
                dst_type,
                relation,
                edge_index,
                edge_weight,
                reverse=True,
            )

        out: Dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            if x.numel() == 0:
                out[node_type] = x
                continue
            normalized_messages = aggregated[node_type] / counts[node_type].clamp_min(1.0)
            h = self.message_norm[node_type](x + self.dropout(normalized_messages))
            h = self.ff_norm[node_type](h + self.dropout(self.ff[node_type](h)))
            out[node_type] = h
        return out

    def _aggregate_relation(
        self,
        x_dict: Dict[str, torch.Tensor],
        aggregated: Dict[str, torch.Tensor],
        counts: Dict[str, torch.Tensor],
        src_type: str,
        dst_type: str,
        relation: str,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        reverse: bool,
    ) -> None:
        if reverse:
            source_type, target_type = dst_type, src_type
            source_index, target_index = edge_index[1], edge_index[0]
            relation_key = self._relation_key(dst_type, f"rev_{relation}", src_type)
        else:
            source_type, target_type = src_type, dst_type
            source_index, target_index = edge_index[0], edge_index[1]
            relation_key = self._relation_key(src_type, relation, dst_type)

        if x_dict[source_type].numel() == 0 or x_dict[target_type].numel() == 0:
            return
        linear = self._relation_linear(relation_key)
        messages = linear(x_dict[source_type][source_index]) * edge_weight.unsqueeze(-1)
        aggregated[target_type].index_add_(0, target_index, messages)
        counts[target_type].index_add_(0, target_index, torch.ones((target_index.numel(), 1), device=messages.device, dtype=messages.dtype))

    def _edge_weight(self, edge_store, n_edges: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if hasattr(edge_store, "stoichiometry"):
            return edge_store.stoichiometry.to(device=device, dtype=dtype).reshape(-1).abs().clamp_min(1e-6)
        return torch.ones(n_edges, device=device, dtype=dtype)

    def _relation_linear(self, relation_key: str) -> nn.Linear:
        if relation_key not in self.relation_linears:
            self.relation_linears[relation_key] = nn.Linear(self.d_model, self.d_model)
        return self.relation_linears[relation_key]

    def _relation_key(self, src_type: str, relation: str, dst_type: str) -> str:
        return f"{src_type}__{relation}__{dst_type}"

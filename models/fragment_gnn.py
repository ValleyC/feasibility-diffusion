"""
Fragment GNN: edge-aware message-passing network over the fragment graph.

Predicts two scores per candidate oriented merge:
  - merge_score: how desirable is this merge?
  - risk_score: estimated infeasibility risk (0 = safe, 1 = infeasible)

The GNN operates on the fragment graph where:
  - Nodes = route fragments (with constraint summary features)
  - Edges = candidate oriented merges (k-NN on endpoint distances)
"""

import numpy as np
import torch
import torch.nn as nn


class EdgeGNNLayer(nn.Module):
    """Message-passing layer with edge feature integration."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_upd = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_node, h_edge, edge_index):
        """
        Args:
            h_node: (F, D) node embeddings
            h_edge: (E, D) edge embeddings
            edge_index: (E, 2) int — [src, dst]

        Returns:
            h_node: (F, D) updated node embeddings
            h_edge: (E, D) updated edge embeddings
        """
        F, D = h_node.shape

        if edge_index.shape[0] == 0:
            return h_node, h_edge

        src = edge_index[:, 0]  # (E,)
        dst = edge_index[:, 1]  # (E,)

        # Messages: src node + dst node + edge → message
        h_src = h_node[src]   # (E, D)
        h_dst = h_node[dst]   # (E, D)
        msg = self.msg(torch.cat([h_src, h_dst, h_edge], dim=-1))  # (E, D)

        # Aggregate messages to dst nodes
        agg = torch.zeros_like(h_node)  # (F, D)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, D), msg)

        # Update nodes
        h_node = self.node_norm(h_node + self.upd(torch.cat([h_node, agg], dim=-1)))

        # Update edges
        h_src_new = h_node[src]
        h_dst_new = h_node[dst]
        h_edge = self.edge_norm(
            h_edge + self.edge_upd(torch.cat([h_src_new, h_dst_new, h_edge], dim=-1))
        )

        return h_node, h_edge


class FragmentGNN(nn.Module):
    """Edge-aware GNN over fragment graph with merge and risk scoring heads.

    Outputs per candidate oriented merge:
      - merge_score: (E,) how desirable is this merge
      - risk_score: (E,) estimated infeasibility risk
    """

    def __init__(self, n_node_feat: int = 12, n_edge_feat: int = 3,
                 hidden_dim: int = 128, n_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.node_embed = nn.Sequential(
            nn.Linear(n_node_feat, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(n_edge_feat, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            EdgeGNNLayer(hidden_dim) for _ in range(n_layers)
        ])

        # Merge score head: src + dst + edge → scalar
        self.merge_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Risk score head: src + dst + edge → scalar in [0, 1]
        self.risk_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feat, edge_index, edge_feat):
        """
        Args:
            node_feat: (F, n_node_feat) fragment features
            edge_index: (E, 2) int — oriented merge candidates
            edge_feat: (E, n_edge_feat) edge features

        Returns:
            merge_scores: (E,) merge desirability
            risk_scores: (E,) infeasibility risk (after sigmoid)
        """
        h_node = self.node_embed(node_feat)      # (F, D)
        h_edge = self.edge_embed(edge_feat)      # (E, D)

        for layer in self.layers:
            h_node, h_edge = layer(h_node, h_edge, edge_index)

        if edge_index.shape[0] == 0:
            return torch.zeros(0, device=node_feat.device), \
                   torch.zeros(0, device=node_feat.device)

        src = edge_index[:, 0]
        dst = edge_index[:, 1]

        h_merge_in = torch.cat([h_node[src], h_node[dst], h_edge], dim=-1)

        merge_scores = self.merge_head(h_merge_in).squeeze(-1)     # (E,)
        risk_scores = torch.sigmoid(
            self.risk_head(h_merge_in).squeeze(-1)                 # (E,)
        )

        return merge_scores, risk_scores

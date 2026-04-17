"""
Move Scorer: neural network that scores candidate 2-opt moves.

Given a current tour, node coordinates, and noise level t, the model
outputs a score for each valid 2-opt move. During training, this is
supervised to predict the best-improving swap. During inference, the
highest-scored move is applied greedily.

Architecture:
  1. Node features: (x, y, tour_position/N, sin(2πt/T), cos(2πt/T))
  2. GNN on tour graph (predecessor/successor edges) for context
  3. For each candidate move (i,j), extract 4-node features at
     positions i, i+1, j, (j+1)%N and score via MLP

The tour graph gives each node its local routing context. The 4-node
MLP captures whether removing edges (i,i+1) and (j,j+1) and adding
edges (i,j) and (i+1,j+1) is beneficial — this is exactly the
information needed for 2-opt decision-making.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GNNLayer(nn.Module):
    """Simple message-passing layer on the tour graph.

    Each node aggregates messages from its predecessor and successor
    in the tour (degree-2 graph). This captures local routing context.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, tour):
        """
        h: (B, N, D) node features
        tour: (B, N) long — permutation giving visit order
        """
        B, N, D = h.shape

        # For each position p in the tour, its predecessor is tour[(p-1)%N]
        # and successor is tour[(p+1)%N].
        # We work in "position space": h_pos[b, p] = h[b, tour[b, p]]
        h_pos = h.gather(1, tour.unsqueeze(-1).expand(-1, -1, D))

        # Predecessor and successor features (in position space)
        h_prev = torch.roll(h_pos, 1, dims=1)   # h at position p-1
        h_next = torch.roll(h_pos, -1, dims=1)  # h at position p+1

        # Messages from both neighbors
        msg_prev = self.msg_mlp(torch.cat([h_pos, h_prev], dim=-1))
        msg_next = self.msg_mlp(torch.cat([h_pos, h_next], dim=-1))
        msg = msg_prev + msg_next

        # Update
        h_new_pos = self.update_mlp(torch.cat([h_pos, msg], dim=-1))
        h_new_pos = self.norm(h_pos + h_new_pos)  # residual

        # Scatter back to node space
        h_new = torch.zeros_like(h)
        h_new.scatter_(1, tour.unsqueeze(-1).expand(-1, -1, D), h_new_pos)

        return h_new


class MoveScorer(nn.Module):
    """Score all candidate 2-opt moves for a given tour.

    Args:
        n_node_features: input feature dim per node (default 5: x, y, pos, sin_t, cos_t)
        hidden_dim: GNN hidden dimension
        n_layers: number of GNN layers
    """

    def __init__(self, n_node_features=5, hidden_dim=64, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.node_embed = nn.Sequential(
            nn.Linear(n_node_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(n_layers)
        ])

        # Score a move from 4 node embeddings (positions i, i+1, j, j+1)
        self.move_scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coords, tour, t, t_max, moves_ij):
        """
        Args:
            coords: (B, N, 2) node coordinates
            tour: (B, N) long — current tour permutation
            t: (B,) or scalar — noise level (number of scrambling steps)
            t_max: scalar — maximum noise level (for normalization)
            moves_ij: (M, 2) long — list of (i, j) move indices, same for all B
                      M = N(N-3)/2

        Returns:
            scores: (B, M) logits, one per candidate move
        """
        B, N, _ = coords.shape
        device = coords.device
        M = moves_ij.shape[0]

        # Build per-node features
        # Tour position: for each node, where it sits in the tour
        # tour[b, p] = node_id → we want pos[b, node_id] = p / N
        pos = torch.zeros(B, N, device=device)
        positions = torch.arange(N, device=device).float() / N
        pos.scatter_(1, tour, positions.unsqueeze(0).expand(B, -1))

        # Time embedding (sinusoidal)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        t_norm = t.float() / t_max
        sin_t = torch.sin(2 * math.pi * t_norm).unsqueeze(-1).expand(-1, N)
        cos_t = torch.cos(2 * math.pi * t_norm).unsqueeze(-1).expand(-1, N)

        # (B, N, 5): x, y, tour_position, sin_t, cos_t
        node_feats = torch.cat([
            coords,
            pos.unsqueeze(-1),
            sin_t.unsqueeze(-1),
            cos_t.unsqueeze(-1),
        ], dim=-1)

        # Embed
        h = self.node_embed(node_feats)  # (B, N, D)

        # GNN message passing on tour graph
        for layer in self.gnn_layers:
            h = layer(h, tour)

        # Gather embeddings in tour-position space for scoring
        # h is in node space; we need position-space for move indexing
        D = self.hidden_dim
        h_pos = h.gather(1, tour.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)

        # For each move (i, j), extract features at positions i, i+1, j, (j+1)%N
        mi = moves_ij[:, 0]  # (M,)
        mj = moves_ij[:, 1]  # (M,)
        mi1 = (mi + 1) % N
        mj1 = (mj + 1) % N

        # Gather: (B, M, D) for each of the 4 positions
        def gather_pos(positions):
            idx = positions.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
            return h_pos.gather(1, idx)

        h_i = gather_pos(mi)    # (B, M, D)
        h_i1 = gather_pos(mi1)  # (B, M, D)
        h_j = gather_pos(mj)    # (B, M, D)
        h_j1 = gather_pos(mj1)  # (B, M, D)

        # Concat and score
        move_feats = torch.cat([h_i, h_i1, h_j, h_j1], dim=-1)  # (B, M, 4D)
        scores = self.move_scorer(move_feats).squeeze(-1)  # (B, M)

        return scores

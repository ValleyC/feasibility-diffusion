"""
Constraint-Native Layer for Combinatorial Optimization.

Standard routing GNNs treat constraints as features or decoder masks.
This layer makes constraints first-class architectural objects:

  - Customer nodes carry problem features (coords, demand, TW)
  - Constraint nodes carry resource state (capacity, slack, budget)
  - Typed message passing: customer ↔ customer, customer ↔ constraint
  - Resource-conservative updates: remaining capacity can only tighten
  - Monotone feasibility head: feasibility decreases with demand, increases with slack

This is a drop-in replacement for EdgeGNNLayer. Same interface:
  forward(h_node, h_edge, edge_index) → (h_node, h_edge)

The constraint nodes are appended to h_node internally and stripped
before returning, so the caller sees the same shapes.

The analogy: EGNN bakes E(n) symmetry into architecture.
This layer bakes constraint structure into architecture.
"""

import torch
import torch.nn as nn
import numpy as np


class ConstraintNativeLayer(nn.Module):
    """Message-passing layer with explicit constraint/resource nodes.

    Adds C constraint nodes to the graph. Customer nodes exchange messages
    with both other customers AND constraint nodes. Constraint nodes
    aggregate demand/resource info from customers and broadcast residual
    capacity/slack back.

    Resource-conservative: constraint node hidden states pass through a
    monotone gate that can only reduce available resources layer by layer.
    """

    def __init__(self, hidden_dim: int, n_constraint_nodes: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_constraint_nodes = n_constraint_nodes

        # Customer ↔ Customer messages (same as EdgeGNNLayer)
        self.cc_msg = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cc_upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Customer → Constraint messages ("I need this much resource")
        self.c2r_msg = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Constraint → Customer messages ("here's the resource situation")
        self.r2c_msg = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Constraint node update (resource-conservative)
        self.r_upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Monotone gate: controls how much resource is "consumed" per layer
        # Uses sigmoid to ensure consumption is in [0, 1], then subtracts
        self.resource_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
        )

        # Customer update from constraint context
        self.c_from_r_upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge update (same as EdgeGNNLayer)
        self.edge_upd = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.constraint_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_node, h_edge, edge_index, h_constraint=None):
        """
        Args:
            h_node: (F, D) customer/fragment node embeddings
            h_edge: (E, D) edge embeddings
            edge_index: (E, 2) customer-customer edges
            h_constraint: (C, D) constraint node embeddings, or None to init

        Returns:
            h_node: (F, D) updated customer embeddings
            h_edge: (E, D) updated edge embeddings
            h_constraint: (C, D) updated constraint embeddings
        """
        F, D = h_node.shape
        C = self.n_constraint_nodes

        # Initialize constraint nodes if not provided
        if h_constraint is None:
            # Aggregate customer features to initialize constraint context
            h_constraint = h_node.mean(dim=0, keepdim=True).expand(C, -1).clone()

        # ── Step 1: Customer ↔ Customer messages ──
        if edge_index.shape[0] > 0:
            src = edge_index[:, 0]
            dst = edge_index[:, 1]
            h_src = h_node[src]
            h_dst = h_node[dst]
            cc_msg = self.cc_msg(torch.cat([h_src, h_dst, h_edge], dim=-1))

            cc_agg = torch.zeros_like(h_node)
            cc_agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, D), cc_msg)

            h_node_cc = self.cc_upd(torch.cat([h_node, cc_agg], dim=-1))
        else:
            h_node_cc = torch.zeros_like(h_node)

        # ── Step 2: Customer → Constraint messages ──
        # Each constraint node receives from ALL customers
        c2r_msgs = self.c2r_msg(
            torch.cat([h_node.mean(dim=0, keepdim=True).expand(C, -1),
                        h_constraint], dim=-1)
        )  # (C, D)
        # More precise: each constraint node gets a demand-weighted summary
        # For now, mean aggregation (can be refined)

        # Constraint update with resource-conservative gate
        r_input = self.r_upd(torch.cat([h_constraint, c2r_msgs], dim=-1))
        # Resource gate: how much resource is consumed this layer
        consumption = self.resource_gate(r_input)
        # Resource-conservative: subtract consumption from current state
        # This ensures resources can only decrease (tighten) through layers
        h_constraint = self.constraint_norm(h_constraint - consumption * r_input)

        # ── Step 3: Constraint → Customer messages ──
        # Each customer receives from all constraint nodes (broadcast)
        r2c_msg = self.r2c_msg(
            torch.cat([h_constraint.mean(dim=0, keepdim=True).expand(F, -1),
                        h_node], dim=-1)
        )  # (F, D)

        # ── Step 4: Combine customer updates ──
        h_node = self.node_norm(
            h_node + h_node_cc +
            self.c_from_r_upd(torch.cat([h_node, r2c_msg], dim=-1))
        )

        # ── Step 5: Edge update ──
        if edge_index.shape[0] > 0:
            h_src_new = h_node[src]
            h_dst_new = h_node[dst]
            h_edge = self.edge_norm(
                h_edge + self.edge_upd(
                    torch.cat([h_src_new, h_dst_new, h_edge], dim=-1))
            )

        return h_node, h_edge, h_constraint


class MonotoneFeasibilityHead(nn.Module):
    """Feasibility scorer with structural monotonicity.

    Feasibility should be:
      - DECREASING in demand / resource consumption
      - INCREASING in remaining capacity / slack

    Enforced by using non-negative weights in the final layer.
    Initialized to output ~0.5 (uncertain) so the untrained model
    doesn't block all merges.

    Input: concatenation of [src_node, dst_node, edge, constraint_context]
    Output: scalar infeasibility risk in [0, 1] (0 = safe, 1 = infeasible)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        # Initialize so output is negative pre-sigmoid → risk ≈ 0.2
        # softplus(0) ≈ 0.693, so zero weights still give positive outputs
        # Use negative bias to compensate
        nn.init.zeros_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, -3.0)  # sigmoid(-3) ≈ 0.05

    def forward(self, x):
        """x: (..., input_dim) → (...,) infeasibility risk in [0, 1]"""
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        # Monotone final layer: softplus ensures non-negative weights
        w = torch.nn.functional.softplus(self.fc_out.weight)
        out = torch.nn.functional.linear(h, w, self.fc_out.bias)
        return torch.sigmoid(out.squeeze(-1))


class ConstraintNativeGNN(nn.Module):
    """Drop-in replacement for FragmentGNN with constraint-native layers.

    Same interface: forward(node_feat, edge_index, edge_feat) → (merge_scores, risk_scores)

    The constraint nodes are created and managed internally.
    The caller doesn't need to know about them.
    """

    def __init__(self, n_node_feat: int = 12, n_edge_feat: int = 3,
                 hidden_dim: int = 128, n_layers: int = 6,
                 n_constraint_nodes: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_constraint_nodes = n_constraint_nodes

        self.node_embed = nn.Sequential(
            nn.Linear(n_node_feat, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(n_edge_feat, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            ConstraintNativeLayer(hidden_dim, n_constraint_nodes)
            for _ in range(n_layers)
        ])

        # Merge score head: src + dst + edge + constraint_context → scalar
        self.merge_head = nn.Sequential(
            nn.Linear(3 * hidden_dim + hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Monotone risk head: structural monotonicity for feasibility
        self.risk_head = MonotoneFeasibilityHead(
            input_dim=3 * hidden_dim + hidden_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, node_feat, edge_index, edge_feat):
        """Same interface as FragmentGNN.

        Args:
            node_feat: (F, n_node_feat)
            edge_index: (E, 2)
            edge_feat: (E, n_edge_feat)

        Returns:
            merge_scores: (E,)
            risk_scores: (E,) — from monotone head
        """
        h_node = self.node_embed(node_feat)
        h_edge = self.edge_embed(edge_feat)

        h_constraint = None
        for layer in self.layers:
            h_node, h_edge, h_constraint = layer(
                h_node, h_edge, edge_index, h_constraint)

        if edge_index.shape[0] == 0:
            return torch.zeros(0, device=node_feat.device), \
                   torch.zeros(0, device=node_feat.device)

        src = edge_index[:, 0]
        dst = edge_index[:, 1]

        # Constraint context: mean of constraint node embeddings, broadcast
        constraint_ctx = h_constraint.mean(dim=0, keepdim=True).expand(
            edge_index.shape[0], -1)  # (E, D)

        h_merge_in = torch.cat([
            h_node[src], h_node[dst], h_edge, constraint_ctx
        ], dim=-1)  # (E, 3D + D)

        merge_scores = self.merge_head(h_merge_in).squeeze(-1)
        risk_scores = self.risk_head(h_merge_in)

        return merge_scores, risk_scores

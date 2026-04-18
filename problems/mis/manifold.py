"""
Maximum Independent Set (MIS) Manifold.

MIS: select the largest subset of nodes such that no two selected nodes
are adjacent in the graph.

State: selected (N,) bool.
Feasibility: no two selected nodes share an edge.
Moves:
  - Add: include an unselected node (if no neighbor is selected)
  - Remove: exclude a selected node (always feasible)
  - Swap: remove a selected node, add an unselected non-neighbor

Cost: -count(selected) (minimize = maximize independent set size).

Non-geometric problem. No equivariance. Demonstrates framework generality
beyond routing/geometric CO.
"""

import numpy as np
from typing import List
from enum import IntEnum
from core.manifold import FeasibilityManifold


class MISMoveType(IntEnum):
    ADD = 0
    REMOVE = 1
    SWAP = 2


class MISManifold(FeasibilityManifold):
    """Maximum Independent Set manifold.

    Instance: dict with 'adj' (N, N) bool adjacency matrix, 'n_nodes' int.
    """

    def _is_independent(self, selected, adj):
        """Check no two selected nodes are adjacent."""
        idx = np.where(selected)[0]
        if len(idx) <= 1:
            return True
        sub_adj = adj[np.ix_(idx, idx)]
        return not sub_adj.any()

    def _can_add(self, node, selected, adj):
        """Check if adding node preserves independence."""
        neighbors = np.where(adj[node])[0]
        return not selected[neighbors].any()

    def sample_random(self, instance):
        n = instance['n_nodes']
        adj = instance['adj']
        selected = np.zeros(n, dtype=bool)
        nodes = list(range(n))
        np.random.shuffle(nodes)
        for i in nodes:
            if self._can_add(i, selected, adj):
                selected[i] = True
        return selected

    def cost(self, solution, instance):
        return -int(solution.sum())  # maximize size

    def is_feasible(self, solution, instance):
        return self._is_independent(solution, instance['adj'])

    def enumerate_moves(self, solution, instance):
        moves = []
        adj = instance['adj']
        n = instance['n_nodes']

        included = [i for i in range(n) if solution[i]]
        excluded = [i for i in range(n) if not solution[i]]

        # Add: if no selected neighbor
        for node in excluded:
            if self._can_add(node, solution, adj):
                moves.append((MISMoveType.ADD, node, -1))

        # Remove: always feasible
        for node in included:
            moves.append((MISMoveType.REMOVE, node, -1))

        # Swap: remove a, add b (if b has no OTHER selected neighbor)
        for a in included:
            for b in excluded:
                test = solution.copy()
                test[a] = False
                if self._can_add(b, test, adj):
                    moves.append((MISMoveType.SWAP, a, b))

        return moves

    def apply_move(self, solution, move):
        new = solution.copy()
        mtype, a, b = move
        if mtype == MISMoveType.ADD:
            new[a] = True
        elif mtype == MISMoveType.REMOVE:
            new[a] = False
        elif mtype == MISMoveType.SWAP:
            new[a] = False
            new[b] = True
        return new

    def move_delta(self, solution, move, instance):
        mtype, a, b = move
        if mtype == MISMoveType.ADD:
            return -1  # size increases by 1
        elif mtype == MISMoveType.REMOVE:
            return 1   # size decreases by 1
        elif mtype == MISMoveType.SWAP:
            return 0   # size unchanged
        return 0

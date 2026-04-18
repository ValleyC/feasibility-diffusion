"""
Knapsack Problem (KP) Manifold.

KP: select items to maximize total value, subject to total weight ≤ capacity.

State: selected (N,) bool.
Moves: add (if weight fits), remove (always), swap (if weight fits).
Cost: -total_value (minimize = maximize value).

This is a pure selection problem — NO routing/TSP component.
Non-geometric (no coordinates involved).
"""

import numpy as np
from typing import List
from enum import IntEnum
from core.manifold import FeasibilityManifold


class KPMoveType(IntEnum):
    ADD = 0
    REMOVE = 1
    SWAP = 2


class KPManifold(FeasibilityManifold):
    """Knapsack manifold.

    Instance: dict with 'values' (N,), 'weights' (N,), 'capacity' float, 'n_items' int.
    """

    def sample_random(self, instance):
        n = instance['n_items']
        weights = instance['weights']
        capacity = instance['capacity']

        selected = np.zeros(n, dtype=bool)
        items = list(range(n))
        np.random.shuffle(items)
        load = 0.0
        for i in items:
            if load + weights[i] <= capacity:
                selected[i] = True
                load += weights[i]
        return selected

    def cost(self, solution, instance):
        return -float(instance['values'][solution].sum())

    def is_feasible(self, solution, instance):
        return instance['weights'][solution].sum() <= instance['capacity'] + 1e-8

    def enumerate_moves(self, solution, instance):
        moves = []
        weights = instance['weights']
        capacity = instance['capacity']
        current_weight = weights[solution].sum()
        n = instance['n_items']

        included = [i for i in range(n) if solution[i]]
        excluded = [i for i in range(n) if not solution[i]]

        for i in excluded:
            if current_weight + weights[i] <= capacity + 1e-8:
                moves.append((KPMoveType.ADD, i, -1))

        for i in included:
            moves.append((KPMoveType.REMOVE, i, -1))

        for a in included:
            for b in excluded:
                if current_weight - weights[a] + weights[b] <= capacity + 1e-8:
                    moves.append((KPMoveType.SWAP, a, b))

        return moves

    def apply_move(self, solution, move):
        new = solution.copy()
        mtype, a, b = move
        if mtype == KPMoveType.ADD:
            new[a] = True
        elif mtype == KPMoveType.REMOVE:
            new[a] = False
        elif mtype == KPMoveType.SWAP:
            new[a] = False
            new[b] = True
        return new

    def move_delta(self, solution, move, instance):
        vals = instance['values']
        mtype, a, b = move
        if mtype == KPMoveType.ADD:
            return -vals[a]
        elif mtype == KPMoveType.REMOVE:
            return vals[a]
        elif mtype == KPMoveType.SWAP:
            return vals[a] - vals[b]
        return 0.0

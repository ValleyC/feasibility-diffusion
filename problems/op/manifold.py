"""
Orienteering Problem (OP) Manifold.

OP: select a subset of nodes to visit, MAXIMIZE total prize,
subject to tour length ≤ budget. Depot is always included.

State: selected (N+1,) bool — which nodes are in the tour.
Feasibility: tour_cost(selected) ≤ budget AND depot selected.
Moves: add / remove / swap (all check budget after sub-TSP re-solve).

Differs from PCTSP:
  - PCTSP: minimize (tour_cost + penalties), subject to prize ≥ threshold
  - OP: maximize prize, subject to tour_cost ≤ budget
  For our manifold, we negate prize so cost = -prize + penalty_if_over_budget.
  Equivalently: cost = tour_length - lambda * prize (scalarized).
"""

import numpy as np
from typing import List
from enum import IntEnum

from core.manifold import FeasibilityManifold
from problems.pctsp.selection import _solve_sub_tsp_2opt


class OPMoveType(IntEnum):
    ADD = 0
    REMOVE = 1
    SWAP = 2


OPMove = tuple  # (type, node_a, node_b)


def _tour_length(selected, dist, depot=0):
    """Compute tour length over selected nodes (sub-TSP)."""
    nodes = [i for i in range(len(selected)) if selected[i]]
    if len(nodes) <= 1:
        return 0.0
    if len(nodes) == 2:
        c = [i for i in nodes if i != depot][0]
        return 2 * dist[depot, c]
    sub_dist = dist[np.ix_(nodes, nodes)]
    return _solve_sub_tsp_2opt(sub_dist)


class OPManifold(FeasibilityManifold):
    """Orienteering Problem manifold.

    Instance: dict with 'coords', 'prizes', 'budget', 'dist', 'n_customers'.
    Cost: -total_prize (minimize = maximize prize). Budget enforced via moves.
    """

    def sample_random(self, instance):
        n = instance['n_customers']
        dist = instance['dist']
        budget = instance['budget']
        prizes = instance['prizes']

        selected = np.zeros(n + 1, dtype=bool)
        selected[0] = True
        customers = list(range(1, n + 1))
        np.random.shuffle(customers)

        for c in customers:
            selected[c] = True
            if _tour_length(selected, dist) > budget:
                selected[c] = False
        return selected

    def cost(self, solution, instance):
        # Negative prize (we minimize, so more prize = lower cost)
        prize = instance['prizes'][solution].sum()
        return -float(prize)

    def is_feasible(self, solution, instance):
        if not solution[0]:
            return False
        tl = _tour_length(solution, instance['dist'])
        return tl <= instance['budget'] + 1e-8

    def enumerate_moves(self, solution, instance):
        moves = []
        dist = instance['dist']
        budget = instance['budget']
        n = len(solution)

        included = [i for i in range(1, n) if solution[i]]
        excluded = [i for i in range(1, n) if not solution[i]]

        # Add: include excluded node (if tour stays within budget)
        for node in excluded:
            new_sel = solution.copy()
            new_sel[node] = True
            if _tour_length(new_sel, dist) <= budget + 1e-8:
                moves.append((OPMoveType.ADD, node, -1))

        # Remove: always feasible (shorter tour, less prize)
        for node in included:
            moves.append((OPMoveType.REMOVE, node, -1))

        # Swap: replace included with excluded (if budget ok)
        for a in included:
            for b in excluded:
                new_sel = solution.copy()
                new_sel[a] = False
                new_sel[b] = True
                if _tour_length(new_sel, dist) <= budget + 1e-8:
                    moves.append((OPMoveType.SWAP, a, b))

        return moves

    def apply_move(self, solution, move):
        new = solution.copy()
        mtype, a, b = move
        if mtype == OPMoveType.ADD:
            new[a] = True
        elif mtype == OPMoveType.REMOVE:
            new[a] = False
        elif mtype == OPMoveType.SWAP:
            new[a] = False
            new[b] = True
        return new

    def move_delta(self, solution, move, instance):
        cost_before = self.cost(solution, instance)
        new = self.apply_move(solution, move)
        cost_after = self.cost(new, instance)
        return cost_after - cost_before

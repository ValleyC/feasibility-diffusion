"""
Stochastic PCTSP (SPCTSP) Manifold.

Same as PCTSP but prizes are stochastic: each node has an expected prize
and a realized prize (drawn at visit time). The selection must ensure
that the EXPECTED total prize >= threshold.

For the manifold, we use expected prizes for feasibility checking and
realized prizes for cost computation (simulating the stochastic nature).
In practice, we sample a realization during cost computation.

State: selected (N+1,) bool — same as PCTSP.
Feasibility: E[prize(selected)] >= min_prize.
Cost: tour_cost + penalties + prize_shortfall_penalty.
"""

import numpy as np
from typing import List
from core.manifold import FeasibilityManifold
from problems.pctsp.selection import (
    _solve_sub_tsp_2opt, apply_selection_move, SelMoveType,
)


class SPCTSPManifold(FeasibilityManifold):
    """Stochastic PCTSP manifold.

    Instance: dict with:
        'coords', 'expected_prizes', 'prize_stdev', 'penalties',
        'min_prize', 'dist', 'n_customers'

    Feasibility uses expected prizes. Cost uses a sampled realization
    (expectation over stochastic prizes).
    """

    def sample_random(self, instance):
        n = instance['n_customers']
        exp_prizes = instance['expected_prizes']
        min_prize = instance['min_prize']

        selected = np.ones(n + 1, dtype=bool)
        customers = list(range(1, n + 1))
        np.random.shuffle(customers)
        for c in customers:
            if exp_prizes[selected].sum() - exp_prizes[c] >= min_prize:
                if np.random.random() < 0.5:
                    selected[c] = False
        return selected

    def cost(self, solution, instance):
        dist = instance['dist']
        penalties = instance['penalties']
        exp_prizes = instance['expected_prizes']
        stdev = instance['prize_stdev']
        min_prize = instance['min_prize']

        # Tour cost
        nodes = [i for i in range(len(solution)) if solution[i]]
        if len(nodes) <= 1:
            tour_cost = 0.0
        elif len(nodes) == 2:
            c = [i for i in nodes if i != 0][0]
            tour_cost = 2 * dist[0, c]
        else:
            sub_dist = dist[np.ix_(nodes, nodes)]
            tour_cost = _solve_sub_tsp_2opt(sub_dist)

        # Penalty for unvisited
        penalty = sum(penalties[i] for i in range(1, len(solution)) if not solution[i])

        # Stochastic prize shortfall: E[max(0, min_prize - realized_prize)]
        # Approximate via expected prize and stdev
        total_exp = exp_prizes[solution].sum()
        total_std = np.sqrt((stdev[solution] ** 2).sum())
        # Penalty for risk of not meeting threshold (simplified)
        shortfall_risk = max(0, min_prize - total_exp + 0.5 * total_std)

        return tour_cost + penalty + shortfall_risk

    def is_feasible(self, solution, instance):
        if not solution[0]:
            return False
        total_exp = instance['expected_prizes'][solution].sum()
        return total_exp >= instance['min_prize'] - 1e-8

    def enumerate_moves(self, solution, instance):
        moves = []
        exp_prizes = instance['expected_prizes']
        min_prize = instance['min_prize']
        n = len(solution)
        current_prize = exp_prizes[solution].sum()

        included = [i for i in range(1, n) if solution[i]]
        excluded = [i for i in range(1, n) if not solution[i]]

        for node in excluded:
            moves.append((SelMoveType.ADD, node, -1))

        for node in included:
            if current_prize - exp_prizes[node] >= min_prize - 1e-8:
                moves.append((SelMoveType.REMOVE, node, -1))

        for a in included:
            for b in excluded:
                new_prize = current_prize - exp_prizes[a] + exp_prizes[b]
                if new_prize >= min_prize - 1e-8:
                    moves.append((SelMoveType.SWAP, a, b))

        return moves

    def apply_move(self, solution, move):
        return apply_selection_move(solution, move)

    def move_delta(self, solution, move, instance):
        cost_before = self.cost(solution, instance)
        new = self.apply_move(solution, move)
        return self.cost(new, instance) - cost_before

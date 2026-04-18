"""
Min-Max Multi-TSP (mTSP) Manifold.

mTSP: partition N customers into exactly K routes (one per agent),
minimize the MAXIMUM route length (balanced workload).

State: assign (N+1,) int — customer-to-agent assignment (same as CVRP).
Feasibility: every customer assigned exactly once (no capacity constraint).
Moves: relocate + swap (same as CVRP but always feasible since no capacity).
Cost: max over K routes of route_cost (min-max objective).

Note: no capacity constraint — the objective encourages balance instead.
All relocate/swap moves are always feasible (unlike CVRP).
"""

import numpy as np
from typing import List
from enum import IntEnum
from core.manifold import FeasibilityManifold
from problems.cvrp.partition import _solve_sub_tsp_2opt


class MTSPMoveType(IntEnum):
    RELOCATE = 0
    SWAP = 1


class MTSPManifold(FeasibilityManifold):
    """Min-max mTSP manifold.

    Instance: dict with 'coords' (N+1, 2), 'dist' (N+1, N+1),
              'n_customers' int, 'n_agents' int.
    """

    def sample_random(self, instance):
        n = instance['n_customers']
        K = instance['n_agents']
        assign = np.full(n + 1, -1, dtype=np.int64)
        customers = list(range(1, n + 1))
        np.random.shuffle(customers)
        for i, c in enumerate(customers):
            assign[c] = i % K
        return assign

    def _route_costs(self, assign, instance):
        """Compute per-route cost."""
        K = instance['n_agents']
        dist = instance['dist']
        costs = np.zeros(K)
        for k in range(K):
            customers = [i for i in range(len(assign)) if assign[i] == k]
            if len(customers) == 0:
                continue
            nodes = [0] + customers
            sub_dist = dist[np.ix_(nodes, nodes)]
            costs[k] = _solve_sub_tsp_2opt(sub_dist)
        return costs

    def cost(self, solution, instance):
        costs = self._route_costs(solution, instance)
        return float(costs.max())  # min-max objective

    def is_feasible(self, solution, instance):
        if solution[0] != -1:
            return False
        n = instance['n_customers']
        K = instance['n_agents']
        visited = set()
        for i in range(1, n + 1):
            if solution[i] < 0 or solution[i] >= K:
                return False
            visited.add(i)
        return visited == set(range(1, n + 1))

    def enumerate_moves(self, solution, instance):
        moves = []
        n = instance['n_customers']
        K = instance['n_agents']

        # Relocate: move customer c to agent k (always feasible, no capacity)
        for c in range(1, n + 1):
            k_src = solution[c]
            for k_dst in range(K):
                if k_dst != k_src:
                    moves.append((MTSPMoveType.RELOCATE, c, k_src, k_dst, -1))

        # Swap: exchange customer a and b (different agents)
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                if solution[a] != solution[b]:
                    moves.append((MTSPMoveType.SWAP, a, solution[a], b, solution[b]))

        return moves

    def apply_move(self, solution, move):
        new = solution.copy()
        mtype = move[0]
        if mtype == MTSPMoveType.RELOCATE:
            customer, _, k_dst = move[1], move[2], move[3]
            new[customer] = k_dst
        elif mtype == MTSPMoveType.SWAP:
            a, k_a, b, k_b = move[1], move[2], move[3], move[4]
            new[a] = k_b
            new[b] = k_a
        return new

    def move_delta(self, solution, move, instance):
        cost_before = self.cost(solution, instance)
        new = self.apply_move(solution, move)
        cost_after = self.cost(new, instance)
        return cost_after - cost_before

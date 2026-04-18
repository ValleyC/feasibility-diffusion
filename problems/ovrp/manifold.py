"""
Open VRP (OVRP) Manifold.

Same as CVRP but vehicles do NOT return to depot. Route cost is
depot → c1 → c2 → ... → ck (no return edge). This changes only
the cost computation, not the moves or feasibility.
"""

import numpy as np
from typing import List
from core.manifold import FeasibilityManifold
from problems.cvrp.partition import (
    random_partition, partition_feasible,
    enumerate_partition_moves, apply_partition_move,
    vehicle_loads, n_vehicles, get_vehicle_customers, PartMove,
    _solve_sub_tsp_2opt,
)


def _open_route_cost(customers, dist, depot=0):
    """Cost of an OPEN route: depot → c1 → ... → ck (no return)."""
    if len(customers) == 0:
        return 0.0
    # Solve sub-TSP to get good ordering, then drop the return edge
    nodes = [depot] + list(customers)
    n = len(nodes)
    if n == 2:
        return dist[depot, customers[0]]
    sub_dist = dist[np.ix_(nodes, nodes)]
    # Get closed tour cost and ordering via 2-opt
    # Then compute open cost (remove longest edge from depot)
    closed_cost = _solve_sub_tsp_2opt(sub_dist)
    # Approximate open cost: closed - longest edge touching depot in the tour
    # Simpler: just compute depot → NN → ... → last (no return)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        curr = tour[-1]
        best, bd = -1, float('inf')
        for j in range(n):
            if not visited[j] and sub_dist[curr, j] < bd:
                bd = sub_dist[curr, j]
                best = j
        tour.append(best)
        visited[best] = True
    # 2-opt on open tour
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 2, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[j + 1] if j + 1 < n else -1
                if d == -1:
                    old = sub_dist[a, b]
                    new = sub_dist[a, c]
                else:
                    old = sub_dist[a, b] + sub_dist[c, d]
                    new = sub_dist[a, c] + sub_dist[b, d]
                if new < old - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    # Open cost: sum of edges (no return)
    cost = sum(sub_dist[tour[k], tour[k + 1]] for k in range(n - 1))
    return cost


class OVRPManifold(FeasibilityManifold):
    """Open VRP: same partition moves as CVRP, open route cost."""

    def sample_random(self, instance):
        return random_partition(instance['n_customers'], instance['demands'],
                                instance['capacity'])

    def cost(self, solution, instance):
        K = n_vehicles(solution)
        total = 0.0
        for k in range(K):
            custs = get_vehicle_customers(solution, k)
            total += _open_route_cost(custs, instance['dist'])
        return total

    def is_feasible(self, solution, instance):
        return partition_feasible(solution, instance['demands'],
                                  instance['capacity'], instance['n_customers'])

    def enumerate_moves(self, solution, instance):
        return enumerate_partition_moves(solution, instance['demands'],
                                         instance['capacity'])

    def apply_move(self, solution, move):
        return apply_partition_move(solution, move)

    def move_delta(self, solution, move, instance):
        cost_before = self.cost(solution, instance)
        new = self.apply_move(solution, move)
        return self.cost(new, instance) - cost_before

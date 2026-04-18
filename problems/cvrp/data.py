"""
CVRP instance generation and solution-to-tensor conversion utilities.

Converts the list-of-routes CVRP solution into tensors suitable for
the GNN move scorer. The GNN operates on a graph where:
  - Nodes = depot (0) + customers (1..N)
  - Edges = consecutive nodes within each route (tour-structure edges)
  - Node features = (x, y, demand/capacity, route_id/K, position/route_len, is_depot)
"""

import numpy as np
from typing import List, Tuple
from problems.tsp.tour import dist_matrix_from_coords
from problems.cvrp.solution import (
    random_solution, solution_cost, enumerate_moves,
    apply_move, delta_move, is_feasible, MoveType,
)


CAPACITIES = {10: 20, 20: 30, 50: 40, 100: 50}


def generate_cvrp_instance(N: int, seed: int = 42) -> dict:
    """Generate a random CVRP instance."""
    np.random.seed(seed)
    coords = np.random.rand(N + 1, 2).astype(np.float32)
    demands = np.zeros(N + 1, dtype=np.float32)
    demands[1:] = np.random.randint(1, 10, N).astype(np.float32)
    capacity = float(CAPACITIES.get(N, 50))
    dist = dist_matrix_from_coords(coords)
    return {
        'coords': coords, 'demands': demands,
        'capacity': capacity, 'dist': dist, 'n_customers': N,
    }


def solution_to_edges(routes: List[List[int]], depot: int = 0) -> List[Tuple[int, int]]:
    """Convert CVRP solution to edge list for GNN.

    Each route depot→c1→c2→...→ck→depot produces edges:
    (depot, c1), (c1, c2), ..., (ck, depot)
    """
    edges = []
    for route in routes:
        if len(route) == 0:
            continue
        edges.append((depot, route[0]))
        for k in range(len(route) - 1):
            edges.append((route[k], route[k + 1]))
        edges.append((route[-1], depot))
    return edges


def solution_to_features(routes: List[List[int]], N: int,
                         demands: np.ndarray, capacity: float) -> dict:
    """Extract per-node features from a CVRP solution.

    Returns dict with:
        route_ids: (N+1,) int — which route each node belongs to (-1 for depot)
        positions: (N+1,) float — position in route / route_length (0-1)
        is_depot: (N+1,) float — 1.0 for depot, 0.0 for customers
    """
    K = len(routes)
    route_ids = np.full(N + 1, -1, dtype=np.int64)
    positions = np.zeros(N + 1, dtype=np.float32)
    is_depot = np.zeros(N + 1, dtype=np.float32)
    is_depot[0] = 1.0

    for ri, route in enumerate(routes):
        n = len(route)
        for pi, c in enumerate(route):
            route_ids[c] = ri
            positions[c] = pi / max(n - 1, 1)

    return {'route_ids': route_ids, 'positions': positions, 'is_depot': is_depot}


def move_to_4nodes(routes: List[List[int]], move, depot: int = 0) -> Tuple[int, int, int, int]:
    """Extract the 4 critical nodes involved in any move type.

    Every move changes exactly 2 edges (removes 2, adds 2). The 4 endpoints
    of the changed edges are the critical nodes for scoring.

    Returns (a, b, c, d) — the 4 node indices.
    """
    mtype, ri, pi, rj, pj = move
    r_i = routes[ri]

    if mtype == MoveType.INTRA_2OPT:
        # Reverse segment r_i[pi:pj+1]
        # Removed edges: (prev_of_pi, r_i[pi]) and (r_i[pj], next_of_pj)
        a = depot if pi == 0 else r_i[pi - 1]
        b = r_i[pi]
        c = r_i[pj]
        d = depot if pj == len(r_i) - 1 else r_i[pj + 1]

    elif mtype == MoveType.RELOCATE:
        # Move customer r_i[pi] from route ri to route rj at position pj
        cust = r_i[pi]
        # Nodes affected in source route (customer removed)
        prev_src = depot if pi == 0 else r_i[pi - 1]
        next_src = depot if pi == len(r_i) - 1 else r_i[pi + 1]
        # Node at insertion point in destination route
        r_j = routes[rj]
        prev_dst = depot if pj == 0 else r_j[pj - 1]
        a, b, c, d = prev_src, cust, next_src, prev_dst

    elif mtype == MoveType.SWAP:
        # Exchange r_i[pi] and r_j[pj]
        r_j = routes[rj]
        cust_a = r_i[pi]
        cust_b = r_j[pj]
        # Neighbors (for context)
        prev_a = depot if pi == 0 else r_i[pi - 1]
        prev_b = depot if pj == 0 else r_j[pj - 1]
        a, b, c, d = cust_a, prev_a, cust_b, prev_b

    else:
        a, b, c, d = depot, depot, depot, depot

    return a, b, c, d


def greedy_improve(routes: List[List[int]], instance: dict,
                   max_iters: int = 500) -> Tuple[List[List[int]], float, list]:
    """Run greedy local search (best-improving move) to convergence.

    Returns:
        final_routes: locally optimal solution
        final_cost: cost of final solution
        trajectory: list of (routes, best_move_index) at each step
    """
    dist = instance['dist']
    demands = instance['demands']
    capacity = instance['capacity']

    trajectory = []
    for iteration in range(max_iters):
        moves = enumerate_moves(routes, demands, capacity)
        if len(moves) == 0:
            break

        deltas = np.array([delta_move(routes, m, dist) for m in moves])
        best_idx = int(np.argmin(deltas))

        if deltas[best_idx] >= -1e-10:
            break  # locally optimal

        trajectory.append((routes, moves, best_idx))
        routes = apply_move(routes, moves[best_idx])

    final_cost = solution_cost(routes, dist)
    return routes, final_cost, trajectory

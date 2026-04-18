"""
CVRP Partition Representation and Feasibility-Preserving Moves.

A CVRP partition is an assignment of customers to vehicles (routes),
where each vehicle's total demand does not exceed capacity Q. The
ordering of customers within each route is NOT part of the state —
that's handled by a separate sub-TSP solver.

Representation:
  assign: np.ndarray of shape (N+1,) int
    assign[0] = -1 (depot, not assigned to any vehicle)
    assign[i] = k means customer i is assigned to vehicle k

Feasibility-preserving moves:
  1. Relocate: move customer c from vehicle k1 to vehicle k2
     Valid if demand(k2) + demand(c) <= capacity
  2. Swap: exchange customer a (vehicle k1) with customer b (vehicle k2)
     Valid if both resulting vehicle demands <= capacity

Both moves preserve: (a) each customer assigned exactly once, (b) demand <= Q.

The sub-TSP cost of each vehicle's route is computed externally (OR-Tools,
2-opt, or any TSP solver). This is the modular separation:
  - Partition diffusion learns WHICH customers go together
  - Sub-TSP solver determines the ORDERING within each route
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import IntEnum


class PartMoveType(IntEnum):
    RELOCATE = 0   # move customer from one vehicle to another
    SWAP = 1       # exchange customers between vehicles


# Move descriptor: (type, customer_a, vehicle_src, customer_b_or_vehicle_dst)
# - RELOCATE: (0, customer, src_vehicle, dst_vehicle, -1)
# - SWAP:     (1, customer_a, vehicle_a, customer_b, vehicle_b)
PartMove = Tuple[int, int, int, int, int]


def random_partition(n_customers: int, demands: np.ndarray,
                     capacity: float) -> np.ndarray:
    """Generate a random feasible partition.

    Greedily assigns randomly shuffled customers to vehicles,
    opening a new vehicle when capacity is exceeded.
    """
    assign = np.full(n_customers + 1, -1, dtype=np.int64)
    customers = list(range(1, n_customers + 1))
    np.random.shuffle(customers)

    vehicle = 0
    load = 0.0
    for c in customers:
        if load + demands[c] > capacity:
            vehicle += 1
            load = 0.0
        assign[c] = vehicle
        load += demands[c]

    return assign


def partition_feasible(assign: np.ndarray, demands: np.ndarray,
                       capacity: float, n_customers: int) -> bool:
    """Check partition feasibility."""
    if assign[0] != -1:
        return False
    K = assign[1:].max() + 1
    for k in range(K):
        mask = (assign == k)
        if demands[mask].sum() > capacity + 1e-8:
            return False
    visited = set(np.where(assign >= 0)[0].tolist())
    expected = set(range(1, n_customers + 1))
    return visited == expected


def vehicle_loads(assign: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """Compute load per vehicle. Returns (K,) array."""
    K = assign[1:].max() + 1
    loads = np.zeros(K)
    for k in range(K):
        loads[k] = demands[assign == k].sum()
    return loads


def get_vehicle_customers(assign: np.ndarray, k: int) -> List[int]:
    """Get list of customer indices assigned to vehicle k."""
    return [i for i in range(len(assign)) if assign[i] == k]


def n_vehicles(assign: np.ndarray) -> int:
    """Number of vehicles used."""
    return int(assign[1:].max()) + 1


def enumerate_partition_moves(assign: np.ndarray, demands: np.ndarray,
                              capacity: float) -> List[PartMove]:
    """Enumerate all feasibility-preserving partition moves.

    Returns list of (type, customer_a, vehicle_a, customer_b_or_dst, vehicle_b_or_-1).
    """
    moves = []
    K = n_vehicles(assign)
    loads = vehicle_loads(assign, demands)
    n = len(assign)

    # 1. Relocate: move customer c from vehicle k_src to vehicle k_dst
    for c in range(1, n):
        k_src = assign[c]
        d_c = demands[c]
        for k_dst in range(K):
            if k_dst == k_src:
                continue
            if loads[k_dst] + d_c <= capacity + 1e-8:
                moves.append((PartMoveType.RELOCATE, c, k_src, k_dst, -1))

    # 2. Swap: exchange customer a (vehicle k_a) with customer b (vehicle k_b)
    for a in range(1, n):
        k_a = assign[a]
        d_a = demands[a]
        for b in range(a + 1, n):
            k_b = assign[b]
            if k_a == k_b:
                continue  # same vehicle, no partition change
            d_b = demands[b]
            if (loads[k_a] - d_a + d_b <= capacity + 1e-8 and
                    loads[k_b] - d_b + d_a <= capacity + 1e-8):
                moves.append((PartMoveType.SWAP, a, k_a, b, k_b))

    return moves


def apply_partition_move(assign: np.ndarray, move: PartMove) -> np.ndarray:
    """Apply a partition move, returning a NEW assignment array."""
    new_assign = assign.copy()
    mtype = move[0]

    if mtype == PartMoveType.RELOCATE:
        customer, _, k_dst = move[1], move[2], move[3]
        new_assign[customer] = k_dst

    elif mtype == PartMoveType.SWAP:
        cust_a, k_a, cust_b, k_b = move[1], move[2], move[3], move[4]
        new_assign[cust_a] = k_b
        new_assign[cust_b] = k_a

    return new_assign


def partition_cost(assign: np.ndarray, coords: np.ndarray,
                   dist: np.ndarray, depot: int = 0,
                   sub_tsp_solver=None) -> float:
    """Compute total CVRP cost by solving sub-TSP per vehicle.

    If sub_tsp_solver is None, uses nearest-neighbor + 2-opt (self-contained).
    Otherwise, calls the provided solver function:
        sub_tsp_solver(coords_subset, dist_subset) -> tour_cost
    """
    K = n_vehicles(assign)
    total = 0.0

    for k in range(K):
        customers = get_vehicle_customers(assign, k)
        if len(customers) == 0:
            continue

        # Nodes in this sub-problem: depot + customers
        nodes = [depot] + customers
        n = len(nodes)

        if n <= 2:
            # Only depot + 1 customer: depot → customer → depot
            total += 2 * dist[depot, customers[0]]
            continue

        # Sub-distance matrix
        sub_dist = dist[np.ix_(nodes, nodes)]

        if sub_tsp_solver is not None:
            sub_cost = sub_tsp_solver(coords[nodes], sub_dist)
        else:
            sub_cost = _solve_sub_tsp_2opt(sub_dist)

        total += sub_cost

    return total


def _solve_sub_tsp_2opt(sub_dist: np.ndarray) -> float:
    """Solve small sub-TSP via nearest-neighbor + 2-opt. Depot is index 0."""
    n = sub_dist.shape[0]
    if n <= 2:
        return 2 * sub_dist[0, 1] if n == 2 else 0.0

    # Nearest-neighbor from depot
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        curr = tour[-1]
        best_next, best_d = -1, float('inf')
        for j in range(n):
            if not visited[j] and sub_dist[curr, j] < best_d:
                best_d = sub_dist[curr, j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True

    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 2, n):
                if i == 1 and j == n - 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (sub_dist[a, c] + sub_dist[b, d]
                         - sub_dist[a, b] - sub_dist[c, d])
                if delta < -1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break

    # Compute tour cost
    cost = sum(sub_dist[tour[k], tour[(k + 1) % n]] for k in range(n))
    return cost


def delta_partition_move(assign: np.ndarray, move: PartMove,
                         coords: np.ndarray, dist: np.ndarray,
                         depot: int = 0, sub_solver=None) -> float:
    """Compute cost change from a partition move.

    Since partition moves change which customers are in which vehicle,
    we need to re-solve sub-TSPs for the affected vehicles.
    Only the affected vehicles (at most 2) need re-solving.

    Returns delta = cost_after - cost_before. Negative = improvement.
    """
    mtype = move[0]

    if mtype == PartMoveType.RELOCATE:
        customer, k_src, k_dst = move[1], move[2], move[3]
        affected = [k_src, k_dst]
    elif mtype == PartMoveType.SWAP:
        _, k_a, _, k_b = move[1], move[2], move[3], move[4]
        affected = [k_a, k_b]
    else:
        return 0.0

    _solver = sub_solver if sub_solver is not None else _solve_sub_tsp_2opt

    # Cost of affected vehicles BEFORE move
    cost_before = 0.0
    for k in affected:
        customers = get_vehicle_customers(assign, k)
        if len(customers) == 0:
            continue
        nodes = [depot] + customers
        if sub_solver is not None:
            cost_before += sub_solver(coords[nodes])
        else:
            cost_before += _solve_sub_tsp_2opt(dist[np.ix_(nodes, nodes)])

    # Cost of affected vehicles AFTER move
    new_assign = apply_partition_move(assign, move)
    cost_after = 0.0
    for k in affected:
        customers = get_vehicle_customers(new_assign, k)
        if len(customers) == 0:
            continue
        nodes = [depot] + customers
        if sub_solver is not None:
            cost_after += sub_solver(coords[nodes])
        else:
            cost_after += _solve_sub_tsp_2opt(dist[np.ix_(nodes, nodes)])

    return cost_after - cost_before

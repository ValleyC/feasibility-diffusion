"""
CVRP Solution Representation and Feasibility-Preserving Moves.

A CVRP solution is a set of K routes, each visiting a subset of customers,
where every customer is visited exactly once and each route's total demand
does not exceed the vehicle capacity Q.

Representation:
  routes: List[List[int]] — each inner list is an ordered sequence of
          customer indices (0-indexed, depot is implicit at start/end).
  Example: [[1, 3, 7], [2, 5], [4, 6, 8]]
           = 3 routes: depot→1→3→7→depot, depot→2→5→depot, depot→4→6→8→depot

Feasibility-preserving moves:
  1. Intra-route 2-opt: reverse a segment within one route (always valid)
  2. Inter-route relocate: move customer from route i to route j (check capacity)
  3. Inter-route swap: exchange customers between routes (check both capacities)

All moves preserve: (a) each customer visited exactly once, (b) demand ≤ capacity.

Complexity:
  - enumerate all moves: O(N^2 + N*K)
  - apply any move: O(N)
  - delta computation: O(1) per move
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import IntEnum


class MoveType(IntEnum):
    INTRA_2OPT = 0    # reverse segment within a route
    RELOCATE = 1       # move customer from one route to another
    SWAP = 2           # exchange customers between two routes


# Move descriptor: (type, route_i, pos_i, route_j, pos_j)
# - INTRA_2OPT: (0, route, i, route, j) — reverse route[i:j+1]
# - RELOCATE:   (1, src_route, src_pos, dst_route, dst_pos)
# - SWAP:       (2, route_i, pos_i, route_j, pos_j)
Move = Tuple[int, int, int, int, int]


def route_demand(route: List[int], demands: np.ndarray) -> float:
    """Total demand of a route."""
    return sum(demands[c] for c in route)


def route_cost(route: List[int], dist: np.ndarray, depot: int = 0) -> float:
    """Cost of a single route: depot → route[0] → ... → route[-1] → depot."""
    if len(route) == 0:
        return 0.0
    cost = dist[depot, route[0]]
    for k in range(len(route) - 1):
        cost += dist[route[k], route[k + 1]]
    cost += dist[route[-1], depot]
    return cost


def solution_cost(routes: List[List[int]], dist: np.ndarray, depot: int = 0) -> float:
    """Total CVRP cost (sum of all route costs)."""
    return sum(route_cost(r, dist, depot) for r in routes)


def is_feasible(routes: List[List[int]], demands: np.ndarray,
                capacity: float, n_customers: int) -> bool:
    """Check CVRP feasibility: all customers visited exactly once, capacity ok."""
    visited = set()
    for route in routes:
        for c in route:
            if c in visited:
                return False  # duplicate
            visited.add(c)
        if route_demand(route, demands) > capacity + 1e-8:
            return False  # capacity violated
    # All customers visited (customers are 1..n_customers, depot is 0)
    expected = set(range(1, n_customers + 1))
    if visited != expected:
        return False
    return True


def random_solution(n_customers: int, demands: np.ndarray,
                    capacity: float) -> List[List[int]]:
    """Generate a random feasible CVRP solution.

    Randomly shuffles customers, greedily assigns to routes respecting capacity.
    """
    customers = list(range(1, n_customers + 1))
    np.random.shuffle(customers)
    routes = [[]]
    current_load = 0.0

    for c in customers:
        if current_load + demands[c] <= capacity:
            routes[-1].append(c)
            current_load += demands[c]
        else:
            routes.append([c])
            current_load = demands[c]

    return routes


def enumerate_moves(routes: List[List[int]], demands: np.ndarray,
                    capacity: float) -> List[Move]:
    """Enumerate all feasibility-preserving moves.

    Returns a list of Move descriptors. Each move, when applied, produces
    a valid CVRP solution.
    """
    moves = []
    K = len(routes)
    loads = [route_demand(r, demands) for r in routes]

    # 1. Intra-route 2-opt: for each route, all valid segment reversals
    for ri in range(K):
        n = len(routes[ri])
        if n < 3:
            continue
        for i in range(n - 1):
            for j in range(i + 2, n):
                moves.append((MoveType.INTRA_2OPT, ri, i, ri, j))

    # 2. Inter-route relocate: move customer from route ri to route rj
    for ri in range(K):
        for pi, cust in enumerate(routes[ri]):
            d_c = demands[cust]
            for rj in range(K):
                if rj == ri:
                    continue
                # Check capacity: can route rj absorb customer cust?
                if loads[rj] + d_c <= capacity + 1e-8:
                    # Insert at end of route rj (position = len)
                    moves.append((MoveType.RELOCATE, ri, pi, rj, len(routes[rj])))

    # 3. Inter-route swap: exchange customers between routes
    for ri in range(K):
        for pi, cust_a in enumerate(routes[ri]):
            for rj in range(ri + 1, K):
                for pj, cust_b in enumerate(routes[rj]):
                    d_a = demands[cust_a]
                    d_b = demands[cust_b]
                    # Check both capacities after swap
                    if (loads[ri] - d_a + d_b <= capacity + 1e-8 and
                            loads[rj] - d_b + d_a <= capacity + 1e-8):
                        moves.append((MoveType.SWAP, ri, pi, rj, pj))

    return moves


def apply_move(routes: List[List[int]], move: Move) -> List[List[int]]:
    """Apply a move, returning a NEW solution (does not modify input).

    INVARIANT: if input is feasible, output is feasible.
    """
    new_routes = [r.copy() for r in routes]
    mtype, ri, pi, rj, pj = move

    if mtype == MoveType.INTRA_2OPT:
        # Reverse segment routes[ri][pi:pj+1]
        new_routes[ri][pi:pj + 1] = new_routes[ri][pi:pj + 1][::-1]

    elif mtype == MoveType.RELOCATE:
        # Remove customer from route ri at position pi
        cust = new_routes[ri].pop(pi)
        # Insert into route rj at position pj
        # Adjust pj if same route (shouldn't happen but safety)
        new_routes[rj].insert(pj, cust)

    elif mtype == MoveType.SWAP:
        # Exchange customers at (ri, pi) and (rj, pj)
        new_routes[ri][pi], new_routes[rj][pj] = \
            new_routes[rj][pj], new_routes[ri][pi]

    return new_routes


def delta_move(routes: List[List[int]], move: Move,
               dist: np.ndarray, depot: int = 0) -> float:
    """Compute cost change from a move WITHOUT applying it. O(1).

    Returns delta = cost_after - cost_before. Negative = improvement.
    """
    mtype, ri, pi, rj, pj = move
    r_i = routes[ri]

    if mtype == MoveType.INTRA_2OPT:
        # Same as TSP 2-opt within route ri
        n = len(r_i)
        a = depot if pi == 0 else r_i[pi - 1]
        b = r_i[pi]
        c = r_i[pj]
        d = depot if pj == n - 1 else r_i[pj + 1]
        return (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

    elif mtype == MoveType.RELOCATE:
        r_j = routes[rj]
        cust = r_i[pi]
        n_i = len(r_i)
        n_j = len(r_j)

        # Cost of removing cust from route ri
        prev_i = depot if pi == 0 else r_i[pi - 1]
        next_i = depot if pi == n_i - 1 else r_i[pi + 1]
        remove_cost = dist[prev_i, next_i] - dist[prev_i, cust] - dist[cust, next_i]

        # Cost of inserting cust into route rj at position pj
        prev_j = depot if pj == 0 else r_j[pj - 1]
        next_j = depot if pj == n_j else r_j[pj]  # pj can be len(r_j)
        insert_cost = dist[prev_j, cust] + dist[cust, next_j] - dist[prev_j, next_j]

        return remove_cost + insert_cost

    elif mtype == MoveType.SWAP:
        r_j = routes[rj]
        cust_a = r_i[pi]
        cust_b = r_j[pj]
        n_i = len(r_i)
        n_j = len(r_j)

        # Cost change in route ri: replace cust_a with cust_b
        prev_a = depot if pi == 0 else r_i[pi - 1]
        next_a = depot if pi == n_i - 1 else r_i[pi + 1]
        delta_i = (dist[prev_a, cust_b] + dist[cust_b, next_a]
                   - dist[prev_a, cust_a] - dist[cust_a, next_a])

        # Cost change in route rj: replace cust_b with cust_a
        prev_b = depot if pj == 0 else r_j[pj - 1]
        next_b = depot if pj == n_j - 1 else r_j[pj + 1]
        delta_j = (dist[prev_b, cust_a] + dist[cust_a, next_b]
                   - dist[prev_b, cust_b] - dist[cust_b, next_b])

        return delta_i + delta_j

    return 0.0

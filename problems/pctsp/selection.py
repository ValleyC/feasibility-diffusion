"""
PCTSP Selection Representation and Feasibility-Preserving Moves.

A PCTSP selection is a subset of nodes to visit, where the total collected
prize meets a minimum threshold P. The routing over the selected nodes is
handled by a sub-TSP solver (modular, same as CVRP).

Representation:
  selected: np.ndarray of shape (N+1,) bool
    selected[0] = True (depot always visited)
    selected[i] = True means customer i is included in the tour

Feasibility constraint:
  sum(prizes[selected]) >= min_prize

Feasibility-preserving moves:
  1. Add: include a node (always feasible — prize only increases)
  2. Remove: exclude a node (feasible IF remaining prize >= min_prize)
  3. Swap: replace node a with node b (feasible IF resulting prize >= min_prize)

Cost = tour_cost(selected_nodes) + sum(penalties[unselected_nodes])

Connectivity: add + remove + swap ensure any feasible selection is reachable
from any other.
"""

import numpy as np
from typing import List, Tuple
from enum import IntEnum


class SelMoveType(IntEnum):
    ADD = 0      # include a currently-excluded node
    REMOVE = 1   # exclude a currently-included node
    SWAP = 2     # replace included node with excluded node


SelMove = Tuple[int, int, int]  # (type, node_a, node_b) — node_b=-1 for add/remove


def random_selection(n_customers: int, prizes: np.ndarray,
                     min_prize: float) -> np.ndarray:
    """Generate a random feasible selection.

    Start with all nodes selected (trivially feasible since total prize is max),
    then randomly remove nodes one at a time while keeping prize >= min_prize.
    """
    selected = np.ones(n_customers + 1, dtype=bool)  # depot + all customers
    customers = list(range(1, n_customers + 1))
    np.random.shuffle(customers)

    for c in customers:
        # Try removing this customer
        if prize_total(selected, prizes) - prizes[c] >= min_prize:
            if np.random.random() < 0.5:  # randomly decide to remove
                selected[c] = False

    return selected


def prize_total(selected: np.ndarray, prizes: np.ndarray) -> float:
    """Total prize collected by selected nodes."""
    return float(prizes[selected].sum())


def selection_feasible(selected: np.ndarray, prizes: np.ndarray,
                       min_prize: float, n_customers: int) -> bool:
    """Check selection feasibility: depot selected AND prize >= min_prize."""
    if not selected[0]:
        return False
    return prize_total(selected, prizes) >= min_prize - 1e-8


def enumerate_selection_moves(selected: np.ndarray, prizes: np.ndarray,
                              min_prize: float) -> List[SelMove]:
    """Enumerate all feasibility-preserving selection moves."""
    moves = []
    n = len(selected)
    current_prize = prize_total(selected, prizes)

    included = [i for i in range(1, n) if selected[i]]
    excluded = [i for i in range(1, n) if not selected[i]]

    # 1. Add: include any excluded node (always feasible — prize increases)
    for node in excluded:
        moves.append((SelMoveType.ADD, node, -1))

    # 2. Remove: exclude an included node (if remaining prize >= min_prize)
    for node in included:
        if current_prize - prizes[node] >= min_prize - 1e-8:
            moves.append((SelMoveType.REMOVE, node, -1))

    # 3. Swap: replace included node a with excluded node b
    for a in included:
        for b in excluded:
            new_prize = current_prize - prizes[a] + prizes[b]
            if new_prize >= min_prize - 1e-8:
                moves.append((SelMoveType.SWAP, a, b))

    return moves


def apply_selection_move(selected: np.ndarray, move: SelMove) -> np.ndarray:
    """Apply a selection move, returning a NEW array."""
    new_sel = selected.copy()
    mtype, node_a, node_b = move

    if mtype == SelMoveType.ADD:
        new_sel[node_a] = True
    elif mtype == SelMoveType.REMOVE:
        new_sel[node_a] = False
    elif mtype == SelMoveType.SWAP:
        new_sel[node_a] = False  # remove a
        new_sel[node_b] = True   # add b

    return new_sel


def selection_cost(selected: np.ndarray, coords: np.ndarray,
                   dist: np.ndarray, penalties: np.ndarray,
                   depot: int = 0) -> float:
    """PCTSP cost = tour_cost(selected) + penalty_cost(unselected).

    Tour cost is computed by solving sub-TSP over selected nodes.
    Penalty cost = sum of penalties for unselected customers.
    """
    nodes = [i for i in range(len(selected)) if selected[i]]
    n = len(nodes)

    # Penalty for unselected
    penalty = sum(penalties[i] for i in range(1, len(selected)) if not selected[i])

    if n <= 1:
        return penalty  # only depot, no tour

    if n == 2:
        # depot + 1 customer
        c = [i for i in nodes if i != depot][0]
        return 2 * dist[depot, c] + penalty

    # Sub-TSP over selected nodes
    sub_dist = dist[np.ix_(nodes, nodes)]
    tour_cost = _solve_sub_tsp_2opt(sub_dist)

    return tour_cost + penalty


def delta_selection_move(selected: np.ndarray, move: SelMove,
                         coords: np.ndarray, dist: np.ndarray,
                         penalties: np.ndarray, depot: int = 0) -> float:
    """Cost change from a selection move.

    Re-solves sub-TSP for the changed selection (exact delta).
    """
    cost_before = selection_cost(selected, coords, dist, penalties, depot)
    new_sel = apply_selection_move(selected, move)
    cost_after = selection_cost(new_sel, coords, dist, penalties, depot)
    return cost_after - cost_before


def _solve_sub_tsp_2opt(sub_dist: np.ndarray) -> float:
    """Solve small sub-TSP via nearest-neighbor + 2-opt. Depot is index 0."""
    n = sub_dist.shape[0]
    if n <= 2:
        return 2 * sub_dist[0, 1] if n == 2 else 0.0

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

    return sum(sub_dist[tour[k], tour[(k + 1) % n]] for k in range(n))

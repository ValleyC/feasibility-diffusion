"""
TSP Tour Representation and 2-opt Operations.

A tour is a permutation of {0, 1, ..., N-1} representing the visit order of
a closed Hamiltonian cycle: tour[0] → tour[1] → ... → tour[N-1] → tour[0].

2-opt neighborhood:
  A 2-opt move removes two non-adjacent edges from the tour and reconnects
  the two resulting paths. Equivalently, it reverses a contiguous segment
  of the tour.

  Given edge indices e1 < e2 (where edge e connects tour[e] to tour[(e+1)%N]),
  the 2-opt move:
    - Removes edges (tour[e1], tour[e1+1]) and (tour[e2], tour[(e2+1)%N])
    - Reverses the segment tour[e1+1 : e2+1]
    - Adds edges (tour[e1], tour[e2]) and (tour[e1+1], tour[(e2+1)%N])

  Validity: e2 - e1 >= 2, and not (e1=0, e2=N-1) which are adjacent via wrap.
  Count: N(N-3)/2 valid moves for an N-node tour.

Complexity:
  - apply_2opt: O(N) worst case (segment reversal)
  - delta_2opt: O(1) (only 4 distance lookups)
  - enumerate_2opt: O(N^2)
  - tour_cost: O(N)
"""

import numpy as np
from typing import List, Tuple


def tour_cost(tour: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Compute the total cost of a closed tour.

    Args:
        tour: (N,) permutation of node indices
        dist_matrix: (N, N) pairwise distance matrix

    Returns:
        Sum of edge weights along the closed tour.
    """
    N = len(tour)
    cost = 0.0
    for k in range(N):
        cost += dist_matrix[tour[k], tour[(k + 1) % N]]
    return cost


def tour_cost_coords(tour: np.ndarray, coords: np.ndarray) -> float:
    """Compute tour cost from 2D coordinates (Euclidean TSP).

    Args:
        tour: (N,) permutation
        coords: (N, 2) node coordinates
    """
    ordered = coords[tour]
    diffs = np.roll(ordered, -1, axis=0) - ordered
    return float(np.linalg.norm(diffs, axis=1).sum())


def dist_matrix_from_coords(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix.

    Args:
        coords: (N, 2)
    Returns:
        (N, N) symmetric distance matrix
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def enumerate_2opt(N: int) -> List[Tuple[int, int]]:
    """Enumerate all valid 2-opt moves for an N-node closed tour.

    A move (i, j) means: reverse segment tour[i+1 : j+1].
    This removes edges (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]),
    and adds edges (tour[i], tour[j]) and (tour[i+1], tour[(j+1)%N]).

    Constraints:
      - 0 <= i < j <= N-1
      - j - i >= 2  (don't remove the same edge twice)
      - NOT (i == 0 and j == N-1)  (these edges are adjacent via wrap-around)

    Returns:
        List of (i, j) pairs. Length = N(N-3)/2.
    """
    moves = []
    for i in range(N):
        for j in range(i + 2, N):
            if i == 0 and j == N - 1:
                continue
            moves.append((i, j))
    assert len(moves) == N * (N - 3) // 2, \
        f"Expected {N * (N - 3) // 2} moves, got {len(moves)}"
    return moves


def apply_2opt(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Apply a 2-opt move: reverse segment tour[i+1 : j+1].

    Args:
        tour: (N,) permutation (not modified)
        i, j: move indices with 0 <= i < j <= N-1

    Returns:
        New tour with the segment reversed.

    INVARIANT: output is a valid permutation if input is a valid permutation.
    """
    new_tour = tour.copy()
    new_tour[i + 1: j + 1] = tour[i + 1: j + 1][::-1]
    return new_tour


def delta_2opt(tour: np.ndarray, i: int, j: int,
               dist_matrix: np.ndarray) -> float:
    """Compute cost change from a 2-opt move WITHOUT applying it.

    delta = cost_after - cost_before
    Negative delta means the move improves the tour.

    The 2-opt move (i, j) removes edges:
        (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N])
    and adds edges:
        (tour[i], tour[j]) and (tour[i+1], tour[(j+1)%N])

    Args:
        tour: (N,) permutation
        i, j: move indices
        dist_matrix: (N, N)

    Returns:
        Cost change (float). O(1) computation.
    """
    N = len(tour)
    a = tour[i]
    b = tour[i + 1]
    c = tour[j]
    d = tour[(j + 1) % N]
    return (dist_matrix[a, c] + dist_matrix[b, d]
            - dist_matrix[a, b] - dist_matrix[c, d])


def random_tour(N: int) -> np.ndarray:
    """Generate a uniformly random permutation (= random valid tour)."""
    return np.random.permutation(N)


def is_valid_tour(tour: np.ndarray) -> bool:
    """Check that tour is a valid permutation of {0, ..., N-1}."""
    N = len(tour)
    return (np.sort(tour) == np.arange(N)).all()


def greedy_nearest_neighbor(coords: np.ndarray, start: int = 0) -> np.ndarray:
    """Construct a tour via nearest-neighbor heuristic (for initialization).

    Args:
        coords: (N, 2) node coordinates
        start: starting node index

    Returns:
        (N,) tour as permutation
    """
    N = len(coords)
    visited = np.zeros(N, dtype=bool)
    tour = np.zeros(N, dtype=np.int64)
    tour[0] = start
    visited[start] = True
    for step in range(1, N):
        curr = tour[step - 1]
        dists = np.linalg.norm(coords - coords[curr], axis=1)
        dists[visited] = np.inf
        tour[step] = np.argmin(dists)
        visited[tour[step]] = True
    return tour

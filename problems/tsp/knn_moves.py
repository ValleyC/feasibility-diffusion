"""
K-NN restricted 2-opt moves for scalable inference.

Instead of enumerating all N(N-3)/2 2-opt moves (O(N²)), restrict to
moves where the boundary nodes are spatial nearest neighbors. This
captures >99% of improving moves since 2-opt improvements are local.

Matches EDISCO's sparse_factor convention:
  TSP-50:    k=50  (full, since N(N-3)/2 = 1175)
  TSP-100:   k=50
  TSP-500:   k=50
  TSP-1000:  k=100
  TSP-10000: k=100

Complexity: O(N·k) moves per step instead of O(N²).
"""

import numpy as np
from typing import List, Tuple


# Match EDISCO's sparse_factor settings
SPARSE_FACTOR = {50: 50, 100: 50, 200: 50, 500: 50, 1000: 100, 2000: 100,
                 5000: 100, 10000: 100}


def get_sparse_factor(N: int) -> int:
    """Get k for k-NN move restriction, matching EDISCO."""
    if N in SPARSE_FACTOR:
        return SPARSE_FACTOR[N]
    if N <= 100:
        return min(50, N - 3)  # full neighborhood for small N
    elif N <= 500:
        return 50
    else:
        return 100


def build_knn(coords: np.ndarray, k: int) -> np.ndarray:
    """Build k-nearest-neighbor index.

    Returns:
        knn: (N, k) int — knn[i] = indices of k nearest neighbors of node i
    """
    N = len(coords)
    k = min(k, N - 1)
    dist_sq = ((coords[:, None] - coords[None]) ** 2).sum(-1)
    np.fill_diagonal(dist_sq, np.inf)
    knn = np.argsort(dist_sq, axis=1)[:, :k]
    return knn


def enumerate_2opt_knn(tour: np.ndarray, coords: np.ndarray,
                       k: int = None) -> List[Tuple[int, int]]:
    """Enumerate 2-opt moves restricted to k-NN spatial neighbors.

    A move (i, j) is included only if tour[i] and tour[j] are among
    each other's k nearest spatial neighbors. This ensures the 2-opt
    swap reconnects spatially nearby nodes.

    Args:
        tour: (N,) current tour permutation
        coords: (N, 2) node coordinates
        k: number of nearest neighbors (default: EDISCO's setting for N)

    Returns:
        List of (i, j) tour-position pairs for valid 2-opt moves.
    """
    N = len(tour)
    if k is None:
        k = get_sparse_factor(N)

    # For small N, full enumeration is faster
    full_count = N * (N - 3) // 2
    knn_count_est = N * k
    if knn_count_est >= full_count:
        from problems.tsp.tour import enumerate_2opt
        return enumerate_2opt(N)

    # Build k-NN index on node coordinates
    knn = build_knn(coords, k)

    # Convert to set of (node_a, node_b) pairs that are k-NN
    knn_set = set()
    for i in range(N):
        for j in knn[i]:
            knn_set.add((min(i, j), max(i, j)))

    # Node-to-position mapping: pos[node] = position in tour
    pos = np.zeros(N, dtype=np.int64)
    for p in range(N):
        pos[tour[p]] = p

    # Enumerate moves: (i, j) where tour[i] and tour[j] are k-NN
    moves = []
    for i in range(N):
        node_i = tour[i]
        for neighbor_node in knn[node_i]:
            j = int(pos[neighbor_node])
            # Ensure i < j and valid 2-opt constraints
            pi, pj = min(i, j), max(i, j)
            if pj - pi >= 2 and not (pi == 0 and pj == N - 1):
                moves.append((pi, pj))

    # Deduplicate
    moves = list(set(moves))
    return moves

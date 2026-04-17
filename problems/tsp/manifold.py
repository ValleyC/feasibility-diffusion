"""
TSP Feasibility Manifold.

State space: all Hamiltonian cycles on N nodes (= permutations up to
cyclic rotation and reflection).

Neighborhood: 2-opt moves. The 2-opt neighborhood graph on N-node tours
is connected (Lin, 1965), ensuring ergodicity of the forward CTMC.

Instance: a distance matrix D ∈ R^{N×N} (or equivalently, 2D coordinates).
Cost: total tour length = Σ_k D[tour[k], tour[(k+1) % N]].
"""

import numpy as np
from typing import Any, List, Tuple

from core.manifold import FeasibilityManifold
from problems.tsp.tour import (
    random_tour, is_valid_tour, tour_cost, enumerate_2opt,
    apply_2opt, delta_2opt,
)


class TSPManifold(FeasibilityManifold):
    """TSP with 2-opt neighborhood.

    Instance format: np.ndarray of shape (N, N) — distance matrix.
    Solution format: np.ndarray of shape (N,) — permutation of {0,...,N-1}.
    Move format: Tuple[int, int] — (i, j) indices for 2-opt reversal.
    """

    def sample_random(self, instance: np.ndarray) -> np.ndarray:
        """Sample a uniformly random valid tour."""
        N = instance.shape[0]
        return random_tour(N)

    def cost(self, solution: np.ndarray, instance: np.ndarray) -> float:
        """Compute tour length under distance matrix."""
        return tour_cost(solution, instance)

    def is_feasible(self, solution: np.ndarray, instance: np.ndarray) -> bool:
        """Check tour validity: must be a permutation of {0,...,N-1}."""
        return is_valid_tour(solution)

    def enumerate_moves(self, solution: np.ndarray,
                        instance: np.ndarray) -> List[Tuple[int, int]]:
        """Enumerate all valid 2-opt moves. O(N^2)."""
        return enumerate_2opt(len(solution))

    def apply_move(self, solution: np.ndarray,
                   move: Tuple[int, int]) -> np.ndarray:
        """Apply 2-opt reversal. Returns a NEW array (does not mutate)."""
        return apply_2opt(solution, move[0], move[1])

    def move_delta(self, solution: np.ndarray, move: Tuple[int, int],
                   instance: np.ndarray) -> float:
        """O(1) cost change computation for a 2-opt move."""
        return delta_2opt(solution, move[0], move[1], instance)

    def num_moves(self, N: int) -> int:
        """Number of valid 2-opt moves for N-node TSP: N(N-3)/2."""
        return N * (N - 3) // 2

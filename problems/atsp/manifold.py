"""
Asymmetric TSP (ATSP) Manifold.

Same 2-opt moves as TSP, but delta computation must account for the
direction change of ALL edges in the reversed segment (not just boundary).
The O(1) symmetric delta formula does not apply to ATSP.
"""

import numpy as np
from typing import Tuple
from problems.tsp.manifold import TSPManifold
from problems.tsp.tour import tour_cost, apply_2opt


class ATSPManifold(TSPManifold):
    """ATSP: inherits TSP moves but overrides delta to handle asymmetry."""

    def move_delta(self, solution: np.ndarray, move: Tuple[int, int],
                   instance) -> float:
        """Brute-force delta for ATSP (O(N) — segment reversal changes all internal edges)."""
        dist = self._get_dist(instance)
        cost_before = tour_cost(solution, dist)
        new_tour = apply_2opt(solution, move[0], move[1])
        cost_after = tour_cost(new_tour, dist)
        return cost_after - cost_before

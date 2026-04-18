"""
PCTSP Selection Manifold: diffusion over prize-feasible node subsets.

State space: all subsets of nodes where total prize >= min_prize.
Neighborhood: add + remove + swap (all check prize constraint).
Cost: tour_cost(selected) + penalties(unselected).

The sub-TSP routing is handled by an external solver — the diffusion
learns WHICH nodes to include, not the routing order.
"""

import numpy as np
from typing import Any, List

from core.manifold import FeasibilityManifold
from problems.pctsp.selection import (
    random_selection, selection_feasible, selection_cost,
    enumerate_selection_moves, apply_selection_move,
    delta_selection_move, SelMove,
)


class PCTSPManifold(FeasibilityManifold):
    """PCTSP selection manifold with add/remove/swap moves.

    Instance format: dict with keys:
        'coords': (N+1, 2)
        'prizes': (N+1,) — prize[0] = 0 (depot)
        'penalties': (N+1,) — penalty[0] = 0 (depot)
        'min_prize': float — minimum total prize required
        'dist': (N+1, N+1)
        'n_customers': int
    """

    def sample_random(self, instance: dict) -> np.ndarray:
        return random_selection(
            instance['n_customers'],
            instance['prizes'],
            instance['min_prize'],
        )

    def cost(self, solution: np.ndarray, instance: dict) -> float:
        return selection_cost(
            solution, instance['coords'], instance['dist'],
            instance['penalties'], depot=0,
        )

    def is_feasible(self, solution: np.ndarray, instance: dict) -> bool:
        return selection_feasible(
            solution, instance['prizes'],
            instance['min_prize'], instance['n_customers'],
        )

    def enumerate_moves(self, solution: np.ndarray,
                        instance: dict) -> List[SelMove]:
        return enumerate_selection_moves(
            solution, instance['prizes'], instance['min_prize'],
        )

    def apply_move(self, solution: np.ndarray, move: SelMove) -> np.ndarray:
        return apply_selection_move(solution, move)

    def move_delta(self, solution: np.ndarray, move: SelMove,
                   instance: dict) -> float:
        return delta_selection_move(
            solution, move, instance['coords'], instance['dist'],
            instance['penalties'], depot=0,
        )

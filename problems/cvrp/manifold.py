"""
CVRP Feasibility Manifold.

State space: all valid CVRP solutions (sets of routes with capacity constraints).
Neighborhood: intra-route 2-opt + inter-route relocate + inter-route swap.

Connectivity: the combined move set ensures any CVRP solution is reachable
from any other (via sequences of relocates and swaps). This guarantees
ergodicity of the forward CTMC.

Instance format: dict with 'coords' (N+1, 2), 'demands' (N+1,),
                 'capacity' float, 'dist' (N+1, N+1).
"""

import numpy as np
from typing import Any, List, Tuple

from core.manifold import FeasibilityManifold
from problems.cvrp.solution import (
    random_solution, is_feasible, solution_cost,
    enumerate_moves, apply_move, delta_move, Move,
)


class CVRPManifold(FeasibilityManifold):
    """CVRP with intra-2opt + relocate + swap neighborhood.

    Instance format: dict with keys:
        'coords': (N+1, 2) — node 0 is depot
        'demands': (N+1,) — demand[0] = 0
        'capacity': float
        'dist': (N+1, N+1) — distance matrix
        'n_customers': int
    """

    def sample_random(self, instance: dict) -> List[List[int]]:
        """Sample a random feasible CVRP solution."""
        return random_solution(
            instance['n_customers'],
            instance['demands'],
            instance['capacity'],
        )

    def cost(self, solution: List[List[int]], instance: dict) -> float:
        """Total CVRP cost (sum of route costs)."""
        return solution_cost(solution, instance['dist'], depot=0)

    def is_feasible(self, solution: List[List[int]], instance: dict) -> bool:
        """Check CVRP validity."""
        return is_feasible(
            solution, instance['demands'],
            instance['capacity'], instance['n_customers'],
        )

    def enumerate_moves(self, solution: List[List[int]],
                        instance: dict) -> List[Move]:
        """All feasibility-preserving moves from current solution."""
        return enumerate_moves(
            solution, instance['demands'], instance['capacity'],
        )

    def apply_move(self, solution: List[List[int]], move: Move) -> List[List[int]]:
        """Apply a move (returns NEW solution, does not modify input)."""
        return apply_move(solution, move)

    def move_delta(self, solution: List[List[int]], move: Move,
                   instance: dict) -> float:
        """O(1) cost change for any move type."""
        return delta_move(solution, move, instance['dist'], depot=0)

"""
CVRP Partition Manifold: diffusion over capacity-respecting customer assignments.

State space: all feasible partitions (assignments of customers to vehicles
where each vehicle's demand <= capacity).

Neighborhood: relocate (move customer between vehicles) + swap (exchange
customers). Both check capacity before allowing.

The sub-TSP routing within each vehicle is handled by an external solver
and is NOT part of the diffusion state. The diffusion learns WHICH customers
go together; routing is modular.

Connectivity: relocate + swap ensure any partition is reachable from any
other (via sequence of single-customer moves).
"""

import numpy as np
from typing import Any, List, Tuple

from core.manifold import FeasibilityManifold
from problems.cvrp.partition import (
    random_partition, partition_feasible, partition_cost,
    enumerate_partition_moves, apply_partition_move,
    delta_partition_move, PartMove,
)


class CVRPPartitionManifold(FeasibilityManifold):
    """CVRP partition manifold with relocate + swap moves.

    Instance format: dict with keys:
        'coords': (N+1, 2) — node 0 is depot
        'demands': (N+1,) — demand[0] = 0
        'capacity': float
        'dist': (N+1, N+1) — distance matrix
        'n_customers': int
    """

    def sample_random(self, instance: dict) -> np.ndarray:
        return random_partition(
            instance['n_customers'],
            instance['demands'],
            instance['capacity'],
        )

    def cost(self, solution: np.ndarray, instance: dict) -> float:
        return partition_cost(
            solution, instance['coords'], instance['dist'], depot=0,
        )

    def is_feasible(self, solution: np.ndarray, instance: dict) -> bool:
        return partition_feasible(
            solution, instance['demands'],
            instance['capacity'], instance['n_customers'],
        )

    def enumerate_moves(self, solution: np.ndarray,
                        instance: dict) -> List[PartMove]:
        return enumerate_partition_moves(
            solution, instance['demands'], instance['capacity'],
        )

    def apply_move(self, solution: np.ndarray, move: PartMove) -> np.ndarray:
        return apply_partition_move(solution, move)

    def move_delta(self, solution: np.ndarray, move: PartMove,
                   instance: dict) -> float:
        return delta_partition_move(
            solution, move, instance['coords'], instance['dist'], depot=0,
        )

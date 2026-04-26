"""
Pure CVRP instance generation. No time windows.

Capacity constraint only: sum of demands per route <= capacity.
"""

import numpy as np
from typing import List, Optional


def generate_cvrp_instance(n_customers: int, seed: Optional[int] = None,
                           capacity_ratio: float = 0.8) -> dict:
    """Generate a random CVRP instance (no time windows).

    Args:
        n_customers: number of customers (depot is node 0)
        seed: random seed
        capacity_ratio: controls tightness. Lower = tighter.

    Returns:
        Instance dict with coords, demands, capacity, dist, n_customers.
        Also includes dummy tw fields so fragment code works unchanged.
    """
    rng = np.random.RandomState(seed)
    N = n_customers
    n_total = N + 1

    # Coordinates: uniform [0, 1]^2, depot at center
    coords = rng.uniform(0, 1, size=(n_total, 2)).astype(np.float32)
    coords[0] = [0.5, 0.5]

    # Distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    # Demands: uniform [1, 10)
    demands = np.zeros(n_total, dtype=np.float32)
    demands[1:] = rng.randint(1, 10, size=N).astype(np.float32)

    # Capacity
    total_demand = demands.sum()
    avg_demand = total_demand / N
    expected_per_route = 8
    expected_routes = max(N / expected_per_route, 1)
    capacity = total_demand * capacity_ratio / expected_routes
    capacity = max(capacity, avg_demand * 2)

    # Dummy TW fields (always feasible — effectively no time windows)
    horizon = 1e6
    tw_early = np.zeros(n_total, dtype=np.float32)
    tw_late = np.full(n_total, horizon, dtype=np.float32)
    service_time = np.zeros(n_total, dtype=np.float32)

    return {
        'coords': coords,
        'dist': dist,
        'demands': demands,
        'capacity': float(capacity),
        'tw_early': tw_early,
        'tw_late': tw_late,
        'service_time': service_time,
        'n_customers': N,
    }


def generate_cvrp_dataset(n_customers: int, n_instances: int,
                          seed: int = 42, **kwargs) -> List[dict]:
    """Generate a batch of CVRP instances."""
    return [generate_cvrp_instance(n_customers, seed=seed + i, **kwargs)
            for i in range(n_instances)]

"""
CVRPTW instance generation and benchmark loading.

Supports:
  - Random Euclidean instances with configurable constraint tightness
  - Solomon benchmark instances (R1/R2/C1/C2/RC1/RC2)
  - Constraint-shift variants for robustness evaluation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def generate_instance(n_customers: int, seed: Optional[int] = None,
                      capacity_ratio: float = 0.8,
                      tw_width: str = 'medium') -> dict:
    """Generate a random CVRPTW instance.

    Args:
        n_customers: number of customers (depot is node 0)
        seed: random seed
        capacity_ratio: controls tightness. Lower = tighter capacity.
            capacity = total_demand * capacity_ratio / expected_routes
        tw_width: 'tight', 'medium', 'loose', or float (absolute width)

    Returns:
        Instance dict with coords, demands, capacity, tw_early, tw_late,
        service_time, dist, n_customers.
    """
    rng = np.random.RandomState(seed)
    N = n_customers
    n_total = N + 1  # depot + customers

    # Coordinates: uniform [0, 1]^2, depot at center
    coords = rng.uniform(0, 1, size=(n_total, 2)).astype(np.float32)
    coords[0] = [0.5, 0.5]  # depot at center

    # Distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    # Demands: uniform [1, 10) for customers, 0 for depot
    demands = np.zeros(n_total, dtype=np.float32)
    demands[1:] = rng.randint(1, 10, size=N).astype(np.float32)

    # Capacity: based on ratio
    total_demand = demands.sum()
    avg_demand = total_demand / N
    expected_per_route = 8  # rough target
    expected_routes = max(N / expected_per_route, 1)
    capacity = total_demand * capacity_ratio / expected_routes
    capacity = max(capacity, avg_demand * 2)  # at least 2 customers per route

    # Time windows
    horizon = 5.0
    service_time = np.zeros(n_total, dtype=np.float32)
    service_time[1:] = 0.1

    tw_early = np.zeros(n_total, dtype=np.float32)
    tw_late = np.full(n_total, horizon, dtype=np.float32)

    # Customer time windows
    if isinstance(tw_width, (int, float)):
        width = float(tw_width)
    elif tw_width == 'tight':
        width = 0.5
    elif tw_width == 'medium':
        width = 1.0
    elif tw_width == 'loose':
        width = 2.0
    else:
        width = 1.0

    for c in range(1, n_total):
        center = rng.uniform(0.5, horizon - 0.5)
        half_w = width / 2
        tw_early[c] = max(0.0, center - half_w)
        tw_late[c] = min(horizon, center + half_w)

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


def generate_dataset(n_customers: int, n_instances: int, seed: int = 42,
                     **kwargs) -> List[dict]:
    """Generate a batch of CVRPTW instances."""
    instances = []
    for i in range(n_instances):
        inst = generate_instance(n_customers, seed=seed + i, **kwargs)
        instances.append(inst)
    return instances


def generate_constraint_shift(base_instances: List[dict],
                              capacity_ratio: Optional[float] = None,
                              tw_width: Optional[float] = None,
                              ) -> List[dict]:
    """Create constraint-shifted variants of existing instances.

    Keeps the same coordinates and demands but changes capacity and/or
    time windows. Used for robustness evaluation.
    """
    shifted = []
    for inst in base_instances:
        new_inst = dict(inst)  # shallow copy

        if capacity_ratio is not None:
            total_demand = inst['demands'].sum()
            N = inst['n_customers']
            avg_demand = total_demand / N
            expected_routes = max(N / 8, 1)
            new_cap = total_demand * capacity_ratio / expected_routes
            new_cap = max(new_cap, avg_demand * 2)
            new_inst['capacity'] = float(new_cap)

        if tw_width is not None:
            n_total = inst['n_customers'] + 1
            horizon = inst['tw_late'][0]
            rng = np.random.RandomState(42)  # deterministic shift
            tw_early = np.zeros(n_total, dtype=np.float32)
            tw_late = np.full(n_total, horizon, dtype=np.float32)
            for c in range(1, n_total):
                center = (inst['tw_early'][c] + inst['tw_late'][c]) / 2
                half_w = tw_width / 2
                tw_early[c] = max(0.0, center - half_w)
                tw_late[c] = min(horizon, center + half_w)
            new_inst['tw_early'] = tw_early
            new_inst['tw_late'] = tw_late

        shifted.append(new_inst)
    return shifted

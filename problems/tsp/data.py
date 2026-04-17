"""
TSP Instance Generation and Optimal Solution Computation.

Generates random Euclidean TSP instances and computes optimal (or
near-optimal) solutions for use as "clean" targets in the forward process.

For small N (≤ 20): brute-force or dynamic programming (Held-Karp).
For medium N (≤ 100): nearest-neighbor + exhaustive 2-opt (local optimum).
For large N: use LKH-3 or Concorde externally.
"""

import numpy as np
from typing import Tuple, List, Optional
from problems.tsp.tour import (
    dist_matrix_from_coords, tour_cost, enumerate_2opt,
    apply_2opt, delta_2opt, greedy_nearest_neighbor, random_tour,
)


def generate_instance(N: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random Euclidean TSP instance.

    Args:
        N: number of nodes
        seed: random seed (optional)

    Returns:
        coords: (N, 2) coordinates in [0, 1]^2
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(N, 2).astype(np.float32)


def solve_2opt(coords: np.ndarray, max_restarts: int = 1,
               max_iters: int = 10000) -> Tuple[np.ndarray, float]:
    """Solve TSP via multi-start nearest-neighbor + 2-opt local search.

    Not optimal for N > ~15, but gives a good "clean" solution for training.
    For truly optimal solutions on larger instances, use Concorde/LKH externally.

    Args:
        coords: (N, 2) node coordinates
        max_restarts: number of NN restarts from different starting nodes
        max_iters: max 2-opt improvement iterations per restart

    Returns:
        (best_tour, best_cost)
    """
    N = len(coords)
    dist = dist_matrix_from_coords(coords)
    moves = enumerate_2opt(N)

    best_tour = None
    best_cost = float('inf')

    starts = range(min(max_restarts, N))
    for start in starts:
        tour = greedy_nearest_neighbor(coords, start=start)

        # Exhaustive 2-opt until no improvement
        improved = True
        n_iter = 0
        while improved and n_iter < max_iters:
            improved = False
            for i, j in moves:
                d = delta_2opt(tour, i, j, dist)
                if d < -1e-10:
                    tour = apply_2opt(tour, i, j)
                    improved = True
                    break  # restart scan after each improvement
            n_iter += 1

        c = tour_cost(tour, dist)
        if c < best_cost:
            best_cost = c
            best_tour = tour.copy()

    return best_tour, best_cost


def generate_dataset(N: int, n_instances: int,
                     seed: int = 42,
                     solver_restarts: int = 5,
                     ) -> Tuple[List[np.ndarray], List[np.ndarray],
                                List[np.ndarray], List[float]]:
    """Generate a dataset of TSP instances with near-optimal solutions.

    Args:
        N: number of nodes per instance
        n_instances: number of instances
        seed: base random seed
        solver_restarts: NN restarts for 2-opt solver

    Returns:
        coords_list: list of (N, 2) coordinate arrays
        dist_list: list of (N, N) distance matrices
        tour_list: list of (N,) near-optimal tours
        cost_list: list of tour costs
    """
    coords_list, dist_list, tour_list, cost_list = [], [], [], []

    for i in range(n_instances):
        coords = generate_instance(N, seed=seed + i)
        dist = dist_matrix_from_coords(coords)
        tour, cost = solve_2opt(coords, max_restarts=solver_restarts)

        coords_list.append(coords)
        dist_list.append(dist)
        tour_list.append(tour)
        cost_list.append(cost)

    return coords_list, dist_list, tour_list, cost_list

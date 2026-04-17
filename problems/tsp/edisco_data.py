"""
Load TSP training data in EDISCO/DIFUSCO format.

File format (one instance per line):
    x1 y1 x2 y2 ... xN yN output t1 t2 t3 ... tN t1

- Coordinates are space-separated floats in [0, 1]
- 'output' is a literal delimiter string
- Tour nodes are 1-indexed integers (we convert to 0-indexed)
- Tour is a closed loop: first node repeated at the end (we drop it)

Example line (N=5):
    0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 output 1 3 5 2 4 1

This produces:
    coords = [[0.1,0.2], [0.3,0.4], [0.5,0.6], [0.7,0.8], [0.9,1.0]]
    tour = [0, 2, 4, 1, 3]  (0-indexed, no repeat)
"""

import numpy as np
from typing import List, Tuple, Optional
from problems.tsp.tour import dist_matrix_from_coords, tour_cost, is_valid_tour


def load_edisco_tsp(filepath: str,
                    max_instances: Optional[int] = None,
                    ) -> Tuple[List[np.ndarray], List[np.ndarray],
                               List[np.ndarray], List[float]]:
    """Load TSP instances from EDISCO-format text file.

    Args:
        filepath: path to the .txt file
        max_instances: if given, only load this many instances

    Returns:
        coords_list: list of (N, 2) float32 arrays
        dist_list: list of (N, N) float32 distance matrices
        tour_list: list of (N,) int64 arrays (0-indexed permutations)
        cost_list: list of tour costs (float)
    """
    coords_list, dist_list, tour_list, cost_list = [], [], [], []

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_instances is not None and i >= max_instances:
                break

            line = line.strip()
            if not line:
                continue

            # Split on ' output '
            parts = line.split(' output ')
            assert len(parts) == 2, f"Line {i}: expected 'output' delimiter, got: {line[:100]}"

            # Parse coordinates
            coord_tokens = parts[0].split()
            assert len(coord_tokens) % 2 == 0, f"Line {i}: odd number of coordinate tokens"
            N = len(coord_tokens) // 2
            coords = np.array(
                [[float(coord_tokens[2*j]), float(coord_tokens[2*j+1])] for j in range(N)],
                dtype=np.float32,
            )

            # Parse tour (1-indexed, closed loop — last node repeats first)
            tour_tokens = parts[1].split()
            tour_1indexed = np.array([int(t) for t in tour_tokens])
            # Convert to 0-indexed
            tour = tour_1indexed - 1
            # Drop the repeated last node (closed loop)
            if len(tour) == N + 1 and tour[-1] == tour[0]:
                tour = tour[:-1]

            assert len(tour) == N, \
                f"Line {i}: tour length {len(tour)} != N={N}"
            assert is_valid_tour(tour), \
                f"Line {i}: invalid tour (not a permutation)"

            # Compute distance matrix and tour cost
            dist = dist_matrix_from_coords(coords)
            cost = tour_cost(tour, dist)

            coords_list.append(coords)
            dist_list.append(dist)
            tour_list.append(tour)
            cost_list.append(cost)

    print(f"Loaded {len(coords_list)} instances from {filepath} "
          f"(N={N}, avg cost={np.mean(cost_list):.4f})")
    return coords_list, dist_list, tour_list, cost_list

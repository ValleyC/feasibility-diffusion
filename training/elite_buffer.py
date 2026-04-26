"""
Elite Solution Buffer: manages high-quality solutions for training.

Initializes with solver-generated solutions (savings heuristic, OR-Tools,
or any classical CVRPTW solver). Supports self-improvement: decoded
solutions that improve the buffer are added automatically.

Extracts merge targets from elite solutions: which customers are adjacent
in the same route (positive), which are on different routes (negative).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from route_objects.fragment import simulate_route
from route_objects.fragment_state import FragmentState
from route_objects.projector import project_savings, project_random, finalize_routes, local_search_repair


@dataclass
class EliteSolution:
    """One elite solution for a CVRPTW instance."""
    routes: List[List[int]]   # list of customer sequences (no depot)
    cost: float               # total route cost


class EliteBuffer:
    """Manages elite solutions for a set of instances."""

    def __init__(self, instances: List[dict], n_elite: int = 10):
        self.instances = instances
        self.n_elite = n_elite
        self.solutions: Dict[int, List[EliteSolution]] = {
            i: [] for i in range(len(instances))
        }

    def initialize_with_savings(self, n_restarts: int = 5):
        """Generate initial elite solutions using savings + random perturbation."""
        for i, inst in enumerate(self.instances):
            # First restart: pure savings (deterministic)
            state = FragmentState.from_singletons(inst)
            state = project_savings(state, max_steps=500)
            state = local_search_repair(state, max_steps=50)
            routes = [f.seq for f in state.fragments]
            cost = state.total_cost()
            self.add(i, routes, cost)

            # Additional restarts: random merges + repair (for diversity)
            for restart in range(1, n_restarts):
                state = FragmentState.from_singletons(inst)
                state = project_random(state, max_steps=500)
                state = local_search_repair(state, max_steps=100)
                routes = [f.seq for f in state.fragments]
                cost = state.total_cost()
                self.add(i, routes, cost)

    def add(self, instance_idx: int, routes: List[List[int]], cost: float) -> bool:
        """Add a solution if it actually enters the retained top-K.

        Returns True only if the solution survives the top-K truncation.
        """
        sol = EliteSolution(routes=routes, cost=cost)
        buf = self.solutions[instance_idx]

        # Duplicate check
        for existing in buf:
            if abs(existing.cost - cost) < 1e-8:
                return False

        # If buffer is full and this is worse than the worst elite, reject
        if len(buf) >= self.n_elite and cost >= buf[-1].cost - 1e-8:
            return False

        buf.append(sol)
        buf.sort(key=lambda s: s.cost)

        # Keep top-K
        if len(buf) > self.n_elite:
            buf.pop()

        return True

    def get_elite(self, instance_idx: int) -> List[EliteSolution]:
        return self.solutions[instance_idx]

    def best_cost(self, instance_idx: int) -> float:
        buf = self.solutions[instance_idx]
        return buf[0].cost if buf else float('inf')

    def avg_cost(self, instance_idx: int) -> float:
        buf = self.solutions[instance_idx]
        return np.mean([s.cost for s in buf]) if buf else float('inf')


def extract_merge_targets(buffer: EliteBuffer, instance_idx: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract soft merge targets from elite consensus.

    For each pair (c_i, c_j) of customers:
      - merge_target[i,j] = fraction of elite solutions where c_i and c_j
        are adjacent on the same route (in order c_i → c_j)
      - This gives a soft, multi-modal target

    Returns:
        adjacency_freq: (N+1, N+1) float — frequency of (i,j) being adjacent
            in elite solutions (i immediately before j). Includes depot (0).
        coroute_freq: (N+1, N+1) float — frequency of (i,j) being on the
            same route (not necessarily adjacent).
    """
    inst = buffer.instances[instance_idx]
    N = inst['n_customers']
    n_total = N + 1

    elites = buffer.get_elite(instance_idx)
    if not elites:
        return np.zeros((n_total, n_total), dtype=np.float32), \
               np.zeros((n_total, n_total), dtype=np.float32)

    adjacency_freq = np.zeros((n_total, n_total), dtype=np.float32)
    coroute_freq = np.zeros((n_total, n_total), dtype=np.float32)
    n_elites = len(elites)

    for sol in elites:
        for route in sol.routes:
            # Co-route frequency
            for ci in route:
                for cj in route:
                    if ci != cj:
                        coroute_freq[ci, cj] += 1.0 / n_elites

            # Adjacency frequency (directed: i → j)
            full_route = [0] + route + [0]  # depot → customers → depot
            for pos in range(len(full_route) - 1):
                i, j = full_route[pos], full_route[pos + 1]
                adjacency_freq[i, j] += 1.0 / n_elites

    return adjacency_freq, coroute_freq


def extract_pairwise_labels(buffer: EliteBuffer, instance_idx: int,
                            edge_index: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-edge labels for the fragment graph.

    For each edge (src_frag, dst_frag) in the fragment graph, compute:
      - merge_label: should these fragments be merged? (from elite consensus)
      - risk_label: is this merge infeasible? (from exact check)

    This is called during training when fragments may be partially merged.

    Args:
        buffer: elite buffer
        instance_idx: which instance
        edge_index: (E, 2) fragment indices for candidate merges

    Returns:
        merge_labels: (E,) float in [0, 1]
        risk_labels: (E,) float in {0, 1}
    """
    adjacency_freq, coroute_freq = extract_merge_targets(buffer, instance_idx)

    # For now, return the raw frequencies — the training loop will
    # map fragment edges to customer pairs
    return adjacency_freq, coroute_freq

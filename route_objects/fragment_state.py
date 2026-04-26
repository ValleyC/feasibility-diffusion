"""
Fragment State: manages a set of route fragments and builds the fragment graph
for GNN scoring.

The fragment graph is a sparse directed graph where:
  - Nodes = fragments
  - Edges = candidate oriented merges (f_i.end → f_j.start within k-NN)
  - Node features = fragment constraint summaries
  - Edge features = connection distance, combined load ratio, TW compatibility
"""

import numpy as np
from typing import List, Tuple, Optional
from route_objects.fragment import (
    RouteFragment, create_singleton, merge_fragments,
    check_merge_feasible, route_cost_with_depot,
)


class FragmentState:
    """Manages a set of route fragments for one instance."""

    def __init__(self, fragments: List[RouteFragment], instance: dict):
        self.fragments = list(fragments)
        self.instance = instance

    @classmethod
    def from_singletons(cls, instance: dict) -> 'FragmentState':
        """Initialize with one singleton fragment per customer."""
        n_customers = instance['n_customers']
        fragments = [create_singleton(c, instance) for c in range(1, n_customers + 1)]
        return cls(fragments, instance)

    @classmethod
    def from_routes(cls, routes: List[List[int]], instance: dict) -> 'FragmentState':
        """Initialize from a set of routes (e.g., from an elite solution)."""
        fragments = []
        for route in routes:
            if not route:
                continue
            frag = create_singleton(route[0], instance)
            for c in route[1:]:
                next_frag = create_singleton(c, instance)
                frag = merge_fragments(frag, next_frag, instance, orientation=0)
            fragments.append(frag)
        return cls(fragments, instance)

    def apply_merge(self, idx_i: int, idx_j: int, orientation: int = 0) -> 'FragmentState':
        """Merge fragments at indices idx_i and idx_j, return new state."""
        f_i = self.fragments[idx_i]
        f_j = self.fragments[idx_j]
        merged = merge_fragments(f_i, f_j, self.instance, orientation)

        new_frags = []
        for k, f in enumerate(self.fragments):
            if k != idx_i and k != idx_j:
                new_frags.append(f)
        new_frags.append(merged)

        return FragmentState(new_frags, self.instance)

    @property
    def n_fragments(self) -> int:
        return len(self.fragments)

    def total_cost(self) -> float:
        """Total route cost: sum of depot → fragment → depot for all fragments."""
        return sum(route_cost_with_depot(f, self.instance) for f in self.fragments)


def build_fragment_graph(state: FragmentState, k: int = 20
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse directed graph of candidate oriented merges.

    For each fragment, connect to the k nearest fragments by endpoint distance.
    Two orientations per pair: (f_i→f_j) and (f_j→f_i).

    Returns:
        node_features: (F, n_node_feat) fragment features
        edge_index: (E, 2) int — (src_fragment, dst_fragment) for oriented merge
        edge_features: (E, n_edge_feat) edge features
    """
    instance = state.instance
    frags = state.fragments
    F = len(frags)
    coords = instance['coords']
    capacity = instance['capacity']
    dist = instance['dist']
    tw_early = instance['tw_early']
    tw_late = instance['tw_late']
    service_time = instance['service_time']
    horizon = tw_late[0] if tw_late[0] > 0 else 1.0
    N = instance['n_customers']

    # -- Node features --
    # 12 features per fragment
    node_feat = np.zeros((F, 12), dtype=np.float32)
    for i, f in enumerate(frags):
        node_feat[i, 0] = coords[f.start_node, 0]  # start x
        node_feat[i, 1] = coords[f.start_node, 1]  # start y
        node_feat[i, 2] = coords[f.end_node, 0]     # end x
        node_feat[i, 3] = coords[f.end_node, 1]     # end y
        node_feat[i, 4] = f.load / capacity
        node_feat[i, 5] = f.size / N
        node_feat[i, 6] = f.travel_cost / (dist.max() + 1e-8)
        node_feat[i, 7] = f.earliest_depart / horizon
        node_feat[i, 8] = f.latest_start / horizon
        node_feat[i, 9] = f.forward_slack / horizon
        node_feat[i, 10] = f.backward_slack / horizon
        node_feat[i, 11] = f.service_time_sum / horizon

    # -- Edges: k-NN on endpoint distances --
    # For oriented merge f_i → f_j: distance from f_i.end_node to f_j.start_node
    # We consider both orientations, so build a distance matrix for both directions
    if F <= 1:
        edge_index = np.zeros((0, 2), dtype=np.int64)
        edge_feat = np.zeros((0, 3), dtype=np.float32)
        return node_feat, edge_index, edge_feat

    k_use = min(k, F - 1)

    edges = []
    e_feats = []

    # For each fragment i, find k nearest fragments j such that
    # dist(f_i.end, f_j.start) is small → oriented merge f_i → f_j
    end_nodes = [f.end_node for f in frags]
    start_nodes = [f.start_node for f in frags]

    # Endpoint distance matrix (F x F): dist from f_i.end to f_j.start
    ep_dist = np.zeros((F, F), dtype=np.float32)
    for i in range(F):
        for j in range(F):
            if i == j:
                ep_dist[i, j] = float('inf')
            else:
                ep_dist[i, j] = dist[end_nodes[i], start_nodes[j]]

    max_dist = dist.max() + 1e-8

    for i in range(F):
        # k-nearest by endpoint distance for orientation i→j
        neighbors = np.argsort(ep_dist[i])[:k_use]
        for j in neighbors:
            if i == j:
                continue
            combined_load = frags[i].load + frags[j].load
            connection_dist = ep_dist[i, j]

            # Rough TW compatibility: can f_j start after f_i finishes?
            # f_i earliest departure + travel ≤ f_j latest start
            travel_ij = dist[end_nodes[i], start_nodes[j]]
            tw_compat = 1.0 if (frags[i].earliest_depart + travel_ij
                                <= frags[j].latest_start + 1e-8) else 0.0

            edges.append([i, j])
            e_feats.append([
                connection_dist / max_dist,
                combined_load / capacity,
                tw_compat,
            ])

    if not edges:
        edge_index = np.zeros((0, 2), dtype=np.int64)
        edge_feat = np.zeros((0, 3), dtype=np.float32)
    else:
        edge_index = np.array(edges, dtype=np.int64)
        edge_feat = np.array(e_feats, dtype=np.float32)

    return node_feat, edge_index, edge_feat

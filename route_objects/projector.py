"""
Constrained Projector: greedy agglomeration from scored fragment merges.

At each step:
  1. Build fragment graph (k-NN on endpoints)
  2. Score candidate merges with the model
  3. Rank by score, filter by risk threshold
  4. For each candidate: capacity screen → exact TW check → accept if feasible
  5. Re-score after each accepted merge (one merge per step)
  6. Repeat until no feasible merge improves or stopping condition

This is a learned constrained agglomeration process. The model predicts
which merges are desirable and which are risky; the projector enforces
exact feasibility.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple

from route_objects.fragment import (
    RouteFragment, check_merge_feasible, merge_fragments,
    route_cost_with_depot, simulate_route,
)
from route_objects.fragment_state import FragmentState, build_fragment_graph


@torch.no_grad()
def project(model, state: FragmentState, device: torch.device,
            k: int = 20, max_steps: int = 200,
            risk_threshold: float = 0.8,
            min_fragments: int = 1) -> FragmentState:
    """Run constrained agglomeration using model-scored merges.

    Args:
        model: FragmentGNN (or None for random baseline)
        state: initial FragmentState (typically singletons)
        device: torch device
        k: k-NN parameter for fragment graph
        max_steps: maximum merge steps
        risk_threshold: reject candidates with risk > threshold
        min_fragments: stop when this many fragments remain

    Returns:
        Final FragmentState with merged fragments.
    """
    if model is not None:
        model.eval()

    for step in range(max_steps):
        if state.n_fragments <= min_fragments:
            break

        # Build fragment graph
        node_feat, edge_index, edge_feat = build_fragment_graph(state, k=k)

        if len(edge_index) == 0:
            break

        # Score with model (or random)
        if model is not None:
            nf_t = torch.tensor(node_feat, dtype=torch.float32, device=device)
            ei_t = torch.tensor(edge_index, dtype=torch.long, device=device)
            ef_t = torch.tensor(edge_feat, dtype=torch.float32, device=device)

            merge_scores, risk_scores = model(nf_t, ei_t, ef_t)
            merge_scores = merge_scores.cpu().numpy()
            risk_scores = risk_scores.cpu().numpy()
        else:
            # Random baseline: uniform scores
            merge_scores = np.random.randn(len(edge_index))
            risk_scores = np.zeros(len(edge_index))

        # Rank by merge score (descending), filter by risk
        order = np.argsort(-merge_scores)

        merged = False
        for rank_idx in order:
            # Risk filter
            if risk_scores[rank_idx] > risk_threshold:
                continue

            src_idx = edge_index[rank_idx, 0]
            dst_idx = edge_index[rank_idx, 1]

            if src_idx == dst_idx:
                continue

            f_src = state.fragments[src_idx]
            f_dst = state.fragments[dst_idx]

            # Orientation: src.end → dst.start (orientation=0 by edge construction)
            feasible, cost = check_merge_feasible(
                f_src, f_dst, state.instance, orientation=0)

            if feasible:
                state = state.apply_merge(src_idx, dst_idx, orientation=0)
                merged = True
                break

        if not merged:
            break

    return state


def project_random(state: FragmentState, max_steps: int = 200) -> FragmentState:
    """Random merge baseline: merge random feasible pairs."""
    return project(None, state, device=torch.device('cpu'), max_steps=max_steps)


def project_savings(state: FragmentState, max_steps: int = 200) -> FragmentState:
    """Clarke-Wright savings heuristic baseline.

    Merge the pair that saves the most distance compared to separate
    depot round-trips. This is a classical non-learned baseline.
    """
    instance = state.instance
    dist = instance['dist']
    depot = 0

    for step in range(max_steps):
        if state.n_fragments <= 1:
            break

        best_saving = -float('inf')
        best_merge = None

        for i in range(state.n_fragments):
            for j in range(i + 1, state.n_fragments):
                f_i = state.fragments[i]
                f_j = state.fragments[j]

                # Savings for merging f_i → f_j (orientation 0)
                saving_0 = (dist[f_i.end_node, depot] + dist[depot, f_j.start_node]
                            - dist[f_i.end_node, f_j.start_node])
                # Savings for merging f_j → f_i (orientation 1)
                saving_1 = (dist[f_j.end_node, depot] + dist[depot, f_i.start_node]
                            - dist[f_j.end_node, f_i.start_node])

                for orient, saving in [(0, saving_0), (1, saving_1)]:
                    if saving > best_saving:
                        feasible, _ = check_merge_feasible(
                            f_i, f_j, instance, orientation=orient)
                        if feasible:
                            best_saving = saving
                            best_merge = (i, j, orient)

        if best_merge is None or best_saving <= 0:
            break

        i, j, orient = best_merge
        state = state.apply_merge(i, j, orientation=orient)

    return state


def finalize_routes(state: FragmentState) -> List[List[int]]:
    """Convert fragment state to list of routes (depot → customers → depot)."""
    routes = []
    for f in state.fragments:
        route = [0] + f.seq + [0]  # depot → seq → depot
        routes.append(route)
    return routes


def local_search_repair(state: FragmentState, max_steps: int = 50) -> FragmentState:
    """Short local search polish using relocate/swap moves.

    Reuses the existing partition-based moves from problems/cvrp/partition.py
    and CVRPTW feasibility checking.
    """
    instance = state.instance
    dist = instance['dist']
    capacity = instance['capacity']
    demands = instance['demands']
    tw_early = instance['tw_early']
    tw_late = instance['tw_late']
    service_time = instance['service_time']

    # Build assignment vector from current fragments
    n_total = instance['n_customers'] + 1
    assign = np.full(n_total, -1, dtype=np.int64)
    route_seqs = []

    for k, frag in enumerate(state.fragments):
        for c in frag.seq:
            assign[c] = k
        route_seqs.append(list(frag.seq))

    # Simple relocate-based local search
    improved = True
    step = 0
    while improved and step < max_steps:
        improved = False
        step += 1

        for k_src in range(len(route_seqs)):
            if not route_seqs[k_src]:
                continue
            for ci, c in enumerate(list(route_seqs[k_src])):
                # Try relocating customer c to another route
                for k_dst in range(len(route_seqs)):
                    if k_dst == k_src:
                        continue

                    # Capacity check
                    dst_load = sum(demands[cc] for cc in route_seqs[k_dst])
                    if dst_load + demands[c] > capacity + 1e-8:
                        continue

                    # Try inserting c into dst route
                    new_dst = route_seqs[k_dst] + [c]
                    feasible_dst, cost_dst = simulate_route(new_dst, instance)
                    if not feasible_dst:
                        continue

                    new_src = [cc for cc in route_seqs[k_src] if cc != c]
                    feasible_src, cost_src = simulate_route(new_src, instance)
                    if not feasible_src and new_src:
                        continue

                    # Compute cost change
                    old_cost_src = route_cost_with_depot(state.fragments[k_src], instance) if route_seqs[k_src] else 0
                    old_cost_dst = route_cost_with_depot(state.fragments[k_dst], instance) if route_seqs[k_dst] else 0
                    new_total = cost_src + cost_dst
                    old_total = old_cost_src + old_cost_dst

                    if new_total < old_total - 1e-8:
                        route_seqs[k_src] = new_src
                        route_seqs[k_dst] = new_dst
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Rebuild fragments from modified routes
    from route_objects.fragment import create_singleton, merge_fragments
    new_frags = []
    for seq in route_seqs:
        if not seq:
            continue
        frag = create_singleton(seq[0], instance)
        for c in seq[1:]:
            next_f = create_singleton(c, instance)
            frag = merge_fragments(frag, next_f, instance, orientation=0)
        new_frags.append(frag)

    return FragmentState(new_frags, instance)

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
    RouteFragment, create_singleton, check_merge_feasible, merge_fragments,
    route_cost_with_depot, simulate_route,
)
from route_objects.fragment_state import FragmentState, build_fragment_graph


@torch.no_grad()
def project(model, state: FragmentState, device: torch.device,
            k: int = 20, max_steps: int = 200,
            risk_threshold: float = 0.8,
            min_fragments: int = 1,
            score_threshold: float = 0.0) -> FragmentState:
    """Run constrained agglomeration using model-scored merges.

    Stops when:
      - No feasible merge exists
      - Best merge score is below score_threshold (utility-based stopping)
      - min_fragments reached
      - max_steps exhausted

    Args:
        model: FragmentGNN (or None for random baseline)
        state: initial FragmentState (typically singletons)
        device: torch device
        k: k-NN parameter for fragment graph
        max_steps: maximum merge steps
        risk_threshold: reject candidates with risk > threshold
        min_fragments: stop when this many fragments remain
        score_threshold: stop when best merge score < this value

    Returns:
        Final FragmentState with merged fragments.
    """
    if model is not None:
        model.eval()

    dist = state.instance['dist']
    depot = 0

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
            score = merge_scores[rank_idx]

            # Utility-based stopping: don't merge if score is too low
            if model is not None and score < score_threshold:
                break  # scores are sorted, so all remaining are worse

            # Risk filter
            if risk_scores[rank_idx] > risk_threshold:
                continue

            src_idx = edge_index[rank_idx, 0]
            dst_idx = edge_index[rank_idx, 1]

            if src_idx == dst_idx:
                continue

            f_src = state.fragments[src_idx]
            f_dst = state.fragments[dst_idx]

            # Exact feasibility check (capacity + TW)
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
    """Short local search polish using relocate moves.

    Computes current route costs freshly at each step (no stale references).
    """
    instance = state.instance
    capacity = instance['capacity']
    demands = instance['demands']

    # Work with mutable route sequences
    route_seqs = [list(f.seq) for f in state.fragments]

    def _route_cost(seq):
        if not seq:
            return 0.0
        ok, c = simulate_route(seq, instance)
        return c if ok else float('inf')

    # Cache current costs (recomputed after every accepted move)
    route_costs = [_route_cost(s) for s in route_seqs]

    step = 0
    while step < max_steps:
        improved = False
        step += 1

        for k_src in range(len(route_seqs)):
            if not route_seqs[k_src]:
                continue
            for c in list(route_seqs[k_src]):
                best_gain = 0.0
                best_move = None

                # Cost of removing c from src
                new_src = [cc for cc in route_seqs[k_src] if cc != c]
                ok_src, cost_new_src = simulate_route(new_src, instance) if new_src else (True, 0.0)
                if not ok_src:
                    continue

                for k_dst in range(len(route_seqs)):
                    if k_dst == k_src:
                        continue

                    # Capacity check
                    dst_load = sum(demands[cc] for cc in route_seqs[k_dst])
                    if dst_load + demands[c] > capacity + 1e-8:
                        continue

                    # Try best insertion position in dst
                    best_insert_cost = float('inf')
                    best_insert_pos = -1
                    for pos in range(len(route_seqs[k_dst]) + 1):
                        new_dst = route_seqs[k_dst][:pos] + [c] + route_seqs[k_dst][pos:]
                        ok_dst, cost_new_dst = simulate_route(new_dst, instance)
                        if ok_dst and cost_new_dst < best_insert_cost:
                            best_insert_cost = cost_new_dst
                            best_insert_pos = pos

                    if best_insert_pos < 0:
                        continue

                    # Gain = old costs - new costs (positive = improvement)
                    gain = (route_costs[k_src] + route_costs[k_dst]) - \
                           (cost_new_src + best_insert_cost)

                    if gain > best_gain + 1e-8:
                        best_gain = gain
                        best_move = (k_dst, best_insert_pos, new_src,
                                     cost_new_src, best_insert_cost)

                if best_move is not None:
                    k_dst, pos, new_src, cost_new_src, cost_new_dst = best_move
                    new_dst = route_seqs[k_dst][:pos] + [c] + route_seqs[k_dst][pos:]
                    route_seqs[k_src] = new_src
                    route_seqs[k_dst] = new_dst
                    route_costs[k_src] = cost_new_src
                    route_costs[k_dst] = cost_new_dst
                    improved = True
                    break  # restart scan
            if improved:
                break

        if not improved:
            break

    # Rebuild fragments from modified routes
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

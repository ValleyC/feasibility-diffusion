"""
Route Fragment: the core state object for route-object compatibility learning.

A fragment is an ordered subsequence of customers that could form part of
a feasible route. Each fragment caches constraint summaries (load, TW slack)
for cheap scoring, but exact feasibility is always verified by the projector
via full sequence simulation.

Fragments start as singletons (one customer each). The generator merges
fragments into longer sequences. The projector enforces that each merge
is capacity- and TW-feasible.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RouteFragment:
    """An ordered subsequence of customers with cached constraint summaries."""

    seq: List[int]                # ordered customer indices (no depot)

    # Constraint summaries (cached, updated on merge)
    load: float = 0.0            # total demand
    travel_cost: float = 0.0     # total internal edge cost (no depot legs)
    service_time_sum: float = 0.0

    # TW summaries (from exact forward/backward pass on seq)
    earliest_depart: float = 0.0  # earliest feasible departure from last node
    latest_start: float = 0.0    # latest feasible start at first node
    forward_slack: float = 0.0   # min TW slack along forward schedule
    backward_slack: float = 0.0  # min TW slack along backward schedule

    @property
    def start_node(self) -> int:
        return self.seq[0]

    @property
    def end_node(self) -> int:
        return self.seq[-1]

    @property
    def size(self) -> int:
        return len(self.seq)


def create_singleton(customer: int, instance: dict) -> RouteFragment:
    """Create a singleton fragment for one customer."""
    demands = instance['demands']
    tw_early = instance['tw_early']
    tw_late = instance['tw_late']
    service_time = instance['service_time']

    return RouteFragment(
        seq=[customer],
        load=float(demands[customer]),
        travel_cost=0.0,
        service_time_sum=float(service_time[customer]),
        earliest_depart=tw_early[customer] + service_time[customer],
        latest_start=tw_late[customer],
        forward_slack=tw_late[customer] - tw_early[customer],
        backward_slack=tw_late[customer] - tw_early[customer],
    )


def compute_tw_summaries(seq: List[int], instance: dict) -> Tuple[float, float, float, float]:
    """Compute TW summaries for a customer sequence via exact simulation.

    Forward pass: depot → seq[0] → ... → seq[-1]
    Backward pass: seq[-1] → ... → seq[0] → depot

    Returns:
        (earliest_depart, latest_start, forward_slack, backward_slack)
    """
    dist = instance['dist']
    tw_early = instance['tw_early']
    tw_late = instance['tw_late']
    service_time = instance['service_time']
    depot = 0

    if not seq:
        return 0.0, float('inf'), float('inf'), float('inf')

    # Forward pass: compute earliest feasible schedule
    min_slack_fwd = float('inf')
    current_time = 0.0  # depart depot at time 0
    prev = depot

    for c in seq:
        travel = dist[prev, c]
        arrive = current_time + travel
        start_service = max(arrive, tw_early[c])
        slack = tw_late[c] - start_service
        min_slack_fwd = min(min_slack_fwd, slack)
        current_time = start_service + service_time[c]
        prev = c

    earliest_depart = current_time  # time after serving last customer

    # Backward pass: compute latest feasible start at seq[0]
    # Work backwards from seq[-1] to seq[0]
    # latest_arrive[i] = tw_late[seq[i]]
    min_slack_bwd = float('inf')
    latest_finish = tw_late[seq[-1]] + service_time[seq[-1]]

    for i in range(len(seq) - 1, -1, -1):
        c = seq[i]
        latest_arrive = tw_late[c]
        if i < len(seq) - 1:
            next_c = seq[i + 1]
            # Must arrive at next_c by tw_late[next_c]
            latest_depart_c = tw_late[next_c] - dist[c, next_c]
            latest_arrive = min(tw_late[c], latest_depart_c - service_time[c])
        slack = latest_arrive - tw_early[c]
        min_slack_bwd = min(min_slack_bwd, slack)

    latest_start = latest_arrive if len(seq) > 0 else float('inf')

    return earliest_depart, latest_start, min_slack_fwd, min_slack_bwd


def check_merge_feasible(f1: RouteFragment, f2: RouteFragment,
                         instance: dict, orientation: int = 0) -> Tuple[bool, float]:
    """Check if merging two fragments is feasible (capacity + TW).

    Args:
        f1, f2: fragments to merge
        orientation: 0 = f1.seq + f2.seq, 1 = f2.seq + f1.seq
        instance: problem instance

    Returns:
        (feasible, cost) where cost is total route distance if feasible,
        else inf.
    """
    capacity = instance['capacity']
    combined_load = f1.load + f2.load

    # Quick capacity screen
    if combined_load > capacity + 1e-8:
        return False, float('inf')

    # Build concatenated sequence
    if orientation == 0:
        merged_seq = f1.seq + f2.seq
    else:
        merged_seq = f2.seq + f1.seq

    # Exact TW feasibility check via forward simulation
    feasible, cost = simulate_route(merged_seq, instance)
    return feasible, cost


def simulate_route(seq: List[int], instance: dict) -> Tuple[bool, float]:
    """Simulate a route (depot → seq → depot) checking TW feasibility.

    Returns (feasible, total_distance).
    """
    if not seq:
        return True, 0.0

    dist = instance['dist']
    tw_early = instance['tw_early']
    tw_late = instance['tw_late']
    service_time = instance['service_time']
    depot = 0

    current_time = 0.0
    total_dist = 0.0
    prev = depot

    for c in seq:
        travel = dist[prev, c]
        arrive = current_time + travel
        if arrive > tw_late[c] + 1e-8:
            return False, float('inf')
        start_service = max(arrive, tw_early[c])
        current_time = start_service + service_time[c]
        total_dist += travel
        prev = c

    # Return to depot
    total_dist += dist[prev, depot]
    arrive_depot = current_time + dist[prev, depot]
    if arrive_depot > tw_late[depot] + 1e-8:
        return False, float('inf')

    return True, total_dist


def merge_fragments(f1: RouteFragment, f2: RouteFragment,
                    instance: dict, orientation: int = 0) -> RouteFragment:
    """Create a new fragment by merging f1 and f2.

    Assumes feasibility has already been checked.
    """
    if orientation == 0:
        merged_seq = f1.seq + f2.seq
    else:
        merged_seq = f2.seq + f1.seq

    dist = instance['dist']
    demands = instance['demands']
    service_time = instance['service_time']

    # Compute internal travel cost (no depot legs)
    travel_cost = 0.0
    for i in range(len(merged_seq) - 1):
        travel_cost += dist[merged_seq[i], merged_seq[i + 1]]

    load = sum(demands[c] for c in merged_seq)
    svc_sum = sum(service_time[c] for c in merged_seq)

    # Exact TW summaries
    ed, ls, fs, bs = compute_tw_summaries(merged_seq, instance)

    return RouteFragment(
        seq=merged_seq,
        load=load,
        travel_cost=travel_cost,
        service_time_sum=svc_sum,
        earliest_depart=ed,
        latest_start=ls,
        forward_slack=fs,
        backward_slack=bs,
    )


def route_cost_with_depot(fragment: RouteFragment, instance: dict) -> float:
    """Compute full route cost: depot → fragment.seq → depot."""
    feasible, cost = simulate_route(fragment.seq, instance)
    return cost if feasible else float('inf')

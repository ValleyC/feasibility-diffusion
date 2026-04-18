"""
CVRPTW (Capacitated VRP with Time Windows) Manifold.

Extends CVRP partition with time window feasibility:
  - Each customer has a time window [earliest, latest] for service start
  - Each customer has a service duration
  - Vehicles can arrive early (wait) but NOT arrive after the latest time
  - Depot has its own time window [0, T_horizon]

State: partition assignment (same as CVRP).
Feasibility: capacity ≤ Q per vehicle AND a TW-feasible route ordering
             exists for each vehicle's customers.
Moves: relocate + swap (same as CVRP, but checked for TW feasibility too).

The key difference from CVRP: a capacity-feasible partition may NOT be
TW-feasible. The move feasibility check must verify both constraints.
"""

import numpy as np
from typing import List
from core.manifold import FeasibilityManifold
from problems.cvrp.partition import (
    random_partition, vehicle_loads, n_vehicles, get_vehicle_customers,
    enumerate_partition_moves, apply_partition_move,
    PartMoveType, PartMove,
)


def _check_tw_feasible(customers, dist, tw_early, tw_late, service_time,
                        depot=0):
    """Check if a TW-feasible route exists for the given customers.

    Uses greedy nearest-feasible-neighbor construction.
    Returns (feasible: bool, cost: float).
    """
    if len(customers) == 0:
        return True, 0.0

    remaining = set(customers)
    route = []
    current = depot
    current_time = 0.0
    total_dist = 0.0

    while remaining:
        # Find nearest feasible customer
        best, best_d = None, float('inf')
        for c in remaining:
            travel = dist[current, c]
            arrive = current_time + travel
            # Can we arrive before the latest time?
            if arrive <= tw_late[c] + 1e-8:
                if travel < best_d:
                    best_d = travel
                    best = c

        if best is None:
            return False, float('inf')  # no feasible next customer

        travel = dist[current, best]
        arrive = current_time + travel
        start_service = max(arrive, tw_early[best])  # wait if early
        current_time = start_service + service_time[best]
        total_dist += travel
        route.append(best)
        remaining.remove(best)
        current = best

    # Return to depot
    total_dist += dist[current, depot]
    return True, total_dist


def _route_cost_tw(customers, dist, tw_early, tw_late, service_time, depot=0):
    """Compute cost of TW-feasible route (or inf if infeasible)."""
    feasible, cost = _check_tw_feasible(
        customers, dist, tw_early, tw_late, service_time, depot
    )
    return cost if feasible else float('inf')


class CVRPTWManifold(FeasibilityManifold):
    """CVRPTW: partition with capacity + time window constraints.

    Instance: dict with:
        'coords' (N+1, 2), 'demands' (N+1,), 'capacity' float,
        'tw_early' (N+1,), 'tw_late' (N+1,), 'service_time' (N+1,),
        'dist' (N+1, N+1), 'n_customers' int
    """

    def sample_random(self, instance):
        """Generate random TW-feasible partition via greedy construction."""
        N = instance['n_customers']
        demands = instance['demands']
        capacity = instance['capacity']
        dist = instance['dist']
        tw_early = instance['tw_early']
        tw_late = instance['tw_late']
        service_time = instance['service_time']

        assign = np.full(N + 1, -1, dtype=np.int64)
        customers = list(range(1, N + 1))
        np.random.shuffle(customers)

        vehicle = 0
        current_customers = []
        load = 0.0

        for c in customers:
            # Try adding to current vehicle
            test_custs = current_customers + [c]
            new_load = load + demands[c]

            if new_load <= capacity:
                feasible, _ = _check_tw_feasible(
                    test_custs, dist, tw_early, tw_late, service_time
                )
                if feasible:
                    current_customers.append(c)
                    assign[c] = vehicle
                    load = new_load
                    continue

            # Start new vehicle
            vehicle += 1
            current_customers = [c]
            assign[c] = vehicle
            load = demands[c]

        return assign

    def cost(self, solution, instance):
        K = n_vehicles(solution)
        total = 0.0
        for k in range(K):
            custs = get_vehicle_customers(solution, k)
            c = _route_cost_tw(
                custs, instance['dist'], instance['tw_early'],
                instance['tw_late'], instance['service_time'],
            )
            if c == float('inf'):
                return float('inf')
            total += c
        return total

    def is_feasible(self, solution, instance):
        if solution[0] != -1:
            return False
        N = instance['n_customers']
        demands = instance['demands']
        capacity = instance['capacity']

        visited = set()
        K = n_vehicles(solution)
        for k in range(K):
            custs = get_vehicle_customers(solution, k)
            # Capacity check
            if sum(demands[c] for c in custs) > capacity + 1e-8:
                return False
            # TW check
            feasible, _ = _check_tw_feasible(
                custs, instance['dist'], instance['tw_early'],
                instance['tw_late'], instance['service_time'],
            )
            if not feasible:
                return False
            visited.update(custs)

        return visited == set(range(1, N + 1))

    def enumerate_moves(self, solution, instance):
        """Enumerate moves that are BOTH capacity AND TW feasible."""
        demands = instance['demands']
        capacity = instance['capacity']
        dist = instance['dist']
        tw_early = instance['tw_early']
        tw_late = instance['tw_late']
        service_time = instance['service_time']

        # Start with capacity-feasible moves (from CVRP)
        cap_moves = enumerate_partition_moves(solution, demands, capacity)

        # Filter: keep only those that are also TW-feasible
        valid_moves = []
        for move in cap_moves:
            new_assign = apply_partition_move(solution, move)
            mtype = move[0]

            if mtype == PartMoveType.RELOCATE:
                affected = [move[2], move[3]]  # src and dst vehicles
            else:
                affected = [move[2], move[4]]  # both vehicles

            tw_ok = True
            for k in set(affected):
                custs = get_vehicle_customers(new_assign, k)
                feasible, _ = _check_tw_feasible(
                    custs, dist, tw_early, tw_late, service_time
                )
                if not feasible:
                    tw_ok = False
                    break

            if tw_ok:
                valid_moves.append(move)

        return valid_moves

    def apply_move(self, solution, move):
        return apply_partition_move(solution, move)

    def move_delta(self, solution, move, instance):
        cost_before = self.cost(solution, instance)
        new = self.apply_move(solution, move)
        cost_after = self.cost(new, instance)
        return cost_after - cost_before
